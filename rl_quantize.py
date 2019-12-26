# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import sys
sys.path.insert(0, '../BERT')

import os
import math
import json
import argparse
import numpy as np
from copy import deepcopy
from pathlib import Path

from lib.utils.utils import prYellow
from lib.env.quantize_env import QuantizeEnv
from lib.rl.ddpg import DDPG
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
# import torchvision.models as models
# import models as customized_models

from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler

from transformers.modeling_bert import BertForPreTraining
from transformers.tokenization_bert import BertTokenizer
from examples.lm_finetuning.finetune_on_pregenerated import PregeneratedDataset

# Models
'''
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names
print('support models: ', model_names)
'''

def train(num_episode, agent, env, output, debug=False):
    # best record
    best_reward = -math.inf
    best_policy = []

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action, action_actor = agent.random_action(observation)
        else:
            action, action_actor = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action, action_actor)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if episode % int(num_episode / 100) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            if debug:
                print('#{}: episode_reward:{:.4f} loss: {:.4f}, w_ratio: {:.4f}'.format(episode, episode_reward,
                                                                                         info['loss'],
                                                                                         info['w_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} loss: {:.4f}, w_ratio: {:.4f}, org_loss: {:.4f}\n'.format(episode, episode_reward,
                                                                                     info['loss'],
                                                                                     info['w_ratio'], env.org_loss))
            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode)[0],
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy

            value_loss = agent.get_value_loss()
            policy_loss = agent.get_policy_loss()
            delta = agent.get_delta()
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/loss', info['loss'], episode)
            tfwriter.add_scalar('info/w_ratio', info['w_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(best_policy), episode)
            tfwriter.add_text('info/current_policy', str(env.strategy), episode)
            tfwriter.add_text('info/actor_policy', str(env.strategy_actor), episode)
            tfwriter.add_scalar('value_loss', value_loss, episode)
            tfwriter.add_scalar('policy_loss', policy_loss, episode)
            tfwriter.add_scalar('delta', delta, episode)
            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate_w/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(best_reward))
            text_writer.write('best policy: {}\n'.format(best_policy))
            
            text_writer.flush()
    text_writer.close()
    return best_policy, best_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    # parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use)')
    # parser.add_argument('--dataset_root', default='data/imagenet', type=str, help='path to dataset)')
    parser.add_argument('--preserve_ratio', default=0.1, type=float, help='preserve ratio of the model size')
    parser.add_argument('--min_bit', default=1, type=int, help='minimum bit to use')
    parser.add_argument('--max_bit', default=8, type=int, help='maximum bit to use')
    parser.add_argument('--float_bit', default=32, type=int, help='the bit of full precision float')
    parser.add_argument('--is_pruned', dest='is_pruned', action='store_true')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=20, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=128, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.993, type=float,
                        help='delta decay during exploration')
    parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./save', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=600, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=234, type=int, help='')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    # parser.add_argument('--data_bsize', default=256, type=int, help='number of data batch size')
    parser.add_argument('--finetune_epoch', default=1, type=int, help='')
    parser.add_argument('--finetune_gamma', default=0.8, type=float, help='finetune gamma')
    parser.add_argument('--finetune_lr', default=0.001, type=float, help='finetune gamma')
    parser.add_argument('--finetune_flag', default=False, type=bool, help='whether to finetune')
    parser.add_argument('--use_top5', default=False, type=bool, help='whether to use top5 acc in reward')
    # parser.add_argument('--train_size', default=20000, type=int, help='number of train data size')
    # parser.add_argument('--val_size', default=10000, type=int, help='number of val data size')
    parser.add_argument('--resume', default='', type=str, help='Resuming model path for testing')
    # Architecture
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet_v2', choices=model_names,
    #               help='model architecture:' + ' | '.join(model_names) + ' (default: mobilenet_v2)')
    # device options
    # parser.add_argument('--gpu_id', default='1', type=str,
    #                     help='id(s) for CUDA_VISIBLE_DEVICES')

    # bert
    parser.add_argument('--epochs', type=int, default=3)    
    parser.add_argument('--bert_model', type=str, required=True)
    parser.add_argument('--do_lower_case', action="store_true")
    parser.add_argument('--reduce_memory', action="store_true")
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--add_extra_state', action='store_true')
    parser.add_argument('--new_reward', action='store_true')
    parser.add_argument('--separate_qkv', action='store_true')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--bert_val_size', default=5000, type=int)
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument('--use_recorder', action='store_true')
    parser.add_argument('--use_diff', action='store_true')

    args = parser.parse_args()
    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"
    '''
    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        metrics = json.loads(metrics_file.read_text())
        samples_per_epoch.append(metrics['num_training_examples'])
    '''
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if not args.no_cuda:
        assert torch.cuda.is_available(), 'CUDA is needed for CNN'
        args.n_gpu = torch.cuda.device_count()
    else:
        args.n_gpu = 0
    print("==> Use ngpu: {}".format(args.n_gpu))

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not args.no_cuda:
            torch.cuda.manual_seed_all(args.seed)


    base_folder_name = '{}_ratio{}_run'.format(args.bert_model, args.preserve_ratio)
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix

    args.output = os.path.join(args.output, base_folder_name)
    tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
    print('==> Output path: {}...'.format(args.output))

    # load model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # total_train_examples = 0
    # for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        # total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    model = BertForPreTraining.from_pretrained('../CBert/bert_weight')
    if not args.no_cuda:
        model.cuda()
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model).cuda()


    pretrained_model = model.state_dict()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True

    epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                        num_data_epochs=3, reduce_memory=args.reduce_memory)
   
    n_eval = len(epoch_dataset)
    indices = list(range(n_eval)) 
    np.random.shuffle(indices)
    if args.debug:
        args.bert_val_size = 50
    eval_sampler = SubsetRandomSampler(indices[:args.bert_val_size])

    eval_dataloader = DataLoader(epoch_dataset, sampler=eval_sampler, 
                                num_workers=args.n_worker, batch_size=args.train_batch_size)

    env = QuantizeEnv(model, pretrained_model, 
                        compress_ratio=args.preserve_ratio, n_data_worker=args.n_worker,
                        args=args, float_bit=args.float_bit, is_model_pruned=args.is_pruned,
                        val_loader=eval_dataloader)

    nb_states = env.layer_embedding.shape[1]
    nb_actions = 1  # actions for weight and activation quantization
    args.rmsize = args.rmsize * len(env.quantizable_idx)  # for each layer
    print('** Actual replay buffer size: {}'.format(args.rmsize))
    agent = DDPG(nb_states, nb_actions, args)
    if args.resume != '':
        print('==> Load weights from {}'.format(args.resume))
        agent.load_weights(args.resume)

    best_policy, best_reward = train(args.train_episode, agent, env, args.output, debug=args.debug)
    print('best_reward: ', best_reward)
    print('best_policy: ', best_policy)

