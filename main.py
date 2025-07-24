import os
import argparse
import torch

from torch.backends import cudnn
from utils.utils import *

# ours
from solver import Solver

# other MMs
# from other_solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train(training_type='first_train')
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'memory_initial':
        solver.get_memory_initial_embedding(training_type='second_train')


    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temp_param', type=float, default=0.05)
    parser.add_argument('--lambd', type=float, default=0.01)
    parser.add_argument('--lambd_con', type=float, default=0.001)
    parser.add_argument('--lambd_kl', type=float, default=0.001)
    parser.add_argument('--lambd_dis', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--cut_len', type=float, default=0.2)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--data_path', type=str, default='/home/stu2023/qtt/project/MEMTO-main/dataset/SMD/')
    parser.add_argument('--model_save_path', type=str, default='/home/stu2023/qtt/project/MEMTO-main/checkpoint')
    parser.add_argument('--anormly_ratio', type=float, default=0.2)
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_memory', type=int, default=10, help='number of memory items')
    parser.add_argument('--a_memory', type=int, default=10, help='number of memory items')
    parser.add_argument('--num_workers', type=int, default=4*torch.cuda.device_count())
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--memory_initial', type=str, default=False, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--phase_type',type=str, default=None, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')

    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
