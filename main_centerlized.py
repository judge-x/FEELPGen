#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool
import pandas as pd

def create_server_n_user(args, i, device):
    server_index=0
    model = create_model(args.model, args.dataset, args.algorithm)
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, i, device,server_index)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i, device,server_index)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    
    #set device
    if args.device=='cuda':
        gpu_id=args.gpu_idx
        device=torch.device(f'cuda:{gpu_id}')
    else:
        device='cpu'
    
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i, device=device)

    if args.train:
        server.train(args)
        server.test()
        # print(server.metrics['glob_acc'])
        # print(server.metrics['glob_loss'])
    
    df = pd.DataFrame({'acc': server.metrics['glob_acc'], 'loss': server.metrics['glob_loss']})
    df.to_excel('{}_{}.xlsx'.format(args.dataset,args.algorithm), index=False)

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist-alpha0.1-ratio0.5")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="FedProx")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--num_mid_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--gpu_idx", type=int, default=0, help="run gpu in ?")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

    args = parser.parse_args()
    main(args)
