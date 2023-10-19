#!/usr/bin/env python
import argparse

import numpy as np

from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
# from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverSD import FedSD
from utils.model_utils import create_model, aggregate_server_parameter
from utils.plot_utils import *
import torch
from multiprocessing import Pool
from torchsummary import summary
import pandas as pd
import time
import sys
from pympler import asizeof

import warnings

warnings.filterwarnings("ignore")


def create_server_n_user(args, model, i, index, device):
    # model = create_model(args.model, args.dataset, args.algorithm)
    # if args.device=="cuda":
    #     torch.cuda.set_device(0)
    # device='cuda'
    # print()
    # model.to(device)
    if ('FedAvg' in args.algorithm):
        server = FedAvg(args, model, i, device, index)
    elif 'FeedPGen' in args.algorithm:
        server = FedGen(args, model, i, device, index)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i, device,index)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, i, device,index)
    elif ('SDFEEL' in args.algorithm):
        server = FedSD(args,model,i,device,index)
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
        
    # init model
    # Generate init model
    init_model = create_model(args.model, args.dataset, args.algorithm)

    # Generate model
    servers = []
    print("\n[starting creating total {} server]".format(args.n_server))
    for j in range(args.n_server):
        servers.append(create_server_n_user(args, init_model, i, j, device=device))
    print("\n[finish]")

    # Generate global model
    glo_model = init_model[0]
    glo_model.to(device)
    # print(next(servers[0].model.parameters()).device)

    #for mnist
    # summary(glo_model, input_size=(1, 28, 28), device="cuda")

    # for celeba
    # summary(glo_model, input_size=(3, 84, 84), device="cuda")
    
    # for generator
    # print('Generator size is',asizeof.asizeof(servers[0].generative_model))




    print("\n\n                        [ Start training iteration {} ]           \n\n".format(i))

    accs_mean = []
    losses_mean = []
    glo_models = []
    for glo_iter in range(args.num_glo_iters):
        
        accs = []
        losses = []
        if ("SDFEEL" not in args.algorithm):
            print("\n                        [ Global Round {} ]           \n".format(glo_iter))

            

            time_sum=0
            for j in range(args.n_server):
                print("\n-------------Train Domain: ", j, " -------------\n")

                # Update models for all servers
                if glo_iter != 0:
                    servers[j].model = glo_models[j]
                     
                
                if args.train:
                    time_start_train=time.time()
                    
                    servers[j].train(args)
                    
                    
                    
                    _, f_test_samples, f_accs, f_losses = servers[j].test()
                    glob_acc = np.sum(f_accs) * 1.0 / np.sum(f_test_samples)
                    glob_loss = np.sum([x * y for (x, y) in zip(f_test_samples, f_losses)]).item() / np.sum(f_test_samples)
                    
                    #calculate compute cost
                    time_sum+=time.time()-time_start_train
                    
                    accs.append(glob_acc)
                    losses.append(glob_loss)
                    
                    
            print('local training time is :', time_sum/(args.n_server*args.num_users))
            print('\nGlobal {} final accuracy{}: '.format(glo_iter, np.mean(accs)))
            print('Global {} final loss{}: '.format(glo_iter, np.mean(losses)))
            accs_mean.append(np.mean(accs))
            losses_mean.append(np.mean(losses))

            # Aggregate the total model
            time_start_inter=time.time()
            glo_models = aggregate_server_parameter(servers, args.total_algorithm, glo_iter, device)
            print('algorithm ', args.algorithm, ' time cost is:', time.time()-time_start_inter)

        # plot_final_result(args, accs_mean, losses_mean)
        else:
            tau_1=1
            tau_2=1
            
            for j in range(args.n_server):
                print("\n-------------Train Domain: ", j, " -------------\n")
                if args.train:
                    servers[j].train(args,glo_iter,tau_1)
                    _, f_test_samples, f_accs, f_losses = servers[j].test()
                    glob_acc = np.sum(f_accs) * 1.0 / np.sum(f_test_samples)
                    glob_loss = np.sum([x * y for (x, y) in zip(f_test_samples, f_losses)]).item() / np.sum(f_test_samples)
                    
                    accs.append(glob_acc)
                    losses.append(glob_loss)
                
            
            print('\nGlobal {} final accuracy{}: '.format(glo_iter, np.mean(accs)))
            print('Global {} final loss{}: '.format(glo_iter, np.mean(losses)))
            accs_mean.append(np.mean(accs))
            losses_mean.append(np.mean(losses))
            
            # Update models for all servers
            alpha=0
                
            if glo_iter%(tau_1*tau_2) == 0 and glo_iter!=0:
                alpha+=1
                glo_models = aggregate_server_parameter(servers, args.total_algorithm, glo_iter, device)
                servers[j].model = glo_models[j]
        
        
    df = pd.DataFrame({'acc': accs_mean, 'loss': losses_mean})
    df.to_excel('{}_{}_{}_glr{}.xlsx'.format(args.dataset,args.algorithm,args.total_algorithm, args.gen_lr), index=False)

    print('The total training accuracy is: ', accs_mean)
    print('The total training loss is: ', losses_mean)


def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist-alpha0.1-ratio0.5")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--algorithm", type=str, default="FeedPGen")
    parser.add_argument("--total_algorithm", type=str, default="FeedPGen")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    # parser.add_argument("--personal_learning_rate", type=float, default=0.01,
    #                     help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--gen_lr", type=float, default=1e-4, help="Gen learning rate.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glo_iters", type=int, default=50)
    parser.add_argument("--num_mid_iters", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=20)
    
    parser.add_argument("--n_server", type=int, default=10, help="number of total server")
    parser.add_argument("--num_users", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--gpu_idx", type=int, default=0, help="run gpu in ?")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

    

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.gen_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glo_iters))
    print("Number of mid rounds       : {}".format(args.num_mid_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
