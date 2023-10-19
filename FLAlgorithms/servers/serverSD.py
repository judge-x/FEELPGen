from FLAlgorithms.users.userSD import UserSD
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# Implementation for FedSD Server
import time
import random

class FedSD(Server):
    def __init__(self, args, model, seed, device="cpu",ser_idx=0):
        device_=device
        super().__init__(args, model, seed, device_)

        # Initialize data for all  users
        data = read_data(args.dataset,ser_idx)
        total_users = len(data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))

        # self.total_train_times=[]
        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserSD(args, id, model, train_data, test_data, use_adam=False, device=self.device)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            # self.total_train_times.append(user.training_time)
    
        self.max_training_time=random.randint(10,20)
        print("")    
        print("server max triain time is", self.max_training_time)
        print("Finished creating FedAvg server.")
    
    def select_users_SD(self, glob_iter):
        pass
        
            
    
    def train(self, args, glob_iter,tau):
        print("\n-------------Round number: ",glob_iter, " -------------\n")
        self.selected_users = self.select_users(glob_iter,self.num_users)
        self.send_parameters(mode=self.mode)
        self.evaluate()
        self.timestamp = time.time() # log user-training start time
        for user in self.selected_users: # allow selected users to train
                user.train(glob_iter, personalized=self.personalized) #* user.train_samples
        curr_timestamp = time.time() # log  user-training end time
        train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
        self.metrics['user_train_time'].append(train_time)
        # Evaluate selected user
        if glob_iter % tau==0:
            self.aggregate_parameters()

        # self.save_results(args)
        # self.save_model()