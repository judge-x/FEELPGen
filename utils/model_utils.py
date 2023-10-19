import json
import numpy as np
import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
import numpy as np
from FLAlgorithms.trainmodel.models import Net
from torch.utils.data import DataLoader
import torch.nn.functional as F
from FLAlgorithms.trainmodel.generator import Generator, DiversityLoss
from pympler import asizeof

from utils.model_config import *
from utils.pca_utils import *

METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time']


def get_data_dir(dataset, server_index):
    if 'EMnist' in dataset:
        # EMnist-alpha0.1-ratio0.1-0-letters
        dataset_ = dataset.replace('alpha', '').replace('ratio', '').split('-')
        alpha, ratio = dataset_[1], dataset_[2]
        types = 'letters'
        path_prefix = os.path.join('data', 'EMnist', f'u20-{types}-alpha{alpha}-ratio{ratio}-domain{server_index}')
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/emnist-n10/'

    elif 'Mnist' in dataset:
        dataset_ = dataset.replace('alpha', '').replace('ratio', '').split('-')
        alpha, ratio = dataset_[1], dataset_[2]
        # path_prefix=os.path.join('data', 'Mnist', 'u20alpha{}min10ratio{}'.format(alpha, ratio))
        path_prefix = os.path.join('data', 'Mnist',
                                   'u20c10-alpha{}-ratio{}-domain{}'.format(alpha, ratio, server_index))
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/mnist-n10/'

    elif 'celeb' in dataset.lower():
        dataset_ = dataset.lower().replace('user', '').replace('agg', '').split('-')
        user, agg_user = dataset_[1], dataset_[2]
        path_prefix = os.path.join('data', 'CelebA', 'user{}-agg{}-domain{}'.format(user, agg_user, server_index))
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
        proxy_data_dir = os.path.join('/user500/', 'proxy')

    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir, proxy_data_dir


def read_data(dataset, ser_idx):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_data_dir, test_data_dir, proxy_data_dir = get_data_dir(dataset, ser_idx)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])

    if proxy_data_dir and os.path.exists(proxy_data_dir):
        proxy_files = os.listdir(proxy_data_dir)
        proxy_files = [f for f in proxy_files if f.endswith('.json') or f.endswith(".pt")]
        for f in proxy_files:
            file_path = os.path.join(proxy_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata = torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            proxy_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data


def read_proxy_data(proxy_data, dataset, batch_size):
    X, y = proxy_data['x'], proxy_data['y']
    X, y = convert_data(X, y, dataset=dataset)
    dataset = [(x, y) for x, y in zip(X, y)]
    proxyloader = DataLoader(dataset, batch_size, shuffle=True)
    iter_proxyloader = iter(proxyloader)
    return proxyloader, iter_proxyloader


def aggregate_data_(clients, dataset, dataset_name, batch_size):
    combined = []
    unique_labels = []
    for i in range(len(dataset)):
        id = clients[i]
        user_data = dataset[id]
        X, y = convert_data(user_data['x'], user_data['y'], dataset=dataset_name)
        combined += [(x, y) for x, y in zip(X, y)]
        unique_y = torch.unique(y)
        unique_y = unique_y.detach().numpy()
        unique_labels += list(unique_y)

    data_loader = DataLoader(combined, batch_size, shuffle=True)
    iter_loader = iter(data_loader)
    return data_loader, iter_loader, unique_labels


def aggregate_user_test_data(data, dataset_name, batch_size):
    clients, loaded_data = data[0], data[3]
    data_loader, _, unique_labels = aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, np.unique(unique_labels)


def aggregate_user_data(data, dataset_name, batch_size):
    # data contains: clients, groups, train_data, test_data, proxy_data
    clients, loaded_data = data[0], data[2]
    data_loader, data_iter, unique_labels = aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, data_iter, np.unique(unique_labels)


def convert_data(X, y, dataset=''):
    if not isinstance(X, torch.Tensor):
        if 'celeb' in dataset.lower():
            X = torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2)
            y = torch.Tensor(y).type(torch.int64)

        else:
            X = torch.Tensor(X).type(torch.float32)
            y = torch.Tensor(y).type(torch.int64)
    return X, y


def read_user_data(index, data, dataset='', count_labels=False):
    # data contains: clients, groups, train_data, test_data, proxy_data(optional)
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=dataset)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    X_test, y_test = convert_data(test_data['x'], test_data['y'], dataset=dataset)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    if count_labels:
        label_info = {}
        unique_y, counts = torch.unique(y_train, return_counts=True)
        unique_y = unique_y.detach().numpy()
        counts = counts.detach().numpy()
        label_info['labels'] = unique_y
        label_info['counts'] = counts
        return id, train_data, test_data, label_info
    return id, train_data, test_data


def get_dataset_name(dataset):
    dataset = dataset.lower()
    passed_dataset = dataset.lower()
    if 'celeb' in dataset:
        passed_dataset = 'celeb'
    elif 'emnist' in dataset:
        passed_dataset = 'emnist'
    elif 'mnist' in dataset:
        passed_dataset = 'mnist'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset


def create_generative_model(dataset, algorithm='', model='cnn', embedding=False, device='cpu'):
    passed_dataset = get_dataset_name(dataset)
    device_=device
    assert any([alg in algorithm for alg in ['FeedPGen', 'FeedPGen']])
    if 'FeedPGen' in algorithm:
        # temporary roundabout to figure out the sensitivity of the generator network & sampling size
        if 'cnn' in algorithm:
            gen_model = algorithm.split('-')[1]
            passed_dataset += '-' + gen_model
        elif '-gen' in algorithm:  # we use more lightweight network for sensitivity analysis
            passed_dataset += '-cnn1'
    return Generator(passed_dataset, model=model, embedding=embedding, latent_layer_idx=-1, device=device_)


def create_model(model, dataset, algorithm):
    passed_dataset = get_dataset_name(dataset)
    model = Net(passed_dataset, model), model
    return model


def polyak_move(params, target_params, ratio=0.1):
    for param, target_param in zip(params, target_params):
        param.data = param.data - ratio * (param.clone().detach().data - target_param.clone().detach().data)


def meta_move(params, target_params, ratio):
    for param, target_param in zip(params, target_params):
        target_param.data = param.clone().data + ratio * (target_param.clone().data - param.clone().data)


def moreau_loss(params, reg_params):
    # return 1/T \sum_i^T |param_i - reg_param_i|^2
    losses = []
    for param, reg_param in zip(params, reg_params):
        losses.append(torch.mean(torch.square(param - reg_param.clone().detach())))
    loss = torch.mean(torch.stack(losses))
    return loss


def l2_loss(params):
    losses = []
    for param in params:
        losses.append(torch.mean(torch.square(param)))
    loss = torch.mean(torch.stack(losses))
    return loss


def update_fast_params(fast_weights, grads, lr, allow_unused=False):
    """
    Update fast_weights by applying grads.
    :param fast_weights: list of parameters.
    :param grads: list of gradients
    :param lr:
    :return: updated fast_weights .
    """
    for grad, fast_weight in zip(grads, fast_weights):
        if allow_unused and grad is None: continue
        grad = torch.clamp(grad, -10, 10)
        fast_weight.data = fast_weight.data.clone() - lr * grad
    return fast_weights


def init_named_params(model, keywords=['encode']):
    named_params = {}
    # named_params_list = []
    for name, params in model.named_layers.items():
        if any([key in name for key in keywords]):
            named_params[name] = [param.clone().detach().requires_grad_(True) for param in params]
            # named_params_list += named_params[name]
    return named_params  # , named_params_list


def get_log_path(args, algorithm, seed, gen_batch_size=32):
    alg = args.dataset + "_" + algorithm
    alg += "_" + str(args.learning_rate) + "_" + str(args.num_users)
    alg += "u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg = alg + "_" + str(seed)
    if 'FedGen' in algorithm:  # to accompany experiments for author rebuttal
        alg += "_embed" + str(args.embedding)
        if int(gen_batch_size) != int(args.batch_size):
            alg += "_gb" + str(gen_batch_size)
    return alg


def aggregate_server_parameter(servers, alg, glo_iter, device):
    global_models = []
    if alg == 'FedAvg':
        global_model = servers[0].model
        
        #test total model size:
        # model_size=0
        # for item in servers:
        #     model_size+=asizeof.asizeof(item)
            
        # print('servers size :',model_size)
        num_samples=[]    
        for i in range(0,len(servers)):
            tmp_num=0
            for client in servers[i].selected_users:
                tmp_num+=client.train_samples
            num_samples.append(tmp_num)
        
        sum_sample=sum(num_samples)

        for global_param in global_model.parameters():
            global_param.data *= num_samples[0]/sum_sample

        for i in range(1, len(servers)):
            for global_param, user_param in zip(global_model.parameters(), servers[i].model.parameters()):
                global_param.data += user_param.data.clone() * (num_samples[i]/sum_sample)
                
        for item in servers:
            global_models.append(global_model)
    elif alg == 'FeedPGen':
        time_start=time.time()
        gen_distribution = []
        y = np.random.choice(servers[0].qualified_labels, servers[0].batch_size)
        y_input = torch.LongTensor(y).to(device)

        # generate latent distribution
        for item in servers:
            gen_data = item.generative_model(y_input, latent_layer_idx=-1, verbose=True)['output']
            gen_distribution.append(gen_data)

        if glo_iter == 0:
            select_all = True
        else:
            select_all = False

        for j in range(len(servers)):
            #time_cost for one server
            # time_start_server=time.time()
            
            if select_all:
                # cold-start
                # compute similiary
                ser_distance = []
                dis_item = gen_distribution[j]
                for i in range(len(gen_distribution)):
                    if i != j:
                        dis_compare = gen_distribution[i]
                        # ser_distance.append(servers[j].generative_model.diversity_loss(dis_item,dis_compare).item()*10000)
                        ser_distance.append(
                            F.kl_div(dis_item.softmax(-1).log(), dis_compare.softmax(-1), reduction='sum'))
                    else:
                        ser_distance.append(0)

                servers[j].DL = ser_distance
                lr = 0.01
                gramma = 0.5
                distance_sum = sum(ser_distance)
                # a=servers[i].max_trainning_time
                # b=1

                weights = []
                for item in servers[j].DL:
                    if item == j:
                        weights.append(0)
                    else:
                        # weight_data=ser_distance[item]/distance_sum
                        # tau=a-servers[item].max_trainning_time
                        # if tau>0:
                        #     weight_asyn=1/tau
                        # else:
                        #     weight_asyn=0
                        # print('weight data: ',weight_data,' weight aysn : ',weight_asyn)
                        weights.append(item/distance_sum)

                global_model = servers[j].model
                for i in range(len(weights)):
                    for global_param, user_param in zip(global_model.parameters(), servers[i].model.parameters()):
                        global_param.data += user_param.data.clone() * weights[i] * lr

                global_models.append(global_model)
                
            else:
                # print("select half of clients:")
                # random select half of server
                available_range = [x for x in range(0, len(servers)) if x != j]
                selected_server = random.sample(available_range, 4)

                # compute similiary
                # ser_distance=[]
                dis_item = gen_distribution[j]
                for i in selected_server:
                    # if i!=j:
                    dis_compare = gen_distribution[i]
                    # ser_distance.append(servers[j].generative_model.diversity_loss(dis_item,dis_compare).item()*10000)
                    # ser_distance.append(F.kl_div(dis_item.softmax(-1).log(),dis_compare.softmax(-1),reduction='sum'))
                    servers[j].DL[i] = F.kl_div(dis_item.softmax(-1).log(), dis_compare.softmax(-1), reduction='sum')
                    # else:
                    #     ser_distance.append(0)

                # update with DL Queue
                max_dl = np.max(servers[j].DL)
                servers[j].DL[j] = max_dl + 0.1

                # get the aggreated index and distance
                DL_ = servers[j].DL
                selected_idx = np.argpartition(DL_, int(0.4 * len(servers)))[:int(0.4 * len(servers))]
                sum_distance = 0
                for item in selected_idx:
                    sum_distance+=servers[j].DL[item]

                lr = 0.01
                gramma = 0.5

                weights = []
                for item in selected_idx:
                    weight_data = servers[j].DL[item] / sum_distance
                    staeless_tau = glo_iter - servers[j].tau[item]
                    servers[j].tau[item] = glo_iter

                    weight_asyn = 0.8 ** staeless_tau

                    # print('weight data: ', weight_data, ' weight aysn : ', weight_asyn)
                    weights.append(weight_data * gramma + weight_asyn * 0.1 * (1 - gramma))

                global_model = servers[j].model
                idx = 0
                for i in selected_idx:
                    for global_param, user_param in zip(global_model.parameters(), servers[i].model.parameters()):
                        global_param.data += user_param.data.clone() * weights[idx] * lr
                    idx += 1

                global_models.append(global_model)
        #     print('FeedGen time cost for one server is',time.time()-time_start_server)
        # print('FeedGen time cost one epoch is',time.time()-time_start)
        

    elif alg == 'FeedPGen+':
        time_start=time.time()
        
        # triainning with the optimized PCA algorithm
        gen_distribution = []
        gen_distribution_=[]
        gen_distribution_pca=[]
        y = np.random.choice(servers[0].qualified_labels, servers[0].batch_size)
        y_input = torch.LongTensor(y).to(device)

        # generate latent distribution
        for item in servers:
            #test for compare communication cost
            gen_data = item.generative_model(y_input, latent_layer_idx=-1, verbose=True)['output']
            gen_distribution.append(gen_data)
            gen_distribution_.append(gen_data.to('cpu').detach().numpy())
            # print('orgin size for transimition:',gen_data.to('cpu').detach().numpy().nbytes+sys.getsizeof(np.array([[]],dtype=np.float32)))
            
            gen_data_pca=copy.deepcopy(getPCA(gen_data.to('cpu').detach().numpy()))
            # tmp=copy.deepcopy(gen_data_pca)
            gen_distribution_pca.append(gen_data_pca)
            # print('pcaed size for taansmition:',np.array(gen_data_pca).nbytes+sys.getsizeof(np.array([[]],dtype=np.float32)))

    
        print('orgin size for transimition:',np.array(gen_distribution_).nbytes+sys.getsizeof(np.array([[]],dtype=np.float32)))
        print('pcaed size for taansmition:',np.array(gen_distribution_pca).nbytes+sys.getsizeof(np.array([[]],dtype=np.float32)))


        if glo_iter == 0:
            select_all = True
        else:
            select_all = False

        for j in range(len(servers)):
            # time_start_server=time.time()
            if select_all:
                # cold-start
                # compute similiary
                ser_distance=[]
                ser_distance_pca = []
                dis_item=gen_distribution[j]
                dis_item_pca = gen_distribution_pca[j]
                for i in range(len(gen_distribution)):
                    if i != j:
                        # compute distance between pca
                        dis_compare_pca = gen_distribution_pca[i]
                        # dis=torch.norm(torch.tensor(dis_item_pca)-torch.tensor(dis_compare_pca),p=2,dim=1)
                        dis=np.linalg.norm((dis_item_pca-dis_compare_pca),ord=2)
                        # ser_distance_pca.append(torch.sum(dis).item()/32)
                        ser_distance_pca.append(dis)
                        # compute distance by KL distance
                        dis_compare=gen_distribution[i]
                        ser_distance.append(
                            F.kl_div(dis_item.softmax(-1).log(), dis_compare.softmax(-1), reduction='sum'))
                        
                        
                    else:
                        ser_distance_pca.append(0)
                        ser_distance.append(0)

                #for test
                sort_=sorted(range(len(ser_distance)), key=lambda i: ser_distance[i])
                sort_pca=sorted(range(len(ser_distance_pca)), key=lambda i: ser_distance_pca[i])

                
                

                servers[j].DL = ser_distance_pca
                
                lr = 0.01
                gramma = 0.5
                distance_sum = sum(ser_distance_pca)

                weights = []
                for item in servers[j].DL:
                    weights.append(item/distance_sum)

                global_model = servers[j].model
                for i in range(len(weights)):
                    for global_param, user_param in zip(global_model.parameters(), servers[i].model.parameters()):
                        global_param.data += user_param.data.clone() * weights[i] * lr

                global_models.append(global_model)
            else:
                # print("select half of clients:")
                # random select half of server
                available_range = [x for x in range(0, len(servers)) if x != j]
                selected_server = random.sample(available_range, 4)

                # compute similiary
                # ser_distance=[]
                dis_item_pca = gen_distribution_pca[j]
                for i in selected_server:
                    # if i!=j:
                    dis_compare_pca = gen_distribution_pca[i]
                    servers[j].DL[i]=torch.sum(torch.norm(torch.tensor(dis_item_pca)-torch.tensor(dis_compare_pca),p=2,dim=1)).item()/32
                    

                # update with DL Queue
                max_dl = np.max(servers[j].DL)
                servers[j].DL[j] = max_dl + 0.1

                # get the aggreated index and distance
                DL_ = servers[j].DL
                selected_idx = np.argpartition(DL_, int(0.4 * len(servers)))[:int(0.4 * len(servers))]
                sum_distance = 0
                for item in selected_idx:
                    sum_distance+=servers[j].DL[item]

                lr = 0.01
                gramma = 0.5

                weights = []
                for item in selected_idx:
                    weight_data = servers[j].DL[item] / sum_distance
                    staeless_tau = glo_iter - servers[j].tau[item]
                    servers[j].tau[item] = glo_iter

                    weight_asyn = (staeless_tau/1)**0.8

                    print('weight data: ', weight_data, ' weight aysn : ', weight_asyn)
                    weights.append(weight_data * gramma + weight_asyn * 10 * (1 - gramma))

                
                weights_sum=sum(weights)
                
                global_model = servers[j].model
                idx = 0
                for i in selected_idx:
                    for global_param, user_param in zip(global_model.parameters(), servers[i].model.parameters()):
                        global_param.data += user_param.data.clone() * weights[idx]/weights_sum * lr
                    idx += 1

                global_models.append(global_model)
    elif alg == 'SDFEEL':
        #synchronous updating
        #constract laplacian metric
        # adjacency_matrix = np.ones((10,10))-np.eye(10)

        # degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

        # L = degree_matrix - adjacency_matrix
        
        # # get omega which is the number of sample domain
        # num_samples=[]
        # omega = np.zeros((10, 10))    
        # for i in range(0,len(servers)):
        #     tmp_num=0
        #     for client in servers[i].selected_users:
        #         tmp_num+=client.train_samples
        #     num_samples.append(tmp_num)
        #     omega[i,i]=tmp_num
        
        # num_sum=sum(num_samples)
        # omega/=num_sum
        # omega_=np.linalg.inv(omega)
        
        # L_=L*omega_
        
        # # get the i-th eigen
        # eigenvalues = np.linalg.eigvals(L_)
        # eigenvalues_sorted = np.sort(eigenvalues)
        # minest_eigenvalue = eigenvalues_sorted[1]
        # maxest_eigenvalue = eigenvalues_sorted[-1]
        
        # I=np.eye(10)
        
        # P=I-(2/(minest_eigenvalue+maxest_eigenvalue))*L_

            
        
                
        #     print('FeedGen+ time cost for one server is',time.time()-time_start_server)
        # print('FeedGen+ time cost one epoch is',time.time()-time_start)
        
        #ayschronous updating
        training_time=[]
        for item in servers:
            training_time.append(item.max_training_time)
        
        for j in range(0,len(servers)):
            train_self=training_time[j]
            agg_index=[]
            agg_time=[]
            for i in range(0,len(servers)):
                if training_time[i]>train_self:
                    agg_index.append(i)
                    agg_time.append(training_time[i]-train_self)
                    

            
            global_model=servers[j].model
            if len(agg_time)==0:
                pass
            else:
                time_sum=sum(agg_time)
                agg_weight=np.array(agg_time)/time_sum
                for global_param in global_model.parameters():
                    global_param.data /= len(agg_weight+1)
                k=0
                for i in agg_index:
                    for global_param, user_param in zip(global_model.parameters(), servers[i].model.parameters()):
                        global_param.data += user_param.data.clone() * agg_weight[k] /len(agg_weight+1)
                    k+=1
            
            
            global_models.append(global_model)
    else:
        raise "no such global aggregation algorithm"

    return global_models