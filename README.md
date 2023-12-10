# FEELPGen:A Heterogeneous-aware  and Privacy-preserving Federated Edge Learning for Cross-silo Network

Research code that accompanies the paper []().
It contains implementation of the following algorithms:
* **FedAvg-c/FedAvg-h** ([paper](https://ieeexplore.ieee.org/document/9148862))
* **FedProx-c/FedProx-h** ([paper](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)).
* **HierFAVG** ([paper])(https://ieeexplore.ieee.org/abstract/document/9148862/)which represent as FedAvg+FedAvg
* **SD-FEEL** ([paper](https://ieeexplore.ieee.org/document/10059225)).

## Install Requirements:
```pip3 install -r requirements.txt```

## Prepare Dataset: 
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FeedPGen/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20
### This will generate a dataset located at FeedPGen/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>
    

- Similarly, to generate *non-iid* **EMnist** Dataset, using 10% of the total available training samples:
<pre><code>cd FeedPGen/data/EMnist
python generate_niid_dirichlet.py --sampling_ratio 0.1 --alpha 0.1 --n_user 20 
### This will generate a dataset located at FeedPGen/data/EMnist/u20-letters-alpha0.1-ratio0.1/
</code></pre> 

## Run Experiments: 
'''
# mnist
<pre><code>
python main.py --dataset Mnist-alpha0.5-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen

python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen

python main.py --dataset Mnist-alpha0.05-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen

python main.py --dataset Mnist-alpha0.05-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen+
</code></pre>


# emnist
<pre><code>
python main.py --dataset EMnist-alpha0.5-ratio0.5 --algorithm FedAvg --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg

python main.py --dataset EMnist-alpha0.5-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg

python main.py --dataset EMnist-alpha0.5-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen

python main.py --dataset EMnist-alpha0.05-ratio0.5 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen+
</code></pre>

# celeba
<pre><code>
python main.py --dataset celeb-user20-agg50 --algorithm FedAvg --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg

python main.py --dataset celeb-user20-agg100 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg

python main.py --dataset celeb-user20-agg100 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen

python main.py --dataset celeb-user20-agg100 --algorithm FeedPGen --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FeedPGen+ --gpu_idx 2
</code></pre>

#run compare
# FedForx-h
<pre><code>
python main.py --dataset celeb-user20-agg100 --algorithm FedProx --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg

python main.py --dataset Mnist-alpha0.05-ratio0.5 --algorithm FedProx --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg

python main.py --dataset EMnist-alpha0.1-ratio0.5 --algorithm FedProx --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1 --total_algorithm FedAvg --gpu_idx 3
</code></pre>

# FedProx-c
<pre><code>
python main_centerlized.py --dataset celeb-user20-agg100 --algorithm FedProx --batch_size 32 --num_glo_iters 50 --num_mid_iters 5 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --times 1
</code></pre>

## Thanks Refer as
https://github.com/zhuangdizhu/FedGen)https://github.com/zhuangdizhu/FedGen
