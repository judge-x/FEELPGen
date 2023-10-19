import numpy as np
import math

def get_budget(n):
    delta=1/(n**2)
    ksi=1/(n**2)
    
    epsilon=math.log((1-ksi-delta)/ksi)
    return epsilon,delta


def generate_noise(n,p,sen=1):
    epsilon,delta=get_budget(n)
    
    delta=1/(n**2)
    # epsilon=100
    
    sigam=math.sqrt(2*math.log(1.25/delta))/epsilon*sen
    
    gaussian_matrix = np.random.normal(0, sigam, (n, p))
    
    return gaussian_matrix