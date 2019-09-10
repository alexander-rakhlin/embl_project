from sklearn.covariance import empirical_covariance
import numpy as np

def pierson_correlation(data):
    cov = empirical_covariance(data.reshape(-1, data.shape[-1]))
    std = np.std(data.reshape(-1, data.shape[-1]), axis=0)
    std2 = std[:, None] * std[None, :]
    return cov / std2

def thresholded_correlation(dat):
    eps = 1e-10
    data = dat.astype(np.float32)
    data[data<-2/3] = -1
    data[(data<-1/3) & (data >-1 + eps)] = -0.5
    data[(data < 1/3) & (data > -0.5 + eps)] = 0
    data[data > 2/3] = 1
    data[(data > eps) & (data < 1 - eps)] = 0.5
    return data

    
