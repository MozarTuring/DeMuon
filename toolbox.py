import numpy as np 
import pickle
import os

def l2_clip(z, tau):
    znorm = np.linalg.norm(z)
    return (tau / znorm) * z if znorm > tau else z

def smooth_clipping(z, phi, eps):
    psi = lambda u: phi * u / (u**2 + eps)**0.5
    return psi(z)

def dump_pickle(dict_ndarray, filename):
    '''
    save dictionary with values of ndarray to filename as a pickle
    '''
    directory, _ = os.path.split(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(dict_ndarray, f)

def save_numpy(ndarray: np.ndarray, filename: str):
    directory, _ = os.path.split(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(filename, ndarray)