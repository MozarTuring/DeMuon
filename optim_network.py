import numpy as np
from toolbox import l2_clip, smooth_clipping

'''
All network algorithms in this module is based on ATC, adapt then combine type; other types of
network SGD can be added in the future. 
'''

def dsgd(weights, X, stepsize, SG):
    '''
    this function does network sgd, 
    weights: d x d matrix for model averaging. 
    X: current iterate, n x d
    SG: n x d ndarray, n is the num_client, d is the gradient len/dim.
    return: next X
    '''
    # row_norms = np.linalg.norm(SG, ord=2, axis=1, keepdims=True)
    # normalized_SG = SG / row_norms
    # return weights @ (X - stepsize * normalized_SG)
    return weights @ (X - stepsize * SG)
    
def dsgd_gclip(weights, X, stepsize, SG, l2_bd):
    '''
    this function does network l2 norm based gradient clipping on agents
    SG: n x d ndarray, n is the num_client, d is the gradient len/dim
    '''
    clip_axis = lambda u: l2_clip(u, l2_bd)
    if SG.ndim == 2:
        SG_clipped = np.apply_along_axis(clip_axis, 1, SG)
    else: # scalar case 
        SG_clipped = np.clip(SG, -l2_bd, l2_bd)
    return weights @ (X - stepsize * SG_clipped)

def sclip_ef_network(weights, X, M, x_stepsize, m_stepsize, phi, eps, SG):
    ''' 
    x: current iterate on the server
    M: local momentums on clients, x x d ndarray
    phi: scaler of the clipping operator
    eps: smoother of the clipping operator
    SG: n x d ndarray, n is the num_client, d is the gradient len/dim
    '''
    M = m_stepsize*M + (1-m_stepsize)*smooth_clipping(SG-M, phi=phi, eps=eps)
    return weights @ (X - x_stepsize * M), M

def gt_nsgdm(weights, X, Y, M, x_stepsize, m_stepsize, SG):
    '''
    this function implements gradient tracking based normalized 
    stochastic gradient descent with momentum
    X: n x d
    Y: n x d, gradient estimators
    M: n x d, local momentum
    SG: n x d, stochastic gradient
    '''
    
    M_temp = m_stepsize*M + (1 - m_stepsize)*SG

    Y = weights @ (Y + M_temp - M)
    M = M_temp 

    row_norms = np.linalg.norm(Y, ord=2, axis=1, keepdims=True)
    normalized_Y = Y / row_norms
    X = weights @ (X - x_stepsize*normalized_Y)

    return X, Y, M

def gt_dsgd(weights, X, Y, SG_pre, x_stepsize, SG):
    '''
    this function implements gradient tracking sgd
    X: n x d
    Y: n x d, gradient estimators
    SG_pre: n x d, last stochastic gradient
    SG: n x d, stochastic gradient
    '''
    Y = weights @ (Y + SG - SG_pre)
    SG_pre = SG
    X = weights @ (X - x_stepsize*Y)
    return X, Y, SG_pre

    