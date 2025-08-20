import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from numpy import linalg as LA
import math


"""
The class of exponential graphs: undirected and directed
As number of nodes increases exponentially, the degree at each node increases linearly.
"""
class Exponential_graph:
    def __init__(self,number_of_nodes):
        self.size = number_of_nodes
    
    def undirected(self):
        U = np.zeros( (self.size,self.size) )
        for i in range( self.size ):
            U[i][i] = 1
            hops = np.array( range( int(math.log(self.size-1,2)) + 1 ) )  
            neighbors = np.mod( i + 2 ** hops, self.size )
            for j in neighbors:
                U[i][j] = 1
                U[j][i] = 1
        return U

    def directed(self):             ## n = 2^x.
        D = np.zeros( (self.size,self.size) )
        for i in range( self.size ):
            D[i][i] = 1
            hops = np.array( range( int(math.log(self.size-1,2)) + 1 ) )  
            neighbors = np.mod( i + 2 ** hops, self.size )
            for j in neighbors:
                D[i][j] = 1
        return D

def connected_cycle_connectivity(num_agents, deg):
    '''
    connected cycle with certain degree, the actual degree is 2 * deg
    '''
    network = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(i+1, i+deg+1):
            network[i, j%num_agents] = 1
    return network + network.T

def complete_graph_connectivity(num_agents):
    network = np.ones((num_agents, num_agents))
    return network

def metropolis_weight(network):
    '''
    # ref: 'A scheme for robust distributed sensor fusion based on average consensus'
    '''
    deg_vec = np.sum(network, axis=1)
    num_nodes = len(deg_vec)
    weights = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if network[i, j] == 1 and i != j:
                weights[i, j] = 1 / (1 + np.maximum(deg_vec[i], deg_vec[j]))
        weights[i, i] = 1 - np.sum(weights[i])
    return weights

def connected_cycle_weights(filename, n=20, degree=5):
    network = connected_cycle_connectivity(n, degree)
    if not os.path.exists(filename):
        weights = metropolis_weight(network)
        np.save(filename, network)
        np.save(filename.replace('.npy', '_weights.npy'), weights)
    else: 
        # print('Using exsisting network')
        weights = np.load(filename.replace('.npy', '_weights.npy'))
    singular_values = np.linalg.svd(weights, compute_uv=False)
    print('The second largest singular value of the ring graph is', singular_values[1])
    return weights

def complete_graph_weights(filename, n=20):
    network = complete_graph_connectivity(n)
    # weights = metropolis_weight(network)
    # use uniform weights
    weights = np.ones((n, n)) / n
    np.save(filename, network)
    np.save(filename.replace('.npy', '_weights.npy'), weights)
    singular_values = np.linalg.svd(weights, compute_uv=False)
    print('The second largest singular value of the complete graph is', singular_values[1])
    return weights

def exponential_graph_weights(filename, n=20):
    graph = Exponential_graph(n)
    network = graph.directed()
    weights = metropolis_weight(network)
    np.save(filename, network)
    np.save(filename.replace('.npy', '_weights.npy'), weights)
    singular_values = np.linalg.svd(weights, compute_uv=False)
    print('The second largest singular value of the directed exponential graph is', singular_values[1])
    return weights

def exponential_graph_weights_undirected(filename, n=20):
    graph = Exponential_graph(n)
    network = graph.undirected()
    weights = metropolis_weight(network)
    np.save(filename, network)
    np.save(filename.replace('.npy', '_weights.npy'), weights)
    singular_values = np.linalg.svd(weights, compute_uv=False)
    print('The second largest singular value of the undirected exponential graph is', singular_values[1])
    return weights

