# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
#torch tensors are multidimentional array with single data type
#here we are using Float as tensor data type
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) #randomly intialize the weights with normal distribution (mean of 0 and variance 1)
        self.a = torch.randn(1, nh) #first dimention for batch
        self.b = torch.randn(1, nv)
    def sample_h(self, x): # x stands for visible nodes
        wx = torch.mm(x, self.W.t()) # torch.mm to multiply two tensors but for mathematical correction we need to take transpose of weights(as weight matrix is for p_v_given_h)
        activation = wx + self.a.expand_as(wx) # bias self.a must be applied to every line of batch. So expan_as(wx)
        p_h_given_v = torch.sigmoid(activation) # hidden node is activated given the condition of visible node
        return p_h_given_v, torch.bernoulli(p_h_given_v) # return probabilities
        # we are making bernoulli rbm as we are just making binary classification
        # now we have to return some bernoulli smaples of theat distribution
        # p_h_given_v is vector of nh element with probability of each hidden node being in active state given the state of visible nodes
        # for each hidden node bernoulli sampling works like this :- if probability of i(th) neuron being active is 0.6 and generated a random number id below 0.6 then
        #                                                            we simply deactivate that neuron else we activate it
        # In the end we get vector of zero and one.
    def sample_v(self, y): #y stands for hidden nodes 
        wy = torch.mm(y, self.W) # as weight matrix is for p_v_given_h we does not need to take transpose here
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    # RBM sre energy based models. So,we have to minimize the energy. It is same as maximize the log likely hood.(log of probability)
    # to minimize the energy or to maximize log likely hood  we need to compute the gradient which is heavy operation here
    # so we approzimate this gradients.
    # This approximating algorithm is called as Constractive gradient ( cdk algorithm) which comes with gibbs sampling (gibbs chain for k iterations).
    def train(self, v0, vk, ph0, phk): 
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0) # just to keep dimentions on equal form we add v0 - vk to 0
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        # Gibbs chain implementation
        # MCMC (markov chain monte carlo technique)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
