"""
SYNOPSIS
    Implementation of multilayer perceptron network using JAX libraries
DESCRIPTION

    Contains one module:
    a) MLP - defines the layers and depth of the multilayer perceptron
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

# import flax
from flax import linen as nn # type: ignore
import torch.nn as tnn # type: ignore
import torch.nn.functional as F # type: ignore
# import torch
# import numpy as np


class MLP(nn.Module):
    num_hidden: list
    num_outputs: int

    def setup(self):
        self.linear = [nn.Dense(features=self.num_hidden[i]) for i in range(len(self.num_hidden))]
        self.linear2 = nn.Dense(features=self.num_outputs)


    def __call__(self, x):
        for i in range(len(self.num_hidden)):
            x = self.linear[i](x)
            x = nn.elu(x)
        x = self.linear2(x)
        return x
        #return x ** 2



class MLP_torch(tnn.Module):
    def __init__(self,
                 inp_size=100,
                 num_hidden=[500, 400, 200]):
      super(MLP_torch, self).__init__()

      self.inp_size = inp_size
      self.num_hidden = num_hidden

      # Hidden layers
      self.hidden = tnn.ModuleList()
      self.hidden.append(tnn.Linear(inp_size, num_hidden[0]))
      for k in range(len(num_hidden) - 1):
        self.hidden.append(tnn.Linear(num_hidden[k], num_hidden[k+1]))

      # Final output layer
      self.linear2 = tnn.Linear(num_hidden[-1], 1)

    def forward(self, x):
        x = x.float()
        for i in range(len(self.num_hidden)):
            #import ipdb;
            #ipdb.set_trace()
            x = self.hidden[i](x)
            x = F.elu(x)
        x = self.linear2(x)
        return x

    """def pred(self, x0, ref):
        ''' The general prediction for NN value functions '''
        d0 = torch.cat([x0, ref]).double()
        return self.network(d0.unsqueeze(0))[0]"""



