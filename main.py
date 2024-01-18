
__version__ = '1.1'
__author__  = "Siddharth Nair (nair40@purdue.edu)"
__date__    = '2022-May-17'

__doc__     ='''

A novel physics-informed neural network (PINN) focused on predicting scattered
pressure field (p_s) with a 2D circular scatterer inside a square domain is developed.
The PINN assumes a volumetric source term satsified by the analytical solution 
of the Helmholtz equation for the governing PDE. 
The total loss is a sum of the MSE losses of governing PDE (Loss_phy) and Neumann
boundary condition (Loss_BC) on the scatterer boundary. 

The usage of the helping functions are detailed below:

1. CoordinatesEvaluation.py :- 
   a. def ScattererBC() - Function takes NURBS scatterer coordinates as inputs 
                          and outputs scatterer coordinates with corresponding
                          unit normals.
   b. def Domain_eval() - Function takes scatterer coordinates and square domain
                          [0,1]x[0,1] and outputs the (square domain -scatterer)
                          domain coordinates. The network is trained on this 
                          updated domain coordinates. 

2. Loss2DCircle.py :-
   class HelmholtzPINNLoss_2Dcircle() - An object of this class evaluates the 
                                        Loss_phy and Loss_BC for both real and
                                        imaginary parts of p_s. Note that a 
                                        function of this class def ps_analytical()
                                        evaluates the analytical solution on the 
                                        input (x,y) coordinate.
   Loss = w1*Loss_phy + w2*Loss_BC       
        = w1*MSE(Helmholtz_eqn) + w2*MSE(Neumann_condition)   

   where,
                      d^2p_s        d^2p_s
   Helmholtz_eqn: =   ------    +   ------   +   k^2 p_s   -   f  =  0
                       dx^2          dy^2              

                    here  f is the analytical solution satisfying the Helmholtz
                    equation. This is called the volumetric source term.   

                           d (p_s + p_i)      
   Neumann_condition :=    ------------   =  0   
                                dn    


3. DenseResNet.py :-
   class DenseResNet() - An object of this class defines a deep residual network.
                         No Fourier features or beta tuning were used in the 
                         current implementation. 
                         class DenseResNet() is used from the open source code 
                         provided in one of the lecture series by Dr. Ilias 
                         Bilionis. Copyright material of prof's group - 
                         ref: https://www.predictivesciencelab.org/    

4. TrainAndTest.py :- 
  a. def run_training_PINN() - Function takes in all the initialized values with
                               network model and trains the PINN. Training loss
                               and optional progress plots can be visualized here.
  b. def run_testing_PINN() - Function takes in trained parameters and network 
                              and evaluates a test model.     
                              
@endofdocs
'''


import numpy as np
import scipy.special as sp
import math
import matplotlib.pyplot as plt
# %matplotlib inline
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable, grad, functional
from torchsummary import summary
from matplotlib import path
import time

from CoordinatesEvaluation import Domain_eval, ScattererBC
from Loss2DCircle import HelmholtzPINNLoss_2Dcircle
from DenseResNet import DenseResNet
from TrainAndTest import run_training_PINN, run_testing_PINN

if __name__ == "__main__":

    xx1 = np.linspace(0, 1, 64)
    X, Y = np.meshgrid(xx1, xx1)   

    #____________________ Coordinates inside the domain ________________________
    X_domainflat = torch.Tensor(np.hstack([X.flatten()[:, None], Y.flatten()[:, None]]))

    #____________________ Coordinates on the boundary __________________________
    n_theta = 200
    r_s = 0.05
    theta = np.linspace(0, 2*math.pi, n_theta)
    x_sc = r_s*np.cos(theta) + 0.5
    y_sc = r_s*np.sin(theta) + 0.5
    XBC_temp = torch.vstack((torch.tensor(x_sc), torch.tensor(y_sc)))  #(x,y) on circle as 2*100
    X_BC = XBC_temp[:,:].t()  
    X_sc, n_sc = ScattererBC(X_BC)  

    #__________________ Initialize the network architecture ____________________
    model = DenseResNet(dim_in=2, dim_out=2, num_resnet_blocks=2, 
                            num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(), 
                            fourier_features=False, m_freqs=50, sigma=10, tune_beta=False)

    #________________________ Initialize parameters ____________________________
    p0 = 1.
    freq = 500
    n_epochs = 1000
    b_size = 128
    learning_rate = 0.001
    X_domain = Domain_eval(X_sc, X_domainflat)
    obj1 = HelmholtzPINNLoss_2Dcircle(X_domain, r_s, p0, freq, model)

    #________________________ Train the network ________________________________
    run_training_PINN(X_sc, n_sc, X_domain, r_s, p0, freq, obj1, n_epochs, b_size, 
                      learning_rate, model, plot_train=True)

    #________________________ Test the network ________________________________
    model.eval()
    
    # test parameters
    n_theta = 200
    r_s = 0.1
    theta = np.linspace(0, 2*math.pi, n_theta)
    x_sc = r_s*np.cos(theta) + 0.5
    y_sc = r_s*np.sin(theta) + 0.5
    XBC_temp = torch.vstack((torch.tensor(x_sc), torch.tensor(y_sc)))  #(x,y) on circle as 2*100
    X_BC = XBC_temp[:,:].t()  
    X_sc, n_sc = ScattererBC(X_BC)  
    obj3 = HelmholtzPINNLoss_2Dcircle(X_domain, r_s, p0, freq, model)    

    N_test = 100
    xx2 = np.linspace(0, 1, N_test)
    X, Y = np.meshgrid(xx2, xx2)
    Coord_domain_temp = torch.Tensor(np.hstack([X.flatten()[:, None], Y.flatten()[:, None]]))
    Coord_domain_test = Domain_eval(X_sc, Coord_domain_temp)

    Ppred = obj3.solution(Coord_domain_test).detach().numpy() 
    Ptrue = obj3.ps_analytical(Coord_domain_test)   

    run_testing_PINN(Coord_domain_test, X_sc, Ppred, Ptrue, freq)

