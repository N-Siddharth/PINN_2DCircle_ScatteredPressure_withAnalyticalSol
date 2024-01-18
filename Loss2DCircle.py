import numpy as np
import scipy.special as sp
import math
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable, grad, functional
from torchsummary import summary

# function to take derivatives
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, allow_unused=True)[0]

#______________________ Begin class HelmholtzPINNLoss() ________________________

class HelmholtzPINNLoss_2Dcircle(object):
    # Class Initialization
    def __init__(self, x_domain, r_s, p0, freq, model):
        """
        rho    =  air density in 'kg/m^3'
        freq   =  freq in 'Hz'
        cs     =  speed of sound in air 'm/s'
        r_s    =  radius of circular scatterer in 'm'
        p0     =  amplitude of incident acoutsic wave in 'Pa'
        k      =  wavenumber in the homogeneous medium. It is calculated with 
                  freq and cs.
        x_domain =  Coordinates (x,y) of the domain. Pressure at each point 
                    inside the domain satisfies Helmholtz equation.
        net    =  neural network
        solution = combination of NN(theta) + ps_analytical 
        """
        self.rho = 1.24
        self.f = freq
        self.cs = 342.21
        self.r_s = r_s
        self.p0 = p0
        self.k = 2*math.pi*self.f/self.cs
        self.x_domain = x_domain
        # calling network
        self.net = model    
        # solution from network
        self._solution = lambda x: self.ps_analytical(x) + self.net(x)

    @property
    def solution(self):
        return self._solution     

    def ps_analytical(self, x):
        """
        This function converts the NN input x, y coordinates into r, theta and 
        evaluates the true solution analytically in the polar coordinates.
        """
        X = x.detach().numpy().copy()
        p0 = self.p0
        #center of the domain
        cx = 0.5
        cy = 0.5
        n_max = 5
        r_s = self.r_s
        ka = self.k*r_s
        n = np.linspace(1,n_max,n_max)
        a0 = -p0*sp.jv(1, ka)/sp.hankel1(1, ka)
        an = -2*(1j)**(n)*p0*0.5*(sp.jv(n-1, ka) - 
                                  sp.jv(n+1, ka))/(n*sp.hankel1(n, ka)/ka - 
                                                    sp.hankel1(n+1, ka))
        N_sample = X.shape[0]
        N = np.repeat(np.linspace(0,n_max,n_max+1).reshape(1, n_max+1), 
                      N_sample, axis=0)
        An = np.repeat(np.concatenate((a0.reshape(1), an),
                                      axis=0).reshape(1,n_max+1),
                                      N_sample, axis=0) 
        R = np.repeat(np.sqrt((X[:,0]-cx)**2 + (X[:,1]-cy)**2).reshape(N_sample, 1),
                      n_max+1, axis=1)
        
        theta = np.arctan((X[:,1]-cy)/(X[:,0]-cx))
        # Note that the theta values vary according to the quadrant in which 
        # the input x,y coordinate is situated 
        for i in range(N_sample):
            if (X[i,0]-cx)<0 and (X[i,1]-cy)>0:
                theta[i] = math.pi - theta[i]
            if (X[i,0]-cx)<0 and (X[i,1]-cy)<0:
                theta[i] = theta[i] - math.pi   
            if (X[i,0]-cx)>0 and (X[i,1]-cy)<0:   
                theta[i] = 2*math.pi - theta[i]             

        Theta = np.repeat(theta.reshape(N_sample, 1), n_max+1, axis=1)  
        ps = np.sum(An*sp.hankel1(N, self.k*R)*np.cos(N*Theta), axis=1)

        # N_sample*2 (real and imag for each sample)
        Ps = np.concatenate((np.real(ps).reshape(N_sample,1), np.imag(ps).reshape(N_sample,1)), axis=1)

        return torch.tensor(Ps)           

    def loss_total(self, X_sc, n_sc):
        
        X_domain = self.x_domain
        
        def LossSc():
            """
            LossSc() evaluates the MSE loss based on the Neumann BC on the 
            scatterer boundary
            """
            #X_sc = self.x_sc
            X_sc.requires_grad = True
            sol_bc = self.solution(X_sc)

            N_sc = list(X_sc[:,0].shape)[0]
            ek = torch.tensor([1.0, 0.0]).repeat(N_sc, 1)

            z = torch.zeros((N_sc))
            for i in range(N_sc):
                z[i] = torch.dot(X_sc[i], self.k*ek[i])

            pi = torch.cos(z) - 1j*torch.sin(z)           
            piRe = torch.real(pi)                                       
            piIm = torch.imag(pi) 
            
            dpdxRe = grad(sol_bc[:,0] + piRe, X_sc)
            dpdxIm = grad(sol_bc[:,1] + piIm, X_sc)

            dpdn_sc = torch.zeros(N_sc, 2)
            for i in range(0, N_sc):
                dpdn_sc[i,0] = torch.dot( (1/self.rho)*dpdxRe[i], n_sc[i])
                dpdn_sc[i,1] = torch.dot( (1/self.rho)*dpdxIm[i], n_sc[i])

            loss_sc = torch.mean((dpdn_sc[:,0])**2 + (dpdn_sc[:,1])**2)    

            return loss_sc
            
        def LossPhy():
            """
            LossPhy() evaluates the MSE loss based on the Helmholtz equation on
            the coordinates inisde the square domain
            """          
            X_domain.requires_grad = True
            sol_dRe= self.solution(X_domain)[:,0]
            sol_dIm= self.solution(X_domain)[:,1]

            sol_dxRe = grad(sol_dRe, X_domain)
            sol_dxxRe = grad(sol_dxRe[:,0], X_domain)[:,0]
            sol_dyyRe = grad(sol_dxRe[:,1], X_domain)[:,1]

            sol_dxIm = grad(sol_dIm, X_domain)
            sol_dxxIm = grad(sol_dxIm[:,0], X_domain)[:,0]
            sol_dyyIm = grad(sol_dxIm[:,1], X_domain)[:,1]

            sol_phyRe = (sol_dxxRe + sol_dyyRe + self.k**2*sol_dRe)   #N_domain*N_domain
            sol_phyIm = (sol_dxxIm + sol_dyyIm + self.k**2*sol_dIm)
            
            loss_phy = torch.mean(sol_phyRe**2 + sol_phyIm**2)

            return loss_phy

        loss_BC = LossSc()     
        loss_phy = LossPhy()
        
        #weights are chosen by trial and error in the initial stage
        w1 = 1
        w2 = 1
        loss_total = w1*loss_phy + w2*loss_BC

        return loss_total, loss_phy

#__________________ End class HelmholtzPINNLoss_2Dcircle() ___________________
