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
from Loss2DCircle import HelmholtzPINNLoss_2Dcircle


#______________________ Begin def run_training_PINN() __________________________

def run_training_PINN(X_sc, n_sc, X_domain, r_s, p0, freq, obj1, n_epochs, b_size,
                      learning_rate, model, plot_train=False):
    input1 = X_sc
    input2 = n_sc
    split_index = int(len(input1))
    train_inputs1  = input1[:split_index,:] *1.  
    train_inputs2  = input2[:split_index,:] *1.  

    BATCH_SIZE = b_size
    trainset = torch.utils.data.TensorDataset(train_inputs1, train_inputs2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    train_start_time = time.time()
    num_epochs = n_epochs
    train_loss = []
    train_phy_loss = []
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_running_phy = 0.0
        # commence training
        model = model.train()

        # training step
        for i, (input1, input2) in enumerate(trainloader):
            input1 = input1.float()
            input2 = input2.float()
            loss, phy = obj1.loss_total(input1, input2)
            # finding the gradient using .backward
            optimizer.zero_grad()
            loss.backward(gradient = None, retain_graph = True) 
            # update model params
            optimizer.step()
            train_running_loss += loss.detach().item()
            train_running_phy += phy.detach().item()
            
        train_loss.append(train_running_loss/ i)
        train_phy_loss.append(train_running_phy /i)
                                                  
        if epoch % 100 == 0:
            print('Epoch: %d | Train Loss: %.4f | Train phy loss: %.4f' \
                   %(epoch, train_running_loss / i, train_running_phy/i))
            
        # Visualize the progress- by default plot_train = False
        if plot_train and (epoch % 100 == 0):
            Coord_domain_test = X_domain 
            obj2 = HelmholtzPINNLoss_2Dcircle(X_domain, r_s, p0, freq, model)
            Ppred = obj2.solution(Coord_domain_test).detach().numpy()

            XX_domain = Coord_domain_test[:,0].detach().numpy()
            YY_domain = Coord_domain_test[:,1].detach().numpy()
            N_pred = Ppred.shape[0]
            PpredRe = torch.tensor(Ppred[:,0]).view(N_pred)  #3645 #896
            PpredIm = torch.tensor(Ppred[:,1]).view(N_pred)
            PpredAbs = (PpredRe**2 + PpredIm**2)**(1/2)

            # plots- Re
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
            plt.jet()
            fig.suptitle('Prediction at %d Hz'%(freq))
            #1
            c1= ax1.scatter(XX_domain, YY_domain, c=PpredRe)  
            ax1.plot(X_sc[:,0], X_sc[:,1], 'k*')
            ax1.set_title('Re(Ps)') 
            fig.colorbar(c1, ax=ax1)   
            #2
            c2=ax2.scatter(XX_domain, YY_domain, c=PpredIm)
            ax2.plot(X_sc[:,0], X_sc[:,1], 'k*')   
            ax2.set_title('Im(Ps)') 
            fig.colorbar(c2, ax=ax2)  
            plt.show();    
            plt.tight_layout();

    #____________________________ Loss plot ____________________________________
    fig, ax = plt.subplots(dpi=115)
    ax.plot(train_loss, label='Train loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend(loc='best');
                   
#___________________ End def run_training_PINN() ________________________



#___________________ Begin def run_testing_loss_PINN() ________________________

def run_testing_PINN(Coord_domain_test, X_sc, Ppred, Ptrue, freq):

    x_domain = Coord_domain_test[:,0]
    y_domain = Coord_domain_test[:,1]

    N_pred = Ppred.shape[0]
    PpredRe = torch.tensor(Ppred[:,0]).view(N_pred) 
    PpredIm = torch.tensor(Ppred[:,1]).view(N_pred)
    PpredAbs = (PpredRe**2 + PpredIm**2)**(1/2)

    # Error values
    abs_true = (Ptrue[:,0]**2 + Ptrue[:,1]**2)**(1/2)
    error_real = np.abs(Ptrue[:,0] -PpredRe)
    error_imag = np.abs(Ptrue[:,1] -PpredIm)
    error_abs = np.abs(abs_true -PpredAbs)

    print('\n \n \n Testing \n')
    # plots- Re
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
    plt.jet()
    fig.suptitle('Re at %d Hz'%(freq))
    #1
    c1=ax1.scatter(x_domain, y_domain, c=PpredRe)    
    ax1.plot(X_sc[:,0], X_sc[:,1], 'k*')
    ax1.set_title('Predicted Ps') 
    fig.colorbar(c1, ax=ax1)   
    #2
    c2=ax2.scatter(x_domain, y_domain, c=Ptrue[:,0])
    ax2.plot(X_sc[:,0], X_sc[:,1], 'k*')   
    ax2.set_title('True Ps') 
    fig.colorbar(c2, ax=ax2)  
    #3
    c3=ax3.scatter(x_domain, y_domain, c=error_real)
    ax3.plot(X_sc[:,0], X_sc[:,1], 'k*')   
    ax3.set_title('abs Error') 
    fig.colorbar(c3, ax=ax3)                  
    plt.show(); 
    plt.tight_layout();


    # plots- Im
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
    plt.jet()
    fig.suptitle('Im at %d Hz'%(freq))
    #1
    c1=ax1.scatter(x_domain, y_domain, c=PpredIm)    
    ax1.plot(X_sc[:,0], X_sc[:,1], 'k*')
    ax1.set_title('Predicted Ps') 
    fig.colorbar(c1, ax=ax1)   
    #2
    c2=ax2.scatter(x_domain, y_domain, c=Ptrue[:,1])
    ax2.plot(X_sc[:,0], X_sc[:,1], 'k*')   
    ax2.set_title('True Ps') 
    fig.colorbar(c2, ax=ax2)  
    #3
    c3=ax3.scatter(x_domain, y_domain, c=error_imag)
    ax3.plot(X_sc[:,0], X_sc[:,1], 'k*')   
    ax3.set_title('abs Error') 
    fig.colorbar(c3, ax=ax3)                  
    plt.show();    
    plt.tight_layout();

    # plots- Abs
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
    plt.jet()
    fig.suptitle('Abs at %d Hz'%(freq))
    #1
    c1=ax1.scatter(x_domain, y_domain, c=PpredAbs)    
    ax1.plot(X_sc[:,0], X_sc[:,1], 'k*')
    ax1.set_title('Predicted Ps') 
    fig.colorbar(c1, ax=ax1)   
    #2
    c2=ax2.scatter(x_domain, y_domain, c=(Ptrue[:,0]**2 + Ptrue[:,1]**2)**(1/2))
    ax2.plot(X_sc[:,0], X_sc[:,1], 'k*')   
    ax2.set_title('True Ps') 
    fig.colorbar(c2, ax=ax2)  
    #3
    c3=ax3.scatter(x_domain, y_domain, c=error_abs)
    ax3.plot(X_sc[:,0], X_sc[:,1], 'k*')   
    ax3.set_title('abs Error') 
    fig.colorbar(c3, ax=ax3)                  
    plt.show();   
    plt.tight_layout();    

#___________________ End def run_testing_PINN() ________________________

