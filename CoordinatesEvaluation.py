import numpy as np
import scipy.special as sp
import math
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
from matplotlib import path

# ___________________________ Begin def ScattererBC() __________________________

def ScattererBC(Coord_matlab, plot=False):
    """
    This function accepts the NURBS generated coordinates and returns 
    scatterer coordinates and corresponding normals. 
    Users can enable the plot toggle to visualize the scatterer shape and the 
    normals.
    """

    #______ Calculating the normals for the obtained coordinates generated _____
    """
    Before using this code for the NURBS generated coordinates-
    check if circle has same coordinates twice. If then delete one.
    We take 1st to second last row from circle_temp and remove the repeating
    final row.
    """
    Coords_temp = Coord_matlab
    N_coord =  list(Coords_temp[:,0].shape)[0]     # number of points on the scatterer

    shape = Coords_temp                    # (x,y) in N_coord*2
    # second point on circle is first point on circle_next_point
    shape_next_point = torch.roll(shape, -1, 0)  

    # normal for 2D scatterer shape
    dy = shape_next_point[:,1] - shape[:,1]              # nx = dy
    dx = shape[:,0] - shape_next_point[:,0]              # ny = -dx
    n_temp = torch.cat((dy.view(N_coord,1), dx.view(N_coord,1)),1)      # norm_temp = [dy, -dx]
    n_temp_abs = (torch.sum(n_temp**2,1))**(1/2) 
    n_abs = n_temp_abs.view(N_coord,1).repeat(1,2)
    n_Sc = n_temp/n_abs            # UNIT Normal: normal = norm_temp/|norm_temp|

    # Plotting normals for comparison
    if plot:
        plt.plot(n_Sc[:,0], 'b-', label='Nx_predicted')
        plt.plot(n_Sc[:,1], 'g-', label='Ny_predicted')
        plt.legend()
        plt.title('Comparing the true and calculated normals: Uniform distribution')
        plt.show()


    #______________________ Finding all outward normals ________________________
    """
    Here in reality, the n_Sc is the normal calculated at the mid point between
    shape and shape_next_point. 
    So, here p can be either shape (approx.) or (shape + shape_next_point)/2 (exact) 
    """

    # The center of the scatterer at (a,b)
    a = 0.5
    b = 0.5
    n_unit_temp =  n_Sc                                 # from above
    C = torch.tensor([a, b]).repeat(N_coord,1)          # center: 99*2
    p = (shape + shape_next_point)/2                    # from above

    n_unit_Sc =  torch.zeros_like(n_unit_temp)
    for i in range(0, N_coord-1):
        if torch.dot((p-C)[i], n_unit_temp[i])<0:
            n_unit_Sc[i] = -1*n_unit_temp[i]
        else:
            n_unit_Sc[i] = n_unit_temp[i]  

    #______________________ Check for inward normals if any ____________________
    """
    [1] No inward normal if m remains as m=0
    [2] Quiver plot: Plot the normals for all the points
    """
    # [1]
    m=0
    for i in range(0, N_coord-1):
        if torch.dot((p-C)[i], n_unit_Sc[i])<0:
            m+=1
    print("\nNumber of wrong inward normals = ", m)

    # [2]  
    if plot:
        for i in range(0, N_coord-1):
            plt.plot(p[i,0], p[i,1], 'bo')
            plt.quiver(p[i,0], p[i,1], n_unit_Sc[i,0], n_unit_Sc[i,1])

        plt.axis('scaled')
        plt.title('Scatterer with corresponding unit normals')
        plt.show()    

    return p, n_unit_Sc        # returns the points on scatterer and the corresponding unit normals

# _____________________________ End def ScattererBC() __________________________




#____________ Evaluate: ([0,1]x[0,1]- scatterer) domain coordinates ____________

#__________________________ Begin def Domain_eval() ____________________________ 

def Domain_eval(Coord_Sc, Coord_domain):
    """
    Giving radius=1e-9 helps to get rid of the inconsistensies with 
    points on the boundary.
    When radius is given:
    1. If points of Scatterer are in CCW: boundary points are included inside the shape
    2. If points of Scatterer are in CW: boundar points are excluded
    """
    p = path.Path(Coord_Sc)
    point = Coord_domain
    Q = p.contains_points(point, radius=1e-9)    # python function for checking if points lie within
    N_coordD = list(Coord_domain[:,0].shape)[0]
    
    Coord_d_temp = []
    for i in range(0, N_coordD):
        if Q[i]== False:
            Coord_d_temp.append(Coord_domain[i])
        else:
            continue  

    Coord_d = torch.cat(Coord_d_temp).view(len(Coord_d_temp),2)  # New domain coordinates
    
    return Coord_d

#___________________________ End def Domain_eval() _____________________________
