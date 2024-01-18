# PINN_2DCircle_ScatteredPressure_withAnalyticalSol

A physics-informed neural network (PINN) to solve the scattered pressure field (p_s) for an infinite acoustic domain with a circular scatterer. The details of different parts of the code are included in the main.py file documentation.

The PINNs can be trained to accurately predict the real and imaginary scattered pressure fields due to circular scatterers of different radii. The figure below highlights the PINN prediction comparison for a scatterer of radius r=0.1m at a frequency of 500Hz.

![Alt text](/test_Re(ps).jpg?raw=true)
![Alt text](/test_Im(ps).jpg?raw=true)
![Alt text](/test_Abs(ps).jpg?raw=true)
