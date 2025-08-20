import sys
import torch
import numpy as np
from torch import unsqueeze as UNZ
from Functions.utils import get_precision

def fourier_geometry(alpha,nhead,resol):

    precision = get_precision('double')

    """
    Fourier geometry function generated a general geometry given nhead and alpha
    """

    def Heaviside(x):
        out = torch.zeros_like(x)
        out[x==0] = 0.5
        out[x>0] = 1 
        return out.to(torch.bool)
    def angle_wrap(X,Y):# consider that X is Y and Y is X
        O = torch.zeros_like(X)
        mask1 = Heaviside(X)*Heaviside(Y)#(X>=0) & (Y>=0)
        O[mask1] = torch.atan2(torch.abs(Y[mask1]),torch.abs(X[mask1]))
        mask2 = Heaviside(-X)*Heaviside(Y)#(X<0) & (Y>=0)
        O[mask2] = torch.atan2(Y[mask2],X[mask2])
        mask3 = Heaviside(-X)*Heaviside(-Y)#(X<0) & (Y<0)
        O[mask3] = torch.atan2(torch.abs(Y[mask3]),torch.abs(X[mask3])) + torch.pi
        mask4 = Heaviside(X)*Heaviside(-Y)#(X>=0) & (Y<0)
        O[mask4] = 2*torch.pi-torch.atan2(torch.abs(Y[mask4]),torch.abs(X[mask4]))
        return O

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nPx = torch.tensor(resol, dtype=precision.int) 
    wvl = torch.tensor((635e-9), dtype=precision.real)
    alpha = torch.tensor(alpha*torch.pi/180, dtype=precision.real,device=device)
    ps = torch.tensor(3.74e-6, dtype=precision.real,device=device)   
    rooftop = (torch.tensor(0,dtype=precision.real,device=device)*ps)
    nhead = torch.tensor(nhead, dtype=precision.int)# con float funca mal
    # grid
    if nhead >= 2:
        x = torch.linspace( -(nPx-1)//2,(nPx-1)//2,nPx , dtype=precision.real, device=device)*ps
        X,Y = torch.meshgrid(x,x, indexing='ij')
        step = torch.tensor(2*np.pi/nhead, dtype=precision.real)
        nTheta = torch.arange(0,2*np.pi+step,step, dtype=precision.real)
        k = 2*np.pi/wvl
        O_wrap = angle_wrap(X,Y)
        pyr = torch.zeros((nPx,nPx), dtype=precision.complex,device=device)
        beforeExp = torch.zeros_like(pyr, dtype=precision.real,device=device)
        for i in range( len(nTheta)-1 ):
            mask = (Heaviside(O_wrap-nTheta[i])*Heaviside(nTheta[i+1]-O_wrap))
            phase = (nTheta[i]+nTheta[i+1])/2
            cor = torch.cos(phase)*X + torch.sin(phase)*Y
            cor = ((cor-rooftop)*(Heaviside(cor-rooftop).to(precision.real)))
            beforeExp += (mask.to(precision.real))*(k*torch.tan(alpha)*cor)
        afterExp = torch.exp(1j*(beforeExp))
        fourierMask = UNZ(UNZ(torch.fft.fftshift(afterExp/torch.sum(torch.abs(afterExp.flatten()))),0),0)
    elif nhead==0:
        fourierMask = torch.exp(1j*torch.zeros((1,1,nPx,nPx),dtype=precision.complex,device=device))
        beforeExp = 0
    else:
        raise KeyError('Incorrect number of heads')
    return fourierMask,beforeExp