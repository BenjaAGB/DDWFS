import math
import torch
import random
import numpy as np
from mpmath import *
from tqdm import tqdm
import scipy.io as scio
from scipy.stats import *
import scipy.special as scp
import matplotlib.pyplot as plt

class dC: pass

def cart2pol(x, y): return np.sqrt(x**2 + y**2),np.arctan2(y, x)

def GetTurbulenceParameters(wfs, atm, r0, device='cpu'):

    nPx = wfs.nPx
    modulation = wfs.modulation
    D = wfs.D

    # atm
    N = wfs.fovPx
    L = (N-1)*D/(nPx-1)
    binning = atm.binning
    nLenslet = atm.nLenslet
    fc = 1/binning*0.5*(nLenslet)/D
    resAO = atm.resAO
    n_lvl = atm.n_lvl
    L0 = atm.L0
    fR0 = atm.fR0
    noiseVariance = atm.noiseVariance
    nTimes = atm.nTimes

    # computation
    fx,fy = freqspace(resAO,"meshgrid")
    fx = fx*fc + 1e-7
    fy = fy*fc + 1e-7
    Rx, Ry    = PerformRxRy(fx,fy,fc,nLenslet+1,D,r0,L0,fR0,modulation,binning,noiseVariance)
    psdFit    = fittingPSD(fx,fy,fc,"square",nTimes,r0,L0,fR0,D)/r0**(-5/3)
    psdNoise  = noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D)/noiseVariance
    fxExt,fyExt = freqspace(np.size(fx,1)*nTimes,"meshgrid")
    fxExt = fxExt*fc*nTimes
    fyExt = fyExt*fc*nTimes
    index = np.logical_and(np.absolute(fxExt)<fc,np.absolute(fyExt)<fc)
    SxAv,SyAv = SxyAv(fx,fy,D,nLenslet)
    psdAO_mean = np.zeros((np.size(fxExt,0),np.size(fxExt,1)))
    aSlPSD = anisoServoLagPSD(fx,fy,fc,Rx,Ry,SxAv,SyAv,r0,L0,fR0,D)
    psdFact = aSlPSD  + psdNoise*np.mean(n_lvl)
    psdAO_mean[index] = psdFact.flatten()
    fx,fy = freqspace(N,"meshgrid")
    fr,a  = cart2pol(fx,fy)
    fr    = np.fft.fftshift(fr*(N-1)/L/2)
    psdAO_mean = psdAO_mean + psdFit*r0**(-5/3)
    N = np.int32(N)
    fourierSampling = 1/L
    idx = np.where(fr.flatten() == 0)
    idx = idx[0][0]+1
    atm = {"psdAO_mean":psdAO_mean, "N":N, "fourierSampling":fourierSampling, "idx":idx, "pupil":wfs.pupil, "nPx":(nPx)}
    return(atm)

def freqspace(n,flag):
    n = [n,n]
    a = np.arange(0,n[1],1)
    b = np.arange(0,n[0],1)
    f1 = (a-np.floor(n[1]/2))*(2/(n[1]))
    f2 = (b-np.floor(n[0]/2))*(2/(n[0]))
    f1,f2 = np.meshgrid(f1,f2,indexing='ij')
    return(f1,f2)

def PerformRxRy(fx,fy,fc,nActuator,D,r0,L0,fR0,modulation,binning,noiseVariance):
    nL      = nActuator - 1
    d       = D/nL
    f       = np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2)
    Wn      = noiseVariance/(2*fc)**2
    Wphi    = spectrum(f,r0,L0,fR0)
    u       = fx  
    umod = 1/(2*d)/(nL/2)*modulation
    Sx = np.zeros((np.size(u,0),np.size(u,1)),dtype=complex)
    idx = np.absolute(u) > umod
    Sx[idx] = 1j*np.sign(u[idx])
    idx = np.absolute(u) <= umod
    Sx[idx] = 2*1j/math.pi*np.arcsin(u[idx]/umod)
    Av = np.sinc(binning*d*u)*np.transpose(np.sinc(binning*d*u))
    Sy = np.transpose(Sx)
    SxAv = Sx*Av
    SyAv = Sy*Av

    #reconstruction filter
    AvRec = Av
    SxAvRec = Sx*AvRec
    SyAvRec = Sy*AvRec
    # --------------------------------------
    #   MMSE filter
    # --------------------------------------
    gPSD = np.absolute(Sx*AvRec)**2 + np.absolute(Sy*AvRec)**2 + Wn/Wphi +1e-7
    Rx = np.conj(SxAvRec)/gPSD
    Ry = np.conj(SyAvRec)/gPSD
    return(Rx,Ry)

def spectrum(f,r0,L0,fR0):
    out = (24.*math.gamma(6/5)/5)**(5./6)*(math.gamma(11/6)**2/(2*math.pi**(11/3)))*r0**(-5/3)
    out = out*(f**2 + 1/L0**2)**(-11/6)
    out = fR0*out
    return(out)

def fittingPSD(fx,fy,fc,shape,nTimes,r0,L0,fR0,D):
    fx,fy = freqspace(np.size(fx,1)*nTimes,"meshgrid")
    fx = fx*fc*nTimes
    fy = fy*fc*nTimes
    out = np.zeros((np.size(fx,0),np.size(fx,1)))
    if shape == "square":
          index  = np.logical_or((np.absolute(fx)>fc),(np.absolute(fy)>fc))        
    else:
          index  = np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2) > fc
    f     = np.sqrt(np.absolute(fx[index])**2+np.absolute(fy[index])**2)
    out[index] = spectrum(f,r0,L0,fR0)
    out = out*pistonFilter(D,np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2))
    return(out)

def pistonFilter(D,f):
    red = math.pi*D*f
    sm = sombrero(1,red)
    out = 1 - 4*sm**2
    return(out)

def noisePSD(fx,fy,fc,Rx,Ry,noiseVariance,D):
    out   = np.zeros((np.size(fx,0),np.size(fx,1)))
    if noiseVariance>0:
       index = np.logical_not(np.logical_and(np.logical_or((np.absolute(fx)>fc),(np.absolute(fy)>fc)),np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2)>0))
       out[index] = noiseVariance/(2*fc)**2*(np.absolute(Rx[index])**2 + np.absolute(Ry[index])**2)
       out = out*pistonFilter(D,np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2))
       return(out)
    
def SxyAv(fx,fy,D,nLenslet):
    d = D/(nLenslet)
    Sx = 1j*2*math.pi*fx*d 
    Sy = 1j*2*math.pi*fy*d 
    Av = np.sinc(d*fx)*np.sinc(d*fy)*np.exp(1j*math.pi*d*(fx+fy))
    SxAv = Sx*Av
    SyAv = Sy*Av
    return(SxAv,SyAv)

def anisoServoLagPSD(fx,fy,fc,Rx,Ry,SxAv,SyAv,r0,L0,fR0,D):
    out   = np.zeros((np.size(fx,0),np.size(fx,1)))
    index = np.logical_not(np.logical_or((np.absolute(fx)>=fc),(np.absolute(fy)>=fc)))
    pf = pistonFilter(D,np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2))
    fx     = fx[index]
    fy     = fy[index]
    rtf = 1
    out[index] =  rtf * spectrum(np.sqrt(np.absolute(fx)**2+np.absolute(fy)**2),r0,L0,fR0)
    out = pf*out.real
    return(out)

def GetPhaseZernike(psdAO_mean,N,fourierSampling,idx,pupil,nPx,CM, wfs):
    device = pupil.device
    randMap = torch.randn(((N),(N)), device=device)
    phaseMap = torch.fft.ifft2(idx*torch.sqrt(torch.fft.fftshift(psdAO_mean))*torch.fft.fft2(randMap)/N)*fourierSampling
    phaseMap = phaseMap.real*(N**2)
    phaseMap = pupil*phaseMap[0:(nPx),0:(nPx)]
    Ze = torch.matmul(CM,torch.reshape(phaseMap,[-1]))
    return phaseMap,Ze

def sombrero(n,x):
    if n==0:
       out = besselj(0,x)/x
    else:
       if n>1:
          out = np.zeros((np.size(x,0),np.size(x,1)))
       else:
          out = 0.5*np.ones((np.size(x,0),np.size(x,1)))
       u = x != 0
       x = x[u]
       aux = scp.j1(x)  #order 1 bessel
       out[u] = aux/x
       return(out)

def getATM(wfs, p, atm_path, **kwargs):
    nData = np.sum(p['nData'])
    random_state = kwargs.get('random_state',42) ####### se puede cambiar
    batch_atm = 5000
    batch = int( min(nData, kwargs.get('batch',batch_atm)) )
    if not (nData%batch ==0): raise TypeError('Residue of nData/batch must be zero')

    # beta distribution
    r = p['r0'][0]+(p['r0'][1]-p['r0'][0])*beta.rvs(p['ab'][0],p['ab'][1], size=nData, random_state=random_state)
    fig, ax = plt.subplots()
    ax.hist(r, bins=30, density=True, alpha=0.6, color='b', edgecolor='k', label='Histogram')
    x = np.linspace(p['r0'][0], p['r0'][1], nData)
    pdf_values = beta.pdf((x - p['r0'][0]) / (p['r0'][1] - p['r0'][0]), p['ab'][0], p['ab'][1]) / (p['r0'][1] - p['r0'][0])
    ax.plot(x, pdf_values, 'r', linewidth=2, label='PDF')
    ax.set_xlabel('$r_0$')
    ax.set_xlim(p['r0'][0], p['r0'][1])
    ax.set_ylabel('Frequency')
    ax.set_title('Beta Distribution ({}{}) in Dr0=[{},{}] (nData={})'.format(p['ab'][0],p['ab'][1],p['Dr0'][0],p['Dr0'][1],nData))
    ax.legend()
    fig.savefig(atm_path+'/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ITERATION 
    print(f"Generating dataset nData={nData} | batch={batch} | parts={nData//batch}")
    for t in range(int(nData//batch)):
        b = (np.arange(batch*t,batch*(t+1))).astype(np.int32)
        PHI,ZGT = ATM(wfs,r0v=r[b],nData=1)
        ZGT = ZGT.squeeze()# T[r0,z]
        scio.savemat(atm_path+f'/part_{t}.mat', {'Phi':PHI.cpu().numpy(), 'Zgt':ZGT.cpu().numpy()})
    print("Dataset finished!") 

    # verify all is OK
    figs,axs = plt.subplots(1,2,figsize=(5*2,5))
    irand = random.randint(0, batch-1)
    im0 = axs[0].imshow(PHI[irand,0,:,:].cpu())
    im1 = axs[1].imshow( (wfs.modes@ZGT[irand,:].t()).reshape((wfs.nPx),(wfs.nPx)).cpu() )
    plt.colorbar(im0,ax=axs[0])
    plt.colorbar(im1,ax=axs[1])
    plt.savefig(atm_path+f'/OK.png', dpi=300, bbox_inches='tight')
    plt.close(figs)

    return PHI,ZGT

def ATM(wfs,r0v=1,nData=1):
    atm = dC()
    atm.nLenslet      = 16              
    atm.resAO         = 2*atm.nLenslet+1       
    atm.L0            = 25
    atm.fR0           = 1
    atm.noiseVariance = 0.7
    atm.n_lvl         = 0.1
    atm.binning       = 1
    atm.nTimes        = wfs.fovPx/atm.resAO

    PHI = torch.empty((len(r0v), nData, (wfs.nPx), (wfs.nPx)), dtype = wfs.precision.real, device = wfs.device)
    ZGT = torch.empty((len(r0v), nData, len(wfs.jModes)), dtype = wfs.precision.real, device = wfs.device)

    PIMat_gt = torch.linalg.pinv(torch.tensor(wfs.modes.clone().detach() ,dtype = wfs.precision.real, device = wfs.device))

    total_atm = len(r0v) * nData
    bar_format = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
    with tqdm(total = total_atm, desc = 'Generating ATM', bar_format = bar_format) as pbar_atm:
        for r in range(len(r0v)):
            atmP = GetTurbulenceParameters(wfs, atm, r0v[r], device = wfs.device)
            psdAO_mean = torch.tensor(atmP['psdAO_mean'], dtype = wfs.precision.real, device = wfs.device)
            N = torch.tensor(atmP['N'], dtype = wfs.precision.int, device = wfs.device)
            fourierSampling = torch.tensor(atmP['fourierSampling'], dtype = wfs.precision.real, device = wfs.device)
            idx = torch.tensor(atmP['idx'], dtype = wfs.precision.int, device = wfs.device)
            pupil = torch.tensor(atmP['pupil'], dtype = wfs.precision.real, device = wfs.device)
            nPx = torch.tensor(atmP['nPx'], dtype = wfs.precision.int, device = wfs.device)
            for t in range(nData):
                PHI[r,t,:,:], ZGT[r,t,:] = GetPhaseZernike(psdAO_mean, N, fourierSampling, idx, pupil, nPx, PIMat_gt, wfs.precision)
                pbar_atm.update(1)
    return PHI,ZGT