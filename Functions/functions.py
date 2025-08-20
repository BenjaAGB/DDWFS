import math
import torch
import numpy as np
import torch.nn.functional as F

cart2pol_torch = lambda x,y: (torch.sqrt(x**2 + y**2), torch.atan2(y,x))
pol2car_torch = lambda r,o: (r*torch.cos(o), r*torch.sin(o))

def factorial(n): return math.factorial(n)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)  

def CreateTelescopePupil_physical(Npx, radius, extent):

    x = np.linspace(-extent/2, extent/2, Npx)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    R = np.sqrt(X**2 + Y**2)
    return (R <= radius)

def CreateTelescopePupil(Npx):
    x = np.arange(-(Npx-1)/2,(Npx-1)/2+1,1)  # change to +1 if torch of tf
    x = x/np.max(x)
    u = x
    v = x
    x,y = np.meshgrid(u,v,indexing='ij')
    r,o = cart2pol(x,y)      
    out = r <= 1
    return(out)


def CreateZernikePolynomials1(wfs): 
    nPx = wfs.nPx
    jModes = wfs.jModes
    pupilLogical = wfs.pupilLogical
    pupil = torch.tensor(wfs.pupil).numpy()
    u = np.linspace(-1,1,nPx)
    x,y = np.meshgrid(u,u,indexing='ij')
    r,theta = cart2pol(x,y) 

    def zrf(n,m,r):# Zernike radial function
        R = np.zeros_like(r)
        for k in range(0, int((n-m)/2)+1):
            num = (-1)**k * factorial( int(n)-k)
            den = factorial(k) * factorial( int((n+m)/2)-k ) * factorial( int((n-m)/2)-k )
            R += (num/den) * r**(n-2*k)
        return R
    
    def j2nm(j):
        n = int( ( -1.+np.sqrt( 8*(j-1)+1 ) )/2. )
        p = ( j-( n*(n+1) )/2. )
        k = n%2
        m = int((p+k)/2.)*2 - k
        if m!=0:
            if j%2==0:
                s = 1
            else:
                s=-1
            m *= s 
        return n,m
    
    zMode = np.zeros((np.prod(wfs.pupil.shape),len(jModes)))
    for i,j in enumerate(jModes):
        # nm2j
        n,m = j2nm(j)
        #
        if m == 0:
            Z = np.sqrt( (n+1) )*zrf(n,0,r)
        else:
            if m > 0:# j is even
                Z = np.sqrt(2*(n+1))*zrf(n,m,r) * np.cos( m*theta )
            else:# j is odd
                m = np.abs(m)
                Z = np.sqrt(2*(n+1))*zrf(n,m,r) * np.sin( m*theta )
        zMode[:,i] = (Z*pupil).flatten()

    return zMode

def CreateZernikePolynomials_camilo(wfs):
    nPx = wfs.nPx
    jModes = wfs.jModes
    pupilLogical = wfs.pupilLogical
    u = nPx
    u = np.linspace(-1,1,u)
    v = u
    x,y = np.meshgrid(u,v,indexing='ij')
    r,o = cart2pol(x,y) 
    mode = jModes
    nf = [0]*len(mode)
    mf = [0]*len(mode)
    for cj in range(len(mode)):
        j = jModes[cj]
        n  = 0
        m  = 0
        j1 = j-1
        while j1 > n:
            n  = n + 1
            j1 = j1 - n
            m  = (-1)**j * (n%2 + 2*np.floor((j1+(n+1)%2)/2))
        nf[cj] = np.int32(n)
        mf[cj] = np.int32(np.abs(m))
    nv = np.array(nf)
    mv = np.array(mf)
    nf  = len(jModes)
    fun = np.zeros((np.size(r),nf))
    r = np.transpose(r)
    o = np.transpose(o)
    r = r[pupilLogical]
    o = o[pupilLogical]
    pupilVec = pupilLogical.flatten()
    def R_fun(r,n,m):
        R=np.zeros(np.size(r))
        sran = int((n-m)/2)+1# ver como poner np.int32
        for s in range(sran):
            Rn = (-1)**s*np.prod(np.arange(1,(n-s)+1,dtype=float))*r**(n-2*s)
            Rd = (np.prod(np.arange(1,s+1))*np.prod(np.arange(1,((n+m)/2-s+1),dtype=float))*np.prod(np.arange(1,((n-m)/2-s)+1)))
            R = R + Rn/Rd
        return(R)    
    ind_m = list(np.array(np.nonzero(mv==0))[0])
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)
    mod_mode = jModes%2
    ind_m = list(np.array(np.nonzero(np.logical_and(mod_mode==0,mv!=0))))
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.cos(m*o)
    ind_m = list(np.array(np.nonzero(np.logical_and(mod_mode==1,mv!=0))))
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.sin(m*o)
    modes = fun
    return(modes)

def center_pad(x, N_target, value=0):

    if isinstance(x, np.ndarray):
        H, W = x.shape[-2], x.shape[-1]
        assert N_target >= H and N_target >= W, "N_target debe ser >= tamaño actual"
        dH, dW = N_target - H, N_target - W
        pT, pB = dH // 2, dH - dH // 2
        pL, pR = dW // 2, dW - dW // 2
        pad_width = [(0,0)] * (x.ndim - 2) + [(pT, pB), (pL, pR)]
        return np.pad(x, pad_width, mode='constant', constant_values=value)

    elif torch.is_tensor(x):
        H, W = x.shape[-2], x.shape[-1]
        assert N_target >= H and N_target >= W, "N_target debe ser >= tamaño actual"
        dH, dW = N_target - H, N_target - W
        pT, pB = dH // 2, dH - dH // 2
        pL, pR = dW // 2, dW - dW // 2
        # F.pad: paddings = (left, right, top, bottom)
        return F.pad(x, (pL, pR, pT, pB), mode='constant', value=value)

    else:
        raise TypeError("x debe ser np.ndarray o torch.Tensor")
    
def sample_phi(prs, strength=2.0, target_wfe_nm=None, pupil=None):
    n     = prs.modes.shape[1]
    dev   = prs.device
    dtype = prs.precision.real

    irand = torch.randperm(n, device=dev)

    j = torch.arange(1, n+1, device=dev, dtype=torch.float32)
    n_rad = torch.ceil((-3.0 + torch.sqrt(9.0 + 8.0*(j-1.0))) / 2.0)
    w = (n_rad + 1.0).pow(-11.0/6.0)
    w = w[irand]

    c = torch.randn(n, device=dev, dtype=dtype) * w.to(dtype) * float(strength)

    modes = prs.modes[:, irand].to(dev, dtype=dtype)      # [N^2, n]
    phi   = torch.sum(c * modes, dim=-1).reshape(1,1,prs.nPx,prs.nPx)  # rad

    if target_wfe_nm is not None:
        lam = getattr(prs, 'wavelength', 635e-9)  # [m]
        if pupil is None:
            mask = (modes[:,0].reshape(prs.nPx, prs.nPx) != 0).to(dev)
        else:
            mask = pupil.to(dev).bool()
        m = mask[None,None]

        phi = phi - phi[m].mean()
        rms_now = torch.sqrt((phi[m]**2).mean()) + 1e-12
        rms_target = 2*math.pi*(float(target_wfe_nm)*1e-9)/lam
        phi = phi * (rms_target / rms_now)

    return phi
