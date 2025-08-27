import sys
import torch
import numpy as np
from Functions.utils import *
from Functions.functions import *
from Functions.propagator import *
from torch import unsqueeze as UNZ

def fourier_geometry(nPx, nhead, alpha, wvl, ps, rooftop, precision, device):

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

    nPx = torch.tensor(nPx, dtype=precision.int) 
    wvl = torch.tensor((wvl), dtype=precision.real)
    alpha = torch.tensor(alpha*torch.pi/180, dtype=precision.real,device=device)
    ps = torch.tensor(ps, dtype=precision.real,device=device)   
    rooftop = (torch.tensor(rooftop,dtype=precision.real,device=device)*ps)
    nhead = torch.tensor(nhead, dtype=precision.int)
    
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

def zernike_geometry(obj, h, lamD, **kwargs):
    precision = kwargs.get('precision',get_precision(type='double')) 
    nPx = torch.tensor(kwargs.get('nPx',128), dtype=precision.int)
    ps = torch.tensor(kwargs.get('ps',3.74e3), dtype=precision.real,device=obj.device)   
    #
    pyrPupil = (obj.Piston.to(precision.complex))*torch.exp(1j*obj.Piston.to(precision.real))
    subscale = torch.tensor( (1/(2*obj.samp)) ,dtype=precision.real)
    sx = torch.tensor(np.round(nPx*subscale), dtype=precision.int)# doesnt require device 
    npv = torch.tensor( (nPx-sx)/2 ,dtype=precision.int)
    PyrQ = torch.nn.functional.pad(pyrPupil, (npv,npv,npv,npv), "constant", 0)
    psf = torch.fft.fftshift( torch.abs(torch.fft.fft2(PyrQ))**2 ).squeeze()# T[b,c,NM]->T[N,M]

    # grid
    x = torch.linspace( -(nPx-1)//2,(nPx-1)//2,nPx, dtype=precision.real,device=obj.device)
    # compute FWHM
    FWHM = torch.sum( ( psf[psf.shape[0]//2,:]/torch.max(psf[psf.shape[0]//2,:]) )>=.5 )# sum of pixels 1D
    R_FWHM = (FWHM)/2# radius from the center (-1) 1D
    #
    h = torch.tensor(h, dtype=precision.real)
    lamD = torch.tensor(lamD, dtype=precision.real)
    X,Y = torch.meshgrid(x,x, indexing='ij')
    R = torch.sqrt(X**2+Y**2)
    phase = (R<=lamD*R_FWHM).to(precision.real)
    fourierMask = torch.fft.fftshift( torch.exp(1j*h*phase) )
    #
    return fourierMask

def get_modPhasor(fovPx, samp, mod, precision, device, **kwargs):
    """
    This function generates the modulation tensor which is used in propagation:
    output: T[1,nTheta,fovPx,fovPx]
    """
    x = torch.arange(0,fovPx,1, device=device,dtype=precision.real)/torch.tensor(fovPx,device=device)
    vv, uu = torch.meshgrid(x,x,indexing='ij')
    r,o = cart2pol_torch(uu,vv)
    nTheta = torch.tensor(np.round(2*np.pi*samp*mod), dtype=precision.int,device=device)
    if nTheta == 0:
        ModPhasor = torch.exp( -1j*torch.zeros((1,1,fovPx,fovPx), dtype=precision.real,device=device) )
    else:
        ModPhasor = torch.zeros((1,nTheta,fovPx,fovPx), dtype=precision.complex,device=device)
        for kTheta in range(nTheta):
            theta = 2*(kTheta)*torch.pi/nTheta
            ph = 4*torch.pi*mod*samp*r*torch.cos(o+theta)# type promotion int->float
            ModPhasor[0,kTheta,:,:] = torch.exp(-1j*ph)
    return nTheta,ModPhasor

def Propagation_Free(obj, phi, fourier_mask):

    phi = center_pad(phi, obj.fovPx)

    Field = TorchField.begin(wavelength = obj.wvl, grid_size = obj.size, Npx = obj.fovPx, precision = obj.precision_name, batch = phi.shape[0], device= obj.device)
    Field_aperture = Field.circ_aperture(R=obj.R)

    Field_aperture = Field_aperture.apply_phase(phi)

    Field_propagated = Field_aperture.propagate_asm_pad(z = obj.f1)
    Field_lens1 = Field_propagated.lens(f = obj.f1)
    Field_propagated = Field_lens1.propagate_asm_pad(z = obj.f1)

    ## PSF ##
    Field_propagated.field *= fourier_mask 
    ## PSF ##

    Field_propagated = Field_propagated.propagate_asm_pad(z = obj.f2)
    Field_lens2 = Field_propagated.lens(f = obj.f2)
    Field_propagated = Field_lens2.propagate_asm_pad(z = obj.f2)

    I = torch.abs(Field_propagated.field)**2

    return I

# def Propagation(obj, phi, fourier_mask, DE_layers):
#     phi = center_pad(phi, obj.fovPx)

#     # obj.DE_layers = kwargs.get('DE_layers', obj.DE_layers)

#     Field = TorchField.begin(wavelength = obj.wvl, grid_size = obj.size, Npx = obj.fovPx, precision = obj.precision_name, batch = phi.shape[0], device= obj.device)
#     Field_aperture = Field.circ_aperture(R=obj.R)

#     Field_aperture = Field_aperture.apply_phase(phi)

#     Field_propagated = Field_aperture.propagate_asm_pad(z = obj.f1)
#     Field_lens1 = Field_propagated.lens(f = obj.f1)
#     Field_propagated = Field_lens1.propagate_asm_pad(z = obj.f1)

#     # --- PSF --- #
#     Field_propagated.field *= fourier_mask

#     if obj.nDE <= 0:
#         Field_propagated = Field_propagated.propagate_asm_pad(z=obj.f2)
#     else:
#         Field_propagated.field *= UNZ(UNZ(torch.exp(1j * DE_layers[0]), 0), 0)
#         if obj.nDE == 1:
#             Field_propagated = Field_propagated.propagate_asm_pad(z=obj.f2)
#         else:
#             delta = obj.f2 / obj.nDE
#             delta_acum = 0
#             for i in range(1, obj.nDE):
#                 Field_propagated = Field_propagated.propagate_asm_pad(z=delta)
#                 Field_propagated.field *= UNZ(UNZ(torch.exp(1j * DE_layers[i]), 0), 0)

#                 delta_acum = delta_acum + delta 
#             Field_propagated = Field_propagated.propagate_asm_pad(z=delta)
#             delta_acum = delta_acum + delta
            
#     Field_lens2      = Field_propagated.lens(f=obj.f2)
#     Field_propagated = Field_lens2.propagate_asm_pad(z=obj.f2)

#     I = torch.abs(Field_propagated.field)**2

#     return I

def Propagation(obj, phi, fourier_mask, DE_layers):

    phi = center_pad(phi, obj.fovPx)

    # --- FIELD INITIALIZATION --- #
    Field = TorchField.begin(wavelength = obj.wvl, grid_size = obj.size, Npx = obj.fovPx, precision = obj.precision_name, batch = phi.shape[0], device= obj.device)
    Field_aperture = Field.circ_aperture(R=obj.R)

    Field_aperture = Field_aperture.apply_phase(phi)
    # --- FIELD INITIALIZATION --- #

    Field_propagated = Field_aperture.propagate_asm_pad(z = obj.f1)
    Field_lens1 = Field_propagated.lens(f = obj.f1)

    if hasattr(DE_layers, "__len__") and not torch.is_tensor(DE_layers):
        de_list = list(DE_layers)
    else:
        de_list = [DE_layers]

    if not obj.posDE:

        Field_propagated = Field_lens1.propagate_asm_pad(z = obj.f1)

        # --- PSF --- #
        Field_propagated.field *= fourier_mask
        # --- PSF --- #
        
        if len(de_list) <= 0: 
            Field_propagated = Field_propagated.propagate_asm_pad(z = obj.f2)
        else:
            Field_propagated.field *= UNZ(UNZ(torch.exp(1j * de_list[0]), 0), 0)
            if len(de_list) == 1:
                Field_propagated = Field_propagated.propagate_asm_pad(z = obj.f2)
            else:
                delta = obj.f2 / len(de_list)
                delta_acum = 0
                for i in range(1, len(de_list)):
                    Field_propagated = Field_propagated.propagate_asm_pad(z = delta)
                    Field_propagated.field *= UNZ(UNZ(torch.exp(1j * de_list[i]), 0), 0)
                    delta_acum = delta_acum + delta

                Field_propagated = Field_propagated.propagate_asm_pad(z = delta)
                delta_acum = delta_acum + delta

        Field_lens2 = Field_propagated.lens(f = obj.f2)
        Field_propagated = Field_lens2.propagate_asm_pad(z = obj.f2)

        I = torch.abs(Field_propagated.field) ** 2

        return I
    
    if len(de_list) != len(obj.posDE):
        raise ValueError('The number of diffractive elements does not match the number of positions specified.')
    
    pos2layer = {int(p): de_list[i] for i, p in enumerate(obj.posDE)}

    neg_positions  = sorted([-p for p in obj.posDE if p < 0], reverse=True)
    pos_positions  = sorted([p for p in obj.posDE if p > 0])
    zero_positions = (0 in pos2layer)

    if obj.dz is not None:
        step_before = obj.dz
        step_after  = obj.dz
    else:
        step_before = obj.dz_before
        step_after  = obj.dz_after

        if step_before is None and len(neg_positions) > 0:
            step_before = obj.f1 / (neg_positions[0] + 1)
        if step_after is None and len(pos_positions) > 0:
            step_after = obj.f2 / (pos_positions[-1] + 1)

    eps = 0.0

    # --- Lens1 -> PSF --- #
    if len(neg_positions) == 0:
        Field_propagated = Field_lens1.propagate_asm_pad(z = obj.f1)
    else:
        delta_len_1 = obj.f1 - neg_positions[0] * step_before

        if delta_len_1 + eps < 0:
            delta_len_1 = 0.0
            
        Field_propagated = Field_lens1.propagate_asm_pad(z = delta_len_1)
        delta_len_1_acum = delta_len_1

        for i, n_pos in enumerate(neg_positions):
            layer = pos2layer[-n_pos]
            Field_propagated.field *= UNZ(UNZ(torch.exp(1j * layer), 0), 0)
            next_n_pos = neg_positions[i + 1] if i + 1 < len(neg_positions) else 0
            dz_neg = (n_pos - next_n_pos) * step_before

            if dz_neg > 0:
                Field_propagated = Field_propagated.propagate_asm_pad(z = dz_neg)

            delta_len_1_acum = delta_len_1_acum + dz_neg
    # --- Lens1 -> PSF --- #

    # --- PSF --- #
    Field_propagated.field *= fourier_mask
    if zero_positions:
        Field_propagated.field *= UNZ(UNZ(torch.exp(1j * pos2layer[0]), 0), 0)
    # --- PSF --- #

    # --- PSF -> Lens2 --- #
    if len(pos_positions) == 0:
        Field_propagated = Field_propagated.propagate_asm_pad(z = obj.f2)
    else:
        first_p = pos_positions[0]
        delta_len_2 = first_p * step_after

        if delta_len_2 + eps < 0:
            delta_len_2 = 0.0
        Field_propagated = Field_propagated.propagate_asm_pad(z = delta_len_2)
        delta_len_2_acum = delta_len_2

        for i, p_pos in enumerate(pos_positions):
            layer = pos2layer[p_pos]
            Field_propagated.field *= UNZ(UNZ(torch.exp(1j * layer), 0), 0)

            if i + 1 < len(pos_positions):
                dz_pos = (pos_positions[i + 1] - p_pos) * step_after
            else:
                dz_pos = obj.f2 - p_pos * step_after

                if dz_pos < 0:
                    dz_pos = 0.0

            if dz_pos > 0:
                Field_propagated = Field_propagated.propagate_asm_pad(z = dz_pos)
            
            delta_len_2_acum = delta_len_2_acum + dz_pos
    # --- PSF -> Lens2 --- #

    Field_lens2 = Field_propagated.lens(f = obj.f2)
    Field_propagated = Field_lens2.propagate_asm_pad(z = obj.f2)

    I = torch.abs(Field_propagated.field)**2

    return I

def norm_I(var, I, norm_nn = None):
    if norm_nn==None:
        return I
    
    elif norm_nn=='max':
        return I/torch.amax(I,dim=(-2,-1),keepdim=True)
    
    elif norm_nn=='sum':
        return (I/torch.sum(I,dim=(-2,-1),keepdim=True)) * var.pupil_sum
    
    elif norm_nn=='rms':
        return I/torch.sqrt( torch.sum(I**2, dim=(-2,-1),keepdim=True)/var.pupil_sum )

    elif norm_nn=='zscore':
        return (I-torch.mean(I,dim=(-2,-1),keepdim=True))/torch.std(I,dim=(-2,-1),keepdim=True)
    
    elif norm_nn=='zmI':
        return (I-var.I0)/torch.std((I-var.I0),dim=(-2,-1),keepdim=True)

def addNoise(iwfs, vNoise):  ################ revisar como funciona ################
    # camilo lo tiene [1,1,1e1,1]   
    if vNoise[1]:
        iwfs = torch.poisson(iwfs*vNoise[2])/vNoise[2]
    iwfs = vNoise[3]*iwfs

    if isinstance(vNoise[0],list):
        rn = torch.tensor(np.random.uniform(vNoise[0][0],vNoise[0][1]))
    else:
        rn = vNoise[0]

    owfs = iwfs + torch.normal(0,rn,size=iwfs.shape,device=iwfs.device)
    owfs.clamp_(min=0)

    return owfs 

def poscrop(obj,I): ##################### posiblemente hay que cambiarla #####################
    if obj.nHead == 4:
        if obj.alpha == 2:
            if obj.nPx == 128:
                w = 35
                p1 = I[...,28:28+w,28:28+w]# sino con 25 y 40
                p2 = I[...,28:28+w,65:65+w]
                p3 = I[...,65:65+w,28:28+w]
                p4 = I[...,65:65+w,65:65+w]
            elif obj.nPx == 256:
                w = 70
                p1 = I[...,55:55+w,55:55+w]# sino con 25 y 40
                p2 = I[...,55:55+w,130:130+w]
                p3 = I[...,130:130+w,55:55+w]
                p4 = I[...,130:130+w,130:130+w]
        elif obj.alpha == 3:
            if obj.nPx==128:
                p1 = I[...,19:53,19:53]
                p2 = I[...,19:53,75:109]
                p3= I[...,75:109,19:53]
                p4 = I[...,75:109,75:109]
            elif obj.nPx==256:
                p1 = I[...,38:105,38:105]
                p2 = I[...,38:105,151:218]
                p3= I[...,151:218,38:105]
                p4 = I[...,151:218,151:218]
        h1 = torch.concat((p1,p2),dim=-1)
        h2 = torch.concat((p3,p4),dim=-1)
        out = torch.concat((h1,h2),dim=-2)   
    elif obj.nHead == 3:
        if obj.alpha == 3:
            if obj.nPx == 128:
                w = 35
                p1 = I[...,27:27+w,13:13+w]
                p2 = I[...,27:27+w,81:81+w]
                p3 = I[...,86:86+w,47:47+w]
                out = torch.zeros((*I.shape[:2],2*w,2*w), device=obj.device)
                out[...,0:w,0:w] = p1
                out[...,0:w,w:2*w] = p2
                out[...,w:2*w,w//2:3*w//2] = p3 
    return out