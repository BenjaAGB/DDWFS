import math
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from Functions.utils import *
from torch import squeeze as NZ
from torch import unsqueeze as UNZ
from Functions.functions_wfs import *

class WFS(nn.Module):
    def __init__(self, params,**kwargs):
        super().__init__()

        self.tag            = 'major_wfs'
        self.device         = kwargs.get('device', 'cpu')
        self.precision      = params.precision
        self.precision_name = params.precision_name
        
        self.D        = params.D
        self.nPx      = params.nPx
        self.resol_nn = params.resol_nn
        self.fovPx    = params.fovPx
        self.samp     = params.samp

        self.size  = params.size
        self.R     = params.R
        self.f1    = params.f1
        self.f2    = params.f2
        self.nDE   = params.nDE

        self.jModes = params.jModes
        self.modes  = torch.tensor(params.modes, dtype = self.precision.real, device = self.device)
        self.bModes = torch.zeros((len(self.jModes), 1, self.nPx, self.nPx), dtype = self.precision.real, device = self.device)
        for k in range(len(self.jModes)):           
            self.bModes[k, 0:1, : , :] = self.modes[:, k].reshape(self.nPx, self.nPx)

        self.pupil        = torch.tensor(params.pupil)
        self.pupilLogical = torch.tensor(params.pupilLogical, dtype = torch.bool, device = self.device)
        
        self.mInorm  = params.mInorm
        self.amp_cal = torch.tensor(params.amp_cal, dtype = self.precision.real, device = self.device)
        self.vNoise  = params.vNoise
        self.Piston  = UNZ(UNZ(self.pupil, 0), 0).to(self.device).to(self.precision.real)
        
        self.alpha   = params.alpha
        self.wvl     = params.wvl
        self.ps      = params.ps_slm
        self.rooftop = params.rooftop

        self.nHead   = params.nHead

        # self.h = params.h if hasattr(params,'h') else np.pi/2 ##ZWFS
        # self.lD = params.lD if hasattr(params,'lD') else 5    ##ZWFS

        self.crop       = params.crop if hasattr(params,'crop') else False
        self.modulation = params.modulation

        print(f'WFS OBJECT INITIALIZED: nPx_data={self.nPx} | nPx_res={self.fovPx} | nZ={len(params.jModes)} | crop={self.crop} | M={self.modulation} | precision={params.precision_name}')

    # FORWARD I0
    def forward_I0(self, phi, **kwargs):
        if (phi.is_complex() or (len(phi.shape)!=4)): raise ValueError('Input cannot be complex')
        
        vNoise       = kwargs.get('vNoise', self.vNoise)
        fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)

        I = Propagation_Free(self, phi, fourier_mask = fourier_mask)

        I = addNoise(I, vNoise)
        if self.crop and self.nHead == 4:
            I = poscrop(self,I)
        return I
    
    def forward(self, phi, **kwargs):
        if (phi.is_complex() or (len(phi.shape)!=4)): raise ValueError('Input cannot be complex')

        vNoise       = kwargs.get('vNoise', self.vNoise)
        fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)
        DE_layers    = kwargs.get('DE_layers')

        I = Propagation(self, phi, fourier_mask = fourier_mask, DE_layers = DE_layers)

        I = addNoise(I, vNoise)
        if self.crop and self.nHead == 4:
            I = poscrop(self,I)
        return I


    # CALIBRATION
    def Calibration(self, **kwargs): #### cambiar forward #####
        fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)
        mInorm = kwargs.get('mInorm', self.mInorm)# sum normalization
        vNoise = kwargs.get('vNoise',[0.,0.,0.,1.])# asumed average of frame/time exposure
        piston = self.Piston*self.amp_cal
        I0_v = self.forward(piston, fourier_mask=fourier_mask, vNoise=vNoise)[:,0,self.idx]# T[z,1,N,M]->T[z,NM]
        z = self.bModes*self.amp_cal
        mIp_v = self.forward(z, fourier_mask=fourier_mask, vNoise=vNoise)[:,0,self.idx]# T[z,1,N,M]->T[z,NM]
        mIn_v = self.forward(-z, fourier_mask=fourier_mask, vNoise=vNoise)[:,0,self.idx]# T[z,1,N,M]->T[z,NM]
        mIn_v,mIp_v = self.mI(mIn_v,I0_v, dim=1,mInorm=mInorm),self.mI(mIp_v,I0_v, dim=1,mInorm=mInorm)# T[z,NM]-I0
        IMat = ( mIp_v-mIn_v )/(2*self.amp_cal)# T[z,NM]
        PIMat = torch.linalg.pinv( IMat.t() )
        return I0_v, PIMat
    
    # META-INTENSITY
    def mI(self, I,I0, mInorm=1,dim=1):  
        if mInorm:
            return (I-I0)/torch.sum(I, dim=dim, keepdim=True)# T[b,NM]
        else:
            return (I-I0)
    # SETTERS

    # modulation
    @property 
    def modulation(self):
        return self._modulation
    @modulation.setter
    def modulation(self, value):
        self.nTheta,self.ModPhasor = get_modPhasor(fovPx = self.fovPx, samp = self.samp, mod = value, precision = self.precision, device = self.device)
        self._modulation = value
        
    # crop of pupil
    @property
    def crop(self):
        return self._crop
    @crop.setter
    def crop(self, value):
        assert isinstance(value,bool), 'crop must be a boolean variable'
        if value:
            if self.nHead == 4: 
                if self.alpha == 2:
                    if self.nPx == 128:
                        self.idx = (torch.ones((70,70), dtype=torch.bool,device=self.device)>0)# T[N,M]
                    elif self.nPx == 256:
                        self.idx = (torch.ones((140,140), dtype=torch.bool,device=self.device)>0)# T[N,M]
                elif self.alpha == 3:
                    if self.nPx == 128:
                        self.idx = (torch.ones((68,68), dtype=torch.bool,device=self.device)>0)# T[N,M]
                    elif self.nPx == 256:
                        self.idx = (torch.ones((134,134), dtype=torch.bool,device=self.device)>0)# T[N,M]
            elif self.nHead == 3:
                if self.alpha == 3:
                    if self.nPx == 128:
                        self.idx = (torch.ones((70,70), dtype=torch.bool,device=self.device)>0)# T[N,M]
        else:
            self.idx = (torch.ones((self.nPx,self.nPx), dtype=torch.bool,device=self.device)>0)# T[N,M]
        self._crop = value

    # nhead fourier mask
    @property
    def nHead(self):
        return self._nHead
    @nHead.setter
    def nHead(self,value):

        if value==-1:
            print('MAKING A ZWFS')
            self.fourier_mask = zernike_geometry(self, nPx = self.fovPx, h = self.h, lamD = self.lD, device = self.device,
                                                 precision = get_precision(type = 'double')).to(self.precision.complex)
        # elif value==-10:# ragazzoni
        #     img_pil = Image.open("./functions/ragazzoni.jpg").convert("RGB").convert("L")
        #     ar = torch.tensor(np.array(img_pil)[45:255,100:310]).unsqueeze(0).unsqueeze(0)
        #     amp_factor = (self.ps)*((self.fovPx-1)//2) * (2*torch.pi/self.wvl) * np.tan(self.alpha*np.pi/180)
        #     rag = torch.exp( 1j*(f.resize(ar,(self.fovPx,self.fovPx),antialias=True).to(torch.float64)/255)*amp_factor   )
        #     #self.fourier_mask = ( rag/torch.sum( torch.abs(rag.flatten()) ) ).to(self.precision.complex).to(self.device)
        #     self.fourier_mask = torch.fft.fftshift( rag/torch.sum( torch.abs(rag.flatten()) ) ).to(self.precision.complex).to(self.device)

        elif value==-2:# random mask
            #amp_factor = (3.74e3)*((512-1)//2) * (2*torch.pi/635)*np.tan(self.alpha*np.pi/180)
            mask = torch.exp(1j * torch.normal(mean = 0., std = math.sqrt(2/self.fovPx), size = (self.fovPx, self.fovPx), dtype = self.precision.real))
            self.fourier_mask = (mask/torch.sum(torch.abs(mask.flatten()))).unsqueeze(0).unsqueeze(0).to(self.precision.complex).to(self.device)

        else:
            self.fourier_mask,_ = fourier_geometry(nPx = self.fovPx, nhead = value, alpha = self.alpha, wvl = self.wvl,
                                                   ps = self.ps, rooftop = self.rooftop, precision = get_precision(type = 'double'), device = self.device)
            self.fourier_mask = torch.fft.fftshift(self.fourier_mask)
            self.fourier_mask = self.fourier_mask.to(self.precision.complex)

        self._nHead = value