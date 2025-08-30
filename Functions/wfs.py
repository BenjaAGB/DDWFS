import math
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from Functions.utils import *
from torch import squeeze as NZ
from torch import unsqueeze as UNZ
from Functions.functions_wfs import *
import torchvision.transforms.functional as f

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
        self.posDE = params.posDE

        ### DISTANCE DEs Layers ###
        self.dz        = params.dz
        self.dz_after  = params.dz_after
        self.dz_before = params.dz_before

        self.jModes = params.jModes

        # self.modes  = torch.tensor(params.modes, dtype = self.precision.real, device = self.device) ### da advertencia ###
        self.modes = (params.modes.detach().clone() if torch.is_tensor(params.modes) else torch.as_tensor(params.modes)).to(dtype = self.precision.real, device = self.device) ### modificado ###

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
        ## TEST ##
        if DE_layers is None:
            nPx   = getattr(self, 'fovPx', 512)         # tamaño H=W
            dtype = self.precision.real
            dev   = self.device
            DE_layers = [torch.zeros((nPx, nPx), dtype=dtype, device=dev)]# for _ in range(nDE)]

        I = Propagation(self, phi, fourier_mask = fourier_mask, DE_layers = DE_layers)

        if self.resol_nn == 128:
            I = f.resize(I,(128,128), antialias = True, interpolation = f.InterpolationMode.BILINEAR)
        elif self.resol_nn == 256:
            I = f.resize(I,(256,256), antialias = True, interpolation = f.InterpolationMode.BILINEAR)
        elif self.resol_nn == 512:
            I = f.resize(I,(512,512), antialias = True, interpolation = f.InterpolationMode.BILINEAR)
        elif self.resol_nn == 1024:
            I = f.resize(I,(1024,1024),antialias = True, interpolation = f.InterpolationMode.BILINEAR)

        I = addNoise(I, vNoise)
        if self.crop and self.nHead == 4:
            I = poscrop(self,I)
        return I


    # CALIBRATION
    def Calibration(self, **kwargs):  ### modificar nombres ###
        vNoise = kwargs.get('vNoise',[0.,0.,0.,1.])
        fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)
        DE_layers = kwargs.get('DE_layers')
        mInorm = kwargs.get('mInorm', self.mInorm)
        ## TEST ##
        if DE_layers is None:
            nPx   = getattr(self, 'fovPx', 512)         # tamaño H=W
            dtype = self.precision.real
            dev   = self.device
            DE_layers = [torch.zeros((nPx, nPx), dtype=dtype, device=dev)]# for _ in range(nDE)]
        
        with torch.no_grad():
            piston = (self.Piston * self.amp_cal).to(device=self.device, dtype=self.precision.real)
            I0_v = self.forward(piston, fourier_mask=fourier_mask, DE_layers=DE_layers, vNoise=vNoise)[:, 0, self.idx]
            I0_v = I0_v.to(self.device, dtype=self.precision.real, non_blocking=True)

            # z (209 modos) en el dtype/device que usas en todo el sistema
            z_full = (self.bModes * self.amp_cal).to(device=self.device, dtype=self.precision.real)
            Z  = z_full.shape[0]
            NM = self.idx.numel()

            # Reservas en CPU para no llenar VRAM
            mIp_cpu = torch.empty((Z, NM), dtype=self.precision.real, device='cpu')
            mIn_cpu = torch.empty((Z, NM), dtype=self.precision.real, device='cpu')

            chunk = 32  # ajusta según tu GPU (16/32/64)
            for s in range(0, Z, chunk):
                e  = min(s + chunk, Z)
                z  = z_full[s:e]                          # ya en device y dtype correcto
                Ip = self.forward(z,  fourier_mask=fourier_mask, DE_layers=DE_layers, vNoise=vNoise)[:, 0, self.idx]
                In = self.forward(-z, fourier_mask=fourier_mask, DE_layers=DE_layers, vNoise=vNoise)[:, 0, self.idx]

                # Normalización mI en el mismo device/dtype
                Ip = self.mI(Ip, I0_v, dim=1, mInorm=mInorm)
                In = self.mI(In, I0_v, dim=1, mInorm=mInorm)

                # Bajar resultados a CPU y liberar
                mIp_cpu[s:e] = Ip.to('cpu', non_blocking=True)
                mIn_cpu[s:e] = In.to('cpu', non_blocking=True)
                del z, Ip, In
                torch.cuda.empty_cache()

        # IMat y pseudo-inversa en CPU (misma precisión)
        IMat  = (mIp_cpu.to(device = self.device) - mIn_cpu.to(device = self.device)) / (2 * self.amp_cal)
        PIMat = torch.linalg.pinv(IMat.t()) 

        return I0_v, PIMat

        
    # def Calibration(self, **kwargs): #### cambiar forward #####
    #     vNoise = kwargs.get('vNoise',[0.,0.,0.,1.])
    #     fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)
    #     DE_layers = kwargs.get('DE_layers')
    #     mInorm = kwargs.get('mInorm', self.mInorm)
        
    #     piston = self.Piston * self.amp_cal
    #     I0_v = self.forward(piston, fourier_mask = fourier_mask, DE_layers = DE_layers, vNoise = vNoise)[:, 0, self.idx]  # T[z,1,N,M]->T[z,NM]
    #     z = self.bModes * self.amp_cal
    #     mIp_v = self.forward(z, fourier_mask = fourier_mask, DE_layers = DE_layers, vNoise = vNoise)[:, 0, self.idx]  # T[z,1,N,M]->T[z,NM]
    #     mIn_v = self.forward(-z, fourier_mask = fourier_mask, DE_layers = DE_layers, vNoise = vNoise)[:, 0, self.idx]  # T[z,1,N,M]->T[z,NM]
    #     mIn_v, mIp_v = self.mI(mIn_v, I0_v, dim = 1, mInorm = mInorm), self.mI(mIp_v, I0_v, dim = 1, mInorm = mInorm)  # T[z,NM]-I0
    #     IMat = (mIp_v - mIn_v)/(2 * self.amp_cal)  # T[z,NM]
    #     PIMat = torch.linalg.pinv(IMat.t())
    #     return I0_v, PIMat
    
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

        elif value==0:
            self.fourier_mask = torch.ones((1,1,self.fovPx,self.fovPx), dtype = self.precision.complex, device = self.device)

        else: ###### se puede modificar la precicsion ####
            self.fourier_mask,_ = fourier_geometry(nPx = self.fovPx, nhead = value, alpha = self.alpha, wvl = self.wvl,
                                                   ps = self.ps, rooftop = self.rooftop, precision = get_precision(type = 'double'), device = self.device)
            self.fourier_mask = torch.fft.fftshift(self.fourier_mask)
            self.fourier_mask = self.fourier_mask.to(self.precision.complex)

        self._nHead = value