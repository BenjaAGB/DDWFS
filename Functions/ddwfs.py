import torch
import importlib
import torch.nn as nn
from Functions.wfs import *
from Functions.utils import *
from Functions.functions_wfs import *
from Functions.functions_nn import select_nn


# from model.pyr import *
# from model.functions_model import *



class FWFS(nn.Module):
    def __init__(self, params, **kwargs):
        super(FWFS,self).__init__() 

        self.fp             = params.fp if hasattr(params,'fp') else [(0,0),(0,0)]
        self.train_state    = [second_value > 0 for _, second_value in self.fp]
        self.init           = params.init if hasattr(params,'init') else ['constant',0]
        self.device         = params.device if hasattr(params,'device') else kwargs.get('device','cpu')
        self.precision_name = params.precision_name if hasattr(params,'precision_name') else 'single'
        self.precision      = params.precision if hasattr(params,'precision') else get_precision(type='single')
        self.nDE            = params.nDE if hasattr(params,'nDE') else 1

        ### WFS MODEL ###
        self.wfs = WFS(params, device=self.device)

        self.mInorm = self.wfs.mInorm
        self.norm_nn = params.norm_nn if hasattr(params,'norm_nn') else None
        self.pupil_sum = torch.sum(self.wfs.pupil).to(self.precision.real)
        # self.nRr = params.nRr if hasattr(params,'nRr') else False

        ### DE elements ###
        if self.train_state[0]:

            self.DE_layers = nn.ParameterList()

            for _ in range(self.nDE):
                de = torch.empty((self.wfs.fovPx, self.wfs.fovPx), dtype=self.precision.real, device=self.device)

                if self.init[0] == 'kaiming':
                    print('KAIMING')
                    nn.init.kaiming_normal_(de, mode='fan_in')
                    # with torch.no_grad():
                    #     de.mul_(0.05)
                else:
                    print('CONSTANT', self.init[1])
                    nn.init.constant_(de, float(self.init[1]))

                self.DE_layers.append(nn.Parameter(de))
                # self.wfs.DE_layers = self.DE_layers

        # self.register_buffer('DE_dummy', torch.zeros((self.wfs.fovPx, self.wfs.fovPx), dtype=self.precision.real, device=self.device))
        self.DE_dummy = torch.zeros((self.wfs.fovPx, self.wfs.fovPx), dtype = self.precision.real, device = self.device)

        ### NN ###
        if self.train_state[1]:

            self.NN = select_nn(self)
            # method = importlib.import_module('models.GcVit')

            # self.NN = method.GCViT(num_classes = len(self.wfs.jModes), depths = [2,2,6,2], num_heads = [2,4,8,16], window_size = [16, 16, 32, 16],
            #                         resolution = self.wfs.resol_nn, in_chans = 1, dim = 64, mlp_ratio = 3, drop_path_rate = 0.2).to(self.precision.real).to(self.device)
    
        self.k = torch.tensor((2*torch.pi)/params.wvl, dtype = self.precision.real, device = self.device)
        self.I0 = self.wfs.forward_I0(self.wfs.Piston) # T[1,1,N,M]
        self.I0_DE = self.wfs.forward(phi = self.wfs.Piston, DE_layers = self.DE_layers) # T[1,1,N,M]
        
    #### MAIN FORWARD(NN and DEs) FUNCTION ### 
    def forward(self, phi, **kwargs):
        vNoise = kwargs.get('vNoise',self.wfs.vNoise)
        mInorm = kwargs.get('mInorm', self.mInorm)
        norm_nn = kwargs.get('norm_nn', self.norm_nn)

        ### Propagation ###
        I = self.wfs.forward(phi, fourier_mask=self.wfs.fourier_mask, DE_layers = self.DE_layers, vNoise=vNoise) # T[b, 1, N, M]
        ### Propagation ###

        I_deg = I.view(I.shape[0],1,-1) # T[b, 1, NM]

        if hasattr(self,'NN'):
            I = norm_I(self, I, norm_nn)# T[b,1,N,M]
            Zest = self.NN(I)# T[b,z]
        else:
            I0_v,PIMat = self.pwfs.Calibration(fourier_mask=fourier_mask, mInorm=mInorm)
            mI_v = self.pwfs.mI(I[:,0,self.pwfs.idx],I0_v, dim=1, mInorm=mInorm)
            Zest = ( PIMat @ mI_v.t() ).t()# T[z,b]-> T[b,z]    
        return Zest,I_deg
    
    # def norm_I(self,I,norm=None):
    #         if norm==None:
    #             return I
    #         elif norm=='max':
    #             return I/torch.amax(I,dim=(-2,-1),keepdim=True)
    #         elif norm=='sum':
    #             return (I/torch.sum(I,dim=(-2,-1),keepdim=True))*self.pupil_sum
    #         elif norm=='rms':
    #             return I/torch.sqrt( torch.sum(I**2, dim=(-2,-1),keepdim=True)/self.pupil_sum )
    #         elif norm=='zscore':
    #             return (I-torch.mean(I,dim=(-2,-1),keepdim=True))/torch.std(I,dim=(-2,-1),keepdim=True)
    #         elif norm=='zmI':
    #             return (I-self.I0)/torch.std((I-self.I0),dim=(-2,-1),keepdim=True)
    
    # def __repr__(self):
    #     print(f'FWFS class: device={self.device} | fp={self.fp} | norm_nn={self.norm_nn}')
    #     return ''