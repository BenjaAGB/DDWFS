import os
import sys
import torch
import numpy as np
import torch.nn as nn
import scipy.io as scio
from math import pi as _pi
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class dC: pass

m = 1.0
cm=1e-2*m
mm=1e-3*m
um=1e-6*m
nm=1e-9*m

rad=1.0
mrad=1e-3*rad
urad=1e-6*rad

deg=(2*_pi*rad)/360

W = 1.0
mW = 1e-3*W

PI = _pi

def get_precision(type):
    precision = dC()
    if type=='hsingle':
        precision.int = torch.int16
        precision.real = torch.float16
        precision.complex = torch.complex64
    elif type=='single':
        precision.int = torch.int32
        precision.real = torch.float32
        precision.complex = torch.complex64
    elif type=='double':
        precision.int = torch.int64
        precision.real = torch.float64
        precision.complex = torch.complex128
    else:
        sys.exit('precision not recognized')
    return precision

def getTrainParam(input, rows):
    matrix = np.zeros((rows, len(input)), dtype=bool)
    for col_idx, (offset, value) in enumerate(input):
        if value != 0:
            positions = (np.arange(rows-offset)) % value == 0
            matrix[offset:rows, col_idx] = positions[0:rows-offset]
    return matrix

def Log(pars, routine, path, name = 'Log'):
    log_path = path + f'/{name}.txt'
    # parameters
    with open(log_path, 'w') as file_object:
        file_object.write(f"================== GENERAL PARAMETERS ==================\n\n")
        max_key_length = max(len(key) for key in pars.keys())
        for key,value in pars.items():
            if isinstance(value, list):
                value_str = ", ".join(map(str, value)) 
            else:
                value_str = str(value)
            file_object.write("{:<{width}} = {}\n".format(key,value_str, width=max_key_length))
        file_object.write('\n\n')
        # routine
        if len(routine)==1:
            file_object.write(f"================== SINGLE ROUTINE ==================\n\n")
        else:
            file_object.write(f"================== MULTIPLE ROUTINE ==================\n\n")
        for i,data in enumerate(routine):
            file_object.write(f"~~~~~~~~~~~~~~~~~ ROUTINE {i} ~~~~~~~~~~~~~~~~~\n")
            for j,d in enumerate(data):
                file_object.write(f"----------- Train/Test {j} -----------\n\n")
                max_key_length = max(len(key) for key in d.keys())
                for key,value in d.items():
                    if isinstance(value, list):
                        value_str = ", ".join(map(str, value)) 
                    else:
                        value_str = str(value)
                    file_object.write("{:<{width}} = {}\n".format(key,value_str, width=max_key_length))
                file_object.write('\n')
    print('LOG DONE!')

class importDataset(Dataset):
    def __init__(self, atm_path):
        super(importDataset, self).__init__()
        Zgt_list = []
        Phi_list = []
        if os.path.exists(atm_path):
            files = os.listdir(atm_path)
            for file_name in files:
                if '.mat' in file_name:
                    path = atm_path + '/' + file_name
                    data = scio.loadmat( path )
                    phi = torch.from_numpy(data["Phi"])# T[b,1,N,M]
                    zgt = torch.from_numpy(data["Zgt"])# T[b,z]
                    Phi_list.append(phi)
                    Zgt_list.append(zgt)
            self.Phi = torch.cat(Phi_list, dim=0)# along batch
            self.Zgt = torch.cat(Zgt_list, dim=0)# along batch
            self.L = self.Phi.shape[0]
            print(f'Dataset imported correctly: Phi={self.Phi.shape} | Zgt={self.Zgt.shape} | L={self.L} | dtype={self.Phi.dtype}{self.Zgt.dtype} | Path={atm_path}')
        else:
            raise FileNotFoundError('file doesnt exist!')
    def __getitem__(self, index):
        Phi = self.Phi[index,0:1,:,:]# T[b,1,nPx,nPx]
        Zgt = self.Zgt[index,:]# T[b,zModes]
        return Phi,Zgt
    def __len__(self):
        return self.L
    
class COST_modal(nn.Module):
    def __init__(self, **kwargs):
        super(COST_modal, self).__init__()
        valid_modes = ['rmse', 'mse', 'mae']
        self.mode = kwargs.get('mode') 
        assert self.mode in valid_modes, 'Incorrect mode of cost function'
        self.dim = kwargs.get('dim',1)
        print(f'COST function initialized: mode={self.mode} | dim={self.dim}')
    # forward function
    def forward(self, zgt,zest):
        if self.mode == 'rmse':
            output = torch.sqrt( torch.mean( (zgt-zest)**2, dim=self.dim ) )
        elif self.mode=='mse':
            output = ( torch.mean( (zgt-zest)**2, dim=self.dim ) )
        elif self.mode=='mae':
            output = ( torch.mean( torch.abs(zgt-zest), dim=self.dim ) )
        return output# T[b]
class COST_spatial(nn.Module):
    def __init__(self, **kwargs):
        super(COST_spatial, self).__init__()
        valid_modes = ['std', 'mad', 'var']
        self.mode = kwargs.get('mode') 
        assert self.mode in valid_modes, 'Incorrect mode of cost function'
        self.dim = kwargs.get('dim',1)
        print(f'COST function initialized: mode={self.mode} | dim={self.dim}')
    # forward function
    def forward(self, x):
        x_mean = torch.mean(x, dim=(1,2),keepdim=True)# T[b,1,1]
        if self.mode == 'std':
            output = torch.sqrt( torch.mean( (x-x_mean)**2, dim=(1,2) ) )
        elif self.mode == 'var':
            output = torch.mean( (x-x_mean)**2, dim=(1,2) ) 
        return output# T[b]

def trainFigure(loss_t,loss_v,path="./"):
    loss_t = np.abs(np.array(loss_t.cpu()))
    loss_v = np.abs(np.array(loss_v.cpu()))
    epochs = loss_t.shape[0]
    x = np.arange(1,epochs+1)
    # Create a figure and an axis
    fig, ax = plt.subplots()
    # Create the errorbar plots
    ax.plot(x, (loss_t), label="Loss_t")
    ax.plot(x, (loss_v), label="Loss_v")
    # Enable grid
    ax.grid(True)
    # Set the x and y limits
    ax.set_xlim(1, epochs)  # Adjust the x-axis, for example from 1 to epochs
    ax.set_ylim(0, max( max(loss_v),max(loss_t)) + 1)
    # Add title and labels
    ax.set_title("Training v/s Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    # Add legend
    ax.legend()
    # Save the figure to a file
    fig.savefig(path + "/lossErrorPlot.jpg", dpi=900, bbox_inches='tight')
    # Close the figure
    plt.close(fig)

def select_routine(params):
    routine = params.routine

    ## cambiar modelos #####


    ab     = [1,1]                 # Dataset distribution
    mInorm = 1                     # mI normalization
    init   = ['kaiming','kaiming'] # kaiming to not diverge the pinv

    
    # if routine == 'P':
    #     nData = np.array([80,20])*int(1)
    #     fp = [(0,0),(0,1)]
    #     cost = 'std'
    #     norm_nn = 'zscore'
    #     routine_lists = [
    #         [{'mInorm':mInorm, 'alpha':3, 'nHead':nH, 'init':init, 
    #           'nnModel':'GcVit', 'epoch':100, 'lr':[2e-3,2e-4], 'dlr':[.8,.8],
    #           'batch':10, 'nData':nData, 'ab':ab, 'Dr0':[10,150], 'fp':fp, 
    #           'cost':cost, 'cl':[2,1], 'vNoise':[0,0,0,1], 'zModes':[2,210], 'crop':crop,
    #           'norm_nn':norm_nn, 'fine tunning':''},
    #         ],
    #     ]
        
    if routine == 'TEST':
        nData = np.array([80,20])*int(1)
        fp = [(0,2),(0,1)] ##
        cost = 'std'
        norm_nn = 'zscore'
        routine_lists = [
            [{'mInorm':mInorm, 'init':init, 
              'nnModel':'GcVit', 'epoch':40, 'lr':[2e-3,2e-4], 'dlr':[.8,.8],
              'batch':10, 'nData':nData, 'ab':ab, 'Dr0':[10,150], 'fp':fp, 
              'cost':cost, 'cl':[1,0], 'vNoise':[0,0,0,1], 'zModes':[2,210], 'crop':params.crop,
              'norm_nn':norm_nn, 'fine tunning':''},
            ],
        ]

    elif routine == 'TEST_1':
        nData = np.array([8000,2000])*int(1)
        nZ = [2,210]
        # nH = 4
        # al = 3
        init = ['CONSTANT',0]
        b = 10
        zN = 0
        vN = [zN,0,0,1.]
        routine_lists = [
            [{'mInorm':mInorm, 'init':init, 
              'nnModel':'GcVit','epoch':100, 'lr':[0.002,0.0002], 'dlr':[.8,.8],
                'batch':b, 'nData':nData, 'ab':ab, 'Dr0':[10,150], 'fp':[(0,2),(0,1)], 
                'cost':'std', 'cl':[2,1], 'vNoise':vN, 'zModes':nZ, 'crop':False,
                'norm_nn':'zscore', 'fine tunning':'', 'lD':5,'h':np.pi/2},
            ],
        ]

    elif routine == 'TEST_PAPER_NICO':

        nData = np.array([8000,2000])*int(5)
        nZ = [2,200]
        nH = 0
        al = 3
        init = ['constant', 'constant']
        b = 10
        lr = np.array([2e-3,2e-4])*1e-1
        routine_lists = [
            # DNN
            [{'mInorm':mInorm, 'alpha':al, 'nHead':nH, 'init':init, 
              'nnModel':'GcVit', 'epoch':20, 'lr':lr, 'dlr':[.8,.8],
                'batch':b, 'nData':nData, 'ab':ab, 'Dr0':[2.5,100], 'fp':[(0,2),(0,1)], 
                'cost':'std', 'cl':[2,1], 'vNoise':[.0,0.,0.,1.], 'zModes':nZ, 'crop':False,
                'norm_nn':'zscore', 'fine tunning':'./MODELS/nD50k/b10/nZ200/n5init_zN0_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth',
                'lD':5,'h':np.pi/2},
            ]
        ]


    # if routine == 'PNN':
    #     nData = np.array([80,20])*int(1)
    #     fp = [(0,2),(0,1)]
    #     cost = 'std'
    #     norm_nn = 'zscore'
    #     nH = -2
    #     init = ['constant',0]
    #     routine_lists = [
    #         [{'mInorm':mInorm, 'alpha':0, 'nHead':nH, 'init':init, 
    #           'nnModel':'GcVit', 'epoch':100, 'lr':[2e-3,2e-4], 'dlr':[.8,.8],
    #             'batch':10, 'nData':nData, 'ab':ab, 'Dr0':[10,100], 'fp':fp, 
    #             'cost':cost, 'cl':[2,1], 'vNoise':[0,0,0,1], 'zModes':[2,24], 'crop':crop,
    #             'norm_nn':norm_nn, 'fine tunning':'',  'lD':2,'h':np.pi/2},
    #         ],
    #     ]
    # if routine == 'CAMILO':
    #     nData = np.array([80,20])*int(10)
    #     nZ = [2,210]
    #     norm_nn = 'max'
    #     b = 100
    #     crop = True
    #     nH = 5
    #     routine_lists = [
    #         [{'mInorm':mInorm, 'alpha':3, 'nHead':nH, 'init':init, 
    #           'nnModel':'GcVit', 'epoch':40, 'lr':[2e-3,1e-4], 'dlr':[.8,.8],
    #             'batch':b, 'nData':nData, 'ab':ab, 'Dr0':[25,150], 'fp':[(0,0),(0,1)], 
    #             'cost':'mae', 'cl':[0,0], 'vNoise':[0,1,1e1,1], 'zModes':nZ, 'crop':crop,
    #             'norm_nn':norm_nn, 'fine tunning':'', 'h':np.pi/2,'lD':5},
    #         {'mInorm':mInorm, 'alpha':3, 'nHead':nH, 'init':init, 
    #                     'nnModel':'GcVit', 'epoch':40, 'lr':[2e-3,1e-5], 'dlr':[.8,.8],
    #                         'batch':b, 'nData':nData, 'ab':ab, 'Dr0':[7.5,150], 'fp':[(0,0),(0,1)], 
    #                         'cost':'mae', 'cl':[0,0], 'vNoise':[0,1,1e1,1], 'zModes':nZ, 'crop':crop,
    #                         'norm_nn':norm_nn, 'fine tunning':'', 'h':np.pi/2,'lD':5},
    #         {'mInorm':mInorm, 'alpha':3, 'nHead':nH, 'init':init, 
    #                     'nnModel':'GcVit', 'epoch':20, 'lr':[2e-3,1e-6], 'dlr':[.8,.8],
    #                         'batch':b, 'nData':nData, 'ab':ab, 'Dr0':[7.5,150], 'fp':[(0,0),(0,1)], 
    #                         'cost':'mae', 'cl':[1,1], 'vNoise':[0,1,1e1,1], 'zModes':nZ, 'crop':crop,
    #                         'norm_nn':norm_nn, 'fine tunning':'', 'h':np.pi/2,'lD':5},
    #         ],
    #     ]

    return routine_lists