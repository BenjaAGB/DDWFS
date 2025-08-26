############################################# Libraries #############################################
import os
import sys
import time
import copy
import torch
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from Functions.utils import *
from Functions.ddwfs import *
from Functions.functions import *
from Functions.propagator import *
from Functions.atmosphere import *
from Functions.fourier_mask import *
from Functions.functions_nn import *
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, random_split
############################################# Libraries #############################################

command = ' '.join(sys.argv)

############################################# Argument Parser #############################################
parser = argparse.ArgumentParser(description='Settings, Training and Fresnel Wavefront Sensor parameters')

#### Main Parameters ####
parser.add_argument('--D',              default = 3.0,      type = float, help = 'Diameter of the aperture in [m]')
parser.add_argument('--nPx',            default = 128,      type = int,   help = 'Number of pixels')
parser.add_argument('--resol_nn',       default = 128,      type = int,   help = 'Resolution of the neural network')
parser.add_argument('--wvl',            default = 635,      type = float, help = 'Wavelength in [nm]')
parser.add_argument('--ps_slm',         default = 3.74,     type = float, help = 'Pixel size of the SLM in [um]')
parser.add_argument('--modulation',     default = 0,        type = float, help = 'Modulation in [λ/D]')
parser.add_argument('--alpha',          default = 3,        type = float, help = 'Angle of the PWFS in [degrees]')
parser.add_argument('--nHead',          default = 4,        type = int,   help = 'Number of faces of the pyramid')
parser.add_argument('--f1',             default = 100,      type = float, help = 'Focal length of the first lens in [mm]')
parser.add_argument('--f2',             default = 100,      type = float, help = 'Focal length of the second lens in [mm]')
parser.add_argument('--nDE',            default = 1,        type = float, help = 'Number of diffractive elements')
parser.add_argument('--device',         default = '0',      type = str,   help = 'Device to use: cpu or cuda: 0, 1, ..., 7')
parser.add_argument('--precision_name', default = 'single', type = str,   help = 'Precision of the calculations: single, double, hsingle')
parser.add_argument('--routine',        default = 'TEST_1', type = str,   help = 'Routine: D (Diffractive), NN (NN), ND (NN + Diffractive)')
parser.add_argument('--expName',        default = "Testa",   type = str,   help = 'Experiment name for saving results')
parser.add_argument('--evol_save',      default = 1,        type = int,   help = 'Save diffractive evolution on a gif')

params = parser.parse_args()
############################################# Argument Parser #############################################

#### More parameters ####
if params.resol_nn <= params.nPx:
    params.resol_nn = params.nPx

params.wvl          = params.wvl * nm                                                               # [m] Wavelength
params.ps_slm       = params.ps_slm * um                                                            # [m] Pixel size of the SLM
params.f1           = params.f1 * mm                                                                # [m] Focal length of the first lens
params.f2           = params.f2 * mm                                                                # [m] Focal length of second lens
params.rooftop      = 0
params.amp_cal      = 0.1                                                                           ####################################################### # Amplitude calibration
params.device       = torch.device(f"cuda:{params.device}" if torch.cuda.is_available() else "cpu") # Device to use
params.precision    = get_precision(type=params.precision_name)                                          # Define the dtype for the experiment

# params.pupil        = CreateTelescopePupil_physical(params.nPx, params.R, params.size) ################
params.samp     = 2
params.fovPx    = 2*params.samp*params.nPx
params.pupil    = CreateTelescopePupil(params.nPx)
params.size     = (4 + params.nPx/64) * mm
params.R        = (params.size/2) * (params.nPx-1)/(params.fovPx-1)  # [m] Radius of the aperture

params.pupilLogical = params.pupil!=0
params.crop         = False
params.pid          = os.getpid()


#### Select routine ####
routine_lists = select_routine(params)

current_date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
params.expName   = f'{params.expName}_resol{params.fovPx}_nPxData{params.nPx}_{params.routine}_{params.precision_name}'

### Create Log ###
Log_Path         = f'./train/{params.precision_name}/{params.expName}'
os.makedirs(Log_Path, exist_ok=True)

pars_log = {'D':params.D, 'resol_Data':params.nPx, 'resol_NN':params.resol_nn, 'rooftop':params.rooftop, 'ampCal':params.amp_cal,
            'wvl':params.wvl, 'ps_slm':params.ps_slm, 'R':params.R, 'size':params.size, 'nDE':params.nDE,
            'f1':params.f1, 'f2':params.f2, 'alpha':params.alpha, 'nHead':params.nHead,
            'routine':params.routine, 'expName':params.expName, 'pid':params.pid, 'device':params.device, 
            'precision_name':params.precision_name, 'evol_save':params.evol_save, 'command':command, 'date':current_date_str}

Log(pars_log, routine_lists, path = Log_Path, name = f'Log')

print(f'TRAINING: device={params.device} | precision={params.precision_name} | expName={params.expName}')
# time.sleep(5)

for i,ro in enumerate(routine_lists, start=0): # Rutines [{}] for each routine
    path_routine = Log_Path + f'/routine_{i}'
    os.makedirs(path_routine, exist_ok=True)

    for j,p in enumerate(ro, start=0): # Fine Tunning {}

            prs = copy.deepcopy(params)  # copia independiente de los parámetros
    
            path_train = os.path.join(path_routine, f"train_{j}")
            os.makedirs(path_train, exist_ok=True)
            checkpoint_path = path_train + f'/Checkpoint'
            os.makedirs(checkpoint_path, exist_ok=True)  
            if prs.evol_save:
                evol_path = f"{path_train}/evol"
                os.makedirs(evol_path, exist_ok=True)


            p_local = {k: (np.array(v) if isinstance(v, list) and k != 'vNoise' else v)
                    for k, v in p.items()}

            p_local['r0'] = np.round(prs.D/p_local['Dr0'][::-1], 2).astype(np.float64)

            prs.vNoise  = p_local['vNoise']
            prs.zModes  = p_local['zModes']
            prs.mInorm  = p_local['mInorm']
            prs.init    = p_local['init']
            prs.device  = prs.device
            prs.crop    = p_local['crop']
            prs.norm_nn = p_local['norm_nn']
            prs.zModes  = torch.tensor(prs.zModes, dtype=prs.precision.real)
            prs.jModes  = torch.arange(prs.zModes[0], prs.zModes[1]+1)

            # prs.h   = p_local['h']   # Zernike WFS
            # prs.lD  = p_local['lD']  # Zernike WFS
            # prs.nRr = p_local['nRr'] # Version estandar GCVIT

            prs.modes = torch.tensor(CreateZernikePolynomials1(prs)).to(prs.device).to(prs.precision.real)

            ab = ''.join(map(str, p_local['ab']))
            fp_str = "".join([str(item) for tup in p_local['fp'] for item in tup])
            freeze_mask = getTrainParam(p_local['fp'], p_local['epoch'])
            prs.fp = p_local['fp']

            ### DATASET ###
            atm_path = f"./dataset/D{int(prs.D)}_ResData{int(prs.nPx)}_Dro{p_local['Dr0'][0]}-{p_local['Dr0'][1]}_Z{p_local['zModes'][-1]}_T{sum(p_local['nData'])}_αβ{ab}_{prs.precision_name}"

            if not os.path.exists(atm_path) or not os.listdir(atm_path):
                if not os.path.exists(atm_path):
                    os.makedirs(atm_path, exist_ok=True)
                getATM(prs, p_local,atm_path)
