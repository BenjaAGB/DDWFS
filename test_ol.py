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
from torch.nn.parallel import DistributedDataParallel as DDP
############################################# Libraries #############################################

bar_format = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]" #### revisar ###
command = ' '.join(sys.argv)

############################################# Argument Parser #############################################
parser = argparse.ArgumentParser(description='Settings, Training and Fresnel Wavefront Sensor parameters')

#### Main Parameters ####
parser.add_argument('--D',              default = 3.0,        type = float, help = 'Diameter of the aperture in [m]')
parser.add_argument('--nPx',            default = 128,        type = int,   help = 'Number of pixels')
parser.add_argument('--resol_nn',       default = 128,        type = int,   help = 'Resolution of the neural network')
parser.add_argument('--wvl',            default = 635,        type = float, help = 'Wavelength in [nm]')
parser.add_argument('--ps_slm',         default = 3.74,       type = float, help = 'Pixel size of the SLM in [um]')
parser.add_argument('--modulation',     default = 0,          type = float, help = 'Modulation in [λ/D]')
parser.add_argument('--alpha',          default = 3,          type = float, help = 'Angle of the PWFS in [degrees]')
parser.add_argument('--nHead',          default = 4,          type = int,   help = 'Number of faces of the pyramid')
parser.add_argument('--f1',             default = 100,        type = float, help = 'Focal length of the first lens in [mm]')
parser.add_argument('--f2',             default = 100,        type = float, help = 'Focal length of the second lens in [mm]')
parser.add_argument('--nDE',            default = 3,          type = int,   help = 'Number of diffractive elements')
parser.add_argument('--dz',             default = 0,          type = float, help = 'Step size for the propagation in [mm]')
parser.add_argument('--dz_before',      default = 0,          type = float, help = 'Step size before the PSF in [mm]')
parser.add_argument('--dz_after',       default = 0,          type = float, help = 'Step size after the PSF in [mm]')
parser.add_argument('--posDE',          default = [-1, 0, 1],         type = int,   help = 'Position of the diffractive propagator', nargs = '+')
parser.add_argument('--device',         default = '4',        type = str,   help = 'Device to use: cpu or cuda: 0, 1, ..., 7')
parser.add_argument('--precision_name', default = 'single',   type = str,   help = 'Precision of the calculations: single, double, hsingle')
parser.add_argument('--routine',        default = 'TEST_P',   type = str,   help = 'Routine: D (Diffractive), NN (NN), ND (NN + Diffractive)')
parser.add_argument('--expName',        default = 'Test',     type = str,   help = 'Experiment name for saving results')
parser.add_argument('--zoom',           default = 1,          type = int,   help = 'Zoom factor for the output images')
parser.add_argument('--nTest',          default = 1000,       type = int,   help = 'Number of examples for Test')
parser.add_argument('--batch',          default = 100,        type = int,   help = 'Batch size')
parser.add_argument('--cost',           default = 'rmse',     type = str,   help = 'Cost function')

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

if params.dz > 0:
    params.dz          = params.dz * mm
else:
    params.dz          = None

if params.dz_before > 0:
    params.dz_before   = params.dz_before * mm
else:
    params.dz_before   = None

if params.dz_after > 0:
    params.dz_after    = params.dz_after * mm
else:
    params.dz_after    = None

# params.pupil        = CreateTelescopePupil_physical(params.nPx, params.R, params.size) ################
params.samp     = 2
params.fovPx    = 2*params.samp*params.nPx
params.pupil    = CreateTelescopePupil(params.nPx)
params.size     = (4 + params.nPx/64) * mm
params.R        = (params.size/2) * (params.nPx-1)/(params.fovPx-1)  # [m] Radius of the aperture

params.pupilLogical = params.pupil!=0
params.crop         = False
params.pid          = os.getpid()

routine_lists = select_routine(params)

current_date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if not params.posDE and params.nDE >= 0:
    if params.nDE == 0:
        params.dz = 0
        params.dz_before = 0
        params.dz_after = 0
    else:
        params.dz = params.f2 / params.nDE
        params.dz_before = 0
        params.dz_after = params.dz

    params.de_info = compute_de_positions_for_log(params)
    DE_z = np.array(params.de_info['DE_z_from_aperture']) * 1000 #--- DEs distance mm---#
    de_z_formatted = [f"{z*1000:.2f}" for z in params.de_info['DE_z_from_aperture']]
    de_z_str = '-'.join(de_z_formatted)
    if params.nDE == 0:
        de_z_str = 'None'

elif params.posDE and params.nDE > 0:
    PosDE_Error(params)
    params.de_info = compute_de_positions_for_log(params)
    DE_z = np.array(params.de_info['DE_z_from_aperture']) * 1000 #--- DEs distance mm---#
    de_z_formatted = [f"{z*1000:.2f}" for z in params.de_info['DE_z_from_aperture']]
    de_z_str = '-'.join(de_z_formatted)

if params.nHead > 0:
    params.expName   = f'{params.expName}_ResolNN_{params.resol_nn}_ResolData_{params.nPx}_Geometry_[NHead{params.nHead}-Alpha{params.alpha}]_NDiffractive_{params.nDE}_PosD_[{de_z_str}][mm]_f1_[{params.f1 * 1000 }][mm]_f2_[{params.f2 * 1000 }][mm]_DistSys[{(params.f1 * 2 + params.f2 * 2) *1000 }][mm]_Routine_{params.routine}'
elif params.nHead == 0:
    params.expName   = f'{params.expName}_ResolNN_{params.resol_nn}_ResolData_{params.nPx}_NO-Geometry_NDiffractive_{params.nDE}_PosD_[{de_z_str}][mm]_f1_[{params.f1 * 1000 }][mm]_f2_[{params.f2 * 1000 }][mm]_DistSys[{(params.f1 * 2 + params.f2 * 2) *1000 }][mm]_Routine_{params.routine}'

params.zModes = routine_lists[0][0]['zModes'] ### cambiar por el numero de carpetas que se entrenaron ###

if params.zoom == 1:
    params.Dr0 = [60,55,50,45,40,35,30,25,20,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    print('ATM with D/r0:', params.Dr0)
elif params.zoom == 2:
    params.Dr0 =  [20,15,10,8,6,4,2,1,.1,.01]
    print('ATM with D/r0:', params.Dr0)
elif params.zoom == 3:
    params.Dr0 =  [10,9,8,7,6,5,4,3,2,1,.1]
    print('ATM with D/r0:', params.Dr0)

params.Dr0 = np.array(params.Dr0)
params.r0  = np.round(params.D / params.Dr0, 5)

params.k = torch.tensor((2*torch.pi)/params.wvl, dtype = params.precision.real, device = params.device)
params.mask_pupil = torch.tensor(params.pupil, dtype = params.precision.real, device = params.device)

params.jModes  = torch.arange(params.zModes[0], params.zModes[1]+1)
params.modes = torch.tensor(CreateZernikePolynomials1(params)).to(params.device).to(params.precision.real)

if params.nTest % params.batch != 0:
    raise KeyError('Please select a batch that divide the T in integer parts')

if params.cost in ['mse','rmse','mae']:
    cost = COST_modal(mode=params.cost, dim=1).to(params.device)
elif params.cost in ['std','mad','var']:
    cost = COST_spatial(mode=params.cost, dim=(-2,-1)).to(params.device)

main_path = params.expName + f'/tests/OL_Data/'
os.makedirs(main_path, exist_ok=True)
