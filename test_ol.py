############################################# Libraries #############################################
import os
from pyexpat import model
import sys
import time
import copy
import torch
import pathlib
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from Functions import wfs
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
parser.add_argument('--device',         default = '7',        type = str,   help = 'Device to use: cpu or cuda: 0, 1, ..., 7')
parser.add_argument('--precision_name', default = 'single',   type = str,   help = 'Precision of the calculations: single, double, hsingle')
parser.add_argument('--routine',        default = 'TEST_P',   type = str,   help = 'Routine: D (Diffractive), NN (NN), ND (NN + Diffractive)')
parser.add_argument('--expName',        default = 'Test',     type = str,   help = 'Experiment name for saving results')
parser.add_argument('--zoom',           default = 1,          type = int,   help = 'Zoom factor for the output images')
parser.add_argument('--nTest',          default = 1000,       type = int,   help = 'Number of examples for Test')
parser.add_argument('--batch',          default = 10,         type = int,   help = 'Batch size')
parser.add_argument('--cost',           default = 'rmse',     type = str,   help = 'Cost function')
parser.add_argument('--pyr_wfs_test',   default = False,      type = bool,  help = 'Test for PYR_WFS')

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

save_path = pathlib.Path(params.expName).name
main_path = pathlib.Path('train') / params.precision_name / save_path

main_test_path = main_path / 'tests' / 'OL_Data'
main_test_path.mkdir(parents=True, exist_ok=True)

vNs = ''.join(map(str, routine_lists[0][0]['vNoise']))

exp_path = main_test_path / f'N_Data_{params.nTest}_Dr0_{params.Dr0[0]}--{params.Dr0[-1]}'
exp_path.mkdir(parents=True, exist_ok=True)

print( f'RMSE SWEEP: dvc={params.device} | T={params.nTest} | b={params.batch} | Dr0={params.Dr0} | M={params.modulation} | nDE={params.nDE} | nHead={params.nHead}')

atm_name = f'atm_Dro{params.Dr0[0]}-{params.Dr0[-1]}_Z{params.zModes[-1]}_T{params.nTest}_{params.precision_name}'

if not os.path.exists(exp_path / (atm_name + '.npz')):
    PHI,ZGT = ATM(params, r0v = params.r0, nData = params.nTest)#, seed = params.pid)
    np.savez(exp_path / (atm_name + '.npz'), Phi=PHI.cpu().numpy(), Zgt=ZGT.cpu().numpy())
else:
    print(f'Importing {atm_name} ATM')
    dataset = np.load(exp_path / (atm_name + '.npz'))
    PHI = torch.from_numpy(dataset['Phi'])
    ZGT = torch.from_numpy(dataset['Zgt'])
    
torch.cuda.empty_cache()

if params.pyr_wfs_test: ## puede ir mas arriba

    print('--- PYR WFS TEST ---')

    exp_path = pathlib.Path(f'test/OL_DATA/PYR_WFS_N_Data_{params.nTest}_Dr0_{params.Dr0[0]}--{params.Dr0[-1]}')
    exp_path.mkdir(parents=True, exist_ok=True)

    pyr_params         = copy.deepcopy(params)
    pyr_params.nDE     = 0
    pyr_params.posDE   = []
    pyr_params.vNoise  = routine_lists[0][0]['vNoise']
    pyr_params.mInorm  = routine_lists[0][0]['mInorm']
    pyr_params.init    = routine_lists[0][0]['init']
    pyr_params.crop    = routine_lists[0][0]['crop']
    pyr_params.norm_nn = routine_lists[0][0]['norm_nn']
    pyr_params.zModes  = torch.tensor(pyr_params.zModes, dtype=pyr_params.precision.real)

    pyr_wfs = FWFS(pyr_params, device=pyr_params.device).eval()

    pyr_params.mod = [0, 1, 2]
    pyr_params.nhs = [2, 3, 4]

    nhead_tag = tag("nHead", pyr_params.nhs)
    mod_tag   = tag("Mod",   pyr_params.mod)

    pyr_wfs_name =  f'pyr_wfs_{nhead_tag}_{mod_tag}_Dro{pyr_params.Dr0[0]}-{pyr_params.Dr0[-1]}_Zmodes{pyr_params.zModes[-1]}_T{pyr_params.nTest}_{pyr_params.precision_name}'
    pyr_wfs_path = exp_path / (pyr_wfs_name + '.npy')
    atm_std_path = exp_path /  'atm_std.npy'

    if not os.path.exists(pyr_wfs_path) or not os.path.exists(atm_std_path): ## adjuntar la carpeta de un del atm para hacer el 
        print(f'PWFS used: alpha={pyr_params.alpha:.4f} | nHead={pyr_params.nHead}')
        raw_pyr_wfs = torch.empty((len(pyr_params.mod), len(pyr_params.nhs), len(pyr_params.r0), pyr_params.nTest), dtype = pyr_params.precision.real).to('cpu')
        total_pyr_wfs = len(pyr_params.nhs) * len(pyr_params.mod) * len(pyr_params.r0) * (pyr_params.nTest // pyr_params.batch)
        raw_atm = torch.empty((len(pyr_params.r0), pyr_params.nTest), dtype = pyr_params.precision.real).to('cpu')

        with tqdm(total=total_pyr_wfs, desc='', bar_format=bar_format) as pbar_pyr_wfs:
            for i, m in enumerate(pyr_params.mod):
                pyr_wfs.wfs.modulation = m

                for j,nh in enumerate(pyr_params.nhs): 
                    pyr_wfs.wfs.nHead = nh
                    _, piston = pyr_wfs(pyr_wfs.wfs.Piston, vNoise=pyr_params.vNoise)
                    pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                    piston = piston.reshape(pos, pos).detach().cpu()
                    plt.imshow(piston.cpu().squeeze())
                    #plt.colorbar()
                    plt.axis('off')
                    plt.savefig(exp_path / f'I0_pyr_wfs_mod{m}_nhead{nh}.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    I0_v, PIMat = pyr_wfs.wfs.Calibration(mInorm=pyr_params.mInorm)

                    for r in range(len(params.r0)):
                        for t in range(params.nTest//params.batch):
                            pbar_pyr_wfs.set_description(f'Testing Pyr_WFS: Nheads={nh} | Mod={m} | r0={params.r0[r]}')
                            batch = np.arange(params.batch * t, params.batch * (t + 1))
                            iwfs = UNZ(PHI[r, batch, :, :], 1).to(params.precision.real).to(params.device)
                            zgt = ZGT[r, batch, :].to(params.precision.real).to(params.device)

                            # --- ESTIMATION --- #
                            I = pyr_wfs.wfs.forward(iwfs, vNoise=pyr_params.vNoise)
                            mI_v = pyr_wfs.wfs.mI(I[:, 0, pyr_wfs.wfs.idx], I0_v, mInorm = pyr_params.mInorm)
                            zest = (PIMat @ mI_v.t()).t()
                            # --- ESTIMATION --- #

                            raw_atm[r, batch] = torch.std(iwfs[..., torch.tensor(params.pupilLogical)].cpu(), dim=(-2,-1))# T[T,1,NM] -> T[r,T] fbasiufbaiufbaiusf
                            error = cost(zgt, zest).detach().cpu()
                            raw_pyr_wfs[i, j, r, batch] = error.squeeze()
                            pbar_pyr_wfs.update(1)

        np.save(pyr_wfs_path, raw_pyr_wfs)
        np.save(atm_std_path, raw_atm)

    else:
        print(f'Pyr_WFS_Mod{mod_tag}_nHead{nhead_tag} Open Loop exists at {pyr_wfs_path} and ATM_std {atm_std_path}')

else:
    print('--- DDWFS TEST ---')

    ddwfs_path   = exp_path / 'ol_data.npy'
    model_path   = main_path / 'routine_0' / 'train_0' / 'Checkpoint' / 'checkpoint_best-all.pth' ## modificar para fine tune
    w_nn_path    = main_path / 'routine_0' / 'train_0' / 'Checkpoint' / 'weights-nn.pt'
    w_de_path    = main_path / 'routine_0' / 'train_0' / 'Checkpoint' / 'de_layers.pth'

    if not os.path.exists(ddwfs_path):
        # raw_ddwfs = torch.empty((len(routine_lists), 1, len(params.r0), params.nTest), dtype = params.precision.real).to('cpu')
        raw_ddwfs = torch.empty((1, 1, len(params.r0), params.nTest), dtype = params.precision.real).to('cpu')

        # total_ddwfs= len(routine_lists) * len(params.r0) * (params.nTest // params.batch)
        total_ddwfs= 1 * len(params.r0) * (params.nTest // params.batch)

        with tqdm(total = total_ddwfs, desc = '', bar_format = bar_format) as pbar_ddwfs:
            
            # --- Nico Loading --- #
            # ddwfs = torch.load(model_path, map_location = params.device, weights_only = False).to(params.device).eval() ###################checkpoint#################
            # print(ddwfs)

            # --- Other Loading --- #
            ddwfs_params         = copy.deepcopy(params)
            ddwfs_params.nDE     = 0
            ddwfs_params.posDE   = []
            ddwfs_params.vNoise  = routine_lists[0][0]['vNoise']
            ddwfs_params.mInorm  = routine_lists[0][0]['mInorm']
            ddwfs_params.init    = routine_lists[0][0]['init']
            ddwfs_params.crop    = routine_lists[0][0]['crop']
            ddwfs_params.norm_nn = routine_lists[0][0]['norm_nn']
            ddwfs_params.zModes  = torch.tensor(ddwfs_params.zModes, dtype=ddwfs_params.precision.real)

            # ddwfs_params.test_ol = True

            ddwfs = FWFS(ddwfs_params, device = ddwfs_params.device).to(ddwfs_params.device)
            state_nn = torch.load(w_nn_path, map_location='cpu')
            ddwfs.NN.load_state_dict(state_nn, strict=True)

            state_de = torch.load(w_de_path, map_location='cpu')
            ddwfs.DE_layers.load_state_dict(state_de, strict=True)

            ddwfs.eval()
