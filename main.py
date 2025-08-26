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
parser.add_argument('--nDE',            default = 2,        type = float, help = 'Number of diffractive elements')
parser.add_argument('--device',         default = '3',      type = str,   help = 'Device to use: cpu or cuda: 0, 1, ..., 7')
parser.add_argument('--precision_name', default = 'double', type = str,   help = 'Precision of the calculations: single, double, hsingle')
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
            atm_path = f"./dataset/D{int(prs.D)}_ResData{int(prs.nPx)}_ResNN{int(prs.resol_nn)}_Dro{p_local['Dr0'][0]}-{p_local['Dr0'][1]}_Z{p_local['zModes'][-1]}_T{sum(p_local['nData'])}_αβ{ab}_{prs.precision_name}"

            if not os.path.exists(atm_path) or not os.listdir(atm_path):
                if not os.path.exists(atm_path):
                    os.makedirs(atm_path, exist_ok=True)
                getATM(prs, p_local,atm_path)

            dataset = importDataset(atm_path)
            train_dataset, test_dataset = random_split(dataset, p_local['nData'])
            train_data = DataLoader(train_dataset, batch_size = p_local['batch'], shuffle = True)
            val_data = DataLoader(test_dataset, batch_size = p_local['batch'], shuffle = False)

            ### Cost Function ###
            if p_local['cost'] in ['mse','rmse','mae']:
                cost = COST_modal(mode=p_local['cost'], dim=1).to(prs.device)
            elif p_local['cost'] in ['std','mad','var']:
                cost = COST_spatial(mode=p_local['cost'], dim=(-2,-1)).to(prs.device)

            ### Fine Tunning ###
            if p_local['fine tunning']: #### revisar si funciona bien ###
                model = torch.load(p_local['fine tunning'], map_location = prs.device).to(prs.device)
                model.pwfs.vNoise = prs.vNoise #### pwfs cambiar
                model.device = prs.device 
                print(model.device)
                lr,dlr = p_local['lr'],p_local['dlr']
                best_loss_v = float('Inf')
                epoch_check = 0
                print('Importing pretrained model')
            else:
                print('Training cero-model')
                if j == 0: # first training
                    epoch_check = 0
                    best_loss_v = float('inf')
                    lr,dlr = p_local['lr'],p_local['dlr']

                    model = FWFS(prs, device=prs.device)

                    print(f'FIRST EXECUTION')
                else: # fine-tuning
                    finetuning_path = path_routine + f'/train_{j-1}/Checkpoint/checkpoint_best-v.pth'

                    model = torch.load(finetuning_path, map_location = prs.device)

                    lr = p_local['lr']
                    dlr = p_local['dlr']
                    best_loss_v = float('inf')
                    epoch_check = 0

                    print(f'FINE TUNNING {j}')
                
            # piston propagation
            with torch.no_grad():
                _, piston = model.eval()(model.wfs.Piston)# T[b,1,NM]
                pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                piston = piston.reshape(pos, pos).detach().cpu()
                plt.imshow(piston)
                plt.axis('off')
                plt.savefig(path_train+f'/I0_e0.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            input = sample_phi(prs, strength = 3.0)
            Irand = model.wfs(input.to(prs.device), DE_layers = model.DE_layers).detach().cpu().squeeze()

            figs,axs = plt.subplots(1, 2, figsize = (5*2, 5))
            im0=axs[0].imshow(input.squeeze().cpu())
            im1=axs[1].imshow(Irand)
            axs[0].set_axis_off()
            axs[1].set_axis_off()
            plt.colorbar(im0,ax=axs[0])
            plt.colorbar(im1,ax=axs[1])
            plt.savefig(path_train+f'/Irand.png', dpi=300, bbox_inches='tight')
            plt.close()

            ### Train ###
            loss_t, loss_v = torch.zeros(p_local['epoch']), torch.zeros(p_local['epoch'])
            t0_total       = time.time()
            tiempo         = torch.zeros((p_local['epoch']))

            for e in range(epoch_check,epoch_check+p['epoch']):
                t0 = time.time()

                par = {'epoch':e, 'batch':p_local['batch'], 'lr':lr, 'dlr':dlr, 'cost':(p_local['cost'], cost), 'fp':freeze_mask[e], 'cl':p_local['cl'], 'vN':p_local['vNoise']}
                loss_t[e] = train(par, model.train(),train_data)   # loss
                loss_v[e],phis = validation(par, model.eval(),val_data)# error 

                ## Checkpoint ##
                evol_par = {'epoch':e, 'lr':lr, 'dlr':dlr, 'loss_v':loss_v[e], 'loss_t':loss_t[e]}
                if loss_v[e] < best_loss_v:
                    best_loss_v = loss_v[e]
                    torch.save(model, f'{checkpoint_path}/checkpoint_best-all.pth')
                    torch.save(model.NN.state_dict(), f"{checkpoint_path}/weights-nn.pt")
                    torch.save(model.DE_layers.state_dict(), f"{checkpoint_path}/weights-de.pt")

                    # print(f'Best model saved at epoch {e} with loss {best_loss_v * 1e6:.5f}')

                    if prs.nDE > 0:
                        for i, de_layer in enumerate(model.DE_layers, start=1):
                            de = de_layer.detach().cpu().numpy()
                            
                            de = ((de + math.pi) % (2*math.pi)) - math.pi  ## solo visualizacion ##

                            plt.imshow(de, vmin = -math.pi, vmax = math.pi, cmap = 'hsv')
                            plt.axis('off')
                            plt.savefig(path_train + f'/DE_{i:02d}.png', dpi = 300, bbox_inches = 'tight')
                            plt.close()
                    else:
                        de = model.DE_dummy.detach().cpu().numpy()
                        de = ((de + math.pi) % (2*math.pi)) - math.pi  ## solo visualizacion ##

                        plt.imshow(de, vmin = -math.pi, vmax = math.pi, cmap = 'hsv')
                        plt.axis('off')
                        plt.savefig(path_train + f'/DE_actual.png', dpi = 300, bbox_inches = 'tight')
                        plt.close()
                    
                    if prs.evol_save:
                        
                        with torch.no_grad():
                            _, piston = model.eval()(model.wfs.Piston)
                            pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                            piston = piston.reshape(pos, pos).detach().cpu()
                            plt.imshow(piston)
                            plt.axis('off')
                            plt.savefig(path_train+f'/I0.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            norm = Normalize(vmin = phis['phi_corr'].min(), vmax = phis['phi_corr'].max())
                            fig, ax = plt.subplots(1, 2, figsize = (10, 5))
                            ax[0].imshow(phis['phi_corr'], norm = norm)
                            ax[1].set_title(f'std(phi_res) = {np.std(phis["phi_res"][prs.pupil]):.2f}')
                            ax[1].imshow(phis['phi_res'], norm = norm)
                            plt.savefig(path_train + f'/phis.png', dpi = 300, bbox_inches = 'tight')
                            plt.close()

                    if (e+1)%5 == 0:
                        lr = dlr*lr
                        print('lr changed to: ' + ' '.join([format(value, '.3f') for value in lr]))

                    t1 = time.time()
                    tiempo[e] = t1 - t0

                    trainFigure(loss_t.detach(), loss_v.detach(), path_train)
                    print(f'epoch finished = {e} | loss_t = {loss_t[e]:.3f} [nm] | loss_v = {loss_v[e]:.3f} [nm] / {best_loss_v:.3f} | time = {t1-t0:.3f}')

            t1_total = time.time()
            print(f'Total time: {t1_total - t0_total:.3f} | Epoch time: {tiempo[e]:.3f} | Best loss: {best_loss_v:.3f}')







