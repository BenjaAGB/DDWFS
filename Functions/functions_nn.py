import torch
import importlib
import numpy as np
from tqdm import tqdm
import torch.optim as optim

bar_format = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"

def select_nn(params):
    method = importlib.import_module('models.GcVit')

    if params.wfs.resol_nn == 128:
        print('hola')

    model = method.GCViT(num_classes = len(params.wfs.jModes), depths = [2,2,6,2], num_heads = [2,4,8,16], window_size = [16, 16, 32, 16],
                                    resolution = params.wfs.resol_nn, in_chans = 1, dim = 64, mlp_ratio = 3, drop_path_rate = 0.2).to(params.precision.real).to(params.device)
    return model

def train(params, model, train_data):
    epoch_loss = 0

    if model.nDE:
        # model.DE_layers.requires_grad = bool(p['fp'][0])
        # optimizer_de = optim.AdamW([{'params': model.DE_layers.parameters()}], lr=p['lr'][0])

        model.DE_layers.requires_grad_(bool(params['fp'][0]))

        if params['fp'][0]:
            optimizer_de = optim.AdamW(model.DE_layers.parameters(), lr = params['lr'][0])
        else:
            optimizer_de = None

    if hasattr(model, 'NN'):
        # model.NN.requires_grad = bool(p['fp'][1])
        # optimizer_nn = optim.AdamW([{'params': model.NN.parameters()}], lr=p['lr'][1])

        model.NN.requires_grad_(bool(params['fp'][1]))

        if params['fp'][1]:
            optimizer_nn = optim.AdamW(model.NN.parameters(), lr = params['lr'][1])
        else:
            optimizer_nn = None
    
    modes = model.wfs.modes

    for _,data in tqdm(enumerate(train_data), total = len(train_data), desc = 'Trainning data', bar_format = bar_format):
        phi = data[0].to(model.device).to(model.precision.real)       # T[b,1,N,M]
        zgt = data[1].to(model.device).to(model.precision.real)[:, :] # T[b,z]

        if model.nDE and optimizer_de is not None:
            optimizer_de.zero_grad()

        if hasattr(model, 'NN'):
            optimizer_nn.zero_grad()

        if (not params['fp'][0]): # if DE freezed or not in the model
            vNoise = model.wfs.vNoise
        else:
            vNoise = [model.wfs.vNoise[0], 0., 0., model.wfs.vNoise[-1]]# neglect poisson noise

        if params['cost'][0] in ['std', 'mad', 'var']:
            for _ in range(int(params['cl'][0])):
                zest_tmp, _ = model(phi, vNoise = vNoise)
                zgt = zgt - params['cl'][1] * zest_tmp    # T[b,z]
                for i in range(phi.shape[0]): 
                    phi[i, 0:1, :, :] = phi[i, 0:1, :, :] - params['cl'][1] * (torch.reshape(modes@zest_tmp[i, :], (1, 1, model.wfs.nPx, model.wfs.nPx))) 
            loss = torch.mean(params['cost'][1](phi[:, :, model.wfs.pupilLogical] * (1/model.k))) * 1e9                 # select only values inside the pupil T[b]-> T[1]

        elif params['cost'][0] in ['mse', 'rmse', 'mae']:
            for _ in range(int(params['cl'][0])):
                with torch.no_grad():
                    zest_tmp, _ = model(phi, vNoise = vNoise)
                    zgt = zgt - params['cl'][1] * zest_tmp# T[b,z]
                    for i in range(phi.shape[0]):
                        phi[i, 0:1, :, :] = phi[i, 0:1, :, :] - params['cl'][1] * (torch.reshape(modes@zest_tmp[i, :], (1, 1, model.wfs.nPx, model.wfs.nPx)))
            zest, _ = model(phi, vNoise = vNoise)# T[b,z]
            loss = torch.mean(params['cost'][1](zgt * (1/model.k), zest * (1/model.k))) * 1e9 # T[b]-> T[1]|
        loss.backward()
        # step optimizer
        if params['fp'][0] and model.nDE:
            optimizer_de.step()
        if params['fp'][1] and hasattr(model, 'NN'):
            optimizer_nn.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss/len(train_data)
    
    return avg_loss
#
def validation(params, model, val_data): # test with the vNoise set regardless it type because this is testing 
    rmse_nn = torch.zeros(len(val_data) * params['batch'], dtype = model.precision.real, device = model.device)
    Zest    = torch.zeros((len(val_data) * params['batch'], len(model.wfs.jModes)), dtype = model.precision.real, device = 'cpu')
    Zgt     = torch.zeros_like(Zest)
    modes   = model.wfs.modes

    for i,data in tqdm(enumerate(val_data), total = len(val_data), desc = 'Validation data', bar_format = bar_format):
        phi = data[0].to(model.device).to(model.precision.real)      # T[b,1,N,M]
        zgt = data[1].to(model.device).to(model.precision.real)[:, :] # T[b,z]##
        b = np.arange(params['batch'] * i, (i + 1) * params['batch'])

        with torch.no_grad(): # force the test mode, gradients frozen
            if params['cost'][0] in ['std', 'mad', 'var']:
                for _ in range(int(params['cl'][0])):
                    zest_tmp, _ = model(phi)
                    zgt = zgt - params['cl'][1] * zest_tmp # T[b,z]
                    for i in range(phi.shape[0]):
                        phi_corr = params['cl'][1] * (torch.reshape(modes@zest_tmp[i, :] , (1, 1, model.wfs.nPx, model.wfs.nPx))) 
                        phi[i, 0:1, :, :] = phi[i, 0:1, :, :] - phi_corr
                loss = (params['cost'][1](phi[:, :, model.wfs.pupilLogical] * (1/model.k))) * 1e9# select only values inside the pupil T[b]-> T[1]

            elif params['cost'][0] in ['mse', 'rmse', 'mae']:
                for _ in range(int(params['cl'][0])):
                    zest_tmp,_ = model(phi)
                    zgt = zgt - params['cl'][1] * zest_tmp# T[b,z]
                    for i in range(phi.shape[0]):
                        phi_corr = params['cl'][1] * (torch.reshape(modes@zest_tmp[i, :], (1, 1, model.wfs.nPx, model.wfs.nPx))) 
                        phi[i, 0:1, :, :] = phi[i, 0:1, :, :] - phi_corr
                zest, _ = model(phi)# T[b,z] 
                loss = (params['cost'][1](zgt * (1/model.k), zest * (1/model.k))) * 1e9# T[b]-> T[1]             
                
                Zest[b, :] = zest.detach().cpu()
                Zgt[b, :] = zgt.detach().cpu()

            rmse_nn[b] = loss

            if int(params['cl'][0]):
                phis = {'phi_res':phi[-1, ...].detach().cpu().numpy().squeeze(), 'phi_corr':phi_corr.detach().cpu().numpy().squeeze()} 
            else:
                phis = {'phi_res':np.zeros((model.wfs.nPx, model.wfs.nPx)), 'phi_corr':np.zeros((model.wfs.nPx, model.wfs.nPx))} 
                
    return torch.mean(rmse_nn), phis