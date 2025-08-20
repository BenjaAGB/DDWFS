import os
import torch
import argparse
import numpy as np
import scipy.io as sio
from LightPipes import Begin, CircAperture, Forvard, Lens, Intensity, cm, mm, nm
import matplotlib.pyplot as plt
from Functions import fourier_mask
import matplotlib.animation as ani
from matplotlib.animation import PillowWriter

parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')

parser.add_argument('--wl',       type = float, default = 635, help = 'Wavelength in nm')
parser.add_argument('--size',     type = float, default = 7,   help = 'Size of the field in mm')
parser.add_argument('--Npx',      type = int,   default = 512, help = 'Number of pixels')
parser.add_argument('--f1',       type = float, default = 100, help = 'Focal length of the first lens in mm')
parser.add_argument('--f2',       type = float, default = 100, help = 'Focal length of the second lens in mm')
parser.add_argument('--R',        type = float, default = .5,  help = 'Radius of the aperture in mm')
parser.add_argument('--alpha',    type = float, default = 3,   help = 'Angle of the PWFS in degrees')
parser.add_argument('--nhead',    type = int,   default = 6,   help = 'Number of heads')
parser.add_argument('--dz',       type = float, default = 1.,  help = 'Propagation step distance in mm')
parser.add_argument('--phase_in', type = int,   default = 3,   help='Phase input type: 0=None, 1=Uniform, 2=Random, 3=ATM Phase')

args = parser.parse_args()

wavelength   = args.wl * nm   # [m] Longitud de onda
size         = args.size * mm # [m] Tamaño del campo
Npx          = args.Npx       # Número de píxeles
f1           = args.f1 * mm   # [m] Distancia focal del primera lente
f2           = args.f2 * mm   # [m] Distancia focal de la segunda lente
R            = args.R * mm    # [m] Radio de la apertura
alpha        = args.alpha     # [°] angulo de la PWFS
nhead        = args.nhead     # Número de caras
dz           = args.dz * mm   # Distancia de paso en la propagación
phase_in     = args.phase_in  # Tipo de fase de entrada: 0=None, 1=Uniforme, 2=Aleatoria, 3=Fase ATM

z_l1  = f1              # Lente 1
z_psf = 2 * f1          # Plano PSF (2f)
z_l2  = 2 * f1 + f2     # Lente 2
z_max = 2 * f1 + 2 * f2 # 4f imagen

folder = 'animations_propagation/'

if not os.path.exists(folder):
    os.makedirs(folder)


fourier_mask, _ = fourier_mask.fourier_geometry(alpha=alpha, nhead=nhead, resol=Npx)
fourier_mask = torch.fft.fftshift(fourier_mask)
mask_np = fourier_mask.squeeze().cpu().numpy()

phase_in = 3

if phase_in == 0:
    phi = None
    phase_in = 'zero'
elif phase_in == 1:
    phi = np.ones((Npx, Npx))
    phase_in = 'uniform'
elif phase_in == 2:
    mask = (np.hypot(np.linspace(-size/2, size/2, Npx)[:, None], np.linspace(-size/2, size/2, Npx)[None, :]) <= R).astype(float)
    phi = mask * np.random.rand(Npx, Npx) * 2 * np.pi
    phase_in = 'random'
elif phase_in == 3:
    phi = sio.loadmat('/data2/bgonzalez/desktop/test/DDWFS/DPWFS/dataset/D3_R512_Dro10-100_Z200_T500_αβ11_single_1/part_0.mat')['Phi'][1,:,:,:].squeeze()
    phase_in = 'atm_phase'
else:
    raise ValueError("Invalid value for phase_in. Use 0, 1, or 3.")

name = 'Phase_ini_' + phase_in + '_Npx_' + str(Npx) + '_alpha_' + str(alpha) + '_nhead_' + str(nhead) + '_R_' + str(R/1e-3) + '_size_' + str(size/1e-3) + '_dz_' + str(dz/1e-3) + '_mm_' + 'f1_' + str(f1/1e-3) + '_f2_' + str(f2/1e-3) + '_mm'


F = Begin(size, wavelength , Npx)
F = CircAperture(F, R)

F.field = F.field * np.exp(1j * phi) if phi is not None else F.field

I, labels, zcoord = [], [], []

def store(tag, zpos, hold=1):
    img = Intensity(1, F)
    for _ in range(hold):
        I.append(img)
        labels.append(tag)
        zcoord.append(zpos/1e-3)

z = 0
store('Field In', z, hold = 15)

point_lens = np.array([z_l1, z_psf, z_l2, z_max])
z_targets  = np.unique(np.concatenate([np.arange(0, z_max, dz), point_lens]))

for z_next in z_targets[0:]:

    dz_step = z_next - z

    if abs(dz_step) < 1e-12:
        continue

    F = Forvard(F, dz_step)

    z += dz_step

    if np.isclose(z, z_l1, atol=dz/2):
        # store(f'Antes L1 (z={z/mm:.0f} mm)', z)
        F = Lens(F, f1)
        # store(f'Después L1', z)

    elif np.isclose(z, z_psf, atol=dz/2):
        # store('Before Mask', z)
        F.field *= mask_np
        # store('After Mask', z)
        continue

    elif np.isclose(z, z_l2, atol=dz/2):
        # store(f'Antes L2 (z={z/mm:.0f} mm)', z)
        F = Lens(F, f2)
        # store(f'Después L2', z)

    elif np.isclose(z, z_max, atol=dz/2):
        store(f'Field out', z, hold=15)

    else:
        store(f'z = {z/1e-3:.0f} mm', z)

fig, ax = plt.subplots(figsize=(5,5))
extent = [-size/2/1e-3, size/2/1e-3, -size/2/1e-3, size/2/1e-3]
fig.dpi = 100

# ttl = ax.set_title(labels[0], fontsize=11)
# ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')

ax.set_axis_off()              # borra ticks + marco
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # sin márgenes

im = ax.imshow(I[0], cmap='hot', extent=extent, animated=True)

txt = ax.text(0.02, 0.98,           # posición (relativa 0–1)
              labels[0],            # primer texto  
              color='white',
              fontsize=12,
              ha='left', va='top',
              transform=ax.transAxes)

# def update(i):
#     im.set_array(I[i])
#     ttl.set_text(labels[i])
#     return im, ttl

def update(k):
    idx = k
    im.set_array(I[idx])
    txt.set_text(labels[idx])
    return im, txt

fps = 10
movie = ani.FuncAnimation(fig, update, frames=len(I), interval=1000/fps, blit=True)

frames = np.stack(I, axis=0)

sio.savemat(folder + name + '.mat', {'frames': frames})

# gif_path = folder + name + '.gif'
# writer   = PillowWriter(fps=fps, codec=None)

# movie.save(gif_path, writer=writer)
# print(f"GIF guardado en → {gif_path}")