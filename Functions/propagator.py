from __future__ import annotations
import torch, types
from dataclasses import dataclass
from typing import Union
from math import pi
from Functions.utils import get_precision

@dataclass
class TorchField:
    field:      torch.Tensor      # [B,1,N,N]  complejo
    grid_size:  float             # [m]
    wavelength: float             # [m]
    precision:  types.SimpleNamespace

    # Propiedades
    @property
    def Npx(self) -> int:        return self.field.shape[-1]
    @property
    def dx (self) -> float:      return self.grid_size / self.Npx
    @property
    def dtype(self):             return self.field.dtype
    @property
    def device(self):            return self.field.device

    # Duplicar
    def copy(self) -> "TorchField":
        return TorchField(self.field.clone(), self.grid_size,
                          self.wavelength, self.precision)

    # Mover a otro dispositivo o cambiar precisión
    def to(self,*args,**kw):     
        return TorchField(self.field.to(*args,**kw), self.grid_size,
                          self.wavelength, self.precision)

    # Crear mallas cartesianas
    @property
    def mgrid_cartesian(self):
        N, dx, dev, rdt = self.Npx, self.dx, self.device, self.precision.real
        idx = torch.arange(N, dtype=rdt, device=dev) - (N//2)
        Y, X = torch.meshgrid(idx, idx, indexing='ij')
        return Y*dx, X*dx

    # Inicializador de campo
    @classmethod
    def begin(cls, grid_size, wavelength, Npx, *, precision='double', batch=1, device='cpu'):
        p = get_precision(precision)
        t = torch.ones((batch,1,Npx,Npx), dtype=p.complex, device=device)
        return cls(t, grid_size, wavelength, p)

    # Apertura circular
    def circ_aperture(self, R):
        Fout = self.copy()
        N1 = Fout.Npx
        dev = Fout.device

        # Índices enteros equivalentes a 2*(i - (N1-1)/2)  →  [- (N1-1), ..., (N1-1)]
        idx2 = torch.arange(N1, device=dev, dtype=torch.int64) * 2 - (N1 - 1)
        Y2, X2 = torch.meshgrid(idx2, idx2, indexing='ij')

        thr = ((N1/4) - 1)                 # radio en "enteros dobles"
        # print(thr)
        mask = (X2*X2 + Y2*Y2) <= (thr*thr)

        Fout.field = Fout.field * mask.to(dtype=Fout.dtype).unsqueeze(0).unsqueeze(0)

        # Fout = self.copy()
        # Y,X   = Fout.mgrid_cartesian
        # #Y -= y_shift
        # #X -= x_shift
        # mask  = (X**2 + Y**2) <= R**2
        # mask_batch = mask.to(dtype=Fout.dtype).unsqueeze(0).unsqueeze(0)

        # Fout.field = Fout.field * mask_batch

        return Fout

    # Aplicar Phase de entrada
    def apply_phase(self, phi: torch.Tensor | float) -> "TorchField":

        Fout = self.copy()

        if not torch.is_tensor(phi):
            phi = torch.tensor(phi, device=self.device, dtype=self.precision.real)

        phi = phi.to(device=self.device, dtype=self.precision.real)

        if   phi.ndim == 2: phi = phi.unsqueeze(0).unsqueeze(0)
        elif phi.ndim == 4: pass
        else:
            raise ValueError("phi debe ser escalar, 2-D o 4-D")

        Fout.field = Fout.field * torch.exp(1j * phi).to(Fout.dtype)
        return Fout
    
    # Aplicar máscara en el dominio espacial
    def apply_mask(self, mask: torch.Tensor, *, inplace: bool = False) -> "TorchField":

        if not torch.is_tensor(mask):
            raise TypeError("mask debe ser torch.Tensor")

        mask = mask.to(device=self.device, dtype=self.dtype)

        if   mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)          # [1,1,N,N]
        elif mask.ndim == 4:
            pass                                     # ya está B,1,N,N
        else:
            raise ValueError("La máscara debe ser 2-D o 4-D")

        if inplace:
            self.field.mul_(mask)
            return self
        else:
            Fout = self.copy()
            Fout.field.mul_(mask)
            return Fout

    
    # Propagación Fresnel aproximada
    def propagate_fresnel(self, z: float) -> "TorchField":
        
        if z == 0.0:
            return self.copy()

        B, _, N, _ = self.field.shape
        dev = self.device
        p   = self.precision
        wavelength = self.wavelength
        size = self.grid_size
        dtype_c = p.complex
        dtype_r = p.real

        field = self.field.clone()# [B,1,N,N]

        # equivalente a fftshift
        iiN  = torch.ones(N, dtype=dtype_c, device=dev)
        iiN[1::2] = -1
        iiij = iiN.view(1,1,N,1) * iiN.view(1,1,1,N)
        field = field * iiij

        _2pi = 2.0 * 3.141592654
        zz   = z
        z    = abs(z)
        kz   = _2pi / wavelength * z
        global_phase = torch.exp(1j * torch.tensor(kz, dtype=dtype_r, device=dev))

        z1   = z * wavelength / 2.0
        No2  = N // 2
        SW   = torch.arange(-No2, N-No2, dtype=dtype_r, device=dev) / size
        SW2  = SW**2
        SSW  = SW2.view(N,1) + SW2.view(1,N)
        Bus  = z1 * SSW
        Ir   = torch.floor(Bus)
        Abus = _2pi * (Ir - Bus)
        CC   = torch.cos(Abus) + 1j*torch.sin(Abus)
        CC   = CC.to(dtype=dtype_c).unsqueeze(0).unsqueeze(0)

        if zz >= 0.0:
            field = torch.fft.fft2(field)
            field = field * CC
            field = torch.fft.ifft2(field)
        else:
            field = torch.fft.ifft2(field)
            field = field * CC.conj()
            field = torch.fft.fft2(field)

        field = field * global_phase
        field = field * iiij

        return TorchField(field, self.grid_size, self.wavelength, self.precision)
    
    # Propagación ASM
    def propagate_asm(self, z: float, *, scale_factor: float = 1.0) -> "TorchField":

        if z == 0.0:
            return self.copy()

        B, _, N, _ = self.field.shape
        dx = self.dx
        wavelength = self.wavelength
        dev = self.device
        p   = self.precision
        k0  = 2.0 * torch.pi / wavelength

        U_f = torch.fft.fftshift(torch.fft.fft2(self.field), dim=(-2, -1))

        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx, device=dev, dtype=p.real))
        fy = fx

        FY, FX = torch.meshgrid(fy, fx, indexing='ij')

        kx = 2 * torch.pi * FX
        ky = 2 * torch.pi * FY

        kz_sq = k0**2 - kx**2 - ky**2
        kz = torch.where(kz_sq >= 0, torch.sqrt(kz_sq), 1j * torch.sqrt(-kz_sq))
        kz = kz.to(dtype=p.complex)
        kz = kz.unsqueeze(0).unsqueeze(0) # [1,1,N,N]

        H = torch.exp(1j * kz * z) # [1,1,N,N]

        U_prop = torch.fft.ifft2(torch.fft.ifftshift(U_f * H, dim=(-2, -1)))

        return TorchField(U_prop, self.grid_size, self.wavelength, self.precision)

    # Propagación ASM con padding
    def propagate_asm_pad(self, z, *, padding=True):
        B,C,N,_ = self.field.shape #### se coloco la C ##
        dx,lam  = self.dx, self.wavelength
        k       = 2*pi/lam; p=self.precision

        if padding:
            Nfft = 2*N
            pad  = torch.zeros((B,C,Nfft,Nfft), dtype=self.dtype, device=self.device)
            pad[..., :N, :N] = self.field
            U = torch.fft.fft2(pad)
        else:
            Nfft = N
            U = torch.fft.fft2(self.field)

        fx = torch.fft.fftfreq(Nfft, d=dx, device=self.device, dtype=p.real)
        fy = fx
        FY,FX = torch.meshgrid(fy,fx,indexing='ij')
        H = torch.exp(1j*k*z*torch.sqrt(
             1 - (lam*FX)**2 - (lam*FY)**2 )).to(dtype=p.complex)

        U2 = torch.fft.ifft2(U*H)
        if padding: U2 = U2[...,:N,:N]

        return TorchField(U2, self.grid_size, self.wavelength, self.precision)

    # Lente
    def lens(self, f: float) -> "TorchField":

        Fout = self.copy()

        k = 2.0 * torch.pi / Fout.wavelength

        Y, X = Fout.mgrid_cartesian

        phi   = -k * (X**2 + Y**2) / (2.0 * f)
        phase = torch.exp(1j * phi).to(dtype=Fout.dtype)

        phase = phase.unsqueeze(0).unsqueeze(0)
        Fout.field = Fout.field * phase
        return Fout



# ──── alias LightPipes style ──────────────────────────────────────
def Begin(size, lam, N, **kw): return TorchField.begin(size, lam, N, **kw)
Ini = Begin
