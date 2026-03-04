# Richards–Wolf Simulation

Python code for simulating highly focused laser beams using the vectorial diffraction formalism developed by Richards and Wolf. The program computes the full three-dimensional electromagnetic field in the focal region of a high numerical aperture (NA) optical system and derives the corresponding 3D Stokes parameters for complete polarization analysis.

---

## Features

- Vectorial Richards–Wolf diffraction integral implementation  
- Full 3D electric field calculation (Ex, Ey, Ez)  
- Support for arbitrary incident beam profiles  
- Custom polarization states via Jones vectors  
- Projection onto the Gaussian reference sphere  
- Multiple numerical integration methods:
  - Direct summation  
  - Trapezoidal rule  
  - Simpson rule  
  - Vectorized accelerated methods  
  - FFT-based approximation  
- 3D Stokes parameter computation  
- Advanced visualization of intensity, phase, and polarization ellipses  

---

## Requirements

- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib  
- tqdm  

Install dependencies with:

```bash
pip install numpy scipy matplotlib tqdm
```

---

## Basic Usage

```python
from RW_simulation import RichardsWolf
import numpy as np

rw = RichardsWolf(
    NA=0.95,
    n_1=1.0,
    n_2=1.0,
    N_theta=100,
    N_phi=100,
    f=5e-3,
    t_s=1,
    t_p=1,
    lamb=500e-9
)

# Example: radially polarized beam
alpha0 = rw.Phi_xy
J_x = np.cos(alpha0)
J_y = np.sin(alpha0)
E_profile = rw.rho_xy * np.exp(-(rw.rho_xy**2))

rw.set_beam(J_x, J_y, E_profile)
rw.calculate_reference_sphere()
rw.calculate_focus(L_xy=1.5*rw.lamb, N_prime_pixel=100, z=0.0, mode="vectorized simpson")
rw.show_focus()
```

---

## Applications

- High-NA microscopy  
- Optical trapping  
- Structured light engineering  
- Nanophotonics  
- Nonparaxial polarization analysis  

---

## Author

David Sáiz Martínez  
Bachelor’s Thesis in Physics  
Universitat de Barcelona  

---
