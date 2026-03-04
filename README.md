# Richards–Wolf Simulation

Python code for simulating highly focused vector beams using the vectorial diffraction formalism developed by Richards and Wolf. The program computes the full three-dimensional electromagnetic field in the focal region of a high numerical aperture (NA) optical system and derives the corresponding generalized 3D Stokes parameters for complete polarization analysis of nonparaxial fields.

---

## Overview

This project implements a numerical framework for:

- Computing the vectorial electromagnetic field near focus using the Richards–Wolf diffraction integral  
- Including longitudinal electric field components  
- Characterizing the local polarization state through a three-dimensional generalization of the Stokes parameters  

The implementation is suitable for strongly nonparaxial regimes where scalar and transverse approximations fail.

---

## Features

- Vectorial Richards–Wolf diffraction integral implementation  
- Full 3D electric field computation (Ex, Ey, Ez)  
- Projection onto the Gaussian reference sphere  
- Custom incident beam profiles  
- Arbitrary polarization states via Jones vectors  
- Multiple numerical integration methods:
  - Direct summation  
  - Trapezoidal rule  
  - Simpson rule  
  - Vectorized accelerated methods  
  - FFT-based approximation  
- 3D Stokes parameter calculation  
- Visualization tools:
  - Intensity maps  
  - Phase distributions  
  - Polarization ellipses  
  - 3D orientation of the polarization normal vector  

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

## Basic Usage Example

```python
from RW_simulation import RichardsWolf
import numpy as np

# Initialize optical system
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

# Projection to reference sphere
rw.calculate_reference_sphere()

# Compute focused field at focal plane (z = 0)
rw.calculate_focus(
    L_xy=1.5*rw.lamb,
    N_prime_pixel=100,
    z=0.0,
    mode="vectorized simpson"
)

# Compute generalized 3D Stokes parameters
rw.calculate_Stokes()

# Visualization
rw.show_focus()
rw.show_Stokes()
```

---

## Theoretical References

[1] Richards, B., & Wolf, E.  
Electromagnetic diffraction in optical systems. II. Structure of the image field in an aplanatic system.  
Proceedings of the Royal Society A, 253, 358–379 (1959).

[2] Martínez-Herrero, R., Maluenda, D., Aviñóà, M., Carnicer, A., Juvells, I., & Sanz, Á. S.  
Local characterization of the polarization state of 3D electromagnetic fields: an alternative approach.  
Photonics Research, 11(7), 1326–1338 (2023).

---

## Applications

- High-NA microscopy  
- Optical trapping  
- Structured light  
- Nanophotonics  
- Nonparaxial polarization analysis  

---

## Author

David Sáiz Martínez  
Bachelor’s Thesis in Physics  
Universitat de Barcelona
