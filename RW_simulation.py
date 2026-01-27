# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 11:08:14 2026

@author: David Saiz Martinez
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.integrate import simpson
from scipy.interpolate import griddata
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import time


class RichardsWolf:
    
    """
    
    This class implements the vectorial Richards–Wolf diffraction integral
    to calculate the electric field (E field) and the intensity from a 
    strongly focused beam.
    
    To do it, it provides a method to define the incident field with its
    profile and polaritzation, calculate the field on the reference sphere
    and integrate it into the chosen z-plane (z = 0.0 is the focal plane).

    Parameters
    ----------
    
    NA : float
        Numerical aperture of the objective.
        
    n_1 : float
        Refractive index of the first medium (incident medium).
        
    n_2 : float
        Refractive index of the second medium (focal medium).
        
    N_theta : int
        Number of sampling points in polar angle (theta).
        
    N_phi : int
        Number of sampling points in azimuthal angle (phi).
        
    f : float
        Focal length of the lens/objective.

    t_s : float
        Transmission coefficient for s-polarized light.
        
    t_p : float
        Transmission coefficient for p-polarized light.
        
    lamb : float
        Wavelength of the light in vacuum.
        
    """

    def __init__(self, NA, n_1, n_2, N_theta, N_phi, f, t_s, t_p, lamb):
        self.NA = NA
        self.n_1 = n_1
        self.n_2 = n_2
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.f = f
        self.t_s = t_s
        self.t_p = t_p
        self.lamb = lamb

        
        self.theta_max = np.arcsin(NA/n_2)
        self.k = n_2 * (2*np.pi/lamb)
        
        self.theta_lin = np.linspace(0, self.theta_max, self.N_theta)
        self.phi_lin = np.linspace(0, 2*np.pi, self.N_phi)
        self.THETA, self.PHI = np.meshgrid(self.theta_lin, self.phi_lin)

        # Unitary vector in cartesian components
        self.n_x = np.array([1, 0, 0])
        self.n_y = np.array([0, 1, 0])
        self.n_z = np.array([0, 0, 1])

        # Cartesian to spherical components transformation (eq. 3.40, 3.41, 3.42 // page 59 [16])
        self.n_rho = np.stack([np.cos(self.PHI), np.sin(self.PHI), np.zeros_like(self.PHI)], axis=-1)
        self.n_phi = np.stack([-np.sin(self.PHI), np.cos(self.PHI), np.zeros_like(self.PHI)], axis=-1)
        self.n_theta = np.stack([np.cos(self.THETA)*np.cos(self.PHI),
                            np.cos(self.THETA)*np.sin(self.PHI),
                            -np.sin(self.THETA)], axis=-1)
        
        # Cartesian coordinates
        self.x_max = self.f * np.sin(self.theta_max)   # length unit same as f
        
        x = np.linspace(-self.x_max, self.x_max, self.N_theta)
        y = np.linspace(-self.x_max, self.x_max, self.N_phi)
        self.X_inc, self.Y_inc = np.meshgrid(x, y)

        # Precompute common polar quantities on that cartesian grid
        self.rho_xy = np.sqrt(self.X_inc**2 + self.Y_inc**2)
        self.Phi_xy = np.arctan2(self.Y_inc, self.X_inc)
        # R (distance from origin) - default plane z=0
        self.R_xy = np.sqrt(self.X_inc**2 + self.Y_inc**2 + 0.0**2)
        
        # Generate attributes that are used to avoid errors
        self.I_inc = None
        self.I_inf = None
        self.I_focus = None
        self.N_focus = None

        
    def set_beam(self, J_x, J_y, E_inc_profile):

        """
        Set the profile and the polaritzation of the incident beam. If the J_x,
        J_y or/and E_inc_profile are scalars, they are converted into a ndarray.

        Parameters
        ----------
        
        J_x: ndarray or complex
        Component along the x-axis Jones matrix that determines the polaritzation 
        of the incident beam.
        
        J_y: ndarray or complex
        Component along the y-axis Jones matrix that determines the polaritzation 
        of the incident beam.
        
        
        E_inc_profile: ndarray or complex
        Profile of the incident beam in spherical coordinates.
        
        """

        # E_inc_vec = E_0 * n_rho
        # E_inc_vec = E_0 * n_phi
        # E_inc_vec = E_0 * ((1/np.sqrt(2)) * (n_x + 1j * n_y))
        # E_inc_vec = E_0 * ((1/np.sqrt(2)) * (n_x - 1j * n_y))

        
        if not isinstance(J_x, np.ndarray):
            J_x *= np.ones_like(self.THETA)
            
        if not isinstance(J_y, np.ndarray):
            J_y *= np.ones_like(self.THETA)
            
        if not isinstance(E_inc_profile, np.ndarray):
            E_inc_profile *= np.ones_like(self.THETA)


        self.Jones_gen = np.stack([J_x, J_y], axis=-1)
        self.E_inc_profile = E_inc_profile

    
        self.E_inc_vec = (self.Jones_gen[...,0, None] * self.n_x[None, None, :] + self.Jones_gen[...,1, None] * self.n_y[None, None, :]) * self.E_inc_profile[..., None]


        # Calculate de 's' (phi) and 'p' (rho) components (eq. 3.38, 3.39 // page 58 [15])
        # We used 'np.sum' because of E_inc_vec size
        self.E_inc_phi = np.sum(self.E_inc_vec * self.n_phi, axis=-1)
        self.E_inc_rho = np.sum(self.E_inc_vec * self.n_rho, axis=-1)
        

        # Cartesian components from the incident field
        self.Ex_inc = self.E_inc_vec[..., 0]
        self.Ey_inc = self.E_inc_vec[..., 1]
        self.Ez_inc = self.E_inc_vec[..., 2]  #  equals 0
        
        # Phases from the incident field
        self.Phase_x_inc = np.angle(self.E_inc_vec[..., 0])
        self.Phase_y_inc = np.angle(self.E_inc_vec[..., 1])
        #self.Phase_yx_inc = np.angle(self.E_inc_vec[..., 1]/self.E_inc_vec[..., 0])
        self.Phase_yx_inc = self.Phase_y_inc - self.Phase_x_inc
        
        # Total incident intensity
        self.I_inc = np.linalg.norm(self.E_inc_vec, axis=-1)**2
        
        # Intensity of the individual field components
        self.Ix_inc = np.abs(self.E_inc_vec[..., 0])**2
        self.Iy_inc = np.abs(self.E_inc_vec[..., 1])**2
        self.Iz_inc = np.abs(self.E_inc_vec[..., 2])**2
        
        
    def show_incident_beam(self):
        
        """
        
        Display the x and y components of the total intensity and the phases 
        from the incident beam. 
        
        """
        
        if self.I_inc is None:
            print("You need to set the incident beam first.")
        
        else:
        
            # Original points in spherical coordinates (THETA, PHI) in "f" units
            X_sph = self.f * np.sin(self.THETA) * np.cos(self.PHI)
            Y_sph = self.f * np.sin(self.THETA) * np.sin(self.PHI)
            
            # Interpolate Ex, Ey, Ez on the Cartesian grid
            points = np.stack((X_sph.ravel(), Y_sph.ravel()), axis=-1)
            
            Ex_inc_cart = griddata(points, self.Ex_inc.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=0)
            Ey_inc_cart = griddata(points, self.Ey_inc.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=0)
            Ez_inc_cart = griddata(points, self.Ez_inc.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=0)
                        
            # Stokes parameters
            S0 = np.abs(Ex_inc_cart)**2 + np.abs(Ey_inc_cart)**2
            S1 = np.abs(Ex_inc_cart)**2 - np.abs(Ey_inc_cart)**2
            S2 = 2 * np.real(Ex_inc_cart * np.conj(Ey_inc_cart))
            S3 = -2 * np.imag(Ex_inc_cart * np.conj(Ey_inc_cart))
            
            # Orientation angle ψ
            psi = 0.5 * np.arctan2(S2, S1)
               
            # Ellipticity angle χ
            chi = 0.5 * np.arcsin(np.clip(S3 / (S0 + 1e-20), -1, 1))
            
            # Ellipse axis ratio b = |tan χ|
            b = np.abs(np.tan(chi)) # 0=lineal, 1=circular
            
            zeros_array1 = np.zeros_like(self.I_inc, dtype=float)
            zeros_array2 = np.zeros_like(self.I_inc, dtype=float)
            zeros_array3 = np.zeros_like(self.I_inc, dtype=float)
            
            fig = plt.figure(figsize = (12, 8))
            fig.suptitle('Incident beam', fontsize=18, fontweight='bold', y = 1.00)
 
        # Intensity    
            grid1 = ImageGrid(fig, (2, 1, 1), nrows_ncols=(1, 3), axes_pad=0.4,
                              cbar_mode='single', cbar_location='right')

            I_inc_cart = np.abs(Ex_inc_cart)**2 + np.abs(Ey_inc_cart)**2 + np.abs(Ez_inc_cart)**2
            Ix_inc_cart = np.abs(Ex_inc_cart)**2
            Iy_inc_cart = np.abs(Ey_inc_cart)**2
            
            intensity_inc = [(np.divide(I_inc_cart, np.max(I_inc_cart), out=zeros_array1, where=(np.max(I_inc_cart) > 1e-20)), 'hot', r'$|E_{inc}(x, y, z)|^2$'),
                             (np.divide(Ix_inc_cart, np.max(Ix_inc_cart), out=zeros_array2, where=(np.max(Ix_inc_cart) > 1e-20)), 'hot', r'$|E_x^{inc}|^2$'),
                             (np.divide(Iy_inc_cart, np.max(Iy_inc_cart), out=zeros_array3, where=(np.max(Iy_inc_cart) > 1e-20)), 'hot', r'$|E_y^{inc}|^2$')]

            
            # --- Automatic polarization sampling (fixed number per axis) ---
            N_pol = 10
            arrow_step = max(1, self.X_inc.shape[0] // N_pol)
            
            for idx, (ax, (data, cmap, title)) in enumerate(zip(grid1, intensity_inc)):
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, origin='lower', extent=[-self.x_max/self.f, self.x_max/self.f, -self.x_max/self.f, self.x_max/self.f])
                ax.set_aspect('equal')
                ax.set_title(title)
                ax.set_xlabel('x/f')
                ax.set_ylabel('y/f')
                ax.axis('equal')
                
                # Set background in black
                ax.set_facecolor('black')
                
                # Visible rang
                x_min, x_max = -self.x_max/self.f, self.x_max/self.f
                y_min, y_max = -self.x_max/self.f, self.x_max/self.f  # assume square grid
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                arrow_scale = 0.08 * x_range  # arrow length = 8% of range
                head_size = 0.05 * x_range   # size of arrow = 5% of range
                
                
                # We only draw the polarization indicators in the first subplot (total intensity)
                if idx == 0:
                    # Draw polarization indicators
                    for i in range(0, self.X_inc.shape[0], arrow_step):
                        for j in range(0, self.X_inc.shape[1], arrow_step):
                        
                        
                            if S0[i, j] < 1e-6 * np.max(S0):
                                continue # Skip very low intensity
                        
                            x0 = self.X_inc[i, j]/self.f
                            y0 = self.Y_inc[i, j]/self.f
                            angle = psi[i, j]
                            bb = b[i, j]
                        
                        
                            # LINEAR polarization → draw line
                            if bb < 0.05:
                                dx = arrow_scale * np.cos(angle)
                                dy = arrow_scale * np.sin(angle)
                                start_x = x0 - dx/2
                                start_y = y0 - dy/2
                                ax.arrow(start_x, start_y, dx, dy, head_width=head_size, head_length=head_size, fc='green', ec='green')
                                
                            
                            # CIRCULAR → draw circle as an ellipse
                            elif bb > 0.95:
                                radius = arrow_scale/2
                            
                                # We assign color according to clockwise/counterclockwise rotation
                                color = 'cyan' if S3[i,j] > 0 else 'magenta'
                                
                                # Treat as an ellipse with equal axes
                                circle_as_ellipse = patches.Ellipse((x0, y0), 2*radius, 2*radius, angle=0,
                                                                    fill=False, color=color, lw=1)
                                ax.add_patch(circle_as_ellipse)
                                
                                # Triangle position on the perimeter
                                theta = -np.pi/4 if S3[i,j] > 0 else np.pi/4
                                x_arrow = x0 + radius*np.cos(theta)
                                y_arrow = y0 + radius*np.sin(theta)
                                
                                # Tangent (for a circle it is simply perpendicular to the radius)
                                dx_rot = -np.sin(theta)
                                dy_rot = np.cos(theta)
                                
                                # Triangle angle
                                angle_marker = np.degrees(np.arctan2(dy_rot, dx_rot))
                                if S3[i,j] < 0:
                                    angle_marker += 180
                                
                                t = Affine2D().rotate_deg(angle_marker)
                                
                                # Draw triangle with same color as the circle
                                ax.plot(x_arrow, y_arrow, marker=MarkerStyle(">", "full", t),
                                        color=color, markersize=arrow_scale*30)
    
                                                
                                                
                            # ELLIPTICAL → draw ellipse
                            else:
                                a = arrow_scale/2 # major axis
                                bb = max(min(bb, 1), 0) # ensure valid
                                minor = a * bb
                            
                                # Set color according to handedness
                                color = 'cyan' if S3[i,j] > 0 else 'magenta'
                                
                                # Draw ellipse
                                e = patches.Ellipse((x0, y0), 2*a, 2*minor, angle=np.degrees(angle),
                                                    fill=False, color=color, lw=1)
                                ax.add_patch(e)
                                
                                # Triangle position on the perimeter
                                theta = -np.pi/4 if S3[i,j] > 0 else np.pi/4
                                x_arrow = x0 + a*np.cos(theta)*np.cos(angle) - minor*np.sin(theta)*np.sin(angle)
                                y_arrow = y0 + a*np.cos(theta)*np.sin(angle) + minor*np.sin(theta)*np.cos(angle)
                                
                                # Tangent
                                dx = -a*np.sin(theta)
                                dy = minor*np.cos(theta)
                                dx_rot = dx*np.cos(angle) - dy*np.sin(angle)
                                dy_rot = dx*np.sin(angle) + dy*np.cos(angle)
                                length = np.hypot(dx_rot, dy_rot)
                                dx_rot /= length
                                dy_rot /= length
                                
                                # Triangle angle
                                angle_marker = np.degrees(np.arctan2(dy_rot, dx_rot))
                                if S3[i,j] < 0:
                                    angle_marker += 180
                                
                                t = Affine2D().rotate_deg(angle_marker)
                                
                                # Draw triangle with same color as the ellipse
                                ax.plot(x_arrow, y_arrow, marker=MarkerStyle(">", "full", t),
                                        color=color, markersize=arrow_scale*30)

            
            # Title row
            fig.text(0.5, 0.95, 'Normalized intensities incident beam', fontsize=14, ha='center', fontweight='bold')
                
            # Common color bar for intensity
            cbar1 = grid1.cbar_axes[0].colorbar(im)
            cbar1.set_label('Normalized Intensity')
                
        # Phase    
            grid2 = ImageGrid(fig, (2, 1, 2), nrows_ncols=(1, 3), axes_pad=0.4, 
                              cbar_mode='single', cbar_location='right')
            
            # Phase interpolation on the Cartesian grid
            Phase_x_inc_cart = griddata(points, self.Phase_x_inc.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=np.nan)
            Phase_y_inc_cart = griddata(points, self.Phase_y_inc.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=np.nan)
            Phase_yx_inc_cart = griddata(points, self.Phase_yx_inc.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=np.nan)
            
            # Low intensity mask to leave the background black
            mask = S0 < 1e-6 * np.max(S0)
            Phase_x_inc_cart[mask] = np.nan
            Phase_y_inc_cart[mask] = np.nan
            Phase_yx_inc_cart[mask] = np.nan
            
            # Create colormap that treats NaN as black
            cmap_hsv_black = plt.cm.hsv.copy()
            cmap_hsv_black.set_bad(color='black')
            
            phase_inc = [(Phase_yx_inc_cart, cmap_hsv_black, r'$\varphi_y^{inc}-\varphi_x^{inc}$'),
                         (Phase_x_inc_cart, cmap_hsv_black, r'$\varphi_x^{inc}$'),
                         (Phase_y_inc_cart, cmap_hsv_black, r'$\varphi_y^{inc}$')]

            for ax, (data, cmap, title) in zip(grid2, phase_inc):
                im2 = ax.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi, origin='lower', extent=[-self.x_max/self.f, self.x_max/self.f, -self.x_max/self.f, self.x_max/self.f])
                ax.set_aspect('equal')                
                ax.set_title(title)
                ax.set_xlabel('x/f')
                ax.set_ylabel('y/f')
                ax.axis('equal')
                
                # Set background in black
                ax.set_facecolor('black')
                
            # Title row
            fig.text(0.5, 0.46, 'Phases of E field components incident beam', fontsize=14, ha='center', fontweight='bold')
                
            # Common color bar for phase
            cbar2 = grid2.cbar_axes[0].colorbar(im2)
            cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar2.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            cbar2.set_label('Field Phase [rad]')
            
            plt.subplots_adjust(top=0.91, bottom=0.05, hspace=0.3)
            plt.show()

        
    def calculate_reference_sphere(self):
        
        """
        
        Calculate the x, y and z components from the E field and the intensity,
        the total intensity and the phases in the reference sphere. They are also 
        transformed into spherical coordinates.
        
        """
        
        if self.I_inc is None:
            print("You need to set the incident beam first.")
        
        else:
            factor = np.sqrt(self.n_1/self.n_2) * np.sqrt(np.cos(self.THETA))
            self.E_inf = (self.t_s * self.E_inc_phi[..., None] * self.n_phi +
                    self.t_p * self.E_inc_rho[..., None] * self.n_theta) * factor[..., None]

            
            # Avoid errors in the representation with a mask (be careful at normalitzation: 0/0)
            eps = 1e-12
            mask_Ex = np.abs(self.E_inf[..., 0]) > eps
            mask_Ey = np.abs(self.E_inf[..., 1]) > eps
            mask_Ez = np.abs(self.E_inf[..., 2]) > eps


            # Decompose the real part of the E field at the surface of the reference sphere
            # into the X, Y and Z coordinates 
            self.Ex_inf = self.E_inf[..., 0]
            self.Ey_inf = self.E_inf[..., 1]
            self.Ez_inf = self.E_inf[..., 2]

            
            # Phases at the reference sphere (Avoid errors with the fases)
            self.Phase_x_inf = np.where(mask_Ex, np.angle(self.E_inf[..., 0]), 0.0)
            self.Phase_y_inf = np.where(mask_Ey, np.angle(self.E_inf[..., 1]), 0.0)
            self.Phase_z_inf = np.where(mask_Ez, np.angle(self.E_inf[..., 2]), 0.0)
    
            # Total intensity in the reference sphere
            self.I_inf = np.linalg.norm(self.E_inf, axis=-1)**2
            
            # Intensity of the individual field components
            self.Ix_inf = np.abs(self.E_inf[..., 0])**2
            self.Iy_inf = np.abs(self.E_inf[..., 1])**2
            self.Iz_inf = np.abs(self.E_inf[..., 2])**2
            
            # In spherical coordinates
            self.E_inf_rho = np.sum(self.E_inf * self.n_rho, axis=-1)
            self.E_inf_phi = np.sum(self.E_inf * self.n_phi, axis=-1)
            self.E_inf_theta = np.sum(self.E_inf * self.n_theta, axis=-1)
            
            
            # Avoid errors with the fases
            mask_E_rho = np.abs(self.E_inf_rho) > eps
            mask_E_phi = np.abs(self.E_inf_phi) > eps
            mask_E_theta = np.abs(self.E_inf_theta) > eps

            self.I_rho_inf = np.abs(self.E_inf_rho)**2
            self.I_phi_inf = np.abs(self.E_inf_phi)**2
            self.I_theta_inf = np.abs(self.E_inf_theta)**2
            
            # Phases at the reference sphere (Avoid errors with the fases)
            self.Phase_rho_inf = np.where(mask_E_rho, np.angle(self.E_inf_rho), 0.0)
            self.Phase_phi_inf = np.where(mask_E_phi, np.angle(self.E_inf_phi), 0.0)
            self.Phase_theta_inf = np.where(mask_E_theta, np.angle(self.E_inf_theta), 0.0)

            # Calculate the difference into the [-pi, pi] interval
            self.Phase_phirho_inf = np.remainder((self.Phase_phi_inf - self.Phase_rho_inf) + np.pi, 2 * np.pi) - np.pi

            
    def show_reference_sphere(self):
        
        """
        
        Display the x, y and z components of the intnesity the total intensity,
        the phases and the intensity and phases of ρ and φ components
        at the reference sphere. 
        
        """
        
        if self.I_inf is None:
            print("You need to calculate the reference sphere first.")
        
        else:

            # Original points in spherical coordinates (THETA, PHI) in "f" units
            X_sph = self.f * np.sin(self.THETA) * np.cos(self.PHI)
            Y_sph = self.f * np.sin(self.THETA) * np.sin(self.PHI)
            
            # Interpolate Ex, Ey, Ez on the Cartesian grid
            points = np.stack((X_sph.ravel(), Y_sph.ravel()), axis=-1)
            
            # Interpolate electric field components to Cartesian grid
            self.Ex_inf_cart = griddata(points, self.Ex_inf.ravel(), (self.X_inc, self.Y_inc),
                                        method='cubic', fill_value=0)
            
            self.Ey_inf_cart = griddata(points, self.Ey_inf.ravel(), (self.X_inc, self.Y_inc),
                                        method='cubic', fill_value=0)
            
            self.Ez_inf_cart = griddata(points, self.Ez_inf.ravel(), (self.X_inc, self.Y_inc),
                                        method='cubic', fill_value=0)
            
            self.E_rho_inf_cart = griddata(points, self.E_inf_rho.ravel(), (self.X_inc, self.Y_inc),
                                           method='cubic', fill_value=0)

            self.E_phi_inf_cart = griddata(points, self.E_inf_phi.ravel(), (self.X_inc, self.Y_inc),
                                           method='cubic', fill_value=0)
            
            
            # Stokes parameters
            S0 = np.abs(self.Ex_inf_cart)**2 + np.abs(self.Ey_inf_cart)**2
            S1 = np.abs(self.Ex_inf_cart)**2 - np.abs(self.Ey_inf_cart)**2
            S2 = 2 * np.real(self.Ex_inf_cart * np.conj(self.Ey_inf_cart))
            S3 = -2 * np.imag(self.Ex_inf_cart * np.conj(self.Ey_inf_cart))
            
            # Orientation angle ψ
            psi = 0.5 * np.arctan2(S2, S1)
               
            # Ellipticity angle χ
            chi = 0.5 * np.arcsin(np.clip(S3 / (S0 + 1e-20), -1, 1))
            
            # Ellipse axis ratio b = |tan χ|
            b = np.abs(np.tan(chi)) # 0=lineal, 1=circular
            
            zeros_array1 = np.zeros_like(self.I_inf, dtype=float)
            zeros_array2 = np.zeros_like(self.I_inf, dtype=float)
            zeros_array3 = np.zeros_like(self.I_inf, dtype=float)
            zeros_array4 = np.zeros_like(self.I_inf, dtype=float)
            zeros_array5 = np.zeros_like(self.I_inf, dtype=float)
            zeros_array6 = np.zeros_like(self.I_inf, dtype=float)


            fig = plt.figure(figsize = (12, 8))
            fig.suptitle('Reference sphere', fontsize=18, fontweight='bold', y = 1.00)
            
            
        # Intensity    
            grid1 = ImageGrid(fig, (2, 1, 1), nrows_ncols=(1, 6), axes_pad=0.4,
                              cbar_mode='single', cbar_location='right')
            
            self.I_inf_cart = np.abs(self.Ex_inf_cart)**2 + np.abs(self.Ey_inf_cart)**2 + np.abs(self.Ez_inf_cart)**2
            self.Ix_inf_cart = np.abs(self.Ex_inf_cart)**2
            self.Iy_inf_cart = np.abs(self.Ey_inf_cart)**2
            self.Iz_inf_cart = np.abs(self.Ez_inf_cart)**2
            self.I_rho_inf_cart = np.abs(self.E_rho_inf_cart)**2
            self.I_phi_inf_cart = np.abs(self.E_phi_inf_cart)**2
            
            
            intensity_inf = [(np.divide(self.I_inf_cart, np.max(self.I_inf_cart), out=zeros_array1, where=(np.max(self.I_inf_cart) > 1e-20)), 'hot', r'$|E_{\infty}(x_{\infty}, y_{\infty}, z_{\infty})|^2$'),
                              (np.divide(self.Ix_inf_cart, np.max(self.Ix_inf_cart), out=zeros_array2, where=(np.max(self.Ix_inf_cart) > 1e-20)), 'hot', r'$|E_x^{\infty}|^2$'),
                              (np.divide(self.Iy_inf_cart, np.max(self.Iy_inf_cart), out=zeros_array3, where=(np.max(self.Iy_inf_cart) > 1e-20)), 'hot', r'$|E_y^{\infty}|^2$'),
                              (np.divide(self.Iz_inf_cart, np.max(self.Iz_inf_cart), out=zeros_array4, where=(np.max(self.Iz_inf_cart) > 1e-20)), 'hot', r'$|E_z^{\infty}|^2$'),
                              (np.divide(self.I_rho_inf_cart, np.max(self.I_rho_inf_cart), out=zeros_array5, where=(np.max(self.I_rho_inf_cart) > 1e-20)), 'hot', r'$|E_{\rho}^{\infty}|^2$'),
                              (np.divide(self.I_phi_inf_cart, np.max(self.I_phi_inf_cart), out=zeros_array6, where=(np.max(self.I_phi_inf_cart) > 1e-20)), 'hot', r'$|E_{\phi}^{\infty}|^2$')]
            
            
            # --- Automatic polarization sampling (fixed number per axis) ---
            N_pol = 10
            arrow_step = max(1, self.X_inc.shape[0] // N_pol)
            
            for idx, (ax, (data, cmap, title)) in enumerate(zip(grid1, intensity_inf)):
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, origin='lower', extent=[-self.x_max/self.f, self.x_max/self.f, -self.x_max/self.f, self.x_max/self.f])
                ax.set_aspect('equal')
                ax.set_title(title)
                ax.set_xlabel('x/f')
                ax.set_ylabel('y/f')
                ax.axis('equal')
                
                # Set background in black
                ax.set_facecolor('black')

                # Visible rang
                x_min, x_max = -self.x_max/self.f, self.x_max/self.f
                y_min, y_max = -self.x_max/self.f, self.x_max/self.f  # assume square grid
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                arrow_scale = 0.08 * x_range  # arrow length = 8% of range
                head_size = 0.05 * x_range   # size of arrow = 5% of range
                
                
                # We only draw the polarization indicators in the first subplot (total intensity)
                if idx == 0:
                    # Draw polarization indicators
                    for i in range(0, self.X_inc.shape[0], arrow_step):
                        for j in range(0, self.X_inc.shape[1], arrow_step):
                        
                        
                            if S0[i, j] < 1e-6 * np.max(S0):
                                continue # Skip very low intensity
                        
                            x0 = self.X_inc[i, j]/self.f
                            y0 = self.Y_inc[i, j]/self.f
                            angle = psi[i, j]
                            bb = b[i, j]
                        
                        
                            # LINEAR polarization → draw line
                            if bb < 0.05:
                                dx = arrow_scale * np.cos(angle)
                                dy = arrow_scale * np.sin(angle)
                                start_x = x0 - dx/2
                                start_y = y0 - dy/2
                                ax.arrow(start_x, start_y, dx, dy, head_width=head_size, head_length=head_size, fc='green', ec='green')
                                
                            
                            # CIRCULAR → draw circle as an ellipse
                            elif bb > 0.95:
                                radius = arrow_scale/2
                            
                                # We assign color according to clockwise/counterclockwise rotation
                                color = 'cyan' if S3[i,j] > 0 else 'magenta'
                                
                                # Treat as an ellipse with equal axes
                                circle_as_ellipse = patches.Ellipse((x0, y0), 2*radius, 2*radius, angle=0,
                                                                    fill=False, color=color, lw=1)
                                ax.add_patch(circle_as_ellipse)
                                
                                # Triangle position on the perimeter
                                theta = -np.pi/4 if S3[i,j] > 0 else np.pi/4
                                x_arrow = x0 + radius*np.cos(theta)
                                y_arrow = y0 + radius*np.sin(theta)
                                
                                # Tangent (for a circle it is simply perpendicular to the radius)
                                dx_rot = -np.sin(theta)
                                dy_rot = np.cos(theta)
                                
                                # Triangle angle
                                angle_marker = np.degrees(np.arctan2(dy_rot, dx_rot))
                                if S3[i,j] < 0:
                                    angle_marker += 180
                                
                                t = Affine2D().rotate_deg(angle_marker)
                                
                                # Draw triangle with same color as the circle
                                ax.plot(x_arrow, y_arrow, marker=MarkerStyle(">", "full", t),
                                        color=color, markersize=arrow_scale*30)
    
                                                
                                                
                            # ELLIPTICAL → draw ellipse
                            else:
                                a = arrow_scale/2 # major axis
                                bb = max(min(bb, 1), 0) # ensure valid
                                minor = a * bb
                            
                                # Set color according to handedness
                                color = 'cyan' if S3[i,j] > 0 else 'magenta'
                                
                                # Draw ellipse
                                e = patches.Ellipse((x0, y0), 2*a, 2*minor, angle=np.degrees(angle),
                                                    fill=False, color=color, lw=1)
                                ax.add_patch(e)
                                
                                # Triangle position on the perimeter
                                theta = -np.pi/4 if S3[i,j] > 0 else np.pi/4
                                x_arrow = x0 + a*np.cos(theta)*np.cos(angle) - minor*np.sin(theta)*np.sin(angle)
                                y_arrow = y0 + a*np.cos(theta)*np.sin(angle) + minor*np.sin(theta)*np.cos(angle)
                                
                                # Tangent
                                dx = -a*np.sin(theta)
                                dy = minor*np.cos(theta)
                                dx_rot = dx*np.cos(angle) - dy*np.sin(angle)
                                dy_rot = dx*np.sin(angle) + dy*np.cos(angle)
                                length = np.hypot(dx_rot, dy_rot)
                                dx_rot /= length
                                dy_rot /= length
                                
                                # Triangle angle
                                angle_marker = np.degrees(np.arctan2(dy_rot, dx_rot))
                                if S3[i,j] < 0:
                                    angle_marker += 180
                                
                                t = Affine2D().rotate_deg(angle_marker)
                                
                                # Draw triangle with same color as the ellipse
                                ax.plot(x_arrow, y_arrow, marker=MarkerStyle(">", "full", t),
                                        color=color, markersize=arrow_scale*30)
                            
            
            # Title row
            fig.text(0.5, 0.95, 'Normalized intensities at reference sphere', fontsize=14, ha='center', fontweight='bold')
                
            # Common color bar for intensity
            cbar1 = grid1.cbar_axes[0].colorbar(im)
            cbar1.set_label('Normalized Intensity')
                
        # Phase    
            grid2 = ImageGrid(fig, (2, 1, 2), nrows_ncols=(1, 4), axes_pad=0.4, 
                              cbar_mode='single', cbar_location='right')
            
            # Phase interpolation on the Cartesian grid
            self.Phase_x_inf_cart = griddata(points, self.Phase_x_inf.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=np.nan)
            self.Phase_y_inf_cart = griddata(points, self.Phase_y_inf.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=np.nan)
            self.Phase_z_inf_cart = griddata(points, self.Phase_z_inf.ravel(), (self.X_inc, self.Y_inc), method='cubic', fill_value=np.nan)
            self.Phase_phirho_inf_cart = griddata(points, self.Phase_phirho_inf.ravel(), (self.X_inc, self.Y_inc), method='cubic',fill_value=np.nan)
            
            # Low intensity mask to leave the background black
            mask = S0 < 1e-6 * np.max(S0)
            self.Phase_x_inf_cart[mask] = np.nan
            self.Phase_y_inf_cart[mask] = np.nan
            self.Phase_z_inf_cart[mask] = np.nan
            self.Phase_phirho_inf_cart[mask] = np.nan
            
            # Create colormap that treats NaN as black
            cmap_hsv_black = plt.cm.hsv.copy()
            cmap_hsv_black.set_bad(color='black')
            
            # Create colormap that treats NaN as black
            cmap_hsv_black = plt.cm.hsv.copy()
            cmap_hsv_black.set_bad(color='black')
            
            phase_inf = [(self.Phase_x_inf_cart, cmap_hsv_black, r'$\varphi_x^{\infty}$'),
                         (self.Phase_y_inf_cart, cmap_hsv_black, r'$\varphi_y^{\infty}$'),
                         (self.Phase_z_inf_cart, cmap_hsv_black, r'$\varphi_z^{\infty}$'),
                         (self.Phase_phirho_inf_cart, cmap_hsv_black, r'$\varphi_{\phi}^{\infty}-\varphi_{\rho}^{\infty}$')]
                        

            for ax, (data, cmap, title) in zip(grid2, phase_inf):
                im2 = ax.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi, origin='lower', extent=[-self.x_max/self.f, self.x_max/self.f, -self.x_max/self.f, self.x_max/self.f])
                ax.set_aspect('equal')                
                ax.set_title(title)
                ax.set_xlabel('x/f')
                ax.set_ylabel('y/f')
                ax.axis('equal')
                
                # Set background in black
                ax.set_facecolor('black')
                
            # Title row
            fig.text(0.5, 0.46, 'Phases of E field components at reference sphere', fontsize=14, ha='center', fontweight='bold')
                
            # Common color bar for phase
            cbar2 = grid2.cbar_axes[0].colorbar(im2)
            cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar2.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            cbar2.set_label('Field Phase [rad]')
            
            plt.subplots_adjust(top=0.91, bottom=0.05, hspace=0.3)
            plt.show()


    def calculate_focus(self, L_xy, N_prime_pixel, z, mode):
        
        """
        Calculate the x, y and z components from the E field and the intensity,
        the total intensity and the phases in the z-plane (z = 0.0 is the focal
        plane) using the Richards–Wolf diffraction integral.

        Parameters
        ----------
        
        L_xy : float
            Half-width of the focal plane.
            
        N_prime_pixel : int
            Number of pixels along each lateral axis.
            
        z : float
            Axial coordinate where the field is calculated.
        
        """
        
        
        
        if self.I_inf is None:
            print("You need to calculate the reference sphere first.")
        
        else:
            
            start_time = time.time()
            self.mode = mode
            
            x = np.linspace(-L_xy, L_xy, N_prime_pixel)   # meters
            y = np.linspace(-L_xy, L_xy, N_prime_pixel)   # meters
            self.X, self.Y = np.meshgrid(x, y)
            
            # Convert units x/lambda, y/lambda
            rho = np.sqrt(self.X**2 + self.Y**2)
            phi_2 = np.arctan2(self.Y, self.X)

            # Calculate the differential of the integral
            diff_theta = self.theta_lin[1]-self.theta_lin[0]
            diff_phi = self.phi_lin[1]-self.phi_lin[0]

            # Create an array where we are gonna put the values of the integral
            E_focus = np.zeros((N_prime_pixel, N_prime_pixel, 3), dtype=complex)
            
            
            if (mode == "sum") or (mode == "trapezoidal") or (mode == "simpson"):
                # Richards–Wolf integral (eq. 3.47 // page 60 [17])
                for theta_i in tqdm(range(self.N_theta)):
                    # Things that ONLY have theta dependence
                    theta_value = self.theta_lin[theta_i]
                    sin_th = np.sin(theta_value)
                    cos_th = np.cos(theta_value)
                    expo_1 = np.exp(1j * self.k * z * cos_th)
                    
                    # Calculate de integral with summations
                    if (mode == "sum"):
                        # Things that have theta and/or phi dependence
                        for phi_i in range(self.N_phi):
                            phi_value = self.phi_lin[phi_i]
                            E_inf_vec = self.E_inf[phi_i, theta_i, :]
                            expo_2 = np.exp(1j * self.k * rho * sin_th * np.cos(phi_value - phi_2))
                            E_focus = E_focus + (E_inf_vec[None,None,:] *
                                        expo_1 * expo_2[...,None] *
                                        sin_th * diff_phi *  diff_theta)
                    
                    # Calculate de integral with trapezoids or Simpson
                    elif (mode == "trapezoidal") or (mode == "simpson"):
                        E_inf_theta = self.E_inf[:, theta_i, :]
                        expo_2 = np.exp(1j * self.k * rho[None, :, :] * sin_th * np.cos(self.phi_lin[:, None, None] - phi_2[None, :, :]))
                        integrand_phi = E_inf_theta[:, None, None, :] * expo_2[..., None]
    
                        if (mode == "trapezoidal"):
                            integral_phi = np.trapz(integrand_phi, x=self.phi_lin, axis=0)
                        
                        elif (mode == "simpson"):
                            integral_phi = simpson(integrand_phi, x=self.phi_lin, axis=0)
                        
                        
                        E_focus = E_focus + integral_phi * expo_1 * sin_th * diff_theta
                        
                    
                    
            elif (mode == 'vectorized trapezoidal') or (mode == 'vectorized simpson'):

                # Angles 1D (no meshgrid)
                sin_th = np.sin(self.theta_lin)[None, :, None, None]
                cos_th = np.cos(self.theta_lin)[None, :, None, None]

                # Coordinates in the focal plane
                rho_b  = rho[None, None, :, :]
                phi2_b = phi_2[None, None, :, :]

                # Phi 1D reshaped
                phi_vals = self.phi_lin[:, None, None, None]

                
                expo_1 = np.exp(1j * self.k * z * cos_th)
                expo_2 = np.exp(1j * self.k * rho_b * sin_th * np.cos(phi_vals - phi2_b))

                # integrand (Nφ, Nθ, Nx, Ny, 3)
                integrand = (self.E_inf[:, :, None, None, :] * expo_1[..., None] *
                              expo_2[..., None] * sin_th[..., None])

                
                if (mode == 'vectorized trapezoidal'):
                    integral1 = np.trapz(integrand, x=self.phi_lin, axis=0)
                    E_focus = np.trapz(integral1, x=self.theta_lin, axis=0)

                elif (mode == 'vectorized simpson'):
                    integral1 = simpson(integrand, x=self.phi_lin, axis=0)
                    E_focus = simpson(integral1, x=self.theta_lin, axis=0)
                    
                    
            elif mode == 'fft':
                Nx = N_prime_pixel
                Ny = N_prime_pixel
            
                # Spatial sample intervals
                dx = x[1] - x[0]
                dy = y[1] - y[0]
            
                # Nyquist frequency
                kx_fft = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)   # length Nx
                ky_fft = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)   # length Ny
                KX, KY = np.meshgrid(kx_fft, ky_fft, indexing='xy')  # shape (Ny, Nx)
            
                # Valors for interpolate
                kx_pts = (self.k * np.sin(self.THETA) * np.cos(self.PHI)).ravel()
                ky_pts = (self.k * np.sin(self.THETA) * np.sin(self.PHI)).ravel()
            
                # Weights and first exponencial
                weights = np.sin(self.THETA) * diff_theta * diff_phi                 # (N_phi, N_theta)
                expo_1 = np.exp(1j * self.k * z * np.cos(self.THETA))
            
                E_weighted = self.E_inf * weights[..., None] * expo_1[..., None]      # (N_phi, N_theta, 3)
            
                # Interpolate every component in the regular grid (kx, ky)
                points = np.column_stack((kx_pts, ky_pts))                           # shape (N_phi*N_theta, 2)
                Ek_grid = np.zeros((Ny, Nx, 3), dtype=complex)                       # (ky index rows, kx index cols)
               
                for comp in range(3):
                    vals = E_weighted[..., comp].ravel()
                    Ek_grid[..., comp] = griddata(points, vals, (KX, KY), method='cubic', fill_value=0.0)
            
                # Calculate the integrate with FFT    
                E_focus = np.zeros_like(Ek_grid, dtype=complex)
                for comp in range(3):
                    E_ifft_comp = np.fft.fftshift(np.fft.ifft2(Ek_grid[..., comp]))
                    
                    # Factor (1/(2π)^2)
                    E_focus[..., comp] = E_ifft_comp / ((2*np.pi)**2)
            
                
            # Multiply by the starting factor of the integral
            self.E_focus = E_focus * ((1j * self.k * self.f * np.exp(-1j * self.k * self.f)) / (2*np.pi))
            
            # Decompose the real part of the E field at the focus
            # into the X, Y and Z coordinates
            self.Ex_focus = np.real(self.E_focus[..., 0])
            self.Ey_focus = np.real(self.E_focus[..., 1])
            self.Ez_focus = np.real(self.E_focus[..., 2])
            
            # Phases at the reference sphere
            self.Phase_x_focus = np.angle(self.E_focus[..., 0])
            self.Phase_y_focus = np.angle(self.E_focus[..., 1])
            self.Phase_z_focus = np.angle(self.E_focus[..., 2])

            # Total intensity
            self.I_focus = np.linalg.norm(self.E_focus, axis=-1)**2
            
            # Intensity of the individual field components (fig. 3.10 (c), (d) and (e) // page 65 [22])
            self.Ix_focus = np.abs(self.E_focus[..., 0])**2
            self.Iy_focus = np.abs(self.E_focus[..., 1])**2
            self.Iz_focus = np.abs(self.E_focus[..., 2])**2
            
            print("--- {} seconds with {} method ---".format(time.time() - start_time, mode))

          
    def show_focus(self, zoom_lim = 2.0):
        
        """
        
        Display the x, y and z components of the intensity, the total intensity
        and the phases in the chosen z-plane. 
        
        """
        
        if self.I_focus is None:
            print("You need to calculate the focus first.")
        
        else:
            
            # Compute polarization orientation angle from Stokes parameters
            Ex_focus = self.E_focus[..., 0]
            Ey_focus = self.E_focus[..., 1]
            
            # Stokes parameters
            S0 = np.abs(Ex_focus)**2 + np.abs(Ey_focus)**2
            S1 = np.abs(Ex_focus)**2 - np.abs(Ey_focus)**2
            S2 = 2 * np.real(Ex_focus * np.conj(Ey_focus))
            S3 = -2 * np.imag(Ex_focus * np.conj(Ey_focus))
            
            # Orientation angle ψ
            psi = 0.5 * np.arctan2(S2, S1)
               
            # Ellipticity angle χ
            chi = 0.5 * np.arcsin(np.clip(S3 / (S0 + 1e-20), -1, 1))
            
            
            # Ellipse axis ratio b = |tan χ|
            b = np.abs(np.tan(chi)) # 0=lineal, 1=circular
                        
            zeros_array1 = np.zeros_like(self.I_focus, dtype=float)
            zeros_array2 = np.zeros_like(self.I_focus, dtype=float)
            zeros_array3 = np.zeros_like(self.I_focus, dtype=float)
            zeros_array4 = np.zeros_like(self.I_focus, dtype=float)
            zeros_array5 = np.zeros_like(self.I_focus, dtype=float)
            

            # We dicide the size of the window dbecause it is not represent correct due to Matplotlib
            fig = plt.figure(figsize=(12, 8), dpi=120)
            plt.rcParams['figure.autolayout'] = False
            fig.suptitle('Focal with mode: {}'.format(self.mode), fontsize=18, fontweight='bold', y = 1.00)
            
            
        # Intensity    
            grid1 = ImageGrid(fig, (2, 1, 1), nrows_ncols=(1, 4), axes_pad=0.4,
                              cbar_mode='single', cbar_location='right')
            
            
            intensity_focus = [(np.divide(self.I_focus, np.max(self.I_focus), out=zeros_array1, where=(np.max(self.I_focus) > 1e-20)), 'hot', r'$|E_{focus}(x,y,z)|^2$'),
                                (np.divide(self.Ix_focus, np.max(self.Ix_focus), out=zeros_array2, where=(np.max(self.Ix_focus) > 1e-20)), 'hot', r'$|E_x^{focus}|^2$'),
                                (np.divide(self.Iy_focus, np.max(self.Iy_focus), out=zeros_array3, where=(np.max(self.Iy_focus) > 1e-20)), 'hot', r'$|E_y^{focus}|^2$'),
                                (np.divide(self.Iz_focus, np.max(self.Iz_focus), out=zeros_array4, where=(np.max(self.Iz_focus) > 1e-20)), 'hot', r'$|E_z^{focus}|^2$')]
            
            # Steps arrow
            x_phys = self.X[0, :] / self.lamb
            dx = x_phys[1] - x_phys[0]
            n_arrows_axis = 8
            Npix_zoom = int((2 * zoom_lim) / dx)
            
            arrow_step = max(1, int(Npix_zoom / n_arrows_axis))
            
            # Physical axes matching imshow extent (IMPORTANT)
            x_axis = np.linspace(self.X.min()/self.lamb,
                                 self.X.max()/self.lamb,
                                 self.X.shape[1])
            
            y_axis = np.linspace(self.Y.min()/self.lamb,
                                 self.Y.max()/self.lamb,
                                 self.X.shape[0])
            
            for idx, (ax, (data, cmap, title)) in enumerate(zip(grid1, intensity_focus)):
                im = ax.imshow(data, cmap=cmap, origin='lower', vmin=0, vmax=1, 
                                extent=[self.X.min()/self.lamb, self.X.max()/self.lamb, self.Y.min()/self.lamb, self.Y.max()/self.lamb])
                ax.set_aspect('equal')
                ax.set_xlim(-zoom_lim, zoom_lim)
                ax.set_ylim(-zoom_lim, zoom_lim)
                ax.set_title(title)
                ax.set_xlabel(r'x/$\lambda$')
                ax.set_ylabel(r'y/$\lambda$')
                
                # Set background in black
                ax.set_facecolor('black')
                
                # Mask for low intensities (same intensity as the plotted image)
                self.I_norm = np.divide(self.I_focus, np.max(self.I_focus), out=zeros_array5, where=(np.max(self.I_focus) > 1e-20))
                mask_pol = self.I_norm < 1e-4
                
                # Zoom
                x_range = 2 * zoom_lim
                
                arrow_scale = 0.08 * x_range  # arrow length = 8% of range
                head_size = 0.05 * x_range   # size of arrow = 5% of range

                
                # We only draw the polarization indicators in the first subplot (total intensity)
                if idx == 0:
                    
                    # Visible physical coordinates
                    x_vals = np.linspace(-zoom_lim, zoom_lim, n_arrows_axis)
                    y_vals = np.linspace(-zoom_lim, zoom_lim, n_arrows_axis)
                    
                    for x0 in x_vals:
                        for y0 in y_vals:
                    
                            # Indices closest to the mesh
                            j = np.argmin(np.abs(x_axis - x0))
                            i = np.argmin(np.abs(y_axis - y0))
                        
                        
                            if (S0[i, j] < 1e-6 * np.max(S0)):
                                continue # Skip very low intensity
                    
                            x0 = x_axis[j]   # j → x
                            y0 = y_axis[i]   # i → y
                            angle = psi[i, j]
                            bb = b[i, j]
                            
                            if abs(x0) > zoom_lim or abs(y0) > zoom_lim:
                                continue
                        
                            # maximum symbol radius
                            R = arrow_scale / 2
                            
                            # DO NOT draw if the symbol does not fit completely
                            if (x0 - R < -zoom_lim or x0 + R > zoom_lim or
                                y0 - R < -zoom_lim or y0 + R > zoom_lim):
                                continue
                        
                            # LINEAR polarization → draw line
                            if bb < 0.05:
                                dx = arrow_scale * np.cos(angle)
                                dy = arrow_scale * np.sin(angle)
                                start_x = x0 - dx/2
                                start_y = y0 - dy/2
                                ax.arrow(start_x, start_y, dx, dy, head_width=head_size, head_length=head_size,
                                         fc='green', ec='green', transform=ax.transData, clip_on=True)
                                
                            
                            # CIRCULAR → draw circle as an ellipse
                            elif bb > 0.95:
                                radius = arrow_scale/2
                            
                                # We assign color according to clockwise/counterclockwise rotation
                                color = 'cyan' if S3[i,j] > 0 else 'magenta'
                                
                                # Treat as an ellipse with equal axes
                                circle_as_ellipse = patches.Ellipse((x0, y0), 2*radius, 2*radius, angle=0,
                                                                    fill=False, color=color, lw=1, transform=ax.transData, clip_on=True)
                                ax.add_patch(circle_as_ellipse)
                                
                                # Triangle position on the perimeter
                                theta = -np.pi/4 if S3[i,j] > 0 else np.pi/4
                                x_arrow = x0 + radius*np.cos(theta)
                                y_arrow = y0 + radius*np.sin(theta)
                                
                                # Tangent (for a circle it is simply perpendicular to the radius)
                                dx_rot = -np.sin(theta)
                                dy_rot = np.cos(theta)
                                
                                # Triangle angle
                                angle_marker = np.degrees(np.arctan2(dy_rot, dx_rot))
                                if S3[i,j] < 0:
                                    angle_marker += 180
                                
                                t = Affine2D().rotate_deg(angle_marker)
                                
                                # Draw triangle with same color as the circle
                                ax.plot(x_arrow, y_arrow, marker=MarkerStyle(">", "full", t),
                                        color=color, markersize=arrow_scale*5, transform=ax.transData, clip_on=True)
    
                                                
                                                
                            # ELLIPTICAL → draw ellipse
                            else:
                                a = arrow_scale/2 # major axis
                                bb = max(min(bb, 1), 0) # ensure valid
                                minor = a * bb
                            
                                # Set color according to handedness
                                color = 'cyan' if S3[i,j] > 0 else 'magenta'
                                
                                # Draw ellipse
                                e = patches.Ellipse((x0, y0), 2*a, 2*minor, angle=np.degrees(angle),
                                                    fill=False, color=color, lw=1, transform=ax.transData, clip_on=True)
                                ax.add_patch(e)
                                
                                # Triangle position on the perimeter
                                theta = -np.pi/4 if S3[i,j] > 0 else np.pi/4
                                x_arrow = x0 + a*np.cos(theta)*np.cos(angle) - minor*np.sin(theta)*np.sin(angle)
                                y_arrow = y0 + a*np.cos(theta)*np.sin(angle) + minor*np.sin(theta)*np.cos(angle)
                                
                                # Tangent
                                dx = -a*np.sin(theta)
                                dy = minor*np.cos(theta)
                                dx_rot = dx*np.cos(angle) - dy*np.sin(angle)
                                dy_rot = dx*np.sin(angle) + dy*np.cos(angle)
                                length = np.hypot(dx_rot, dy_rot)
                                dx_rot /= length
                                dy_rot /= length
                                
                                # Triangle angle
                                angle_marker = np.degrees(np.arctan2(dy_rot, dx_rot))
                                if S3[i,j] < 0:
                                    angle_marker += 180
                                
                                t = Affine2D().rotate_deg(angle_marker)
                                
                                # Draw triangle with same color as the ellipse
                                ax.plot(x_arrow, y_arrow, marker=MarkerStyle(">", "full", t),
                                        color=color, markersize=arrow_scale*5, transform=ax.transData, clip_on=True)
            
            # Title row
            fig.text(0.5, 0.95, 'Normalized intensities at z-plane', fontsize=14, ha='center', fontweight='bold')
                
            # Common color bar for intensity
            cbar1 = grid1.cbar_axes[0].colorbar(im)
            cbar1.set_label('Normalized Intensity')
            
            
                
        # Phase    
            grid2 = ImageGrid(fig, (2, 1, 2), nrows_ncols=(1, 3), axes_pad=0.4, 
                              cbar_mode='single', cbar_location='right')
            
            phase_focus = [(self.Phase_x_focus, 'hsv', r'$\varphi_x^{focus}$'),
                           (self.Phase_y_focus, 'hsv', r'$\varphi_y^{focus}$'),
                           (self.Phase_z_focus, 'hsv', r'$\varphi_z^{focus}$')]
                        

            for ax, (data, cmap, title) in zip(grid2, phase_focus):
                im2 = ax.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi, origin='lower', 
                                extent=[self.X.min()/self.lamb, self.X.max()/self.lamb, self.Y.min()/self.lamb, self.Y.max()/self.lamb])
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(-zoom_lim, zoom_lim)
                ax.set_ylim(-zoom_lim, zoom_lim)
                ax.set_title(title)
                ax.set_xlabel(r'x/$\lambda$')
                ax.set_ylabel(r'y/$\lambda$')
                
                
            # Title row
            fig.text(0.5, 0.46, 'Phases of E field components at z-plane', fontsize=14, ha='center', fontweight='bold')
                
            # Common color bar for phase
            cbar2 = grid2.cbar_axes[0].colorbar(im2)
            cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar2.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            cbar2.set_label('Field Phase [rad]')
            
            plt.subplots_adjust(top=0.91, bottom=0.05, hspace=0.3)
            plt.show()

    
    def calculate_Stokes(self):
        
        """
        Calculate the Stokes vector generalization to 3D components
        
        """
        if self.I_focus is None:
            print("You need to calculate the focus first.")
        
        else:
        
            # Calculate the real and imaginary components
            Er_focus = np.real(self.E_focus)
            Ei_focus = np.imag(self.E_focus)
            Er_focus_2 = np.sum(Er_focus * Er_focus, axis=-1)        # ||Er||^2
            Ei_focus_2 = np.sum(Ei_focus * Ei_focus, axis=-1)        # ||Ei||^2
            dot_r_i_focus = np.sum(Er_focus * Ei_focus, axis=-1)     # Er · Ei
            
            # Calculate alpha (eq. (3) article)
            self.alpha_focus = (1.0/2.0) * np.arctan2( 2.0 *  dot_r_i_focus, Er_focus_2 - Ei_focus_2)
            
            # Calculate N (eq. (6) article)
            self.N_focus = np.cross(Er_focus, Ei_focus, axis=-1)
            
            # Calculate de 3D Stokes components (eq. (9) article)
            self.S0_focus = Er_focus_2 + Ei_focus_2
            cos2a = np.cos(2.0 * self.alpha_focus)
            self.S1_focus = np.zeros_like(self.S0_focus)
            mask = np.abs(cos2a) > 1e-3
            self.S1_focus[mask] = (Er_focus_2[mask] - Ei_focus_2[mask]) / cos2a[mask]
            #self.S1_focus = (Er_focus_2 - Ei_focus_2) / (np.cos(2.0 * self.alpha_focus) + 1e-12)
            self.S2_focus = 0.0                                                   # Always
            self.S3_focus = 2.0 * np.linalg.norm(self.N_focus, axis=-1)
            #self.S3_focus = np.linalg.norm(self.N_focus, axis=-1)
            
            # self.P_focus = np.cos(self.alpha_focus)[...,None]*Er_focus + np.sin(self.alpha_focus)[...,None]*Ei_focus
            # self.Q_focus = -np.sin(self.alpha_focus)[...,None]*Er_focus + np.cos(self.alpha_focus)[...,None]*Ei_focus
        
            # Normalitzation
            # self.S0_focus = self.S0_focus / (np.max(self.S0_focus) + 1e-12)
            # self.S1_focus = self.S1_focus / (np.max(self.S0_focus) + 1e-12)
            # self.S3_focus = self.S3_focus / (np.max(self.S0_focus) + 1e-12)
            S0_max = np.max(self.S0_focus) + 1e-12
            self.S0_focus /= S0_max
            self.S1_focus /= S0_max
            self.S3_focus /= S0_max
            
    
    def show_Stokes(self, zoom_lim = 2.0):
        
        if self.N_focus is None:
            print("You need to calculate the Stokes vector in the focus first.")
        
        else:
        
        # 2D Stokes representation
            # Components N Stokes vector
            Nx_focus = self.N_focus[...,0]
            Ny_focus = self.N_focus[...,1]
            Nz_focus = self.N_focus[...,2]
            
            # Normalitzation
            Nmod = np.sqrt(Nx_focus**2 + Ny_focus**2)
            Nx_plot = Nx_focus / (np.max(Nmod) + 1e-12)
            Ny_plot = Ny_focus / (np.max(Nmod) + 1e-12)
            
            # Figure 1: N
            fig, axs = plt.subplots(2, 2)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            fig.suptitle("Stokes 2D at the focus")
            ax = axs[0, 0]
            im0 = ax.imshow( (self.I_focus / (np.max(self.I_focus) + 1e-12)), origin='lower', vmin=0, vmax=1,
                            extent=[self.X.min()/self.lamb, self.X.max()/self.lamb, self.Y.min()/self.lamb, self.Y.max()/self.lamb],
                            cmap='hot')

            skip = 5
            
            ax.quiver(self.X[::skip, ::skip]/self.lamb, self.Y[::skip, ::skip]/self.lamb,
                      Nx_plot[::skip, ::skip], Ny_plot[::skip, ::skip],
                      color='cyan', pivot='middle', scale=20, width=0.003)
            
            ax.set_title(r'$\mathbf{N}_\perp$')
            ax.set_aspect('equal')
            ax.set_xlabel(r'x/$\lambda$')
            ax.set_ylabel(r'y/$\lambda$')
            
            ax.set_xlim(-zoom_lim, zoom_lim)
            ax.set_ylim(-zoom_lim, zoom_lim)
            
            fig.colorbar(im0, ax=ax)
            
            # Figure 2: S0
            ax = axs[0, 1]
            im1 = ax.imshow(self.S0_focus, origin='lower', vmin=-1, vmax=1,
                            extent=[self.X.min()/self.lamb, self.X.max()/self.lamb, self.Y.min()/self.lamb, self.Y.max()/self.lamb],
                            cmap='seismic')

            ax.set_title(r'$\widetilde{S}_0$')
            ax.set_aspect('equal')
            ax.set_xlabel(r'x/$\lambda$')
            ax.set_ylabel(r'y/$\lambda$')
            
            ax.set_xlim(-zoom_lim, zoom_lim)
            ax.set_ylim(-zoom_lim, zoom_lim)
            
            fig.colorbar(im1, ax=ax)
            
            # Figure 3: S1
            ax = axs[1, 0]
            im2 = ax.imshow(self.S1_focus, origin='lower', vmin=-1, vmax=1,
                            extent=[self.X.min()/self.lamb, self.X.max()/self.lamb, self.Y.min()/self.lamb, self.Y.max()/self.lamb],
                            cmap='seismic')
            
            ax.set_title(r'$\widetilde{S}_1$')
            ax.set_aspect('equal')
            ax.set_xlabel(r'x/$\lambda$')
            ax.set_ylabel(r'y/$\lambda$')
            
            ax.set_xlim(-zoom_lim, zoom_lim)
            ax.set_ylim(-zoom_lim, zoom_lim)
            
            fig.colorbar(im2, ax=ax)
            
            # Figure 4: S3
            ax = axs[1, 1]
            im2 = ax.imshow(self.S3_focus, origin='lower', vmin=-1, vmax=1,
                            extent=[self.X.min()/self.lamb, self.X.max()/self.lamb, self.Y.min()/self.lamb, self.Y.max()/self.lamb],
                            cmap='seismic')
            
            ax.set_title(r'$\widetilde{S}_3$')
            ax.set_aspect('equal')
            ax.set_xlabel(r'x/$\lambda$')
            ax.set_ylabel(r'y/$\lambda$')
            
            ax.set_xlim(-zoom_lim, zoom_lim)
            ax.set_ylim(-zoom_lim, zoom_lim)
            
            fig.colorbar(im2, ax=ax)
            plt.show()

            
        # 3D Stokes representation
            # Normalitzation
            # S0_max = np.max(self.S0_focus)
            # Nx_plot = Nx_focus / (S0_max + 1e-12)
            # Ny_plot = Ny_focus / (S0_max + 1e-12)
            # Nz_plot = Nz_focus / (S0_max + 1e-12)
            Nmod = np.sqrt(Nx_focus**2 + Ny_focus**2 + Nz_focus**2)
            Nx_plot = Nx_focus / (np.max(Nmod) + 1e-12)
            Ny_plot = Ny_focus / (np.max(Nmod) + 1e-12)
            Nz_plot = Nz_focus / (np.max(Nmod) + 1e-12)
            
            # Downsampling
            skip = 5
            X_ds = self.X[::skip, ::skip]/self.lamb
            Y_ds = self.Y[::skip, ::skip]/self.lamb
            Z_ds = np.zeros_like(X_ds)
            
            U_ds = Nx_plot[::skip, ::skip]
            V_ds = Ny_plot[::skip, ::skip]
            W_ds = Nz_plot[::skip, ::skip]
            
            # Flatten (REQUIRED for 3D)
            x = X_ds.ravel()
            y = Y_ds.ravel()
            z = Z_ds.ravel()
            u = U_ds.ravel()
            v = V_ds.ravel()
            w = W_ds.ravel()
            
            # Figure 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.quiver(x, y, z,
                      u, v, w,
                      length=0.25, arrow_length_ratio=0.3, linewidth=1.5, color='orange')
            
            ax.set_xlim(X_ds.min(), X_ds.max())
            ax.set_ylim(Y_ds.min(), Y_ds.max())
            ax.set_zlim(-0.3, 0.3)
            ax.set_box_aspect((1, 1, 0.6))
            
            # Initial view
            ax.view_init(elev=25, azim=45)
            
            ax.set_xlabel(r'x/$\lambda$')
            ax.set_ylabel(r'y/$\lambda$')
            ax.set_zlabel(r'z/$\lambda$')
            ax.set_title(r'$\mathbf{N}$ at focus (3D)')
            plt.show()
        
#%%

""" Create the object """

lambd = 500.0 # nm
f = 5.0 # mm
NA = 0.95

f_m = f * 1e-3      # if "f" is in mm → meters
lambd_m = lambd * 1e-9  # if "lamb" is in nm → meters

#f_m = 1
#lambd_m = 1

rw = RichardsWolf(NA, 1, 1, 100, 100, f_m, 1, 1, lambd_m)

#%%

""" Decide the polaritzation and the profile of the incident light """

# E_0 : float
# Amplitude of the incident electric field.
E_0 = 1
     
# omega_0 : float
# Beam waist of the incident beam.
omega_0 = 1e-3 # m
#omega_0 = 1

# Delta is the relative phase between E_x and E_y
delta = np.full((rw.N_theta, rw.N_phi), 0)

# Alpha is the orientation of the ellipse
alpha = rw.PHI
#alpha = np.full((rw.N_theta, rw.N_phi), np.pi/4)

J_x = np.cos(alpha)
J_y = np.exp(1j * delta) * np.sin(alpha)

# J_x = rw.X_inc
# J_y = rw.Y_inc

#J_x = 1
#J_y = 0


# Incident field in spherical coordinates (eq. 3.52 // page 61 [18])
# E_inc_profile = E_0 * np.exp(-rw.f**2 * (np.sin(rw.THETA))**2/omega_0**2)
# E_inc_profile = (2*E_0*rw.f/omega_0) * np.sin(rw.THETA) * np.cos(rw.PHI) * np.exp(-rw.f**2 * (np.sin(rw.THETA))**2/omega_0**2)
# E_inc_profile = (2*E_0*rw.f/omega_0) * np.sin(rw.THETA) * np.sin(rw.PHI) * np.exp(-rw.f**2 * (np.sin(rw.THETA))**2/omega_0**2)
# E_inc_profile = E_0


rho = rw.f * np.sin(rw.THETA)  # radial coordinate at the entrance
sigma = 1  # beam width, adjustable
E_inc_profile = rho * np.exp(-rho**2 / sigma**2) * np.exp(1j * alpha)

rw.set_beam(J_x, J_y, E_inc_profile)
rw.show_incident_beam()

#%%

""" Surface of the reference sphere """

rw.calculate_reference_sphere()
rw.show_reference_sphere()

#%%

""" Focal plane sum"""

N_prime_pixel = 100
#L_xy = 4.0 * rw.lamb      # Lambda units
# L_z = 0*rw.lamb
# N_z = 1*rw.lamb

# In this version "z" is specificate here
# An alternative idea is to make a linspace and calculate de integral for differents "z"
z = 0.0

# rw.calculate_focus(L_xy, N_prime_pixel, z, "sum")
# zoom_lim = 1.0 
# rw.show_focus(zoom_lim)

# zoom_lim = 3.0 
# rw.show_focus(zoom_lim)

L_xy = 3.0 * rw.lamb
rw.calculate_focus(L_xy, N_prime_pixel, z, "sum")
zoom_lim = 3.0 
rw.show_focus(zoom_lim)

rw.calculate_Stokes()
rw.show_Stokes(zoom_lim)


#%%

""" Focal plane: trapezoidal"""

# rw.calculate_focus(L_xy, N_prime_pixel, z, "trapezoidal")
# rw.show_focus()

#%%

""" Focal plane: Simpson"""

# rw.calculate_focus(L_xy, N_prime_pixel, z, "simpson")
# rw.show_focus()

#%%

""" Focal plane: trapezoidal vectorized, broadcasted"""

# rw.calculate_focus(L_xy, N_prime_pixel, z, "vectorized trapezoidal")
# rw.show_focus()

#%%

""" Focal plane: Simpson vectorized, broadcasted"""

# rw.calculate_focus(L_xy, N_prime_pixel, z, "vectorized simpson")
# rw.show_focus()

#%%

# Apartat "e"

""" Focal plane: FFT"""

# rw.calculate_focus(L_xy, N_prime_pixel, z, "fft")
# rw.show_focus(zoom_lim)


#%%

# help(RichardsWolf)

# help(RichardsWolf.set_beam)
# help(RichardsWolf.show_incident_beam)
# help(RichardsWolf.calculate_reference_sphere)
# help(RichardsWolf.show_reference_sphere)
# help(RichardsWolf.calculate_focus)
# help(RichardsWolf.show_focus)

