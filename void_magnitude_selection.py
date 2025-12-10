#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:39:52 2025

@author: noahpierce
"""

import numpy as np
from jax import numpy as jnp
from quadax import cumulative_simpson
from jax.scipy.stats.norm import logcdf as norm_logcdf
from jax.scipy.special import gamma
import pandas as pd
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import gammaincinv
import sys

import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(0)

c = 2.998 * 10**5

omega_m = 0.3

# redshift from comoving distance
def dc_to_redshift(D_c, c, omega_m):
    
    """
    Takes inputs: comoving distance (in Mpc/h), c, and matter density param
    Outputs: redshift
    """
    
    D = (D_c * 100) / c
    return (1 - jnp.sqrt(1 - 3 * omega_m * D)) / (1.5 * omega_m)


def dc2mu(dc):
    """Convert distance in Mpc to distance modulus."""
    z = dc_to_redshift(dc, c, omega_m)
    return 5 * jnp.log10(dc * (1 + z)) + 25


def simpson2d(f_val, x_grid, y_grid):
    """Evaluate a 2D integral using Simpson's rule."""
    inner = cumulative_simpson(f_val, x=y_grid, axis=1, initial=0.0)
    outer = cumulative_simpson(inner, x=x_grid, axis=0, initial=0.0)
    return outer[-1, -1]


# def log_pdf_LF(M, alpha, M_star, phi_star):
#     """Simple Gaussian-like luminosity function. Replace with Schechter or other."""
#     # return -0.5 * ((M - M_star) / width) ** 2 - jnp.log(width * jnp.sqrt(2 * jnp.pi))
#     # M = 10**log_M
#     # norm = phi_star * gamma(1 - alpha) * M_star
#     # log_pdf = jnp.log(phi_star) - alpha * jnp.log(M / M_star) - (M / M_star) - jnp.log(norm)
#     # return log_pdf + jnp.log(jnp.log(10) * M / 2.5)
#     return jnp.log((0.4*jnp.log(10)*phi_star)) + (alpha+1) * jnp.log(10 ** (0.4*(M_star-M))) - 10**(0.4*(M_star-M))

def log_pdf_LF(M, alpha, M_star, M_abs_Sun):
    """Simple Gaussian-like luminosity function. Replace with Schechter or other."""
    norm = gamma(1 - alpha) * jnp.exp(-0.4*jnp.log(10)*(M_star - M_abs_Sun))
    Ln_L_by_Lstar = -0.4*jnp.log(10)*(M - M_star)
    log_pdf_L = -alpha * Ln_L_by_Lstar - jnp.exp(Ln_L_by_Lstar) - jnp.log(norm) #Wrt L
    log_pdf_M = log_pdf_L + jnp.log(0.4*jnp.log(10)) - 0.4*jnp.log(10)*(M - M_abs_Sun) #Wrt M
    return log_pdf_M
    # return jnp.log((0.4*jnp.log(10)*phi_star)) + (alpha+1) * jnp.log(10 ** (0.4*(M_star-M))) - 10**(0.4*(M_star-M))




def log_integrand_p_det(M, dc, M_star, alpha, psi, mlim, m_err, M_abs_Sun):
    """Logarithmic integrand for the detection probability."""
    
    # L = jnp.exp(L_tilde)
    
   # return norm_logcdf((mlim - (dc2mu(dc) + (-6 - 2.5*jnp.log10(L)))) / sigma_m) + 2 * jnp.log(dc) + jnp.log(1 + psi*dc) + log_pdf_LF(L, alpha, L_star) - jnp.log(r_max**3.0*(1.0/3.0 + 0.25*psi))
    app_mag = M + dc2mu(dc)
    x = (mlim - app_mag) / m_err
    
    # mask = x < 0
    
    #Radial distribution.
    norm = jnp.log(dc_max**3.0*(1.0/3.0 + 0.25*psi))
    ln_prior_dc = 2 * jnp.log(dc) + jnp.log(1 + psi*dc/dc_max) - norm
    
    # output = ln_prior_dc + log_pdf_LF(M, alpha, M_star, M_abs_Sun)
    # updated = jnp.where(mask, -jnp.inf, output)
    
    return norm_logcdf(x) + ln_prior_dc + log_pdf_LF(M, alpha, M_star, M_abs_Sun)
    # return updated, mask
    # return ln_prior_dc + log_pdf_LF(M, alpha, M_star, M_abs_Sun)


L_star = 10**6
alpha = 0.7
N_tot = 750000
phi_star = abs(N_tot/(gamma(1 - alpha) * L_star))
psi = 0.8
# width = 0.5
M_abs_Sun = -6.0
M_star = M_abs_Sun - 2.5*jnp.log10(L_star)
M_min = M_star - 25
M_max = M_star + 25
# # M_min = -30
# # M_max = -10
M_grid = jnp.linspace(M_min, M_max, 1001)
L_grid = 10**-((M_grid+6)/2.5)

schechter_vals = jnp.exp(log_pdf_LF(M_grid, alpha, M_star, M_abs_Sun))

schechter_integral = integrate.simpson(schechter_vals, M_grid)
print('Simpson integral for M: ', schechter_integral)
print('Manual sum for M: ', sum(schechter_vals * (M_grid[1] - M_grid[0])))


# L_min = 1e-5
# L_max = 10**8
# L_grid = jnp.linspace(L_min, L_max, 251)

L_tilde_grid = jnp.log(L_grid)

dc_min = 1e-5
dc_max = 2000

dc_grid = jnp.linspace(dc_min, dc_max, 1001)

mlim = 18
m_err = 0.05

def ln_prior_dc(dc, dc_max, psi):
    
    norm = jnp.log(dc_max**3.0*(1.0/3.0 + 0.25*psi))
    ln_prior_dc = 2 * jnp.log(dc) + jnp.log(1 + psi*dc/dc_max) - norm
    
    return ln_prior_dc

dc_prior_vals = jnp.exp(ln_prior_dc(dc_grid, dc_max, psi))

dc_integral = integrate.simpson(dc_prior_vals, dc_grid)
print('Simpson integral for dc: ', dc_integral)
print('Manual sum for dc: ', sum(dc_prior_vals * (dc_grid[1] - dc_grid[0])))


X, Y = jnp.meshgrid(M_grid, dc_grid, indexing='ij')
log_integrand = log_integrand_p_det(X, Y, M_star, alpha, psi, mlim, m_err, M_abs_Sun)


# This is p(S = 1 | Lambda) from the Overleaf notation.
p_det = simpson2d(jnp.exp(log_integrand), M_grid, dc_grid)

print('Analytically expected remaining fraction:', p_det)

manual_p_det = sum(sum(jnp.exp(log_integrand))) * (M_grid[1] - M_grid[0]) * (dc_grid[1] - dc_grid[0])

print('Manual integration of acceptance fraction: ', manual_p_det)



plt.figure()
plt.pcolormesh(M_grid, dc_grid, jnp.exp(log_integrand.T), shading='auto')
plt.colorbar(label='Probability Density')
plt.xlabel('Absolute Magnitude M')
plt.ylabel('Comoving Distance (Mpc/h)')
plt.title('Detection Probability Density')

# plt.savefig("selection_integral.png", transparent=True, dpi=300)
plt.show()




############

sys.exit()

############


#optimisation checks

# varying alpha
alpha_vals = np.linspace(0.5, 0.9, 100)
alpha_likelihoods = []
for i in alpha_vals:
    log_integrand_alpha = log_integrand_p_det(X, Y, M_star, i, psi, mlim, m_err, M_abs_Sun)
    p_det_alpha = simpson2d(jnp.exp(log_integrand_alpha), M_grid, dc_grid)
    alpha_likelihoods.append(p_det_alpha)
    # print('alpha = ', i, 'p_det = ', p_det_alpha)
    
plt.plot(alpha_vals, alpha_likelihoods)
plt.title('How p_det varies with alpha')
plt.xlabel('Alpha')
plt.ylabel('Detection probability')
plt.show()


# varying M_star
M_star_vals = np.linspace(-16, -28, 100)
M_star_likelihoods = []
for i in M_star_vals:
    log_integrand_M_star = log_integrand_p_det(X, Y, i, alpha, psi, mlim, m_err, M_abs_Sun)
    p_det_M_star = simpson2d(jnp.exp(log_integrand_M_star), M_grid, dc_grid)
    M_star_likelihoods.append(p_det_M_star)
    # print('M_star = ', i, 'p_det = ', p_det_M_star)
    
plt.plot(M_star_vals, M_star_likelihoods)
plt.title('How p_det varies with M_star')
plt.xlabel('M_star')
plt.ylabel('Detection probability')
plt.show()


# varying psi
psi_vals = np.linspace(0.4, 1.2, 100)
psi_likelihoods = []
for i in psi_vals:
    log_integrand_psi = log_integrand_p_det(X, Y, M_star, alpha, i, mlim, m_err, M_abs_Sun)
    p_det_psi = simpson2d(jnp.exp(log_integrand_psi), M_grid, dc_grid)
    psi_likelihoods.append(p_det_psi)
    # print('psi = ', i, 'p_det = ', p_det_psi)
    
plt.plot(psi_vals, psi_likelihoods)
plt.title('How p_det varies with psi')
plt.xlabel('Psi')
plt.ylabel('Detection probability')
plt.show()



# # varying grid spacing
# grid_pts = np.linspace(5, 400, 20)
# likelihood = []
# for i in grid_pts:
#     M_grid = jnp.linspace(M_min, M_max, int(i))
#     dc_grid = jnp.linspace(dc_min, dc_max, int(i))
    
#     X, Y = jnp.meshgrid(M_grid, dc_grid, indexing='ij')
#     log_integrand_grid = log_integrand_p_det(X, Y, M_star, alpha, psi, mlim, m_err, M_abs_Sun)
#     p_det_grid = simpson2d(jnp.exp(log_integrand_grid), M_grid, dc_grid)
#     likelihood.append(p_det_grid)
    
# plt.plot(grid_pts, likelihood)
# plt.title('How p_det varies with no. grid points')
# plt.xlabel('No. grid pts per axis')
# plt.ylabel('Detection probability')
# plt.show()


    



# set params
# alpha = 0.7
# L_star = 10**6.4
# N_tot = 50000
# phi_star = N_tot/(gamma(1 - alpha) * L_star)
dc_max = dc_max
dc_max_inv = 1.0/dc_max
# H_0 = 70
# h = H_0 / 100
# psi = 0.8
# # np.random.seed((rank * N_runs) + i)
# np.random.seed(1)
u = np.random.uniform(0, 1, 2*N_tot)


## lookup table for distance sampling
x_vals = np.linspace(0.0, dc_max, 1000000) #dc

f_x_vals = (x_vals**3 / 3) + ((psi / (dc_max * 4)) * x_vals**4)

norm = f_x_vals[-1]
u_vals = f_x_vals / norm


# root finder
inv_interp = interp1d(u_vals, x_vals, bounds_error=True, fill_value=(0, dc_max), kind='cubic')

u_rand = np.random.rand(N_tot)
u_rand = np.clip(u_rand, 0, 1 - np.finfo(float).eps)

d_c_ground = inv_interp(u_rand)



def schechter_cdf_inv(u, alpha, L_star):
    
    """
    Inverse CDF of the Schechter function. Returns L based on u and Schechter parameters
    
    0 <= u <= 1, alpha and L_star should be scalars
    """
    
    return gammaincinv(1-alpha, u) * L_star


# generate sample of galaxies with absolute magnitudes
#M_ground =  # check this is correct - drawing from luminosity function
M_abs_Sun = -6.0 #Look up for relevant filter from Willmer_2018 (ApJS, 236, 47) or similar
M_L_ratio = 1.0 #Ask about appropriate value.
L_ground = schechter_cdf_inv(u[:N_tot], alpha, L_star) # units of solar luminosities
M_ground = M_abs_Sun - 2.5*jnp.log10(L_ground) + 2.5*jnp.log10(M_L_ratio) #Absolute magnitude in relevant filter.
#Schecter_cdf_inv returns stellar mass in Solar units.


# log pdf of generated magnitudes



hist = plt.hist(M_ground, density=True, bins='auto')
n, bins = hist[0], hist[1]


# calculate redshift from distance
z_ground = dc_to_redshift(d_c_ground, c, omega_m)

# Distance modulus calculation.
mu = dc2mu(d_c_ground)

# calculate apparent magnitude for the sample galaxies
m_ground = M_ground + mu


# measurement error

m_err, z_err = .05, .001

# add uncertainty to redshift
# z_mock = z_ground * 10**(np.random.normal(0, z_err, size=len(M_ground)))
z_mock = (np.random.normal(z_ground, z_err, size=len(M_ground)))


# add Gaussian noise to m_ground
m_mock = (np.random.normal(m_ground, m_err, size=len(M_ground)))


# observed values
m_obs = m_mock[m_mock<18]
z_obs = z_mock[m_mock<18]
N_obs = len(m_obs)

frac_remaining = N_obs/N_tot
print('Fraction of galaxies remaining ', frac_remaining)
