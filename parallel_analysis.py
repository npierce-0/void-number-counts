#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:42:32 2025

@author: noahpierce
"""

import numpy as np
import math
import pandas as pd
from scipy import integrate, optimize
import scipy as sp
import numpyro
import numpyro.distributions as dist
from jax import random, jit
from jax import numpy as jnp
from jax import vmap
from jaxopt import Bisection
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import arviz as az
from scipy.special import gammaincinv
from jax.scipy.special import gamma, gammaincc, gammainc
from scipy.interpolate import interp1d
import pickle
import sys
og_sys_path = sys.path.copy()
sys.path.append('/Users/noahpierce/Portsmouth/bias_testing/functions/')
import dkw_tarp
import dkw_pit
sys.path = og_sys_path



# params from HMC script
alpha = 0.7
L_star = 10**6.4
L_star = -22.0
N_tot = 10000
phi_star = N_tot/(gamma(1 - alpha) * L_star)
dc_max = 2000
dc_max_inv = 1.0/dc_max
omega_m = 0.3
c = 2.998 * 10**5
H_0 = 70
h = H_0 / 100   # needs to be implamented, currently only defined here
psi = 0.8



# load in pickle from HMC script    
with open("/Users/noahpierce/Portsmouth/bias_testing/HMC_output/rand_truths_nonpara_100trial_40kgals.pkl", "rb") as f:
    MPI_master = pickle.load(f)
    
    
    
def samples_dict_to_array(samples_dict):
    """
    Convert a NumPyro samples dictionary to a 2D numpy array of shape (num_samples, num_params).
    """
    arrays = []
    for param in samples_dict.values():
        # Flatten each sample: shape (num_samples, param_dims...)
        param_flat = param.reshape(param.shape[0], -1)
        arrays.append(param_flat)
    
    # Concatenate all parameter arrays along axis 1
    return np.concatenate(arrays, axis=1)


def truths_list(truths_list):
    
    output = []
    
    for i in range(len(truths_list)):
        output += truths_list[i]
    
    return output


alpha_dict = MPI_master['alpha']
L_star_dict = MPI_master['M_star']
phi_star_dict = MPI_master['phi_star']
# psi_dict = MPI_master['psi']
f_pts_dict = MPI_master['f_pts']

alpha_truth_list = MPI_master['alpha_truth']
L_star_truth_list = MPI_master['M_star_truth']
# psi_truth_list = MPI_master['psi_truth']
f_pts_truth_list = MPI_master['f_pts_truth']


# convert dict to array
alpha_array = samples_dict_to_array(alpha_dict)
L_star_array = samples_dict_to_array(L_star_dict)
phi_star_array = samples_dict_to_array(phi_star_dict)
# psi_array = samples_dict_to_array(psi_dict)
f_array = samples_dict_to_array(f_pts_dict)

alpha_truth_array = truths_list(alpha_truth_list)
L_star_truth_array = truths_list(L_star_truth_list)
# psi_truth_array = truths_list(psi_truth_list)
f_truth_array = truths_list(f_pts_truth_list)









# standardised errors on alpha

alpha_samples = alpha_array

# alpha_median = np.median(alpha_samples, axis=0)
# alpha_std = np.std(alpha_samples, axis=0)

# alpha_true = np.ones(len(alpha_median)) * alpha

# hist_pts = (alpha_median - alpha_true) / alpha_std

# sns.histplot(hist_pts, bins='auto', stat='density', color='skyblue', label='Standrdised Error')

# # Overlay standard normal
# x = np.linspace(-4, 4, 100000)
# plt.plot(x, sp.stats.norm.pdf(x, scale=1), 'r--', label='Standard Normal')
# # plt.xlim(-5,5)
# plt.xlabel('Standardised Error')
# plt.ylabel('Density')
# plt.title('Comparison of Standardised Error on alpha to Standard Normal')
# plt.legend()
# plt.tight_layout()
# plt.show()


# standardised errors on L_star

L_star_samples = L_star_array

# L_star_median = np.median(L_star_samples, axis=0)
# L_star_std = np.std(L_star_samples, axis=0)

# L_star_true = np.ones(len(L_star_median)) * L_star

# hist_pts = (L_star_median - L_star_true) / L_star_std

# sns.histplot(hist_pts, bins='auto', stat='density', color='skyblue', label='Standrdised Error')

# # Overlay standard normal
# x = np.linspace(-4, 4, 100000)
# plt.plot(x, sp.stats.norm.pdf(x, scale=1), 'r--', label='Standard Normal')
# # plt.xlim(-5,5)
# plt.xlabel('Standardised Error')
# plt.ylabel('Density')
# plt.title('Comparison of Standardised Error on M star to Standard Normal')
# plt.legend()
# plt.tight_layout()
# plt.show()


# standardised errors on phi_star

phi_star_samples = phi_star_array

# phi_star_median = np.median(phi_star_samples, axis=0)
# phi_star_std = np.std(phi_star_samples, axis=0)

# phi_star_true = np.ones(len(phi_star_median)) * phi_star

# hist_pts = (phi_star_median - phi_star_true) / phi_star_std

# sns.histplot(hist_pts, bins='auto', stat='density', color='skyblue', label='Standrdised Error')

# # Overlay standard normal
# x = np.linspace(-4, 4, 100000)
# plt.plot(x, sp.stats.norm.pdf(x, scale=1), 'r--', label='Standard Normal')
# # plt.xlim(-5,5)
# plt.xlabel('Standardised Error')
# plt.ylabel('Density')
# plt.title('Comparison of Standardised Error on phi star to Standard Normal')
# plt.legend()
# plt.tight_layout()
# plt.show()


# standardised errors on psi

# psi_samples = psi_array

# psi_median = np.median(psi_samples, axis=0)
# psi_std = np.std(psi_samples, axis=0)

# psi_true = np.ones(len(psi_median)) * psi

# hist_pts = (psi_median - psi_true) / psi_std

# sns.histplot(hist_pts, bins='auto', stat='density', color='skyblue', label='Standrdised Error')

# # Overlay standard normal
# x = np.linspace(-4, 4, 100000)
# plt.plot(x, sp.stats.norm.pdf(x, scale=1), 'r--', label='Standard Normal')
# # plt.xlim(-5,5)
# plt.xlabel('Standardised Error')
# plt.ylabel('Density')
# plt.title('Comparison of Standardised Error on latent M to Standard Normal')
# plt.legend()
# plt.tight_layout()
# plt.show()





# alpha probability integral transform

f_alpha = alpha_samples

frac_alpha = np.mean(f_alpha > alpha_truth_array, axis=0)

sns.histplot(frac_alpha, bins='auto', stat='density', color='skyblue')
plt.xlabel('Fraction of samples above true value')
plt.title("alpha PIT")
plt.show()


# L_star probability integral transform

f_L_star = L_star_samples

frac_L_star = np.mean(f_L_star > L_star_truth_array, axis=0)

sns.histplot(frac_L_star, bins='auto', stat='density', color='skyblue')
plt.xlabel('Fraction of samples above true value')
plt.title("L_star PIT")
plt.show()


# phi_star probability integral transform

# f_phi_star = phi_star_samples

# frac_phi_star = np.mean(f_phi_star > phi_star, axis=0)

# sns.histplot(frac_phi_star, bins='auto', stat='density', color='skyblue')
# plt.xlabel('Fraction of samples above true value')
# plt.title("phi_star PIT")
# plt.show()


# psi probability integral transform

# f_psi = psi_samples

# frac_psi = np.mean(f_psi > psi_truth_array, axis=0)

# sns.histplot(frac_psi, bins='auto', stat='density', color='skyblue')
# plt.xlabel('Fraction of samples above true value')
# plt.title("psi PIT")
# plt.show()





# M_true_array = np.array(samples['M_true'])

# M_true_ground = np.array(M_ground)

# frac_M_true = np.mean(M_true_array > M_true_ground, axis=0)

# sns.histplot(frac_M_true, bins='auto', stat='density', color='skyblue')
# plt.xlabel('Fraction of samples above true value')
# plt.title("M truths PIT")
# plt.show()

# dkw_pit.plot_pit_with_dkw(frac_M_true, alpha=0.05)
# plt.show()

# from scipy.stats import kstest
# uniform = np.random.uniform(0, 1, 1000)
# statistic, p_value = kstest(frac_M_true, 'uniform')
# print(f"Latent M truths KS statistic = {statistic:.4f}, p-value = {p_value:.4g}")


# dc_true_array = np.array(samples['d_c_true'])

# dc_true_ground = np.array(d_c_ground)

# frac_dc_true = np.mean(dc_true_array > dc_true_ground, axis=0)

# sns.histplot(frac_dc_true, bins='auto', stat='density', color='skyblue')
# plt.xlabel('Fraction of samples above true value')
# plt.title("dc truths PIT")
# plt.show()

# dkw_pit.plot_pit_with_dkw(frac_dc_true, alpha=0.05)
# plt.show()

# from scipy.stats import kstest
# uniform = np.random.uniform(0, 1, 1000)
# statistic, p_value = kstest(frac_dc_true, 'uniform')
# print(f"Latent dc truths KS statistic = {statistic:.4f}, p-value = {p_value:.4g}")


# # standardised errors on latent M and dc
# M_samples = M_true_array

# M_median = np.median(M_samples, axis=0)
# M_std = np.std(M_samples, axis=0)

# M_true = M_true_ground

# hist_pts = (M_median - M_true) / M_std

# sns.histplot(hist_pts, bins='auto', stat='density', color='skyblue', label='Standrdised Error')

# # Overlay standard normal
# x = np.linspace(-4, 4, 100000)
# plt.plot(x, sp.stats.norm.pdf(x, scale=1), 'r--', label='Standard Normal')
# # plt.xlim(-5,5)
# plt.xlabel('Standardised Error')
# plt.ylabel('Density')
# plt.title('Comparison of Standardised Error on latent M to Standard Normal')
# plt.legend()
# plt.tight_layout()
# plt.show()


# dc_samples = dc_true_array

# dc_median = np.median(dc_samples, axis=0)
# dc_std = np.std(dc_samples, axis=0)

# dc_true = dc_true_ground

# hist_pts = (dc_median - dc_true) / dc_std

# sns.histplot(hist_pts, bins='auto', stat='density', color='skyblue', label='Standrdised Error')

# # Overlay standard normal
# x = np.linspace(-4, 4, 100000)
# plt.plot(x, sp.stats.norm.pdf(x, scale=1), 'r--', label='Standard Normal')
# # plt.xlim(-5,5)
# plt.xlabel('Standardised Error')
# plt.ylabel('Density')
# plt.title('Comparison of Standardised Error on latent dc to Standard Normal')
# plt.legend()
# plt.tight_layout()
# plt.show()










# PIT plotted like TARP

for i, j in zip((frac_alpha, frac_L_star, frac_psi), ('alpha', 'M_star', 'psi')):
    # ignore, num_sims = psi_array.shape
    # num_gamma_bins = num_sims // 10
    # h, gamma = np.histogram(i, density=True, bins=num_gamma_bins)
    # dx = gamma[1] - gamma[0]
    # ecp = np.cumsum(h) * dx
    # ecp = np.concatenate([[0], ecp])
    
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    # ax.plot(gamma, ecp, label='PIT')
    # ax.legend()
    # ax.set_ylabel("Expected Coverage")
    # ax.set_xlabel("Credibility Level")
    # ax.set_title(f"{j} PIT")
    
    # plt.subplots_adjust(wspace=0.4)
    # plt.show()
    
    
    dkw_pit.plot_pit_with_dkw(i, alpha=0.05)
    plt.suptitle(f"{j}, with selection, 40k gals")
    

    
    
# dkw_pit.plot_pit_with_dkw(frac_psi, alpha=0.05)
# plt.suptitle("Bias testing for density parameter")
# plt.savefig("pit.png", transparent=True, dpi=300)

# dkw_tarp.plot_tarp_with_dkw(frac_psi)


# # TARP setup

# nsamples, nsimulations = psi_array.shape

# samples_for_tarp = np.stack([alpha_array, L_star_array, phi_star_array, psi_array], axis=-1)
# truth_values = np.array([alpha, L_star, phi_star, psi])

# # For testing with a single parameter
# #sample_for_tarp = np.stack([alpha_array])
# #truth_values = np.array([0.7]) 

# theta_true = np.tile(truth_values, (nsimulations, 1))  # Shape: (nsimulations, nparams)

# print(f"Truth shapes: {theta_true.shape}")
# print('Samples shapes', samples_for_tarp.shape)

# from tarp import get_tarp_coverage

# ecp, beta, f = get_tarp_coverage(
#     samples_for_tarp, 
#     theta_true, 
#     references="random", 
#     metric="euclidean", 
#     norm=True, 
# )

# ecp, beta, f = get_tarp_coverage(
#     samples_for_tarp[:,:,3][:,:,np.newaxis], 
#     theta_true[:,3][:,np.newaxis], 
#     references="random", 
#     metric="euclidean", 
#     norm=True,
# )

# # samples, theta = get_tarp_coverage(
# #     samples_for_tarp[:,:,0][:,:,np.newaxis], 
# #     theta_true[:,0][:,np.newaxis], 
# #     references="random", 
# #     metric="euclidean", 
# #     norm=True, 
# # )

# # test = np.mean(samples)

# sns.histplot(f, bins='auto', stat='density', color='skyblue')
# plt.xlabel('Fraction of samples below reference value')
# plt.title("tarp 'hist")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(4, 4))

# ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
# ax.plot(beta, ecp, label='TARP')
# ax.legend()
# ax.set_ylabel("Expected Coverage")
# ax.set_xlabel("Credibility Level")

# plt.subplots_adjust(wspace=0.4)
# plt.show()


# dkw_tarp.plot_tarp_with_dkw(f, alpha=0.05)
# plt.suptitle('')
# plt.show()



## KS test
from scipy.stats import kstest
uniform = np.random.uniform(0, 1, 1000)

# Run the KS test against a true uniform(0,1)
statistic, p_value = kstest(frac_alpha, 'uniform')
print(f"Alpha KS statistic = {statistic:.4f}, p-value = {p_value:.4g}")


statistic, p_value = kstest(frac_L_star, 'uniform')
print(f"M* KS statistic = {statistic:.4f}, p-value = {p_value:.4g}")


statistic, p_value = kstest(frac_psi, 'uniform')
print(f"Psi KS statistic = {statistic:.4f}, p-value = {p_value:.4g}")


sys.exit()



# Set up figure size based on grid
rows, cols = 5, 5
fig = plt.figure(figsize=(40, 30))

for i in range(25):
    ax = plt.subplot(rows, cols, i + 1)  # subplot index starts at 1
    sns.histplot(psi_array[:,i], stat='density', bins='auto', color='skyblue', ax=ax)
    sns.kdeplot(psi_array[:,i], ax=ax)
    # Remove all text
    ax.set_title("")          # remove title
    ax.set_xlabel("")          # remove x-axis label
    ax.set_ylabel("")          # remove y-axis label
    ax.set_yticks([])          # remove y-axis ticks

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.suptitle("psi posteriors, Jeffrey's prior and wide integral", fontsize=64)
plt.show()

# sns.histplot(psi_array[:,0], bins='auto', color='skyblue')
# plt.show()



for i in range(25):
    
    data = np.column_stack([alpha_samples[:,i], L_star_samples[:,i], psi_samples[:,i]])
    
    # Create the corner plot
    fig = corner.corner(
        data,
        labels=[r"$\alpha$", r"$M_\ast$", r"$\psi$"],
        show_titles=True,
        truths=[alpha_truth_array[i], L_star_truth_array[i], psi_truth_array[i]]  # your reference/true values
    )
    
    plt.show()


# alpha predictions distribution
alpha_mean = np.mean(alpha_samples, axis=0)

sns.histplot(alpha_mean, bins='auto', stat='density', color='skyblue', label='Standrdised Error')
plt.axvline(alpha)
plt.show()


L_star_mean = np.mean(L_star_samples, axis=0)

sns.histplot(L_star_mean, bins='auto', stat='density', color='skyblue', label='Standrdised Error')
plt.axvline(L_star)
plt.show()


psi_mean = np.mean(psi_samples, axis=0)

sns.histplot(psi_mean, bins='auto', stat='density', color='skyblue', label='Standrdised Error')
plt.axvline(psi)
plt.show()

