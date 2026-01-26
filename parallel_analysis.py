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
import functions.dkw_tarp as dkw_tarp
import functions.dkw_pit as dkw_pit
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
with open("/Users/noahpierce/Portsmouth/bias_testing/HMC_output/numerical_schechter_nonpara_100trial_40kgals.pkl", "rb") as f:
    MPI_master = pickle.load(f)
    
    
    
def samples_dict_to_array(samples_dict):
    """
    Convert a NumPyro samples dictionary to a 2D numpy array of shape (num_samples, num_runs).
    """
    arrays = []
    for param in samples_dict.values():
        # Flatten each sample: shape (num_samples, param_dims...)
        param_flat = param.reshape(param.shape[0], -1)
        arrays.append(param_flat)
    
    # Concatenate all parameter arrays along axis 1
    return np.concatenate(arrays, axis=1)

# def samples_dict_to_array_nodes(samples_dict):
#     """
#     Converts dictionary to 3D numpy array of shape (num_samples, num_runs, num_nodes)
#     """
#     nodes_raw = np.array(list(samples_dict.values()))
#     nodes_raw = nodes_raw.squeeze(1)
#     nodes_samples = np.transpose(nodes_raw, (1, 0, 2))
    
#     return nodes_samples

def samples_dict_to_array_nodes(samples_dict):
    """
    Converts dictionary to 3D numpy array of shape (num_samples, num_runs, num_nodes)
    """
    keys = list(samples_dict.keys())
    
    arr = np.stack([samples_dict[k] for k in keys], axis=0)
    arr = np.transpose(arr, (2, 0, 1, 3))
    
    num_draws, num_cores, n_chains, num_nodes = arr.shape
    
    return arr.reshape(num_draws, num_cores * n_chains, num_nodes)


def truths_list(truths):
    
    output = []
    
    for i in range(len(truths)):
        output += truths[i]
    
    return output

def truths_nodes(truths):
    """
    truths_dict_values: something like f_pts_truth_list.values()
      - length n_runs
      - each element is [Array(11,)] (list of length 1)

    returns: (n_runs, n_components) np.ndarray
    """
    blocks = []
    for item in truths:
        # item is a list like [Array(11,)]
        arr = np.asarray(item)   # grab the Array(11,)
        blocks.append(arr)
    return np.concatenate(blocks, axis=0) 
    


alpha_dict = MPI_master['alpha']
L_star_dict = MPI_master['M_star']
phi_star_dict = MPI_master['phi_star']
# psi_dict = MPI_master['psi']
f_pts_dict = MPI_master['rho_pts']
sm_dict = MPI_master['shell_mass']

alpha_truth_list = MPI_master['alpha_truth']
L_star_truth_list = MPI_master['M_star_truth']
# psi_truth_list = MPI_master['psi_truth']
f_pts_truth_list = MPI_master['rho_pts_truth']
sm_truth_list = MPI_master['shell_mass_truth']

# f_raw = np.array(list(f_pts_dict.values()))
# print('f_raw.shape: ', f_raw.shape)

# f_raw = f_raw.squeeze(1)

# f_samples = np.transpose(f_raw, (1, 0, 2)) # (samples, runs, f_pts)
# print('f_samples.shape: ', f_samples.shape)

# # i = run index (0..9), k = f_pt index (0..10)
# alpha_i   = alpha_array[:, i]      # (1001,)
# f_i_all   = f_samples[:, i, :]     # (1001, 11)
# f_i_k     = f_samples[:, i, k]     # (1001,)


# convert dict to array
alpha_array = samples_dict_to_array(alpha_dict)
L_star_array = samples_dict_to_array(L_star_dict)
phi_star_array = samples_dict_to_array(phi_star_dict)
# psi_array = samples_dict_to_array(psi_dict)
f_array = samples_dict_to_array_nodes(f_pts_dict)
sm_array = samples_dict_to_array_nodes(sm_dict)

alpha_truth_array = truths_list(alpha_truth_list)
L_star_truth_array = truths_list(L_star_truth_list)
# psi_truth_array = truths_list(psi_truth_list)
f_truth_array = truths_nodes(f_pts_truth_list.values())
sm_truth_array = truths_nodes(sm_truth_list.values())








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




# nodes and shell mass PITs
frac_f = np.mean(f_array > f_truth_array[None, :, :], axis=0)

n_runs, n_fpts = frac_f.shape

ncols = 4
nrows = math.ceil(n_fpts / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

for k in range(n_fpts):
    r = k // ncols
    c = k % ncols
    ax = axes[r, c]
    sns.histplot(frac_f[:, k], bins='auto', stat='density', color='skyblue', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel('')
    ax.set_title(f"f_{k}")

# Hide any unused subplots
for k in range(n_fpts, nrows*ncols):
    r = k // ncols
    c = k % ncols
    axes[r, c].axis('off')

fig.suptitle("f_pts PIT histograms", y=0.95)
fig.text(0.5, 0.04, 'Fraction of samples above true value', ha='center')
plt.tight_layout()
plt.show()




frac_sm = np.mean(sm_array > sm_truth_array[None, :, :], axis=0)

n_runs, n_smpts = frac_sm.shape

ncols = 4
nrows = math.ceil(n_smpts / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

for k in range(n_smpts):
    r = k // ncols
    c = k % ncols
    ax = axes[r, c]
    sns.histplot(frac_sm[:, k], bins='auto', stat='density', color='skyblue', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel('')
    ax.set_title(f"sm {k}")

# Hide any unused subplots
for k in range(n_smpts, nrows*ncols):
    r = k // ncols
    c = k % ncols
    axes[r, c].axis('off')

fig.suptitle("shell mass PIT histograms", y=0.95)
fig.text(0.5, 0.04, 'Fraction of samples above true value', ha='center')
plt.tight_layout()
plt.show()




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

for i, j in zip((frac_alpha, frac_L_star), ('alpha', 'M_star')):
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
    plt.show()
    
    
    
# for nodes    
n_runs, n_fpts = frac_f.shape

ncols = 4
nrows = math.ceil(n_fpts / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

for k in range(n_fpts):
    r, c = divmod(k, ncols)
    ax = axes[r, c]

    # Make this the "current" axes so your function draws here
    

    # Call your existing PIT plotting function on this column
    dkw_pit.plot_pit_with_dkw(frac_f[:, k], alpha=0.05, ax=ax)

    ax.set_title(f"f_pt {k}")

# Hide unused subplots if n_fpts doesn’t fill the grid
for k in range(n_fpts, nrows * ncols):
    r, c = divmod(k, ncols)
    axes[r, c].axis("off")

fig.suptitle("f_pts PIT with DKW, 40k gals", y=0.95)
plt.tight_layout()
plt.show()



# for shell masses
n_runs, n_smpts = frac_sm.shape

ncols = 4
nrows = math.ceil(n_smpts / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

for k in range(n_fpts):
    r, c = divmod(k, ncols)
    ax = axes[r, c]

    # Make this the "current" axes so your function draws here
    

    # Call your existing PIT plotting function on this column
    dkw_pit.plot_pit_with_dkw(frac_sm[:, k], alpha=0.05, ax=ax)

    ax.set_title(f"sm {k}")

# Hide unused subplots if n_fpts doesn’t fill the grid
for k in range(n_fpts, nrows * ncols):
    r, c = divmod(k, ncols)
    axes[r, c].axis("off")

fig.suptitle("shell mass PIT with DKW, 40k gals", y=0.95)
plt.tight_layout()
plt.show()
    


##  ln(rho(r)) reconstruction
from numpyro.diagnostics import hpdi

n_splines = 11

r_points = np.linspace(0, dc_max, n_splines)

def f_of_r(r, f_pts):
    """
    Interpolates f(r) for given spline points (f_pts)

    """
    
    # clip r to ensure within min/max of r_points
    r_clipped = jnp.clip(r, r_points[0], r_points[-1])
    return jnp.interp(r_clipped, r_points, f_pts)

# Posterior samples of the spline control points
# shape: (n_samples, n_splines)
f_samps = f_array[:, 0, :]

scaling = f_truth_array[0, 0] / jnp.mean(f_samps[:, 0])

f_samps = f_samps * scaling

r_vals = jnp.linspace(0, dc_max, 10000)

# Convert to JAX
f_samps_jax = jnp.array(f_samps)   # (n_samples, n_splines)
r_vals_jax  = jnp.array(r_vals)    # (n_r,)

# Evaluate rho(r) on r_vals for each posterior sample
# f_of_r(r, f_pts) should already be defined and JAX-compatible
f_vals_samps = vmap(lambda f_pts: f_of_r(r_vals_jax, f_pts))(f_samps_jax)  # (n_samples, n_r)

# To avoid log(0), add a tiny epsilon
eps = 1e-30
log_f_vals_samps = jnp.log(f_vals_samps + eps)  # ln rho(r)

# Posterior mean and 95% HPDI in log space
log_fmean = jnp.mean(log_f_vals_samps, axis=0)         # (n_r,)
log_hpdi  = hpdi(log_f_vals_samps, prob=0.68)          # (2, n_r)

# Convert to numpy for plotting
log_fmean_np   = np.array(log_fmean)
log_hpdi_low   = np.array(log_hpdi[0])
log_hpdi_high  = np.array(log_hpdi[1])

# True profile in log-space
f_vals_ground = f_of_r(r_vals, f_truth_array[0, :])         # your true rho(r)
log_f_ground  = np.log(f_vals_ground + eps)

# Plot ln rho(r) vs r
plt.figure(figsize=(7, 4))

plt.plot(r_vals, log_f_ground, label='True ln ρ(r)', color='C0', lw=1)
plt.plot(r_vals, log_fmean_np, label='Sampled ln ρ(r) (mean)', color='C1', lw=1)

plt.fill_between(
    r_vals,
    log_hpdi_low,
    log_hpdi_high,
    color='C1',
    alpha=0.3,
    interpolate=True,
    label='95% HPDI (ln ρ)'
)

plt.title(r"$\ln \rho(r)$ ground vs MCMC (scaled to fix $\rho_0$ to truth)")
plt.xlabel(r'$D_c$')
plt.ylabel(r'$\ln \rho(r)$')
plt.legend()
plt.tight_layout()
plt.show()
    


import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
import math

# Settings
n_panels = 9
ncols = 3
nrows = 3

r_vals = jnp.linspace(0, dc_max, 10000)
eps = 1e-30

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), squeeze=False)

for run in range(n_panels):
    ax = axes[run // ncols, run % ncols]

    # --- samples for this run ---
    f_samps = f_array[:, run, :]                 # (n_samples, 11)
    
    scaling = f_truth_array[run, 0] / jnp.mean(f_samps[:, 0])
    f_samps = f_samps * scaling
    
    f_samps_jax = jnp.array(f_samps)

    # Evaluate rho(r) for each posterior sample: (n_samples, n_r)
    f_vals_samps = vmap(lambda f_pts: f_of_r(r_vals, f_pts))(f_samps_jax)

    # log space summaries
    log_f_vals_samps = jnp.log(f_vals_samps + eps)
    log_fmean = jnp.mean(log_f_vals_samps, axis=0)         # (n_r,)
    log_hpdi  = hpdi(log_f_vals_samps, prob=0.68)          # (2, n_r)

    # to numpy for plotting
    log_fmean_np  = np.array(log_fmean)
    log_hpdi_low  = np.array(log_hpdi[0])
    log_hpdi_high = np.array(log_hpdi[1])

    # --- truth for this run ---
    f_vals_ground = np.array(f_of_r(np.array(r_vals), f_truth_array[run, :]))
    log_f_ground  = np.log(f_vals_ground + eps)

    r_np = np.array(r_vals)

    # --- plot into this axis ---
    ax.plot(r_np, log_f_ground, lw=1, color='C0', label="True ln ρ(r)")
    ax.plot(r_np, log_fmean_np, lw=1, color='C1', label="Posterior mean ln ρ(r)")
    ax.fill_between(r_np, log_hpdi_low, log_hpdi_high, alpha=0.3, color='C1', label="95% HPDI")

    ax.set_title(f"Run {run}")
    ax.set_xlabel(r"$D_c$")
    ax.set_ylabel(r"$\ln \rho(r)$")

# Put one legend for the whole figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

fig.suptitle(r"$\ln \rho(r)$: ground vs MCMC (scaled to fix $\rho_0$ to truth)", y=0.94)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

    
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


# statistic, p_value = kstest(frac_psi, 'uniform')
# print(f"Psi KS statistic = {statistic:.4f}, p-value = {p_value:.4g}")


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

