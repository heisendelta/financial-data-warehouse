import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity

from scipy.optimize import minimize
from scipy.linalg import block_diag


# Marcenko-Pastur PDF

def marcenko_pastur_pdf(var, q, pts): # q = T / N
    e_min = var * (1 - (1.0 / q) ** 0.5) ** 2
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    
    pdf = q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min)) ** 0.5
    pdf = pd.Series(pdf, index=e_val)
    return pdf

def get_PCA(matrix):
    # get e_val, e_vec form a hermitian matrix

    e_val, e_vec = np.linalg.eigh(matrix)
    indices = e_val.argsort()[::-1] # arguments for sorting e_val in desc order

    e_val = e_val[indices]
    e_vec = e_vec[:, indices]

    e_val = np.diagflat(e_val)
    return e_val, e_vec

def fit_KDE(obs, b_width=0.25, kernel='gaussian', x=None):
    # fit kernel to a series of obs and derive the prob of obs
    # x is the array of values on whihc the fit KDE will be evaluated

    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)

    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(obs)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf

def get_rnd_cov(n_cols, n_facts):
    w = np.random.normal(size=(n_cols, n_facts))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=n_cols))
    return cov

def cov2corr(cov):
    # derive correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    corr[corr < -1] = -1 # numerical error
    corr[corr > 1] = 1

    return corr

def err_pdfs(var, e_val, q, b_width, pts=1000):
    # fit error
    pdf0 = marcenko_pastur_pdf(var, q, pts) # theoretial pdf
    pdf1 = fit_KDE(e_val, b_width, x=pdf0.index.values) # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse

def find_max_eval(e_val, q, b_width):
    out = minimize(lambda x: err_pdfs(x[0], e_val, q, b_width),
                   x0=[0.5],
                   bounds=[(1e-5, 1 - 1e-5)])
    if out['success']:
        var = out['x'][0]
    else:
        var = 1

    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    return e_max, var

def denoised_corr(e_val, e_vec, n_facts):
    # remove noise from corr by fixing random eigenvalues

    e_val_ = np.diag(e_val).copy()
    e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
    e_val_ = np.diag(e_val_)

    corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
    corr1 = cov2corr(corr1)
    return corr1

def denoised_corr2(e_val, e_vec, n_facts, alpha=0):
    # remove noise from corr through targeted shrinkage

    e_val_l, e_vec_l = e_val[:n_facts, :n_facts], e_vec[:, :n_facts]
    e_val_r, e_vec_r = e_val[n_facts:, n_facts:], e_vec[:, n_facts:]

    corr0 = np.dot(e_vec_l, e_val_l).dot(e_vec_l.T)
    corr1 = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)

    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return corr2

def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


def form_block_matrix(n_blocks, b_size, b_corr):
    block = np.ones((b_size, b_size)) * b_corr
    block[range(b_size), range(b_size)] = 1
    corr = block_diag(*([block] * n_blocks))
    return corr

def form_true_matrix(n_blocks, b_size, b_corr):
    corr0 = form_block_matrix(n_blocks, b_size, b_corr)
    corr0 = pd.DataFrame(corr0)

    cols = corr0.columns.to_list()
    np.random.shuffle(cols)

    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])

    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0

def sim_cov_mu(mu0, cov0, n_obs, shrink=False):
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=n_obs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    
    if shrink:
        from sklearn.covariance import LedoitWolf
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)

    return mu1, cov1

def estimate_cov(returns, shrink=False):
    if shrink:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(returns).covariance_
    else:
        cov = returns.cov().values
    return cov

def denoise_cov(cov0, q, b_width, type):
    corr0 = cov2corr(cov0)
    e_val0, e_vec0 = get_PCA(corr0)
    e_max0, var0 = find_max_eval(np.diag(e_val0), q, b_width)

    n_facts0 = e_val0.shape[0] - np.diag(e_val0)[::-1].searchsorted(e_max0)

    if type == 1: # remove noise from corr by fixing random eigenvalues
        corr1 = denoised_corr(e_val0, e_vec0, n_facts0)
    elif type == 2: # remove noise from corr through targeted shrinkage
        corr1 = denoised_corr2(e_val0, e_vec0, n_facts0, alpha=0.5)

    cov1 = corr2cov(corr1, np.diag(cov0) ** .5)
    return cov1

def opt_port(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    
    if mu is None:
        mu = ones
    
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w
