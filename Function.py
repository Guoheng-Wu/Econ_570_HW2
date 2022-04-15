import numpy as np
import random
from tqdm import tqdm
import statsmodels.api as sm


def fn_generate_cov(dim,corr):
    acc  = []
    for i in range(dim):
        row = np.ones((1,dim)) * corr
        row[0][i] = 1
        acc.append(row)
    return np.concatenate(acc,axis=0)

# Generate all Covariates X variable
def fn_generate_multnorm(nobs, corr, nvar):
    std = (np.abs(np.random.normal(loc=1, scale=0.5, size=(nvar, 1)))) ** (1 / 2)
    acc = []
    for i in range(nvar):
        acc.append(np.reshape(np.random.normal(0, std[i], nobs), (nobs, 1)))

    normvars = np.concatenate(acc, axis=1)
    cov = fn_generate_cov(nvar)
    c = np.linalg.cholesky(cov)
    X = np.transpose(np.dot(c, np.transpose(normvars)))
    return X


# Generate Confounder variable C
def fn_randomize_conf(nobs):
    conf = np.reshape(np.random.normal(4, 2, nobs), (nobs, 1))
    return conf


# Generate Treatment variable T
def fn_randomize_treatment(nobs, conf, C):
    T = []
    if conf == False:
        for i in range(nobs):
            t = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])
            T.append(t)
    if conf == True:
        for i in range(nobs):
            t = np.random.choice(np.arange(0, 2), p=[0.5 - 0.025 * float(C[i]), 0.5 + 0.025 * float(C[i])])
            T.append(t)
    return np.reshape(T, (nobs, 1))


# Generate Variable between the path from the treatment to the outcome S
def fn_randomize_s(nobs, T):
    S = []
    for i in range(nobs):
        s = np.random.choice(np.arange(0, 2), p=[0.5 - 0.1 * float(T[i]), 0.5 + 0.1 * float(T[i])])
        S.append(s)
    return np.reshape(S, (nobs, 1))


# Generate Data
def fn_generate_data(tau, N, p, corr, conf, sv):
    if conf == False:
        conf_mult = 0
    if conf == True:
        conf_mult = 1
    if sv == False:
        s_mult = 0
    if sv == True:
        s_mult = 1

    X = fn_generate_multnorm(N, corr, p)
    C = fn_randomize_conf(N)
    T = fn_randomize_treatment(N, conf, C)
    S = fn_randomize_s(N, T)
    err = np.random.normal(0, 1, [N, 1])
    beta0 = np.random.normal(5, 5, [p, 1])

    Yab = tau * T + X @ beta0 + conf_mult * 0.6 * C + s_mult * 1 * S + err

    if conf == False and sv == False:
        return (Yab, T, X)
    if conf == True and sv == False:
        return (Yab, T, X, C)
    if conf == False and sv == True:
        return (Yab, T, X, S)
    if conf == True and sv == True:
        return (Yab, T, X, C, S)


def fn_tauhat_means(Yt, Yc):
    nt = len(Yt)
    nc = len(Yc)
    tauhat = np.mean(Yt) - np.mean(Yc)
    se_tauhat = (np.var(Yt, ddof=1) / nt + np.var(Yc, ddof=1) / nc) ** (1 / 2)
    return (tauhat, se_tauhat)


# Return estimate tauhats and sehats
def fn_get_estimate(tau, Nrange, p, corr, conf, sv, control_conf, control_flagX, control_s):
    tauhats = []
    sehats = []
    for r in tqdm(Nrange):
        if conf == False and sv == False:
            if control_flagX == False:
                Yexp, T, X = fn_generate_data(tau, N, p, corr, conf, sv)
                Yt = Yexp[np.where(T == 1)[0], :]
                Yc = Yexp[np.where(T == 0)[0], :]
                tauhat, se_tauhat = fn_tauhat_means(Yt, Yc)
            if control_flagX == True:
                Yexp, T, X = fn_generate_data(tau, N, p, corr, conf, sv)
                covars = np.concatenate([T, X], axis=1)
                mod = sm.OLS(Yexp, covars)
                res = mod.fit()
                tauhat = res.params[0]
                se_tauhat = res.HC1_se[0]
        if conf == True and sv == False:
            if control_conf == False:
                Yexp, T, X, C = fn_generate_data(tau, N, p, corr, conf, sv)
                covars = np.concatenate([T, X], axis=1)
                mod = sm.OLS(Yexp, covars)
                res = mod.fit()
                tauhat = res.params[0]
                se_tauhat = res.HC1_se[0]
            if control_conf == True:
                Yexp, T, X, C = fn_generate_data(tau, N, p, corr, conf, sv)
                covars = np.concatenate([T, X, C], axis=1)
                mod = sm.OLS(Yexp, covars)
                res = mod.fit()
                tauhat = res.params[0]
                se_tauhat = res.HC1_se[0]
        if conf == False and sv == True:
            if control_s == False:
                Yexp, T, X, S = fn_generate_data(tau, N, p, corr, conf, sv)
                covars = np.concatenate([T, X], axis=1)
                mod = sm.OLS(Yexp, covars)
                res = mod.fit()
                tauhat = res.params[0]
                se_tauhat = res.HC1_se[0]
            if control_s == True:
                Yexp, T, X, S = fn_generate_data(tau, N, p, corr, conf, sv)
                covars = np.concatenate([T, X, S], axis=1)
                mod = sm.OLS(Yexp, covars)
                res = mod.fit()
                tauhat = res.params[0]
                se_tauhat = res.HC1_se[0]

        tauhats = tauhats + [tauhat]
        sehats = sehats + [se_tauhat]
    return (tauhats, sehats)


# Return bias,rmse,size of treatment effect estimate
def fn_bias_rmse_size(theta0, thetahat, se_thetahat, cval=1.96):
    b = thetahat - theta0
    bias = np.mean(b)
    rmse = np.sqrt(np.mean(b ** 2))
    tval = b / se_thetahat
    size = np.mean(1 * (np.abs(tval) > cval))
    return (bias, rmse, size)