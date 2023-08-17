#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions to DA

Author: Esteban Alonso González - alonsoe@cesbio.cnes.fr

"""

import numpy as np
from numpy.random import random
from scipy import special
from scipy.linalg import sqrtm
import config as cfg
import constants as cnt
if cfg.numerical_model == 'FSM2':
    import modules.fsm_tools as model
elif cfg.numerical_model == 'dIm':
    import modules.dIm_tools as model
elif cfg.numerical_model == 'snow17':
    import modules.snow17_tools as model
else:
    raise Exception('Model not implemented')
import modules.met_tools as met
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as krn
import shutil
import os


def ens_klm(prior, obs, pred, alpha, R, rho_AB=1, rho_BB=1,
            stochastic=True, dosvd=True):
    """
    EnKA: Implmentation of the Ensemble Kalman Analysis
    """
    # TODO: Using a Bessel correction (Ne-1 normalization) for sample
    # covariances.
    # TODO: Perturbing the predicted observations rather than the observations
    #      following van Leeuwen (2020, doi: 10.1002/qj.3819) which is
    #      the formally when defining the covariances matrices.
    # TODO: Looking into Sect. 4.5.3. in Evensen
    # (2019, doi: 10.1007/s10596-019-9819-z)
    # TODO: Augmenting with artificial momentum to get some inertia.
    # TODO: Better understand/build on the underlying tempering/annealing
    # idea in ES-MDA.
    # TODO: Versions that may scale better for large n such as those
    #       working in the ensemble subspace (see e.g. formulations
    #       in Evensen, 2003, Ocean Dynamics).

    # Dimensions.
    m = np.size(obs)  # Number of obs
    n = np.shape(prior)[0]  # Tentative number of state vars and/or parameters
    if np.size(prior) == n:  # If prior is 1D correct the above.
        N = n  # Number of ens. members.
        n = 1  # Number of state vars and/or parameters.
    else:
        # If prior is 2D then the lenght of the second dim is the ensemble size
        N = np.shape(prior)[1]

    if pred.ndim == 1:
        pred = pred[np.newaxis, :]

    # Checks on the observation error covariance matrix.
    if R.ndim == 2:
        if np.shape(R)[0] == m and np.shape(R)[1] == m:
            Rsqrt = sqrtm(R)

        else:
            raise Exception('r_cov bad dimensions')

    else:
        if np.size(R) == 1:
            Rsqrt = np.sqrt(R)
            R = R*np.identity(m)
            Rsqrt = Rsqrt*np.identity(m)
        elif np.size(R) == m:
            # print('diag')
            # Square root of a diag matrix is the square root of its elements.
            Rsqrt = np.sqrt(R)
            R = np.diag(R)
            Rsqrt = np.diag(Rsqrt)
            # Convert to matrix if specified as a vector.
        else:
            raise Exception('R must be a scalar, m x 1 vector,\
                            or m x m matrix.')

    # pdb.set_trace()
    # Anomaly calculations.
    mprior = np.mean(prior, -1)  # Prior ensemble mean
    if n == 1:
        A = prior-mprior
    else:
        A = prior-mprior[:, None]
    mpred = np.mean(pred, -1)  # Prior predicted obs ensemble mean
    if m == 1:
        B = pred-mpred
    else:
        B = pred-mpred[:, None]

    Bt = B.T  # Tranposed -"-

    # Covariance matrices
    C_AB = A@Bt  # Prior-predic obs covariance matrix multiplied by N (n x m)
    C_BB = B@Bt  # Predicted obs covariance matrix multiplied by N (m x m)
    # Ap=np.linalg.pinv(A)
    # C_BB=(B@(Ap@A))@((B@(Ap@A)).T)
    # Localize covariance matrices
    C_AB = rho_AB * C_AB
    C_BB = rho_BB * C_BB
    aR = (N*alpha)*R  # Scaled observation error covariance matrix (m x m)

    if dosvd:
        L = np.linalg.cholesky(aR)
        Linv = np.linalg.inv(L)
        Ctilde = Linv@C_BB@(Linv.T)+np.eye(m)
        # This is a shortcut, but then you don't set the SVD ratio explicitly
        # Cinv=np.linalg.pinv(Ctilde,rcond=1e-2)
        [U, S, _] = np.linalg.svd(Ctilde)
        # Note, in np S is already a vector.
        # Since Ctilde is pos def, then U=V hence why V output is supressed
        Svr = np.cumsum(S)/np.sum(S)
        minds = np.arange(m)
        # Singular value ratio threshold (typically between 0.9-0.99=90%-99%)
        thresh = 0.9
        keep = min(minds[Svr > thresh])  # Number of singular values to keep
        St = S[:(keep+1)]  # Exclusive indexing (yay python!)
        Ut = U[:, :(keep+1)]
        Sti = 1/St  # Vector (representing a diagonal matrix)
        Ctildei = (Ut*Sti)@Ut.T  # Same as Ut@np.diag(Sti)@Ut.T
        Cinv = Linv.T@Ctildei@Linv
        # U,S=np.linalg.svd(Ctilde)
    else:
        Cinv = np.linalg.inv(C_BB+aR)

    if stochastic:
        # Perturbed observations.
        pert = np.random.randn(m, N)
        Y = np.outer(obs, np.ones(N))+np.sqrt(alpha)*(Rsqrt@pert)
        # Analysis step
        if n == 1 and m == 1:  # Scalar case
            K = C_AB*Cinv
            inno = Y-pred
            post = prior+K*inno
        else:
            K = C_AB@Cinv  # Kalman gain (n x m)
            inno = Y-pred  # Innovation (m x N)
            post = prior+K@inno  # Posterior (n x N)
    else:
        Y = np.squeeze(obs)
        # Analysis step
        if n == 1 and m == 1:  # Scalar case
            K = C_AB * Cinv
            inno = Y - mpred
            mpost = mprior + K * inno  # Posterior (n x N)
            A = A - 0.5*K*B
            post = mpost+A
        else:
            K = C_AB@Cinv  # Kalman gain (n x m)
            inno = Y - mpred  # Innovation (m x N)
            mpost = mprior + K @ inno  # Posterior (n x N)
            A = A - 0.5*K@B
            post = (mpost+A.T).T

    return post


def pbs(obs, pred, R):
    """
    PBS: Implmentation of the Particle Batch Smoother
    Inputs:
        obs: Observation vector (m x 1 array)
        pred: Predicted observation ensemble matrix (m x N array)
        r_cov: Observation error covariance 'matrix' (m x 1 array, or scalar)
    Outputs:
        w: Posterior weights (N x 1 array)
    Dimensions:
        ens_mem is the number of ensemble members and m is the number
        of observations.

    Here we have implemented the particle batch smoother, which is
    a batch-smoother version of the particle filter (i.e. a particle filter
    without resampling), described in Margulis et al.
    (2015, doi: 10.1175/JHM-D-14-0177.1). As such, this routine can also be
    used for particle filtering with sequential data assimilation. This scheme
    is obtained by using a particle (mixture of Dirac delta functions)
    representation of the prior and applying this directly to Bayes theorem. In
    other words, it is just an application of importance sampling with the
    prior as the proposal density (importance distribution). It is also the
    same as the Generalized Likelihood Uncertainty Estimation (GLUE) technique
    (with a formal Gaussian likelihood)which is widely used in hydrology.

    This particular scheme assumes that the observation errors are additive
    Gaussian white noise (uncorrelated in time and space). The "logsumexp"
    trick is used to deal with potential numerical issues with floating point
    operations that occur when dealing with ratios of likelihoods that vary by
    many orders of magnitude.

    Based on a previous version from  K. Aalstad (14.12.2020)
    """
    # TODO: Implement an option for correlated observation errors if R is
    #      specified as a matrix (this would slow down this function).
    # TODO: Consier other likelihoods and observation models.
    # TODO: Look into iterative versions of importance sampling.

    # Dimensions.
    n_obs = np.size(obs)  # Number of obs
    ens_mem = np.shape(pred)[-1]

    # Checks on the observation error covariance matrix.
    if np.size(R) == 1:
        R = R * np.ones(n_obs)
    elif np.size(R) == n_obs:
        pass
    else:
        raise Exception('r_cov must be a scalar, m x 1 vector.')

    # Residual and log-likelihood
    if n_obs == 1:
        residual = obs - pred
        llh = -0.5 * ((residual**2) * (1/R))
    else:
        residual = obs - pred
        llh = -0.5 * ((1/R)@(residual**2))

    # Log of normalizing constant
    # A properly scaled version of this could be output for model comparison.
    log_z = special.logsumexp(llh)  # from scipy.special import logsumexp

    # Weights
    logw = llh - log_z  # Log of posterior weights
    weights = np.exp(logw)  # Posterior weights

    if np.size(weights) == ens_mem and np.round(np.sum(weights), 10) == 1:
        pass
    else:
        raise Exception('Something wrong with the PBS')

    weights = np.squeeze(weights)  # Remove axes of length one

    Neff = 1/np.sum(weights**2)
    Neff = np.round(Neff)

    return weights, Neff


def ProPBS(obs, pred, R, priormean, priorcov, proposal):
    """
    ProPBS: Implmentation of the Proposal Particle Batch Smoother
    Inputs:
        obs: Observation vector (m x 1 array)
        pred: Predicted observation ensemble matrix (m x N array)
        R: Observation error covariance 'matrix' (m x 1 array, or scalar)
        priormean: The mean (vector) of the prior (n x 1 array)
        priorcov: The covariance (matrix) of the prior (n x n array)
        proposal: Samples from the proposal (n x N array)

    Outputs:
        w: Posterior weights (N x 1 array)
    Dimensions
        N is the number of ensemble members, n is the number of state
        variables and/or parameters, and m is the number of observations.

    The PIES scheme is obtained by using the output of the Iterative Ensemble
    Smoother, i.e. a Gaussian distribution, as the proposal in importance
    sampling. Note that we do not need to use the final posterior from IES
    as the proposal, we can instead use the "posterior" at any iteration
    l=0,...,Na where Na is the number of assimilation cycles.Note that the
    in the special case l=0 we end up with the standard PBS scheme if we
    disregardthe differences between the true and sampled prior statistics.

    Code by: K. Aalstad (14.12.2020) based on an earlier Matlab version.
    """

    # Dimensions.
    m = np.size(obs)  # Number of obs
    N = np.shape(pred)[-1]

    # Checks on the observation error covariance matrix.
    if np.size(R) == 1:
        R = R*np.ones(m)
    elif np.size(R) == m:
        pass
    else:
        raise Exception('R must be a scalar, m x 1 vector.')

    if m == 1:
        residual = obs-pred
        Phid = -0.5*((residual**2))
    else:
        residual = obs.flatten() - pred.T
        residual = residual.T
        Phid = -0.5*((1/R)@(residual**2))  # (1 x m) x (m x N) == 1 x N

    # Transpose for broadcasting (N x n) - (n) = (N x n) [row major logic]
    # TODO: Substitute np.linalg.inv by SVD invesion
    # use scipy.linalg.pinvh ? (rtol parameter instead of numpy rcond)
    A0 = proposal.T-priormean
    A0 = A0.T  # n x N
    C0inv = np.linalg.inv(priorcov)
    Phi0 = -0.5*(A0.T)@C0inv@(A0)  # (N x n)  x (n x n) x (n x N) = N x N
    Phi0 = np.diag(Phi0)  # N x 1

    proposalmean = proposal.mean(-1)
    A = (proposal.T-proposalmean)
    A = A.T  # n x N
    C = (1/N)*(A@A.T)
    Cinv = np.linalg.inv(C)
    Phip = -0.5*(A.T)@Cinv@A  # N x N
    Phip = np.diag(Phip)  # N x 1

    Phi = Phid+Phi0-Phip
    Phimax = Phi.max()
    Phis = Phi-Phimax  # Scaled to avoid numerical overflow
    # See e.g. Chopin and Papaspiliopoulos (2020) book on SMC

    w = np.exp(Phis)
    w = w/sum(w)

    # Neff = 0 if degenerate and 1 if equal weights
    Neff = 1/np.sum(w**2)
    Neff = np.round(Neff)

    return w, Neff


def AMIS(obs, pred, R, prim, pric, propm, propc, props):
    """
    AMIS: Adaptive Multiple Importance Sampling
    Inputs:
        obs: Observation vector (No x 1 array)
        pred: Predicted observation ensemble (No x Ne x Nl array)
        R: Observation error covariance 'matrix' (No x 1 array, or scalar)
        prim: The mean (vector) of the prior (Np x 1 array)
        pric: The covariance (matrix) of the prior (Np x Np array)
        propm: Means of the DM proposal (Np x Nl)
        propc: Covariances of the DM proposal (Np x Nl)
        proposal: Samples from the DM proposal (Np x Ne x Nl array)
    Outputs:
        w: Posterior weights (Ne x 1 array)
    Dimensions
        Ne is the number of ensemble members, Np is the number of state
        variables and/or parameters, No is the number of observations, Nl
        is the number of iterations thus far.

    This AMIS scheme is based on the work of Cornuet et al. (2012) which is
    based on combining the Population Monte Carlo approach with the
    so-called deterministic mixture (DM) approach of Owen and Zhou (2000).
    It is less wasteful, more stable, and often faster to converge than
    simpler AIS algorithms.

    Code by: K. Aalstad (August 2023) based on an earlier Matlab version.
    """

    # Dimensions.
    No = np.shape(pred)[0]
    Ne = np.shape(pred)[1]
    Nl = np.shape(pred)[2]
    # Np=np.shape(propm)[0]

    # Checks on the observation error covariance matrix.
    if np.size(R) == 1:
        R = R*np.ones(No)
    elif np.size(R) == No:
        pass
    else:
        raise Exception('R must be a scalar, m x 1 vector.')

    phi = np.zeros([Ne, Nl])  # negative log of target
    lsepsi = np.zeros([Ne, Nl])  # logsumexp of the DM proposal
    for ell in range(Nl):
        # Terms related to the target
        propell = props[:, :, ell]  # Np x Ne
        A0ell = (propell.T-prim).T
        phi0ell = 0.5*(A0ell.T)@np.linalg.solve(pric, A0ell)  # Ne x Ne
        phi0ell = np.diag(phi0ell)  # Ne
        predell = pred[:, :, ell]  # No x Ne
        residuell = (obs-predell.T).T  # No x Ne
        phidell = 0.5*(1/R)@(residuell**2)  # Ne
        phi[:, ell] = phi0ell+phidell

        psij = np.zeros([Ne, Nl])
        for j in range(Nl):
            mj = propm[:, j]
            Cj = propc[:, :, j]
            cj = np.linalg.det(2*np.pi*Cj)**(-0.5)
            lcj = np.log(cj)
            Aj = (propell.T-mj).T
            psi = 0.5*(Aj.T)@np.linalg.solve(Cj, Aj)
            psi = np.diag(psi)
            psi = psi-lcj
            psij[:, j] = psi
        psijx = np.max(psij, 1)  # Ne
        psijs = (psij.T-psijx).T  # Ne x Nl
        lsepsiell = psijx+np.log(np.sum(np.exp(psijs), 1))
        lsepsi[:, ell] = lsepsiell

    logw = -phi-lsepsi
    logw = logw.flatten('F')  # Purposely flattening column major order
    logw = logw-np.max(logw)
    w = np.exp(logw)
    w = w/np.sum(w)
    Neff = 1/np.sum(w**2)

    return w, Neff


def mcmc(Ensemble, observations_sbst_masked, R,
         chain_len, adaptive, histcov):

    vars_to_perturbate = cfg.vars_to_perturbate
    SD0 = np.asarray([cnt.sd_errors[x] for x in vars_to_perturbate])
    m0 = np.asarray([cnt.mean_errors[x] for x in vars_to_perturbate])

    # starting ensemble
    starting_parameters = Ensemble.train_parameters[-2]
    starting_parameters = transform_space(starting_parameters, 'to_normal')
    predicted = Ensemble.train_pred[-2][:, 0]

    # NOTE:
    # the initial conditions will be defined by a random particle after the
    # IES. This is fine for years 0 but not optimal for the following seasons.
    # Here it might make sense to use PIES and use the highest weight
    init_conditions = Ensemble.out_members_iter[0]

    # create tmp storage of the model
    temp_dest = model.model_copy(Ensemble.lat_idx, Ensemble.lon_idx)

    # Init chain
    phic = np.reshape(starting_parameters.T[0, :],
                      (1, starting_parameters.shape[0]))
    nll = negloglik(predicted[:, np.newaxis], observations_sbst_masked, R)

    Uc = neglogpost(nll, phic, SD0, m0)
    mcmc_storage = np.zeros((chain_len, len(vars_to_perturbate)))
    mcmc_storage[:] = np.nan
    sigp = 0.1  # Gaussian proposal width. Only used if histcov is False
    Np = len(vars_to_perturbate)
    # Using ensemble Kalman methods to speed up the burn in was insired
    # by Zhang et al. (2020, https://doi.org/10.1029/2019WR025474)
    if histcov:  # Posterior IES covariance as proposal covariance
        Ne = starting_parameters.shape[0]
        # Anom is based on IES posterior mean, not prior mean
        mpo = np.mean(starting_parameters, 1)
        anom = (starting_parameters.T-mpo).T
        # Posterior covariance of IES (in transformed space)
        C0 = (anom@anom.T)/Ne
        C0 = 0.01*C0  # Scale this to not be too large
    else:  # Isotropic covariance as proposal covariance
        C0 = (sigp**2)*np.eye(Np)
    Sc = np.linalg.cholesky(C0)
    Id = np.eye(Np)
    # import pdb; pdb.set_trace()

    accepted = 0
    for nsteps in range(chain_len):
        r = np.random.randn(Np)
        prop = Sc@r
        phip = phic+prop
        phip = transform_space(phip.T, 'from_normal').T

        # run the model
        # create forcing candidate
        forcing_mcmcstep, noise_mcmc = met.perturb_parameters(Ensemble.forcing,
                                                              noise=phip.T,
                                                              update=True)
        phip = transform_space(phip.T, 'to_normal').T

        # write perturbed forcing
        model.model_forcing_wrt(forcing_mcmcstep, temp_dest, Ensemble.step)
        # Write init conditions or dump file from previous run if step != 0
        if cfg.numerical_model in ['FSM2']:
            if Ensemble.step != 0:
                model.write_dump(init_conditions, temp_dest)
            # create open loop simulation
            model.model_run(temp_dest)
            # read model outputs
            state_tmp, dump_tmp =\
                model.model_read_output(temp_dest)

        elif cfg.numerical_model in ['dIm', 'snow17']:
            if Ensemble.step != 0:
                state_tmp, dump_tmp =\
                    model.model_run(forcing_mcmcstep, temp_dest)
            else:
                state_tmp, dump_tmp =\
                    model.model_run(forcing_mcmcstep)

        mcmc_step_simulations = get_predictions([state_tmp], cfg.var_to_assim)
        _, predicted, _ = \
            tidy_obs_pred_rcov(mcmc_step_simulations,
                               Ensemble.observations,
                               Ensemble.errors)

        nll = negloglik(predicted[:, np.newaxis], observations_sbst_masked, R)
        Up = neglogpost(nll, phip, SD0, m0)

        mh = min(1, np.exp(-Up+Uc))
        u = np.random.rand(1)
        accept = (mh > u)
        if accept:
            phic = phip
            Uc = Up
            accepted = accepted + 1

        mcmc_storage[nsteps] = phic
        # If adaptive, update proposal covariance for next step.
        # RAM algorithm by Vihola (https://doi.org/10.1007/s11222-011-9269-5)
        if adaptive:
            mhopt = 0.234  # Hard coded hyper-parameters for RAM
            gam = 2.0/3.0
            stepc = nsteps+1  # Step counter with 1-based indexing.
            eta = min(1, Np*stepc**(-gam))
            rinner = r@r
            router = np.outer(r, r)
            roi = router/rinner
            Cp = Sc@(Id+eta*(mh-mhopt)*roi)@(Sc.T)
            Sc = np.linalg.cholesky(Cp)

    # Clean tmp directory
    try:
        shutil.rmtree(os.path.split(temp_dest)[0], ignore_errors=True)
    except TypeError:
        pass
    printstr = 'mcmc done, adaptive=%s, acceptance rate=%4.2f' % (
        adaptive, (1.0*accepted)/nsteps)
    print(printstr)
    return accepted, mcmc_storage


def AI_mcmc(starting_parameters, predicted, observations_sbst_masked, R,
            chain_len, adaptive):

    vars_to_perturbate = cfg.vars_to_perturbate

    nll = negloglik(predicted, observations_sbst_masked, R)

    # train emulator
    gp_emul = gp_emulator(X_train=starting_parameters.T, y_train=nll)

    # MCMC parameters
    # Rinv = None
    sigp = 0.1
    Np = len(vars_to_perturbate)
    C0 = (sigp**2)*np.eye(Np)
    Sc = np.linalg.cholesky(C0)
    Id = np.eye(Np)

    SD0 = np.asarray([cnt.sd_errors[x] for x in vars_to_perturbate])
    m0 = np.asarray([cnt.mean_errors[x] for x in vars_to_perturbate])

    mcmc_storage = np.zeros((chain_len, len(vars_to_perturbate)))
    mcmc_storage[:] = np.nan

    # Init chain
    phic = np.reshape(starting_parameters.T[0, :],
                      (1, starting_parameters.shape[0]))
    nll, sd = gp_emul.predict(phic, return_std=True)

    Uc = neglogpost(nll, phic, SD0, m0)

    accepted = 0

    for nsteps in range(chain_len):

        r = np.random.randn(len(vars_to_perturbate))

        prop = Sc@r
        phip = phic+prop

        nll, sd = gp_emul.predict(phip, return_std=True)

        Up = neglogpost(nll, phip, SD0, m0)
        mh = min(1, np.exp(-Up+Uc))
        u = np.random.rand(1)
        accept = (mh > u)
        if accept:
            phic = phip
            Uc = Up
            accepted = accepted + 1

        mcmc_storage[nsteps] = phic

        # If adaptive, update proposal covariance for next step.
        # RAM algorithm by Vihola (https://doi.org/10.1007/s11222-011-9269-5)
        if adaptive:
            mhopt = 0.234  # Hard coded hyper-parameters for RAM
            gam = 2.0/3.0
            stepc = nsteps+1  # Step counter with 1-based indexing.
            eta = min(1, Np*stepc**(-gam))
            rinner = r@r
            router = np.outer(r, r)
            roi = router/rinner
            Cp = Sc@(Id+eta*(mh-mhopt)*roi)@(Sc.T)
            Sc = np.linalg.cholesky(Cp)

    printstr = 'mcmc done, adaptive=%s, acceptance rate=%4.2f' % (
        adaptive, (1.0*accepted)/nsteps)
    print(printstr)
    return accepted, mcmc_storage


def negloglik(predicted, observations_sbst_masked, R):

    if len(R.shape) == 1:
        R = R[:, np.newaxis]

    nres = (observations_sbst_masked - predicted)/np.sqrt(R)

    nll = 0.5 * np.sum(nres**2, axis=0)

    return nll


def neglogpost(nll, chain_par, SD0, m0):

    ndev = (chain_par - m0)/SD0

    nlpri = 0.5 * np.sum(ndev**2, axis=1)

    nlp = nll + nlpri

    return nlp


def gp_emulator(X_train, y_train):

    # kernel = 1 * krn.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    kernel = 1.0 * krn.Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                  n_restarts_optimizer=9, normalize_y=True)
    gp.fit(X_train, y_train)
    return gp


def bootstrapping(weights):

    weights = np.asarray(weights)

    # Sort weights and positions
    sorted_index = np.argsort(-weights)
    weights_sorted = -np.sort(-weights)

    w_sc = np.cumsum(weights_sorted)

    # Init results vector
    N = len(weights)
    indexes = np.empty(N, dtype=int)

    for j in range(N):

        u = np.random.rand()
        d = u - w_sc
        here = d == max(d[d < 0])

        # HACK: there is a small chance of len(here) > 2.
        # included np.random.choice to select randomlly if that happens
        indexes[j] = np.random.choice(sorted_index[here])

    return indexes


def residual_resample(weights):
    """
    Performs the residual resampling algorithm used by particle filters.
    Based on observation that we don't need to use random numbers to select
    most of the weights. Take int(N*w^i) samples of each particle i, and then
    resample any remaining using a standard resampling algorithm [1]
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    References
    ----------
    .. [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    """

    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is one
    indexes[k:N] = np.searchsorted(cumulative_sum, random(N-k))

    return indexes


def stratified_resample(weights):
    """
    Performs the stratified resampling algorithm used by particle filters.
    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """

    N = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights):
    """
    Performs the systemic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def resampled_indexes(weights):
    """
    Parameters
    ----------
    weights : TYPE
        DESCRIPTION.

    Returns
    -------
    indexes : TYPE
        DESCRIPTION.

    """

    resampling_algorithm = cfg.resampling_algorithm

    if resampling_algorithm == "stratified_resample":
        indexes = stratified_resample(weights)
    elif resampling_algorithm == "residual_resample":
        indexes = residual_resample(weights)
    elif resampling_algorithm == "bootstrapping":
        indexes = bootstrapping(weights)
    elif resampling_algorithm == "systematic_resample":
        indexes = systematic_resample(weights)
    elif resampling_algorithm == "no_resampling":
        indexes = np.arange(0, len(weights), 1, dtype=int)
    return indexes


def weighted_std(values, axis, weights):
    """
    Return the weighted  standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=axis, weights=weights)

    av_list = []
    for i in range(len(values)):
        av_list.append(average)

    # Fast and numerically precise:
    variance = np.average((np.asarray(values)-np.asarray(av_list))**2,
                          axis=axis, weights=weights)
    return np.sqrt(variance)


def get_predictions(list_state, var_to_assim):

    predicted = []

    for var in var_to_assim:
        assim_idx = model.get_var_state_position(var)

        predicted_tmp = [list_state[x].iloc[:, assim_idx].to_numpy()
                         for x in range(len(list_state))]
        predicted_tmp = np.asarray(predicted_tmp)
        # predicted_tmp = np.squeeze(predicted_tmp)

        predicted.append(predicted_tmp)

    return predicted


def tidy_obs_pred_rcov(predicted, observations_sbst, errors_sbst,
                       ret_mask=False):

    # tidy list of predictions
    predicted = np.concatenate(predicted.copy(), axis=1)

    # flaten obs and errors
    r_cov_f = errors_sbst.flatten(order='F')
    observations_sbst_f = observations_sbst.flatten(order='F')

    # create mask of nan
    mask = np.argwhere(~np.isnan(observations_sbst_f))

    # mask everithing
    observations_sbst_f_masked = observations_sbst_f[mask]
    r_cov_f = np.array(np.squeeze(r_cov_f[mask]))

    predicted = np.squeeze(predicted[:, mask])
    predicted = np.ndarray.transpose(predicted)

    if ret_mask:
        return observations_sbst_f_masked, predicted, r_cov_f, mask

    return observations_sbst_f_masked, predicted, r_cov_f


def tidy_predictions(predicted, mask):

    predicted_sf = predicted.copy()

    predicted_sf = np.concatenate(predicted_sf, axis=1)
    predicted_sf = np.squeeze(predicted_sf[:, mask])
    predicted_sf = np.ndarray.transpose(predicted_sf)

    return predicted_sf


def get_posterior_vars(list_state, var_to_assim, wgth):

    predicted = get_predictions(list_state, var_to_assim)
    posterior = []
    for n in range(len(var_to_assim)):

        posterior_tmp = np.vstack(predicted[n])
        posterior_tmp = np.average(posterior_tmp, axis=0, weights=wgth)
        posterior.append(posterior_tmp)

    return posterior


def get_OL_vars(Ensemble, var_to_assim):

    OL = []

    OL_state = Ensemble.origin_state.copy()

    for var in var_to_assim:

        assim_idx = model.get_var_state_position(var)
        OL_tmp = OL_state.iloc[:, assim_idx].to_numpy()

        OL.append(OL_tmp)

    return OL


def transform_space(parameters, trans_direction):

    safe_pars = parameters.copy()
    perturbation_strategy = cfg.perturbation_strategy
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds
    vars_to_perturbate = cfg.vars_to_perturbate

    if trans_direction == 'to_normal':
        # translate lognormal variables to normal distribution
        for cont, var in enumerate(perturbation_strategy):

            var_tmp = vars_to_perturbate[cont]

            if var == "lognormal":
                safe_pars[cont, :] = np.log(safe_pars[cont, :])

            elif var in ["logitnormal_mult",
                         "logitnormal_adi"]:

                safe_pars[cont, :] = met.glogit(safe_pars[cont, :],
                                                lower_bounds[var_tmp],
                                                upper_bounds[var_tmp])
            else:
                pass

    elif trans_direction == 'from_normal':

        for cont, var in enumerate(perturbation_strategy):
            var_tmp = vars_to_perturbate[cont]

            if var == "lognormal":
                safe_pars[cont, :] = np.exp(safe_pars[cont, :])

            elif var in ["logitnormal_mult",
                         "logitnormal_adi"]:

                safe_pars[cont, :] = met.gexpit(safe_pars[cont, :],
                                                lower_bounds[var_tmp],
                                                upper_bounds[var_tmp])
            else:
                pass

    else:
        raise Exception("transformation not found")

    return safe_pars


def get_parameters(Ensemble, j):

    param_arr = np.ones((len(cfg.vars_to_perturbate), Ensemble.members))

    for cont, var in enumerate(cfg.vars_to_perturbate):
        if j == 0:
            var_tmp = [Ensemble.noise[x][var]
                       for x in range(Ensemble.members)]
        else:
            var_tmp = [Ensemble.noise_iter[x][var]
                       for x in range(Ensemble.members)]
        var_tmp = np.asarray(var_tmp)
        var_tmp = np.squeeze(var_tmp)
        # HACK: next lines have to be modified with if time varying
        # perturbations are implemented
        # var_tmp = np.squeeze(var_tmp[:, mask])

        # Trick to handle the shape of the noise when there is an
        # observation in the first timestep
        if var_tmp.ndim == 1:
            param_arr[cont, :] = var_tmp
        else:
            if cfg.da_algorithm == "EnKF":
                param_arr[cont, :] = var_tmp[:, -1]
            else:  # da_algorithm == "IEnKF"
                param_arr[cont, :] = var_tmp[:, 0]
    return param_arr


def implement_assimilation(Ensemble, step):

    vars_to_perturbate = cfg.vars_to_perturbate
    mean_errors = cnt.mean_errors
    sd_errors = cnt.sd_errors
    da_algorithm = cfg.da_algorithm
    max_iterations = cfg.max_iterations
    var_to_assim = cfg.var_to_assim

    Result = {}  # initialice results dict

    list_state = Ensemble.state_membres
    errors = Ensemble.errors
    observations = Ensemble.observations

    # implement assimilation
    if da_algorithm == "PBS":
        # Check if there are observations to assim, or all weitgs = 1
        if np.isnan(Ensemble.observations).all():

            Ensemble.season_rejuvenation()
            pass

        else:

            predicted = get_predictions(list_state, var_to_assim)

            observations_sbst_masked, predicted, r_cov = \
                tidy_obs_pred_rcov(predicted, observations, errors)

            wgth, Neff = pbs(observations_sbst_masked, predicted, r_cov)

            if Neff/Ensemble.members < cnt.Neffthrs:
                print('Low Neff ({Neff}) found at cell: Lat:{lat}, Lon:{lon}'.
                      format(lat=Ensemble.lat_idx,
                             lon=Ensemble.lon_idx,
                             Neff=int(Neff)))
                Ensemble.lowNeff = True

            else:
                Ensemble.lowNeff = False

            Ensemble.wgth = wgth

            Ensemble.season_rejuvenation()

    elif da_algorithm == "PF":
        if np.isnan(Ensemble.observations).all():

            Result["resampled_particles"] = np.arange(Ensemble.members)

        else:

            predicted = get_predictions(list_state, var_to_assim)

            observations_sbst_masked, predicted, r_cov = \
                tidy_obs_pred_rcov(predicted, observations, errors)

            wgth, Neff = pbs(observations_sbst_masked, predicted, r_cov)

            if Neff/Ensemble.members < cnt.Neffthrs:
                print('Low Neff ({Neff}) found at cell: Lat:{lat}, Lon:{lon}'.
                      format(lat=Ensemble.lat_idx,
                             lon=Ensemble.lon_idx,
                             Neff=int(Neff)))
                Ensemble.lowNeff = True

            else:
                Ensemble.lowNeff = False

            Ensemble.wgth = wgth

            resampled_particles = resampled_indexes(wgth)

            Result["resampled_particles"] = resampled_particles

    elif da_algorithm == 'AdaPBS':
        # Check if there are observations to assim, or all weitgs = 1
        if np.isnan(Ensemble.observations).all():

            Ensemble.season_rejuvenation()
            pass

        else:

            priormean = np.zeros(len(vars_to_perturbate))
            priorsd = np.zeros(len(vars_to_perturbate))

            for count, var in enumerate(vars_to_perturbate):
                priormean[count] = mean_errors[var]
                priorsd[count] = sd_errors[var]
            priorcov = np.diag(priorsd**2)

            for j in range(max_iterations):

                predicted = get_predictions(
                    Ensemble.state_membres, var_to_assim)

                observations_sbst_masked, predicted, r_cov = \
                    tidy_obs_pred_rcov(predicted, observations, errors)

                proposal = get_parameters(Ensemble, j)
                proposal = transform_space(proposal, 'to_normal')

                wgth, Neff = ProPBS(observations_sbst_masked, predicted, r_cov,
                                    priormean, priorcov, proposal)
                print('Neff: {Neff} in j:{j}'.format(Neff=int(Neff),
                                                     j=j))

                # exit if not collapsed
                if (Neff/Ensemble.members > cnt.Neffthrs):
                    Ensemble.wgth = wgth
                    break

                resampled_particles = resampled_indexes(wgth)
                Ensemble.resample(resampled_particles, do_res=j != 0)

                # get resampled parameters
                thetaprop = get_parameters(Ensemble, j)
                # transform to normal space
                thetaprop = transform_space(thetaprop, 'to_normal')
                diversity = Neff/Ensemble.members
                # Calculate the mean vector of the proposed parameter ensemble
                thetapropm = np.mean(thetaprop, axis=1)
                # Calculate the covariance matrix of the proposed parameters
                thetapropA = (thetaprop.T - thetapropm).T
                thetapropc = (thetapropA@thetapropA.T) / Ensemble.members
                # Inflate the covariance slightly in case of degeneracy
                # thetapropc = thetapropc+0.1*(1-diversity)*priorcov
                thetapropc = thetapropc+(0.5**(8*j))*(1-diversity)*priorcov
                L = np.linalg.cholesky(thetapropc)

                # Draw from this Gaussian for the next iteration.
                thetaprop = thetapropm[:, np.newaxis] +\
                    L@np.random.standard_normal(size=(len(vars_to_perturbate),
                                                      Ensemble.members))
                thetaprop = transform_space(thetaprop, 'from_normal')
                Ensemble.iter_update(step, thetaprop,
                                     create=True, iteration=j)

            Ensemble.season_rejuvenation()

    elif da_algorithm == 'AdaMuPBS':
        # Check if there are observations to assim, or all weitgs = 1
        if np.isnan(Ensemble.observations).all():

            Ensemble.season_rejuvenation()
            pass

        else:

            priormean = np.zeros(len(vars_to_perturbate))
            priorsd = np.zeros(len(vars_to_perturbate))

            for count, var in enumerate(vars_to_perturbate):
                priormean[count] = mean_errors[var]
                priorsd[count] = sd_errors[var]
            priorcov = np.diag(priorsd**2)

            for j in range(max_iterations):

                predicted = get_predictions(
                    Ensemble.state_membres, var_to_assim)

                observations_sbst_masked, predicted, r_cov = \
                    tidy_obs_pred_rcov(predicted, observations, errors)

                proposal = get_parameters(Ensemble, j)
                proposal = transform_space(proposal, 'to_normal')

                if j == 0:
                    No = np.size(observations_sbst_masked)
                    Ne = Ensemble.members
                    Nl = max_iterations
                    Np = np.shape(proposal)[0]
                    predall = np.zeros([No, Ne, Nl])
                    predall[:] = np.nan
                    propsall = np.zeros([Np, Ne, Nl])
                    propsall[:] = np.nan
                    propmall = np.zeros([Np, Nl])
                    propmall[:] = np.nan
                    propmall[:, j] = priormean
                    propcall = np.zeros([Np, Np, Nl])
                    propcall[:] = np.nan
                    propcall[:, :, j] = priorcov
                    adapt_thresh = cnt.Neffthrs

                propsall[:, :, j] = proposal
                predall[:, :, j] = predicted
                ells = np.arange(j+1)
                obs = observations_sbst_masked.flatten()
                wgth, Neff = AMIS(obs, predall[:, :, ells],
                                  r_cov, priormean, priorcov,
                                  propmall[:, ells], propcall[:, :, ells],
                                  propsall[:, :, ells])

                print('Neff: {Neff} in j:{j}'.format(Neff=int(Neff),
                                                     j=j))

                diversity = Neff/Ne
                doadapt = diversity < adapt_thresh
                notlast = (j+1) < max_iterations
                w = wgth.flatten('F')

                # Can instead always set clip to 1 if you don't want to clip
                doclip=doadapt and notlast
                if doclip:
                    clip = int(np.round(adapt_thresh*Ne))
                    ws = -np.sort(-w)
                    wc = ws[clip-1]
                    nonzero=wc>0
                    if nonzero:
                        toclip=w>wc
                        w[toclip]=wc
                        w=w/np.sum(w)
                    else:
                        doclip=False
                        
                Nw = np.size(w)
                pinds = np.arange(Nw)
                reinds = np.random.choice(pinds, Ne, p=w)
                thetap = propsall[:, :, ells]
                thetap = np.reshape(thetap, [Np, Nw], order='F')
                thetap = thetap[:, reinds]
                pm = np.mean(thetap, axis=1)
                if doclip:
                    A = (thetap.T-pm).T
                    pc = (A@A.T)/Ne
                else:
                    pc=np.copy(priorcov)*(0.5**j)

                # Draw from this Gaussian for the next adaptive iteration
                # if there will be one
                if doadapt and notlast:
                    L = np.linalg.cholesky(pc)
                    Z = np.random.randn(Np, Ne)
                    thetap = (pm+(L@Z).T).T
                    propcall[:, :, j+1] = pc
                    propmall[:, j+1] = pm

                # Update parameters for next iteration (it is just
                # resampling if not adapt and/or last)

                thetap = transform_space(thetap, 'from_normal')
                Ensemble.iter_update(step, thetap, create=True, iteration=j)

                # exit if not collapsed
                if not doadapt:
                    break

            Ensemble.season_rejuvenation()

    elif da_algorithm in ["EnKF", 'IEnKF']:
        if da_algorithm == "EnKF":

            max_iterations = 1

        if np.isnan(Ensemble.observations).all():

            Ensemble.iter_update(create=False)
            pass

        else:
            for j in range(max_iterations):

                list_state = Ensemble.state_membres

                # Get ensemble of predictions
                predicted = get_predictions(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations,
                                           errors, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)

                # get parameters
                param_array = get_parameters(Ensemble, j)

                # translate lognormal variables to normal distribution
                param_array = transform_space(param_array, 'to_normal')

                alpha = max_iterations
                updated_pars = ens_klm(param_array, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                updated_pars = transform_space(updated_pars, 'from_normal')

                Ensemble.iter_update(step, updated_pars,
                                     create=True, iteration=j)

    elif da_algorithm in ['ES', 'IES']:

        if da_algorithm == "ES":

            max_iterations = 1

        if np.isnan(Ensemble.observations).all():

            Ensemble.iter_update(create=False)
            Ensemble.season_rejuvenation()
            pass

        else:
            for j in range(max_iterations):

                list_state = Ensemble.state_membres

                # Get ensemble of predictions
                predicted = get_predictions(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations,
                                           errors, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)

                # get parameters
                param_array = get_parameters(Ensemble, j)

                # translate lognormal variables to normal distribution
                param_array = transform_space(param_array, 'to_normal')

                alpha = max_iterations
                updated_pars = ens_klm(param_array, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                updated_pars = transform_space(updated_pars, 'from_normal')

                Ensemble.iter_update(step, updated_pars,
                                     create=True, iteration=j)
            Ensemble.season_rejuvenation()

    elif da_algorithm == 'PIES':
        if np.isnan(Ensemble.observations).all():

            Ensemble.iter_update(create=False)
            Ensemble.season_rejuvenation()
            Result["resampled_particles"] = np.arange(Ensemble.members)
            pass

        else:
            for j in range(max_iterations):

                list_state = Ensemble.state_membres

                # Get ensemble of predictions
                predicted = get_predictions(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations,
                                           errors, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)

                # get parameters
                param_array = get_parameters(Ensemble, j)

                # translate lognormal variables to normal distribution
                param_array = transform_space(param_array, 'to_normal')

                alpha = max_iterations
                updated_pars = ens_klm(param_array, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                updated_pars = transform_space(updated_pars, 'from_normal')

                Ensemble.iter_update(step, updated_pars,
                                     create=True, iteration=j)

            list_state = Ensemble.state_membres
            predicted = get_predictions(list_state, var_to_assim)
            predicted = tidy_predictions(predicted, mask)

            priormean = np.zeros(len(vars_to_perturbate))
            priorsd = np.zeros(len(vars_to_perturbate))

            for count, var in enumerate(vars_to_perturbate):
                priormean[count] = mean_errors[var]
                priorsd[count] = sd_errors[var]

            priorcov = np.diag(priorsd**2)
            proposal = transform_space(updated_pars, 'to_normal')

            wgth, Neff = ProPBS(observations_sbst_masked, predicted, r_cov,
                                priormean, priorcov, proposal)

            if Neff/Ensemble.members < cnt.Neffthrs:
                print('Low Neff ({Neff}) found at cell: Lat:{lat}, Lon:{lon}'.
                      format(lat=Ensemble.lat_idx,
                             lon=Ensemble.lon_idx,
                             Neff=int(Neff)))
                Ensemble.lowNeff = True

            else:
                Ensemble.lowNeff = False

            Ensemble.wgth = wgth
            Ensemble.season_rejuvenation()
            resampled_particles = resampled_indexes(wgth)
            Result["resampled_particles"] = resampled_particles

    elif da_algorithm in ['IES-MCMC']:
        if np.isnan(Ensemble.observations).all():

            Ensemble.iter_update(create=False)
            pass

        else:
            for j in range(max_iterations):

                list_state = Ensemble.state_membres

                # Get ensemble of predictions
                predicted = get_predictions(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations,
                                           errors, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)

                # get prior
                param_array = get_parameters(Ensemble, j)

                # Store parameters for gaussian process regresion

                Ensemble.store_train_data(param_array, predicted, j)

                # translate lognormal variables to normal distribution
                param_array = transform_space(param_array, 'to_normal')

                alpha = max_iterations
                updated_pars = ens_klm(param_array, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                updated_pars = transform_space(updated_pars, 'from_normal')

                Ensemble.iter_update(step, updated_pars,
                                     create=True, iteration=j)

            # Run the MCMC
            accepted, mcmc_storage = mcmc(Ensemble, observations_sbst_masked,
                                          r_cov, chain_len=cfg.chain_len,
                                          adaptive=cfg.adaptive,
                                          histcov=cfg.histcov)

            # Burn in:  discard the first samples
            ini = int(mcmc_storage.shape[0] * cfg.burn_in)
            end = mcmc_storage.shape[0]
            mcmc_storage = mcmc_storage[ini:end, :]

            # Sample n number of members
            idx = np.random.randint(mcmc_storage.shape[0],
                                    size=Ensemble.members)
            post_sample = mcmc_storage[idx, :].T

            # Create Ensemble from mcmc
            # translate to log space
            post_sample = transform_space(post_sample, 'from_normal')

            Ensemble.create_MCMC(post_sample, step)

            # Generate new parameters at the end of the season
            Ensemble.season_rejuvenation()

    elif da_algorithm in ['IES-MCMC_AI']:
        if np.isnan(Ensemble.observations).all():

            Ensemble.iter_update(create=False)
            pass

        else:
            for j in range(max_iterations):

                list_state = Ensemble.state_membres

                # Get ensemble of predictions
                predicted = get_predictions(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations,
                                           errors, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)

                # get prior
                param_array = get_parameters(Ensemble, j)

                # Store parameters for gaussian process regresion

                Ensemble.store_train_data(param_array, predicted, j)

                # translate lognormal variables to normal distribution
                param_array = transform_space(param_array, 'to_normal')

                alpha = max_iterations
                updated_pars = ens_klm(param_array, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                updated_pars = transform_space(updated_pars, 'from_normal')

                Ensemble.iter_update(step, updated_pars,
                                     create=True, iteration=j)

            # MCMC starting point
            list_state = Ensemble.state_membres

            # Get ensemble of predictions
            predicted = get_predictions(list_state, var_to_assim)
            predicted = tidy_predictions(predicted, mask)

            # get prior
            param_array = get_parameters(Ensemble, j)

            Ensemble.store_train_data(param_array, predicted, j+1)

            # Start MCMC
            starting_parameters = np.concatenate(Ensemble.train_parameters,
                                                 axis=1)
            predicted = np.concatenate(Ensemble.train_pred, axis=1)

            # translate to gaussian space
            starting_parameters = transform_space(starting_parameters,
                                                  'to_normal')

            # Run MCMC with gaussian emulator
            accepted, mcmc_storage = AI_mcmc(starting_parameters, predicted,
                                             observations_sbst_masked, r_cov,
                                             chain_len=cfg.chain_len,
                                             adaptive=cfg.adaptive)

            # Burn in:  discard the first samples

            ini = int(mcmc_storage.shape[0] * cfg.burn_in)
            end = mcmc_storage.shape[0]
            mcmc_storage = mcmc_storage[ini:end, :]

            # Sample n number of members
            idx = np.random.randint(mcmc_storage.shape[0],
                                    size=Ensemble.members)
            post_sample = mcmc_storage[idx, :].T

            # Create Ensemble from mcmc
            # translate to log space
            post_sample = transform_space(post_sample, 'from_normal')

            Ensemble.create_MCMC(post_sample, step)

            # Generate new parameters at the end of the season
            Ensemble.season_rejuvenation()

    else:
        raise Exception("Assim not implemented")

    # get perturbations

    for cont, var_p in enumerate(vars_to_perturbate):

        # Get perturbation parameters"""
        noise_ens_temp = [Ensemble.noise[x][var_p]
                          for x in range(len(Ensemble.noise))]
        noise_ens_temp = np.vstack(noise_ens_temp)

        noise_tmp_avg = np.average(noise_ens_temp, axis=0,
                                   weights=Ensemble.wgth)
        noise_tmp_sd = weighted_std(noise_ens_temp, axis=0,
                                    weights=Ensemble.wgth)

        Result[var_p + "_noise_mean"] = noise_tmp_avg
        Result[var_p + "_noise_sd"] = noise_tmp_sd

    return Result
