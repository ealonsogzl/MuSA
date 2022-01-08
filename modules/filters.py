#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description

Author: Esteban Alonso González - e.alonso@ipe.csic.es
Thu Oct  7 16:50:27 2021
"""

import numpy as np
from numpy.random import random
import pandas as pd
from scipy import special
import config as cfg
import constants as cnt
import modules.fsm_tools as fsm


def ens_klm(prior, obs, pred, alpha, r_cov):
    """
    EnKA: Implmentation of the Ensemble Kalman Analysis
    Inputs:
        prior: Prior ensemble matrix (n x N array)
        obs: Observation vector (m x 1 array)
        pred: Predicted observation ensemble matrix (m x N array)
        alpha: Observation error inflation coefficient (scalar)
        r_cov: Observation error covariance matrix (m x m array, m x 1 array,
                                                    or scalar)
    Outputs:
        post: Posterior ensemble matrix (n x N array)
    Dimensions:
        N is the number of ensemble members, n is the number of state
        variables and/orparameters, and m is the number of boservations.

    For now, we have impelemnted the classic (stochastic) version of the
    Ensemble Kalman analysis that involves perturbing the observations.
    Deterministic variants (Sakov and Oke, 2008) also exist that may be worth
    looking into. This analysis step can be used both for filtering, i.e.
    Ensemble Kalman filter, when observations are assimilated sequentially or
    smoothing, i.e. Ensemble (Kalman) smoother, when also be carried out in a
    multiple data assimilation (i.e. iterative) approach.

    Based on a previous version from K. Aalstad (10.12.2020)
    """
    # TODO: Using a Bessel correction (Ne-1 normalization) for sample
    # covariances.
    # TODO: Perturbing the predicted observations rather than the observations
    #      following van Leeuwen (2020, doi: 10.1002/qj.3819) which is
    #      the formally when defining the covariances matrices.
    # TODO: Looking into Sect. 4.5.3. in Evensen
    # (2019, doi: 10.1007/s10596-019-9819-z)
    # TODO: Using a deterministic rather than stochastic analysis step.
    # TODO: Augmenting with artificial momentum to get some inertia.
    # TODO: Better understand/build on the underlying tempering/annealing
    # idea in ES-MDA.
    # TODO: Versions that may scale better for large n such as those
    #       working in the ensemble subspace (see e.g. formulations
    #       in Evensen, 2003, Ocean Dynamics).

    # Dimensions.
    n_obs = np.size(obs)  # Number of obs
    n_state = np.shape(prior)[0]  # Tentative n of state vars and/or parameters

    if pred.ndim == 1:
        pred = np.expand_dims(pred, 1)
        pred = pred.T

    if np.size(prior) == n_state:  # If prior is 1D correct the above.
        n_ens_mem = n_state  # Number of ens. members.
        n_state = 1  # Number of state vars and/or parameters.
    else:  # If prior is 2D then the len of the second dim is the ensemble size
        n_ens_mem = np.shape(prior)[1]

    # Checks on the observation error covariance matrix.
    if np.size(r_cov) == 1:
        r_cov = r_cov * np.identity(n_obs)
    elif np.size(r_cov) == n_obs:
        r_cov = np.diag(r_cov)
        # Convert to matrix if specified as a vector.
    elif np.shape(r_cov)[0] == n_obs and np.shape(r_cov)[1] == n_obs:
        pass
        # Do nothing if specified as a matrix.
    else:
        raise Exception(
            'r_cov must be a scalar, m x 1 vector, or m x m matrix.')

    # Anomaly calculations.
    mprior = np.mean(prior, -1)  # Prior ensemble mean
    if n_state == 1:
        A = prior - mprior
    else:
        A = prior - mprior[:, None]
    mpred = np.mean(pred, -1)  # Prior predicted obs ensemble mean
    if n_obs == 1:
        B = pred-mpred
    else:
        B = pred-mpred[:, None]

    Bt = B.T  # Tranposed -"-

    # Perturbed observations.
    Y = np.outer(obs, np.ones(n_ens_mem)) + np.sqrt(alpha) * \
        np.random.randn(n_obs, n_ens_mem)
    # Y=np.outer(obs,np.ones(N))

    # Covariance matrices.
    C_AB = A @ Bt  # Prior-predicted obs covariance matrix mult by N (n x m)
    C_BB = B @ Bt  # Predicted obs covariance matrix mult by N (m x m)

    # Scaled observation error cov matrix (m x m)
    aR = (n_ens_mem * alpha) * r_cov

    # Analysis step
    if n_state == 1 and n_obs == 1:  # Scalar case
        K = C_AB * (np.linalg.inv(C_BB + aR))
        inno = Y - pred
        post = prior + K * inno
    else:
        K = C_AB@(np.linalg.inv(C_BB+aR))  # Kalman gain (n x m)
        inno = Y - pred  # Innovation (m x N)
        post = prior + K @ inno  # Posterior (n x N)

    return post


def pbs(obs, pred, r_cov):
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
    if np.size(r_cov) == 1:
        r_cov = r_cov * np.ones(n_obs)
    elif np.size(r_cov) == n_obs:
        pass
    else:
        raise Exception('r_cov must be a scalar, m x 1 vector.')

    # Residual and log-likelihood
    if n_obs == 1:
        residual = obs - pred
        llh = -0.5 * ((residual**2) * (1/r_cov))
    else:
        residual = obs - pred
        llh = -0.5 * ((1/r_cov)@(residual**2))

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

    return weights


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
        indexes = stratified_resample(weights)
    elif resampling_algorithm == "bootstrapping":
        indexes = bootstrapping(weights)
    elif resampling_algorithm == "systematic_resample":
        indexes = systematic_resample(weights)
    return indexes


def weighted_std(values, axis, weights):
    """
    Return the weighted average and standard deviation.

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


def get_predicitons(list_state, var_to_assim):

    predicted = []

    for var in var_to_assim:
        assim_idx = fsm.get_var_state_position(var)

        predicted_tmp = [list_state[x].iloc[:, assim_idx].to_numpy()
                         for x in range(len(list_state))]
        predicted_tmp = np.asarray(predicted_tmp)
        # predicted_tmp = np.squeeze(predicted_tmp)

        predicted.append(predicted_tmp)

    return predicted


def tidy_obs_pred_rcov(predicted, observations_sbst, r_cov, ret_mask=False):

    var_to_assim = cfg.var_to_assim

    # tidy list of predictions
    predicted = np.concatenate(predicted, axis=1)

    # check that there are the same error than vasr to assim
    if len(r_cov) != len(var_to_assim):
        raise Exception('Provide one error covariance value per var_to_assim')

    # expand errors to observation estructure
    r_cov_expand = np.tile(np.asarray(r_cov),
                           (np.shape(observations_sbst)[0], 1))

    # flaten obs and errors
    r_cov_expand_f = r_cov_expand.flatten(order='F')
    observations_sbst_f = observations_sbst.flatten(order='F')

    # create mask of nan
    mask = np.argwhere(~np.isnan(observations_sbst_f))

    # mask everithing
    observations_sbst_f_masked = observations_sbst_f[mask]
    r_cov_expand_f = np.squeeze(r_cov_expand_f[mask])

    predicted = np.squeeze(predicted[:, mask])
    predicted = np.ndarray.transpose(predicted)

    if ret_mask:
        return observations_sbst_f_masked, predicted, r_cov_expand_f, mask

    return observations_sbst_f_masked, predicted, r_cov_expand_f


def tidy_predictions(predicted, mask):
    predicted = np.concatenate(predicted, axis=1)
    predicted = np.squeeze(predicted[:, mask])
    predicted = np.ndarray.transpose(predicted)
    return predicted


def get_posterior_vars(list_state, var_to_assim, wgth):

    predicted = get_predicitons(list_state, var_to_assim)
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

        assim_idx = fsm.get_var_state_position(var)
        OL_tmp = OL_state.iloc[:, assim_idx].to_numpy()

        OL.append(OL_tmp)

    return OL


def implement_assimilation(Ensemble, observations_sbst,
                           step, forcing_sbst):
    """


    Parameters
    ----------
    Ensemble : Ensemble class
        An Ensemble of snow simulations.
    observations_sbst : array
        Array of observations to assimilate.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    vars_to_perturbate = cfg.vars_to_perturbate
    perturbation_strategy = cfg.perturbation_strategy
    filter_algorithm = cfg.filter_algorithm
    assimilation_strategy = cfg.assimilation_strategy
    Kalman_iterations = cfg.Kalman_iterations
    var_to_assim = cfg.var_to_assim
    r_cov = cfg.r_cov

    # Avoid to create ensemblestatistics if direct insertion
    if assimilation_strategy != "direct_insertion":

        list_state = Ensemble.state_membres

        # Retrieve ensemble SWE/SD mean, standar deviation and members

        SWE_ens = [list_state[x].iloc[:, 5].to_numpy()
                   for x in range(len(list_state))]
        SD_ens = [list_state[x].iloc[:, 4].to_numpy()
                  for x in range(len(list_state))]

        SD_ens = np.vstack(SD_ens)
        SWE_ens = np.vstack(SWE_ens)

        SWE_ens_mean = np.average(SWE_ens, axis=0)
        SD_ens_mean = np.average(SD_ens, axis=0)

        SWE_ens_sd = np.std(SWE_ens, axis=0)
        SD_ens_sd = np.std(SD_ens, axis=0)

    # implement assimilation
    if assimilation_strategy == "smoothing" and filter_algorithm == "PBS":
        # Check if there are observations to assim, or all weitgs = 1
        if np.isnan(observations_sbst).all():

            wgth = np.ones(len(list_state))

            SWE_assim_mean = np.average(SWE_ens, axis=0, weights=wgth)
            SD_assim_mean = np.average(SD_ens, axis=0, weights=wgth)

            SWE_assim_sd = weighted_std(SWE_ens, axis=0, weights=wgth)
            SD_assim_sd = weighted_std(SD_ens, axis=0, weights=wgth)

            # Get openloop, and posterior assimilated vars

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}
        else:

            predicted = get_predicitons(list_state, var_to_assim)

            observations_sbst_masked, predicted, r_cov = \
                tidy_obs_pred_rcov(predicted, observations_sbst, r_cov)

            wgth = pbs(observations_sbst_masked, predicted, r_cov)

            SWE_assim_mean = np.average(SWE_ens, axis=0, weights=wgth)
            SD_assim_mean = np.average(SD_ens, axis=0, weights=wgth)

            SWE_assim_sd = weighted_std(SWE_ens, axis=0, weights=wgth)
            SD_assim_sd = weighted_std(SD_ens, axis=0, weights=wgth)

            # Get openloop, and posterior assimilated vars

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}

    elif assimilation_strategy == "filtering" and filter_algorithm == "PBS":
        if np.isnan(observations_sbst).all():

            wgth = np.ones(len(list_state))

            SWE_assim_mean = np.average(SWE_ens, axis=0, weights=wgth)
            SD_assim_mean = np.average(SD_ens, axis=0, weights=wgth)

            SWE_assim_sd = weighted_std(SWE_ens, axis=0, weights=wgth)
            SD_assim_sd = weighted_std(SD_ens, axis=0, weights=wgth)

            if cfg.redraw_prior:
                Ensemble.wgth = wgth

            # Get openloop, and posterior assimilated vars

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}
        else:

            predicted = get_predicitons(list_state, var_to_assim)

            observations_sbst_masked, predicted, r_cov = \
                tidy_obs_pred_rcov(predicted, observations_sbst, r_cov)

            wgth = pbs(observations_sbst_masked, predicted, r_cov)

            SWE_assim_mean = np.average(SWE_ens, axis=0, weights=wgth)
            SD_assim_mean = np.average(SD_ens, axis=0, weights=wgth)

            SWE_assim_sd = weighted_std(SWE_ens, axis=0, weights=wgth)
            SD_assim_sd = weighted_std(SD_ens, axis=0, weights=wgth)

            resampled_particles = resampled_indexes(wgth)

            if cfg.redraw_prior:
                Ensemble.wgth = wgth

            # Get openloop, and posterior assimilated vars
            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars,
                      "resampled_particles": resampled_particles}

    elif assimilation_strategy == "filtering" and filter_algorithm == "Kalman":
        if np.isnan(observations_sbst).all():

            wgth = np.ones(len(list_state))

            SWE_assim_mean = np.average(SWE_ens, axis=0, weights=wgth)
            SD_assim_mean = np.average(SD_ens, axis=0, weights=wgth)

            SWE_assim_sd = weighted_std(SWE_ens, axis=0, weights=wgth)
            SD_assim_sd = weighted_std(SD_ens, axis=0, weights=wgth)

            Ensemble.kalman_update(forcing_sbst, step, updated_pars=None,
                                   create=False, iteration=None)

            list_state = Ensemble.state_membres

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}
        else:
            for j in range(Kalman_iterations):

                list_state = Ensemble.state_membres

                wgth = np.ones(len(list_state))

                # Get ensemble of predictions
                predicted = get_predicitons(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations_sbst,
                                           r_cov, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)
                # get prior
                prior = np.ones((len(vars_to_perturbate), len(list_state)))
                for cont, var in enumerate(vars_to_perturbate):
                    if j == 0:
                        var_tmp = [Ensemble.noise[x][var]
                                   for x in range(len(list_state))]
                    else:
                        var_tmp = [Ensemble.noise_kalman[x][var]
                                   for x in range(len(list_state))]
                    var_tmp = np.asarray(var_tmp)
                    var_tmp = np.squeeze(var_tmp)
                    var_tmp = np.squeeze(var_tmp[:, mask])
                    prior[cont, :] = var_tmp

                # translate lognormal variables to normal distribution
                for cont, var in enumerate(perturbation_strategy):
                    if var in ["constant_lognormal", "time_varing_lognormal",
                               "time_dcor_lognormal"]:
                        prior[cont, :] = np.log(prior[cont, :])

                alpha = Kalman_iterations
                updated_pars = ens_klm(prior, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                for cont, var in enumerate(perturbation_strategy):
                    if var in ["constant_lognormal", "time_varing_lognormal",
                               "time_dcor_lognormal"]:
                        updated_pars[cont, :] = np.exp(updated_pars[cont, :])

                Ensemble.kalman_update(forcing_sbst, step, updated_pars,
                                       create=True, iteration=j)

            # Retrieve the updated SWE and SD
            # get new list state
            list_state = Ensemble.state_membres

            SWE_ens_updated = [list_state[x].iloc[:, 5].to_numpy()
                               for x in range(len(list_state))]
            SD_ens_updated = [list_state[x].iloc[:, 4].to_numpy()
                              for x in range(len(list_state))]

            SD_ens_updated = np.vstack(SD_ens_updated)
            SWE_ens_updated = np.vstack(SWE_ens_updated)

            SWE_assim_mean = np.average(SWE_ens_updated, axis=0)
            SD_assim_mean = np.average(SD_ens_updated, axis=0)

            SWE_assim_sd = np.std(SWE_ens_updated, axis=0)
            SD_assim_sd = np.std(SD_ens_updated, axis=0)

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}

    elif assimilation_strategy == "smoothing" and filter_algorithm == "Kalman":

        if np.isnan(observations_sbst).all():

            wgth = np.ones(len(list_state))

            SWE_assim_mean = np.average(SWE_ens, axis=0, weights=wgth)
            SD_assim_mean = np.average(SD_ens, axis=0, weights=wgth)

            SWE_assim_sd = weighted_std(SWE_ens, axis=0, weights=wgth)
            SD_assim_sd = weighted_std(SD_ens, axis=0, weights=wgth)

            Ensemble.kalman_update(forcing_sbst, step, updated_pars=None,
                                   create=False, iteration=None)

            list_state = Ensemble.state_membres

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}
        else:
            for j in range(Kalman_iterations):

                list_state = Ensemble.state_membres

                wgth = np.ones(len(list_state))

                # Get ensemble of predictions
                predicted = get_predicitons(list_state, var_to_assim)

                if j == 0:
                    observations_sbst_masked, predicted, r_cov, mask = \
                        tidy_obs_pred_rcov(predicted, observations_sbst,
                                           r_cov, ret_mask=True)
                else:
                    predicted = tidy_predictions(predicted, mask)

                # get prior
                prior = np.ones((len(vars_to_perturbate), len(list_state)))
                for cont, var in enumerate(vars_to_perturbate):
                    if j == 0:
                        var_tmp = [Ensemble.noise[x][var]
                                   for x in range(len(list_state))]
                    else:
                        var_tmp = [Ensemble.noise_kalman[x][var]
                                   for x in range(len(list_state))]
                    var_tmp = np.asarray(var_tmp)
                    var_tmp = np.squeeze(var_tmp)
                    var_tmp = np.squeeze(var_tmp[:, mask])
                    prior[cont, :] = var_tmp[:, 0]

                # translate lognormal variables to normal distribution
                for cont, var in enumerate(perturbation_strategy):
                    if var in ["constant_lognormal", "time_varing_lognormal",
                               "time_dcor_lognormal"]:
                        prior[cont, :] = np.log(prior[cont, :])

                alpha = Kalman_iterations
                updated_pars = ens_klm(prior, observations_sbst_masked,
                                       predicted, alpha, r_cov)

                for cont, var in enumerate(perturbation_strategy):
                    if var in ["constant_lognormal", "time_varing_lognormal",
                               "time_dcor_lognormal"]:
                        updated_pars[cont, :] = np.exp(updated_pars[cont, :])

                Ensemble.kalman_update(forcing_sbst, step, updated_pars,
                                       create=True, iteration=j)

            # Retrieve the updated SWE and SD
            # get new list state
            list_state = Ensemble.state_membres

            SWE_ens_updated = [list_state[x].iloc[:, 5].to_numpy()
                               for x in range(len(list_state))]
            SD_ens_updated = [list_state[x].iloc[:, 4].to_numpy()
                              for x in range(len(list_state))]

            SD_ens_updated = np.vstack(SD_ens_updated)
            SWE_ens_updated = np.vstack(SWE_ens_updated)

            SWE_assim_mean = np.average(SWE_ens_updated, axis=0)
            SD_assim_mean = np.average(SD_ens_updated, axis=0)

            SWE_assim_sd = np.std(SWE_ens_updated, axis=0)
            SD_assim_sd = np.std(SD_ens_updated, axis=0)

            post_vars = get_posterior_vars(list_state, var_to_assim, wgth)

            Result = {"SWE_ens_mean": SWE_ens_mean,
                      "SD_ens_mean": SD_ens_mean,
                      "SWE_ens_sd": SWE_ens_sd,
                      "SD_ens_sd": SD_ens_sd,
                      "SWE_assim_mean": SWE_assim_mean,
                      "SD_assim_mean": SD_assim_mean,
                      "SWE_assim_sd": SWE_assim_sd,
                      "SD_assim_sd": SD_assim_sd,
                      "post_vars": post_vars}

    elif assimilation_strategy == "direct_insertion":
        # raise Exception("direct_insertion not implemented yet")

        if np.isnan(observations_sbst).all():
            # If there are not observations do nothing
            return None

        # Remove nan from obs and predict
        mask = np.argwhere(~np.isnan(observations_sbst))
        observations_sbst_masked = observations_sbst[mask]

        # Point to last init file
        dump_to_insert = Ensemble.origin_dump[len(Ensemble.origin_dump)-1]

        # get layers from sim and obser
        n, l1, l2, l3 = fsm.get_layers(observations_sbst_masked[0])
        n_sim = dump_to_insert.iloc[2, 0]

        if (n == n_sim):  # Equal number of layers
            # Update mass, keeping density constant
            # ice content
            ice_dens = dump_to_insert.iloc[5] / dump_to_insert.iloc[1]
            new_icecont = ice_dens * pd.Series([l1, l2, l3, float("NaN")])
            dump_to_insert.iloc[5] = new_icecont.fillna(0)

            # Water content
            liquid_dens = dump_to_insert.iloc[6] / dump_to_insert.iloc[1]
            new_liqcont = liquid_dens * pd.Series([l1, l2, l3, float("NaN")])
            dump_to_insert.iloc[6] = new_liqcont.fillna(0)

            # Insert snowdepth in last init file
            dump_to_insert.at[1, 0] = l1
            dump_to_insert.at[1, 1] = l2
            dump_to_insert.at[1, 2] = l3

            return None

        else:   # if the number of layers is different.
            # Guess mass and temperature, insert layers
            # TODO : try to retrieve density and temperature from simulation
            # if any layers exist instead of overwrite with fixed values
            # TODO : Think on the albedo, if there is no snow in the simulation
            # The albedo of the direct insertion is too low

            liquid_dens = cnt.rSNOW * cnt.lSNOW
            new_liqcont = liquid_dens * pd.Series([l1, l2, l3, float("NaN")])
            dump_to_insert.iloc[6] = new_liqcont.fillna(0)

            ice_dens = cnt.rSNOW - liquid_dens
            new_icecont = ice_dens * pd.Series([l1, l2, l3, float("NaN")])
            dump_to_insert.iloc[5] = new_icecont.fillna(0)

            # Insert snowdepth in last init file
            dump_to_insert.at[1, 0] = l1
            dump_to_insert.at[1, 1] = l2
            dump_to_insert.at[1, 2] = l3

            # Insert snow temp in last init file
            dump_to_insert.at[9, 0] = cnt.tSNOW
            dump_to_insert.at[9, 1] = cnt.tSNOW
            dump_to_insert.at[9, 2] = cnt.tSNOW

            # Insert snow/ground sfc temp
            dump_to_insert.at[11, 0] = cnt.sfcTEMP

            # Insert number of layers
            dump_to_insert.at[2, 0] = n

            # Insert snow grain radius
            dump_to_insert.at[4, 0] = cnt.grRADI
            dump_to_insert.at[4, 1] = cnt.grRADI
            dump_to_insert.at[4, 2] = cnt.grRADI

            return None

    else:
        raise Exception("Assim not implemented")

    # Adding perturbing information to results
    for cont, var_p in enumerate(vars_to_perturbate):

        # Get perturbation parameters"""
        noise_ens_temp = [Ensemble.noise[x][var_p]
                          for x in range(len(Ensemble.noise))]
        noise_ens_temp = np.vstack(noise_ens_temp)

        noise_tmp_avg = np.average(noise_ens_temp, axis=0, weights=wgth)
        noise_tmp_sd = weighted_std(noise_ens_temp, axis=0, weights=wgth)

        Result[var_p + "_noise_mean"] = noise_tmp_avg
        Result[var_p + "_noise_sd"] = noise_tmp_sd

    return Result
