#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple example of how to display the results of a cell

@author: Esteban Alonso González - alonsoe@ipe.csic.es
"""
import numpy as np
import modules.internal_fns as ifn
import matplotlib.pyplot as plt

# name of the file to be visualized
name = './DATA/RESULTS/cell_0_2.pkl.blp'


def main(name):
    output = ifn.io_read(name)

    obs = output['DA_Results']
    ol = output['OL_Sim']
    sd = output['std_Post']
    updated = output['mean_Post']
    prior_mean = output['mean_Prior']
    prior_sd = output['std_Prior']
    try:
        sdmcmc = output['mcmcSD_Sim']
        updatedmcmc = output['mcmc_Sim']
    except Exception:
        pass

    # Get obs positions
    x = np.where(~np.isnan(obs['snd']))
    y = obs['snd'][~np.isnan(obs['snd'])]
    ids = np.linspace(1, obs.shape[0], obs.shape[0])

    # get and truncate posterior & prior lower marging
    minimun = updated['snd'] - sd['snd']
    minimun[minimun < 0] = 0
    minimun_prior = prior_mean['snd'] - prior_sd['snd']
    minimun_prior[minimun_prior < 0] = 0

    try:
        minimun_mcmc = updatedmcmc['snd'] - sdmcmc['snd']
        minimun_mcmc[minimun_mcmc < 0] = 0
    except Exception:
        pass

    plt.figure(figsize=(7, 2), dpi=600)
    plt.ylabel("Snow depth [m]")

    plt.fill_between(ids,
                     minimun_prior,
                     prior_mean['snd'] + prior_sd['snd'],
                     color='lightsalmon')
    plt.plot(minimun_prior,
             color="tomato", lw=0.5, linestyle='dashed')
    plt.plot(prior_mean['snd'] + prior_sd['snd'],
             color="tomato", lw=0.5, linestyle='dashed')
    plt.plot(prior_mean['snd'], color="red", lw=1,
             label="Open-loop")

    plt.fill_between(ids,
                     minimun,
                     updated['snd'] + sd['snd'],
                     color='mediumaquamarine')
    plt.plot(minimun,
             color="seagreen", lw=0.5, linestyle='dashed')
    plt.plot(updated['snd'] + sd['snd'],
             color="seagreen", lw=0.5, linestyle='dashed')
    plt.plot(updated['snd'], color="darkgreen", lw=1,
             label="Updated")

    try:
        plt.fill_between(ids,
                         minimun_mcmc,
                         updatedmcmc['snd'] + sdmcmc['snd'],
                         color='lightskyblue')
        plt.plot(updatedmcmc['snd'] + sdmcmc['snd'],
                 color="dodgerblue", lw=0.5, linestyle='dashed')
        plt.plot(minimun_mcmc,
                 color="dodgerblue", lw=0.5, linestyle='dashed')
        plt.plot(updatedmcmc['snd'], color="dodgerblue", lw=1,
                 label="Updated-MCMC")
    except Exception:
        pass

    plt.plot(ol['snd'], color="black", lw=0.8,
             label="Reference run")
    plt.scatter(x, y, color="cornflowerblue", edgecolors='black', s=9,
                label="Observations")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


if __name__ == "__main__":
    main(name)
