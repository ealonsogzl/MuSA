#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple example of how to display the results of a cell

@author: Esteban Alonso González - alonsoe@cesbio.cnes.fr
"""
import numpy as np
import modules.internal_fns as ifn
import matplotlib.pyplot as plt


output = ifn.io_read('./DATA/RESULTS/cell_0_2.pkl.blp')

obs = output['DA_Results']
ol = output['OL_Sim']
sd = output['sd_Sim']
updated = output['updated_Sim']
prior_mean = output['prior_mean']
prior_sd = output['prior_sd']

# Get obs positions
x = np.where(~np.isnan(obs['snd']))
y = obs['snd'][~np.isnan(obs['snd'])]
ids = np.linspace(1, obs.shape[0], obs.shape[0])

# get and truncate posterior & prior lower marging
minimun = updated['snd'] - sd['snd']
minimun[minimun < 0] = 0
minimun_prior = prior_mean['snd'] - prior_sd['snd']
minimun_prior[minimun_prior < 0] = 0


plt.figure(figsize=(7, 2), dpi=600)
plt.ylabel("Snow depth [m]")
props = dict(boxstyle='round', facecolor='white')
ax = plt.gca()

plt.fill_between(ids,
                 minimun_prior,
                 prior_mean['snd'] + prior_sd['snd'],
                 color='lightsalmon')
plt.fill_between(ids,
                 minimun,
                 updated['snd'] + sd['snd'],
                 color='mediumaquamarine')

plt.plot(minimun_prior,
         color="tomato", lw=0.5, linestyle='dashed')
plt.plot(prior_mean['snd'] + prior_sd['snd'],
         color="tomato", lw=0.5, linestyle='dashed')

plt.plot(minimun,
         color="seagreen", lw=0.5, linestyle='dashed')
plt.plot(updated['snd'] + sd['snd'],
         color="seagreen", lw=0.5, linestyle='dashed')
plt.plot(updated['snd'], color="darkgreen", lw=1,
         label="Updated")

plt.plot(prior_mean['snd'], color="red", lw=1,
         label="Open-loop")

plt.plot(ol['snd'], color="black", lw=0.8,
         label="Reference run")
plt.scatter(x, y, color="cornflowerblue", edgecolors='black', s=9,
            label="Observations")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
