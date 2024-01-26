#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module can be used to modify the outputs of the numerical model used,
for example by calculating statistics or coupling other models.
In a real use case, functions will be included here that will add new 
variables to the output of the model.

Warning: VERY experimental, its use can be complex.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import numpy as np


def snd_ord(df, day_time=11, nigth_time=4):

    # NOCHE anterior MENOS DIA actual, timepo de analisis dia actual
    lst = df['Tsrf'].values
    Ampli = np.zeros_like(lst)
    Ampli[:] = np.nan

    idnigth = np.argwhere(df['hour'].values == nigth_time)
    idday = np.argwhere(df['hour'].values == day_time)

    simpli_lst = lst[int(idnigth[0]):int(idday[-1])]

    def shift(xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = np.nan
            e[n:] = xs[:-n]
        else:
            e[n:] = np.nan
            e[:n] = xs[-n:]
        return e
    # dia menos noche anterior
    Ampli_simpli = simpli_lst - shift(simpli_lst, day_time-nigth_time)
    Ampli[int(idnigth[0]):int(idday[-1])] = Ampli_simpli

    # Mask values that can not be solved
    Ampli[np.argwhere(np.isnan(Ampli))] = np.nanmean(Ampli)

    # Add to df
    df['Ampli'] = Ampli
    return df
