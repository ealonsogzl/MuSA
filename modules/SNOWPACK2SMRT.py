#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

# import warnings
import pandas as pd
import numpy as np
import config as cfg
from modules.threading_utils import threadpool_limits_safe
from smrt import make_snowpack, make_model
import modules.smrt_utils as su


def run_SMRT(rows, statedataframe, k):

    results_list = []
    sensor = su.get_sensor()
    names = su.column_names(sensor)  # also without observation rows

    for row in rows:

        # Snowpack
        cols = statedataframe.filter(like="Rgrnm_layer").columns
        Rgrn = statedataframe[cols].values[row]  # now layer grain radius

        cols = statedataframe.filter(like="Dsnw_layer").columns
        Dsnw = statedataframe[cols].values[row]  # snow thikness

        cols = statedataframe.filter(like="lWE_layer").columns
        lWE = statedataframe[cols].values[row]  # liq water

        cols = statedataframe.filter(like="rhosnw_layer").columns
        rhosnw = statedataframe[cols].values[row]  # density

        cols = statedataframe.filter(like="Tsnow_layer").columns
        tsnow = statedataframe[cols].values[row]  # snow temperature

        # Remove empty layers y any
        Rgrnm = Rgrn[~np.isnan(rhosnw)]
        Dsnw = Dsnw[~np.isnan(rhosnw)]
        lWE = lWE[~np.isnan(rhosnw)]
        tsnow = tsnow[~np.isnan(rhosnw)]
        rhosnw = rhosnw[~np.isnan(rhosnw)]

        # prepare inputs
        # SNOWPACK gives the height of the top of each element above the
        # ground (bottom to top); SMRT wants thicknesses from top to bottom
        thickness = np.diff(Dsnw, prepend=0.0)[::-1]
        rhosnw, Rgrnm, lWE = rhosnw[::-1], Rgrnm[::-1], lWE[::-1]
        tsnow = tsnow[::-1]
        corr_length = k * (4 / 3) * (1 - rhosnw / 917.0) * Rgrnm
        temperature = tsnow
        density = rhosnw
        # Soil

        substrate = su.make_substrate(temperature=273.15)

        temperature[lWE > 0.0] = 273.15
        temperature[temperature > 273.15] = 273.15
        # create the snowpack
        snowpack = make_snowpack(
            thickness=thickness,
            microstructure_model="exponential",
            density=density,
            temperature=temperature,
            volumetric_liquid_water=lWE,  # fraction (/100 in snowpack_tools)
            corr_length=corr_length,
            substrate=substrate,
        )

        # create the model
        m = make_model("iba", "dort")
        # run the model
        with threadpool_limits_safe(limits=cfg.numpy_threads, user_api="blas"):
            result = m.run(sensor, snowpack, parallel_computation="none")

        # outputs, in the order of column names
        results_list.append(su.result_values(result, sensor))

    # store results
    # same index as the state (it starts at 1 after the first window)
    results_df = pd.DataFrame(
        np.nan, index=statedataframe.index, columns=names
    )
    for i, row in enumerate(rows):
        results_df.iloc[row] = results_list[i]

    statedataframe = statedataframe.join(results_df)

    cols = statedataframe.filter(like="layer").columns
    statedataframe = statedataframe.drop(columns=cols)

    return statedataframe


def return_col_names():

    return su.column_names(su.get_sensor())
