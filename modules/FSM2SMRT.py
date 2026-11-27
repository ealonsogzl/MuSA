#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import warnings
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
        Rgrn = statedataframe[["Rgrn1", "Rgrn2", "Rgrn3"]].values[
            row
        ]  # now layer grain radius
        Dsnw = statedataframe[["Dsnw1", "Dsnw2", "Dsnw3"]].values[
            row
        ]  # snow thikness
        lWE = statedataframe[["Sliq1", "Sliq2", "Sliq3"]].values[
            row
        ]  # liq water

        sWE = statedataframe[["Sice1", "Sice2", "Sice3"]].values[row]  # ice
        SWE = lWE + sWE

        with warnings.catch_warnings():  # This warning is necesary
            warnings.simplefilter("ignore")
            rhosnw = SWE / Dsnw  # density
        tsnow = statedataframe[["Tsnow1", "Tsnow2", "Tsnow3"]].values[
            row
        ]  # snot temperature

        # Remove empty layers y any
        Rgrn = Rgrn[~np.isnan(rhosnw)]
        Dsnw = Dsnw[~np.isnan(rhosnw)]
        lWE = lWE[~np.isnan(rhosnw)]
        sWE = sWE[~np.isnan(rhosnw)]
        SWE = SWE[~np.isnan(rhosnw)]
        tsnow = tsnow[~np.isnan(rhosnw)]
        rhosnw = rhosnw[~np.isnan(rhosnw)]

        # Soil
        Tsoil = statedataframe[
            ["Tsoil1", "Tsoil2", "Tsoil3", "Tsoil4"]
        ].values[
            row
        ]  # soil temperature
        Vsmc = statedataframe[["Vsmc1", "Vsmc2", "Vsmc3", "Vsmc4"]].values[
            row
        ]  # QUIARESTO DE LA SALIDA

        # prepare inputs
        thickness = list(Dsnw)
        corr_length = k * (4 / 3) * (1 - rhosnw / 917.0) * Rgrn
        temperature = tsnow
        density = rhosnw

        substrate = su.make_substrate(sensor, temperature=Tsoil[0])

        # SMRT requires the temperature of snow layer to be 273.15
        # or higher if there is liquid water in the snowpack. Since we are in
        # float 32 (FORTRAN) and python is float64 by default, this may cause
        # issues because of the tolerance of >

        temperature = temperature.astype("float64")
        temperature[lWE > 0.0] = 273.15
        temperature[temperature > 273.15] = 273.15
        # create the snowpack
        snowpack = make_snowpack(
            thickness=thickness,
            microstructure_model="exponential",
            density=density,
            temperature=temperature,
            # fraction: Sliq (kg/m^2) / 1000 is metres of water, per layer m
            volumetric_liquid_water=lWE / 1000.0 / Dsnw,
            corr_length=corr_length,
            substrate=substrate,
        )

        # create the model
        m = make_model("iba", "dort")
        # run the model
        with threadpool_limits_safe(limits=cfg.numpy_threads, user_api="blas"):
            result = m.run(sensor, snowpack, parallel_computation="none")
        
        # outputs, in the order of the column names        
        results_list.append(su.result_values(result, sensor))

    # store results
    results_df = pd.DataFrame(
        np.nan, index=range(len(statedataframe)), columns=names
    )
    for i, row in enumerate(rows):
        results_df.iloc[row] = results_list[i]

    # remove some columns from the state?
    statedataframe = statedataframe.join(results_df)

    return statedataframe


def return_col_names():

    return su.column_names(su.get_sensor())