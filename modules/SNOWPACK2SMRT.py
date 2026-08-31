#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import warnings
import pandas as pd
import numpy as np
import config as cfg
from threadpoolctl import threadpool_limits
from smrt import make_snowpack, make_model, make_soil, sensor_list


def run_SMRT(rows, statedataframe, k):

    results_list = []

    for row in rows:

        # Snowpack
        cols = statedataframe.filter(like="Rgrn").columns
        Rgrn = statedataframe[cols].values[row]  # now layer grain radius

        cols = statedataframe.filter(like="Dsnw").columns
        Dsnw = statedataframe[cols].values[row]  # snow thikness

        cols = statedataframe.filter(like="lWE").columns
        lWE = statedataframe[cols].values[row]  # liq water

        cols = statedataframe.filter(like="rhosnw").columns
        rhosnw = statedataframe[cols].values[row]  # density

        cols = statedataframe.filter(like="Tsnow").columns
        tsnow = statedataframe[cols].values[row]  # snow temperature

        # Remove empty layers y any
        Rgrn = Rgrn[~np.isnan(rhosnw)]
        Dsnw = Dsnw[~np.isnan(rhosnw)]
        lWE = lWE[~np.isnan(rhosnw)]
        tsnow = tsnow[~np.isnan(rhosnw)]
        rhosnw = rhosnw[~np.isnan(rhosnw)]

        # prepare inputs

        thickness = Dsnw
        corr_length = k * (4 / 3) * (1 - rhosnw / 917.0) * Rgrn
        temperature = tsnow
        density = rhosnw
        # Soil

        substrate = make_soil(
            "soil_wegmuller",
            "soil_permittivity_dobson85_peplinski95",
            temperature=273.15,
            moisture=0.20435,
            sand=0.6,
            clay=0.3,
            drymatter=1100,
            roughness_rms=5e-3,
        )

        temperature[lWE > 0.0] = 273.15
        temperature[temperature > 273.15] = 273.15
        # create the snowpack
        snowpack = make_snowpack(
            thickness=thickness,
            microstructure_model="exponential",
            density=density,
            temperature=temperature,
            volumetric_liquid_water=lWE / 1000.0,
            corr_length=corr_length,
            substrate=substrate,
        )

        # create the sensor
        radiometer_str = "sensor_list.{cfgsensorlist}".format(
            cfgsensorlist=cfg.SMRT_sensor_list
        )
        radiometer = eval(radiometer_str)
        # create the model
        m = make_model("iba", "dort")
        # run the model
        with threadpool_limits(limits=cfg.numpy_threads, user_api="blas"):
            result = m.run(radiometer, snowpack, parallel_computation="none")

        # outputs
        result = result.Tb()

        # Flatten the array
        flat_values = result.values.flatten()

        # Convert to numpy arrays
        flat_values = np.array(flat_values)
        names = [
            f"{int(round(f/1e9))}{p}_i{int(round(t))}"
            for f in result.frequency.values
            for t in result.theta.values
            for p in result.polarization.values
        ]

        results_list.append(flat_values)

    # store results
    results_df = pd.DataFrame(
        np.nan, index=range(len(statedataframe)), columns=names
    )
    for i, row in enumerate(rows):
        results_df.iloc[row] = results_list[i]

    statedataframe = statedataframe.join(results_df)

    smrt_names = return_col_names()
    if cfg.DAsord:
        model_columns = (
            "snd",
            "SWE",
            *smrt_names,
            *cfg.DAord_names,
        )

    else:
        model_columns = (
            "snd",
            "SWE",
            *smrt_names,
        )

    statedataframe = statedataframe[list(model_columns)]  # remove some columns

    return statedataframe


def return_col_names():

    # create the sensor
    radiometer_str = "sensor_list.{cfgsensorlist}".format(
        cfgsensorlist=cfg.SMRT_sensor_list
    )
    radiometer = eval(radiometer_str)

    names = [
        f"{int(round(f/1e9))}{p}_i{int(round(t))}"
        for f in radiometer.frequency
        for t in radiometer.theta_deg
        for p in radiometer.polarization
    ]

    return names
