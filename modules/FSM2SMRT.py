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

        substrate = make_soil(
            "soil_wegmuller",
            "soil_permittivity_dobson85_peplinski95",
            temperature=Tsoil[0],
            moisture=0.20435,
            sand=0.6,
            clay=0.3,
            drymatter=1100,
            roughness_rms=5e-3,
        )

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
        with threadpool_limits(limits=cfg.smrt_cores, user_api="blas"):
            result = m.run(radiometer, snowpack)

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

    # remove some columns from the state?
    statedataframe = statedataframe.join(results_df)

    return statedataframe


def return_col_names():

    # prepare inputs
    thickness = [200]
    corr_length = [5e-5]
    temperature = [270]
    density = [420]

    # create the snowpack
    snowpack = make_snowpack(
        thickness=thickness,
        microstructure_model="exponential",
        density=density,
        temperature=temperature,
        corr_length=corr_length,
    )
    # create the sensor
    radiometer_str = "sensor_list.{cfgsensorlist}".format(
        cfgsensorlist=cfg.SMRT_sensor_list
    )
    radiometer = eval(radiometer_str)
    # create the model
    m = make_model("iba", "dort")
    # run the model
    result = m.run(radiometer, snowpack)

    # outputs
    result = result.Tb()

    names = [
        f"{int(round(f/1e9))}{p}_i{int(round(t))}"
        for f in result.frequency.values
        for t in result.theta.values
        for p in result.polarization.values
    ]

    return names
