#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants declarations

"""

# -----------------------------------
# Physical constants
# -----------------------------------

KELVING_CONVER = 273.15         # Kelvin to Celsius conversion
R = 8.31441                     # Universal Gas Constant [J mol**-1 K**-1]
MW = 0.01801528                 # Molecular weight of water [kg mol**-1]
LF = 334000                     # latent heat of fusion [J kg**-1]
SCA0 = 0.25                     # fSCA threshold for SCA [-]
sdfrac = 0.7                    # fraction of the sd_errors to use if collapse
Neffthrs = 0.1                  # Low Neff threshold
DMF = 3/24                      # Degree melt index
FIX_density = 0.3
aprox_lat = 50

# -----------------------------------
# Default fSCA parameters
# -----------------------------------
# SWE threshold where SCA = 1 (SWEsca) [SNFRAC = 3]
SWEsca = 40
# shape of the fSCA. [SNFRAC = 3]
Taf = 4.0
# coefficient of variation for the subgrid snow variation [SNFRAC = 4]
subgrid_cv = 2.0

# -----------------------------------
# Default vegetation characteristics
# -----------------------------------
VAI = 0    # Vegetation area index
vegh = 0   # Canopy height (m)
fsky = 1   # Sky view fraction for remote shading
hbas = 2   # Canopy base height

# -----------------------------------
# Mean errors
# -----------------------------------

mean_errors = {"SW": 0,
               "LW": 0,
               "Prec": -1.6,
               "Ta": 0,
               "RH": 0,
               "Ua": -0.14,
               "Ps": 0}


# -----------------------------------
# Standar deviation errors
# -----------------------------------

sd_errors = {"SW": 0.1,
             "LW": 20.8,
             "Prec": 1,
             "Ta": 0.5,
             "RH": 8.9,
             "Ua": 0.53,
             "Ps": 100}

# -----------------------------------
# upper bounds errors
# -----------------------------------

upper_bounds = {"SW": 10,
                "LW": 10,
                "Prec": 8,
                "Ta": 8,
                "RH": 10,
                "Ua": 10,
                "Ps": 10}

# -----------------------------------
# Lower bounds errors
# -----------------------------------

lower_bounds = {"SW": -10,
                "LW": -10,
                "Prec": 0,
                "Ta": -8,
                "RH": -10,
                "Ua": -10,
                "Ps": -10}


# -----------------------------------
# Dynamic noise
# -----------------------------------
dyn_noise = {"SW": 0.01,
             "LW": 0.01,
             "Prec": 0.01,
             "Ta": 0.01,
             "RH": 0.01,
             "Ua": 0.01,
             "Ps": 0.01}


# -----------------------------------
# Unit conversions
# -----------------------------------
forcing_offset = {"SW": 0,
                  "LW": 0,
                  "Prec": 0,
                  "Ta": 0,
                  "RH": 0,
                  "Ua": 0,
                  "Ps": 0}

forcing_multiplier = {"SW": 1,
                      "LW": 1,
                      "Prec": 1,
                      "Ta": 1,
                      "RH": 1,
                      "Ua": 1,
                      "Ps": 1}
