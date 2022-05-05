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
rSNOW = 300                     # Guessed Snow bulk density [kg m**-3]
tSNOW = 273                     # Guessed Snow temperature [K]
lSNOW = 0.1                     # Guessed Snow liquid content [-]
sfcTEMP = 273                   # Guessed Snow/ground sfc temp [K]
grRADI = 0.00005                # Guessed grain radius [m]
Hfsn = 0.1                      # fSCA depth scale [m] (linear/asymptotic fSCA)
SWEsca = 13                     # SWE where SCA=1 [mm] (Noah fSCA)
Taf = 4                         # fSCA shape parameter [-] (Noah fSCA)
SCA0 = 0.25                     # fSCA threshold for SCA [-]
sdfrac = 0.1                    # fraction of the sd_errors to use if collapse
# -----------------------------------
# Mean errors
# -----------------------------------

mean_errors = {"SW": 0,
               "LW": 0,
               "Prec": 0,
               "Ta": 0,
               "RH": 0,
               "Ua": -0.14,
               "Ps": 0}


# -----------------------------------
# Standar deviation errors
# -----------------------------------

sd_errors = {"SW": 0.1,
             "LW": 20.8,
             "Prec": 0.63,
             "Ta": 1,
             "RH": 8.9,
             "Ua": 0.53,
             "Ps": 100}


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

