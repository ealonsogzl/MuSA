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
DMF = 3/24                      # Degree melt index [dIm model]
FIX_density = 0.3               # Fixed snow density [dIm model]
aprox_lat = 50

# -----------------------------------
# Default FSM2 vegetation characteristics
# -----------------------------------
VAI = 0    # Vegetation area index
vegh = 0   # Canopy height (m)
fsky = 1   # Sky view fraction for remote shading
alb0 = 0.2  # Snow-free ground albedo

# -----------------------------------
# Default snow FSM parameters
# -----------------------------------
asmn = 0.5            # Minimum albedo for melting snow
asmx = 0.85           # Maximum albedo for fresh snow
eta0 = 3.7e7          # Reference snow viscosity (Pa s)
hfsn = 0.1            # Snowcover fraction depth scale (m)
kfix = 0.24           # Fixed thermal conductivity of snow (W/m/K)
rcld = 300            # Maximum density for cold snow (kg/m^3)
rfix = 300            # Fixed snow density (kg/m^3)
rgr0 = 5e-5           # Fresh snow grain radius (m)
rhof = 100            # Fresh snow density (kg/m^3)
rhow = 300            # Wind-packed snow density (kg/m^3)
rmlt = 500            # Maximum density for melting snow (kg/m^3)
Salb = 10             # Snowfall to refresh albedo (kg/m^2)
snda = 2.8e-6         # Thermal metamorphism parameter (1/s)
Talb = -2             # Snow albedo decay temperature threshold (C)
tcld = 3.6e6          # Cold snow albedo decay time scale (s)
tmlt = 3.6e5          # Melting snow albedo decay time scale (s)
trho = 200*3600       # Snow compaction timescale (s)
Wirr = 0.03           # Irreducible liquid water content of snow
z0sn = 0.001          # Snow roughness length (m)

# SWE threshold where SCA = 1 (SWEsca) [SNFRAC = 3]
SWEsca = 40
# shape of the fSCA. [SNFRAC = 3]
Taf = 4.0
# coefficient of variation for the subgrid snow variation [SNFRAC = 4]
subgrid_cv = 2.0

# -----------------------------------
# Ground surface and soil FSM2parameters
# -----------------------------------
fcly = 0.3            # Soil clay fraction
fsnd = 0.6            # Soil sand fraction
gsat = 0.01           # Surface conductance for saturated soil (m/s)
z0sf = 0.1            # Snow-free surface roughness length (m)

# -----------------------------------
# Vegetation FSM2 parameters
# -----------------------------------
acn0 = 0.1            # Snow-free dense canopy albedo
acns = 0.4            # Snow-covered dense canopy albedo
avg0 = 0.21           # Canopy element reflectivity
avgs = 0.6            # Canopy snow reflectivity
cvai = 3.6e4          # Vegetation heat capacity per unit VAI (J/K/m^2)
gsnf = 0.01           # Snow-free vegetation moisture conductance (m/s)
hbas = 2              # Canopy base height (m)
kext = 0.5            # Vegetation light extinction coefficient
leaf = 20             # Leaf boundary resistance (s/m)^(1/2)
svai = 4.4            # Intercepted snow capacity per unit VAI (kg/m^2)
tunl = 240*3600       # Canopy snow unloading time scale (s)
wcan = 2.5            # Canopy wind decay coefficient


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
