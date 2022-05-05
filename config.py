#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the MuSA configuration file.
Note that not all the options will be used in all the experimental setups.

"""

# -----------------------------------
# Directories
# -----------------------------------

nc_obs_path = "./DATA/Obs/"
nc_forcing_path = "./DATA/Forcing/"
nc_maks_path = "./DATA/mask/"
fsm_src_path = "./FSM2"
intermediate_path = "./DATA/INTERMEDIATE/"
output_path = "./DATA/RESULTS/"
tmp_path = None


# -----------------------------------
# Data Assim
# -----------------------------------

# assimilation_strategy from "filtering", "smoothing" or "direct_insertion"
assimilation_strategy = "smoothing"

# da_method from "Kalman" or "Particle"
da_method = "Particle"
redraw_prior = True
Kalman_iterations = 4

# resampling_algorithm from "bootstrapping", residual_resample,
# stratified_resample or systematic_resample
resampling_algorithm = "bootstrapping"
ensemble_members = 100
r_cov = [0.15]

# var_to_assim from "snd", "SWE", "Tsrf","fSCA", "SCA", "alb"
var_to_assim = ["snd"]

# fSCA_implementation from "Noah", "linear" or "asymptotic"
fSCA_implementation = "Noah"

# vars_to_perturbate from "SW", "LW", "Prec", "Ta", "RH", "Ua", "PS
vars_to_perturbate = ["Prec", "Ta"]

# perturbation_strategy from "constant_normal" or "constant_lognormal"
perturbation_strategy = ["constant_lognormal",
                         "constant_normal"]

# precipitation_phase from "Harder" or "temp_thld"
precipitation_phase = "Harder"

# Save ensembles as a pkl object
save_ensemble = False
save_ensemble_path = "./DATA/ENSEMBLES/"

# -----------------------------------
# Domain
# -----------------------------------

# implementation from "point_scale" or "distributed"
implementation = "distributed"

# parallelization from "sequential", "multiprocessing", "MPI" or "PBS.array"
parallelization = "multiprocessing"
nprocess = None  # if None, the number of processors will be estimated
aws_lat = 4735490.19
aws_lon = 711249.00
date_ini = "2018-09-01 00:00"
date_end = "2020-08-30 23:00"

season_ini_month = 9
season_ini_day = 1


# -----------------------------------
# Observations
# -----------------------------------

# Note: Dates and obs files will be sorted internally. Ensure the alphabetical
# order of the obs files fits the list of dates (dates_obs)
# Note 2: dates_obs supports list indentation to not have to write many dates
# in very long runs. example for generating a list of dailly strings:
# =============================================================================
# import datetime as dt
#
# start = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
# end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
# dates_obs = [(start + dt.timedelta(days=x) + dt.timedelta(hours=12)).
#             strftime('%Y-%m-%d %H:%M') for x in range(0, (end-start).days+1)]
#
# =============================================================================

dates_obs = ["2019-02-21 12:00",
             "2019-03-26 12:00",
             "2019-05-05 12:00",
             "2019-05-09 12:00",
             "2019-05-23 12:00",
             "2019-05-30 12:00",
             "2020-01-14 12:00",
             "2020-02-03 12:00",
             "2020-02-24 12:00",
             "2020-03-11 12:00",
             "2020-04-29 12:00",
             "2020-05-03 12:00",
             "2020-05-12 12:00",
             "2020-05-19 12:00",
             "2020-05-26 12:00",
             "2020-06-02 12:00",
             "2020-06-10 12:00",
             "2020-06-21 12:00"]

obs_var_names = ["HS"]
lat_obs_var_name = "northing"
lon_obs_var_name = "easting"


# -----------------------------------
# Forcing
# -----------------------------------

frocing_var_names = {"SW_var_name": "SW",
                     "LW_var_name": "LW",
                     "Precip_var_name": "PRECC",
                     "Press_var_name": "PRESS",
                     "RH_var_name": "RH",
                     "Temp_var_name": "TEMP",
                     "Wind_var_name": "UA"}

forcing_dim_names = {"lat_forz_var_name": "northing",
                     "lon_forz_var_name": "easting",
                     "time_forz_var_name": "time"}

# -----------------------------------
# FSM configuration
# -----------------------------------
Dzsnow = [0.1, 0.2, 0.4]
