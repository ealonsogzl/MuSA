#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the MuSA configuration file.
Note that not all the options will be used in all the experimental setups.

"""
# Note: not all options have been tested with dIm and snow17
numerical_model = 'FSM2'  # model to use from FSM2, dIm or snow17
# -----------------------------------
# Directories
# -----------------------------------

nc_obs_path = "./DATA/Obs/"
nc_forcing_path = "./DATA/Forcing/"
nc_maks_path = "./DATA/mask/mask.nc"
dem_path = "./DATA/DEM/DEM.nc"
fsm_src_path = "./FSM2/"
intermediate_path = "./DATA/INTERMEDIATE/"
save_ensemble_path = "./DATA/ENSEMBLES/"
output_path = "./DATA/RESULTS/"
spatial_propagation_storage_path = "./DATA/SPATIAL_PROP/"
tmp_path = None

# If restart_run is enabled, the outputs will not be overwritten
restart_run = False
# If restart_forcing, the forcing will be read from intermediate files
restart_forcing = False

# -----------------------------------
# Data Assim
# -----------------------------------

# da_algorithm from PF, EnKF, IEnKF, PBS, ES, IES, deterministic_OL,
# IES-MCMC_AI, IES-MCMC, AdaMuPBS, AdaPBS or PIES
da_algorithm = 'PBS'
redraw_prior = False  # PF and PBS only
max_iterations = 4  # IEnKF, IES, IES-MCMC and AdaPBS
# resampling_algorithm from "bootstrapping", residual_resample,
# stratified_resample,  systematic_resample, no_resampling
resampling_algorithm = "no_resampling"
ensemble_members = 100
Neffthrs = 0.1           # Low Neff threshold
# MCMC parameters
chain_len = 20000   # Length of the mcmcm
adaptive = True    # Update proposal covariance for next step.
histcov = True     # Use posterior IES covariance as proposal covariance
burn_in = 0.1      # discard the first x proportion of samples

# r_cov can be a list of scalars of length equal to var_to_assim or the string
# 'dynamic_error'. If 'dynamic_error' is selected, errors may change in space
# and time. If this option is selected, the errors will be stored in a new
# variable in the observation files, and will have the same dimensions as
# the observations.
r_cov = [0.04]
add_dynamic_noise = False
# var_to_assim from "snd", "SWE", "Tsrf","fSCA", "SCA", "alb", "LE", "H"
var_to_assim = ["snd"]

# DA second order variables and/or statistics (experimental)
DAsord = False
DAord_names = ["Ampli"]

# vars_to_perturbate from "SW", "LW", "Prec", "Ta", "RH", "Ua", "PS
vars_to_perturbate = ["Ta", "Prec"]

# In smoothers, re-draw new parameters for each season
season_rejuvenation = [True, True]
# seed to initialise the random number generator
seed = None

# perturbation_strategy from "normal", "lognormal",
# "logitnormal_adi" or "logitnormal_mult"
perturbation_strategy = ["logitnormal_adi", "logitnormal_mult"]

# precipitation_phase from "Harder" or "temp_thld"
precipitation_phase = "Harder"

# Save ensembles as a pkl object
save_ensemble = False

# -----------------------------------
# Domain
# -----------------------------------

# implementation from "point_scale", "distributed" or "Spatial_propagation"
implementation = "distributed"

# if implementation = "Spatial_propagation" : specify which observation
# variables are spatially propagated in a list
# if var_to_prop = var_to_assim -> All the variables are spatially propagated
# if var_to_prop = [] -> Any variable is spatially propagated
var_to_prop = var_to_assim

# parallelization from "sequential", "multiprocessing" or "HPC.array"
parallelization = "multiprocessing"
MPI = False  # Note: not tested
nprocess = 8  # Note: if None, the number of processors will be estimated

aws_lat = 4735225.54  # Latitude in case of point_scale
aws_lon = 710701.28   # Longitude in case of point_scale

date_ini = "2018-09-01 00:00"
date_end = "2020-08-30 23:00"

season_ini_month = 9  # In smoothers, beginning of DA window (month)
season_ini_day = 1    # In smoothers, beginning of DA window (day)

# -----------------------------------
# Spatial propagation configuration
# -----------------------------------

# Cut-off distance for the Gaspari and Cohn function.
c = [5, 5]

# Calculate the distances internally (topo_dict_external = None) or read an
# external file with the dimensions
topo_dict_external = None
dist_algo = 'euclidean'

# Optionally perform dimension reduction
dimension_reduction = 'None'  # LMDS, PCA or None

# try to find closePD or raise exception (closePDmethod = None)
closePDmethod = None  # 'clipped' (the faster but less accurate) or 'nearest'

# Topographical dimensions to compute the distances
topographic_features = {'Ys': True,     # Latitude
                        'Xs': True,     # Longitude
                        'Zs': False,    # Elevation
                        'slope': False,  # Slope
                        'DAH': False,   # Diurnal Anisotropic Heat
                        'TPI': False,   # Topographic Position Index
                        'Sx': False}    # Upwind Slope index (Winstral)

# Topographical hyperparameters
DEM_res = 5              # DEM resolution
TPI_size = 25            # TPI window size
Sx_dmax = 15             # Sx search distance
Sx_angle = 315           # Sx main wind direction angle
nc_dem_varname = "DEM"     # Name of the elevation variable in the DEM

# -----------------------------------
# Observations
# -----------------------------------

# Note: Dates and obs files will be sorted internally. Ensure the alphabetical
# order of the obs files fits the list of dates (dates_obs)

# Note 2: dates_obs supports list indentation to not have to write many dates
# in very long runs. Example for generating a list of dailly strings:

# =============================================================================
# import datetime as dt
#
# start = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
# end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
# dates_obs = [(start + dt.timedelta(days=x) + dt.timedelta(hours=12)).
#             strftime('%Y-%m-%d %H:%M') for x in range(0, (end-start).days+1)]
#
# =============================================================================

# Note 3: A single column .cvs without headers with the dates in the
# format "%Y-%m-%d %H:%M" is also accepted substituting:
# dates_obs = '/path/to/file/dates.csv'


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
obs_error_var_names = ['sdError']  # In case of r_cov = 'dynamic_error'
lat_obs_var_name = "northing"
lon_obs_var_name = "easting"


# -----------------------------------
# Forcing and some parameters
# -----------------------------------
# Note: not all parameters/forcing variables are needed for all models
# Note II: param_var_names is optional. It can be used to change some of the
# model parameters, including vegetation ones. If they are not included as
# part of the forcing, those defined in constants.py will be used.
# These parameters can be included within the assimilation

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

param_var_names = {"RealLat_var_name": " XLAT",
                   "vegh_var_name": "vegh",
                   "VAI_var_name": "VAI",
                   "fsky_var_name": "fsky",
                   "hbas_var_name": "hbas",
                   "SWEsca_var_name": "SWEsca",
                   "Taf_var_name": "Taf",
                   "cv_var_name": "subgrid_cv"}

# -----------------------------------
# FSM configuration (Namelist)
# -----------------------------------

# Number and thickness of snow layers
Dzsnow = [0.1, 0.2, 0.4]

# -----------------------------------
# FSM configuration (Compilation)
# -----------------------------------

# Optimization flag. Choose from -O (no optimization), -O1, -O2, -O3 or -Ofast.
# Note: -O3 is recommended. -Ofast may be slightly faster (~10%), but its
# numerical accuracy is lower.
# Note II: Can be used to pass any other flag(s) to gfortran if you know
# what you are doing
OPTIMIZATION = '-O3'

# Parameterizations, see FSM2 documentation
ALBEDO = 2
CONDCT = 1
DENSITY = 2
EXCHNG = 1
HYDROL = 2
SGRAIN = 2
SNFRAC = 2
CANMOD = 2
CANRAD = 2
