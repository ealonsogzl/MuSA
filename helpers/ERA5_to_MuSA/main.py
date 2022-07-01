#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small application to download data from ERA5_land ready to be used by MuSA,
using Google Earth Engine. To use it you need to have the GEE API installed
and to be authenticated.

It creates a flat netcdf. Dimensions: (time x 1 x number of locations)

Author: Esteban Alonso González - e.alonsogzl@gmail.com
"""
import ee
import myfuns as myf

# forcing name
nc_name = "MuSA_forz.nc"

# Initial date of interest (inclusive).
i_date = "2015-01-16"

# Final date of interest (exclusive).
f_date = "2017-01-16"

# Define the locations of interest in degrees.
era_lon = [100.233, 0.63, 7.02]
era_lat = [64.266, 42.67, 45.49]


###
# ------- Hopefully it will not be necessary to touch under this line ------- #
###

# Create netcdf
myf.init_netcdf(nc_name, era_lon, era_lat, i_date, f_date)

# Initialize the library.
ee.Initialize()

# Import the ERA5_LAND collection.
era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

# Separate dates in seq years to avoid EEException: User memory limit exceeded.
dates = myf.seq_dates(i_date, f_date)

# Check if same n of lat & lon
if len(era_lon) != len(era_lat):
    raise Exception('check coordinates')

# Loop over coordinates
for n in range(len(era_lon)):

    print("Solving cell: " + str(n+1)+" of " + str(len(era_lon)))
    # Construct a point from coordinates.
    era_poi = ee.Geometry.Point([era_lon[n], era_lat[n]])

    # get pandas df form GEE
    era5df = myf.fromGEE_to_df(dates, era5_land, era_poi)

    # prepare columns for MuSA
    MuSA_era5 = myf.format_forz(era5df)

    myf.store_era_nc(nc_name, MuSA_era5, n)

