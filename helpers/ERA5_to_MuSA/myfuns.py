#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My functions

Author: Esteban Alonso González - e.alonsogzl@gmail.com
"""
import pandas as pd
import numpy as np
import datetime as dt
import netCDF4 as nc4


def ee_array_to_df(arr, list_of_bands):

    # Transforms client-side ee.Image.getRegion array to pandas.DataFrame
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['datetime',  *list_of_bands]]

    return df


def myRange(start, end, step):
    i = start
    while i < end:
        yield i
        i += step
    yield end


def seq_dates(i_date, f_date):
    sdate = dt.datetime.strptime(i_date, "%Y-%m-%d")
    edate = dt.datetime.strptime(f_date, "%Y-%m-%d")

    diff_days = edate - sdate
    if diff_days.days > 365:
        seq_dates = myRange(0, diff_days.days, 365)
        date_list = [sdate + dt.timedelta(days=x) for x in seq_dates]
    else:
        date_list = [sdate, edate]

    return [x.strftime("%Y-%m-%d") for x in date_list]


def fromGEE_to_df(dates, era5_land, era_poi):

    # scale in meters
    scale = 1000

    # initialize storage df
    era5_df = pd.DataFrame(columns=['datetime',
                                    'temperature_2m',
                                    'dewpoint_temperature_2m',
                                    'surface_solar_radiation_downwards_hourly',
                                    'surface_thermal_radiation_downwards_hourly',
                                    'total_precipitation_hourly',
                                    'u_component_of_wind_10m',
                                    'v_component_of_wind_10m',
                                    'surface_pressure'])

    for dat in range(len(dates)-1):

        i_dt_tmp = dates[dat]
        f_dt_tmp = dates[dat+1]

        # Selection of appropriate bands and dates for LST.
        era5_selc = era5_land.select('temperature_2m',
                                     'dewpoint_temperature_2m',
                                     'surface_solar_radiation_downwards_hourly',
                                     'surface_thermal_radiation_downwards_hourly',
                                     'total_precipitation_hourly',
                                     'u_component_of_wind_10m',
                                     'v_component_of_wind_10m',
                                     'surface_pressure').filterDate(i_dt_tmp,
                                                                    f_dt_tmp)

        era5_data = era5_selc.getRegion(era_poi, scale).getInfo()

        era5_dftmp = ee_array_to_df(era5_data,
                                    ['temperature_2m',
                                     'dewpoint_temperature_2m',
                                     'surface_solar_radiation_downwards_hourly',
                                     'surface_thermal_radiation_downwards_hourly',
                                     'total_precipitation_hourly',
                                     'u_component_of_wind_10m',
                                     'v_component_of_wind_10m',
                                     'surface_pressure'])

        # append rows to storage df
        era5_df = era5_df.append(era5_dftmp, ignore_index=True)

    return era5_df


def format_forz(era5df):

    tidy_df = era5df.copy()

    # Silence copy warning
    pd.set_option('mode.chained_assignment', None)

    # Shortwave
    # To W
    tidy_df['SW_flux'] = \
        tidy_df['surface_solar_radiation_downwards_hourly']/3600
    # Remove negative noise
    tidy_df['SW_flux'][tidy_df['SW_flux'] < 0] = 0

    # Longwave
    # To W
    tidy_df['LW_flux'] = \
        tidy_df['surface_thermal_radiation_downwards_hourly']/3600
        # Remove negative noise
    tidy_df['LW_flux'][tidy_df['LW_flux'] < 0] = 0
    
    # RH
    TD = tidy_df['dewpoint_temperature_2m'] - 273.15
    T = tidy_df['temperature_2m'] - 273.15
    tidy_df['RH'] = 100 * (np.exp((17.625 * TD) / (243.04 + TD)) /
                           np.exp((17.625 * T) / (243.04 + T)))
    # Force bounds
    tidy_df['RH'][tidy_df['RH'] > 100] = 100
    tidy_df['RH'][tidy_df['RH'] < 0] = 0

    # Wind
    U = tidy_df['u_component_of_wind_10m']
    V = tidy_df['v_component_of_wind_10m']
    tidy_df['Wind'] = np.sqrt(U**2 + V**2)

    # Precipitation
    tidy_df['PRECC'] = \
        tidy_df['total_precipitation_hourly']*1000/3600
    tidy_df['PRECC'][tidy_df['PRECC'] < 0] = 0

    # remove old columns
    del tidy_df['surface_solar_radiation_downwards_hourly']
    del tidy_df['surface_thermal_radiation_downwards_hourly']
    del tidy_df['dewpoint_temperature_2m']
    del tidy_df['u_component_of_wind_10m']
    del tidy_df['v_component_of_wind_10m']
    del tidy_df['total_precipitation_hourly']

    return tidy_df


def init_netcdf(nc_name, era_lon, era_lat, i_date, f_date):
    lon = range(len(era_lon))
    lat = 0

    # Create ncdf
    f = nc4.Dataset(nc_name, 'w', format='NETCDF4')

    # create dimensions (spatial are fake)
    f.createDimension('lon', len(lon))
    f.createDimension('lat', 1)
    f.createDimension('time', None)

    # create dim vars
    longitude = f.createVariable('Lon', 'f4', 'lon')
    longitude.units = 'number of points'
    longitude.long_name = 'longitude'

    latitude = f.createVariable('Lat', 'f4', 'lat')
    latitude.units = 'Just one'
    latitude.long_name = 'latitude'

    time = f.createVariable('time', np.float64, ('time',))
    time.units = 'hours since 1900-01-01'
    time.long_name = 'time'

    # calculate number of hours
    sdate = dt.datetime.strptime(i_date, "%Y-%m-%d")
    edate = dt.datetime.strptime(f_date, "%Y-%m-%d")

    hour_since = (sdate - dt.datetime.strptime
                  ('1900-01-01', "%Y-%m-%d")).total_seconds()/3600
    hour_until = (edate - dt.datetime.strptime
                  ('1900-01-01', "%Y-%m-%d")).total_seconds()/3600

    hours_era = range(int(hour_since), int(hour_until))

    # add dim values
    longitude[:] = lon
    latitude[:] = lat
    time[:] = np.asarray(hours_era)

    f.description = "Frozing for MuSA from GEE"

    # Create met vars
    sw = f.createVariable('SW', 'f4', ('time', 'lon', 'lat'))
    sw.long_name = 'surface_solar_radiation_downwards_hourly'
    sw.units = 'W/m2'
    lw = f.createVariable('LW', 'f4', ('time', 'lon', 'lat'))
    lw.long_name = 'surface_thermal_radiation_downwards_hourly'
    lw.units = 'W/m2'
    precc = f.createVariable('PRECC', 'f4', ('time', 'lon', 'lat'))
    precc.long_name = 'precipitation_flux_hourly'
    precc.units = 'kg/m2s'
    press = f.createVariable('PRESS', 'f4', ('time', 'lon', 'lat'))
    press.long_name = 'surface_pressure'
    press.units = 'Pa'
    rh = f.createVariable('RH', 'f4', ('time', 'lon', 'lat'))
    rh.long_name = 'relative_humidity'
    rh.units = '-'
    temp = f.createVariable('TEMP', 'f4', ('time', 'lon', 'lat'))
    temp.long_name = 'temperature'
    temp.units = 'K'
    ua = f.createVariable('UA', 'f4', ('time', 'lon', 'lat'))
    ua.long_name = 'wind_speed'
    ua.units = 'm/s2'

    f.close()


def store_era_nc(nc_name, MuSA_era5, n):

    f = nc4.Dataset(nc_name, "r+")
    f.variables["SW"][:, n, 0] = MuSA_era5['SW_flux']
    f.variables["LW"][:, n, 0] = MuSA_era5['LW_flux']
    f.variables["PRECC"][:, n, 0] = MuSA_era5['PRECC']
    f.variables["PRESS"][:, n, 0] = MuSA_era5['surface_pressure']
    f.variables["RH"][:, n, 0] = MuSA_era5['RH']
    f.variables["TEMP"][:, n, 0] = MuSA_era5['temperature_2m']
    f.variables["UA"][:, n, 0] = MuSA_era5['Wind']
    f.close()
