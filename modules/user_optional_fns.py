#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module can be used to modify the outputs of the numerical model used,
for example by calculating statistics or coupling other models.
In a real use case, functions will be included here that will add new 
variables to the output of the model.

Warning: VERY experimental, its use can be complex.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import pandas as pd

import config as cfg


def snd_ord(df, snd_min=0.1):
    """
    DAsord: compute the variables of DAord_names (config.py) from the output
    of the model and add them as columns of df.

    df has one row per time step of the simulation: the output of the model
    (its model_columns, e.g. snd, SWE, fSCA... depending on the model) and the
    columns year, month, day and hour of each row. Any column can be used;
    only the DAord_names columns are kept afterwards.

    Example: snow_days, the number of days with snow (daily mean snow depth
    above snd_min, in m) since the beginning of the season (season_ini_month
    and season_ini_day in config.py). All the models give snd, so it works
    with any of them. Only the days of the simulated period can be counted:
    with short assimilation windows (filters) it counts from the beginning of
    each window.
    """
    date = pd.to_datetime(df[["year", "month", "day"]])

    # season of each row: the year in which its season began
    start = pd.to_datetime(
        pd.DataFrame(
            {
                "year": date.dt.year,
                "month": cfg.season_ini_month,
                "day": cfg.season_ini_day,
            }
        )
    )
    season = date.dt.year - (date < start)

    # one value per day: True if the mean snow depth of the day is above
    # snd_min; the cumulative sum in each season counts its snow days
    daily = df["snd"].groupby(date.to_numpy()).mean()
    daily_season = season.groupby(date.to_numpy()).first()
    snow_days = (daily > snd_min).groupby(daily_season).cumsum()

    # back to one value per time step
    df["snow_days"] = snow_days.reindex(date.to_numpy()).to_numpy(float)
    return df
