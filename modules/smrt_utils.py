#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alvaro Robledano - arobledano@ipe.csic.es
SMRT helpers shared by SNOWPACK2SMRT and FSM2SMRT: the sensor, the names of
the SMRT columns of the state and the values of those columns from an SMRT
result.

Passive sensors give brightness temperatures (K), columns <GHz><pol>_i<angle>
(e.g. 36V_i55).
"""

import numpy as np
import config as cfg
from smrt import make_soil, sensor_list  # noqa: F401 (used by eval)

# Soil used before cfg.SMRT_soil existed; also fills keys missing from it
DEFAULT_SOIL = {
    "moisture": 0.20435,
    "sand": 0.6,
    "clay": 0.3,
    "drymatter": 1100,
    "roughness_rms": 5e-3,
}

def get_sensor():
    """Sensor from cfg.SMRT_sensor_list, e.g. "passive(...)" """
    return eval("sensor_list.{}".format(cfg.SMRT_sensor_list))

def make_substrate(temperature):
    """Soil under the snow from cfg.SMRT_soil. The temperature (K) comes
    from each model"""
    soil = {**DEFAULT_SOIL, **getattr(cfg, "SMRT_soil", {})}
    return make_soil(
        "soil_wegmuller",
        "soil_permittivity_dobson85_peplinski95",
        temperature=temperature,
        moisture=soil["moisture"],
        sand=soil["sand"],
        clay=soil["clay"],
        drymatter=soil["drymatter"],
        roughness_rms=soil["roughness_rms"],
    )


def _axes(sensor):
    """(dimension, values in the sensor order) in the order of the columns.
    atleast_1d: SMRT stores a single frequency as a scalar"""
    return [
        ("frequency", np.atleast_1d(sensor.frequency)),
        ("theta", np.atleast_1d(sensor.theta_deg)),
        ("polarization", list(np.atleast_1d(sensor.polarization))),
    ]


def column_names(sensor):
    axes = dict(_axes(sensor))
    return [
        f"{int(round(f/1e9))}{p}_i{int(round(t))}"
        for f in axes["frequency"]
        for t in axes["theta"]
        for p in axes["polarization"]
    ]


def result_values(result, sensor):
    """Flat values of an SMRT result, in the same order as column_names
    (the state columns are relabelled by position)"""
    axes = _axes(sensor)
    out = result.Tb()

    # SMRT drops the dimensions of length 1 (a single frequency, angle or
    # polarization), so they are restored before transposing
    for dim, values in axes:
        if dim in out.dims:
            continue
        if dim in out.coords:  # kept as a scalar coordinate
            out = out.expand_dims(dim)
        else:
            out = out.expand_dims({dim: values})

    # same order as the column names: frequency, theta, polarization.
    # SMRT always returns V before H, so the polarizations are selected in
    # the sensor order
    out = out.transpose(*[dim for dim, _ in axes])
    out = out.sel(polarization=dict(axes)["polarization"])

    return out.values.flatten()
