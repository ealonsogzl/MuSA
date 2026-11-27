#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alvaro Robledano - arobledano@ipe.csic.es
SMRT helpers shared by SNOWPACK2SMRT and FSM2SMRT: the sensor, the names of
the SMRT columns of the state and the values of those columns from an SMRT
result.

Passive sensors give brightness temperatures (K), columns <GHz><pol>_i<angle>
(e.g. 36V_i55). Active sensors give backscatter (dB), columns
<GHz><pol_inc><pol>_i<angle> (e.g. 5VH_i35: emitted V, received H).
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
    "corr_length": 5e-2,
}

# Below this value, SMRT gives zero backscatter (i.e. -200 dB for 0)
SIGMA_DB_MIN = -150.0


def get_sensor():
    """Sensor from cfg.SMRT_sensor_list, e.g. 
    "passive(...)" or "active(...)" """
    return eval("sensor_list.{}".format(cfg.SMRT_sensor_list))

def is_active(sensor):
    return sensor.mode == "A"


def make_substrate(sensor, temperature):
    """Soil under the snow from cfg.SMRT_soil. The temperature (K) comes
    from each model. Passive: soil_wegmuller; Active: iem_fung92"""
    soil = {**DEFAULT_SOIL, **getattr(cfg, "SMRT_soil", {})}
    common = dict(
        temperature=temperature,
        moisture=soil["moisture"],
        sand=soil["sand"],
        clay=soil["clay"],
        drymatter=soil["drymatter"],
        roughness_rms=soil["roughness_rms"],
    )
    if is_active(sensor):
        return make_soil(
            "iem_fung92",
            "soil_permittivity_dobson85_peplinski95",
            corr_length=soil["corr_length"],
            **common,
        )
    return make_soil(
        "soil_wegmuller", "soil_permittivity_dobson85_peplinski95", **common
    )


def _axes(sensor):
    """(dimension, values in the sensor order) in the order of the columns.
    atleast_1d: SMRT stores a single frequency as a scalar"""
    pol = list(np.atleast_1d(sensor.polarization))
    if is_active(sensor):
        return [
            ("frequency", np.atleast_1d(sensor.frequency)),
            ("theta_inc", np.atleast_1d(sensor.theta_inc_deg)),
            ("polarization_inc", list(np.atleast_1d(sensor.polarization_inc))),
            ("polarization", pol),
        ]
    return [
        ("frequency", np.atleast_1d(sensor.frequency)),
        ("theta", np.atleast_1d(sensor.theta_deg)),
        ("polarization", pol),
    ]


def column_names(sensor):
    axes = dict(_axes(sensor))
    if is_active(sensor):
        return [
            f"{int(round(f/1e9))}{pi}{p}_i{int(round(t))}"
            for f in axes["frequency"]
            for t in axes["theta_inc"]
            for pi in axes["polarization_inc"]
            for p in axes["polarization"]
        ]

    return [
        f"{int(round(f/1e9))}{p}_i{int(round(t))}"
        for f in axes["frequency"]
        for t in axes["theta"]
        for p in axes["polarization"]
    ]


def result_values(result, sensor):
    """Flat values of an SMRT result, Tb (passive) or backscatter in dB
    (active) in the same order as column_names (the state columns are 
    relabelled by position)"""
    axes = _axes(sensor)
    out = result.sigma_dB() if is_active(sensor) else result.Tb()

    # SMRT drops the dimensions of length 1 (a single frequency, angle or
    # polarization), so they are restored before transposing
    for dim, values in axes:
        if dim in out.dims:
            continue
        if dim in out.coords:  # kept as a scalar coordinate
            out = out.expand_dims(dim)
        else:
            out = out.expand_dims({dim: values})

    # same order as the column names. The polarizations are also selected
    # by name: SMRT always returns V before H, and in active mode, adds a
    # third Stokes component (U) that is not a channel
    out = out.transpose(*[dim for dim, _ in axes])
    out = out.sel({dim: values for dim, values in axes if "polarization" in dim})

    values = out.values.flatten()
    if is_active(sensor) and not np.all(
        np.isfinite(values) & (values > SIGMA_DB_MIN)
    ):
        raise Exception(
            "SMRT gives zero or non-finite backscatter: "
            f"{dict(zip(column_names(sensor), values))}"
        )
    return values
