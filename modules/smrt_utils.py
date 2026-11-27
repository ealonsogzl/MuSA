#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alvaro Robledano - arobledano@ipe.csic.es
SMRT helpers shared by SNOWPACK2SMRT and FSM2SMRT: the sensor, the names of
the SMRT columns of the state and the values of those columns from an SMRT
result.

Passive sensors give brightness temperatures (K), columns <GHz><pol>_i<angle>
(e.g. 36V_i55). Active sensors give backscatter (dB), columns
<GHz><pol><pol>_i<angle> (e.g. 5VV_i35), copolarization only (VV, HH).
"""

import numpy as np
import xarray as xr
import config as cfg
from smrt import make_model, make_soil, sensor_list  # noqa: F401 (sensor list used by eval)

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
    sensor = eval("sensor_list.{}".format(cfg.SMRT_sensor_list))
    if is_active(sensor):
        _copol(sensor)  # stops here if the sensor is not co-polarized
    return sensor


def is_active(sensor):
    return sensor.mode == "A"


def _copol(sensor):
    """Polarizations of an active sensor, used only as co-polarized pairs
    (VV, HH). Cross-polarization (VH, HV) is not supported yet"""
    pol_inc = [str(p) for p in np.atleast_1d(sensor.polarization_inc)]
    pol = [str(p) for p in np.atleast_1d(sensor.polarization)]
    if pol_inc != pol:
        raise Exception(
            "Active SMRT sensors: only co-polarization (VV, HH) is supported. "
            "Use the same polarization_inc and polarization, e.g. "
            "polarization_inc='V', polarization='V' for VV. "
            f"Got polarization_inc={pol_inc}, polarization={pol}"
        )
    return pol


def make_em_model(snowpack):
    """Electromagnetic model (cfg.SMRT_emmodel, "iba" if not set) for now with the
    DORT solver:
    "iba": IBA, default. Not valid for layers denser than about
        458 kg m-3 (half the ice density); SMRT warns.
    "iba_dense": IBA with dense_snow_correction="auto": those layers are
        modelled as air in ice. Tb and sigma0 jump when a layer crosses
        458 kg m-3.
    "symsce_torquato21": symmetric SCE, valid and continuous at all
        densities, but differs from IBA. Work in progress.
    See Picard et al. 2022, doi:10.5194/tc-16-3861-2022"""
    name = getattr(cfg, "SMRT_emmodel", "iba")
    if name not in ("iba", "iba_dense", "symsce_torquato21"):
        raise Exception(
            f"SMRT_emmodel = {name!r} not supported. "
            'Use "iba", "iba_dense" or "symsce_torquato21"'
        )
    if name == "symsce_torquato21":
        return make_model("symsce_torquato21", "dort")
    # Without snow, make_snowpack adds a transparent layer: plain IBA then
    if name == "iba_dense" and all(
        layer.emmodel is None for layer in snowpack.layers
    ):
        return make_model(
            "iba", "dort", emmodel_options=dict(dense_snow_correction="auto")
        )
    return make_model("iba", "dort")
    


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
            f"{int(round(f/1e9))}{p}{p}_i{int(round(t))}"
            for f in axes["frequency"]
            for t in axes["theta_inc"]
            for p in _copol(sensor)
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
    if is_active(sensor):
        # co-polarized pairs only: (V, V), (H, H)
        pol = xr.DataArray(_copol(sensor), dims="copol")
        out = out.sel(polarization_inc=pol, polarization=pol)
        out = out.transpose("frequency", "theta_inc", "copol")
    else:
        out = out.transpose(*[dim for dim, _ in axes])
        out = out.sel(polarization=axes[-1][1])

    values = out.values.flatten()
    if is_active(sensor) and not np.all(
        np.isfinite(values) & (values > SIGMA_DB_MIN)
    ):
        raise Exception(
            "SMRT gives zero or non-finite backscatter: "
            f"{dict(zip(column_names(sensor), values))}"
        )
    return values
