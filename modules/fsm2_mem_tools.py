#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-memory FSM2 backend for MuSA.

This module adapts the Python/Numba package :mod:`fsm2` to the same
``model_*`` contract used by ``modules.fsm_tools``, ``modules.dIm_tools``,
``modules.snow17_tools`` and ``modules.snowpack_tools``.  MuSA imports it as
``model`` when ``config.numerical_model == "FSM2mem"``.

The model runs entirely in memory, like ``modules.dIm_tools`` and
``modules.snow17_tools``.  ``model_run(forcing_df, init)`` returns the state
table and the final dump; no meteorological, state or output files are written
or read.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import numpy as np
import pandas as pd

import config as cfg

from modules.fsm2 import (
    Condiciones,
    Configuracion,
    Forzamiento,
    compilar_numba,
    fsm2 as _fsm2,
)

# ---------------------------------------------------------------------------
# Output columns expected by MuSA
# ---------------------------------------------------------------------------

model_columns = [
    "snd",
    "SWE",
    "Tsrf",
    "fSCA",
    "alb",
    "H",
    "LE",
]

if cfg.run_smrt:
    # Keep exactly the same per-layer columns as the Fortran FSM2 backend,
    # because modules.FSM2SMRT.run_SMRT expects these names.
    model_columns.extend(
        [
            "Rgrn1",
            "Rgrn2",
            "Rgrn3",
            "Tsoil1",
            "Tsoil2",
            "Tsoil3",
            "Tsoil4",
            "Tsnow1",
            "Tsnow2",
            "Tsnow3",
            "Vsmc1",
            "Vsmc2",
            "Vsmc3",
            "Vsmc4",
            "Dsnw1",
            "Dsnw2",
            "Dsnw3",
            "Sice1",
            "Sice2",
            "Sice3",
            "Sliq1",
            "Sliq2",
            "Sliq3",
        ]
    )

    import modules.FSM2SMRT as smrt

    model_columns.extend(smrt.return_col_names())

if cfg.DAsord:
    from modules.user_optional_fns import snd_ord

    model_columns.extend(cfg.DAord_names)

model_columns = tuple(model_columns)


# ---------------------------------------------------------------------------
# FSM2 in-memory helpers
# ---------------------------------------------------------------------------


def _fsm_tools():
    """Import the Fortran backend lazily to reuse its generic helpers.

    Importing ``modules.fsm_tools`` at module load time would make the
    existing circular imports between MuSA modules harder to follow.  The
    helpers reused here (forcing table, result storage, ...) are backend
    independent.
    """
    import modules.fsm_tools as fsm_tools

    return fsm_tools


# ---------------------------------------------------------------------------
# FSM2 configuration
# ---------------------------------------------------------------------------


def _nsmax():
    return len(cfg.Dzsnow)


def _nsoil():
    return int(getattr(cfg, "FSM2_NSOIL", 4))


def _ncnpy():
    return 1 if int(cfg.CANMOD) == 1 else 2


def _opciones_compilacion():
    """Map MuSA's FSM2 options to a :class:`fsm2.Configuracion` dictionary."""
    opciones = {
        "dt": float(cfg.dt),
        # FSM2's namelist defaults used by MuSA (nlst_base does not override
        # lat, noon, zT, zU or trans).
        "lat": float(getattr(cfg, "FSM2_LAT", 0.0)),
        "noon": float(getattr(cfg, "FSM2_NOON", 12.0)),
        "zT": float(getattr(cfg, "FSM2_ZT", 2.0)),
        "zU": float(getattr(cfg, "FSM2_ZU", 10.0)),
        "trans": float(getattr(cfg, "FSM2_TRANS", 0.0)),
        # Grid
        "nsmax": _nsmax(),
        "dzsnow": [float(x) for x in cfg.Dzsnow],
        "nsoil": _nsoil(),
        "fvg1": float(getattr(cfg, "FSM2_FVG1", 0.5)),
        "zsub": float(getattr(cfg, "FSM2_ZSUB", 1.5)),
        # Compile-time options.  ZOFFST and SWPART are fixed in MuSA's
        # compil_base.sh; ENSMBL is undefined there, hence 0.
        "albedo": int(cfg.ALBEDO),
        "canint": int(cfg.CANINT),
        "canmod": int(cfg.CANMOD),
        "canrad": int(cfg.CANRAD),
        "canunl": int(cfg.CANUNL),
        "condct": int(cfg.CONDCT),
        "densty": int(cfg.DENSTY),
        "exchng": int(cfg.EXCHNG),
        "hydrol": int(cfg.HYDROL),
        "sgrain": int(cfg.SGRAIN),
        "snfrac": int(cfg.SNFRAC),
        "swpart": int(getattr(cfg, "FSM2_SWPART", 0)),
        "zoffst": int(getattr(cfg, "FSM2_ZOFFST", 1)),
        "ensmbl": int(getattr(cfg, "FSM2_ENSMBL", 0)),
        "nhyd": int(getattr(cfg, "FSM2_NHYD", 10)),
    }
    if hasattr(cfg, "FSM2_DZSOIL"):
        opciones["dzsoil"] = [float(x) for x in cfg.FSM2_DZSOIL]
    return opciones


# Parameters that the Fortran backend reads from the first row of the forcing
# (they are written to the FSM2 namelist).  ``tunl`` is lower case in MuSA's
# forcing table, while the FSM2 API uses ``Tunl``.
_PARAM_MAP = {
    "alb0": "alb0",
    "vegh": "vegh",
    "VAI": "VAI",
    "asmn": "asmn",
    "asmx": "asmx",
    "eta0": "eta0",
    "hfsn": "hfsn",
    "kfix": "kfix",
    "rcld": "rcld",
    "rfix": "rfix",
    "rgr0": "rgr0",
    "rhof": "rhof",
    "rmlt": "rmlt",
    "Salb": "Salb",
    "snda": "snda",
    "Talb": "Talb",
    "tcld": "tcld",
    "tmlt": "tmlt",
    "trho": "trho",
    "Wirr": "Wirr",
    "z0sn": "z0sn",
    "fcly": "fcly",
    "fsnd": "fsnd",
    "gsat": "gsat",
    "z0sf": "z0sf",
    "acn0": "acn0",
    "acns": "acns",
    "avg0": "avg0",
    "avgs": "avgs",
    "cvai": "cvai",
    "gsnf": "gsnf",
    "hbas": "hbas",
    "kext": "kext",
    "leaf": "leaf",
    "svai": "svai",
    "tunl": "Tunl",
    "wcan": "wcan",
}


def _configuracion_from_forcing(forcing_df):
    """Build the FSM2 configuration using the first forcing row.

    The Fortran backend writes the parameter columns of the first row to the
    namelist, so this reproduces exactly the same behaviour.
    """
    if forcing_df.empty:
        raise ValueError("El forzamiento está vacío")

    opciones = _opciones_compilacion()
    primera = forcing_df.iloc[0]
    for origen, destino in _PARAM_MAP.items():
        if origen in forcing_df.columns:
            opciones[destino] = float(primera[origen])

    # Per-timestep profiles are only required by SMRT.
    opciones["guardar_estado"] = bool(cfg.run_smrt)
    return Configuracion(**opciones)


def _configuracion_base():
    return Configuracion(**_opciones_compilacion())


# ---------------------------------------------------------------------------
# Contract functions
# ---------------------------------------------------------------------------


def model_copy(y_id, x_id):
    """FSM2mem runs in memory: no working directory is needed."""
    return None


def model_compile():
    """Compile the Numba core once per process, without a full simulation."""
    segundos = compilar_numba(_configuracion_base())
    print(f"FSM2mem: núcleo Numba compilado en {segundos:.2f} s")
    return None


def model_compile_HPC(HPC_task_id):
    """Compile in every HPC task.

    Numba compiled code is process-local, so there is no binary to share
    between job-array tasks.  Each task compiles before creating its worker
    pool; the workers inherit the compiled functions when ``fork`` is used.
    """
    model_compile()
    return None


def model_run(forcing_df, init=None):
    """Run FSM2 in memory, like dIm and snow17.

    Parameters
    ----------
    forcing_df : pandas.DataFrame
        Forcing table with one row per timestep.
    init : pandas.DataFrame or None
        Optional FSM2 dump from the previous run, used as restart state.

    Returns
    -------
    state : pandas.DataFrame
        Model output with one row per forcing timestep.
    dump : pandas.DataFrame
        Final model state, ready to be used as ``init`` in the next run.
    """
    forzamiento, configuracion = _prepare_forcing(forcing_df)

    # Ensure the optional profile output is enabled when SMRT needs it.
    configuracion.guardar_estado = bool(cfg.run_smrt)

    reinicio = None
    if init is not None:
        reinicio = _condiciones_from_dump(init, configuracion)

    simulacion, condiciones_finales = _fsm2(
        forzamiento,
        reinicio,
        configuracion,
    )

    state = _state_dataframe(simulacion)

    if cfg.DAsord:
        state = snd_ord(state)

    if state.isnull().values.any():
        raise Exception(
            "NaN found in FSM2mem output. Check forcing units, "
            "perturbation_strategy and the configured model parameters."
        )

    return state, _dump_dataframe(condiciones_finales)


def model_remove(fsm_path):
    """FSM2mem has no working directory to remove."""
    return None


def get_var_state_position(var):
    return model_columns.index(var)


def init_result(del_t, DA=False, OL=False):
    return _fsm_tools().init_result(del_t, DA=DA, OL=OL)


def storeDA(
    Result_df, step_results, observations_sbst, error_sbst, time_dict, step
):
    return _fsm_tools().storeDA(
        Result_df,
        step_results,
        observations_sbst,
        error_sbst,
        time_dict,
        step,
    )


def storeOL(OL_FSM, Ensemble, observations_sbst, time_dict, step):
    return _fsm_tools().storeOL(
        OL_FSM, Ensemble, observations_sbst, time_dict, step
    )


def store_sim(
    sim_stat, Ensemble, time_dict, step, MCMC=False, save_prior=False
):
    return _fsm_tools().store_sim(
        sim_stat,
        Ensemble,
        time_dict,
        step,
        MCMC=MCMC,
        save_prior=save_prior,
    )


def forcing_table(lat_idx, lon_idx, step=0):
    return _fsm_tools().forcing_table(lat_idx, lon_idx, step)


def unit_conversion(forcing_df):
    return _fsm_tools().unit_conversion(forcing_df)


# ---------------------------------------------------------------------------
# Forcing preparation
# ---------------------------------------------------------------------------


def stable_forcing(forcing_df):
    """Apply the same stability filters as the Fortran FSM2 backend."""
    temp_forz_def = forcing_df.copy()

    temp_forz_def.loc[temp_forz_def["SW"] < 0, "SW"] = 0
    temp_forz_def.loc[temp_forz_def["LW"] < 0, "LW"] = 0
    temp_forz_def.loc[temp_forz_def["Prec"] < 0, "Prec"] = 0
    temp_forz_def.loc[temp_forz_def["Ua"] < 0, "Ua"] = 0

    temp_forz_def.loc[temp_forz_def["RH"] > 100, "RH"] = 100
    temp_forz_def.loc[temp_forz_def["RH"] < 0, "RH"] = 1

    mask = temp_forz_def["vegh"] < (temp_forz_def["hbas"] + 1)
    temp_forz_def.loc[mask, "vegh"] = 0.0
    temp_forz_def.loc[mask, "VAI"] = 0.0

    temp_forz_def.loc[temp_forz_def["Prec"] < 0.01 / 3600, "Prec"] = 0

    return temp_forz_def


def _prepare_forcing(forcing_df):
    """Build the FSM2 forcing and configuration from MuSA's forcing table."""
    import modules.met_tools as met

    temp_forz_def = stable_forcing(forcing_df.copy())

    if cfg.precipitation_phase == "Harder":
        Rf, Sf = met.pp_psychrometric(
            temp_forz_def["Ta"].values,
            temp_forz_def["RH"].values,
            temp_forz_def["Prec"].values,
        )
    elif cfg.precipitation_phase == "temp_thld":
        Rf, Sf = met.pp_temp_thld_log(
            temp_forz_def["Ta"].values,
            temp_forz_def["Prec"].values,
        )
    elif cfg.precipitation_phase == "Liston":
        Rf, Sf = met.linear_liston(
            temp_forz_def["Ta"].values,
            temp_forz_def["Prec"].values,
        )
    else:
        raise Exception("Precipitation phase partitioning not implemented")

    # ``pp_*`` returns (liquid, solid) in all three cases.
    forzamiento = Forzamiento(
        year=temp_forz_def["year"].to_numpy(dtype=np.int64),
        month=temp_forz_def["month"].to_numpy(dtype=np.int64),
        day=temp_forz_def["day"].to_numpy(dtype=np.int64),
        hour=temp_forz_def["hours"].to_numpy(dtype=np.float64),
        SW=temp_forz_def["SW"].to_numpy(dtype=np.float64),
        LW=temp_forz_def["LW"].to_numpy(dtype=np.float64),
        Sf=np.asarray(Sf, dtype=np.float64),
        Rf=np.asarray(Rf, dtype=np.float64),
        Ta=temp_forz_def["Ta"].to_numpy(dtype=np.float64),
        RH=temp_forz_def["RH"].to_numpy(dtype=np.float64),
        Ua=temp_forz_def["Ua"].to_numpy(dtype=np.float64),
        Ps=temp_forz_def["Ps"].to_numpy(dtype=np.float64),
    )
    configuracion = _configuracion_from_forcing(temp_forz_def)

    return forzamiento, configuracion


def model_forcing_wrt(forcing_df, temp_dest, step=0):
    """Kept for compatibility: FSM2mem builds forcing inside ``model_run``."""
    return None


# ---------------------------------------------------------------------------
# State dump (restart)
# ---------------------------------------------------------------------------


# Order used by FSM2's out_dump (FSM2.F90).  Row positions are important:
# modules.fsm_tools reads the file in this order, regardless of the labels it
# assigns to the resulting DataFrame.
_DUMP_ROWS = (
    ("albs", "scalar"),
    ("Dsnw", "snow"),
    ("Nsnow", "scalar"),
    ("Qcan", "canopy"),
    ("Rgrn", "snow"),
    ("Sice", "snow"),
    ("Sliq", "snow"),
    ("Sveg", "canopy"),
    ("Tcan", "canopy"),
    ("Tsnow", "snow"),
    ("Tsoil", "soil"),
    ("Tsrf", "scalar"),
    ("Tveg", "canopy"),
    ("Vsmc", "soil"),
)


def _capas(tipo, configuracion=None):
    if tipo == "scalar":
        return 1
    if tipo == "snow":
        return configuracion.nsmax if configuracion is not None else _nsmax()
    if tipo == "soil":
        return configuracion.nsoil if configuracion is not None else _nsoil()
    if tipo == "canopy":
        if configuracion is not None:
            return 1 if int(configuracion.canmod) == 1 else 2
        return _ncnpy()
    raise ValueError(tipo)


def _dump_dataframe(final):
    """Create an out_dump-like DataFrame from the final FSM2 conditions."""
    if final is None:
        raise RuntimeError("No hay condiciones finales para generar el dump")

    filas = []
    for nombre, tipo in _DUMP_ROWS:
        valores = np.asarray(getattr(final, nombre)).ravel()
        filas.append(valores.astype(np.float64, copy=False))

    ancho = max(len(fila) for fila in filas)
    datos = np.full((len(filas), ancho), np.nan, dtype=np.float64)
    for i, fila in enumerate(filas):
        datos[i, : len(fila)] = fila

    return pd.DataFrame(
        datos,
        index=[nombre for nombre, _ in _DUMP_ROWS],
        columns=range(ancho),
    )


def _values_from_dump_row(row):
    """Return the finite values of one dump row, dropping padding NaNs."""
    values = []
    for value in row:
        if pd.isna(value):
            continue
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _condiciones_from_dump(dump, configuracion):
    """Convert an out_dump-like DataFrame back into FSM2 conditions."""
    if dump is None:
        return None
    if len(dump.index) < len(_DUMP_ROWS):
        raise ValueError("El dump de FSM2mem no tiene las 14 filas esperadas")

    # The first row is albs, with one value per point.
    albs = _values_from_dump_row(dump.iloc[0, :].to_numpy())
    if albs.size == 0:
        raise ValueError("El dump de FSM2mem no contiene albs")
    npuntos = albs.size

    valores = {}
    for i, (nombre, tipo) in enumerate(_DUMP_ROWS):
        fila = _values_from_dump_row(dump.iloc[i, :].to_numpy())
        capas = _capas(tipo, configuracion)
        esperado = npuntos * capas
        if fila.size != esperado:
            raise ValueError(
                f"La fila {nombre} del dump tiene {fila.size} valores; "
                f"se esperaban {esperado} ({npuntos} puntos × {capas} capas)"
            )

        if tipo == "scalar":
            valores[nombre] = fila
        else:
            valores[nombre] = fila.reshape(npuntos, capas)

    nsnow = valores["Nsnow"]
    if not np.all(nsnow == np.round(nsnow)):
        raise ValueError("La fila Nsnow del dump contiene valores no enteros")
    valores["Nsnow"] = nsnow.astype(np.int64)

    return Condiciones(**valores)


def write_dump(dump, fsm_path):
    """Kept for compatibility: restart is passed directly to ``model_run``."""
    return None


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------


def _state_dataframe(sim):
    if sim.snd.shape[1] != 1:
        raise ValueError(
            "El backend FSM2mem de MuSA espera un único punto por celda"
        )

    datos = {
        "snd": sim.snd[:, 0],
        "SWE": sim.snw[:, 0],
        "Tsrf": sim.Tsrf[:, 0],
        "fSCA": sim.fsnow[:, 0],
        "alb": sim.albs[:, 0],
        "H": sim.H[:, 0],
        "LE": sim.LE[:, 0],
    }

    if cfg.run_smrt:
        nt = sim.snd.shape[0]
        perfiles = getattr(sim, "perfiles", None)
        if perfiles is None:
            raise RuntimeError(
                "FSM2mem no generó los perfiles de estado necesarios para "
                "SMRT. Revisa Configuracion.guardar_estado."
            )

        def capa(nombre, indice):
            valores = perfiles[nombre][:, 0, :]
            if indice < valores.shape[1]:
                return valores[:, indice]
            return np.full(nt, np.nan)

        for i in range(3):
            datos[f"Rgrn{i + 1}"] = capa("Rgrn", i)

        for i in range(4):
            if i < sim.Tsoil.shape[2]:
                datos[f"Tsoil{i + 1}"] = sim.Tsoil[:, 0, i]
            else:
                datos[f"Tsoil{i + 1}"] = np.full(nt, np.nan)

        for i in range(3):
            datos[f"Tsnow{i + 1}"] = capa("Tsnow", i)

        for i in range(4):
            datos[f"Vsmc{i + 1}"] = capa("Vsmc", i)

        for i in range(3):
            datos[f"Dsnw{i + 1}"] = capa("Dsnw", i)

        for i in range(3):
            datos[f"Sice{i + 1}"] = capa("Sice", i)

        for i in range(3):
            datos[f"Sliq{i + 1}"] = capa("Sliq", i)

    return pd.DataFrame(datos)
