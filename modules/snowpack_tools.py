#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import datetime as dt
import pandas as pd
import config as cfg
import constants as cnt
import modules.met_tools as met
import secrets
import copy
import pdcast as pdc
import warnings
import numpy as np
import netCDF4 as nc
import modules.internal_fns as ifn
from statsmodels.stats.weightstats import DescrStatsW

if cfg.DAsord:
    from modules.user_optional_fns import snd_ord
if cfg.run_smrt:
    import modules.SNOWPACK2SMRT as smrt
# TODO: homogenize documentation format

if cfg.run_smrt:

    smrt_names = smrt.return_col_names()
    if cfg.DAsord:
        model_columns = (
            "snd",
            "SWE",
            *smrt_names,
            *cfg.DAord_names,
        )

    else:
        model_columns = (
            "snd",
            "SWE",
            *smrt_names,
        )
else:
    if cfg.DAsord:
        model_columns = (
            "snd",
            "SWE",
            tuple(cfg.DAord_names),
        )

    else:
        model_columns = ("snd", "SWE")


def model_copy(y_id, x_id):

    to_directory = cfg.tmp_path

    if to_directory is None:
        tmp_dir = tempfile.mkdtemp()
        final_directory = os.path.join(tmp_dir, (str(y_id) + "_" + str(x_id)))
    else:
        token = secrets.token_urlsafe(16)  # safe path to run multiple sesions
        final_directory = os.path.join(
            to_directory, token, (str(y_id) + "_" + str(x_id))
        )
    if os.path.exists(final_directory):
        shutil.rmtree(final_directory, ignore_errors=True)

    os.mkdir(final_directory)
    os.mkdir(os.path.join(final_directory, "input"))
    os.mkdir(os.path.join(final_directory, "output"))
    os.mkdir(os.path.join(final_directory, "RESTARTDATA"))

    return final_directory


def model_compile():
    """
    just check that just check that it can be called
    """

    try:
        subprocess.call(
            ["snowpack"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    except Exception as e:
        print(e)
        print("Error when calling SNOWPACK – is it in the PATH?")


def model_compile_HPC(HPC_task_id):

    # Just check if its in the PATH
    model_compile()


def model_remove(wd_path):
    """
    Remove the temporal wd directory
    Parameters
    ----------
    wd_path : string
        wd temporal location.

    Returns
    -------
    None.

    """
    if os.path.exists(wd_path):
        if cfg.tmp_path is not None:  # Remove the random path
            wd_path = os.path.split(wd_path)[0]

        shutil.rmtree(wd_path, ignore_errors=True)


def forcing_table(lat_idx, lon_idx, step=0):

    nc_forcing_path = cfg.nc_forcing_path
    forcing_var_names = cfg.forcing_var_names
    param_var_names = cfg.param_var_names
    date_ini = cfg.date_ini
    date_end = cfg.date_end
    intermediate_path = cfg.intermediate_path

    # Path to intermediate file
    final_directory = os.path.join(
        intermediate_path, (str(lat_idx) + "_" + str(lon_idx) + ".pkl")
    )

    # try to read the forcing from a dumped file
    if os.path.exists(final_directory) and (
        cfg.restart_forcing
        or (cfg.implementation == "Spatial_propagation" and step != 0)
    ):

        forcing_df = ifn.io_read(final_directory)

    else:

        short_w = ifn.nc_array_forcing(
            nc_forcing_path,
            lat_idx,
            lon_idx,
            forcing_var_names["SW_var_name"],
            date_ini,
            date_end,
        )

        long_wave = ifn.nc_array_forcing(
            nc_forcing_path,
            lat_idx,
            lon_idx,
            forcing_var_names["LW_var_name"],
            date_ini,
            date_end,
        )

        prec = ifn.nc_array_forcing(
            nc_forcing_path,
            lat_idx,
            lon_idx,
            forcing_var_names["Precip_var_name"],
            date_ini,
            date_end,
        )

        temp = ifn.nc_array_forcing(
            nc_forcing_path,
            lat_idx,
            lon_idx,
            forcing_var_names["Temp_var_name"],
            date_ini,
            date_end,
        )

        rel_humidity = ifn.nc_array_forcing(
            nc_forcing_path,
            lat_idx,
            lon_idx,
            forcing_var_names["RH_var_name"],
            date_ini,
            date_end,
        )

        wind = ifn.nc_array_forcing(
            nc_forcing_path,
            lat_idx,
            lon_idx,
            forcing_var_names["Wind_var_name"],
            date_ini,
            date_end,
        )

        if forcing_var_names["Press_var_name"] == "from_DEM":

            with nc.Dataset(cfg.dem_path) as dem:
                topo = dem.variables[cfg.nc_dem_varname][lat_idx, lon_idx]
                sfc_pres = met.pres_from_dem(topo)
                press = np.full_like(wind, sfc_pres)

        else:
            press = ifn.nc_array_forcing(
                nc_forcing_path,
                lat_idx,
                lon_idx,
                forcing_var_names["Press_var_name"],
                date_ini,
                date_end,
            )

        # Search for parameters or use the default settings
        # example
        """
        try:
            vegh = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["vegh_var_name"],
                                        date_ini, date_end)
        except KeyError:
            vegh = np.repeat(cnt.vegh, len(prec))
        """

        date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
        del_t = ifn.generate_dates(date_ini, date_end)

        forcing_df = pd.DataFrame(
            {
                "year": del_t,
                "month": del_t,
                "day": del_t,
                "hours": del_t,
                "SW": short_w,
                "LW": long_wave,
                "Prec": prec,
                "Ta": temp,
                "RH": rel_humidity,
                "Ua": wind,
                "Ps": press,
            }
        )

        forcing_df["year"] = forcing_df["year"].dt.year
        forcing_df["month"] = forcing_df["month"].dt.month
        forcing_df["day"] = forcing_df["day"].dt.day
        forcing_df["hours"] = forcing_df["hours"].dt.hour

        if cfg.run_smrt:
            try:
                k = ifn.nc_array_forcing(
                    nc_forcing_path,
                    lat_idx,
                    lon_idx,
                    param_var_names["k_var_name"],
                    date_ini,
                    date_end,
                )
            except KeyError:
                k = np.repeat(cnt.k, len(prec))

            forcing_df["k"] = k

        forcing_df = unit_conversion(forcing_df)
        if len(del_t) != len(forcing_df.index):
            raise Exception("date_end - date_ini longuer than forcing")

        # write intermediate file to avoid re-reading the nc files
        if cfg.save_int_forcing:
            ifn.io_write(final_directory, forcing_df)

    return forcing_df


def unit_conversion(forcing_df):

    forcing_offset = cnt.forcing_offset
    forcing_multiplier = cnt.forcing_multiplier

    forcing_df.SW = forcing_df.SW * forcing_multiplier["SW"]
    forcing_df.LW = forcing_df.LW * forcing_multiplier["LW"]
    forcing_df.Prec = forcing_df.Prec * forcing_multiplier["Prec"]
    forcing_df.Ta = forcing_df.Ta * forcing_multiplier["Ta"]
    forcing_df.RH = forcing_df.RH * forcing_multiplier["RH"]
    forcing_df.Ua = forcing_df.Ua * forcing_multiplier["Ua"]
    forcing_df.Ps = forcing_df.Ps * forcing_multiplier["Ps"]

    forcing_df.SW = forcing_df.SW + forcing_offset["SW"]
    forcing_df.LW = forcing_df.LW + forcing_offset["LW"]
    forcing_df.Prec = forcing_df.Prec + forcing_offset["Prec"]
    forcing_df.Ta = forcing_df.Ta + forcing_offset["Ta"]
    forcing_df.RH = forcing_df.RH + forcing_offset["RH"]
    forcing_df.Ua = forcing_df.Ua + forcing_offset["Ua"]
    forcing_df.Ps = forcing_df.Ps + forcing_offset["Ps"]

    # Save some space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        forcing_df = pdc.downcast(forcing_df, numpy_dtypes_only=True)

    return forcing_df


def stable_forcing(forcing_df):

    temp_forz_def = forcing_df.copy()

    # Negative SW to 0
    temp_forz_def.loc[temp_forz_def["SW"] < 0, "SW"] = 0

    # Negative LW to 0
    temp_forz_def.loc[temp_forz_def["LW"] < 0, "LW"] = 0

    # Negative Prec to 0
    temp_forz_def.loc[temp_forz_def["Prec"] < 0, "Prec"] = 0

    # Negative wind to 0
    temp_forz_def.loc[temp_forz_def["Ua"] < 0, "Ua"] = 0

    # RH limits
    temp_forz_def.loc[temp_forz_def["RH"] > 100, "RH"] = 100
    temp_forz_def.loc[temp_forz_def["RH"] < 0, "RH"] = 1

    return temp_forz_def


def model_forcing_wrt(forcing_df, wd_path, step=0):

    temp_forz_def = forcing_df.copy()
    temp_forz_def = stable_forcing(temp_forz_def)

    liquid_prec, solid_prec, rain_fr = met.pp_psychrometric(
        temp_forz_def["Ta"].values,
        temp_forz_def["RH"].values,
        temp_forz_def["Prec"].values,
        ret_fr=True,
    )
    temp_forz_def.insert(4, "PSUM_PH", rain_fr)
    temp_forz_def.insert(4, "TSG", np.full_like(rain_fr, 273.15))

    temp_forz_def["Prec"] = temp_forz_def["Prec"] * cfg.dt
    temp_forz_def["RH"] = temp_forz_def["RH"] / 100

    firstdate = pd.Timestamp(
        year=int(temp_forz_def["year"].iloc[0]),
        month=int(temp_forz_def["month"].iloc[0]),
        day=int(temp_forz_def["day"].iloc[0]),
        hour=int(temp_forz_def["hours"].iloc[0]),
    )

    # El forzamiento tiene que emepzar en el timestep anterior:
    firstdate_minus_one_dt = firstdate - pd.Timedelta(hours=cfg.dt / 3600)

    temp_forz_def = pd.concat([temp_forz_def.iloc[[0]], temp_forz_def])

    temp_forz_def.iloc[
        0, temp_forz_def.columns.get_indexer(["year", "month", "day", "hours"])
    ] = [
        firstdate_minus_one_dt.year,
        firstdate_minus_one_dt.month,
        firstdate_minus_one_dt.day,
        firstdate_minus_one_dt.hour,
    ]

    file_name = os.path.join(wd_path, "input", "justacell.smet")
    write_smet(temp_forz_def, file_name)

    if step == 0:  # FSM no, pero snowpack siempre hay qeu dar init con
        write_empty_smet(
            Path(os.path.join(wd_path, "input", "justacell.sno")),
            firstdate.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    params = None

    write_nlst(wd_path, params, step)


def write_smet(
    df: pd.DataFrame,
    output_file: str | Path,
    station_id: str = "justacell",
    station_name: str = "justacell",
    latitude: float = 42.0,
    longitude: float = 0.0,
    altitude: float = 2000.0,
    timezone: float = 0.0,
) -> None:

    # Timestamps como strings solo una vez
    timestamps = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hours"],
        }
    ).dt.strftime("%Y-%m-%dT%H:%M")

    # NumPy es más rápido que iterar sobre filas pandas
    values = (
        df[["TSG", "PSUM_PH", "SW", "LW", "Prec", "Ta", "RH", "Ua", "Ps"]]
        .fillna(-999)
        .to_numpy(dtype=np.float64)
    )

    header = (
        "SMET 1.1 ASCII\n\n"
        "[HEADER]\n"
        f"station_id = {station_id}\n"
        f"station_name = {station_name}\n"
        f"latitude = {latitude}\n"
        f"longitude = {longitude}\n"
        f"altitude = {altitude}\n"
        "nodata = -999\n"
        f"tz = {timezone}\n"
        "fields = timestamp TSG PSUM_PH ISWR ILWR PSUM TA RH VW P\n"
        "[DATA]\n"
    )

    output_file = Path(output_file)

    with output_file.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write(header)

        np.savetxt(
            f,
            np.column_stack((timestamps, values)),
            fmt=[
                "%s",  # timestamp
                "%.2f",  # TSG
                "%.1f",  # PSUM_PH
                "%.1f",  # ISWR
                "%.2f",  # ILWR
                "%.1f",  # PSUM
                "%.2f",  # TA
                "%.1f",  # RH
                "%.2f",  # VW
                "%.2f",  # P
            ],
            delimiter=" ",
        )


def read_smet(filepath, variables=None):
    """
    Read a SMET 1.1 ASCII file into a pandas DataFrame.

    Values in the [DATA] section are corrected using the
    units_multiplier and units_offset values defined in the header:

        corrected_value = raw_value * multiplier + offset

    Parameters
    ----------
    filepath : str
        Path to the .smet file.
    variables : list[str], optional
        Variables to return. If None, all variables are returned.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the SMET data.
    """

    fields = None
    multipliers = None
    offsets = None
    nodata = None

    # Read file and parse header
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line or stripped_line.startswith("#"):
            continue

        if "=" not in stripped_line:
            continue

        key, value = stripped_line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "fields":
            fields = value.split()

        elif key in {"units_multiplier", "multiplier"}:
            multipliers = [float(v) for v in value.split()]

        elif key in {"units_offset", "offset"}:
            offsets = [float(v) for v in value.split()]

        elif key == "nodata":
            nodata = float(value)

    if fields is None:
        raise ValueError("No 'fields' line found in SMET file.")

    # Find [DATA] section
    try:
        data_start = (
            next(
                i
                for i, line in enumerate(lines)
                if line.strip().upper() == "[DATA]"
            )
            + 1
        )
    except StopIteration:
        raise ValueError("No [DATA] section found in SMET file.")

    # Read data
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=data_start,
        names=fields,
        comment="#",
    )

    # Use neutral values if multiplier or offset are not specified
    if multipliers is None:
        multipliers = [1.0] * len(fields)

    if offsets is None:
        offsets = [0.0] * len(fields)

    if len(multipliers) != len(fields):
        raise ValueError(
            "The number of multiplier values does not match the "
            f"number of fields: {len(multipliers)} != {len(fields)}"
        )

    if len(offsets) != len(fields):
        raise ValueError(
            "The number of offset values does not match the "
            f"number of fields: {len(offsets)} != {len(fields)}"
        )

    # Correct values from the [DATA] section
    for field, multiplier, offset in zip(fields, multipliers, offsets):

        # Timestamp and other non-numeric columns remain unchanged
        if not pd.api.types.is_numeric_dtype(df[field]):
            continue

        # Do not transform nodata values
        if nodata is not None:
            valid = df[field].notna() & df[field].ne(nodata)
        else:
            valid = df[field].notna()

        df.loc[valid, field] = df.loc[valid, field] * multiplier + offset

    # Convert timestamp
    """
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    """

    # Select requested variables
    if variables is not None:
        missing = [var for var in variables if var not in df.columns]

        if missing:
            raise ValueError(
                f"Variables not found in SMET file: {missing}\n"
                f"Available variables: {list(df.columns)}"
            )

        df = df[variables]

    return df


def write_empty_smet(
    output_file: str | Path,
    profile_date: str | dt.datetime,
) -> None:
    """
    Escribe un archivo SMET vacío con una fecha de perfil modificable.

    Parameters
    ----------
    output_file : str | Path
        Ruta del archivo SMET que se va a crear.

    profile_date : str | datetime
        Fecha del perfil. Puede proporcionarse como:
        - datetime
        - String con formato 'YYYY-MM-DDTHH:MM:SS'
    """

    if isinstance(profile_date, dt.datetime):
        profile_date = profile_date.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        # Comprueba que el string tenga el formato esperado
        profile_date = dt.datetime.strptime(
            profile_date,
            "%Y-%m-%dT%H:%M:%S",
        ).strftime("%Y-%m-%dT%H:%M:%S")

    header = f"""SMET 1.1 ASCII
[HEADER]
station_id       = justacell
station_name     = justacell
longitude        = 0.00
latitude         = 42.00
altitude         = 2000.0
nodata           = -999
tz               = 0.0
ProfileDate      = {profile_date}
HS_Last          = 0.0000
SlopeAngle       = 0.0
SlopeAzi         = 0.0
nSoilLayerData   = 0
nSnowLayerData   = 0
SoilAlbedo       = 0.09
BareSoil_z0      = 0.200
CanopyHeight     = 0.00
CanopyLeafAreaIndex = 0.00
CanopyDirectThroughfall = 1.00
WindScalingFactor = 1.00
ErosionLevel     = 0
TimeCountDeltaHS = 0.000000
fields           = timestamp Layer_Thick  T  Vol_Frac_I  Vol_Frac_W  Vol_Frac_V  Vol_Frac_S Rho_S Conduc_S HeatCapac_S  rg  rb  dd  sp  mk mass_hoar ne CDot metamo
[DATA]
"""

    Path(output_file).write_text(header, encoding="utf-8")


def write_nlst(wd_path, params=None, step=None):

    content = f"""[General]
    BUFFER_SIZE = 370
    BUFF_BEFORE = 1.5
    BUFF_GRIDS = 10
    
    [Input]
    COORDSYS = CH1903
    TIME_ZONE = 0.00
    METEO = SMET
    METEOPATH = ./input
    STATION1 = justacell
    SNOWFILE1 = justacell
    
    [InputEditing]
    ENABLE_TIMESERIES_EDITING = FALSE
    
    [Output]
    COORDSYS = CH1903
    TIME_ZONE = 1
    METEO\t= SMET
    METEOPATH = ./output
    WRITE_PROCESSED_METEO = FALSE
    EXPERIMENT = EXP
    SNOW_WRITE = TRUE
    SNOW = SMET
    SNOWPATH = RESTARTDATA
    SNOW_DAYS_BETWEEN = 9999.000000
    FIRST_BACKUP = 9999.000000
    PROF_WRITE = {cfg.run_smrt}
    PROF_DAYS_BETWEEN = 4.1666e-2
    TS_WRITE = TRUE
    TS_FORMAT = SMET
    ACDD_WRITE = FALSE
    TS_START = 0
    TS_DAYS_BETWEEN = 4.1666e-2
    AVGSUM_TIME_SERIES = TRUE
    CUMSUM_MASS = FALSE
    PRECIP_RATES = FALSE
    OUT_CANOPY = FALSE
    OUT_HAZ = FALSE
    OUT_SOILEB = FALSE
    OUT_HEAT = FALSE
    OUT_T = FALSE
    OUT_LW = FALSE
    OUT_SW = TRUE
    OUT_MASS = TRUE
    OUT_METEO = TRUE
    OUT_STAB = FALSE
    
    [Snowpack]
    CALCULATION_STEP_LENGTH = 60.000
    ROUGHNESS_LENGTH = 0.001
    HEIGHT_OF_METEO_VALUES = 2.0
    HEIGHT_OF_WIND_VALUE = 10.0
    ENFORCE_MEASURED_SNOW_HEIGHTS = FALSE
    SW_MODE = INCOMING
    ATMOSPHERIC_STABILITY = MO_HOLTSLAG
    CANOPY = FALSE
    MEAS_TSS = FALSE
    CHANGE_BC = FALSE
    SNP_SOIL = FALSE
    
    [SnowpackAdvanced]
    FIXED_POSITIONS = 0.25 0.5 1.0 -0.25 -0.10
    SNOW_EROSION = TRUE
    COMBINE_ELEMENTS = TRUE
    REDUCE_N_ELEMENTS = TRUE
    HEIGHT_NEW_ELEM = 0.03
    MINIMUM_L_ELEMENT = 0.01
    """
    filename = Path(os.path.join(wd_path, "io_justacell.ini"))
    with open(filename, "w") as f:
        f.write(content)


def read_snowpack_profile(filename, variables, rename=None):
    """
    Lee un perfil .pro de SNOWPACK y devuelve un DataFrame.

    Parameters
    ----------
    filename : str
        Ruta al fichero .pro.

    variables : list
        Variables a extraer. Puede contener códigos ('0502')
        o nombres ('element density').

    rename : dict, optional
        Diccionario para renombrar variables.
        Ejemplo:
        {
            "0502": "density",
            "0503": "temperature"
        }

    Returns
    -------
    pd.DataFrame
        Índice temporal y columnas variable_capa.
    """

    if rename is None:
        rename = {}

    # ------------------------------------------------------------------
    # Leer cabecera y construir mapa código -> descripción
    # ------------------------------------------------------------------
    code_to_name = {}

    with open(filename, "r") as f:

        in_header = False

        for line in f:

            line = line.strip()

            if line == "[HEADER]":
                in_header = True
                continue

            if line == "[DATA]":
                break

            if not in_header:
                continue

            parts = line.split(",", 2)

            if len(parts) < 3:
                continue

            code = parts[0]
            description = parts[2]

            code_to_name[code] = description

    # ------------------------------------------------------------------
    # Resolver las variables pedidas
    # ------------------------------------------------------------------
    variable_codes = []

    for var in variables:

        if var in code_to_name:
            variable_codes.append(var)
            continue

        matches = [
            code
            for code, desc in code_to_name.items()
            if var.lower() in desc.lower()
        ]

        if len(matches) == 0:
            raise ValueError(f"No encuentro variable '{var}'")

        variable_codes.append(matches[0])

    variable_codes = list(dict.fromkeys(variable_codes))

    # ------------------------------------------------------------------
    # Crear nombres finales
    # ------------------------------------------------------------------
    code_to_output_name = {}

    for code in variable_codes:

        if code in rename:
            code_to_output_name[code] = rename[code]
        else:
            code_to_output_name[code] = code

    # ------------------------------------------------------------------
    # Leer datos
    # ------------------------------------------------------------------
    rows = []

    current_time = None
    current_data = {}

    with open(filename, "r") as f:

        in_data = False

        for line in f:

            line = line.strip()

            if line == "[DATA]":
                in_data = True
                continue

            if not in_data:
                continue

            parts = line.split(",")

            code = parts[0]

            # Nuevo timestep
            if code == "0500":

                if current_time is not None:
                    row = {"timestamp": current_time}
                    row.update(current_data)
                    rows.append(row)

                current_time = pd.to_datetime(
                    parts[1],
                    format="%d.%m.%Y %H:%M:%S",
                )

                current_data = {}
                continue

            if code not in variable_codes:
                continue

            if len(parts) < 3:
                continue

            n_elems = int(parts[1])
            values = parts[2:]

            base_name = code_to_output_name[code]

            for i in range(min(n_elems, len(values))):

                colname = f"{base_name}_layer{i + 1}"

                try:
                    value = float(values[i])
                except ValueError:
                    value = values[i]

                current_data[colname] = value

    # último timestep
    if current_time is not None:

        row = {"timestamp": current_time}
        row.update(current_data)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp")

    return df


def model_read_output(wd_path, step, read_dump=True):
    """
    Read snowpack outputs and return it in a dataframe

    Parameters:

    wd_path : str
        Location of snowpack outputs

    """
    directory = os.path.join(wd_path, "output/")
    directory = Path(directory)

    files = list(directory.glob("*.smet"))

    if len(files) > 1:
        raise Exception("Too many .smet files in output")

    state = read_smet(files[0], variables=["HS_mod", "SWE"])
    # Traslate into MuSA (vars and units)
    state.columns = ["snd", "SWE"]

    if cfg.run_smrt:

        file_pro = list(directory.glob("*.pro"))
        df = read_snowpack_profile(
            file_pro[0],
            variables=[
                "0501",  # height (cm)
                "0502",  # density (kg m-3)
                "0503",  # Temp (C)
                "0506",  # liquid water content by volume (%)
                "0535",  # optical equivalent grain size (mm)
            ],
            rename={
                "0501": "Dsnw",
                "0502": "rhosnw",
                "0503": "Tsnow",
                "0506": "lWE",
                "0535": "Rgrnm",
            },
        )
        # Convertir unodades

        cols = df.filter(like="Dsnw").columns
        df[cols] = df[cols] / 100

        cols = df.filter(like="Tsnow").columns
        df[cols] = df[cols] + 273.15

        cols = df.filter(like="Rgrn").columns
        df[cols] = (df[cols] / 2) / 1000

        cols = df.filter(like="lWE").columns
        df[cols] = df[cols] / 100

        state = pd.concat(
            [state.reset_index(drop=True), df.reset_index(drop=True)], axis=1
        )

    # Hay que  quitar la fila de solape en las siguientes simulaciones
    if step > 0:
        state = state.iloc[1:]
    # add optional variables
    if cfg.DAsord:
        state = snd_ord(state)

    if read_dump:
        directory = os.path.join(wd_path, "RESTARTDATA/")
        directory = Path(directory)
        files = list(directory.glob("*.sno"))
        if len(files) > 1:
            raise Exception("Too many .sno (restart) files in output")

        with open(files[0], "r") as f:
            dump = f.readlines()

    # Clean Directories
    folder = Path(os.path.join(wd_path, "output/"))
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()

    folder = Path(os.path.join(wd_path, "RESTARTDATA/"))
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()

    folder = Path(os.path.join(wd_path, "input/"))
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()

    if read_dump:
        return state, dump
    else:
        return state


def write_dump(dump, wd_path):

    file_name = os.path.join(wd_path, "input", "justacell.sno")

    with open(file_name, "w") as f:

        f.writelines(dump)


def storeDA(
    Result_df, step_results, observations_sbst, error_sbst, time_dict, step
):

    vars_to_perturbate = cfg.vars_to_perturbate
    var_to_assim = cfg.var_to_assim
    error_names = cfg.obs_error_var_names

    rowIndex = Result_df.index[
        time_dict["Assimilation_steps"][step] : time_dict[
            "Assimilation_steps"
        ][step + 1]
    ]

    if len(var_to_assim) > 1:
        for i, var in enumerate(var_to_assim):
            Result_df.loc[rowIndex, var] = observations_sbst[:, i]
            Result_df.loc[rowIndex, error_names[i]] = error_sbst[:, i]
    else:
        var = var_to_assim[0]
        Result_df.loc[rowIndex, var] = observations_sbst
        Result_df.loc[rowIndex, error_names] = error_sbst

    # Add perturbation parameters to Results
    for var_p in vars_to_perturbate:
        Result_df.loc[rowIndex, var_p + "_noise_mean"] = step_results[
            var_p + "_noise_mean"
        ]
        Result_df.loc[rowIndex, var_p + "_noise_sd"] = step_results[
            var_p + "_noise_sd"
        ]


def storeOL(OL_FSM, Ensemble, observations_sbst, time_dict, step):

    ol_data = Ensemble.origin_state.copy()

    # TODO: modify directly FSM code to not to output time id's

    # Store colums
    for n, name_col in enumerate(ol_data.columns):
        OL_FSM[name_col] = ol_data.iloc[:, [n]].to_numpy()


def store_sim(
    sim_stat, Ensemble, time_dict, step, MCMC=False, save_prior=False
):

    if MCMC:
        list_state = copy.deepcopy(Ensemble.state_members_mcmc)
    else:
        list_state = copy.deepcopy(Ensemble.state_membres)
    # remove time ids fomr FSM output
    # TODO: modify directly FSM code to not to output time id's

    rowIndex = sim_stat["mean"].index[
        time_dict["Assimilation_steps"][step] : time_dict[
            "Assimilation_steps"
        ][step + 1]
    ]

    if save_prior:
        pesos = np.ones_like(Ensemble.wgth)
    else:
        pesos = Ensemble.wgth

    for n, name_col in enumerate(list(list_state[0].columns)):

        # create matrix of colums
        col_arr = [
            list_state[x].iloc[:, n].to_numpy() for x in range(len(list_state))
        ]
        col_arr = np.vstack(col_arr)

        d1 = DescrStatsW(col_arr, weights=pesos)

        if len(sim_stat.keys()) == 2:  # Mean, Std

            sim_stat["mean"].loc[rowIndex, name_col] = d1.mean
            sim_stat["std"].loc[rowIndex, name_col] = d1.std
        else:
            perc = d1.quantile([0, 0.25, 0.5, 0.75, 1]).values
            sim_stat["min"].loc[rowIndex, name_col] = perc[0, :]
            sim_stat["Q1"].loc[rowIndex, name_col] = perc[1, :]
            sim_stat["median"].loc[rowIndex, name_col] = perc[2, :]
            sim_stat["Q3"].loc[rowIndex, name_col] = perc[3, :]
            sim_stat["max"].loc[rowIndex, name_col] = perc[4, :]
            sim_stat["mean"].loc[rowIndex, name_col] = d1.mean
            sim_stat["std"].loc[rowIndex, name_col] = d1.std
    return sim_stat


def init_result(del_t, DA=False, OL=False):

    if DA:
        # Concatenate
        col_names = ["Date"]

        # Create results dataframe
        Results = pd.DataFrame(
            np.nan, index=range(len(del_t)), columns=col_names
        )

        Results["Date"] = [x.strftime("%d/%m/%Y-%H:%S") for x in del_t]
        return Results

    else:

        # Create results dataframe
        Results = pd.DataFrame(
            np.nan, index=range(len(del_t)), columns=model_columns
        )

        Results["Date"] = [x.strftime("%d/%m/%Y-%H:%S") for x in del_t]
        # Reordenar las columnas para que 'Date' sea la primera
        cols = ["Date"] + [col for col in Results if col != "Date"]
        Results = Results[cols]

        if cfg.write_stat_full:
            stat_name_list = [
                "min",
                "max",
                "Q1",
                "Q3",
                "median",
                "mean",
                "std",
            ]
        else:
            stat_name_list = ["mean", "std"]

        sim_stat = {key: Results.copy() for key in stat_name_list}

        if OL:
            return sim_stat["mean"]

        return sim_stat


def get_var_state_position(var):

    state_columns = model_columns

    return state_columns.index(var)


def get_last_date_smet(filename):
    with open(filename, "rb") as f:
        f.seek(0, 2)  # final del archivo

        pos = f.tell() - 1
        line = b""

        while pos >= 0:
            f.seek(pos)
            char = f.read(1)

            if char == b"\n" and line:
                break

            line = char + line
            pos -= 1

    timestamp = line.decode().strip().split()[0]
    return pd.to_datetime(timestamp)


def model_run(wd_path):
    """
    Just run FSM in a directory

    Parameters:

    fsm_path : str
        Location of FSM binary

    Returns:

    [None]

    """

    file_name = os.path.join(wd_path, "io_justacell.ini")
    end_date = get_last_date_smet(
        os.path.join(wd_path, "input", "justacell.smet")
    )

    try:
        subprocess.run(
            [
                "snowpack",
                "-c",
                file_name,
                "-e",
                end_date.strftime("%Y-%m-%dT%H:%M"),
            ],
            cwd=wd_path,  # <-- directorio de trabajo
            capture_output=True,
            text=True,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Snowpack terminó con código {e.returncode}\n\n"
            f"STDERR:\n{e.stderr}\n\n"
            f"STDOUT:\n{e.stdout}"
        )

    except Exception as e:
        raise Exception(f"Error al ejecutar Snowpack: {e}")
