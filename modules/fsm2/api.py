"""In-memory Python API for the Flexible Snow Model FSM2.

The public entry point is :func:`fsm2`::

    simulacion, condiciones_finales = fsm2(forzamiento, condiciones_iniciales,
                                           configuracion)

All three arguments are Python objects and no meteorological or state data
are read from or written to disk.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from typing import Optional, Sequence, Union

import numpy as np

from ._constants import (
    E0,
    EPS,
    HCON_AIR,
    HCON_CLAY,
    HCON_SAND,
    I0,
    PI,
    TM,
)
from ._params import Parametros
from ._solar import solar_elevation
from ._timestep import fsm2_timestep
from ._loop import fsm2_run

ArrayLike = Union[float, Sequence[float], np.ndarray]

_DAYS_IN_MONTH = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64
)


@dataclass
class Forzamiento:
    """Meteorological forcing, time first.

    Each array has shape ``(ntiempo, npuntos)``; one-dimensional arrays are
    interpreted as a time series for a single point.  ``SW``, ``LW``, ``Sf``,
    ``Rf``, ``Ta``, ``RH``, ``Ua`` and ``Ps`` are the standard FSM driving
    variables.  ``Sdif`` and ``Sdir`` may be supplied instead of ``SW`` when
    the diffuse/direct partition is already known.

    ``year``, ``month``, ``day`` and ``hour`` are optional timestamp arrays
    of length ``ntiempo``.  If they are omitted, timestamps are generated
    from ``Configuracion.fecha_inicio`` and ``Configuracion.dt``.
    """

    SW: Optional[np.ndarray] = None
    LW: Optional[np.ndarray] = None
    Sf: Optional[np.ndarray] = None
    Rf: Optional[np.ndarray] = None
    Ta: Optional[np.ndarray] = None
    RH: Optional[np.ndarray] = None
    Ua: Optional[np.ndarray] = None
    Ps: Optional[np.ndarray] = None
    Sdif: Optional[np.ndarray] = None
    Sdir: Optional[np.ndarray] = None
    year: Optional[np.ndarray] = None
    month: Optional[np.ndarray] = None
    day: Optional[np.ndarray] = None
    hour: Optional[np.ndarray] = None


@dataclass
class Condiciones:
    """FSM2 state variables, point first.

    Arrays have shapes ``(npuntos,)``, ``(npuntos, nsmax)``,
    ``(npuntos, ncnpy)`` or ``(npuntos, nsoil)``.  Any field left as ``None``
    is filled with the FSM2 default initial state, so
    ``Condiciones()`` is a valid default initial condition.
    """

    albs: Optional[np.ndarray] = None
    Tsrf: Optional[np.ndarray] = None
    Dsnw: Optional[np.ndarray] = None
    Nsnow: Optional[np.ndarray] = None
    Qcan: Optional[np.ndarray] = None
    Rgrn: Optional[np.ndarray] = None
    Sice: Optional[np.ndarray] = None
    Sliq: Optional[np.ndarray] = None
    Sveg: Optional[np.ndarray] = None
    Tcan: Optional[np.ndarray] = None
    Tsnow: Optional[np.ndarray] = None
    Tsoil: Optional[np.ndarray] = None
    Tveg: Optional[np.ndarray] = None
    Vsmc: Optional[np.ndarray] = None


# Descriptive aliases for the two ends of a simulation.
CondicionesIniciales = Condiciones
CondicionesFinales = Condiciones


@dataclass
class Configuracion:
    """Site, grid, physics options and parameter values.

    The defaults reproduce ``compil.sh`` and ``FSM2_PARAMS.F90``.  Vegetation
    properties (``alb0``, ``vegh``, ``VAI``) may be scalars or arrays of
    length ``npuntos``.  ``fsat`` and ``Tprf`` set up the default initial soil
    moisture and temperature profiles.
    """

    # Site and time
    dt: float = 3600.0
    lat: float = 0.0
    noon: float = 12.0
    zT: float = 2.0
    zU: float = 10.0
    trans: float = 0.0
    fecha_inicio: Union[str, datetime] = "2004-10-01 01:00"

    # Grid
    nsmax: int = 3
    nsoil: int = 4
    dzsnow: Optional[Sequence[float]] = None
    dzsoil: Optional[Sequence[float]] = None
    fvg1: float = 0.5
    zsub: float = 1.5

    # Vegetation characteristics (per point, or scalar)
    alb0: ArrayLike = 0.2
    vegh: ArrayLike = 0.0
    VAI: ArrayLike = 0.0

    # Default initial soil profiles
    fsat: ArrayLike = 0.5
    Tprf: ArrayLike = 285.0

    # Compile-time options (same values as compil.sh)
    albedo: int = 2
    canint: int = 1
    canmod: int = 1
    canrad: int = 1
    canunl: int = 1
    condct: int = 1
    densty: int = 1
    exchng: int = 1
    hydrol: int = 1
    sgrain: int = 1
    snfrac: int = 1
    swpart: int = 0
    zoffst: int = 0
    ensmbl: int = 0

    # Vegetation parameters
    acn0: float = 0.1
    acns: float = 0.3
    avg0: float = 0.27
    avgs: float = 0.65
    cvai: float = 3.6e4
    eunl: float = 240.0 * 3600.0
    gsnf: float = 0.01
    hbas: float = 2.0
    kext: float = 0.5
    leaf: float = 20.0
    munl: float = 0.4
    svai: float = 4.4
    Tunl: float = 1.87e5
    Uunl: float = 1.56e5
    wcan: float = 2.5

    # Snow parameters
    asmn: float = 0.5
    asmx: float = 0.85
    eta0: float = 3.7e7
    hfsn: float = 0.1
    kfix: float = 0.24
    nhyd: int = 10
    rcld: float = 300.0
    rfix: float = 300.0
    rgr0: float = 5e-5
    rhof: float = 100.0
    rhow: float = 300.0
    rmlt: float = 500.0
    Salb: float = 10.0
    snda: float = 2.8e-6
    Talb: float = -2.0
    tcld: float = 1000.0 * 3600.0
    tmlt: float = 100.0 * 3600.0
    trho: float = 200.0 * 3600.0
    Wirr: float = 0.03
    z0sn: float = 0.001

    # Ground and soil parameters
    fcly: float = 0.3
    fsnd: float = 0.6
    gsat: float = 0.01
    z0sf: float = 0.1

    # Ensemble parameters
    Pmlt: float = 1.0
    Tadd: float = 0.0

    # Optional per-timestep state profiles.  They are only needed by
    # applications (such as MuSA/SMRT) that require the full snow, canopy
    # and soil state at every output time.  They are disabled by default to
    # keep the output small and to avoid unnecessary copies.
    guardar_estado: bool = False


@dataclass
class Simulacion:
    """Model output time series.

    Diagnostic arrays have shape ``(ntiempo, npuntos)``.  Profiles have shape
    ``(ntiempo, npuntos, nsmax)``, ``(ntiempo, npuntos, nsoil)`` or
    ``(ntiempo, npuntos, ncnpy)``.

    If ``Configuracion.guardar_estado`` is true, the object also exposes the
    per-timestep state profiles ``Dsnw``, ``Rgrn``, ``Sice``, ``Sliq``,
    ``Tsnow``, ``Qcan``, ``Sveg``, ``Tcan`` and ``Vsmc``, and the dictionary
    ``perfiles`` that contains them.
    """

    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    hour: np.ndarray
    tiempo: np.ndarray
    fsnow: np.ndarray
    albs: np.ndarray
    Nsnow: np.ndarray
    H: np.ndarray
    LE: np.ndarray
    LWout: np.ndarray
    LWsub: np.ndarray
    Melt: np.ndarray
    Roff: np.ndarray
    snd: np.ndarray
    snw: np.ndarray
    subl: np.ndarray
    svg: np.ndarray
    SWout: np.ndarray
    SWsub: np.ndarray
    Tsub: np.ndarray
    Usub: np.ndarray
    Tsoil: np.ndarray
    Tsrf: np.ndarray
    Tveg: np.ndarray
    Wflx: np.ndarray

    def __getitem__(self, key):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self):
        """Return all output arrays in a plain dictionary."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _validate_options(cfg):
    allowed = {
        "albedo": (1, 2),
        "canint": (1, 2),
        "canmod": (1, 2),
        "canrad": (1, 2),
        "canunl": (1, 2),
        "condct": (0, 1),
        "densty": (0, 1, 2),
        "exchng": (0, 1),
        "hydrol": (0, 1, 2),
        "sgrain": (0, 1, 2),
        "snfrac": (1, 2, 3),
        "swpart": (0, 1),
        "zoffst": (0, 1),
        "ensmbl": (0, 1),
    }
    for name, values in allowed.items():
        value = int(getattr(cfg, name))
        if value not in values:
            raise ValueError(
                f"Configuracion.{name} debe ser uno de {values}, recibido {value}"
            )
    if int(cfg.nhyd) < 1:
        raise ValueError("Configuracion.nhyd debe ser >= 1")
    if float(cfg.dt) <= 0.0:
        raise ValueError("Configuracion.dt debe ser > 0")


def _grid(cfg):
    nsmax = int(cfg.nsmax)
    nsoil = int(cfg.nsoil)
    if nsmax < 1:
        raise ValueError("Configuracion.nsmax debe ser >= 1")
    if nsoil < 2:
        raise ValueError("Configuracion.nsoil debe ser >= 2")

    if cfg.dzsnow is None:
        base = np.array([0.1, 0.2, 0.4], dtype=np.float64)
        if nsmax > base.size:
            raise ValueError(
                "Para nsmax > 3 debe indicar Configuracion.dzsnow"
            )
        dzsnow = base[:nsmax].copy()
    else:
        dzsnow = np.ascontiguousarray(cfg.dzsnow, dtype=np.float64).ravel()
        if dzsnow.size != nsmax:
            raise ValueError(
                f"Configuracion.dzsnow debe tener {nsmax} elementos"
            )

    if cfg.dzsoil is None:
        base = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
        if nsoil > base.size:
            raise ValueError(
                "Para nsoil > 4 debe indicar Configuracion.dzsoil"
            )
        dzsoil = base[:nsoil].copy()
    else:
        dzsoil = np.ascontiguousarray(cfg.dzsoil, dtype=np.float64).ravel()
        if dzsoil.size != nsoil:
            raise ValueError(
                f"Configuracion.dzsoil debe tener {nsoil} elementos"
            )

    if np.any(dzsnow <= 0.0) or np.any(dzsoil <= 0.0):
        raise ValueError("Los espesores de capa deben ser positivos")
    return nsmax, nsoil, dzsnow, dzsoil


def _soil_properties(cfg):
    fcly = float(cfg.fcly)
    fsnd = float(cfg.fsnd)
    if fcly < 0.0 or fsnd < 0.0 or fcly + fsnd <= 0.0:
        raise ValueError("Las fracciones de arcilla y arena no son válidas")
    b = 3.1 + 15.7 * fcly - 0.3 * fsnd
    hcap_soil = (2.128 * fcly + 2.385 * fsnd) * 1e6 / (fcly + fsnd)
    sathh = 10.0 ** (0.17 - 0.63 * fcly - 1.58 * fsnd)
    Vsat = 0.505 - 0.037 * fcly - 0.142 * fsnd
    Vcrit = Vsat * (sathh / 3.364) ** (1.0 / b)
    # Keep Fortran's operator precedence: the final exponent applies only
    # to hcon_sand**(1-fcly), not to the whole clay/sand product.
    hcon_soil = (HCON_AIR ** Vsat) * (
        (HCON_CLAY ** fcly)
        * (HCON_SAND ** (1.0 - fcly)) ** (1.0 - Vsat)
    )
    return b, hcap_soil, hcon_soil, sathh, Vcrit, Vsat


def _build_params(cfg):
    _validate_options(cfg)
    nsmax, nsoil, dzsnow, dzsoil = _grid(cfg)
    canmod = int(cfg.canmod)
    ncnpy = 1 if canmod == 1 else 2
    b, hcap_soil, hcon_soil, sathh, Vcrit, Vsat = _soil_properties(cfg)
    rhof = float(cfg.rfix) if int(cfg.densty) == 0 else float(cfg.rhof)

    return Parametros(
        acn0=float(cfg.acn0),
        acns=float(cfg.acns),
        avg0=float(cfg.avg0),
        avgs=float(cfg.avgs),
        cvai=float(cfg.cvai),
        eunl=float(cfg.eunl),
        gsnf=float(cfg.gsnf),
        hbas=float(cfg.hbas),
        kext=float(cfg.kext),
        leaf=float(cfg.leaf),
        munl=float(cfg.munl),
        svai=float(cfg.svai),
        Tunl=float(cfg.Tunl),
        Uunl=float(cfg.Uunl),
        wcan=float(cfg.wcan),
        asmn=float(cfg.asmn),
        asmx=float(cfg.asmx),
        eta0=float(cfg.eta0),
        hfsn=float(cfg.hfsn),
        kfix=float(cfg.kfix),
        rcld=float(cfg.rcld),
        rfix=float(cfg.rfix),
        rgr0=float(cfg.rgr0),
        rhof=rhof,
        rhow=float(cfg.rhow),
        rmlt=float(cfg.rmlt),
        Salb=float(cfg.Salb),
        snda=float(cfg.snda),
        Talb=float(cfg.Talb),
        tcld=float(cfg.tcld),
        tmlt=float(cfg.tmlt),
        trho=float(cfg.trho),
        Wirr=float(cfg.Wirr),
        z0sn=float(cfg.z0sn),
        fcly=float(cfg.fcly),
        fsnd=float(cfg.fsnd),
        gsat=float(cfg.gsat),
        z0sf=float(cfg.z0sf),
        b=b,
        hcap_soil=hcap_soil,
        hcon_soil=hcon_soil,
        sathh=sathh,
        Vcrit=Vcrit,
        Vsat=Vsat,
        Pmlt=float(cfg.Pmlt),
        Tadd=float(cfg.Tadd),
        ncnpy=ncnpy,
        nsmax=nsmax,
        nsoil=nsoil,
        nhyd=int(cfg.nhyd),
        albedo=int(cfg.albedo),
        canint=int(cfg.canint),
        canmod=canmod,
        canrad=int(cfg.canrad),
        canunl=int(cfg.canunl),
        condct=int(cfg.condct),
        densty=int(cfg.densty),
        exchng=int(cfg.exchng),
        hydrol=int(cfg.hydrol),
        sgrain=int(cfg.sgrain),
        snfrac=int(cfg.snfrac),
        swpart=int(cfg.swpart),
        zoffst=int(cfg.zoffst),
        ensmbl=int(cfg.ensmbl),
        fvg1=float(cfg.fvg1),
        zsub=float(cfg.zsub),
        dzsnow=dzsnow,
        dzsoil=dzsoil,
    )


def _broadcast_vector(value, n, name):
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(n, arr[0], dtype=np.float64)
    if arr.size == n:
        return np.ascontiguousarray(arr, dtype=np.float64)
    raise ValueError(f"{name} debe ser un escalar o tener {n} elementos")


def _as_2d(value, name, nt, npuntos):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((nt, npuntos), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == 1:
            return np.full((nt, npuntos), arr[0], dtype=np.float64)
        if arr.size == nt and npuntos == 1:
            return np.ascontiguousarray(arr.reshape(nt, 1))
        if arr.size == npuntos:
            return np.broadcast_to(arr, (nt, npuntos)).copy()
        if arr.size == nt:
            return np.broadcast_to(arr[:, None], (nt, npuntos)).copy()
        raise ValueError(
            f"{name} tiene longitud {arr.size}; se esperaba {nt} o {npuntos}"
        )
    if arr.ndim == 2:
        if arr.shape[0] != nt:
            raise ValueError(
                f"{name} tiene {arr.shape[0]} tiempos; se esperaba {nt}"
            )
        if arr.shape[1] == npuntos:
            return np.ascontiguousarray(arr, dtype=np.float64)
        if arr.shape[1] == 1:
            return np.broadcast_to(arr, (nt, npuntos)).copy()
        raise ValueError(
            f"{name} tiene {arr.shape[1]} puntos; se esperaba {npuntos}"
        )
    raise ValueError(f"{name} debe ser un escalar, vector o matriz")


def _prepare_time(forz, cfg, nt):
    provided = [
        getattr(forz, "year", None),
        getattr(forz, "month", None),
        getattr(forz, "day", None),
        getattr(forz, "hour", None),
    ]
    n_provided = sum(value is not None for value in provided)
    if n_provided not in (0, 4):
        raise ValueError(
            "Forzamiento debe incluir year, month, day y hour, o ninguno"
        )

    if n_provided == 4:
        year = np.asarray(forz.year, dtype=np.int64).ravel()
        month = np.asarray(forz.month, dtype=np.int64).ravel()
        day = np.asarray(forz.day, dtype=np.int64).ravel()
        hour = np.asarray(forz.hour, dtype=np.float64).ravel()
        if not (year.size == month.size == day.size == hour.size == nt):
            raise ValueError(
                "Los vectores de tiempo deben tener longitud ntiempo"
            )
        # Fast path: build the datetime64 series with vectorized NumPy
        # arithmetic.  Invalid dates fall back to the scalar loop below so
        # that they raise the same ValueError as before.
        in_range = (
            np.all((year >= 1) & (year <= 9999))
            and np.all((month >= 1) & (month <= 12))
            and np.all(np.isfinite(hour))
            and np.all(np.abs(hour) <= 1.0e6)
        )
        if in_range:
            days_in_month = _DAYS_IN_MONTH[month - 1]
            leap = ((year % 4 == 0) & (year % 100 != 0)) | (
                year % 400 == 0
            )
            days_in_month = days_in_month + ((month == 2) & leap)
            if np.all((day >= 1) & (day <= days_in_month)):
                base = np.datetime64("1970", "Y") + (
                    year - 1970
                ).astype("timedelta64[Y]")
                months = base + (
                    month.astype("timedelta64[M]") - np.timedelta64(1, "M")
                )
                days = months.astype("datetime64[D]") + (
                    day.astype("timedelta64[D]") - np.timedelta64(1, "D")
                )
                micros = np.rint(hour * 3.6e9).astype(np.int64)
                tiempo = (
                    days.astype("datetime64[us]")
                    + micros.astype("timedelta64[us]")
                ).astype("datetime64[s]")
                return year, month, day, hour, tiempo

        tiempo = np.empty(nt, dtype="datetime64[s]")
        for i in range(nt):
            stamp = datetime(
                int(year[i]), int(month[i]), int(day[i])
            ) + timedelta(hours=float(hour[i]))
            tiempo[i] = np.datetime64(stamp.replace(microsecond=0), "s")
        return year, month, day, hour, tiempo

    if isinstance(cfg.fecha_inicio, str):
        start = datetime.fromisoformat(cfg.fecha_inicio)
    elif isinstance(cfg.fecha_inicio, datetime):
        start = cfg.fecha_inicio
    else:
        raise TypeError("Configuracion.fecha_inicio debe ser texto o datetime")

    current = start.date()
    hour_value = (
        start.hour + start.minute / 60.0 + start.second / 3600.0
    )
    year = np.empty(nt, dtype=np.int64)
    month = np.empty(nt, dtype=np.int64)
    day = np.empty(nt, dtype=np.int64)
    hour = np.empty(nt, dtype=np.float64)
    for i in range(nt):
        year[i] = current.year
        month[i] = current.month
        day[i] = current.day
        hour[i] = hour_value
        hour_value += float(cfg.dt) / 3600.0
        while hour_value >= 24.0:
            hour_value -= 24.0
            current = current + timedelta(days=1)

    tiempo = np.array(
        [
            np.datetime64(
                start + timedelta(seconds=i * float(cfg.dt)), "s"
            )
            for i in range(nt)
        ],
        dtype="datetime64[s]",
    )
    return year, month, day, hour, tiempo


def _partition_sw(cfg, nt, npuntos, SW, Sdif, Sdir, year, month, day, hour):
    lat = float(cfg.lat) * PI / 180.0
    if Sdif is not None and Sdir is not None:
        sdif = _as_2d(Sdif, "Sdif", nt, npuntos)
        sdir = _as_2d(Sdir, "Sdir", nt, npuntos)
        if int(cfg.swpart) == 1:
            elev = np.array(
                [
                    solar_elevation(
                        year[i], month[i], day[i], hour[i],
                        lat, float(cfg.noon)
                    )
                    for i in range(nt)
                ],
                dtype=np.float64,
            )
        else:
            elev = np.zeros(nt, dtype=np.float64)
        return sdif, sdir, elev

    if SW is None:
        raise ValueError(
            "Forzamiento necesita SW o, alternativamente, Sdif y Sdir"
        )
    sw = _as_2d(SW, "SW", nt, npuntos)
    if int(cfg.swpart) == 0:
        return sw.copy(), np.zeros_like(sw), np.zeros(nt, dtype=np.float64)

    elev = np.array(
        [
            solar_elevation(
                year[i], month[i], day[i], hour[i], lat, float(cfg.noon)
            )
            for i in range(nt)
        ],
        dtype=np.float64,
    )
    sin_elev = np.sin(elev)
    denom = np.where(sin_elev > 0.0, I0 * sin_elev, 1.0)[:, None]
    Kt = np.where(sin_elev[:, None] > 0.0, sw / denom, 0.0)
    dfrac = 1.0 - 0.09 * Kt
    mask = Kt > 0.22
    kt = Kt[mask]
    kt2 = kt * kt
    dfrac[mask] = (
        0.95 - 0.16 * kt + 4.39 * kt2
        - 16.64 * kt2 * kt + 12.34 * (kt2 * kt2)
    )
    dfrac[Kt > 0.8] = 0.165
    return dfrac * sw, (1.0 - dfrac) * sw, elev


def _npoints_hint(cfg):
    """Number of points implied by the per-point vegetation arrays."""
    lengths = []
    for name in ("alb0", "vegh", "VAI"):
        arr = np.asarray(getattr(cfg, name), dtype=np.float64).ravel()
        if arr.size > 1:
            lengths.append(arr.size)
    if not lengths:
        return 1
    n = lengths[0]
    if any(length != n for length in lengths):
        raise ValueError(
            "Configuracion.alb0, vegh y VAI deben tener igual longitud"
        )
    return n


def _prepare_forcing(forz, cfg, params, npuntos_hint=1):
    required = ["LW", "Sf", "Rf", "Ta", "RH", "Ua", "Ps"]
    has_sw = getattr(forz, "SW", None) is not None
    has_parts = (
        getattr(forz, "Sdif", None) is not None
        and getattr(forz, "Sdir", None) is not None
    )
    if not has_sw and not has_parts:
        raise ValueError(
            "Forzamiento necesita SW o, alternativamente, Sdif y Sdir"
        )
    names = list(required)
    if has_sw:
        names.append("SW")
    raw = {
        name: np.asarray(getattr(forz, name), dtype=np.float64)
        for name in names
    }

    shapes = [arr.shape for arr in raw.values() if arr.ndim == 2]
    if shapes:
        times = {shape[0] for shape in shapes}
        if len(times) != 1:
            raise ValueError("Las matrices de forzamiento no tienen igual ntiempo")
        nt = times.pop()
        points = {shape[1] for shape in shapes}
        if len(points) == 1:
            npuntos = max(points.pop(), npuntos_hint)
        elif 1 in points:
            npuntos = max(max(points), npuntos_hint)
        else:
            if max(points) != npuntos_hint:
                raise ValueError(
                    "Las matrices de forzamiento no tienen igual npuntos"
                )
            npuntos = max(points)
    else:
        lengths = {
            arr.size for arr in raw.values()
            if arr.ndim == 1 and arr.size > 1
        }
        if len(lengths) > 1:
            raise ValueError(
                "Los vectores de forzamiento no tienen igual longitud"
            )
        nt = lengths.pop() if lengths else 1
        npuntos = npuntos_hint

    values = {
        name: _as_2d(value, name, nt, npuntos)
        for name, value in raw.items()
    }
    LW = values["LW"]
    Sf = values["Sf"]
    Rf = values["Rf"]
    Ta = values["Ta"]
    RH = values["RH"]
    Ua = values["Ua"]
    Ps = values["Ps"]

    Tc = Ta - TM
    es = E0 * np.exp(17.5043 * Tc / (241.3 + Tc))
    Qa = (RH / 100.0) * EPS * es / Ps
    Ua = np.maximum(Ua, 0.1)

    if int(cfg.ensmbl) == 1:
        Ta = Ta + float(cfg.Tadd)
        Pr = float(cfg.Pmlt) * (Rf + Sf)
        Tc = Ta - TM
        es = E0 * np.exp(17.5043 * Tc / (241.3 + Tc))
        Qs = EPS * es / Ps
        Qa = np.minimum(Qa, Qs)
        fs = np.clip(1.0 - 0.5 * Tc, 0.0, 1.0)
        Rf = (1.0 - fs) * Pr
        Sf = fs * Pr

    year, month, day, hour, tiempo = _prepare_time(forz, cfg, nt)
    if has_parts:
        Sdif = _as_2d(forz.Sdif, "Sdif", nt, npuntos)
        Sdir = _as_2d(forz.Sdir, "Sdir", nt, npuntos)
    else:
        Sdif = None
        Sdir = None
    Sdif, Sdir, elev = _partition_sw(
        cfg, nt, npuntos, values.get("SW"), Sdif, Sdir,
        year, month, day, hour
    )
    return (LW, Ps, Qa, Rf, Sdif, Sdir, Sf, Ta, Ua, year, month, day,
            hour, tiempo, elev, nt, npuntos)


def _coerce_state(value, shape, name, default, integer=False):
    if value is None:
        return default
    if integer:
        raw = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.round(raw)):
            raise ValueError(f"Condiciones.{name} debe contener enteros")
        arr = raw.astype(np.int64)
    else:
        arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=arr.dtype)
    if arr.shape == shape:
        # Copia defensiva: el núcleo actualiza el estado en sitio y no debe
        # modificar los arrays que haya pasado el usuario.
        return np.array(arr, dtype=arr.dtype, copy=True, order="C")
    if arr.size == 1:
        return np.full(shape, arr.ravel()[0], dtype=arr.dtype)
    if len(shape) > 1 and arr.shape == shape[1:]:
        return np.ascontiguousarray(np.broadcast_to(arr, shape))
    if len(shape) > 1 and arr.shape == (shape[0],):
        target = (shape[0],) + (1,) * (len(shape) - 1)
        return np.ascontiguousarray(
            np.broadcast_to(arr.reshape(target), shape)
        )
    raise ValueError(
        f"Condiciones.{name} tiene forma {arr.shape}; se esperaba {shape}"
    )


def _prepare_conditions(cond, cfg, params, npuntos, VAI):
    if cond is None:
        cond = Condiciones()
    if not isinstance(cond, Condiciones):
        raise TypeError(
            "condiciones_iniciales debe ser una instancia de Condiciones"
        )

    ncnpy = params.ncnpy
    nsmax = params.nsmax
    nsoil = params.nsoil

    Tprf = _broadcast_vector(cfg.Tprf, nsoil, "Tprf")
    fsat = _broadcast_vector(cfg.fsat, nsoil, "fsat")
    Tsoil_default = np.tile(Tprf, (npuntos, 1))
    Vsmc_default = np.tile(fsat * params.Vsat, (npuntos, 1))

    Sveg_default = np.zeros((npuntos, ncnpy))
    Tveg_default = np.full((npuntos, ncnpy), 285.0)
    if npuntos > 0:
        no_veg = VAI == 0.0
        Sveg_default[no_veg, :] = -999.0 / ncnpy
        Tveg_default[no_veg, :] = -999.0

    albs = _coerce_state(
        cond.albs, (npuntos,), "albs",
        np.full(npuntos, 0.8)
    )
    Tsrf = _coerce_state(
        cond.Tsrf, (npuntos,), "Tsrf", Tsoil_default[:, 0].copy()
    )
    Dsnw = _coerce_state(
        cond.Dsnw, (npuntos, nsmax), "Dsnw",
        np.zeros((npuntos, nsmax))
    )
    Nsnow = _coerce_state(
        cond.Nsnow, (npuntos,), "Nsnow",
        np.zeros(npuntos, dtype=np.int64), integer=True
    )
    Rgrn = _coerce_state(
        cond.Rgrn, (npuntos, nsmax), "Rgrn",
        np.full((npuntos, nsmax), params.rgr0)
    )
    Sice = _coerce_state(
        cond.Sice, (npuntos, nsmax), "Sice",
        np.zeros((npuntos, nsmax))
    )
    Sliq = _coerce_state(
        cond.Sliq, (npuntos, nsmax), "Sliq",
        np.zeros((npuntos, nsmax))
    )
    Tsnow = _coerce_state(
        cond.Tsnow, (npuntos, nsmax), "Tsnow",
        np.full((npuntos, nsmax), 273.0)
    )
    Qcan = _coerce_state(
        cond.Qcan, (npuntos, ncnpy), "Qcan",
        np.zeros((npuntos, ncnpy))
    )
    Sveg = _coerce_state(
        cond.Sveg, (npuntos, ncnpy), "Sveg", Sveg_default
    )
    Tcan = _coerce_state(
        cond.Tcan, (npuntos, ncnpy), "Tcan",
        np.full((npuntos, ncnpy), 285.0)
    )
    Tveg = _coerce_state(
        cond.Tveg, (npuntos, ncnpy), "Tveg", Tveg_default
    )
    Tsoil = _coerce_state(
        cond.Tsoil, (npuntos, nsoil), "Tsoil", Tsoil_default
    )
    Vsmc = _coerce_state(
        cond.Vsmc, (npuntos, nsoil), "Vsmc", Vsmc_default
    )

    if np.any(Nsnow < 0) or np.any(Nsnow > nsmax):
        raise ValueError(
            f"Condiciones.Nsnow debe estar entre 0 y {nsmax}"
        )

    return (albs, Tsrf, Dsnw, Nsnow, Qcan, Rgrn, Sice, Sliq, Sveg,
            Tcan, Tsnow, Tsoil, Tveg, Vsmc)


def compilar_numba(configuracion=None):
    """Compile the Numba core without running a complete simulation.

    A single dummy timestep is evaluated with representative arrays.  This
    triggers compilation of :func:`fsm2_timestep` and all the physics
    routines it calls, for every compile-time option.  No state is exposed
    and no output is produced.

    Parameters
    ----------
    configuracion : Configuracion, optional
        Configuration used to determine the grid dimensions and soil
        defaults.  ``None`` uses :class:`Configuracion`.

    Returns
    -------
    float
        Elapsed time in seconds.  If the core was already compiled in this
        process, the returned value is close to zero.
    """
    if configuracion is None:
        configuracion = Configuracion()
    if not isinstance(configuracion, Configuracion):
        raise TypeError("Configuracion debe ser una instancia de Configuracion")

    params = _build_params(configuracion)
    ncnpy = params.ncnpy
    nsmax = params.nsmax
    nsoil = params.nsoil
    Tprf = _broadcast_vector(configuracion.Tprf, nsoil, "Tprf")
    fsat = _broadcast_vector(configuracion.fsat, nsoil, "fsat")

    # Defaults equivalent to Condiciones(); the arrays are local to this
    # call, so the dummy timestep cannot modify any user state.
    Dsnw = np.zeros(nsmax, dtype=np.float64)
    Qcan = np.zeros(ncnpy, dtype=np.float64)
    Rgrn = np.full(nsmax, params.rgr0, dtype=np.float64)
    Sice = np.zeros(nsmax, dtype=np.float64)
    Sliq = np.zeros(nsmax, dtype=np.float64)
    Sveg = np.zeros(ncnpy, dtype=np.float64)
    Tcan = np.full(ncnpy, 285.0, dtype=np.float64)
    Tsnow = np.full(nsmax, 273.0, dtype=np.float64)
    Tsoil = np.array(Tprf, dtype=np.float64)
    Tveg = np.full(ncnpy, 285.0, dtype=np.float64)
    Vsmc = np.array(fsat * params.Vsat, dtype=np.float64)

    inicio = time.perf_counter()
    fsm2_timestep(
        params,
        float(configuracion.dt),
        0.0,
        float(configuracion.zT),
        float(configuracion.zU),
        250.0,       # LW
        101325.0,    # Ps
        0.005,       # Qa
        0.0,         # Rf
        0.0,         # Sdif
        0.0,         # Sdir
        0.0,         # Sf
        270.0,       # Ta
        0.0,         # trans
        1.0,         # Ua
        0.2,         # alb0
        0.0,         # vegh
        0.0,         # VAI
        0.8,         # albs
        float(Tprf[0]),  # Tsrf
        Dsnw,
        0,           # Nsnow
        Qcan,
        Rgrn,
        Sice,
        Sliq,
        Sveg,
        Tcan,
        Tsnow,
        Tsoil,
        Tveg,
        Vsmc,
    )

    # Compile the fused time loop as well.  It receives the same state
    # arrays reshaped to (npuntos, ...) and one dummy forcing point.
    one = np.ones((1, 1), dtype=np.float64)
    out2d = np.empty((1, 1), dtype=np.float64)
    out_nsnow = np.empty((1, 1), dtype=np.int64)
    dummy3 = np.empty((1, 1, 1), dtype=np.float64)
    perfiles = [dummy3 for _ in range(9)]
    fsm2_run(
        params,
        float(configuracion.dt),
        float(configuracion.zT),
        float(configuracion.zU),
        float(configuracion.trans),
        np.zeros(1, dtype=np.float64),  # elev
        one, one, one, one, one, one, one, one, one,  # forcing
        np.full(1, 0.2), np.zeros(1), np.zeros(1),     # alb0, vegh, VAI
        np.full(1, 0.8), np.full(1, Tprf[0]),
        Dsnw.reshape(1, -1), np.zeros(1, dtype=np.int64),
        Qcan.reshape(1, -1), Rgrn.reshape(1, -1),
        Sice.reshape(1, -1), Sliq.reshape(1, -1),
        Sveg.reshape(1, -1), Tcan.reshape(1, -1),
        Tsnow.reshape(1, -1), Tsoil.reshape(1, -1),
        Tveg.reshape(1, -1), Vsmc.reshape(1, -1),
        out_nsnow, out2d,
        np.empty((1, 1, nsoil)), np.empty((1, 1, ncnpy)),
        np.empty((1, 1, nsmax)),
        out2d, out2d, out2d, out2d, out2d, out2d, out2d, out2d,
        out2d, out2d, out2d, out2d, out2d, out2d, out2d, out2d,
        False,
        *perfiles,
    )
    return time.perf_counter() - inicio


def fsm2(forzamiento, condiciones_iniciales=None, configuracion=None):
    """Run FSM2 entirely in memory.

    Parameters
    ----------
    forzamiento : Forzamiento
        Meteorological forcing arrays, time first.
    condiciones_iniciales : Condiciones, optional
        Initial state.  ``None`` or ``Condiciones()`` uses the FSM2 defaults.
    configuracion : Configuracion, optional
        Site, grid, physics options and parameters.

    Returns
    -------
    simulacion : Simulacion
        Output time series.
    condiciones_finales : CondicionesFinales
        State at the end of the run, suitable for restarting.
    """
    if configuracion is None:
        configuracion = Configuracion()
    if not isinstance(configuracion, Configuracion):
        raise TypeError("Configuracion debe ser una instancia de Configuracion")
    if not isinstance(forzamiento, Forzamiento):
        raise TypeError("forzamiento debe ser una instancia de Forzamiento")

    cfg = configuracion
    params = _build_params(cfg)

    npuntos_hint = _npoints_hint(cfg)
    (LW, Ps, Qa, Rf, Sdif, Sdir, Sf, Ta, Ua, year, month, day, hour,
     tiempo, elev, nt, npuntos) = _prepare_forcing(
        forzamiento, cfg, params, npuntos_hint
    )

    alb0 = _broadcast_vector(cfg.alb0, npuntos, "alb0")
    vegh = _broadcast_vector(cfg.vegh, npuntos, "vegh")
    VAI = _broadcast_vector(cfg.VAI, npuntos, "VAI")

    (albs, Tsrf, Dsnw, Nsnow, Qcan, Rgrn, Sice, Sliq, Sveg, Tcan,
     Tsnow, Tsoil, Tveg, Vsmc) = _prepare_conditions(
        condiciones_iniciales, cfg, params, npuntos, VAI
    )

    diag_names = (
        "fsnow", "albs", "H", "LE", "LWout", "LWsub", "Melt", "Roff",
        "snd", "snw", "subl", "svg", "SWout", "SWsub", "Tsub", "Usub",
    )
    diag = {
        name: np.empty((nt, npuntos), dtype=np.float64)
        for name in diag_names
    }
    Nsnow_out = np.empty((nt, npuntos), dtype=np.int64)
    Tsrf_out = np.empty((nt, npuntos), dtype=np.float64)
    Tsoil_out = np.empty((nt, npuntos, params.nsoil), dtype=np.float64)
    Tveg_out = np.empty((nt, npuntos, params.ncnpy), dtype=np.float64)
    Wflx_out = np.empty((nt, npuntos, params.nsmax), dtype=np.float64)

    guardar_estado = bool(cfg.guardar_estado)
    perfiles = {}
    if guardar_estado:
        for nombre, capas in (
            ("Dsnw", params.nsmax),
            ("Rgrn", params.nsmax),
            ("Sice", params.nsmax),
            ("Sliq", params.nsmax),
            ("Tsnow", params.nsmax),
            ("Qcan", params.ncnpy),
            ("Sveg", params.ncnpy),
            ("Tcan", params.ncnpy),
            ("Vsmc", params.nsoil),
        ):
            perfiles[nombre] = np.empty(
                (nt, npuntos, capas), dtype=np.float64
            )

    if guardar_estado:
        perf_Dsnw = perfiles["Dsnw"]
        perf_Rgrn = perfiles["Rgrn"]
        perf_Sice = perfiles["Sice"]
        perf_Sliq = perfiles["Sliq"]
        perf_Tsnow = perfiles["Tsnow"]
        perf_Qcan = perfiles["Qcan"]
        perf_Sveg = perfiles["Sveg"]
        perf_Tcan = perfiles["Tcan"]
        perf_Vsmc = perfiles["Vsmc"]
    else:
        dummy = np.empty((1, 1, 1), dtype=np.float64)
        perf_Dsnw = dummy
        perf_Rgrn = dummy
        perf_Sice = dummy
        perf_Sliq = dummy
        perf_Tsnow = dummy
        perf_Qcan = dummy
        perf_Sveg = dummy
        perf_Tcan = dummy
        perf_Vsmc = dummy

    fsm2_run(
        params, float(cfg.dt), float(cfg.zT), float(cfg.zU),
        float(cfg.trans), elev, LW, Ps, Qa, Rf, Sdif, Sdir, Sf, Ta, Ua,
        alb0, vegh, VAI,
        albs, Tsrf, Dsnw, Nsnow, Qcan, Rgrn, Sice, Sliq, Sveg,
        Tcan, Tsnow, Tsoil, Tveg, Vsmc,
        Nsnow_out, Tsrf_out, Tsoil_out, Tveg_out, Wflx_out,
        diag["fsnow"], diag["albs"], diag["H"], diag["LE"],
        diag["LWout"], diag["LWsub"], diag["Melt"], diag["Roff"],
        diag["snd"], diag["snw"], diag["subl"], diag["svg"],
        diag["SWout"], diag["SWsub"], diag["Tsub"], diag["Usub"],
        guardar_estado,
        perf_Dsnw, perf_Rgrn, perf_Sice, perf_Sliq, perf_Tsnow,
        perf_Qcan, perf_Sveg, perf_Tcan, perf_Vsmc,
    )

    final = CondicionesFinales(
        albs=albs.copy(),
        Tsrf=Tsrf.copy(),
        Dsnw=Dsnw.copy(),
        Nsnow=Nsnow.copy(),
        Qcan=Qcan.copy(),
        Rgrn=Rgrn.copy(),
        Sice=Sice.copy(),
        Sliq=Sliq.copy(),
        Sveg=Sveg.copy(),
        Tcan=Tcan.copy(),
        Tsnow=Tsnow.copy(),
        Tsoil=Tsoil.copy(),
        Tveg=Tveg.copy(),
        Vsmc=Vsmc.copy(),
    )

    simulacion = Simulacion(
        year=year,
        month=month,
        day=day,
        hour=hour,
        tiempo=tiempo,
        fsnow=diag["fsnow"],
        albs=diag["albs"],
        Nsnow=Nsnow_out,
        H=diag["H"],
        LE=diag["LE"],
        LWout=diag["LWout"],
        LWsub=diag["LWsub"],
        Melt=diag["Melt"],
        Roff=diag["Roff"],
        snd=diag["snd"],
        snw=diag["snw"],
        subl=diag["subl"],
        svg=diag["svg"],
        SWout=diag["SWout"],
        SWsub=diag["SWsub"],
        Tsub=diag["Tsub"],
        Usub=diag["Usub"],
        Tsoil=Tsoil_out,
        Tsrf=Tsrf_out,
        Tveg=Tveg_out,
        Wflx=Wflx_out,
    )
    if guardar_estado:
        simulacion.perfiles = perfiles
        for nombre, valores in perfiles.items():
            setattr(simulacion, nombre, valores)
    return simulacion, final




