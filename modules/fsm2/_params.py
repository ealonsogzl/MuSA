"""Compile-time options and physical parameters passed to the Numba core."""

from collections import namedtuple

# All fields are scalars or one-dimensional arrays.  A namedtuple is used
# instead of a dataclass because Numba supports it directly as a function
# argument in nopython mode.
PARAM_FIELDS = (
    # Vegetation parameters
    "acn0", "acns", "avg0", "avgs", "cvai", "eunl", "gsnf", "hbas",
    "kext", "leaf", "munl", "svai", "Tunl", "Uunl", "wcan",
    # Snow parameters
    "asmn", "asmx", "eta0", "hfsn", "kfix", "rcld", "rfix", "rgr0",
    "rhof", "rhow", "rmlt", "Salb", "snda", "Talb", "tcld", "tmlt",
    "trho", "Wirr", "z0sn",
    # Ground and soil parameters
    "fcly", "fsnd", "gsat", "z0sf", "b", "hcap_soil", "hcon_soil",
    "sathh", "Vcrit", "Vsat",
    # Ensemble parameters
    "Pmlt", "Tadd",
    # Grid dimensions
    "ncnpy", "nsmax", "nsoil", "nhyd",
    # Compile-time options
    "albedo", "canint", "canmod", "canrad", "canunl", "condct",
    "densty", "exchng", "hydrol", "sgrain", "snfrac", "swpart",
    "zoffst", "ensmbl",
    # Geometry
    "fvg1", "zsub", "dzsnow", "dzsoil",
)

Parametros = namedtuple("Parametros", PARAM_FIELDS)
