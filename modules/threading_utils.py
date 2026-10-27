# -*- coding: utf-8 -*-
"""
Helpers to configure BLAS/OpenMP threads safely.

``threadpoolctl`` is used to avoid oversubscription when several MuSA worker
processes run at the same time. On some platforms (notably Windows with
certain BLAS distributions) threadpoolctl can fail while introspecting the
BLAS library. Thread limiting is only a performance hint, so MuSA must never
crash because of it.
"""

from contextlib import nullcontext
import warnings

from threadpoolctl import threadpool_limits


def threadpool_limits_safe(limits, user_api=None):
    """Return a ``threadpool_limits`` context manager or a no-op fallback.

    Parameters
    ----------
    limits : int or None
        Maximum number of threads. ``None`` means no limit.
    user_api : str or None
        Restrict the limit to a given user API (e.g. ``"blas"``).

    Returns
    -------
    context manager
        ``threadpool_limits`` when it works, ``nullcontext`` otherwise.
    """
    try:
        return threadpool_limits(limits=limits, user_api=user_api)
    except (OSError, ValueError) as exc:
        warnings.warn(
            "Could not configure the number of BLAS/OpenMP threads ("
            + str(exc)
            + "). Continuing without thread limiting."
        )
        return nullcontext()
