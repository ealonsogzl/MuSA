#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the Landmark MDS implementation
Created on Wed Mar 15 14:08:54 2023
@inproceedings{Silva2004SparseMS,
@author: alonsoe
  title={Sparse multidimensional scaling using land-mark points},
  author={Vin de Silva and Joshua B. Tenenbaum},
  year={2004}
}

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import numpy as np
import scipy as sp


def landmark_MDS(D, lands, dim):
    """
    Sparse Multidimensional Scaling using Landmark Points
    Modified from original implementation in 

    https://github.com/danilomotta/LMDS
    """

    Dl = D[:, lands]
    n = len(Dl)

    # Centering matrix
    H = - np.ones((n, n))/n
    np.fill_diagonal(H, 1-1/n)
    # YY^T
    H = -H.dot(Dl**2).dot(H)/2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    if dim:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            raise Exception('Not enough positive eigenvalues'
                            'for the selected dim.')
        if w.size == 0:
            raise Exception('matrix is negative definite.')
            return []

        V = evecs[:, w]
        # L = V.dot(np.diag(np.sqrt(evals[w]))).T
        N = D.shape[1]
        Lh = V.dot(np.diag(1./np.sqrt(evals[w]))).T
        Dm = D - np.tile(np.mean(Dl, axis=1), (N, 1)).T
        dim = w.size
        X = -Lh.dot(Dm)/2.
        X -= np.tile(np.mean(X, axis=1), (N, 1)).T

        _, evecs = sp.linalg.eigh(X.dot(X.T))

        return (evecs[:, ::-1].T.dot(X)).T
