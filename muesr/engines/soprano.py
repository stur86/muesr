"""
Functions and classes using the Python library Soprano for aid with
dipolar calculations. This is used mainly for powder averages and random
orientations
"""

import numpy as np
from soprano.utils import minimum_supcell, supcell_gridgen


def find_smallest_scell(sample, radius):
    """
    Simple function to evaluate the minimum supercell for sample
    that contains a sphere of given radius.

    :param sample: the sample object
    :param radius: the radius of the biggest sphere that should be inscribed.
    :return: the size of the supercell along the lattice coordinates.
    :rtype: list
    :raises: ValueError, TypeError
    """

    cell = sample.cell.get_cell()
    return list(minimum_supcell(radius, cell))

def _get_spins(sample, scell):

    cell = sample.cell.get_cell()
    fxyz, xyz = supcell_gridgen(cell, scell)

    fpos = sample.cell.get_scaled_positions()
    pos = sample.cell.get_positions()

    sfpos = fpos[None,:,:]+fxyz[:,None,:]
    spos = pos[None,:,:]+xyz[:,None,:]

    spins = (spos*0.0j)

    k = sample.mm.k
    fc = sample.mm.fc

    ker = np.exp(2.0j*np.pi*np.dot(k, fxyz.T))
    print(fc[None,:,:]*ker[:,None,:])