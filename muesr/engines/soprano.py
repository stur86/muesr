"""
Functions and classes using the Python library Soprano for aid with
dipolar calculations. This is used mainly for powder averages and random
orientations
"""

import numpy as np
import scipy.constants as cnst
from soprano.utils import minimum_supcell, supcell_gridgen
from soprano.properties.nmr.utils import _get_isotope_data


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


def _get_spins(sample, scell, momtype='e'):

    cell = sample.cell.get_cell()
    fxyz, xyz = supcell_gridgen(cell, scell)

    fpos = sample.cell.get_scaled_positions()
    pos = sample.cell.get_positions()

    sfpos = fpos[None, :, :]+fxyz[:, None, :]
    spos = pos[None, :, :]+xyz[:, None, :]

    spins = (spos*0.0j)

    k = sample.mm.k
    fc = sample.mm.fc

    ker = np.exp(-2.0j*np.pi*np.dot(k, sfpos.T))
    spins = np.real(fc[None, :, :]*ker[:, None, None])

    # Now turn those into actual dipole moments...
    if momtype == 'e':
        g_e = (cnst.physical_constants['Bohr magneton'][0] *
               cnst.physical_constants['electron g factor'][0])/cnst.hbar
        gammas = np.repeat([g_e], len(pos))
    elif momtype == 'n':
        gammas = _get_isotope_data(sample.cell.get_chemical_symbols(),
                                   'gamma')
    else:
        raise ValueError('Invalid momtype argument passed to _get_spins')
    gammas = np.repeat(gammas[None, :], len(fxyz), axis=0)

    return spos, spins, gammas


def _dipten_all(sample, radius, momtype='e'):

    scell = find_smallest_scell(sample, radius)
    spos, spins, gammas = _get_spins(sample, scell, momtype)

    # Build the full dipolar tensors
    dipten = []
    fields = []

    for mu in sample.muons:
        mupos = np.dot(sample.cell.get_cell(), mu)
        r = spos-mupos[None, None]
        d = np.linalg.norm(r, axis=-1)
        sph_i = np.where(d <= radius)

        r = r[sph_i]
        d = d[sph_i]
        s = spins[sph_i]
        g = gammas[sph_i]

        # Build the tensors
        dyad = (3*r[:, :, None]*r[:, None, :] /
                d[:, None, None]**2-np.eye(3)[None])
        dyad *= (-cnst.mu_0*cnst.hbar/(8*np.pi*d[:, None, None]**3)
                 * 1e30*g[:, None, None])

        dipten.append(dyad)
        fields.append(np.sum(dyad*s[:,:,None], axis=1))

    dipten = np.array(dipten)
    fields = np.array(fields)

    return dipten, fields
