"""
Functions and classes using the Python library Soprano for aid with
dipolar calculations. This is used mainly for powder averages and random
orientations
"""

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
