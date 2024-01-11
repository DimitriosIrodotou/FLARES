import time
import numpy as np
import healpy as hlp
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.style as style

from astropy.constants import G
from astropy_healpix import HEALPix
from morpho_kinematics import MorphoKinematic


def test_where_iterative(**conditions):
    """
    Test if specific conditions are met.
    :param conditions: dictionary of keyword arguments.
    :return: None
    """

    # Iterating over the keys and values of the 'conditions' dictionary #
    for key, value in zip(conditions, conditions.values()):
        if np.where(key == value) is True:
            print(key, value)

    return None
