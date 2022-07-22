from astropy.io import fits
import argparse
import numpy as np
import matplotlib
import os
import pathlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.modeling import models
from astropy.modeling.models import custom_model
from astropy.modeling import fitting

#from analysis import smooth
from pyappss.analysis import multigauss
from pyappss.analysis.util import Util
from pyappss.analysis.gauss import Gauss
from pyappss.analysis.trap import Trap
from pyappss.analysis.twopeak import Twopeak
# import pyappss.analysis.fits -- let this be a folder containing the 4 fit types

class Measure(Util):
    """
    Measure qualities of an HI spectrum.
    Use an FITS file or data arrays to calculate spectral features:
     - Integrated flux density
     - Systemic velocity
     - Source width
     - Signal to noise

     Parameters
     ----------
     filename : int
     AGC number of the galaxy, e.g, 104365

     smo : int
     Value for smoothing the spectrum, if nothing is passed Hanning smoothing smoothing will occur.
     e.g smo = 7 will result in boxcar smooth of value 7

     gauss : bool
     When True gaussian fit will be applied to the spectrum. Default is False

     twopeak : bool
     When True twopeak fit will be applied to the spectrum. Default is False.

     trap : bool
     When True trapezoidal fit will be applied to the spectrum. Default is False.

     multigauss : bool
     When True multigauss fit will be applied to the spectrum. Default is False.

     dark_mode: bool
     Dark mode for the spectrum. When True the spectrum will plotted in dark mode.
     Default is Light Mode.
    """

    def __init__(self, smo=None, gauss=False, twopeak=False, trap=False, path="", dark_mode=False,
                 vel=None, spec=None, rms=None, agc=None, noconfirm=False, overlay=False):

        super().__init__(agc, smo, path, dark_mode)

            # currently the way reduce is written, filename is never set, only agc
        # if filename is not None:
        #     self.filename = 'AGC{}.fits'.format(filename)
        #     self.path = pathlib.PurePath(path + "/" + self.filename)
        #     self.load()
        #
        #     # smooth, if None, the smoothing will do nothing
        #     self.res = smooth.smooth(self.spec, smooth_type=smo)
        #     if smo is not None:
        #         self.boxcar = True
        # else:

        self.vel = vel
        self.res = spec  # smoothed version is passed from baseline
        self.spec = spec
        self.rms = rms

        self.base = True
        if smo is not None:
            self.boxcar = True
        else:
            self.boxcar = False
        self.n = 0
        self.smo = smo

        self.overlay = overlay # better way to do this?
            # like one function
        length = len(vel)
        if self.overlay:
            self.y = []
            self.x = []
            # Setting these conditions so it can run directly.
                # Maybe I should have defined these arrays in the gaussfilter function, however.
            for i in range(1000, length - 1001):
                self.x.append(self.vel[i])
                self.y.append(self.spec[i])
            self.y = np.nan_to_num(self.y)
            self.x = np.nan_to_num(self.x)
            self.convolved = multigauss.ManyGauss.gaussfilter(self)

        # for information bubble
        self.currstep = "Measuring"
        self.stepcolor = "crimson"

        self.plot()

        # again reduce is written so this never happens
        # if self.filename is not None:
        #     self.calcRMS()

        # Adds in a choice to change fit if baseline/viewable data leads reducer to want some different fit type.
        if not noconfirm:

            # if none of the flags are set, skip question
            if not twopeak and not trap and not gauss:
                response = 'no'
            else:
                print('Do you want to keep your previously selected fit type?\n'
                      'Type "yes" and press Enter to keep, type anything else and press Enter to pick a new fit type')
                response = input()

            # allow the user to select new fits types
            if response != 'yes':
                print(
                    'The accepted fit methods are: "gauss" for a gaussian, "twopeak" for a double-horned profile fit, or "trap" for a trapezoidal fit.\n'
                    'Once done, hit Enter, with no text input, to move on. Multiple fit options can still be selected in this step!')
                chosen = False
                twopeak = False
                trap = False
                gauss = False
                while not chosen:
                    response = input()
                    if response == 'gauss':
                        gauss = True
                    elif response == 'twopeak':
                        twopeak = True
                    elif response == 'trap':
                        trap = True
                    elif response == '':
                        chosen = True
                    else:
                        print('Please enter a valid fit option!\nAccepted values are "gauss", "twopeak", and "trap"')

        plt.close()

        if gauss:
            Gauss(self.agc, path=self.path, dark_mode=dark_mode, vel=self.vel, spec=self.spec, rms=self.rms)

        if trap:
            Trap(self.agc, self.boxcar, self.path, dark_mode, self.vel, self.spec, self.rms)

        if twopeak:
            Twopeak(self.agc, self.boxcar, self.path, dark_mode, self.vel, self.spec, self.rms)


        plt.close()

    def calcRMS(self):
        """
        Helper method to select regions for RMS claculation.
        To be depreciated since masking takes care of RMS calculation.
        :return:
        """
        print("\n Select a region without rfi or the source for RMS calculation")
        v, s = self.markregions()
        self.rms = np.std(s)
        print('\n RMS of Spectrum: ', self.rms)

