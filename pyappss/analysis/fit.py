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

from pyappss.analysis.util import Util


class Fit(Util):
    def __init__(self, agc=None, boxcar=None, path="", dark_mode=False, vel=None, spec=None, rms=None):
        super().__init__(agc=agc, dark_mode=dark_mode)

        self.boxcar = boxcar
        self.path = path  # measure already calc correct path so use that, dont recalc path
        self.vel = vel
        self.spec = spec
        self.rms = rms
        self.stepcolor = 'crimson'

        self.w50 = 0
        self.w50err = 0
        self.w20 = 0
        self.w20err = 0
        self.vsys = 0
        self.vsyserr = 0
        self.flux = 0
        self.fluxerr = 0
        self.SN = 0

        # call measure calc?

    def get_header(self):
        """
        Returns the header of the FITS file.
        """
        hdul = fits.open(self.path)
        hdr = hdul[1].header
        return hdr

    def write_file(self, comments):
        """
        Writes the values of the quantities to the CSV file.
        :param comments: comments on the fit
        """

        file_exists = os.path.exists('ReducedData.csv')
        if file_exists == False:
            file = open('ReducedData.csv', 'x')
            message_info = (
                    'AGCnr,RA,DEC,Vsys(km/s),W50(km/s),W50err,W20(km/s),flux(Jy*km/s),fluxerr,SN,rms,smo,FitType,comments' + '\n')
            file.write(message_info)

        hdr = self.get_header()
        file = open('ReducedData.csv', 'a')
        # Modified to match the changes made to filename - pulls 4th entry and extends 6 further - should match the longest galaxy numbers.
        message = (str(self.filename[3:-5]) + ',' +
                   # Currently commented as GBT files lack attached galaxy names
                   # str(hdr[16]) + ',' +
                   str(hdr['RA']) + ',' + str(hdr['DEC']) + ',' +
                   # Similarly, there is not a comparison between optical and radio coordinates.
                   # + str(hdr[18]) + ',' + str(hdr[19]) + ','
                   str(self.vsys) + ',' +
                   str(self.w50) + ',' + str(self.w50err) + ',' +
                   str(self.w20) + ',' +
                   str(self.flux) + ',' + str(self.fluxerr) + ',' +
                   str(self.SN) + ',' + str(self.rms) + ',' + str(self.smo) + ',' +
                   str(self.currstep) + ',' + str(comments) + '\n'
                   )
        file.write(message)

    def get_comments(self):
        """
        helper method thats gets the user comments on the fit.
        """
        comment = input('\nEnter any comments: ')
        return comment

    def print_values(self):
        print('\n')
        print('W50 = ', self.w50, ' +/- ', self.w50err, ' km/s ')
        print('W20 = ', self.w20, ' +/- ', self.w20err, ' km/s ')
        print('vsys = ', self.vsys, ' +/- ', self.vsyserr, ' km/s')
        print('flux = ', self.flux, ' +/- ', self.fluxerr, ' Jy km/s')
        print('SN: ' + str(self.SN))

    def calculate_lambda(self, deltav, logSNR):
        """
        This function is just a straight-up adaptation of Springob 2005 Table 2 into Python code
        We need to check separately for deltav and logSNR
        """
        if not self.boxcar:
            if deltav < 5.:
                if logSNR < 0.6:
                    return 0.005
                if logSNR > 1.1:
                    return 0.395
                return 0
            if deltav > 11.:
                if logSNR < .6:
                    return .227
                if logSNR > 1.1:
                    return 0.533
                return -0.1523 + 0.623 * logSNR
            if logSNR < 0.6:
                return 0.037 * deltav - 0.18
            if logSNR > 1.1:
                return 0.023 * deltav + 0.28
            return (0.0527 * deltav - 0.732) + (-0.027 * deltav + 0.92) * logSNR
        if self.boxcar:
            if deltav < 5.:
                if logSNR < 0.6:
                    return 0.020
                if logSNR > 1.1:
                    return 0.430
                return -0.4705 + 0.820 * logSNR
            if deltav > 11.:
                if logSNR < .6:
                    return 0.332
                if logSNR > 1.1:
                    return 0.802
                return -0.2323 + 0.940 * logSNR
            if logSNR < 0.6:
                return 0.052 * deltav - 0.24
            if logSNR > 1.1:
                return 0.062 * deltav + 0.12
            return (0.0397 * deltav - 0.669) + (0.020 * deltav + 0.72) * logSNR
