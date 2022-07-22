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

from pyappss.analysis.fit import Fit

class Twopeak(Fit):

    def __init__(self, agc=None, boxcar=None, path="", dark_mode=False, vel=None, spec=None, rms=None):

        super().__init__(agc, boxcar, path, dark_mode, vel, spec, rms)
        self.currstep = 'twopeak'

        self.twopeak()
        self.write_file(self.get_comments())
        input('Two-peaked fit complete! Press Enter to end.\n')
        plt.close()

    def twopeak(self, first_region=True):
        """
        Method to fit a two peak profile
        :return:
        """
        # if self.base == False:
        #     response = input(
        #         'You have not baselined this spectrum yet. If you would like to proceed with baselining, press \'Enter\' to continue. Otherwise, press any other key for No.')
        #     if response == '':
        #         self.baseline()
        #     else:
        #         sys.exit('Program exited. Not baselined.')
        self.plot()

        print("Please select the region for a two-peaked fit.")
        v, s = self.mark_regions(first_region)
        self.plot(min(v), max(v), min(s), max(s))

        left = True  # left fitting starts as true, so user fits the left side of the emission first.
        right = False  # user will fit right side of emission second.
        leftcoef = []
        rightcoef = []
        leftvel, leftvel20, leftsigma, leftcov, leftmaxval, lefterror = 0, 0, 0, 0, 0, 0
        rightvel, rightvel20, rightsigma, rightcov, rightmaxval, righterror = 0, 0, 0, 0, 0, 0

        # left fitting
        while left:
            print('\nSelect a region around one of the outer slopes of the profile.')
            first_region = False
            leftv, lefts = self.mark_regions(first_region)
            fitedge = [min(leftv), max(leftv)]

            # self.plot(None, None, min(self.res), max(self.res))
            leftcoef, leftvel, leftvel20, leftsigma, leftcov, leftmaxval, lefterror = self.edgefit(leftv, lefts,
                                                                                                   left, right)
            # plotting the fit and checking
            self.plot(min(v), max(v), min(s), max(s))
            self.ax.plot(v, leftcoef[1] + leftcoef[0] * v)

            response = input('Is this fit OK? Press Enter to accept or any other key for No.')
            if response == '':
                break  # move on to right fit. if not, keep looping through the left until an appropriate fit is found.
        # right fitting
        right = True  # user fits right side of emission
        while right:
            first_region = False
            print('\nSelect the region around the second slope of the profile.')
            rightv, rights = self.mark_regions(first_region)
            fitedge = [min(rightv), max(rightv)]
            rightcoef, rightvel, rightvel20, rightsigma, rightcov, rightmaxval, righterror = self.edgefit(rightv, rights,
                                                                                                          left, right)
            # plotting the fit and checking
            self.plot(min(v), max(v), min(s), max(s))
            self.ax.plot(v, leftcoef[1] + leftcoef[0] * v)  # orange represents the left side
            self.ax.plot(v, rightcoef[1] + rightcoef[0] * v)  # red represents the right side

            response = input('Is this fit OK? Press Enter to accept or any other key for No.')
            if response == '':
                break  # moves on to calculations. if not, keep looping through right fit until an appropriate fit is found.

        # swap values if the user selected left/right backwards
            # check the slopes from the coefs
            # if left is negative, then it is the right slope
        if leftcoef[0] < 0:
            rightcoef, leftcoef = leftcoef, rightcoef
            rightvel, leftvel = leftvel, rightvel
            rightvel20, leftvel20 = leftvel20, rightvel20
            rightsigma, leftsigma = leftsigma, rightsigma
            rightcov, leftcov = leftcov, rightcov
            rightmaxval, leftmaxval = leftmaxval, rightmaxval
            righterror, lefterror = lefterror, righterror

        deltav, fluxerr, sn, totflux, vsys, vsyserr, w20, w20err, w50, w50err = self.__twopeak_calc(leftcoef, lefterror,
                                                                                                    leftvel, leftvel20,
                                                                                                    rightcoef,
                                                                                                    righterror,
                                                                                                    rightvel,
                                                                                                    rightvel20, s, v)
        # use measure's plot and add in the extra lines
        self.plot(min(v), max(v), min(s), max(s))
        self.ax.plot(v, leftcoef[1] + leftcoef[0] * v)
        self.ax.plot(v, rightcoef[1] + rightcoef[0] * v)
        self.ax.plot([vsys, vsys], [-100, 1e4], linestyle='--', color='red', linewidth=0.5)
        self.ax.plot([leftvel, rightvel], [0.25 * (leftmaxval + rightmaxval), 0.25 * (leftmaxval + rightmaxval)],
                linestyle='--', color='red', linewidth=0.5)

        self.w50 = w50
        self.w50err = w50err
        self.w20 = w20
        self.w20err = w20err
        self.vsys = vsys
        self.vsyserr = vsyserr
        self.flux = totflux
        self.fluxerr = fluxerr
        self.SN = sn

        self.print_values()

    def __twopeak_calc(self, leftcoef, lefterror, leftvel, leftvel20, rightcoef, righterror, rightvel, rightvel20, s, v):
        """
        Helper method that calculates all the qualities of the twopeak fit and returns them
        """
        # calculating w50 and vsys
        between = []
        for i in range(len(v)):
            if -leftcoef[1] / leftcoef[0] < v[i] < -rightcoef[1] / rightcoef[0]:
                between.append(i)
        leftedge = min(between)
        rightedge = max(between)
        fluxpeak = max(s[leftedge:rightedge])  # highest point of the source
        SNR = (fluxpeak - self.rms)  # signal-noise ratio
        logSNR = np.log10(SNR)
        centerchan = int((leftedge + rightedge) / 2.)
        deltav = abs(v[centerchan + 1] - v[centerchan - 1]) / 2.
        Lambda = self.calculate_lambda(deltav, logSNR)
        w50 = abs(
            rightvel - leftvel) - 2 * deltav  ##width at half the height of the source. subtracting off noise and instrumental broadening effect
        w50err = np.sqrt(lefterror ** 2 + righterror ** 2)
        w20 = abs(rightvel20 - leftvel20)  # width at 1/5 the height of the source
        w20err = w50err
        w50 = w50 - 2 * deltav * Lambda  # Subtract off noise+instrumental broadening effect
        w20 = w20 - 2 * deltav * Lambda
        vsys = .5 * (rightvel + leftvel)  # where rightvel and leftvel are half vels in the right and left fit
        vsyserr = w50err / np.sqrt(2)
        # integrate the flux
        totflux = 0
        for i in range(leftedge, rightedge):  # finding the area of the total region
            totflux += deltav * s[i]
        totflux = totflux / 1000  # from mJy to Jy
        fluxerr = 2 * (self.rms / 1000) * np.sqrt(1.4 * w50 * deltav)
        # Calculate signal to noise (the ALFALFA way)
        sn = 1000 * totflux / w50 * np.sqrt((w50.clip(min=None, max=400.) / 20.)) / self.rms
        return deltav, fluxerr, sn, totflux, vsys, vsyserr, w20, w20err, w50, w50err

    def edgefit(self, v, s, left=None, right=None):

        # passing left and right booleans to indicate which side we are fitting.
        # edgefit works for both left and right fit.

        minspec = min(s)  # finding min y-val in spectrum (selected region)
        # finding the last place that the baselined spectrum crosses the x-axis
        maxval = max(s)  # finding max y-val in selected region
        # finding location of maxval, relative to the selected region
        maxchan = np.argmax(s)  # where does the max y-val occur?
        if left and not right:  # leftfit
            zerochan = []
            for i in range(len(v) - 1):
                if (s[i] <= 0 and s[i + 1] >= 0) or (
                        s[i] >= 0 and s[i + 1] <= 0):  # keeps track of crossings of the x-axis
                    zerochan.append(i)
            if (len(zerochan)) == 0:
                zerochan.append(np.argmin(s))
            edge = [maxchan, zerochan[0]]  # since leftfit, we want the last zero crossing
        else:  # rightfit
            zerochan = []
            for i in range(len(v) - 1):
                if (s[i] <= 0 and s[i + 1] >= 0) or (s[i] >= 0 and s[i + 1] <= 0):
                    zerochan.append(i)
            if (len(zerochan)) == 0:
                zerochan.append(np.argmin(s))
            edge = [maxchan, zerochan[-1]]  # since rightfit, we want the first zero crossing
        if v[edge[0]] > v[edge[1]]:
            edge.reverse()  # keeps numbers in order
        maxval = maxval - self.rms
        percent15 = .15 * maxval
        percent85 = .85 * maxval
        # restrict further to the region between 15% and 85% of maximum
        region = []
        for i in range(len(self.vel)):  # iterating over the whole spectrum so we have consistency in the indices
            if self.res[i] >= percent15 and self.res[i] <= percent85 and self.vel[i] >= v[edge[0]] and self.vel[i] <= v[
                edge[1]]:
                region.append(
                    i)  # appending the indices where the y values are between 15 and 85%, and where the x values are within the selected region.
        p15chan = min(region)
        p85chan = max(region)
        midchan = [p15chan, p85chan]
        midchan.sort()
        # Perform a a linear fit over the 15%-85% profile edge
        xvals = []
        yvals = []
        xvals = self.vel[midchan[0]:midchan[1]]  # vel from 15% to 85%
        yvals = self.res[midchan[0]:midchan[1]]  # res from 15% to 85%
        errors = np.zeros(len(xvals)) + self.rms
        # Not enough points in the region to fit
        # if len(xvals) <= 3 or abs(edge[1] - edge[0]) < 3:
        #     print(
        #         'There are not enough points in the selected edge region to produce a good fit! Try performing a boxcar smooth on the spectrum using smooth(int) first, or selecting a new region.')
        #     sys.exit('Not enough points to fit. Please try again.')
        coef, cov = np.polyfit(xvals, yvals, 1, cov=True)
        sigma = np.sqrt(np.diag(cov))
        intercept = coef[1]  # Unpack the coefficient array
        slope = coef[0]
        inter_err = sigma[1]  # Unpack the uncertainty array
        slope_err = sigma[0]
        yfit = []

        # Use the fit information to find the velocity at half the peak intensity
        halfpeak = 0.5 * maxval
        velocity = (halfpeak - intercept) / slope

        peak20 = 0.2 * maxval
        vel20 = (peak20 - intercept) / slope

        variance = 1 / slope ** 2 * (
                self.rms ** 2 / 4 + slope_err ** 2 * velocity ** 2 + inter_err ** 2 + 2 * velocity * cov[0, 1])
        error = np.sqrt(variance)

        return coef, velocity, vel20, sigma, cov, maxval, error