from astropy.io import fits
import argparse
import numpy as np
import matplotlib
import os

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.modeling import models
from astropy.modeling.models import custom_model
from astropy.modeling import fitting

from analysis import smooth

class Measure:
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

     light_mode: bool
     Light mode for the spectrum. When True the spectrum will plotted in light mode.
     Default is Dark Mode.

    """

    def __init__(self, filename=None, smo=None, gauss=False, twopeak=False, trap=False, dark_mode=False,
                 vel=None, spec=None, rms=None, agc=None, noconfirm=False):
        self.base = True  # for now
        self.smoothed = False
        self.boxcar = False  # tracks if the spectrum has been boxcar smoothed
        self.n = 0
        self.vel = []
        self.spec = []
        self.freq = []
        self.yfit = []
        self.res = []  # and smooth same variable
        self.rms = 0

        self.w50 = 0
        self.w50err = 0
        self.w20 = 0
        self.w20err = 0
        self.vsys = 0
        self.vsyserr = 0
        self.flux = 0
        self.fluxerr = 0
        self.SN = 0

        self.currfit = ""  # current fit type

        if dark_mode:
            plt.style.use('dark_background')
        plt.ion()
        plt.rcParams["figure.figsize"] = (10, 6)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.cid = None
        # Filename modified here, as well
        if filename is not None:
            self.filename = 'AGC{:0}.fits'.format(filename)
            self.load()
            if smo is None:
                self.res = smooth.smooth(self.spec)
            else:
                self.res = smooth.smooth(self.spec, smooth_type=smo)
                self.boxcar = True
        else:
            self.vel = vel
            self.res = spec
            self.spec = spec
            self.rms = rms
            self.filename = 'AGC{}.fits'.format(agc)

        self.smoothed = True

        self.plot()
        props = dict(boxstyle='round', facecolor='crimson')
        self.ax.text(0.1, 1.05, "Measuring", transform=self.ax.transAxes, fontsize=14, bbox=props)

        if filename is not None:
            self.calcRMS()
        # Adds in a choice to change fit if baseline/viewable data leads reducer to want some different fit type.
        if noconfirm == False:

            # if none of the flags are set, skip question
            if not twopeak and not trap and not gauss:
                response = 'no'
            else:
                print('Do you want to keep your previously selected fit type?\n'
                      'Type "yes" and press Enter to keep, type anything else and press Enter to pick a new fit type')
                response = input()

            # allow the user to select new fits types
            if response != 'yes':
                print('The accepted fit methods are: "gauss" for a gaussian, "twopeak" for a double-horned profile fit, or "trap" for a trapezoidal fit.\n'
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

        if gauss:
            first_region = True
            self.currfit = 'gauss'
            self.gauss(first_region)
            self.__write_file(self.__get_comments(), self.currfit)
            input('Gaussian fit complete! Press Enter to end.\n')

        if twopeak:
            self.currfit = 'twopeak'
            first_region = True
            self.twopeakfit(first_region)
            self.__write_file(self.__get_comments(), self.currfit)
            input('Two-peaked fit complete! Press Enter to end.\n')

        if trap:
            self.currfit = 'trap'
            first_region = True
            self.trapezoidal_fit(first_region)
            self.__write_file(self.__get_comments(), self.currfit)
            input('Trapezoidal fit complete! Press Enter to end.\n')

        plt.close()

    def load(self):
        """
        Reads the FITS file and loads the data into the arrays.
        """
        hdul = fits.open(self.filename)
        fitsdata = hdul[1].data
        entries = len(fitsdata)

        self.freq = np.zeros(entries)
        self.vel = np.zeros(entries)
        self.spec = np.zeros(entries)

        for i in range(len(fitsdata)):
            self.vel[i] = fitsdata[i][0]
            self.freq[i] = fitsdata[i][1]
            self.spec[i] = fitsdata[i][2]
            self.n = -1  # masking variable. set to -1 so we know that masking hasn't been done yet. after masking, this changes to the length of the list of the selected region.
            self.smoothed = False  # smoothing boolean. If a hanning or boxcar smooth hasn't been performed, this indicates that smoothing needs to occur before showing the spectrum.

    def plot(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Plots the vel by the spec.
        :param xmin: optional xmin on the spectrum
        :param xmax: optional xmax on the spectrum
        :param ymin: optional ymin on the spectrum
        :param ymax: optional ymax on the spectrum
        """
        plt.cla()
        if not self.smoothed:
            smooth.smooth(self.spec)  # smooth the function (hanning) if not already done

        # plt.ion()
        # fig, ax = plt.subplots()

        if not self.base:
            self.ax.plot(self.vel, self.res, linewidth=1)
            # ymin = min(self.smo)  # if not baselined, use max/min of the smoothed spectrum values
            # ymax = max(self.smo)

        else:
            self.ax.plot(self.vel, self.res, linewidth=1)
            # ymin = min(self.res)  # if already baselined, use max/min of residual values
            # ymax = max(self.smo)

        self.ax.axhline(y=0, dashes=[5, 5])
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        # textbox to tell the user what step they are on
        message = self.currfit
        props = dict(boxstyle='round', facecolor='crimson')
        self.ax.text(0.1, 1.05, message, transform=self.ax.transAxes, fontsize=14, bbox=props)

        title = self.filename[3:-5] # get just the AGC number
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(title))
        self.fig.canvas.draw()

    # def smooth(self, smoothtype=None):
    #     """
    #     Smooths the spec array values
    #     :param smoothtype: Value for smoothing the spectrum, if nothing is passed Hanning smoothing smoothing will occur
    #                         else, boxcar
    #     """
    #     self.n = smoothtype
    #     self.yfit = np.zeros(len(self.spec))
    #     if self.n == -1:  # if masking hasn't occured, neither has baselining. So for the purposes of the smooth function, the yfit array is just zeros.
    #         self.yfit = np.zeros(len(self.spec))
    #     smo = []
    #     for i in range(len(self.spec)):
    #         smo.append(self.spec[i] - self.yfit[i])
    #         # Hanning smooth
    #     window = [.25, .5, .25]
    #     smo = np.convolve(smo, window, mode='same')
    #     if smoothtype is not None:  # Boxcar smooth
    #         window = []
    #         if smoothtype % 2 == 1:  # if the user selected an even int for boxcar smooth, make it odd by adding 1.
    #             smoothtype += 1
    #         for i in range(int(smoothtype)):  # range
    #             window.append(1 / float(smoothtype))
    #         self.boxcar = True
    #         smo = np.convolve(smo, window, 'same')
    #     self.res = smo  # allows the plot to reflect the smoothing.
    #     self.smoothed = True  # function is now smoothed, set to True

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

    def markregions(self, first_region=True):
        """
        Method to interactively select regions on the spectrum
        :return: v, the velocity values in the region
                 s, the spec values in the region
        """

        global mark_regions
        global regions
        regions = []
        #self.plot()

        mark_regions = self.fig.canvas.mpl_connect('button_press_event', self.__markregions_onclick)
        if first_region == True:
            region_message = 'Select a region free of RFI that has the source within it.\n' \
                             'Once done, press Enter if the region is OK, or type "clear" and press Enter to clear region selection.\n'
        else:
            region_message = 'Once done, press Enter to accept the region, or type \'clear\' and press Enter to clear region selection.\n'
        response = input(region_message)
        regions_good = False
        while not regions_good:
            if response == '':
                if len(regions) < 2:
                    response = input("Please complete the region.\n")
                else:
                    regions_good = True
            elif response == 'clear':
                del regions
                regions = []
                # regions.clear()
                self.plot()
                mark_regions = self.fig.canvas.mpl_connect('button_press_event', self.__markregions_onclick)
                response = input('Region cleared! Select a new region now. Press Enter if the region is OK, '
                                 'or type "clear" and press Enter to clear region selection.\n')
            else:
                response = input('Please press Enter if the region is OK, or type "clear" and press enter to clear region selection.\n')
        # self.fig.canvas.mpl_disconnect(mark_regions)
        regions.sort()
        v = list()
        s = list()
        for i in range(len(self.vel)):
            for j in range(len(regions) - 1):
                # constructing v and s lists if they are within the selected region.
                if regions[j] <= self.vel[i] <= regions[j + 1]:
                    v.append(self.vel[i])
                    if len(self.res) is not 0:
                        s.append(self.res[i])
                    else:
                        s.append(self.spec[i])
        # changing v and s into numpy arrays so calculations become shorter.

        del mark_regions, regions
        v = np.asarray(v)
        s = np.asarray(s)
        return v, s

    def __markregions_onclick(self, event):
        """
        Helper method to connect with the GUI and register the clicks for the markregions() method.
        """
        if len(regions) < 2:
            ix, iy = event.xdata, event.ydata
            self.ax.plot([ix, ix], [-100, 1e4], linestyle='--', linewidth=0.7, color='green')
            regions.append(ix)
            if len(regions) is 2:
                self.fig.canvas.mpl_disconnect(mark_regions)

    def gaussfunc(self, v, s, v0, sigma):
        """
        Defenition of the gaussian function to be used in the Gaussian fit.
        """
        return s * np.exp(-(v - v0) ** 2 / (2 * sigma ** 2))

    def gauss(self, first_region=True):
        """
        Method to fit a gaussian fit.
        Assigns the spectrum qualities to their instance variables.
        """
        vel, spec = self.markregions(first_region)
        plt.cla()
        a, aerr, fluxerr, peakmJy, popt, totflux, vsys, vsyserr, w20, w20err, w50, w50err = self.__gaussian_fit(
            vel, spec)
        if self.rms != 0:
            SN = peakmJy / self.rms
        else:  # should not be 0. If it is 0 means the spectrum is not smoothed (rms value has not been calculated)
            self.rms = np.std(self.spec)  # check with prof
            SN = peakmJy / self.rms
        # print(self.rms)
        # print('Area: ' + str(a))
        # print('Area Error: ' + str(aerr))

        #self.__print_values()

        self.ax.plot(vel, spec)  # plotting v and s (notice how the graph zooms into this part of the spectrum)
        self.ax.plot(vel, self.gaussfunc(vel, popt[0], popt[1], popt[2]),
                     'r')  # plotting the gaussian fit to the spectrum
        # something to keep as reference: popt[0] = peak, popt[1] = central velocity, popt[2] = sigma
        self.ax.axhline(y=0, dashes=[5, 5])
        title = self.filename[3:-5]
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(title))
        # plt.pause(1000)
        self.w50 = w50
        self.w50err = w50err
        self.w20 = w20
        self.w20err = w20err
        self.vsys = vsys
        self.vsyserr = vsyserr
        self.flux = totflux
        self.fluxerr = fluxerr
        self.SN = SN

        self.__print_values()

    def __print_values(self):
        print('\n')
        print('W50 = ', self.w50, ' +/- ', self.w50err, ' km/s ')
        print('W20 = ', self.w20, ' +/- ', self.w20err, ' km/s ')
        print('vsys = ', self.vsys, ' +/- ', self.vsyserr, ' km/s')
        print('flux = ', self.flux, ' +/- ', self.fluxerr, ' Jy km/s')
        print('SN: ' + str(self.SN))

    def __gaussian_fit(self, vel, spec):
        """
        Helper method that does all the calculations to derive the qualities.
        Assigns the spectrum qualities to their instance variables.
        """

        peak = max(spec)  # peak of the spectrum
        v0 = np.argmax(spec)  # finding the location of the peak
        plt.style.use('dark_background')
        #mean = sum(vel * spec) / sum(spec)
        #sigma = np.sqrt(abs(sum(spec * (vel - mean) ** 2) / sum(spec)))
        # fitting the gaussian to the spectrum
        #popt, pcov = opt.curve_fit(self.gaussfunc, vel, spec, p0=[peak, mean, sigma])
        
        #Attempt to use astropy's implementation of least squares rather than curve fit
        
        fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
        gauss_model = models.Gaussian1D(amplitude=peak, mean=vel[v0])
        
        gaussfit = fitter(gauss_model, vel, spec)
        
        popt = []
        unc = []
        popt.append(gaussfit.amplitude)
        popt.append(gaussfit.mean)
        popt.append(gaussfit.stddev)
        unc.append(gaussfit.stds[0])
        unc.append(gaussfit.stds[1])
        unc.append(gaussfit.stds[2])
        pcov = gaussfit.cov_matrix
        #unc = np.diag(pcov)  # uncertainty array
        # calculate area
        a = abs(popt[0] * popt[2] * np.sqrt(2 * np.pi))
        aerr = a * np.sqrt((unc[0] ** 2) / popt[0] ** 2 + (unc[2] ** 2) / popt[2] ** 2)
        # calculate w50
        w50 = abs(popt[2] * 2.35482)  # width of gaussian at 1/2 the height
        w50err = 2.35482 * np.sqrt(abs(pcov[2, 2]))
        # calculate w20
        w20 = abs(2 * np.sqrt(2 * np.log(5)) * popt[2])
        w20err = 2 * np.sqrt(2 * np.log(5)) * np.sqrt(abs(pcov[2, 2]))
        # calculate the central velocity
        vsys = popt[1].value  # central velocity occurs at the peak for gaussian
        vsyserr = np.sqrt(abs(pcov[1, 1]))
        # calculate the flux under the curve
        totflux = a / 1000  # from mJy to Jy
        fluxerr = aerr / 1000  # from mJy to Jy
        # calculate signal to noise
        peakmJy = a / (popt[2] * np.sqrt(2 * np.pi))  # peakflux calculation: area/(sigma * sqrt(2pi))
        return a, aerr, fluxerr, peakmJy, popt, totflux, vsys, vsyserr, w20, w20err, w50, w50err

    def twopeakfit(self, first_region=True):
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
        v, s = self.markregions(first_region)
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
            leftv, lefts = self.markregions(first_region)
            fitedge = [min(leftv), max(leftv)]

            # self.plot(None, None, min(self.res), max(self.res))
            leftcoef, leftvel, leftvel20, leftsigma, leftcov, leftmaxval, lefterror = self.edgefit(leftv, lefts, left,
                                                                                                   right)
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
            rightv, rights = self.markregions(first_region)
            fitedge = [min(rightv), max(rightv)]
            rightcoef, rightvel, rightvel20, rightsigma, rightcov, rightmaxval, righterror = self.edgefit(rightv,
                                                                                                          rights, left,
                                                                                                          right)
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

        self.__print_values()

    def __twopeak_calc(self, leftcoef, lefterror, leftvel, leftvel20, rightcoef, righterror, rightvel, rightvel20, s,
                       v):
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

    def trapezoidal_fit(self, first_region=True):
        """
        Method to fit a trapezoidal fit
        :return:
        """
        self.plot()
        print("Please select the region for a trapezoidal fit.")
        v, s = self.markregions(first_region)

        self.plot(min(v), max(v), min(s), max(s))

        global cid
        global coords_trap
        coords_trap = list()

        cid = self.fig.canvas.mpl_connect('button_press_event', self.__trapezoidal_onclick)
        response = input("\nSelect 4 points around the emission."
                         " Two points should be on the outside slope of the left side, and two points should be on the outside slope of the right side."
                         "\nPress Enter when done or type \'clear\' and press Enter to restart selection at anytime.\n")

        trap_good = False
        while not trap_good:
            if response == '':
                # do nothing if the user presses enter to early
                if len(coords_trap) < 4:
                    response = input("Please select 4 points.\n")
                else:

                    coords_trap.sort(key=lambda x: x[0])

                    # unpack the tuple
                    x = [x[0] for x in coords_trap]
                    y = [y[1] for y in coords_trap]

                    SN, W20, W20err, W50, W50err, fluxerr, totflux, vsys, vsyserr = self.__trap_calc(x, y)
                    response = input("Is this fit OK? Press Enter to accept the fit or type \'clear\' and press Enter to restart.\n")
                    if response == '':
                        trap_good = True
                    # if clear should restart

            elif response == 'clear':
                # reset the plot and restart the interaction
                del coords_trap
                coords_trap = []

                self.plot(min(v), max(v), min(s), max(s))
                cid = self.fig.canvas.mpl_connect('button_press_event', self.__trapezoidal_onclick)
                response = input(
                    "Points cleared! Select 4 new points. Press Enter when done or type \'clear\' and press Enter to clear the selection.\n")
            else:
                response = input("Please press Enter when done or type \'clear\' and press Enter to restart.\n")

        # if good, then return values
        self.w50 = W50
        self.w50err = W50err
        self.w20 = W20
        self.w20err = W20err
        self.vsys = vsys
        self.vsyserr = vsyserr
        self.flux = totflux
        self.fluxerr = fluxerr
        self.SN = SN

        self.__print_values()

        # plt.pause(100)

    def __trapezoidal_onclick(self, event):
        """
        Method which connects with the GUI and
        :param event:
        :return:
        """
        ix, iy = event.xdata, event.ydata

        # print(f'x = %d, y = %d' % (ix, iy))
        self.ax.plot(ix, iy, 'ro')

        if (ix, iy) not in coords_trap:
            coords_trap.append((ix, iy))
        if len(coords_trap) == 4:
            self.fig.canvas.mpl_disconnect(cid)

    def __trap_calc(self, x, y):
        """
        Method that does the caluclations of the trapezoidal fit and return them.
        """
        # Figure out the "real" bases, i.e. where the spectrum intersects 0.
        slope = [(y[0] - y[1]) / (x[0] - x[1]), (y[3] - y[2]) / (x[3] - x[2])]
        x_intercept = [x[0] - y[0] / slope[0], x[2] - y[2] / slope[1]]
        # print("Slope ", slope)
        # print("X-intercept ",  x_intercept)
        base_vel = np.array(x_intercept)
        leftedge = []
        rightedge = []
        for i in range(len(self.vel)):
            if self.vel[i] > base_vel[1]:
                rightedge.append(i)
            if self.vel[i] > base_vel[0]:
                leftedge.append(i)
        leftedge = max(leftedge) - 1
        # This throws an error with max, and works correctly with min, so has been modified according.
        rightedge = min(rightedge) + 1
        # rightedge = max(rightedge) + 1
        # Figure out the "peak" locations, i.e. where the spectrum hits a value of peak-rms
        # In the range given by the bases.
        peakval = max(self.spec[rightedge:leftedge]) - self.rms
        peak_vel = [(peakval - y[0]) / slope[0] + x[0], (peakval - y[2]) / slope[1] + x[2]]
        # print(peakval, peak_vel)
        # halfmax = [i*0.5 for i in base_vel] + [i*0.5 for i in peak_vel]
        halfmax = []
        for i in range(len(base_vel)):
            halfmax.append((base_vel[i] + peak_vel[i]) * 0.5)
        e_halfmax = np.mean(abs(self.rms / slope))
        W50 = abs(halfmax[1] - halfmax[0])
        W50err = e_halfmax
        # twentymax = [i*0.8 for i in base_vel] + [i*0.2 for i in peak_vel]
        twentymax = []
        for i in range(len(base_vel)):
            twentymax.append(0.8 * base_vel[i] + 0.2 * peak_vel[i])
        W20 = abs(twentymax[1] - twentymax[0])
        W20err = e_halfmax
        vsys = np.mean(halfmax)
        vsyserr = e_halfmax / np.sqrt(2)
        # y_intercept = [y[0] - slope[0] * x[0], y[2] - slope[1] * x[2]]
        self.ax.plot([base_vel[0], peak_vel[0], peak_vel[1], base_vel[1]], [0., peakval, peakval, 0.])
        self.ax.plot([vsys, vsys], [0, peakval], color='red', linestyle='--', linewidth=0.5)
        self.ax.plot(halfmax, [peakval / 2, peakval / 2], color='red', linestyle='--', linewidth=0.5)
        # Find the delta-v at the center channel
        centerchan = int((leftedge + rightedge) / 2.)
        deltav = abs(self.vel[centerchan + 1] - self.vel[centerchan - 1]) / 2.
        totflux = 0.  # Running total of the integrated flux density
        for i in range(rightedge, leftedge):  # finding the area of the total region
            deltav = abs(self.vel[i] - self.vel[i - 2]) / 2.
            totflux += deltav * self.spec[i]
        totflux = totflux / 1000.
        # SN = 1000 * totflux / W50 * np.sqrt((np.choose(np.greater(W50, 400.), (W50, 400.))) / 20.) / self.rms
        SN = 1000 * totflux / W50 * np.sqrt(W50.clip(min=None, max=400.) / 20.) / self.rms
        fluxerr = 2 * (self.rms / 1000) * np.sqrt(1.4 * W50 * deltav)
        logSN = np.log10(SN)
        Lambda = self.calculate_lambda(deltav, logSN)
        # print(Lambda)
        W50 = W50 - 2 * deltav * Lambda  # Subtract off noise+instrumental broadening effect
        W20 = W20 - 2 * deltav * Lambda
        # Recalculate SN based on new W50.
        SN = 1000 * totflux / W50 * np.sqrt(W50.clip(min=None, max=400.) / 20.) / self.rms
        return SN, W20, W20err, W50, W50err, fluxerr, totflux, vsys, vsyserr

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

    def __get_header(self):
        """
        Returns the header of the FITS file.
        """
        hdul = fits.open(self.filename)
        hdr = hdul[1].header
        return hdr

    def __write_file(self, comments, fittype):
        """
        Writes the values of the quantities to the CSV file.
        :param comments: comments on the fit
        """

        file_exists = os.path.exists('ReducedData.csv')
        if file_exists == False:
            file = open('ReducedData.csv', 'x')
            message_info = (
                        'AGCnr,RA,DEC,Vsys(km/s),W50(km/s),W50err,W20(km/s),flux(Jy*km/s),fluxerr,SN,rms,FitType,comments' + '\n')
            file.write(message_info)

        hdr = self.__get_header()
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
                   str(self.SN) + ',' + str(self.rms) + ',' + 
                   str(fittype) + ',' + str(comments) + '\n'
                )
        file.write(message)

    def __get_comments(self):
        """
        helper method thats gets the user comments on the fit.
        """
        comment = input('\nEnter any comments: ')
        return comment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reduce HI Spectrum using a Gaussian, Two Peak or Trapezoidal Fit.")
    parser.add_argument('filename', metavar='AGC', type=int, help="AGC number of the galaxy, e.g, 104365")
    parser.add_argument('-smo', metavar='smooth', type=int,
                        help="Value for smoothing the spectrum, if nothing is passed Hanning smooth will occur by default;"
                             "\n 'X' for boxcar where X is a postive integer.")
    parser.add_argument('-gauss', action='store_true', help='Do a Gaussian fit of the spectrum')
    parser.add_argument('-twopeak', action='store_true', help='Do a Two Peak fit of the spectrum')
    parser.add_argument('-trap', action='store_true', help='Do a Trapezoidal fit of the spectrum')
    parser.add_argument('-dark_mode', action='store_true', help='Enable dark mode, but not recommended for publication.')

    args = parser.parse_args()

    Measure(filename=args.filename, smo=args.smo, gauss=args.gauss, twopeak=args.twopeak, trap=args.trap,
            dark_mode=args.dark_mode)