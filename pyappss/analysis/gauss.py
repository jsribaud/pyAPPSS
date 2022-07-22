import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.modeling import fitting

from pyappss.analysis.fit import Fit


class Gauss(Fit):

    def __init__(self, agc=None, path="", dark_mode=False, vel=None, spec=None, rms=None):

        super().__init__(agc=agc, path=path, dark_mode=dark_mode, vel=vel, spec=spec, rms=rms)
        self.currstep = 'gauss'

        self.gauss()
        self.write_file(self.get_comments())
        input('Gaussian fit complete! Press Enter to end.\n')
        plt.close()

    def gauss(self, first_region=True):
        """
        Method to fit a gaussian fit.
        Assigns the spectrum qualities to their instance variables.
        """
        self.plot()

        gauss_good = False
        while not gauss_good:
            v, s = self.mark_regions(first_region)
            self.plot()
            a, aerr, fluxerr, peakmJy, popt, totflux, vsys, vsyserr, w20, w20err, w50, w50err = self.__gaussian_fit(v, s)
            if self.rms != 0:
                SN = peakmJy / self.rms
            else:  # should not be 0. If it is 0 means the spectrum is not smoothed (rms value has not been calculated)
                self.rms = np.std(self.spec)  # check with prof
                SN = peakmJy / self.rms
            # print(self.rms)
            # print('Area: ' + str(a))
            # print('Area Error: ' + str(aerr))

            #self.__print_values()

            self.plot(min(v), max(v), min(s), max(s))
            self.ax.plot(v, self.gaussfunc(v, popt[0], popt[1], popt[2]), 'r')  # plotting the gaussian fit to the spectrum
            # something to keep as reference: popt[0] = peak, popt[1] = central velocity, popt[2] = sigma

            response = input("Is this fit OK? Press Enter to accept or any other key for No.")
            if response == "":
                gauss_good = True
            else:
                self.plot()

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

        self.print_values()

    def gaussfunc(self, v, s, v0, sigma):
        """
        Definition of the gaussian function to be used in the Gaussian fit.
        """
        return s * np.exp(-(v - v0) ** 2 / (2 * sigma ** 2))

    def __gaussian_fit(self, vel, spec):
        """
        Helper method that does all the calculations to derive the qualities.
        Assigns the spectrum qualities to their instance variables.
        """

        peak = max(spec)  # peak of the spectrum
        v0 = np.argmax(spec)  # finding the location of the peak
        # mean = sum(vel * spec) / sum(spec)
        # sigma = np.sqrt(abs(sum(spec * (vel - mean) ** 2) / sum(spec)))
        # fitting the gaussian to the spectrum
        # popt, pcov = opt.curve_fit(self.gaussfunc, vel, spec, p0=[peak, mean, sigma])

        # Attempt to use astropy's implementation of least squares rather than curve fit

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
        # unc = np.diag(pcov)  # uncertainty array
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
