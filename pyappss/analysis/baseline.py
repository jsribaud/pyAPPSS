from astropy.io import fits
from scipy.stats import t

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib

from pyappss.analysis import smooth
from pyappss.analysis.util import Util

matplotlib.use('Qt5Agg')


class Baseline(Util):
    """
    Interactively baseline a spectrum.
    First asks user to select emission/RFI-free regions
    Then interactively fits polynomials of different orders

    When the instance is called the vel, spec, and rms are returned.
    for e.g, b = Baseline(12159)
    b()[0] is vel, b()[1] is spec, b()[2] is rms
    Parameters
    ----------
    filename : int
    AGC number of the galaxy, e.g, 104365
    """

    def __init__(self, agc, smo, path="", noconfirm=None, dark_mode=False):
        super().__init__(agc, smo, path, dark_mode)

        self.n = -1
        self.m = []

        self.load()
        self.smo = smooth.smooth(self.spec, smooth_type=smo)
        self.res = self.smo
        self.smoothed = True

        # for information bubble
        self.currstep = "Baselining"
        self.stepcolor = "skyblue"

        self.plot()
        self.baseline(noconfirm)
        self.plot()
        input('Press Enter to end Baseline.')
        plt.close()  # close the window (measure makes a new one)

    def __call__(self):
        return self.vel, self.res, self.rms

    def baseline(self, noconfirm=False):
        # self.smooth()  # smoothing the function
        self.__mask()  # masking the function
        recommended, rmsval, pval = self.calcpoly()  # calculating the recommended order of the function
        titles = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']
        print('Statistics for each fit order:')
        print(' order  rms(mJy)')
        for i in range(len(titles)):
            if i is recommended:
                print('  ' + titles[i] + '   ' + str(rmsval[i]) + '*')
            else:
                print('  ' + titles[i] + '   ' + str(rmsval[i]))

        print('Plotting a fit of the recommended order (' + titles[recommended] + ').')
        print('  Enter an order [0-' + '9' + '] to plot and select.')
        print('  Press [enter] to accept.')
        order = recommended

        accepted = False
        while not accepted:
            (self.rms, self.p, self.yfit) = self.fitpoly(
                order)  # receiving rms, p, and yfit from the fitpoly function, using previously recommended order
            self.res = (np.asarray(self.smo) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
            self.ax.plot(self.vel, self.yfit, linestyle='--', color='black', linewidth='1', label='yfit')

            response = input()
            if response is '':
                if noconfirm:
                    accepted = True
                else:
                    self.plot()
                    response = input('Press Enter again to confirm this baseline fit.'
                                     ' Type anything else and hit enter to try again.\n')
                    if response is '':
                        accepted = True
                    else:
                        self.res = np.asarray(self.smo)
                        self.plot()
                        if order < 10:
                            print('Plotting a ' + titles[order] + ' order fit.')
                        else:
                            print('Plotting a ' + str(order) + 'order fit.')
            elif int(response) is -1:
                accepted = True
            else:
                order = int(response)
                line = [line for line in self.ax.lines if line.get_label() == 'yfit'][0]
                self.ax.lines.remove(line)
                if order < 10:
                    print('Plotting a ' + titles[order] + ' order fit.')
                else:
                    print('Plotting a ' + str(order) + ' order fit.')

    def __mask(self):

        global mask_regions
        global regions
        regions = []

        mask_regions = self.fig.canvas.mpl_connect('button_press_event', self.__mask_regions_onclick)
        response = input(
            'Please select regions to be used for baselining. These regions should be free of RFI and the source.'
            '\nPress Enter once done selecting regions, or type \'clear\' and press Enter to clear region selection and start over.\n')
        done_baselining = False
        while not done_baselining:
            if response == '':
                if len(regions) % 2 == 1 or len(regions) == 0:  # do nothing if enter and only an odd number of region selections
                    response = input("Please complete your region(s).\n")
                else:
                    self.fig.canvas.mpl_disconnect(mask_regions)
                    print('Calculating best baseline fit. Please wait.\n')
                    done_baselining = True
            elif response == 'clear':
                regions.clear()
                self.plot()
                response = input('Regions cleared! Select new regions now.\n'
                                 'Press Enter once done selecting regions, or type \'clear\' and press Enter to clear region selection and start over.\n')
                # self.fig.canvas.mpl_connect('button_press_event', self.__maskregions_onclick)
            else:
                response = input()
        X = []
        self.m = []
        regions.sort()
        for i in range(len(self.vel)):
            j = 0
            inRegion = False
            while j < len(regions) - 1:
                # Used to set regions as between each pair of entries.
                if self.vel[i] >= regions[j] and self.vel[i] <= regions[j + 1]:
                    X.append(self.vel[i])
                    inRegion = True
                j = j + 2
            self.m.append(inRegion)
        self.n = len(X)
        self.m = np.array(self.m)

    def __mask_regions_onclick(self, event):

        if event.inaxes:
            ix, iy = event.xdata, event.ydata
            # Bounds have been extended in case of originally odd baselines.
            self.ax.plot([ix, ix], [-1e4, 1e4], linestyle='--', linewidth=0.7, color='green')
            regions.append(ix)

    def fitpoly(self, order):
        """
        called by calcpoly to determine best order, returns yfit list, rms and p values
        """
        if self.n == -1:
            print('You have not masked this source yet. '
                  'Please proceed with masking and call fitpoly() again when masking is complete.')
            self.__mask()  # if not already masked, call the mask function before proceeding
        else:
            if order != 0:
                # vel = []
                # spec = []
                # for i in range(len(self.m)):
                #     if self.m[i]:
                #         vel.append(self.vel[i])
                #         spec.append(self.spec[i])

                vel = [self.vel[i] for i in range(len(self.m)) if self.m[i]]
                spec = [self.spec[i] for i in range(len(self.m)) if self.m[i]]

                vel = np.asarray(vel)
                spec = np.asarray(spec)
                coeff, cov = np.polyfit(vel, spec, deg=order, cov=True)  # (list)
                if min(np.diag(cov)) < 0:
                    sigma = np.ones(order + 1) * 1e6
                else:
                    sigma = np.sqrt(np.diag(cov))  # (list)
                t_test = coeff[0] / sigma[0]  # sigma = std dev
                dof = self.n - (order + 1) - 1  # degree of freedom
                p = 2 * t._pdf((-abs(t_test)), dof)  # probability distribution function

                # produce fitted y values
                yfit = []  # note that this is not the class variable, self.yfit. That comes later.
                for i in range(len(self.vel)):
                    yval = 0
                    for j in range(order + 1):
                        yval += (coeff[j] * (self.vel[i] ** (order - j)))  # generating y-values based on the calculated coefficient array.
                    yfit.append(yval)

                # calculate the rms
                # res = []  # list of baseline-subtracted spectrum values. Again, note that this is not the class variable, self.res. Also comes later.
                # rmsarr = []
                # for i in range(len(yfit)):
                #     res.append(self.spec[i] - yfit[i])  # subtracting the baseline
                #     np.asarray(res)
                #     if self.m[i]:  # if part of the masked region, include it in the rms.
                #         rmsarr.append(res[i])

                # reduces computation time significantly
                res = self.spec - yfit  # list of baseline-subtracted spectrum values.
                    # Again, note that this is not the class variable, self.res. Also comes later.
                rmsarr = np.asarray([res[i] for i in range(len(yfit)) if self.m[i]])
                rms = np.std(rmsarr)  # (number)
            else:
                coeff = np.mean(self.spec)  # coefficient: fitted y val
                yfit = np.zeros(len(self.vel)) + coeff  # constant nchan
                rms = np.std(self.spec)  # standard deviation of spec

                t_test = np.mean(self.spec) / (rms / self.n)  # t_test-test statistic
                dof = self.n - 2  # N-2? Degrees of freedom
                p = float(t.pdf(2 * -abs(t_test), dof))  # probability distribution function
            return rms, p, yfit

    def calcpoly(self):
        """
        iterates through orders to find the best rms value
        """
        recommend = -1  # recommended order by program
        omax = 9  # maximum order is 9
        cutoff = 0.05  # determines how much of a change is significant.
        rmsval = []  # array for rms vals
        pval = []  # list for p-values

        for order in range(omax + 1):  # exclusive end
            (rms, p, yfit) = self.fitpoly(order)  # call fitpolylbw to get rms and p values
            rmsval.append(rms)
            pval.append(p)

        # find recommended order
        for i in range(omax):
            if recommend == -1 and pval[i] > cutoff:
                recommend = i - 1
                if recommend == -1:
                    recommend = 0
                break

        if recommend == -1:
            recommend = omax  # if nothing else seems to work, recommend the 9th order polynomial
        return recommend, rmsval, pval
