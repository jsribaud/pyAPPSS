from astropy.io import fits
from scipy.stats import t

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analysis import smooth
matplotlib.use('Qt5Agg')


class Baseline:
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

    def __init__(self, filename):
        self.filename = 'A{:06}.fits'.format(filename)
        self.smoothed = False
        self.n = -1
        self.m = []
        self.vel = []
        self.spec = []
        self.freq = []
        self.yfit = []
        self.res = []
        self.smo = []
        self.rms = 0

        self.__load()
        self.smo = smooth.smooth(self.spec)
        self.res = self.smo
        self.smoothed = True

        plt.ion()
        plt.rcParams["figure.figsize"] = (10, 6)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.cid = None

        self.__plot()
        self.baseline()
        self.__plot()
        input('Press Enter to end Baseline.')

    def __call__(self):
        return self.vel, self.res, self.rms

    def __load(self):
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
            self.spec[i] = fitsdata[i][2] + fitsdata[i][3]
            self.n = -1  # masking variable. set to -1 so we know that masking hasn't been done yet. after masking, this changes to the length of the list of the selected region.
            self.smoothed = False  # smoothing boolean. If a hanning or boxcar smooth hasn't been performed, this indicates that smoothing nee

    def __plot(self, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.cla()
        if not self.smoothed:
            self.res = smooth.smooth(self.spec)  # smooth the function (hanning) if not already done

        self.ax.plot(self.vel, self.res, linewidth=1)
        self.ax.axhline(y=0, dashes=[5, 5])
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)")
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.fig.canvas.draw()

    def __mask(self):  # interactive masking
        regions = []  # a list of the user selected numbers
        cont = True
        while cont == True:
            num = input('Enter a number, or hit \'Enter\' to finish masking: ')
            if (num == ''):
                cont = False
            else:
                regions.append(int(num))
        regions.sort()
        X = []
        self.m = []  # a list of booleans where True is if the velocity is within the selected region
        for i in range(len(self.vel)):
            j = 0
            inRegion = False
            while (j < len(regions) - 1):
                if (self.vel[i] >= regions[j] and self.vel[i] <= regions[j + 1]):  # in between the marked regions
                    X.append(self.vel[i])
                    inRegion = True
                j = j + 2  # going to the next region
            self.m.append(inRegion)
        self.n = len(X)  # Number of points being fit to
        self.m = np.array(
            self.m)  # converting this to a numpy array so we can make use of other functionalities of the numpy class.

    def markregions(self):

        global mark_regions
        global regions
        regions = []
        # self.plot()

        mark_regions = self.fig.canvas.mpl_connect('button_press_event', self.__markregions_onclick)
        response = input('Press Enter if this region is OK ')
        if response != '':
            regions.clear()
            self.fig.canvas.mpl_connect('button_press_event', self.__markregions_onclick)
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
        if len(regions) < 2:
            ix, iy = event.xdata, event.ydata
            self.ax.__plot([ix, ix], [-100, 1e4], linestyle='--', linewidth=0.7, color ='green')
            regions.append(ix)
            if len(regions) is 2:
                self.fig.canvas.mpl_disconnect(mark_regions)

    def fitpoly(self, order):
        """
        called by calcpoly to determine best order, returns yfit list, rms and p values
        """
        if self.n == -1:
            print(
                'You have not masked this source yet. Please proceed with masking and call fitpoly() again when masking is complete.')
            self.__mask()  # if not already masked, call the mask function before proceeding
        else:
            if order != 0:
                vel = []
                spec = []
                for i in range(len(self.m)):
                    if self.m[i]:
                        vel.append(self.vel[i])
                        spec.append(self.spec[i])
                vel = np.asarray(vel)
                spec= np.asarray(spec)
                coeff, cov= np.polyfit(vel, spec, deg=order, cov=True)  # (list)
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
                        yval += (coeff[j] * (self.vel[i] ** (
                                    order - j)))  # generating y-values based on the calculated coefficient array.
                    yfit.append(yval)

                # calculate the rms
                res = []  # list of baseline-subtracted spectrum values. Again, note that this is not the class variable, self.res. Also comes later.
                rmsarr = []
                for i in range(len(yfit)):
                    res.append(self.spec[i] - yfit[i])  # subtracting the baseline
                    np.asarray(res)
                    if (self.m[i]):  # if part of the masked region, include it in the rms.
                        rmsarr.append(res[i])
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
        rms = 0
        p = 0
        yfit = []

        for order in range(omax, -1, -1):
            (rms, p, yfit) = self.fitpoly(order)  # call fitpolylbw to get rms and p values
            rmsval.insert(0, rms)  # putting each value at the front of the list so it's in the right order
            pval.insert(0, p)
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

    def baseline(self):
        # self.smooth()  # smoothing the function
        self.__mask()  # masking the function
        recommended, rmsval, pval = self.calcpoly()  # calculating the recommended order of the function
        titles = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']
        print('Statistics for each fit order:')
        print(' order  rms(mJy)')
        for i in range(len(titles)):
            if i is recommended:
                print('  '+titles[i]+'   '+str(rmsval[i]) + '*')
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
            self.ax.plot(self.vel, self.yfit, linestyle='--', color='green', linewidth='1', label ='yfit')

            response = input()
            if response is '':
                accepted = True
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Baseline HI spectra")
    parser.add_argument('agc_number', metavar='AGC', type=int, help="AGC number of the galaxy, e.g, 104365")
    args = parser.parse_args()
    Baseline(args.agc_number)


