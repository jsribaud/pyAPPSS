from astropy.io import fits
from scipy.stats import t

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib

from pyappss.analysis import smooth

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

    def __init__(self, filename, smooth_int, path="", noconfirm=False, dark_mode=False):
        # Filename modified: to AGCxxxxx.fits
        # May align more favorably with desired format, may not. Matches convert.py naming.
        print('in baseline, filename = ',filename)
        if '.fits' in filename:
            self.filename = filename
        else:
            self.filename = 'AGC{}.fits'.format(filename)
        print('in baseline ',self.filename)
        self.path = pathlib.PurePath(path + "/" + self.filename)
        self.path = self.filename
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
        self.smo = smooth.smooth(self.spec, smooth_type=smooth_int)
        self.res = self.smo
        self.smoothed = True

        if dark_mode:
            plt.style.use('dark_background')
        plt.ion()
        plt.rcParams["figure.figsize"] = (10, 6)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.cid = None

        noconfirm = noconfirm

        self.__plot()
        self.fit_baseline(noconfirm)
        self.__plot()
        input('Press Enter to end Baseline.')
        plt.close()  # close the window (measure makes a new one)

    def __call__(self):
        return self.vel, self.res, self.rms

    def __load(self):
        """
        Reads the FITS file and loads the data into the arrays.
        """
        from astropy.table import Table
        hdul = Table.read(self.path)
        self.freq = hdul['FREQUENCY']
        self.vel = hdul['VHELIO']
        self.spec = hdul['FLUX']
        self.weight = hdul['WEIGHT']
        self.baseline = hdul['BASELINE']

        self.n = -1  # masking variable. set to -1 so we know that masking hasn't been done yet. after masking, this changes to the length of the list of the selected region.
        self.smoothed = False  # smoothing boolean. If a hanning or boxcar smooth hasn't been performed, this indicates that smoothing nee

    def __plot(self, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.cla()
        if not self.smoothed:
            self.res = smooth.smooth(self.spec)  # smooth the function (hanning) if not already done
        props = dict(boxstyle='round', facecolor='skyblue')
        self.ax.text(0.1, 1.05, "Baselining", transform=self.ax.transAxes, fontsize=14, bbox=props)
        self.ax.plot(self.vel, self.res, linewidth=1)
        self.ax.axhline(y=0, dashes=[5, 5])
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        title = self.filename[3:-5]  # get just the AGC number
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(title))

        self.fig.canvas.draw()

    # def __mask(self):  # interactive masking
    #     regions = []  # a list of the user selected numbers
    #     cont = True
    #     while cont == True:
    #         num = input('Enter a number, or hit \'Enter\' to finish masking: ')
    #         if (num == ''):
    #             cont = False
    #         else:
    #             regions.append(int(num))
    #     regions.sort()
    #     X = []
    #     self.m = []  # a list of booleans where True is if the velocity is within the selected region
    #     for i in range(len(self.vel)):
    #         j = 0
    #         inRegion = False
    #         while (j < len(regions) - 1):
    #             if (self.vel[i] >= regions[j] and self.vel[i] <= regions[j + 1]):  # in between the marked regions
    #                 X.append(self.vel[i])
    #                 inRegion = True
    #             j = j + 2  # going to the next region
    #         self.m.append(inRegion)
    #     self.n = len(X)  # Number of points being fit to
    #     self.m = np.array(
    #         self.m)  # converting this to a numpy array so we can make use of other functionalities of the numpy class.

    # Alternative code written using functions from measure.py, reapplied to masking. Original method is above, but requires manually entering values.

    # This masking variant requires only clicking to select regions.
    def __mask(self):

        global mask_regions
        global regions
        regions = []

        mask_regions = self.fig.canvas.mpl_connect('button_press_event', self.__maskregions_onclick)
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
                self.__plot()
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

    def __maskregions_onclick(self, event):

        ix, iy = event.xdata, event.ydata
        # Bounds have been extended in case of originally odd baselines.
        self.ax.plot([ix, ix], [-1e4, 1e4], linestyle='--', linewidth=0.7, color='green')
        regions.append(ix)
        # self.fig.canvas.mpl_disconnect(mask_regions)

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
        regions.sort()  # allows the user to insert the regions in any order
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
            self.ax.__plot([ix, ix], [-100, 1e4], linestyle='--', linewidth=0.7, color='green')
            regions.append(ix)
            if len(regions) is 2:
                self.fig.canvas.mpl_disconnect(mark_regions)

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

    def fit_baseline(self, noconfirm=False):
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
                    self.__plot()
                    response = input('Press Enter again to confirm this baseline fit.'
                                     ' Type anything else and hit enter to try again.\n')
                    if response is '':
                        accepted = True
                    else:
                        self.res = np.asarray(self.smo)
                        self.__plot()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Baseline HI spectra")
    parser.add_argument('agc_number', metavar='AGC', type=int, help="AGC number of the galaxy, e.g, 104365")
    args = parser.parse_args()
    Baseline(args.agc_number)

