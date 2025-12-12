from astropy.io import fits
from astropy.table import Table
from scipy.stats import t
from astropy import units as u

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import os
import sys

#from pyappss.analysis import smooth
import smooth2
import load2

matplotlib.use('Qt5Agg')


class Baseline:
    """
    Interactively baseline a spectrum.
    First asks user to select emission/RFI-free regions
    Then interactively fits polynomials of different orders

    When the instance is called the vel, spec, and rms are returned.
    for e.g, b = Baseline(12159)
    b()[0] is vel, b()[1] is spec, b()[2] is rms , b()[3] is specrms, b()[4] is selected baseline, b()[5] is data table - with updated baseline, b()[6] is file header
    Parameters
    ----------
    filename : int
    AGC number of the galaxy, e.g, 104365
    """

    def __init__(self, filename, smooth_int=1, path="", noconfirm=False, dark_mode=False,
                 gbtidl_fits=False,no_smooth=False):
        # Filename modified: to AGCxxxxx.fits
        if '.fits' in filename:
            self.filename = filename
        else:
            self.filename = 'AGC{}.fits'.format(filename)

        self.path = self.filename
        self.smoothed = False
        self.n = -1
        self.m = []
        self.vel = []
        self.spec = []
        self.freq = []
        self.yfit = []
        self.res = []
        self.smo = smooth_int
        self.xrms = 0
        self.specrms = 0
        self.rms = 0

        self.tab, self.hdr = load2.load(self.filename,gbtidl_fits=gbtidl_fits)
        self.hdr['ORIGFILE'] = self.filename
        self.freq = np.array(self.tab['FREQUENCY'])
        self.vel = np.array(self.tab['VELOCITY'])
        self.spec = np.array(self.tab['FLUX'])
        self.base = np.array(self.tab['BASELINE'])
        self.weight = np.array(self.tab['WEIGHT'])

        if dark_mode:
            plt.style.use('dark_background')
        plt.ion()
        plt.rcParams["figure.figsize"] = (10, 6)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.cid = None
        noconfirm = noconfirm

        if no_smooth:
            stab, smx = smooth2.smooth(self.tab, no_smooth=True)
            self.hdr['Hanning'] = 'False'
            self.hdr['Boxcar'] = 1
            self.hdr['VELRES'] = str(smx[1])
        else:
            stab, smx = smooth2.smooth(self.tab, box_width=smooth_int)
            self.hdr['Hanning'] = 'True'
            self.hdr['Boxcar'] = smooth_int
            self.hdr['VELRES'] = str(smx[1])
        self.smoothed = True
        self.stab = stab
        self.sflx = stab['FLUX']
        self.svel = stab['VELOCITY']
        self.sbase = stab['BASELINE']
        self.sweight = stab['WEIGHT']
        #self.res = smx

        self.__plot(first_plot=True)

        response = input('The current baseline for the loaded data is shown in RED. \n'+
                         'Would you like to go through the basline process again? \n'+
                         'Enter "y" to fit a new baseline. Hit "Return" to continue with the current baseline. \n')

        if response=='y':
            self.fit_baseline(noconfirm)
            # update flux and baseline arrays
            self.stab['FLUX'] = self.sflx - self.yfit
            self.stab['BASELINE'] = self.yfit
            self.sflx = self.stab['FLUX']
            self.sbase = self.stab['BASELINE']
            self.hdr['RMS'] = str(self.rms)+' mJy'
            ##
            self.__plot()
            ##
            input('Press Enter to end Baseline.')

        else:
            self.fit_baseline(noconfirm,rms_only=True)
            self.hdr['RMS'] = str(self.rms)+' mJy'
            ##
            self.__plot()

    def __call__(self):
        return self.tab, self.hdr, self.stab, self.xrms, self.rms

    def __plot(self, xmin=None, xmax=None, ymin=None, ymax=None, first_plot=False):
        plt.cla()

        props = dict(boxstyle='round', facecolor='skyblue')
        self.ax.text(0.1, 1.05, "Baselining", transform=self.ax.transAxes, fontsize=14, bbox=props)
        if first_plot:
            self.ax.step(self.svel, self.sflx + self.sbase, linewidth=1, color='k')
            self.ax.plot(self.svel, self.sbase, linewidth=2, ls='--', color='r')
        else:
            self.ax.step(self.svel, self.sflx, linewidth=1, color='k')
        self.ax.axhline(y=0, dashes=[5, 5])
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        title = self.filename[:-5]  # get just the AGC number
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='{}'.format(title))

        self.fig.canvas.draw()

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
            else:
                response = input()
        X = []
        sX = []
        self.m = []
        self.sm = []
        regions.sort()
        self.specrms=0
        nrms=0
        for i in range(len(self.vel)):
            j = 0
            inRegion = False
            while j < len(regions) - 1:
                # Used to set regions as between each pair of entries.
                if self.vel[i] >= regions[j] and self.vel[i] <= regions[j + 1]:
                    X.append(self.vel[i])
                    inRegion = True
                    self.specrms=self.specrms+self.spec[i]*self.spec[i]
                    nrms=nrms+1
                j = j + 2
            self.m.append(inRegion)
        self.specrms = 0
        for i in range(len(self.svel)):
            j = 0
            inRegion = False
            while j < len(regions) - 1:
                # Used to set regions as between each pair of entries.
                if self.svel[i] >= regions[j] and self.svel[i] <= regions[j + 1]:
                    sX.append(self.svel[i])
                    inRegion = True
                    self.specrms=self.specrms+self.sflx[i]*self.sflx[i]
                    nrms = nrms + 1
                j = j + 2
            self.sm.append(inRegion)
        self.n = len(X)
        self.sn = len(sX)
        self.m = np.array(self.m)
        self.sm = np.array(self.sm)
        self.specrms = np.sqrt(self.specrms / nrms)
        self.xrms = np.std(self.spec[self.m])
        self.rms = np.std(self.sflx[self.sm])
        print('Original spectrum rms in selected regions is', f"{self.xrms:.2f}")
        print('Smoothed spectrum rms in selected regions is', f"{self.rms:.2f}")

    def __mask_rms(self):
        global mask_regions
        global regions
        regions = []

        mask_regions = self.fig.canvas.mpl_connect('button_press_event', self.__maskregions_onclick)
        response = input(
            'Please select regions to be used for rms calculation. These regions should be free of RFI and the source (if present).'
            '\nPress Enter once done selecting regions, or type \'clear\' and press Enter to clear region selection and start over.\n')
        done_rmsing = False
        while not done_rmsing:
            if response == '':
                if len(regions) % 2 == 1 or len(
                        regions) == 0:  # do nothing if enter and only an odd number of region selections
                    response = input("Please complete your region(s).\n")
                else:
                    self.fig.canvas.mpl_disconnect(mask_regions)
                    done_rmsing = True
            elif response == 'clear':
                regions.clear()
                self.__plot()
                response = input('Regions cleared! Select new regions now.\n'
                                 'Press Enter once done selecting regions, or type \'clear\' and press Enter to clear region selection and start over.\n')
            else:
                response = input()
        X = []
        self.m = []
        sX = []
        self.sm = []
        regions.sort()
        self.specrms = 0
        nrms = 0
        for i in range(len(self.vel)):
            j = 0
            inRegion = False
            while j < len(regions) - 1:
                # Used to set regions as between each pair of entries.
                if self.vel[i] >= regions[j] and self.vel[i] <= regions[j + 1]:
                    X.append(self.vel[i])
                    inRegion = True
                    self.specrms=self.specrms+self.spec[i]*self.spec[i]
                    nrms = nrms + 1
                j = j + 2
            self.m.append(inRegion)
        self.specrms = 0
        nrms = 0
        for i in range(len(self.svel)):
            j = 0
            inRegion = False
            while j < len(regions) - 1:
                # Used to set regions as between each pair of entries.
                if self.svel[i] >= regions[j] and self.svel[i] <= regions[j + 1]:
                    sX.append(self.svel[i])
                    inRegion = True
                    self.specrms=self.specrms+self.sflx[i]*self.sflx[i]
                    nrms = nrms + 1
                j = j + 2
            self.sm.append(inRegion)
        self.n = len(X)
        self.m = np.array(self.m)
        self.sn = len(sX)
        self.sm = np.array(self.sm)
        self.specrms = np.sqrt(self.specrms/nrms)
        self.xrms = np.std(self.spec[self.m])
        self.rms = np.std(self.sflx[self.sm])
        print('Original spectrum rms in selected regions is', f"{self.xrms:.2f}")
        print('Smoothed spectrum rms in selected regions is', f"{self.rms:.2f}")

    def __maskregions_onclick(self, event):

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

            # Bic implementation for basline.py
            vel = self.svel[self.sm]
            spec = self.sflx[self.sm]

            # fit the polynomial of order (order)
            fitted_series = np.polynomial.polynomial.Polynomial.fit(vel, spec, order).convert().coef

            # create and fit polynonial object
            p = np.polynomial.Polynomial(fitted_series)
            y_fit = p(self.svel)

            ## Alternative calculation for mse
            # mse = np.mean((self.spec - y_fit) ** 2) # this calculates the mean squared error
            # for the entire spectrum, not just the masked region.

            # Keeping the same rms calculation as before for rms
            res = self.sflx - y_fit  # list of baseline-subtracted spectrum values.
                    # Again, note that this is not the class variable, self.res. Also comes later.
            rmsarr = np.asarray([res[i] for i in range(len(y_fit)) if self.sm[i]])
            rms = np.std(rmsarr)  # (number)

            # calculate MSE for bic/aic values
            mse = np.mean([(res[i])**2 for i in range(len(y_fit)) if self.sm[i]])
            # aic = 2 * (order + 1) + self.n * np.log(2 * np.pi) + self.n * np.log(mse) + self.n
            bic = (order + 1)*np.log(self.sn) + self.sn * np.log(2 * np.pi) + self.sn * np.log(mse) + self.sn
            return rms, bic, y_fit


    def calcpoly(self):
        """
        iterates through orders to find the best rms value
        """
        recommend = -1  # recommended order by program
        omax = 9  # maximum order is 9
        #cutoff = 0.05  # determines how much of a change is significant.
        rmsval = []  # array for rms vals
        bicval = []  # list for p-values

        print('Calculating best fit order. Please wait.\n')
        for order in range(omax + 1):  # exclusive end
            (rms, bic, yfit) = self.fitpoly(order)  # call fitpoly to get rms and p values
            # append value to the list
            rmsval.append(rms)
            bicval.append(bic)
        # grabs the three lowest bic indicies 
        # index corresponds to order
        recommend = np.argsort(bicval)[:3]     

        return recommend, rmsval, bicval

    def fit_baseline(self, noconfirm=False,rms_only=False):
        if rms_only==True:
            self.__plot()
            self.__mask_rms()
        else:
            self.__mask()  # masking the function
            recommended, rmsval, bicval = self.calcpoly()  # calculating the recommended order of the function
            titles = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']  # titles for the orders
            print('Statistics for each fit order:')
            print(' order  rms(mJy)')
            for i in range(len(titles)):
                if i in recommended:
                    print('  ' + titles[i] + '   ' + str(rmsval[i]) + '*' * (3 - np.where(recommended == i)[0][0]))
                else:
                    print('  ' + titles[i] + '   ' + str(rmsval[i]))

            print('Plotting a fit of the recommended order (' + titles[recommended[0]] + ').')
            print('  An asterisk (***) indicates the recommended order.')
            print('  Enter an order [0-' + '9' + '] to plot and select.')
            print('  Enter [10] to shuffle through the top three orders.')
            print('  Press [enter] to accept.')
            order = recommended[0]

            accepted = False
            while not accepted:
                plt.title(f"Plot for order {order}")
                (self.frms, self.aic, self.yfit) = self.fitpoly(
                    order)  # receiving rms, p, and yfit from the fitpoly function, using previously recommended order
                self.res = (np.asarray(self.sflx) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
                self.ax.plot(self.svel, self.yfit, linestyle='--', color='orange', linewidth='2', label='yfit')

                response = input()

                if response == '':
                    if noconfirm:
                        accepted = True
                    else:
                        self.__plot()
                        self.ax.plot(self.svel, self.yfit, linestyle='--', color='orange', linewidth='2', label='yfit')
                        response = input('Press Enter again to confirm this baseline fit.'
                                         ' Type anything else and hit enter to try again.\n')
                        if response == '':
                            accepted = True
                        else:
                            self.res = np.asarray(self.sflx)
                            #self.res = np.asarray(self.spec)
                            self.__plot()
                            if order < 10:
                                print('Plotting a ' + titles[order] + ' order fit.')
                            else:
                                print('Plotting a ' + str(order) + 'order fit.')
                elif int(response) == -1:
                    accepted = True

                # this allows us to print each recomended order
                elif int(response) == 10:
                    print("Plotting the three recomended orders.")
                    # iterate through the top three recommended orders
                    for i in range(3):
                        # remove the previous fit
                        line = [line for line in self.ax.lines if line.get_label() == 'yfit'][0]
                        print(line)
                        # plot the given order then wait 5 seconds to remove it
                        (self.frms, self.aic, self.yfit) = self.fitpoly(
                            recommended[i])  # receiving rms, p, and yfit from the fitpoly function, using previously recommended order
                        self.res = (np.asarray(self.sflx) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
                        self.ax.plot(self.svel, self.yfit, linestyle='--', color='orange', linewidth='1', label='yfit')
                        plt.title(f"Plot for order {recommended[i]}")
                        plt.pause(3)

                    # restore to the reccomened order
                    line = [line for line in self.ax.lines if line.get_label() == 'yfit'][0]
                    #self.ax.lines.remove(line)
                    print("Shuffle has concluded.")
                    order = recommended[0]

                else:
                    order = int(response)
                    line = [line for line in self.ax.lines if line.get_label() == 'yfit'][0]
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

