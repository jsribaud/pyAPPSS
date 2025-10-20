from astropy.io import fits
from astropy.table import Table
from scipy.stats import t

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import os
import sys

#from pyappss.analysis import smooth
import smooth2

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

    def __init__(self, filename, smooth_int, path="", noconfirm=False, dark_mode=False):
        # Filename modified: to AGCxxxxx.fits
        # May align more favorably with desired format, may not. Matches convert.py naming.
        #print('in baseline, filename = ',filename)
        if '.fits' in filename:
            self.filename = filename
        else:
            self.filename = 'AGC{}.fits'.format(filename)
        #print('in baseline ',self.filename)
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
        self.specrms = 0
        self.srms = 0
        

        self.__load()



        if dark_mode:
            plt.style.use('dark_background')
        plt.ion()
        plt.rcParams["figure.figsize"] = (10, 6)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.cid = None

        noconfirm = noconfirm



        self.smo = smooth2.smooth(self.spec, smooth_type=smooth_int)
        self.smoothed = True

        self.res = self.smo

        self.__first_plot()

        response = input('The current baseline for the loaded data is shown in RED. \n'+
                         'Would you like to go through the basline process again? \n'+
                         'Enter "y" to fit a new baseline. Hit "Return" to continue with the current baseline. \n')

        if response=='y':
            self.fit_baseline(noconfirm)
            self.__plot()

            #update flux and baseline arrays before ending
            self.data['FLUX']=self.spec-self.yfit
            self.data['BASELINE']=self.yfit
            self.data['SM_FLUX']=self.smo
            input('Press Enter to end Baseline.')
            self.hdr['RMS'] = self.rms
            self.hdr['Hanning'] = 'True'
            if smooth_int is not None:
                self.hdr['Boxcar'] = smooth_int
            else:
                self.hdr['Boxcar'] = 'False'
            self.hdr['SM_RMS'] = self.srms
            #plt.close()  # close the window (measure makes a new one)
        else:

            #self.res = self.smo#self.data['FLUX']
            self.fit_baseline(noconfirm,rms_only=True)
            self.smoothed = False
            self.data['SM_FLUX'] = self.smo
            self.hdr['RMS'] = self.rms
            self.hdr['Hanning'] = 'True'
            if smooth_int is not None:
                self.hdr['Boxcar'] = smooth_int
            else:
                self.hdr['Boxcar'] = 'False'
            self.hdr['SM_RMS'] = self.srms
            self.__plot()
            #plt.close()

    def __call__(self):
        return self.vel, self.res, self.rms, self.srms, self.yfit, self.data, self.hdr

    def __load(self):
        """
        Reads the FITS file and loads the data into the arrays.
        """
        hdulx = fits.open(self.path)
        tdata = Table(hdulx[1].data)#Table.read(self.path)
        self.hdr = hdulx[1].header
        #hdul = Table.read(self.path)
        #tmptab = Table.read(self.path)
        velname = 'VELOCITY'
        fluxname = 'FLUX'
        freqname = 'FREQUENCY'
        weightname = 'WEIGHT'
        baselinename = 'BASELINE'
        vel_list = ['VHELIO', 'Vhelio', 'VELOCITY', 'Velocity', 'VEL', 'Vel']
        freq_list = ['FREQUENCY', 'Frequency', 'FREQ', 'Freq']
        flux_list = ['FLUX', 'Flux', 'SPEC', 'Spec']
        weight_list = ['WEIGHT', 'Weight']
        baseline_list = ['BASELINE', 'Baseline']
        for ii in range(len(tdata.colnames)):
            if tdata.colnames[ii] in vel_list:
                velname = tdata.colnames[ii]
            if tdata.colnames[ii] in freq_list:
                freqname = tdata.colnames[ii]
            if tdata.colnames[ii] in flux_list:
                fluxname = tdata.colnames[ii]
            if tdata.colnames[ii] in weight_list:
                weightname = tdata.colnames[ii]
            if tdata.colnames[ii] in baseline_list:
                baselinename = tdata.colnames[ii]

        #cnames = [velname,freqname,fluxname,weightname,baselinename]

        try:
            tmpsort = np.argsort(tdata[velname])  # checks for spectra format (increasing v or f)
            hdul = tdata[tmpsort]  # forces spectra to increase in v
            self.vel = np.array(hdul[velname].value, 'd')
            self.freq = np.array(hdul[freqname].value, 'd')
            self.spec = np.array(hdul[fluxname].value, 'd')
            self.weight = np.array(hdul[weightname].value, 'd')
            self.baseline = np.array(hdul[baselinename].value, 'd')
            self.data = hdul
            self.data.rename_column(velname, 'VELOCITY')
            self.data.rename_column(freqname, 'FREQUENCY')
            self.data.rename_column(fluxname, 'FLUX')
            self.data.rename_column(weightname, 'WEIGHT')
            self.data.rename_column(baselinename, 'BASELINE')
        except:
            print('Unable to properly load fits table - column names unrecognized. Run gbtfits_load() on your GBT'+
                  'data to produce an appropriate file format. Or use the flag -gbt_fits to load a GBT .FITS file.')
            sys.exit()
        '''try:
            tmpsort = np.argsort(tdata['VHELIO'])  # checks for spectra format (increasing v or f)
            hdul = tdata[tmpsort]  # forces spectra to increase in v
            self.vel = np.array(hdul['VHELIO'].value,'d')
        except:
            tmpsort = np.argsort(tdata['VELOCITY'])  # checks for spectra format (increasing v or f)
            hdul = tdata[tmpsort]  # forces spectra to increase in v
            self.vel = np.array(hdul['VELOCITY'].value,'d')
        try:
            self.freq = np.array(hdul['FREQUENCY'].value,'d')
            self.spec = np.array(hdul['FLUX'].value,'d')
            self.weight = np.array(hdul['WEIGHT'].value,'d')
            self.baseline = np.array(hdul['BASELINE'].value,'d')
            self.data = hdul
        except:
            self.freq = np.zeros(len(hdul))
            self.spec = np.array(hdul['FLUX'].value,'d')
            self.weight = np.ones(len(hdul))
            self.baseline = np.zeros(len(hdul))
            self.data = hdul
        '''

        self.n = -1  # masking variable. set to -1 so we know that masking hasn't been done yet. after masking, this changes to the length of the list of the selected region.
        self.smoothed = False  # smoothing boolean. If a hanning or boxcar smooth hasn't been performed, this indicates that smoothing nee

        #hdul = fits.open(self.path)
        #fitsdata = hdul[1].data
        #entries = len(fitsdata)

        #self.freq = np.zeros(entries,'d')
        #self.vel = np.zeros(entries,'d')
        #self.spec = np.zeros(entries,'d')

        #for i in range(len(fitsdata)):
        #    self.vel[i] = fitsdata[i][0]
        #    self.freq[i] = fitsdata[i][1]
        #    self.spec[i] = fitsdata[i][2]
        #self.vel = np.array(fitsdata['VHELIO'],'d')
        #self.freq = np.array(fitsdata['FREQUENCY'],'d')
        #self.spec = np.array(fitsdata['FLUX'],'d')
        #self.n = -1  # masking variable. set to -1 so we know that masking hasn't been done yet. after masking, this changes to the length of the list of the selected region.
        #self.smoothed = False  # smoothing boolean. If a hanning or boxcar smooth hasn't been performed, this indicates that smoothing nee

    def __first_plot(self, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.cla()
        #if not self.smoothed:
        #    self.res = smooth2.smooth(self.spec)  # smooth the function (hanning) if not already done
        props = dict(boxstyle='round', facecolor='skyblue')
        self.ax.text(0.1, 1.05, "Baselining", transform=self.ax.transAxes, fontsize=14, bbox=props)
        self.ax.step(self.vel, self.res+self.baseline, linewidth=1,color='k')
        self.ax.plot(self.vel, self.baseline, linewidth=2, ls='--', color='r')
        self.ax.axhline(y=0, dashes=[5, 5])
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        title = self.filename[3:-5]  # get just the AGC number
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(title))

        self.fig.canvas.draw()

    def __plot(self, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.cla()
        #if not self.smoothed:
        #    self.res = smooth2.smooth(self.spec)  # smooth the function (hanning) if not already done
        props = dict(boxstyle='round', facecolor='skyblue')
        self.ax.text(0.1, 1.05, "Baselining", transform=self.ax.transAxes, fontsize=14, bbox=props)
        self.ax.step(self.vel, self.res, linewidth=1,color='k')
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
        self.n = len(X)
        self.m = np.array(self.m)
        self.specrms = np.sqrt(self.specrms / nrms)
        self.rms = np.std(self.spec[self.m])
        self.srms = np.std(self.smo[self.m])
        #print('Spectrum rms in selected regions is', f"{self.specrms:.2f}")
        print('Spectrum rms in selected regions is', f"{self.rms:.2f}")
        print('Smoothed spectrum rms in selected regions is', f"{self.srms:.2f}")

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
                # self.fig.canvas.mpl_connect('button_press_event', self.__maskregions_onclick)
            else:
                response = input()
        X = []
        self.m = []
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
        self.n = len(X)
        self.m = np.array(self.m)
        self.specrms = np.sqrt(self.specrms/nrms)
        self.rms = np.std(self.spec[self.m])
        self.srms = np.std(self.smo[self.m])
        #print('Spectrum rms in selected regions is', f"{self.specrms:.2f}")
        print('Spectrum rms in selected regions is', f"{self.rms:.2f}")
        print('Smoothed spectrum rms in selected regions is', f"{self.srms:.2f}")

        '''file_exists = os.path.exists('RMS.csv')
        if file_exists == False:
            file = open('RMS.csv', 'x')
            message_info = (
                    'AGCnr,Specrms' + '\n')
            file.write(message_info)

        file = open('RMS.csv', 'a')
        try: 
            message = (str(self.filename) + ',' +
                       str(self.specrms) + '\n'
                       )
            file.write(message)
        except:
            print('Error: Unable to write output to RMS.csv')
	'''

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
                    if len(self.res) != 0:
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
            if len(regions) == 2:
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

            # original code implementation for p-value criteria
            # coeff, cov = np.polyfit(vel, spec, deg=order, cov=True)  # (list)

            # Bic implementation for basline.py
            vel = self.vel[self.m]
            spec = self.spec[self.m]

            # fit the polynomial of order (order)
            fitted_series = np.polynomial.polynomial.Polynomial.fit(vel, spec, order).convert().coef

            # create and fit polynonial object
            p = np.polynomial.Polynomial(fitted_series)
            y_fit = p(self.vel)

            ## Alternative calculation for mse
            # mse = np.mean((self.spec - y_fit) ** 2) # this calculates the mean squared error
            # for the entire spectrum, not just the masked region.

            # Keeping the same rms calculation as before for rms
            res = self.spec - y_fit  # list of baseline-subtracted spectrum values.
                    # Again, note that this is not the class variable, self.res. Also comes later.
            rmsarr = np.asarray([res[i] for i in range(len(y_fit)) if self.m[i]])
            rms = np.std(rmsarr)  # (number)

            # calculate MSE for bic/aic values
            mse = np.mean([(res[i])**2 for i in range(len(y_fit)) if self.m[i]])
            # aic = 2 * (order + 1) + self.n * np.log(2 * np.pi) + self.n * np.log(mse) + self.n
            bic = (order + 1)*np.log(self.n) + self.n * np.log(2 * np.pi) + self.n * np.log(mse) + self.n
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
            # self.smooth()  # smoothing the function
            self.__mask()  # masking the function
            recommended, rmsval, bicval = self.calcpoly()  # calculating the recommended order of the function
            titles = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']  # titles for the orders
            print('Statistics for each fit order:')
            print(' order  rms(mJy)')
            for i in range(len(titles)):
                # if our order is recommended print *
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


            # Block of code that will shuffle through polynomial fits
            #if response == 's':
             #   for i in range(len(recommended)):
              #      self.rms, self.aic, self.yfit = self.fitpoly(recommended[i])
               #     self.res = (np.asarray(self.smo) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
                #    # Plot the fit for each recommended order with a label
                 #   self.ax.plot(self.vel, self.yfit, linestyle='--', color='black', linewidth=1, label=f'yfit for order {recommended[i]}')
                  #  time.sleep(5)  # Wait for 5 seconds before moving to the next plot

            accepted = False
            while not accepted:
                plt.title(f"Plot for order {order}")
                (self.rms, self.aic, self.yfit) = self.fitpoly(
                    order)  # receiving rms, p, and yfit from the fitpoly function, using previously recommended order
                self.res = (np.asarray(self.smo) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
                #self.res = (np.asarray(self.spec) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
                self.ax.plot(self.vel, self.yfit, linestyle='--', color='orange', linewidth='2', label='yfit')

                response = input()

                if response == '':
                    if noconfirm:
                        accepted = True
                    else:
                        self.__plot()
                        response = input('Press Enter again to confirm this baseline fit.'
                                         ' Type anything else and hit enter to try again.\n')
                        if response == '':
                            accepted = True
                        else:
                            self.res = np.asarray(self.smo)
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
                        self.ax.lines.remove(line)
                        # plot the given order then wait 5 seconds to remove it
                        (self.rms, self.aic, self.yfit) = self.fitpoly(
                            recommended[i])  # receiving rms, p, and yfit from the fitpoly function, using previously recommended order
                        self.res = (np.asarray(self.smo) - np.asarray(self.yfit))  # baseline subtracted spectrum (residual)
                        self.ax.plot(self.vel, self.yfit, linestyle='--', color='black', linewidth='1', label='yfit')
                        plt.title(f"Plot for order {recommended[i]}")
                        plt.pause(3)

                    # restore to the reccomened order
                    line = [line for line in self.ax.lines if line.get_label() == 'yfit'][0]
                    self.ax.lines.remove(line)
                    print("Shuffle has concluded.")
                    order = recommended[0]

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

