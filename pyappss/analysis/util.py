import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pyappss.analysis import smooth


class Util(object):

    # parent class to hold common methods/attributes between Measure, Baseline, and Fit
    def __init__(self, agc=None, smo=None, path="", dark_mode=False):
        self.agc = agc
        self.filename = 'AGC{}.fits'.format(self.agc)
        self.path = pathlib.PurePath(path + "/" + self.filename)
        self.smoothed = False
        self.n = None
        self.vel = []
        self.spec = []
        self.freq = []
        self.yfit = []
        self.res = []
        self.smo = None  # = smo??
        self.rms = 0

        if dark_mode:
            plt.style.use('dark_background')
        plt.ion()
        plt.rcParams["figure.figsize"] = (10, 6)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.cid = None

        # to be set by children
        self.currstep = ""  # location in reduction e.g. Baselining, Measuring, ...
        self.stepcolor = ""  # color of bubble

    def load(self):
        """
        Reads the FITS file and loads the data into the arrays.
        """
        hdul = fits.open(self.path)
        fitsdata = hdul[1].data
        entries = len(fitsdata)

        self.freq = np.zeros(entries)
        self.vel = np.zeros(entries)
        self.spec = np.zeros(entries)

        for i in range(len(fitsdata)):
            self.vel[i] = fitsdata[i][0]
            self.freq[i] = fitsdata[i][1]
            self.spec[i] = fitsdata[i][2]

            # masking variable. set to -1 so we know that masking hasn't been done yet.
                # after masking, this changes to the length of the list of the selected region.
            self.n = -1

            # smoothing boolean. If a hanning or boxcar smooth hasn't been performed,
                # this indicates that smoothing needs to occur before showing the spectrum.
            self.smoothed = False

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
            self.res = smooth.smooth(self.spec)  # smooth the function (hanning) if not already done

        # textbox to tell the user what step they are on
        props = dict(boxstyle='round', facecolor=self.stepcolor)
        self.ax.text(0.1, 1.05, self.currstep, transform=self.ax.transAxes, fontsize=14, bbox=props)

        self.ax.plot(self.vel, self.res, linewidth=1)
        self.ax.axhline(y=0, dashes=[5, 5])
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(self.agc))

        self.fig.canvas.draw()

    def mark_regions(self, first_region=True):
        """
        Method to interactively select regions on the spectrum
        :return: v, the velocity values in the region
                 s, the spec values in the region
        """

        global mark_regions
        global regions
        regions = []

        mark_regions = self.fig.canvas.mpl_connect('button_press_event', self.mark_region_onclick)

        if first_region:
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
                    self.fig.canvas.mpl_disconnect(mark_regions)
            elif response == 'clear':
                del regions
                regions = []
                # regions.clear()
                self.plot()
                mark_regions = self.fig.canvas.mpl_connect('button_press_event', self.mark_region_onclick)
                response = input('Region cleared! Select a new region now. Press Enter if the region is OK, '
                                 'or type "clear" and press Enter to clear region selection.\n')
            else:
                response = input(
                    'Please press Enter if the region is OK, or type "clear" and press enter to clear region selection.\n')

        regions.sort()
        v = list()
        s = list()
        # constructing v and s lists if they are within the selected region.
        for i in range(len(self.vel)):
            for j in range(len(regions) - 1):
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

    def mark_region_onclick(self, event):
        # ensure the user has marked a point within the plot/their view
        if event.inaxes:
            ix, iy = event.xdata, event.ydata
            self.ax.plot([ix, ix], [-1e4, 1e4], linestyle='--', linewidth=0.7, color='green')
            regions.append(ix)
