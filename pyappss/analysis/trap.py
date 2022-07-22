from astropy.io import fits
import argparse
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.modeling import models
from astropy.modeling.models import custom_model
from astropy.modeling import fitting

from pyappss.analysis.fit import Fit


class Trap(Fit):

    def __init__(self, agc=None, boxcar=None, path="", dark_mode=False, vel=None, spec=None, rms=None):

        super().__init__(agc, boxcar, path, dark_mode, vel, spec, rms)
        self.currstep = 'trap'

        self.trap()
        self.write_file(self.get_comments())
        input('Trapezoidal fit complete! Press Enter to end.\n')
        plt.close()

    def trap(self, first_region=True):
        """
        Method to fit a trapezoidal fit
        :return:
        """
        self.plot()
        print("Please select the region for a trapezoidal fit.")
        v, s = self.mark_regions(first_region)

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

        self.print_values()

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