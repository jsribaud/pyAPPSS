import argparse
import os
import urllib.request

import webbrowser
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from astropy.convolution import convolve, Box1DKernel
from astropy.io import fits


class Plot():
    """
    Plot the spectrum of a galaxy who's data is stored in a FITS file. The name of the file should be the galaxy's AGC number.
    The script has to be in the same directory (as of now) as the FITS files.
    Parameters
    ----------
    filename : int
        AGC number of the galaxy, e.g, 104365s
    xmin : float
        Minumum x value on the plot
    xmax : float
        Maximum x value on the plot
    ymin : float
        Minumum y value on the plot
    ymax : float
        Maximum y value on the plot
    showimage : bool
        When true, the program will obtain the SDSS DR14 inverted image of the galaxy. Default is true.
    saveplot : bool
        When true, the program will save the plot as png image with the AGC number as the name of the file in the same directory the program is being run from.
        Default is false.
    smo : str
       Value for smoothing the spectrum, if nothing is passed smoothing will not occur; 'h' for hanning smoothing, 'bX' for boxcar where X is a postive integer.
        e.g, 'smo' = 'b7'
    h : bool
        When true, prints the doc string of this class. Default is false.
    """

    def __init__(self, filename, filetwo=None, xmin=None, xmax=None, ymin=None, ymax=None, file2impose=False,
                 showimage=True, ned=None,
                 saveplot=False,
                 smo=None,
                 h=False):

        self.filename = 'A{:06}.fits'.format(filename)
        self.run = False  # boolean to check whether or not to run smoothing operation. If true, yes.

        if filetwo is not None:
            self.file_two = 'A{:06}.fits'.format(filetwo)
        else:
            self.file_two = None

        if h:
            print(self.__doc__)

        if smo is not None:
            self.run = True
            if smo == 'h':
                self.smotype = 'h'
            else:
                self.smotype = smo[1:]  # Just sends the value for boxcar smoothing

        if showimage:
            self.displayimage(self.filename)

        if ned is not None:
            self.open_ned(ned)

        if file2impose and (self.file_two is not None):
            self.plot(xmin, xmax, ymin, ymax, saveplot)
        else:
            self.impose_plots(xmin, xmax, ymin, ymax, saveplot)

    # Filename can be accessed by calling this method.
    def name(self):
        return self.filename

    # Smoothtype can be accessed by calling this method
    def smo_type(self):
        return self.smotype

    # Boolean value whether to smooth the plot or not
    def run_smo(self):
        return self.run

    def displayimage(self, filename):
        hdr = self.getheader(self.readdata(filename)[1])
        url = f'http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={hdr[18]}&dec={hdr[19]}&scale=0.2&width=500&height=500&opt=I'
        imgfilename = 'grab.jpg'
        urllib.request.urlretrieve(url, imgfilename)

        img = Image.open('grab.jpg')
        img.show()  # does not work in a jupyter notebook
        os.remove('grab.jpg')
        del img

    def open_ned(self, ned):
        hdr = self.getheader(self.readdata(self.filename)[1])
        url = "http://ned.ipac.caltech.edu/cgi-bin/objsearch?in_csys=Equatorial&in_equinox=J2000.0&lon={ra}d&lat={dec}d&radius={arc}&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&search_type=Near+Position+Search&z_constraint=Unconstrained&z_value1=&z_value2=&z_unit=z&ot_include=ANY&nmp_op=ANY&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=Distance+to+search+center&of=pre_text&zv_breaker=30000.0&list_limit=5&img_stamp=YES".format(
            ra=hdr[18], dec=hdr[19], arc=ned)
        webbrowser.open(url, new=0, autoraise=True)

    # Read in the data.
    # The 1 is because SDFits uses an extension on the normal data (which is in [0] and empty)
    def readdata(self, filename):
        hdul = fits.open(filename)
        fitsdata = hdul[1].data
        return fitsdata, hdul  # returns a tuple where the index=0 contains the fitsdata and the index=1 contains the hdul

    # the method will return a tuple which will contain each numpy array
    def fillarrays(self, fitsdata):  # make instance variable!!!

        entries = len(fitsdata)

        vel = np.zeros(entries)
        freq = np.zeros(entries)
        spec = np.zeros(entries)
        base = np.zeros(entries)
        weight = np.zeros(entries)

        # Fill the arrays
        for i in range(len(fitsdata)):
            vel[i] = fitsdata[i][0]
            freq[i] = fitsdata[i][1]
            spec[i] = fitsdata[i][2]
            base[i] = fitsdata[i][3]
            weight[i] = fitsdata[i][4]

        return vel, freq, spec, base, weight

    def smooth(self, filename):

        data = self.fillarrays(self.readdata(filename)[0])
        vel = data[0]
        freq = data[1]
        spec = data[2]
        base = data[3]
        weight = data[4]

        if (self.smo_type() == 'h'):  # Hanning Smoothing
            smoothed_signal = np.convolve(spec, [0.25, 0.5, 0.25], mode='same')
        else:  # Boxcar Smoothing
            mag = int(self.smo_type())
            box_kernel = Box1DKernel(mag)
            smoothed_signal = convolve(spec, box_kernel)
        return smoothed_signal

    def plot(self, xmin, xmax, ymin, ymax, saveplot):

        if self.file_two is not None:
            data1 = self.fillarrays(self.readdata(self.filename)[0])
            vel1 = data1[0]
            freq1 = data1[1]
            spec1 = data1[2]
            base1 = data1[3]
            weight1 = data1[4]

            signal1 = spec1

            if self.run is True:
                signal1 = self.smooth(self.filename)

            fig, ax = plt.subplots(2)
            ax[0].plot(vel1, signal1, color='black', linewidth=1)
            ax[0].axhline(y=0, color='grey', dashes=[5, 5], linewidth=1)
            ax[0].set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(self.name()[1:-5]))
            ax[0].set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax[0].grid(False)

            data2 = self.fillarrays(self.readdata(self.file_two)[0])
            vel2 = data2[0]
            freq2 = data2[1]
            spec2 = data2[2]
            base2 = data2[3]
            weight1 = data2[4]

            signal2 = spec2

            if self.run is True:
                signal2 = self.smooth(self.file_two)

            ax[1].plot(vel2, signal2, color='black', linewidth=1)
            ax[1].axhline(y=0, color='grey', dashes=[5, 5], linewidth=1)
            ax[1].set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(self.file_two[1:-5]))
            ax[1].set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax[1].grid(False)

            fig.tight_layout()
            plt.show(block=True)
        else:
            data1 = self.fillarrays(self.readdata(self.filename)[0])
            vel1 = data1[0]
            freq1 = data1[1]
            spec1 = data1[2]
            base1 = data1[3]
            weight1 = data1[4]

            signal1 = spec1

            if self.run is True:
                signal1 = self.smooth(self.filename)

            fig, ax = plt.subplots()
            ax.plot(vel1, signal1, color='black', linewidth=1)
            ax.axhline(y=0, color='grey', dashes=[5, 5], linewidth=1)
            ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(self.name()[1:-5]))
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.grid(False)

            plt.show(block=True)

        if saveplot is True:
            fig.savefig('{}_plot.png'.format(self.name()[:-5]))
        # plt.pause(3)
        plt.close()

    def impose_plots(self, xmin, xmax, ymin, ymax, saveplot):
        data1 = self.fillarrays(self.readdata(self.filename)[0])
        vel1 = data1[0]
        freq1 = data1[1]
        spec1 = data1[2]
        base1 = data1[3]
        weight1 = data1[4]

        signal1 = spec1
        if self.run is True:
            signal1 = self.smooth(self.filename)
        fig, ax1 = plt.subplots()
        ax1.plot(vel1, signal1, color='black', linewidth=1)
        ax1.axhline(y=0, color='grey', dashes=[5, 5], linewidth=1)
        ax1.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)", title='AGC {}'.format(self.name()[1:-5]))
        ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax1.grid(False)

        data2 = self.fillarrays(self.readdata(self.file_two)[0])
        vel2 = data2[0]
        freq2 = data2[1]
        spec2 = data2[2]
        base2 = data2[3]
        weight1 = data2[4]

        signal2 = spec2
        if self.run is True:
            signal2 = self.smooth(self.file_two)
        offset = 0

        ax2 = ax1.twinx()
        ax2.plot(vel2, signal2 + offset, linewidth=1)
        # ax2.axhline(y=offset, color='grey', dashes=[5, 5], linewidth=1)
        ax2.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy) \n AGC {}".format(self.file_two[1:-5]))
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax2.grid(False)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    # Returns  the FITS header
    def getheader(self, hdul):
        hdr = hdul[1].header
        return hdr  # returning hdr[14] returns just the AGC number but does it like 'A 2532' instead of 'A002532'

    def __str__(self):
        return str(self.getheader(self.readdata(self.filename)[1]))

    def __repr__(self):
        return str(self)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot the spectrum of a galaxy who's data is stored in a FITS file.\nThe name of the file should be the galaxy's AGC number.\nThe script has to be in the same directory (as of now) as the FITS files.")

    parser.add_argument('filename', metavar='AGC', type=int, help="AGC number of the galaxy, e.g, '104365'")

    parser.add_argument('-ftwo', metavar='Second File', type=int, help='Optional second galaxy to plot', default=None)

    parser.add_argument('-f2impose', metavar='Second File', type=str2bool, nargs='?', const=False,
                        help='Optional second galaxy to superimpose on plot', default=True)

    parser.add_argument('-xmin', metavar='Xmin', type=float,
                        help='Minumum x value on the plot', default=None)

    parser.add_argument('-xmax', metavar='Xmax', type=float,
                        help='Maximum x value on the plot', default=None)

    parser.add_argument('-ymin', metavar='Ymin', type=float,
                        help='Minumum y value on the plot', default=None)

    parser.add_argument('-ymax', metavar='Ymax', type=float,
                        help='Maximum y value on the plot', default=None)

    parser.add_argument('-ned', metavar='NED', type=float,
                        help='Search NED database within given radius (in arcmin)', default=None)

    parser.add_argument('-smo', metavar='smooth', type=str,
                        help="Value for smoothing the spectrum, if nothing is passed smoothing will not occur; 'h' for hanning smoothing, 'bX' for boxcar where X is a postive integer.")

    parser.add_argument('-showimage', metavar='Galaxy image', nargs='?', type=str2bool, const=True,
                        help='When true, the program will obtain the SDSS DR14 inverted image of the galaxy. Default is True.',
                        default=True)

    parser.add_argument('-saveplot', metavar='Save the Plot', nargs='?', type=str2bool, const=False,
                        help='When true, the program will save the plot as png image with the AGC number as the name of the file in the same directory the program is being run from. Default is False.',
                        default=False)

    #     parser.add_argument('-l', metavar = 'X Range on plot', nargs = 2, type = float, required = False, default = [None, None])

    args = parser.parse_args()
    #     print(args.l)
    Plot(args.filename, args.ftwo, args.xmin, args.xmax, args.ymin, args.ymax, file2impose=args.f2impose, smo=args.smo,
         showimage=args.showimage,
         ned=args.ned, saveplot=args.saveplot)
