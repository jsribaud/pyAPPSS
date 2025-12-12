import argparse

#from pyappss.analysis import baseline
import sys

#from pyappss.analysis import measure
#from pyappss.analysis import multigauss
import measure2 as measure
import baseline2 as baseline

import os

def reduce():
    parser = argparse.ArgumentParser(description="Baseline and Reduce HI Spectrum.")
    parser.add_argument('filename', metavar='AGC', nargs='+', type=str,
                        help="AGC number of the galaxy, e.g, 104365. You may list multiple. e.g. 3232 104365 1993 .... OR you can enter the full filename, including the .fits extension.")
    parser.add_argument('--smo', metavar='smooth', type=int, default=1,
                        help="Value for smoothing the spectrum, if nothing is passed Hanning smooth will occur by default;"
                             "\n 'X' for boxcar where X is a postive integer.")
    parser.add_argument('--no_smooth', metavar='no_smooth', type=bool, default=False,
                        help='No smoothing done on data - for final version data, i.e. ALFALFA. Default is no_smooth = False')
    parser.add_argument('--gauss', action='store_true', help='Do a Gaussian fit of the spectrum')
    parser.add_argument('--twopeak', action='store_true', help='Do a Two Peak fit of the spectrum')
    parser.add_argument('--trap', action='store_true', help='Do a Trapezoidal fit of the spectrum')
    parser.add_argument('-p', '--path', metavar='path', type=str, nargs="?", default=".",
                        help='Specify the location containing the files you want to work on')
    parser.add_argument('--dark_mode', action='store_true',
                        help='Enable dark mode, but not recommended for publication.')
    parser.add_argument('--noconfirm', action='store_true',
                        help='Additional option to remove confirmations, both for baseline selection and for fit model choice')

    # new routine!
    parser.add_argument('--multigauss', action='store_true',
                        help='Do a Many Gaussian fit of the spectrum, calculated near-automatically. As a note, this does take noticeably longer than other methods, typically.')
    parser.add_argument('--overlay', action='store_true', help='Additional option to display a helpful filter overlay when plotting during main measure routine')
    args = parser.parse_args()

    agcs = args.filename
    smo = args.smo
    gauss = args.gauss
    twopeak = args.twopeak
    trap = args.trap
    path = args.path
    dark_mode = args.dark_mode
    noconfirm = args.noconfirm
    mgauss = args.multigauss
    overlay = args.overlay
    nosmooth = args.no_smooth
    
    # check to see if agc

    for agc in agcs:
        #print(agc)
        try:
            # check if agc contains '.fits'.  If yes, assume this is a filename

            if '.fits' in agc:
                filename = agc

            else:
                filename = f"AGC{agc}.fits"
                # assume it's AGC number

            # check if filename exists
            #if os.path.exists(filename):
            #    print(f'file {filename} does exist')

            if not mgauss:
                b = baseline.Baseline(filename, smooth_int=smo, path=path, noconfirm=noconfirm, dark_mode=dark_mode,
                                      no_smooth=nosmooth)
                tab, header, stab, xrms, rms = b()
                response = input('Is there an emission profile to measure?\n'
                      'Enter "y" to start the measuring procedure or hit "Return" to exit the program. \n')
                #response = input()
                if response != 'y':
                    print('No profile to measure - writing out rms information to file...')
                    measure.Measure(smo=smo, gauss=gauss, twopeak=twopeak, trap=trap, path=path, dark_mode=dark_mode,
                                xrms=xrms, rms=rms, tab=tab, header=header, agc=agc,
                                    noconfirm=noconfirm, overlay=overlay, stab=stab, detection=False)
                measure.Measure(smo=smo, gauss=gauss, twopeak=twopeak, trap=trap, path=path, dark_mode=dark_mode,
                                xrms=xrms, rms=rms, tab=tab, header=header, agc=agc,
                                noconfirm=noconfirm, stab=stab, overlay=overlay)

            if mgauss:
                if smo is None:
                    smo = 21
                elif smo > 21:
                    smo = 21

                print("You have selected to use the multigauss fit. Please note this will automatically smooth the data"
                      "with a boxcar of 21.\n")
                b = baseline.Baseline(agc, smooth_int=smo, path=path, noconfirm=noconfirm, dark_mode=dark_mode)
                vel, spec, rms = b()
                print('Continue to profile measurement?\n'
                      'Type y or n')
                response = input()
                if response != 'y':
                    sys.exit()
                multigauss.ManyGauss(smo=smo, vel=vel, spec=spec, rms=rms, agc=agc, path=path)

        except IOError:
            if path == ".":
                p = ""
            else:
                p = path + "/"
            print(f"Could not open {filename}. Please check the filename.\n")


if __name__ == '__main__':
    reduce()

