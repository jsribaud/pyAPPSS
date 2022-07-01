import argparse

from analysis import baseline
from analysis import measure


def main(agcs, smo=None, gauss=False, twopeak=False, trap=False, dark_mode=False, noconfirm=False):

    for agc in agcs:

        try:

            b = baseline.Baseline(agc, smooth_int=smo, noconfirm=noconfirm, dark_mode=dark_mode)
            vel, spec, rms = b()
            measure.Measure(smo=smo, gauss=gauss, twopeak=twopeak, trap=trap, dark_mode=dark_mode,
                            vel=vel, spec=spec, rms=rms, agc=agc, noconfirm=noconfirm)
        except IOError:
            print("Could not open AGC{}.fits\n".format(agc))


if __name__ == '__main__':
    # NOTE: you need to have the FITS file in the same directory as this script!
    parser = argparse.ArgumentParser(
        description="Baseline and Reduce HI Spectrum.")
    parser.add_argument('filename', metavar='AGC', nargs='+', type=str, help="AGC number of the galaxy, e.g, 104365. "
                                                                             "You may list multiple. e.g. 3232 104365 1993 ...")
    parser.add_argument('-smo', metavar='smooth', type=int,
                        help="Value for smoothing the spectrum, if nothing is passed Hanning smooth will occur by default;"
                             "\n 'X' for boxcar where X is a postive integer.")
    parser.add_argument('-gauss', action='store_true', help='Do a Gaussian fit of the spectrum')
    parser.add_argument('-twopeak', action='store_true', help='Do a Two Peak fit of the spectrum')
    parser.add_argument('-trap', action='store_true', help='Do a Trapezoidal fit of the spectrum')
    parser.add_argument('-dark_mode', action='store_true', help='Enable dark mode, but not recommended for publication.')
    parser.add_argument('-noconfirm', action='store_true', help='Additional option to remove confirmations, both for baseline selection and for fit model choice')

    args = parser.parse_args()
    main(agcs=args.filename, smo=args.smo, gauss=args.gauss, twopeak=args.twopeak, trap=args.trap,
         dark_mode=args.dark_mode, noconfirm=args.noconfirm)
