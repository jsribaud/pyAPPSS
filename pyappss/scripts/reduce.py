import argparse

from analysis import baseline
from analysis import measure


def main(agc, smo=None, gauss=False, twopeak=False, trap=False, light_mode=False):
    b = baseline.Baseline(agc)
    vel, spec, rms = b()
    measure.Measure(smo=smo, gauss=gauss, twopeak=twopeak, trap=trap, light_mode=light_mode,
                    vel=vel, spec=spec, rms=rms, agc=agc)


if __name__ == '__main__':
    # NOTE: you need to have the FITS file in the same directory as this script!
    parser = argparse.ArgumentParser(
        description="Baseline and Reduce HI Spectrum.")
    parser.add_argument('filename', metavar='AGC', type=int, help="AGC number of the galaxy, e.g, 104365")
    parser.add_argument('-smo', metavar='smooth', type=int,
                        help="Value for smoothing the spectrum, if nothing is passed Hanning smooth will occur by default;"
                             "\n 'X' for boxcar where X is a postive integer.")
    parser.add_argument('-gauss', action='store_true', help='Do a Gaussian fit of the spectrum')
    parser.add_argument('-twopeak', action='store_true', help='Do a Two Peak fit of the spectrum')
    parser.add_argument('-trap', action='store_true', help='Do a Trapezoidal fit of the spectrum')
    parser.add_argument('-pub', action='store_true', help='Plot in Light Mode which is good for publication')

    args = parser.parse_args()
    main(agc=args.filename, smo=args.smo, gauss=args.gauss, twopeak=args.twopeak, trap=args.trap,
         light_mode=args.pub)
