import argparse

from analysis import baseline
from analysis import measure
from analysis import multigauss as multigaussfit

def main(agc, smo=None, gauss=False, twopeak=False, trap=False, multigauss=False, light_mode=False, noconfirm=False, overlay=False):
    if multigauss == True:
        if smo is None:
            smo = 21
        elif smo > 21:
            smo = 21
            #Multigauss is untested with no smoothing, and probably will not work with a light boxcar or no boxcar. 21 is the tried and true
            #value.
    
    b = baseline.Baseline(agc, smooth_int=smo, noconfirm=noconfirm)
    vel, spec, rms = b()
    if gauss or twopeak or trap:
        measure.Measure(smo=smo, gauss=gauss, twopeak=twopeak, trap=trap, light_mode=light_mode,
                    vel=vel, spec=spec, rms=rms, agc=agc, noconfirm=noconfirm, overlay=overlay)
    if multigauss == True:
        multigaussfit.ManyGauss(vel=vel, spec=spec, rms=rms, agc=agc)


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
    #New routine!
    parser.add_argument('-multigauss', action='store_true', help='Do a Many Gaussian fit of the spectrum, calculated near-automatically. As a note, this does take noticeably longer than other methods, typically')
    parser.add_argument('-pub', action='store_true', help='Plot in Light Mode which is good for publication')
    parser.add_argument('-noconfirm', action='store_true', help='Additional option to remove confirmations, both for baseline selection and for fit model choice')
    parser.add_argument('-overlay', action='store_true', help='Additional option to display a helpful filter overlay when plotting during main measure routine')

    
    args = parser.parse_args()
    main(agc=args.filename, smo=args.smo, gauss=args.gauss, twopeak=args.twopeak, trap=args.trap, multigauss=args.multigauss,
         light_mode=args.pub, noconfirm=args.noconfirm, overlay=args.overlay)
