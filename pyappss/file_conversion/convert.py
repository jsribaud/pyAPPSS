from astropy.io import fits
import numpy as np
import os
import glob
from datetime import date as d
import argparse
import sys
import pathlib


# Parser and arguements, based upon code in reduce.py. Added to account for missing header information. This should be expandable to more than just AGBT 22A-430, with minimal effort.
def convert():
    parser = argparse.ArgumentParser(description="Convert .fits files to APPSS format")
    parser.add_argument('-flag', metavar='ProgramFlag', type=str,
                        help="Code related to specific program. See Readme.md for more information")
    args = parser.parse_args()
    flag = args.flag

    if flag != None:
        if flag == 'AGBT22A-430':
            observatory = 'Green Bank Observatory'
            instrument = 'GBT L-band Receiver'
            beam = '8.96'
        else:
            sys.exit('Please Use a Valid Flag.\nCurrently supported flags are:\nAGBT22A-430')
    else:
        observatory = ''
        beam = ''
        instrument = ''

    # Defining some directories to make the working space much, much cleaner.
    current_dir = pathlib.PurePath(os.getcwd() + '/')
    raw_dir = current_dir / 'raw/'
    process_dir = current_dir / 'processed/'
    os.makedirs(str(process_dir), exist_ok=True)

    # Define the speed of light and other variables that will not change, in kilometer space.
    c = 299792.458
    # Today is today. For tracking when this program was run, from inside the fits file!
    today = d.today()
    # Month abbreviation, day and year
    date = today.strftime("%b-%d-%Y")

    # Create the glob, and work within the globspace to effectively write files en masse.
    # This retrieves all files in the raw directory and
    image_list = glob.glob(str(raw_dir / '*.fits'), recursive=True)
    N_images = len(image_list)

    string_raw = str(pathlib.PurePath('raw'))
    if sys.platform.startswith('win32'):
        morelength = len(string_raw) + 1
    else:
        morelength = len(string_raw)
    for i in range(N_images):
        root_name = image_list[i][image_list[i].rfind(string_raw) + morelength:image_list[i].rfind('.fits')]
        outname = 'AGC' + root_name + '.fits'
        print(outname)
        hdul = fits.open(image_list[i])
        # We need the second entry in the fits, since there needs to be a dummy primary hdu in all fits files, even if not an image.
        hdr = hdul[1].header
        data = hdul[1].data
        # Defining a set of values/variables using values pulled from the fits data/header
        rest = data['RESTFREQ'][0]
        vhel = data['VELOCITY'][0] / 1000
        freq_topo = data['CRVAL1'][0]
        length = len(data['DATA'][0])
        freq_res = data['CDELT1'][0]
        center = data['CRPIX1'][0]
        freq_hel = rest / ((vhel / c) + 1)
        bw = data['BANDWID'][0]

        # There are some other values we would like to define here, but they are not defined in the initial GBT data
        ra = str(data['CRVAL2'][0])
        dec = str(data['CRVAL3'][0])
        source = str(data['OBJECT'][0])
        telescope = str(data['TELESCOP'][0])
        extname = hdr['EXTNAME']
        equinox = str(data['EQUINOX'][0])
        rest_freq = str(data['RESTFREQ'][0])
        bw_mhz = str(data['BANDWID'][0] / 1e6)
        chan = str(len(data['DATA'][0]))
        frontend = str(data['FRONTEND'][0])
        telescope = str(data['TELESCOP'][0])

        # The meat of the work: starting to define values we will need specifically for our array. Beginning with the heliocentric frequency start. The "center" point is at data point 4097, so there are 4096 steps of the channel resolution to reach the beginning. Similarly, when defining the maximum, there are 8191 steps, rather than 8192 steps, to reach the end. This is assuming a number of channels equal to 8192, the standard for our data, of course. But the procedure doesn't assume this
        hel_start = (freq_hel - ((center - 1) * freq_res)) - (freq_res / 3)
        hel_end = (freq_hel - ((center - 1) * freq_res)) + ((length - 1) * freq_res) + (freq_res / 3)
        # Note the 1/3rd of a channel length of fudge factor, shifting the max and min outwards. I am frankly unsure why this creates more accurate data, but this is effectively the most accurate method to the true values, which would be convenient if they could actually be pulled directly in a fits file...
        hel_freq = np.reshape(np.linspace((hel_start), (hel_end), num=length), (length, 1))
        # Next, just a simply conversion to MHz, to match what the APPSS format has!
        freq = hel_freq / 1e6
        # The data is reshaped because astropy fits is picky. The next bit just converts frequency to velocity. In velocity space.
        vel = np.reshape(((rest / hel_freq - 1) * c), (length, 1))
        # Finally, defining the data as the reshaped numpy array.
        flux = np.reshape((data['DATA'][0] * 1000), (length, 1))
        # Two additional arrays are created to match the APPSS data - the "baseline" array and the "weight" array. Because the baseline is created through this procedure anyways, and is added flat in reduce.py, it can just be an array of zeroes. The weight array should likely just be an array of ones - it is not called in reduce.py
        baseline = np.zeros((length, 1))
        weight = np.ones((length, 1))

        # Next, the data is defining as fits columns so it can be written out!
        freq_col = fits.Column(name='Frequency', format='E', array=freq, unit='MHz')
        vel_col = fits.Column(name='Velocity', format='E', array=vel, unit='km/s')
        flx_col = fits.Column(name='Flux', format='E', array=flux, unit='mJy')
        baseline_col = fits.Column(name='Baseline', format='E', array=baseline, unit='mJy')
        weight_col = fits.Column(name='Weight', format='E', array=weight)

        # The data now gets written into a fits binary table format!
        table_hdu = fits.BinTableHDU.from_columns([vel_col, freq_col, flx_col, baseline_col, weight_col])
        # Creating an empty primary header so the data will function. Additionally, writing in the header values we can!
        empty_primary = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([empty_primary, table_hdu])
        hdr = hdul[1].header
        hdr['EXTNAME'] = extname
        hdr['OBSERVAT'] = observatory
        hdr['TELESCOP'] = telescope
        # hdr['INSTRUME']=frontend
        # This was changed because the frontend does read the receiver, but only what vegas knows it as, which is inaccurate
        hdr['INSTRUME'] = instrument
        hdr['BEAM'] = beam
        hdr['OBJECT'] = source
        hdr['RA'] = ra
        hdr['DEC'] = dec
        hdr['EQUINOX'] = equinox
        hdr['RESTFREQ'] = rest_freq
        hdr['BW'] = bw_mhz
        hdr['CHAN'] = chan
        # we can in fact define the receiver - 'FRONTEND'
        hdul[1].header = hdr
        # Add a note about how the data was created!
        hdr[
            'NOTE01'] = 'This fits file was created using a translation program from GBT vegas data to the APPSS format, ' + date
        hdul.writeto(str(process_dir / (outname)), overwrite=True)

        hdul.close()

    print(image_list)


if __name__ == '__main__':
    convert()
