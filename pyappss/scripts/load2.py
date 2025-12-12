import numpy as np
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy import constants as c
from datetime import date as d

import os
import sys

def load(filename, gbtidl_fits=False, frame='helio'):
    # if masking hasn't occured, neither has baselining.
    # So for the purposes of the smooth function, the yfit array is just zeros
    if gbtidl_fits:
        thd = fits.open(filename)
        ftab = Table(thd[1].data)
        fhdr = thd[1].header
        ##Define various telescope/observation values (for header info)
        ra = str(ftab['CRVAL2'][0])
        dec = str(ftab['CRVAL3'][0])
        source = str(ftab['OBJECT'][0])
        telescope = str(ftab['TELESCOP'][0])
        extname = fhdr['EXTNAME']
        equinox = str(ftab['EQUINOX'][0])
        rest_freq = str(ftab['RESTFREQ'][0])
        bw_mhz = str(ftab['BANDWID'][0] / 1e6)
        chan = str(len(ftab['DATA'][0]))
        frontend = str(ftab['FRONTEND'][0])
        telescope = str(ftab['TELESCOP'][0])
        ##Assuming GBT
        observatory = 'Green Bank Observatory'
        instrument = 'GBT L-band Receiver'
        beam = '8.96'
        ##Determine frame corrections
        gbt = EarthLocation.from_geodetic(lat=ftab['SITELAT'][0] * u.deg, lon=ftab['SITELONG'][0] * u.deg,
                                          height=ftab['SITEELEV'][0] * u.m)
        sc = SkyCoord(ra=ftab['CRVAL2'][0] * u.deg, dec=ftab['CRVAL3'][0] * u.deg)
        otime = Time(ftab['DATE-OBS'][0])
        barycorr = sc.radial_velocity_correction(kind='barycentric', obstime=otime, location=gbt)
        v_bcorr = barycorr.to(u.km / u.s)
        heliocorr = sc.radial_velocity_correction(kind='heliocentric', obstime=otime, location=gbt)
        v_hcorr = heliocorr.to(u.km / u.s)
        vframe_bary = -1. * v_bcorr  # convert correction to frame for freq. calculation
        vframe_helio = -1. * v_hcorr  # convert correction to frame for freq. calculation

        ##Apply corrections
        dflux = np.array(ftab['DATA'][0])*u.Jy
        length = len(dflux)
        chans = np.arange(length) + 1.
        freq_topo = ((chans - ftab['CRPIX1'][0]) * ftab['CDELT1'][0] + ftab['CRVAL1'][0]) * u.Hz
        vel_topo = c.c.to(u.km / u.s) * (ftab['RESTFREQ'][0] * u.Hz / freq_topo - 1.0)
        freq_bary = freq_topo * np.sqrt((c.c.to(u.km / u.s) + vframe_bary) / (c.c.to(u.km / u.s) - vframe_bary))
        vel_bary = c.c.to(u.km / u.s) * (ftab['RESTFREQ'][0] * u.Hz / freq_bary - 1.0)
        freq_helio = freq_topo * np.sqrt((c.c.to(u.km / u.s) + vframe_helio) / (c.c.to(u.km / u.s) - vframe_helio))
        vel_helio = c.c.to(u.km / u.s) * (ftab['RESTFREQ'][0] * u.Hz / freq_helio - 1.0)

        #Set frame
        if frame=='topo':
            frame = 'TOPO'
            freq = freq_topo.to(u.MHz)
            vel = vel_topo
        elif frame=='bary':
            frame = 'BARY'
            freq = freq_bary.to(u.MHz)
            vel = vel_bary
        else:
            frame = 'HELIO'
            freq = freq_helio.to(u.MHz)
            vel = vel_helio
        flux = dflux.to(u.mJy)
        # set fits table columns
        freq_col = fits.Column(name='FREQUENCY', format='D', array=freq*u.MHz, unit='MHz')
        vel_col = fits.Column(name='VELOCITY', format='D', array=vel, unit='km/s')
        flx_col = fits.Column(name='FLUX', format='D', array=flux, unit='mJy')
        baseline_col = fits.Column(name='BASELINE', format='D', array=np.zeros(length), unit='mJy')
        weight_col = fits.Column(name='WEIGHT', format='D', array=np.ones(length))

        # The data now gets written into a fits binary table format!
        table_hdu = fits.BinTableHDU.from_columns([vel_col, freq_col, flx_col, baseline_col, weight_col])
        # Creating an empty primary header so the data will function. Additionally, writing in the header values we can!
        empty_primary = fits.PrimaryHDU(header=fhdr)
        hdul = fits.HDUList([empty_primary, table_hdu])
        thdr = hdul[1].header
        thdr['EXTNAME'] = extname
        thdr['OBSERVAT'] = observatory
        thdr['TELESCOP'] = telescope
        thdr['INSTRUME'] = instrument
        thdr['BEAM'] = beam
        thdr['OBJECT'] = source
        thdr['RA'] = ra
        thdr['DEC'] = dec
        thdr['EQUINOX'] = equinox
        thdr['RESTFREQ'] = rest_freq
        thdr['BW'] = bw_mhz
        thdr['CHAN'] = chan
        thdr['FRAME'] = frame
        # we can in fact define the receiver - 'FRONTEND'
        hdul[1].header = thdr
        # Add a note about how the data was created!
        today = d.today()
        date = today.strftime("%b-%d-%Y")
        thdr[
            'NOTE01'] = 'This fits file was created using pyHISD, ' + date
        #hdul.writeto(str(process_dir / (outname)), overwrite=True)
        tab = QTable(hdul[1].data)
        ##Add units to arrays
        tab['VELOCITY'] = tab['VELOCITY'] * u.km / u.s
        tab['FREQUENCY'] = tab['FREQUENCY'] * u.MHz
        tab['FLUX'] = tab['FLUX'] * u.mJy
        tab['BASELINE'] = tab['BASELINE'] * u.mJy
        header = thdr
        hdul.close()

        #tab_wframes = Table([chans, freq_bary, vel_bary, freq_helio, vel_helio, freq_topo, vel_topo, flux * u.Jy],
        #                     names=['Channel', 'FREQUENCY_bary', 'VELOCITY_bary', 'FREQUENCY_helio', 'VELOCITY_helio',
        #                            'FREQUENCY_topo', 'VELOCITY_topo', 'FLUX'])
        #tab = Table([freq_helio,vel_helio,flux*u.Jy,baseline,weight],
        #            names=['FREQUENCY','VELOCITY','FLUX','BASELINE','WEIGHT'])

    else:
        """
                Reads the FITS file and loads the data into the arrays.
                """
        hdu = fits.open(filename)
        tmptab = QTable(hdu[1].data)
        velname = 'VELOCITY'
        fluxname = 'FLUX'
        freqname = 'FREQUENCY'
        weightname = 'WEIGHT'
        baselinename = 'BASELINE'
        vel_list = ['VHELIO','Vhelio','vhelio','VELOCITY','Velocity','velocity','VEL','Vel','vel']
        freq_list = ['FREQUENCY','Frequency','frequency','FREQ','Freq','freq']
        flux_list = ['FLUX','Flux','flux','SPEC','Spec','spec','SPECTRA','Spectra','spectra']
        weight_list = ['WEIGHT','Weight','weight']
        baseline_list = ['BASELINE','Baseline','baseline','BASE','Base','base']
        vec = 0
        frc = 0
        flc = 0
        wec = 0
        bac = 0
        for ii in range(len(tmptab.colnames)):
            if tmptab.colnames[ii] in vel_list:
                velname = tmptab.colnames[ii]
                vec=3
            if tmptab.colnames[ii] in freq_list:
                freqname = tmptab.colnames[ii]
                frc=3
            if tmptab.colnames[ii] in flux_list:
                fluxname = tmptab.colnames[ii]
                flc=3
            if tmptab.colnames[ii] in weight_list:
                weightname = tmptab.colnames[ii]
                wec=1
            if tmptab.colnames[ii] in baseline_list:
                baselinename = tmptab.colnames[ii]
                bac=1
        tmpsort = np.argsort(tmptab[velname])  # checks for spectra format (increasing v or f)
        tab = tmptab[tmpsort]# forces spectra to increase in v
        header = hdu[1].header
        if vec==0:
            tab['VELOCITY']=np.zeros(len(tab))*u.km/u.s
            print('Unable to properly load velocity array.\n')
        if frc == 0:
            tab['FREQUENCY'] = np.zeros(len(tab)) * u.MHz
            print('Unable to properly load frequency array.\n')
        if flc == 0:
            tab['FLUX'] = np.ones(len(tab)) * u.mJy
            print('Unable to properly load flux array.\n')
        if bac==0:
            tab['BASELINE']=np.zeros(len(tab))*u.mJy
            print('Unable to properly load baseline array. Baseline set to 0 mJy.\n')
        if wec==0:
            tab['WEIGHT']=np.ones(len(tab))
            print('Unable to properly load weight array. Weight set to 1 for all elements of flux array.\n')
        # self.cnames = {'Velocity':velname,'Frequency':freqname,'Flux':fluxname,'Weight':weightname,'Baseline':baselinename}
        if vec+frc+flc+bac+wec<9:
            print('Unable to properly load data - table/column names unrecognized.\n')
            print('Recognized column names are:\n')
            print('Required - one of: Velocity, VELOCITY, velocity, Vhelio, VHELIO, vhelio, VEL, Vel, vel')
            print('Required - one of: Frequency, FREQUENCY, frequency, Freq, FREQ, freq')
            print('Required - one of: Flux, FLUX, flux, Spectra, SPECTRA, spectra, Spec, SPEC, spec')
            print('Optional - one of: Baseline, BASELINE, baseline, Base, BASE, base')
            print('Optional - one of: Weight, WEIGHT, weight\n')
            print('To load a spectrum directly from a gbtidl FITS file, set flag gbtidl_fits==True.\n')
            sys.exit()
        try:
            tab.rename_column(velname, 'VELOCITY')
            tab['VELOCITY']=tab['VELOCITY']*u.km/u.s
            tab.rename_column(freqname, 'FREQUENCY')
            tab['FREQUENCY'] = tab['FREQUENCY'] * u.MHz
            tab.rename_column(fluxname, 'FLUX')
            tab['FLUX'] = tab['FLUX'] * u.mJy
            tab.rename_column(baselinename, 'BASELINE')
            tab['BASELINE'] = tab['BASELINE'] * u.mJy
            tab.rename_column(weightname, 'WEIGHT')

        except:
            print('Unable to properly load fits table - column names unrecognized. Run gbtfits_load() on your GBT' +
                  'data to produce an appropriate file format. Or use the flag -gbt_fits to load a GBT .FITS file.')
            sys.exit()
    return tab, header