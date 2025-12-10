import numpy as np
from astropy.convolution import convolve, Box1DKernel, Trapezoid1DKernel
from astropy.table import Table

def smoothold(spec, smooth_type=None, yfit= None):
    # if masking hasn't occured, neither has baselining.
    # So for the purposes of the smooth function, the yfit array is just zeros
    if yfit is None:
        yfit = np.zeros(len(spec))
    smo = []
    for i in range(len(spec)):
        smo.append(spec[i] - yfit[i])
        # Hanning smooth
    window = [.25, .5, .25]
    smo = np.convolve(smo, window, mode='same')
    if smooth_type is not None:  # Boxcar smooth
        window = []
        if smooth_type % 2 == 1:  # if the user selected an even int for boxcar smooth, make it odd by adding 1.
            smooth_type += 1
        smo = convolve(smo, Box1DKernel(smooth_type))
        # for i in range(int(smooth_type)):  # range
        #     window.append(1 / float(smooth_type))
        # smo = np.convolve(smo, window, 'same')
    res = smo  # allows the plot to reflect the smoothing.
    return res

def smooth(tab, box_width=1, yfit=None, no_smooth=False):
    #boxcar smooth by 7 (or whatever box_width is set - should be 7 for our GBT22A-430 data)
    #print(box_width)
    if no_smooth:
        fvel=tab['VELOCITY']
        fflux=tab['FLUX']
        ffreq=tab['FREQUENCY']
        fbase = tab['BASELINE']
        ftab=Table([ffreq,fvel,fflux,fbase],names=['FREQUENCY','VELOCITY','FLUX','BASELINE'])
        smflux2=fflux
        smooth_note = 'No smoothing applied to data.'
        hflag = 'NO'
        sflag = 'NO'
    else:
        if box_width > 1:
            box1d = Box1DKernel(width=box_width)
            smflux1 = convolve(tab['FLUX'],box1d.array)
            #decimate by x7 -> res. element ~20 kHz, 4.2 km/s
            tfreq = tab['FREQUENCY'][::box_width]
            #
            tvel = tab['VELOCITY'][::box_width]
            #
            tflux = smflux1[::box_width]
            #
            tbase = tab['BASELINE'][::box_width]
        else:
            smflux1 = tab['FLUX']
            tfreq = tab['FREQUENCY']
            tvel = tab['VELOCITY']
            tbase = tab['BASELINE']
            tflux = smflux1
        #hanning smooth
        hann1d = Trapezoid1DKernel(width=1)
        smflux2 = convolve(smflux1,hann1d.array)
        hanflux = convolve(tflux,hann1d.array)
        #decimate by x2 -> res. element ~40 kHz, 8.5 km/s
        ffreq = tfreq[::2]
        fvel = tvel[::2]
        fflux = hanflux[::2]
        fbase = tbase[::2]
        ftab = Table([ffreq,fvel,fflux,fbase],names=['FREQUENCY','VELOCITY','FLUX','BASELINE'])
    delf = str(ffreq[0]-ffreq[1])[:6]+' MHz'
    delvh = fvel[1]-fvel[0]
    delvl = fvel[-1]-fvel[-2]
    delv = str((delvh+delvl)/2)[:4]+' km/s'
    delvxh = tab['VELOCITY'][1] - tab['VELOCITY'][0]
    delvxl = tab['VELOCITY'][-1] - tab['VELOCITY'][-2]
    delvx = str((delvxh + delvxl) / 2)[:4] + ' km/s'
    print('\n')
    print('Velocity resolution for Original Data is: ', delvx)
    print('Velocity resolution for Smoothed Data is: ',delv,'\n')
    if box_width > 1:
        smooth_note = 'Boxcar smooth and decimated by '+str(box_width)+' followed by Hanning smooth and decimated by 2'
        hflag = 'YES'
        sflag = 'YES'
    else:
        smooth_note = 'Hanning smooth and decimated by 2'
        hflag = 'YES'
        sflag = 'NO'
    sm_width = str(box_width)
    smooth_info = [delf, delv, sflag, hflag, sm_width, smooth_note]
    return ftab, smflux2