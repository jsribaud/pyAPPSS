## This file includes the most current version of the COG width-finder, as well as all helper functions necessary to run this width finder.

import numpy as np, scipy.optimize as sco, matplotlib.pyplot as plt
from astropy.io import ascii, fits
import pandas as pd

##-------------------------------------------------- load in profiles --------------------------------------------------##
def getProfVals(agcn, key, adir):
    #this program returns the profile values for a given AGC number.
    #need to make sure you have all necessary leading zeros:
    agcnstr = str(agcn)
    numz = 6 - len(agcnstr)
    for i in range(numz): 
        agcnstr = '0'+agcnstr
    ## if you have the full ALFALFA spectraFITS folder, you can change string to include the full path to that file.    
    string = adir+'/A'+agcnstr+'.fits'
    currfits = fits.open(string)
    data = currfits[1].data
    vals = data[key]
    return vals

def displayprof(agcn, datatab, kws = ['AGCNr', 'Vhelio', 'W20', 'W50', 'RMS', 'SNR'], lightweight = False, alfdir = '../alfalfa_data/spectraFITS'):
    #displays the line in the dataset that corresponds to index in the data table...
    # lightweight allows for minimum necessary similarities between ALFALFA formatting (AGCNr and Vhelio columns in datatab) to make the program work
    index = np.where(datatab[kws[0]] == agcn)
    flux = getProfVals(agcn,'FLUX', adir = alfdir)
    vels = getProfVals(agcn, 'VHELIO', adir = alfdir)
    vhel = datatab[kws[1]][index]
    if not lightweight:
        w20 = datatab[kws[2]][index]
        w50 = datatab[kws[3]][index]
        rms = 0.001*datatab[kws[4]][index][0]
        snr = datatab[kws[5]][index][0]
    
        vlow = vhel - np.maximum(500., w20)
        vhigh = vhel + np.maximum(500., w20)
    else:
        vlow = vhel - 500.
        vhigh = vhel + 500.
    
    indhigh = np.argmin(np.abs(vels - vlow))
    indlow = np.argmin(np.abs(vels - vhigh))
    if lightweight:
        retvals = flux, vels, indlow, indhigh, vhel
    else:
        retvals = flux,vels,indlow,indhigh, rms, vhel, snr, w50
    return retvals

##----------------------------------- HELPER FUNCTIONS FOR CCF V_helio-finding ------------------------------------------##
def sigma_from_dist(peakdist):
    #this just puts everything into the one function that you run. peakdist can be in indices, we'll convert to km/s
    a0oa2 = 1.75    
    coeff = 1./np.sqrt(5./2.-a0oa2/np.sqrt(2.))
    if peakdist < 60.0 and peakdist > 6.0:
        pdcorr = (19.3+0.66*peakdist)
        sigma = pdcorr/2.0*coeff
    elif peakdist <= 6.0:
        #these ones are just going to be gaussians.
        sigma = peakdist
    elif peakdist >= 60.0:
        sigma = (peakdist/2.0)*coeff
    return sigma

def binProfFac(fullprofile, binfactor):
    #this creates lower-resolution versions of profiles by combining the flux from the binfactor adjacent bins.
    binnedprof = np.zeros(int(len(fullprofile)/binfactor))
    for i in range(len(binnedprof)):
        for j in range(binfactor):
            binnedprof[i]+=fullprofile[binfactor*i + j]
    return binnedprof

def findZeroes(array):
    #this finds the indices that bracket zeroes in an array with discrete values - it accounts both for zero values and also for places where the gradient sign changes from one position to the next, which means that the zero is between the two indices
    zeroes = []
    for i in range(len(array)-1):
        currval = array[i]
        nextval = array[i+1]
        if currval == 0.0 and i not in zeroes:
            zeroes.append(i)
        elif nextval == 0.0 and i+1 not in zeroes:
            zeroes.append(i+1)
        else:
            currsign = currval / np.abs(currval)
            nextsign = nextval / np.abs(nextval)
            if not currsign == nextsign:
                #this tells you there's a sign change between the indices. approximate using the closer index.
                if np.abs(currval) < np.abs(nextval) and i not in zeroes:
                    zeroes.append(i)
                elif i+1 not in zeroes:
                    zeroes.append(i+1)
    return zeroes

def findhigh(array,maxzeroes):
    if len(array) == 0:
        maxval = 0.0
        maxind = 512
    else:
        maxval = np.nanmax(array)
        maxind = maxzeroes[np.argmax(array)]
    return maxval, maxind
##----------------------------------------- THE FUNCTION THAT FINDS THE CORRELATIONS --------------------------------------## 

def correlationFinder(profile, indlow, indhigh, templates):
    '''provides an initial estimate at the width of the profile and the center of the profile using cross-correlation functions between Hermite polynomials and the profile. Runs through Hermite polynomials that have peaks at distances that are equal to the distance between peaks in the profile, in order to lower the number of CCFs that must be computed.'''
    truncprof = profile[indlow:indhigh]
    #this section of the function finds the distances between peaks by finding zeroes in the gradient of the binned profile - the profile is binned to make these peaks less sensitive to spurious noise peaks, and we use the zeroes of the gradient so that we are selecting local maxima
    binnedprofile = binProfFac(truncprof,4)
    lquad = int(0.2*float(len(binnedprofile)))
    rquad = int(0.85*float(len(binnedprofile)))
    peakfinder = binnedprofile[lquad:rquad]
    if len(peakfinder) <= 5:
        peakfinder = binnedprofile
    grad = np.gradient(peakfinder)
    zpf = findZeroes(grad) 
    #plt.plot(peakfinder)
    #plt.plot(grad)
    #plt.show()
    zeroes = [4*(p + lquad)+indlow for p in zpf]
    locmaxes = [z-4+np.argmax(profile[z-4:z+8]) for z in zeroes]
    distances = np.array([[np.abs(locmaxes[i] - locmaxes[j]) for j in range(len(locmaxes))] for i in range(len(locmaxes))])
    
    flatdists = distances.flatten()

    sing, freq = np.unique(flatdists, return_counts=True)
    flatdists = np.trim_zeros(sing)
    #flatdists contains the distances between peaks that we will use to select profile templates for use in our CCF
    flatdists = [f for f in flatdists if f > 2 and f <= len(templates)]
    if len(flatdists) == 0:
        flatdists = [2,4,6]
    #we compute the profile fft once for use in all of our CCFs (which we compute using fft of profile * fft of template)
    proffft = np.fft.fft(profile)
    profacf = np.fft.ifft(proffft*np.conj(proffft)).real
    N = 1024.
    gsquared = [p**2.0 for p in profile]
    sigg = np.sqrt(1./N*np.sum(gsquared))
    
    #GIVEN SOME LIST OF **ROLLED** TEMPLATE FFTS, WHERE THE TEMPLATE INDEX CORRESPONDS TO THE DIST PARAMETER.
    #we have tbankfile so we can load in the same file of templates every time.
    #oops... The profiles in templatebank are 1 indexed (i.e. the 0th entry has a distance of 1 points between entries)
    selectedtemps = [templates[d-1] for d in flatdists]
    #print('template length: ',len(selectedtemps[0]),' prof length: ',len(proffft))
    crosscorrs = [np.fft.ifft(proffft*tempfft) for tempfft in selectedtemps]
    #finds the peak of each ccf (maxvalues), as well as the position of that maximum within the profile (maxinds)
    maxfinder = [np.gradient(ccorr[indlow:indhigh]) for ccorr in crosscorrs]
    maxzeroes = [[z for z in findZeroes(maxfinderforprof)] for maxfinderforprof in maxfinder]
    localextrvals = [[crosscorrs[i][z+indlow] for z in maxzeroes[i]] for i in range(len(maxzeroes))]
    maxzsinds = [findhigh(localextrvals[i],maxzeroes[i]) for i in range(len(localextrvals))]
    maxvalues = [maxvals[0] for maxvals in maxzsinds]
    maxinds = [maxvals[1] + indlow for maxvals in maxzsinds]    

    if len(maxvalues) > 0:
        maxcorr = np.nanmax(maxvalues)
        maxdists = [flatdists[i] for i in range(len(flatdists)) if maxvalues[i] == maxcorr]
    else:
        maxcorr = 0
        maxdists = 0    
   
    maxtemp = np.fft.ifft(selectedtemps[np.argmax(maxvalues)])
    return maxvalues, maxinds, flatdists, maxtemp

def find_first_zero(array, interpolate = False):
    '''given an array, finds the index of the first place where the array is either equal to zero or crosses zero'''
    cross_inds = [i for i in range(len(array)-1) if np.sign(array[i]) != np.sign(array[i+1]) or array[i] == 0]
    if len(cross_inds) == 0:
        zval = np.argmin(array)
    elif not interpolate:
        zval = np.min(cross_inds)
    else:
        #in this case, we interpolate linearly to find the place where our curve equals zero
        lower = np.min(cross_inds)
        upper = lower+1
        slope = (array[upper]-array[lower])
        b = array[upper]-slope*float(upper)
        
        if slope != 0:
            zval = -b/slope
        else: 
            zval = lower + 0.5
    if np.isnan(zval):
        zval = 0.
    return zval

def piecewise(chan, sl1, int1, pivot, sl2):
    '''Piecewise function that we use to find the curve of growth turnover.'''
    int2 = pivot*(sl1 - sl2) + int1
    return [sl1*c + int1 if c < pivot else sl2*c + int2 for c in chan]


def calculate_full_integral(flux, velocities, vel_cent, low_bound, high_bound, which_integral = 0, diag = False, resolution = 5.5, find_slope = False, plot_title = '', rms = 0.0):
    '''this returns the flux integral as a function of velocity from center. the additional argument tells if you
    are calculating the lower velocity integral (-1), the upper velocity integral (1), or both (0)'''
    #arguments:
    #    flux: flux values
    #    velocities: velocities that correspond to the channels in flux
    #    vel_cent: the rounded channel index of the intensity weighted mean velocity for the real emission in the profile
    #    low_bound, high_bound: the bounds for the region used to calculate the curve of growth
    
    #the 0.001 here converts the fluxes from mJy km/s to Jy km/s
    left_vels = [0.001*np.trapz(flux[i:vel_cent], -velocities[i:vel_cent]) for i in np.arange(vel_cent-1, low_bound, -1)]
    right_vels = [0.001*np.trapz(flux[vel_cent:i], -velocities[vel_cent:i]) for i in np.arange(vel_cent+1, high_bound, 1)]
    full_vels = [l+r for l,r in zip(left_vels,right_vels)]

    #now we select which vels we are using here:
    if which_integral == -1 and len(left_vels) > 1:
        sel_int = left_vels
        #sel_errs = flux_errs[low_bound:vel_cent]
    elif which_integral == 1 and len(right_vels) > 1:
        sel_int = right_vels
        #sel_errs = flux_errs[vel_cent:high_bound]
    elif which_integral == 0 and len(full_vels) > 1:
        sel_int = full_vels
        rms = np.sqrt(2.)*rms
    else:
        if len(left_vels) > 1:
            sel_int = left_vels
        else:
            sel_int = right_vels
            if len(right_vels) < 1:
                print('bad news sir!')
    ## okay... I think we're going to just say that the error is the sum of the values from the preceding and current channels...?
    cogerr = np.array([np.sqrt(n-0.375)*resolution*rms for n in range(1,len(sel_int)+1)])
    #these help us determine where the "flat part of our curve of growth" begins, which is important for normalization and finding the flux
    velgrad = np.gradient(sel_int)
    piv_0 = find_first_zero(velgrad)
    if piv_0 == 0:
        piv_0 = int(len(sel_int) / 2)
    
    sl1_0 = np.median(sel_int[piv_0:])/(piv_0)

    opt, covariance = sco.curve_fit(piecewise, range(len(sel_int)), sel_int, p0 = [sl1_0, 0, piv_0, 0])#, sigma = cogerr)
    flux_val = opt[0]*opt[2] + opt[1]
    cross_ind = int(opt[2])
    if flux_val == 0:
        #this is a contingency measure 
        flux_val = np.amax(sel_int[cross_ind:])
    normalized_cog = [f/flux_val for f in sel_int]
       
    param_errs = np.sqrt(np.diag(covariance))
    int2_opt = opt[2]*(opt[0] - opt[3]) + opt[1]
    u_ps = [v+e for v,e in zip(opt, param_errs)]
    d_ps = [v-e for v,e in zip(opt, param_errs)]
    resid_diff = np.std([s - opt[3]*c - int2_opt  for s,c in zip(sel_int[cross_ind:], range(cross_ind, len(sel_int)))])
    
    upper_flux = u_ps[0]*u_ps[2] + u_ps[1] + resid_diff
    lower_flux = d_ps[0]*d_ps[2] + d_ps[1] - resid_diff
    
    ## Find slopes is for my use in making shapes....
    #okay, i think we do polyfit for the normalized_cog[0:crossind] and sel_int[0:crossind]?
    if find_slope and cross_ind != 0:
        slope, intc = np.polyfit(sel_int[0:cross_ind], normalized_cog[0:cross_ind],1)
    elif find_slope:
        slope = -100.
    if diag:
        plt.plot(left_vels, c = 'forestgreen', linestyle = 'dashed', label = 'Left COG')
        plt.plot(right_vels, c = 'gold', label = 'Right COG')
        plt.errorbar(range(len(sel_int)), sel_int, yerr = cogerr, marker = '*', c = 'black', label = 'Symmetric COG')
        my_pw = np.array(piecewise(range(len(sel_int)), opt[0], opt[1], opt[2], opt[3]))
        if which_integral == 0:
            left_flux = flux[low_bound:vel_cent]
            left_flux = left_flux[::-1]
            right_flux = flux[vel_cent:high_bound]
            full_flux = [l+r for l,r in zip(left_flux,right_flux)]
            plt.step(range(len(right_flux)), [s*flux_val/np.nanmax(full_flux) for s in right_flux], c = 'gold')
            plt.step(range(len(left_flux)), [s*flux_val/np.nanmax(full_flux) for s in left_flux], c = 'forestgreen')
            plt.step(range(len(full_flux)), [s*flux_val/np.nanmax(full_flux) for s in full_flux], c = 'grey', lw = 7)
            plt.fill_between(range(len(full_flux)),np.zeros(len(full_flux)), [s*flux_val/np.nanmax(full_flux) for s in full_flux], alpha = 0.7, color = 'grey', step = 'pre')
        plt.plot(range(len(sel_int)), [flux_val for z in range(len(sel_int))], c = 'red', label = 'Flux')
        plt.plot(range(len(sel_int)), [upper_flux for z in range(len(sel_int))], c = 'red', linestyle = '--', label = 'Upper flux')
        plt.plot(range(len(sel_int)), [lower_flux for z in range(len(sel_int))], c = 'red', linestyle = 'dotted', label = 'Lower flux')
        plt.plot(range(len(sel_int)), my_pw, c = 'grey', label = 'Best-fit piecewise')
        #add the median value
        plt.legend()
        plt.title(plot_title+' Curve of Growth')
        plt.xlabel('Width (channels)')
        plt.ylabel('Integrated flux of COG (Jy km/s)')
        plt.show()
        
        '''## and then the other version of this plot, with the integration outward from profile center:
        ax1 = plt.subplot()
        plt.step(velocities[lowbound:highbound],flux[lowbound:highbound], where = 'pre', c = 'grey', linewidth = 2)
        plt.fill_between(velocities[lowbound:highbound], flux[lowbound:highbound], step = 'pre', color='grey', alpha = 0.5)

        ax2 = ax1.twinx()
        ax2.set_ylim(-0.3,1.2)
        ## ax1 ylim should be based on the max flux in flux
        max_val = 1.1*np.nanmax(flux[lowbound:highbound])
        ax1.set_ylim(-max_val/3.0, max_val)
        ax2.plot(vels[centind+1:high_integrange],np.array(norm_cog_l), c = 'gold', label = 'Right COG', linewidth = 2)
        ax2.plot(vels[centind+1:high_integrange],np.array(norm_cog_b), c = 'black', label = 'Sum COG', linewidth = 2)
        ax2.plot(vels[low_integrange+1:centind],np.array(norm_cog_r[::-1]), c = 'forestgreen', linestyle = '--', label = 'Left COG', linewidth = 2)
        ax2.plot(vels[low_integrange+1:centind],np.array(norm_cog_b[::-1]), c = 'black', linewidth = 2)

        ax2.fill_between([vels[low_integrange], vels[high_integrange]],[flx_b[0]/flx_b[1], flx_b[0]/flx_b[1]], [flx_b[2]/flx_b[1], flx_b[2]/flx_b[1]], color='red', alpha = 0.35, label = 'Integrated flux from COG')
        ax2.plot([vels[low_integrange], vels[high_integrange]], [0.75,0.75], c = 'black')

        low_line = -0.3
        high_line = 1.2
        ax2.plot([vhel, vhel],[low_line, high_line], c = 'dodgerblue', label = r'ALFALFA $V_{hel}$')
        ax2.plot([vhel+w50/2., vhel+w50/2],[low_line, high_line], c = 'dodgerblue', linestyle = '--', label = r'ALFALFA $W_{50,P}$')
        ax2.plot([vhel-w50/2., vhel-w50/2],[low_line, high_line], c = 'dodgerblue', linestyle = '--')
        ax2.plot([rvs[3]+rvs[0], rvs[3]+rvs[0]],[low_line, high_line], c = 'black', linestyle = 'dotted', label = r'$V_{75}$')
        ax2.plot([rvs[3]-rvs[0], rvs[3]-rvs[0]],[low_line, high_line], c = 'black', linestyle = 'dotted')

        plt.legend()
        ax1.set_xlabel('Heliocentric velocity (km/s)')
        ax1.set_ylabel('Flux density (Jy)')
        ax2.set_ylabel('Fraction of integrated flux')
        plt.xlim(vels[low_integrange],vels[high_integrange])'''
        
        
    if find_slope:
        retvals = [lower_flux, flux_val, upper_flux], normalized_cog, opt, slope
    else:
        retvals = [lower_flux, flux_val, upper_flux], normalized_cog, opt
    return retvals

def find_velocity(normalized_curve, velocity_thresh, velocities, interpolate = False):
    '''given a normalized curve of growth, finds the width of the profile at velocity_thresh percent of the flux density.'''
    curve_through_thresh = [n-velocity_thresh for n in normalized_curve]

    first_zero_ind = float(find_first_zero(curve_through_thresh, interpolate = interpolate))
    if first_zero_ind.is_integer():
        rotvel = velocities[int(first_zero_ind)]
    else:
        #this interpolates the velocity to the non-integer index found if interpolate = True
        int_index = int(first_zero_ind)
        if int_index < len(velocities)-1 and int_index > 0:
            delta = velocities[int_index+1] - velocities[int_index]
            vel_delt = (first_zero_ind - float(int_index))*delta
            rotvel = velocities[int_index] + vel_delt
        elif int_index == 0:
            rotvel = velocities[0]
        elif len(velocities) == 0:
            rotvel = 0.
        else:
            rotvel = velocities[len(velocities)-1]
    return rotvel

def find_velrange_centvel(flux, vels, indlow, indhigh, templatebank):
    corrvals, corrinds, peakdists, centtemp = correlationFinder(flux, indlow, indhigh,templatebank)
    profcenter = int(corrinds[np.argmax(corrvals)]) 
    centvel = vels[profcenter]
    
    mostlikelydistance = peakdists[np.argmax(corrvals)]
    sig = sigma_from_dist(mostlikelydistance)
    low_integrange = np.maximum(0, int(profcenter-6*mostlikelydistance))
    high_integrange = np.minimum(int(profcenter+6*mostlikelydistance), len(vels)-1)
    return profcenter, low_integrange, high_integrange, centvel 

##------------------------------------------- PRE-PROCESSING FUNCTION --------------------------------------------------------##

def blank_galactic(profile, velocities, vhel, diag = False):
    '''This function does some very coarse removing galactic emission -- this is necessary so that the correlation finder doesn't select the galactic emission as the heliocentric velocity'''
    peak_flux = np.argmax(profile)
    peak_vel = velocities[peak_flux]
    if np.abs(peak_vel - vhel) > np.abs(peak_vel) and np.abs(peak_vel) < 200.:
        #print('probably from galactic emission')
        high = peak_flux
        low = peak_flux
        while(profile[high]) > 0 and high < len(profile):
            high += 1
        while(profile[low]) > 0 and low > 0:
            low -= 1
        if diag:
            plt.plot(velocities, profile)
            plt.plot(velocities[low:high], profile[low:high])
        profile = [profile[p] if p < low or p > high else 0 for p in range(len(profile))]
        if diag:
            plt.plot(velocities, profile)
            plt.show()
    return profile

##--------------------------------------------Function to be called outside----------------------------------------------##

def cog_velocities(velocities, flux, v_helio, low_ind, high_ind, templatebank, rms = 0.0, vel_thresh = 0.75, which_cog = 0, diagnose = True, interp = False, return_vhel = False, resolution = 5.5, return_params = False, agcn = ''):
    ''' this is the function that you call outside of this script in order to find COG velocities.'''
    #arguments:
    #check code for cases where data resolution is different(ASD)
    #check for/add plotting on same level as measure
    #    vel_thresh: the fraction of the integrated flux that defines the velocity width
    #    which_cog: this tells whether you calculate using just the profile integrated to higher velocities (-1), lower velocities (1), or using the full profile (0)
    #    diag: for diagnosing problems - if true, each method spits out diagnostic images.
    #    return_slopes: if true, returns the long-term slope for the left and right sides of the PW function -- useful for determining if profiles have underestimated widths
    centind, low_integrange, high_integrange, centvel = find_velrange_centvel(flux, velocities, low_ind, high_ind, templatebank)
    # We need the full profiles to get the maximum resolution in our ccfs for find_velrange_centvel, but once we're integrating the profiles we don't....
    fluxes, norm_cog, params = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = which_cog, diag = diagnose, plot_title = 'AGC '+str(agcn), rms = rms, resolution = resolution)
    ## This converts the arrays which have the velocities as a function of channel to arrays that give delta_V from the center to a given channel, which are useful in finding the width of the profile.
    if which_cog == 0:
        vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,np.minimum(centind-low_integrange, high_integrange-centind))]
    elif which_cog == -1:
        vels_for_cog = [velocities[centind-j]-velocities[centind] for j in range(1,centind-low_integrange)]
    else:
        vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,high_integrange-centind)]
    
    vel = find_velocity(norm_cog, vel_thresh, vels_for_cog, interpolate = interp)
    vel_low = find_velocity([n*fluxes[1]/fluxes[0] for n in norm_cog], vel_thresh, vels_for_cog, interpolate = interp)
    vel_high = find_velocity([n*fluxes[1]/fluxes[2] for n in norm_cog], vel_thresh, vels_for_cog, interpolate = interp)
    if diagnose:
        plt.plot(vels_for_cog,norm_cog, c= 'red', label = 'Normalized COG')
        plt.plot(vels_for_cog, [n*fluxes[1]/fluxes[0] for n in norm_cog], c = 'red',linestyle = '--', label = 'Higher normalization')
        plt.plot(vels_for_cog, [n*fluxes[1]/fluxes[2] for n in norm_cog], c = 'red',linestyle = 'dotted',label = 'Lower normalization')
        plt.plot([vels_for_cog[0], vels_for_cog[-1]], [vel_thresh,vel_thresh], c = 'black', label = 'Threshold for velocity width')
        plt.xlabel('Velocity width')
        plt.ylabel('Normalized value of COG')
        plt.legend()
        plt.show()
    if return_params:
        fluxes_n1, norm_cog_n1, params_n1 = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = -1, diag = False, plot_title = str(agcn), rms = rms)
        lts_n1 = params_n1[-1]
        fluxes_1, norm_cog_1, params_1 = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = 1, diag = False, plot_title = str(agcn), rms = rms)
        lts_1 = params_1[-1]
        if return_vhel and return_params:
            retvals = vel, vel-vel_low, vel_high-vel, centvel, [params_n1, params, params_1], [fluxes_n1, fluxes, fluxes_1]
        elif return_vhel: 
            retvals = vel, vel-vel_low, vel_high-vel, centvel, [params_n1, params, params_1], [fluxes_n1, fluxes, fluxes_1]
        else:
            retvals = vel, vel-vel_low, vel_high-vel, [lts_n1, lts_1], [fluxes_n1, fluxes_1]        
    elif return_vhel:
        retvals = vel, vel-vel_low, vel_high-vel, centvel
    else:
        retvals = vel, vel-vel_low, vel_high-vel
    return retvals

def all_info_fromprev(velocities, flux, v_helio, low_ind, high_ind, templatebank, rms, agcn = '', diagnose = False):
    ''' this is the function that you call outside of this script in order to find COG velocities.'''
    #arguments:
    #    vel_thresh: the fraction of the integrated flux that defines the velocity width
    #    which_cog: this tells whether you calculate using just the profile integrated to higher velocities (-1), lower velocities (1), or using the full profile (0)
    #    diag: for diagnosing problems - if true, each method spits out diagnostic images.
    #    return_slopes: if true, returns the long-term slope for the left and right sides of the PW function -- useful for determining if profiles have underestimated widths
    centind, low_integrange, high_integrange, centvel = find_velrange_centvel(flux, velocities, low_ind, high_ind, templatebank)
    # We need the full profiles to get the maximum resolution in our ccfs for find_velrange_centvel, but once we're integrating the profiles we don't....
    fluxes, norm_cog, params = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = 0, diag = False, plot_title = str(agcn), rms = rms)
    ## This converts the arrays which have the velocities as a function of channel to arrays that give delta_V from the center to a given channel, which are useful in finding the width of the profile.
    vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,np.minimum(centind-low_integrange, high_integrange-centind))]
    vel = find_velocity(norm_cog, 0.75, vels_for_cog, interpolate = True)
    vel_low = find_velocity([n*fluxes[1]/fluxes[0] for n in norm_cog], 0.75, vels_for_cog, interpolate = True)
    vel_high = find_velocity([n*fluxes[1]/fluxes[2] for n in norm_cog], 0.75, vels_for_cog, interpolate = True)
    if diagnose:
        plt.plot(vels_for_cog,norm_cog, c= 'red', label = 'Normalized COG')
        plt.plot(vels_for_cog, [n*fluxes[1]/fluxes[0] for n in norm_cog], c = 'red',linestyle = '--', label = 'Higher normalization')
        plt.plot(vels_for_cog, [n*fluxes[1]/fluxes[2] for n in norm_cog], c = 'red',linestyle = 'dotted',label = 'Lower normalization')
        plt.plot([vels_for_cog[0], vels_for_cog[-1]], [0.75,0.75], c = 'black', label = 'Threshold for velocity width')
        plt.xlabel('Velocity width')
        plt.ylabel('Normalized value of COG')
        plt.legend()
        plt.show()
    fluxes_n1, norm_cog_n1, params_n1 = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = -1, diag = False, plot_title = str(agcn), rms = rms)
    lts_n1 = params_n1[-1]
    fluxes_1, norm_cog_1, params_1 = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = 1, diag = False, plot_title = str(agcn), rms = rms)
    lts_1 = params_1[-1]
    
    a_f = np.nanmax([fluxes_n1[1], fluxes_1[1]])/np.nanmin([fluxes_n1[1], fluxes_1[1]])
    a_c = np.nanmax([params_n1[0], params_1[0]])/np.nanmin([params_n1[0], params_1[0]])
    ## PIV_DIFF addition on 10/20 tells you additional info about the asymmetry maybe / about the asymmetry in the integration...
    piv_diff = np.abs(params_n1[2] - params_1[2])/params[2]
    low_err = vel - vel_low
    high_err = vel_high - vel
    d_v_hel = centvel - v_helio
    v_85 = find_velocity(norm_cog, 0.85, vels_for_cog, interpolate = True)
    v_25 = find_velocity(norm_cog, 0.25, vels_for_cog, interpolate = True)
    c_v = v_85/v_25
    
    ## NOW WE NEED TO TELL IF THE PROFILE IS RECOMMENDED OR NOT....
    recd = True
    flag_step = 0
    if d_v_hel > 50:
        flag_step = 1
        recd = False
    elif low_err < 0 or high_err < 0:
        flag_step = 2
        recd = False
    elif np.abs(params[3]) > 2.5e-3:
        flag_step = 3
        recd = False
    
    dict_vals = {'V_hel':centvel,'V75s':vel, 'eV75s_low':low_err, 'eV75s_high':high_err, 'A_C':a_c,'A_F':a_f,'C_V':c_v, 'd_V_hel':d_v_hel, 'BF_slope1':params[0],'BF_intc':params[1],'BF_LTS':params[3],'BF_pivot':params[2], 'COG_flux':fluxes[1],'eCOG_flux':(fluxes[2]-fluxes[0])/2., 'Pivot_diff':piv_diff, 'Flag_Step':flag_step, 'Recommended':recd}
    df = pd.DataFrame(dict_vals, index = [agcn])
    return dict_vals, df