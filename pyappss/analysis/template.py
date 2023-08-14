'''The code here is used to generate the templates for the Curve of Growth code.This is taken directly from Catie Ball's github repo.
The templates are generated using the following function:'''

import numpy as np
import pickle
from os import listdir
def herm0(x,sig):
    coeff = np.sqrt(sig*np.sqrt(np.pi))**-1.0
    return coeff*np.exp(-(x/sig)**2.0/2.0)
    
def herm2(x,sig):
    coeff = np.sqrt(sig*np.sqrt(np.pi))**-1.0
    hermpol = -(2.0)**(-0.5) + (2.0)**0.5*(x/sig)**2.0
    return coeff*hermpol*np.exp(-(x/sig)**2.0/2.0)

def template(sig,a0,a2,reso = 5.5, width = 1024, normalize = False):
    #i promise, i did the math out and this is the projection that follows for projection of the first two polynomials onto a 5 sigma boxcar.
    a0 = 0.8642766554208592
    a2 = 0.5030167620413195
    an0 = a0/np.sqrt(a0**2.0+a2**2.0)
    an2 = a2/np.sqrt(a0**2.0+a2**2.0)
    txes = np.arange(-width*reso/2.,width*reso/2.,reso)
    sig1 = 25.48
    sig2 = 150.0
    #f is a function that varies from 1 at sig = 25.48 to 0 at sig=150.0
    f = (sig - sig2)/(sig1-sig2)
    if sig < sig1:
        templatevals = [herm0(x,sig) for x in txes]
        retcoeff = 1.0
        rc2 = 0
    if sig >= sig1 and sig <= sig2:
        bf = f*(1.0 - an0) + an0
        b2 = np.sqrt(1.0-bf**2.0)
        templatevals = [bf*herm0(x,sig)+b2*herm2(x,sig) for x in txes]
        retcoeff=bf
        rc2 = b2
    if sig > sig2:
        templatevals = [an0*herm0(x,sig)+an2*herm2(x,sig) for x in txes]
        retcoeff = an0
        rc2 = an2
    normfac = np.sum(templatevals)
    return templatevals
def finda0a2(sigma):
    boxcar = np.zeros(1024)
    vels = [i-512 for i in range(len(boxcar))]#5.5*i-2816.0 for i in range(len(boxcar))]
    for i in range(len(vels)):
        if np.abs(vels[i]) < 2.5*sigma:
            boxcar[i] = 1.
            
    h0 = [herm0(x, sigma) for x in vels]
    h2 = [herm2(x, sigma) for x in vels]
    
    proj0 = [h*b for h,b in zip(h0, boxcar)]
    proj2 = [h*b for h,b in zip(h2, boxcar)]
    
    
    coeff0 = np.sum([h*b for h,b in zip(h0, boxcar)])
    coeff2 = np.sum([h*b for h,b in zip(h2, boxcar)])

    normalization = np.sqrt(coeff0**2.+coeff2**2.)
    coeff0 = coeff0 / normalization
    coeff2 = coeff2 / normalization
    
    projection = [coeff0*h0v+coeff2*h2v for h0v,h2v in zip(h0,h2)]
    return coeff0, coeff2
def ccf_template_check(reso, prof_len, tb_dir = 'templatebanks'):
    # This function checks to see if there is a template that matches the profile's specifications, either returning the name of the bank if yes or creating the bank and then returning the name of the bank
    banknames = listdir(tb_dir)
    numchans = [float(c[len(c)-11:len(c)-7]) for c in banknames]
    bankres = [float(c[len(c)-6:len(c)-2])/10.0 for c in banknames]
    ind_acceptable = [i for i in range(len(bankres)) if np.abs(bankres[i]-reso) < 1 and numchans[i] == prof_len]
    #print('number of acceptable temps:',[banknames[i] for i in ind_acceptable])
    if len(ind_acceptable) == 0:
        bank_name = make_templatebank(round(reso*2.0)/2.0, sp_w = prof_len, bankdir = tb_dir+'/')
    elif len(ind_acceptable) == 1:
        bank_name = tb_dir+'/'+banknames[ind_acceptable[0]]
    else:
        print('Ambiguous, using closer resolution')
        close_ind = np.argmin([np.abs(bankres[i]-reso) for i in ind_acceptable])
        print(close_ind, len(ind_acceptable))
        bank_name = tb_dir+'/'+banknames[ind_acceptable[close_ind]]
    return bank_name
def makeTemplate(peakdist, width = 1024, res = 5.5):
    #this just puts everything into the one function that you run. Peakdist is in the number of channels between peaks --
    # we do it this way rather than in km/s because the place where you decide WHICH templates go into your computed CCFs is
    # automated
    a0oa2 = 1.75
    coeff = 1./np.sqrt(5./2.-a0oa2/np.sqrt(2.))
    if peakdist*res < 300.0 and peakdist*res > 30.0:
        # the 5.5 is here because these coefficients / the transition between them was fit using 5.5 km/s data
        pdcorr = (19.3+0.66*(peakdist*res/5.5))
        sigma = pdcorr*5.5/2.0*coeff
        a0, a2 = finda0a2(sigma)
        #print(sigma)
        templ = template(sigma, a0, a2, reso = res, width = width)
    elif peakdist*res <= 30.0:
        #these ones are just going to be gaussians.
        sigma = (peakdist*res)
        a0, a2 = finda0a2(sigma)
        templ = template(sigma, a0, a2, reso = res, width = width)
    elif peakdist*res >= 300.0:
        sigma = (peakdist*res/2.0)*coeff
        a0, a2 = finda0a2(sigma)
        templ = template(sigma, a0, a2, reso = res, width = width)
    #print('sigma: ',sigma,' res: ',res)
    return templ
def make_templatebank(resolution, sp_w = 1024, bankdir = ''):
    ## OKAY, so we want one per channel out to... 800 km/s?
    peakdists = np.arange(1,800/resolution,1)
    print('Now making ',len(peakdists),' templates....')
    templates = [makeTemplate(p, res = resolution, width = sp_w) for p in peakdists]
    # next, we need to roll and then FFT these guys so that they can be used the same 
    print('Templates made, now to roll and FFT for ease of use...')
    ffted_rolled_templates = [np.fft.fft(np.roll(t, int(len(t)/2))) for t in templates]
    # now we need to format the name of the files... we want wxy.z resolution:
    str_res = str(int(resolution*10))
    numzs = 4 - len(str_res)
    for i in range(numzs):
        str_res = '0'+str_res
    str_len = str(sp_w)
    numz_len = 4 - len(str_len)
    for i in range(numz_len):
        str_len = '0'+str_len
    bankname = bankdir+'templatebank_'+str_len+'_'+str_res+'.p'
    pickle.dump(ffted_rolled_templates, open(bankname,'wb'))
    return bankname

# Added function to get or create the required template bank given resolution and profile length:

def get_or_create_template_bank(resolution, sp_w, bankdir='templatebanks'):
    return ccf_template_check(resolution, sp_w, bankdir)
