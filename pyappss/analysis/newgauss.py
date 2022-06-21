import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.modeling.models import custom_model
from astropy.modeling import fitting

from astropy.io import fits
from astropy import units as u

from scipy.signal import butter,filtfilt
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel
from bisect import bisect, bisect_left
import baseline
import os
import argparse
#from scipy.optimization import optimize

class ManyGauss:
	def __init__(self, filename, smooth_int=None, dev=10, vel=None, spec=None, rms=None):
		datapath = 'AGC' + str(filename) + '.fits'
		hdul = fits.open(datapath)
		data = hdul[1].data
		length = len(data)

		self.x = []
		self.y = []
		self.spec = spec
		self.vel = vel
		self.rms = rms
		self.convolved = []
		self.boxcar = True

		#Clipping 1000 channels in either direction to ignore oddities at the ends.
		for i in range(1000, 7192):

			self.y.append(self.spec[i])
			self.x.append(self.vel[i])
		#Should not be an issue with the GBT data, but this double-checks for any NAN values which can completely mess up the convolution and fit.
		self.y=np.nan_to_num(self.y)
		self.x=np.nan_to_num(self.x)
		
		
		self.dev = int(input("\n Please enter the standard deviation for the gaussian convolution."
						"\nRecommended is 10 - this value works well for all tested galaxies. Other values have not been tested.\n"))
		
		plt.ion()
		plt.rcParams["figure.figsize"] = (15, 9)
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot()

		self.gaussfilter()
		self.plot()
		self.manygaussfitter()
		
		self.plot(fitted=True)
		
		self.deltav, self.fluxerr, self.SN, self.flux, self.vsys, self.vsyserr, self.w20, self.w20err, self.w50, self.w50err = self.manygauss_calc()
		self.__print_values()

		input('Press Enter to exit')

	def __print_values(self):
		print('\n')
		print('W50 = ', self.w50, ' +/- ', self.w50err, ' km/s ')
		print('W20 = ', self.w20, ' +/- ', self.w20err, ' km/s ')
		print('vsys = ', self.vsys, ' +/- ', self.vsyserr, ' km/s')
		print('flux = ', self.flux, ' +/- ', self.fluxerr, ' Jy km/s')
		print('SN: ' + str(self.SN))

	def plot(self, xmin=None, xmax=None, ymin=None, ymax=None, fitted=False):
		plt.cla()
		self.ax.plot(self.vel, self.spec, linewidth=1)
		self.ax.axhline(y=0, dashes=[5, 5])
		self.ax.set(xlabel="Velocity (km/s)", ylabel="Flux (mJy)")
		self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
		self.ax.plot(self.x, self.convolved)
		
		if fitted == True:
			self.ax.plot(self.x, self.gaussian_model)
		
		self.fig.canvas.draw()
		

	def __call__(self):
		return self.x, self.y, self.spec, self.vel, self.convolved, self.gaussian_model
	
	def region_selection(self):
		global regions
		global set_bounds
		regions = []
		set_bounds = self.fig.canvas.mpl_connect('button_press_event', self.region_selection_mark)
		response = input("\nPlease select the region to attach the fits to! Use the overlaid, convolution plot as your guide."
						"\nFit the regions as closely as possible to the signal, as this determines the placement and quantity of gaussians."
						"\nThis fitting method requires a region at the minimum centered on the profile, and really fit exactly to the signal, optimally."
						"\nPress Enter if the region is OK, or type \'clear\' and press Enter to restart region selection\n")
		regions_good = False
		while not regions_good:
			if response =='':
				regions_good = True
			elif response == 'clear':
				del regions
				regions =[]
				self.plot()
				set_bounds = self.fig.canvas.mpl_connect('button_press_event', self.region_selection_mark)
				response = input("\nRegion cleared! Select a new region now. Press Enter if the region is OK, or type \'clear\' and press Enter to clear region selection.\n")
			else:
				response = input("\nPlease press Enter if the region is OK, or type \'clear\' and press enter to clear region selection.\n")
			# self.fig.canvas.mpl_disconnect(set_bounds)
		regions.sort()
		v = list()
		s = list()
		for i in range(len(self.x)):
			for j in range(len(regions) - 1):
		# constructing v and s lists if they are within the selected region.
				if regions[j] <= self.x[i] <= regions[j + 1]:
					v.append(self.x[i])
					s.append(self.convolved[i])
		# changing v and s into numpy arrays so calculations become shorter.

		del set_bounds, regions
		v = np.asarray(v)
		s = np.asarray(s)
		return v, s

	def region_selection_mark(self, event):
		ix, iy = event.xdata, event.ydata
		regions.append(ix)
		self.ax.plot([ix, ix], [-100, 100], linestyle='--', linewidth=0.7, color='green')
		if len(regions) >=2:
			self.fig.canvas.mpl_disconnect(set_bounds)


	def gaussfilter(self):
		#Gaussian convolution which reduces noise and aids in approximation of order of required gaussians
		gauss_kernel = Gaussian1DKernel(self.dev)
		self.convolved = convolve(self.y, gauss_kernel)
	
	def manygaussfitter(self, iterations=500):
		fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
		length=len(self.x)

		global regions
		global set_regions

		regions=[]
		lines=[]

		v, s = self.region_selection()
		vel1 = v[0]
		vel2 = v[len(v) - 1]
		pos1 = bisect_left(self.x, vel1)
		pos2 = bisect(self.x, vel2)
		deltav = (abs((self.x[pos1] - self.x[pos1 - 2])/2) + abs((self.x[pos2] - self.x[pos2 - 2])/2)) / 2
		
		changes = []
		positions = []
		self.shorter_x = []
		self.shorter_y = []
		#This section calculates local maxima and minima within the chosen region, using the convolved, and therefore smoother, plot to find them. Too many is a bad thing, until I can make it iterate absurdly
		for i in range(pos1, pos2):
			delta_for = self.convolved[i+1] - self.convolved[i]
			delta_back = self.convolved[i] - self.convolved[i-1]
			signage = delta_for * delta_back
			if signage < 0:
				changes.append(self.x[i])
				positions.append(i)

		#Here, the subsection of the data to attach the fit to is created. As fitting to the convolution is imprecise, since the convolution does not give error values, this is the better alternative
		#It takes the selection region, and tacks on 100 channels of 0'd flux on either end. This assists the fit.The model can still be set equivalent to x, or even vel, since gaussian definitions and stuff. It's great
		for i in range (100):
			fakevel = (vel1 - 100*deltav) + i * deltav 
			self.shorter_x.append(fakevel)
			self.shorter_y.append(0)
		for i in range ((pos1), (pos2)):
			self.shorter_x.append(self.x[i])
			self.shorter_y.append(self.y[i])
		for i in range (1, 100):
			fakevel = (vel2) + i * deltav 
			self.shorter_x.append(fakevel)
			self.shorter_y.append(0)
		#The fit is attached. Thankfully, astropy has a good setup for creating compound models, so this just assigns a gaussian to each max/min and lets the fitting algorithm do the rest. Unfortunately, astropy's use of scipy's optimize 
		#is not perfect - for trial galaxies, the fit tended to be near perfect on the left part of the signal, and fade away from perfection (although still present an OK optimization) on the right part of the signal. 
		#The data is simply too complex for the number of attempts that astropy allows, so unless the signal is very narrow, it will stop at a certain point. The closer the guesses for amplitude, mean, and standard deviation are
		#to their actual values, the easier type the optimization routine has. Unfortunately, consistently making accurate guesses for these values has not been figured out yet. The isolation of the first gaussian from the for loop allows the 
		#compound model to be exclusively gaussian, and the shortened and re-extended arrays to fit to aided in precision, as less data leads to fewer integrations for the same result, but no zeroed baseline hurts the fit's accuray.
		self.gauss_number = len(changes)
		gauss_tot = models.Gaussian1D(mean=changes[0])
		for i in range(1, self.gauss_number):
			one_gauss = models.Gaussian1D(mean=changes[i])
			gauss_tot += one_gauss
		self.gaussfit = fitter(gauss_tot, self.shorter_x, self.shorter_y, maxiter=iterations)
		self.gaussian_model = self.gaussfit(self.x)
	
	def manygauss_calc(self):
		
	
		midchan_l, midchan_r = self.gauss_edges()
		leftvel50, leftvel20, leftvelerr, leftcoef = self.gauss_edgefit(midchan_l)
		rightvel50, rightvel20, rightvelerr, rightcoef = self.gauss_edgefit(midchan_r)
		
		self.ax.plot(self.x, leftcoef[1] + leftcoef[0] * self.x, color='red')
		self.ax.plot(self.x, rightcoef[1] + rightcoef[0] * self.x, color='red')
		print(midchan_l)
		print(midchan_r)
		
		
		#The old method of error calculation. This is a true calculation, based upon the errors of each of my (sometimes many) gaussians. Unfortunately, calculation error this way, while strictly
		#the most accurate method, yields an error far larger than the flux - probably due to the fact that there are negative gaussians that contribute to the error but subtract from the signal
		
		aerr_list = []
		a_list = []
		a = 0
		aerr_sum = 0
		for i in range(self.gauss_number):
			spot = 3 * i
			model_amp = self.gaussfit[i].amplitude
			model_amp_err = self.gaussfit.stds[spot]
			model_dev = self.gaussfit[i].stddev
			model_dev_err = self.gaussfit.stds[spot+2]
			a_inst = model_amp * model_dev * np.sqrt(2 * np.pi)
			a += a_inst
			#aerr_inst = abs(a_inst * np.sqrt((model_amp_err ** 2) / (model_amp ** 2) + (model_dev_err ** 2) / (model_dev ** 2)))
			aerr_inst = np.sqrt( (( (model_dev * np.sqrt(2* np.pi)) ** 2) * ((model_amp_err) ** 2)) + ( ((model_amp * np.sqrt(2* np.pi) ) ** 2) * ((model_dev_err) **2)) )
			aerr_sum += (aerr_inst ** 2)
			
			aerr_list.append(aerr_inst)
			a_list.append(abs(a_inst))
		
		#New and "improved" method of error calculation. Reasons that the error could be explained as the deviation of the total gaussian model from the data. 
		#As a result, an instantaneous error is calculated for each channel, using the difference between the data and the total model. Then, incorporate that we
		#are calculating area, and add in an accounting for RMS, which utilizes the method in measure.py
		#Add those in quadrature for each channel, and then add each channel's error in quadrature, and a final, total error value can be found.
		#This is predicated on the fact that RMS and the deviation of the model from the data are not linked.
		
		y_err_sum = 0
		for i in range(self.leftedge_chan - 1, self.rightedge_chan - 1):
				delta_y = abs(self.gaussian_model[i] - self.y[i])
				delta_y_fixed = delta_y / 1000
				delta_v = abs(self.x[i + 1] - self.x[i - 1]) / 2.
				#The 1.4 * w50 is in reference to the total spectrum width, I believe, so turning it into just delta v should be instantaneous
				rms_err = 2 * (self.rms / 1000) * np.sqrt(delta_v)
				delta_a = delta_y_fixed * delta_v
				y_err_inst = np.sqrt( (delta_a ** 2) + (rms_err **2) )
				y_err_sum += (y_err_inst ** 2)
				
		y_err_tot = np.sqrt(y_err_sum)
		
		fluxpeak = max(self.gaussian_model)  # highest point of the source
		SNR = (fluxpeak - self.rms)  # signal-noise ratio
		logSNR = np.log10(SNR)
		deltav = abs(self.x[self.centerchan + 1] - self.x[self.centerchan - 1]) / 2.
		Lambda = self.calculate_lambda(deltav, logSNR)
		w50 = abs(rightvel50 - leftvel50) - 2 * deltav  ##width at half the height of the source. subtracting off noise and instrumental broadening effect
		w50err = np.sqrt(leftvelerr ** 2 + rightvelerr ** 2)
		w20 = abs(rightvel20 - leftvel20)  # width at 1/5 the height of the source
		w20err = w50err
		w50 = w50 - 2 * deltav * Lambda  # Subtract off noise+instrumental broadening effect
		w20 = w20 - 2 * deltav * Lambda
		vsys = .5 * (rightvel50 + leftvel50)  # where rightvel and leftvel are half vels in the right and left fit
		vsyserr = w50err / np.sqrt(2)
		# Setting flux and flux error equal to our calculated gaussian areas
		totflux = a / 1000
		#fluxerr = aerr / 1000
		fluxerr = y_err_tot
		# Calculate signal to noise (the ALFALFA way)
		sn = 1000 * totflux / w50 * np.sqrt((w50.clip(min=None, max=400.) / 20.)) / self.rms


		return deltav, fluxerr, sn, totflux, vsys, vsyserr, w20, w20err, w50, w50err
	
	def gauss_edges(self):
		left_edge = self.gaussfit[0].mean - (3 * self.gaussfit[0].stddev)
		#left_err = np.sqrt((gfit.stds[1] ** 2) + (9 * (gfit.stds[2] ** 2)))
		right_edge = self.gaussfit[self.gauss_number-1].mean - (3 * self.gaussfit[self.gauss_number-1].stddev)
		#spot_right = 3 * (self.gauss_number -1)
		#right_err = np.sqrt((gfit.stds[spot_right+1] ** 2) + (9 * (gfit.stds[spot_right+2] ** 2)))
		#Error definitions left in in case I somehow need them again.
		
		#Quick and dirty, no user-input method to find peak values for an edge fit. Certainly could fail in some instances, but since it fits to 85% of the maximum anyways, it just might be fine.
		#Splits in two and does each side; centered on center of shorter_x, or the region selected to attach fit to.\
		#This determining of the max values should be nigh-errorless, as the model fits the peaks, ie the max values, very very well. 
		
		#That's actually a bald-faced lie. This technique does not work for non-doublehorned spectra. Definitely a few ways that I can approach this problem, but the initial build
		#Works only for double horned profiles, probably.
		shorter_chan = len(self.shorter_x)
		self.centerchan = bisect_left(self.x, self.shorter_x[round((shorter_chan - 1)/2)])
		leftmax = np.where(self.gaussian_model == max(self.gaussian_model[0:self.centerchan]))
		rightmax = np.where(self.gaussian_model == max(self.gaussian_model[self.centerchan:(len(self.x)-1)]))
		
		self.leftedge_chan = bisect_left(self.x, left_edge)
		self.rightedge_chan = bisect(self.x, right_edge)
		
		#This code is nearly identical to twopeak's edgefit, but of course modified.
		maxval_l = self.gaussian_model[leftmax[0]]
		percent15_l = .15 * maxval_l
		percent85_l = .85 * maxval_l
		maxval_r = self.gaussian_model[rightmax[0]]
		percent15_r = .15 * maxval_r
		percent85_r = .85 * maxval_r
		region_l = []
		region_r = []
		for i in range(len(self.x)):  # iterating over the shortened spectrum so we have consistency in the indices
			if self.gaussian_model[i] >= percent15_l and self.gaussian_model[i] <= percent85_l and self.x[i] >= left_edge and self.x[i] <= self.x[leftmax]: #LEFT
				region_l.append(i)  # appending the indices where the y values are between 15 and 85%, and where the x values are within the selected region.
			#Wrapped into one for loop since I define stuff like this. It'd be six of one half a dozen of another to do it the way twopeak calc/edgefit does, where it does if left and then if right
			if self.gaussian_model[i] >= percent15_r and self.gaussian_model[i] <= percent85_r and self.x[i] >= self.x[rightmax] and self.x[i] <= right_edge: #RIGHT
				region_r.append(i)  # appending the indices where the y values are between 15 and 85%, and where the x values are within the selected region.
				
		p15chan_l, p15chan_r = min(region_l), min(region_r)
		p85chan_l, p85chan_r = max(region_l), max(region_r)
		midchan_l = [p15chan_l, p85chan_l]
		midchan_r = [p15chan_r, p85chan_r]
		midchan_l.sort()
		midchan_r.sort()
		
		return midchan_l, midchan_r
		
	def gauss_edgefit(self, midchan):
		
		#Perform a a linear fit over the 15%-85% profile edge
		xvals = []
		yvals = []
		xvals = self.x[midchan[0]:midchan[1]]  # vel from 15% to 85%
		yvals = self.gaussian_model[midchan[0]:midchan[1]]  # res from 15% to 85%
		errors = np.zeros(len(xvals))

		coef, cov = np.polyfit(xvals, yvals, 1, cov=True)
		sigma = np.sqrt(np.diag(cov))
		intercept = coef[1]  # Unpack the coefficient array
		slope = coef[0]
		inter_err = sigma[1]  # Unpack the uncertainty array
		slope_err = sigma[0]
		yfit = []
		
		maxval = max(yvals)
		halfpeak = 0.5 * maxval
		vel50 = (halfpeak - intercept) / slope

		peak20 = 0.2 * maxval
		vel20 = (peak20 - intercept) / slope

		#variance = 1 / slope ** 2 * (slope_err ** 2 * vel50 ** 2 + inter_err ** 2 + 2 * vel50 * cov[0, 1])
		#Previous maths ignored rms because it was prebaselined for getting the initial setup to work.
		variance = 1 / slope ** 2 * (self.rms ** 2 / 4 + slope_err ** 2 * vel50 ** 2 + inter_err ** 2 + 2 * vel50 * cov[0, 1])
		velerr = np.sqrt(variance)
		
		return vel50, vel20, velerr, coef
		

	
	def calculate_lambda(self, deltav, logSNR):
		"""
		This function is just a straight-up adaptation of Springob 2005 Table 2 into Python code
		We need to check separately for deltav and logSNR
		"""
		if not self.boxcar:
			if deltav < 5.:
				if logSNR < 0.6:
					return 0.005
				if logSNR > 1.1:
					return 0.395
				return 0
			if deltav > 11.:
				if logSNR < .6:
					return .227
				if logSNR > 1.1:
					return 0.533
				return -0.1523 + 0.623 * logSNR
			if logSNR < 0.6:
				return 0.037 * deltav - 0.18
			if logSNR > 1.1:
				return 0.023 * deltav + 0.28
			return (0.0527 * deltav - 0.732) + (-0.027 * deltav + 0.92) * logSNR
		if self.boxcar:
			if deltav < 5.:
				if logSNR < 0.6:
					return 0.020
				if logSNR > 1.1:
					return 0.430
				return -0.4705 + 0.820 * logSNR
			if deltav > 11.:
				if logSNR < .6:
					return 0.332
				if logSNR > 1.1:
					return 0.802
				return -0.2323 + 0.940 * logSNR
			if logSNR < 0.6:
				return 0.052 * deltav - 0.24
			if logSNR > 1.1:
				return 0.062 * deltav + 0.12
			return (0.0397 * deltav - 0.669) + (0.020 * deltav + 0.72) * logSNR
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="A gaussian filter for galaxy spectra")
	parser.add_argument('filename', metavar='AGC', type=int, help="AGC number of the galaxy, eg. 104365")
	parser.add_argument('-smo', metavar='smooth', type=int, help="Value for smoothing the spectrum. A Hanning smooth will always occur, but if an integer is specified, the spectrum will be boxcared by that amount")
	
	args = parser.parse_args()
	if args.smo is None:
		args.smo = 21
	b = baseline.Baseline(args.filename, smooth_int=args.smo, noconfirm=True)
	vel, spec, rms = b()
	ManyGauss(args.filename, smooth_int=args.smo, vel=vel, spec=spec, rms=rms)