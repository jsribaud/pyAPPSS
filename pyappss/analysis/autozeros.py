import numpy as np
from bisect import bisect, bisect_left
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel

def auto_zero(vel, spec, clipped=False):
    length_initial = len(vel)
    # Define new, shortened arrays that aid in convolution and excise the edge effects that often appear
    if clipped == False:
        x = []
        y = []
        for i in range(999, length_initial-1001):
            x.append(vel[i])
            y.append(spec[i])
    elif clipped == True:
        x = vel
        y = spec
            
    gauss_kernel = Gaussian1DKernel(10)
    convolved = convolve(y, gauss_kernel)

    length = len(x)

    centervel = x[round(length/2) - 1]

    point1_1 = bisect_left(x, (centervel - 1000) )
    point1_2 = bisect(x, (centervel - 500) )
    point2_1 = bisect_left(x, (centervel + 500) )
    point2_2 = bisect(x, (centervel + 1000) )

    rms_1 = np.std(convolved[point1_1:point1_2])
    rms_2 = np.std(convolved[point2_1:point2_2])

    rms_avg = (rms_1 +rms_2)/2
    positions = []
    shortvel = []
    shortspec = []

    rms_full = np.std(convolved)

    centerchan = bisect_left(x, centervel)
    # Searches between 1000 km/s in either direction. Looks for 5 channel wide chunks, where the mean is greater than 3 * rms
    # Does this because we can't necessarily be precisely certain with our resolution, and additionally this minimizes the impact of noise
    # Using the convolved spectrum does likewise
    for i in range(point1_1, point2_2):

        tinyspec = []
        for j in range (i-2, i+2):
            tinyspec.append(convolved[j])
            #truth = np.all(tinyspec > 4*rms_avg)
            spec_mean = np.mean(tinyspec)
            #if truth:
            if spec_mean > 3*rms_avg:
                positions.append(i)
                shortvel.append(x[i])
                shortspec.append(convolved[i])

    shortspec = np.asarray(shortspec)
    shortvel = np.asarray(shortvel)

    # snappy channel 'width' definition
    deltav =  ((shortvel[2] - shortvel[0])/2 + (shortvel[len(shortvel)-1] - shortvel[len(shortvel)-3] )/2 ) / 2

    # If the problem of lone-ish peaks were to arise and be significant, the solution would go here.
    # In my testing, using the convolved spectrum eliminated this issue, even for low S/N profiles.

    


    line_spec = []
    line_vel = []
    for i in range(0, 29):
        line_spec.append(shortspec[i])
        line_vel.append(shortvel[i])

    fit_l = np.polyfit(line_vel, line_spec, 1)


    line_spec = []
    line_vel = []
    for i in range(len(shortvel)-30, len(shortvel)-1):
        line_spec.append(shortspec[i])
        line_vel.append(shortvel[i])

    fit_r = np.polyfit(line_vel, line_spec, 1)

        # The first coefficient is slope, and the second is y intercept. As a result, will need to calculate an x intercept.

    x_int_l = - (fit_l[1] / fit_l[0])
    x_int_r = - (fit_r[1] / fit_r[0])
        
    return x_int_l, x_int_r
