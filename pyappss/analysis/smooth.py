import numpy as np


def smooth(spec, smooth_type=None, yfit= None):
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
        for i in range(int(smooth_type)):  # range
            window.append(1 / float(smooth_type))
        smo = np.convolve(smo, window, 'same')
    res = smo  # allows the plot to reflect the smoothing.
    return res
