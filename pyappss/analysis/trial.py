# Temporary


def trial(file):
    import numpy as np
    from astropy.io import ascii

    tab = ascii.read(file)
    print(tab.colnames)
