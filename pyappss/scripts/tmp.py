#!/usr/bin/env python
"""
Build CASBAH bits and pieces
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import pdb

try:  # Python 3
    ustr = unicode
except NameError:
    ustr = str

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Build parts of the CASBAH database; Output_dir = $CASBAH_GALAXIES [v1.1]')
    parser.add_argument("items", type=str, help="One or more things to build ['galaxies', 'specdb']")
    parser.add_argument("--sdss", default=False, action="store_true", help="Build SDSS bits and pieces too?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import warnings
    from pyappss import tmp

    if pargs.items == 'go':
        tmp.tmp()

