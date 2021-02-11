#!/bin/bash
# Nk  Nmu nbins P  C  Pr  EDE  Alin
python DESI.py                 2000 100 3 False True False False False & 
python euclid.py               2000 100 4 False True False False False &
python MegaMapper_fiducial.py  2000 100 3 False True False False False &
python MSE.py                  2000 100 3 False True False False False &
python HIRAX.py                2000 100 3 False True False False False &
python MegaMapper_like.py      2000 100 3 False True False False False &
python PUMA5K.py               2000 100 3 False True False False False &
python PUMA32K.py              2000 100 3 False True False False False
wait
