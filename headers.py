import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special, optimize, integrate, stats
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, interp1d, interp2d, BarycentricInterpolator
from time import time
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from timeit import timeit
from time import time
from copy import copy
from classy import Class
from experiment import *
from fisherForecast import *
import sys

##################################################################################
# for pretty plots

#from matplotlib import rc
#rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
#rc('mathtext', **{'fontset':'cm'})
#rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{amsmath}, \usepackage{amssymb}')
#rc('legend',**{'fontsize':'18'})
