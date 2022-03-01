#!/usr/bin/env python
#
# Script to generate inputs needed for Fisher forecasts of
# BAO and RSD performance of specified surveys.
# This is the "driver" used by the "BAO_sigma8" Jupyter
# notebook.
#
import sys

# Import revelant packages
from headers       import *
from twoPoint      import *
from twoPointNoise import *


# Set the default cosmological parameters.
default_cosmo = {'A_s': 2.10732e-9,\
                 'n_s': 0.96824,\
                 'alpha_s': 0.,\
                 'h': 0.6770,\
                 'N_ur': 1.0196,\
                 'N_ncdm': 2,\
                 'm_ncdm': '0.01,0.05',\
                 'tau_reio': 0.0568,\
                 'omega_b': 0.02247,\
                 'omega_cdm': 0.11923,\
                 'Omega_k': 0.,\
                 'P_k_max_h/Mpc': 2.,\
                 'z_pk': '0.0,6'}



def make_forecast(cosmo,survey_filebase,nbins=3,fsky=0.5):
    """Generate an appropriate forecast instance."""
    # Load fiducial linear bias/number density from table
    # The table file name should be the survey file basename
    # with extension ".txt".
    zs, bs, ns = np.loadtxt(survey_filebase+".txt").T
    #
    # Assume zs spans the full survey
    zmin = zs[0]
    zmax = zs[-1]
    #
    # Interpolate the bias and number density
    b = interp1d(zs,bs)
    n = interp1d(zs,ns)
    #
    exp = experiment(zmin=zmin,zmax=zmax,nbins=nbins,fsky=fsky,b=b,n=n)
    #
    if typ == 'setup':
        setup = True
    else:
        setup = False
    # Generate the fisherForecast object with directory set by
    # the survey file basename and properties read from the text file.
    forecast = fisherForecast(experiment=exp,cosmo=cosmo,name=sfb,setup=setup)
    return forecast




def do_task(sfb,typ,nbins,fsky):
    """Does the work, performing task "typ" on survey file base name "sfb"."""
    # When taking derivatives of P(k,mu) you don't need to get the lensing
    # Cell's from class. So to speed things up we'll use a different CLASS
    # object depending on the derivatives being calculated. 
    if typ == 'setup':
       params = {'output': 'tCl lCl mPk',\
                 'non linear':'halofit', \
                 'l_max_scalars': 1000,\
                 'lensing': 'yes'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class() 
       cosmo.set(params) 
       cosmo.compute() 
       forecast = make_forecast(cosmo,sfb,nbins,fsky)
       #
    elif typ == 'rec':
       params = {'output': 'mPk'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class() 
       cosmo.set(params) 
       cosmo.compute() 
       forecast = forecast = make_forecast(cosmo,sfb,nbins,fsky)
       basis = np.array(['alpha_perp','alpha_parallel','b'])
       forecast.recon = True
       forecast.marg_params = basis
       forecast.compute_derivatives()
       forecast.recon = False
       #
    elif typ == 'fs':
       params = {'output': 'mPk'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class() 
       cosmo.set(params) 
       cosmo.compute() 
       forecast = forecast = make_forecast(cosmo,sfb,nbins,fsky)
       basis = np.array(['log(A_s)','N','alpha0','b','b2','bs','N2','N4','alpha2','alpha4'])
       forecast.marg_params = basis
       forecast.compute_derivatives()
       # 
    elif typ == 'lens':
       params = {'output': 'tCl lCl mPk',\
              'l_max_scalars': 1000,\
              'lensing': 'yes',\
              'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class() 
       cosmo.set(params) 
       cosmo.compute() 
       forecast = forecast = make_forecast(cosmo,sfb,nbins,fsky)
       basis = np.array(['log(A_s)','N','alpha0','b','b2','bs','alphax'])
       forecast.marg_params = basis
       forecast.compute_Cl_derivatives()
    else:
        raise RuntimeError("Unknown task "+str(typ))
    #





if __name__=="__main__":
    if len(sys.argv)<3:
        outstr = "Usage: "+sys.argv[0]+" <survey-filename> <task-name> [nbins=3] [fsky=0.5]"
        raise RuntimeError(outstr)
    if len(sys.argv)==3:
        nbins,fsky = 3,0.5
    if len(sys.argv)==4:
        nbins,fsky = int(sys.argv[3]),0.5
    if len(sys.argv)==5:
        nbins,fsky = int(sys.argv[3]),float(sys.argv[4])
    # Extract the arguments.
    sfb = sys.argv[1]
    typ = sys.argv[2]
    # Do the actual work.
    do_task(sfb,typ,nbins,fsky)
    #
