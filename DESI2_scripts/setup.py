#!/usr/bin/env python
#
# Script to generate inputs for a survey, using 
# redshift bins of width \Delta z = 0.2.
#
import sys
from math import ceil

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





def make_forecast(cosmo,sfb,task,fsky):
    """Generate an appropriate forecast instance."""
    # Load fiducial linear bias/number density from table
    # The table file name should be the survey file basename
    # with extension ".txt".
    zs, bs, ns = np.loadtxt(sfb+".txt").T
    #
    # Assume zs spans the full survey, but round the
    # bin edges to 0.1.
    zmin = 0.1*int(ceil(zs[0]*10))
    zmax = 0.1*int(zs[-1]*10)
    zedg = np.arange(zmin,zmax+0.2,0.2)
    #
    # Interpolate the bias and number density
    b = interp1d(zs,bs)
    n = interp1d(zs,ns)
    # Set up the experiment object.
    exp = experiment(zedges=zedg,fsky=fsky,b=b,n=n)
    #
    if task == 'setup':
        setup = True
    else:
        setup = False
    # Generate the fisherForecast object with directory set by
    # the survey file basename and properties read from the text file.
    forecast = fisherForecast(experiment=exp,cosmo=cosmo,name=sfb,setup=setup)
    return(forecast)




def do_task(sfb,task,fsky):
    """Does the work, performing task "task" on survey file base name "sfb"."""
    # When taking derivatives of P(k,mu) you don't need to get the lensing
    # Cell's from class. So to speed things up we'll use a different CLASS
    # object depending on the derivatives being calculated. 
    if task == 'setup':
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
       forecast = make_forecast(cosmo,sfb,task,fsky)
       #
    elif task == 'rec':
       params = {'output': 'mPk'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class() 
       cosmo.set(params) 
       cosmo.compute() 
       forecast = forecast = make_forecast(cosmo,sfb,task,fsky)
       basis = np.array(['alpha_perp','alpha_parallel','b'])
       forecast.recon = True
       forecast.marg_params = basis
       forecast.compute_derivatives(five_point=False)
       forecast.recon = False
       #
    elif task == 'fs':
       params = {'output': 'mPk'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class() 
       cosmo.set(params) 
       cosmo.compute() 
       forecast = forecast = make_forecast(cosmo,sfb,task,fsky)
       basis = np.array(['log(A_s)','N','alpha0','b','b2','bs','N2','N4','alpha2','alpha4'])
       forecast.marg_params = basis
       forecast.compute_derivatives(five_point=False)
       # 
    elif task == 'lens':
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
       forecast = forecast = make_forecast(cosmo,sfb,task,fsky)
       basis = np.array(['log(A_s)','N','alpha0','b','b2','bs','alphax'])
       forecast.marg_params = basis
       forecast.compute_Cl_derivatives(five_point=False)
    else:
        raise RuntimeError("Unknown task "+str(task))
    #





if __name__=="__main__":
    if len(sys.argv)==3:
        fsky = 0.5
    elif len(sys.argv)==4:
        fsky = float(sys.argv[3])
    else:
        outstr = "Usage: "+sys.argv[0]+" <survey-filename> <task-name> [fsky=0.5]"
        raise RuntimeError(outstr)
    # Extract the arguments.
    filebase = sys.argv[1]
    task     = sys.argv[2]
    # Do the actual work.
    do_task(filebase,task,fsky)
    #
