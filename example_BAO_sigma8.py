import sys
typ = str(sys.argv[1])

# import revelant packages
from headers import *
from twoPoint import *
from twoPointNoise import *


def make_forecast(cosmo):
   # load fiducial linear bias/number density from table
   zs, bs, ns = np.loadtxt('example_survey_info.txt').T

   # assume zs spans the full survey
   zmin = zs[0]
   zmax = zs[-1]

   # interpolate
   b = interp1d(zs,bs)
   n = interp1d(zs,ns)

   nbins = 3
   fsky = 0.5

   exp = experiment(zmin=zmin, zmax=zmax, nbins=nbins, fsky=fsky, b=b, n=n)

   setup = False
   if typ == 'setup': setup = True

   forecast = fisherForecast(experiment=exp,cosmo=cosmo,name='example',setup=setup)
   return forecast


# when taking derivatives of P(k,mu) you don't need to get the lensing
# Cell's from class. So to speed things up we'll use a different CLASS
# object depending on the derivatives being calculated. 

if typ == 'setup':
   params = {'output': 'tCl lCl mPk',
          'l_max_scalars': 1000,
          'lensing': 'yes',
          'P_k_max_h/Mpc': 2.,
          'non linear':'halofit', 
          'z_pk': '0.0,6',
          'A_s': 2.10732e-9,
          'n_s': 0.96824,
          'alpha_s': 0.,
          'h': 0.6770,
          'N_ur': 1.0196,
          'N_ncdm': 2,
          'm_ncdm': '0.01,0.05',
          'tau_reio': 0.0568,
          'omega_b': 0.02247,
          'omega_cdm': 0.11923,
          'Omega_k': 0.}

   cosmo = Class() 
   cosmo.set(params) 
   cosmo.compute() 
   forecast = make_forecast(cosmo)

if typ == 'rec':
   params = {'output': 'mPk',
          'P_k_max_h/Mpc': 2.,
          'z_pk': '0.0,6',
          'A_s': 2.10732e-9,
          'n_s': 0.96824,
          'alpha_s': 0.,
          'h': 0.6770,
          'N_ur': 1.0196,
          'N_ncdm': 2,
          'm_ncdm': '0.01,0.05',
          'tau_reio': 0.0568,
          'omega_b': 0.02247,
          'omega_cdm': 0.11923,
          'Omega_k': 0.}
   cosmo = Class() 
   cosmo.set(params) 
   cosmo.compute() 
   forecast = forecast = make_forecast(cosmo)
   basis = np.array(['alpha_perp','alpha_parallel','b'])
   forecast.recon = True
   forecast.marg_params = basis
   forecast.compute_derivatives()
   forecast.recon = False
   
if typ == 'fs':
   params = {'output': 'mPk',
          'P_k_max_h/Mpc': 2.,
          'z_pk': '0.0,6',
          'A_s': 2.10732e-9,
          'n_s': 0.96824,
          'alpha_s': 0.,
          'h': 0.6770,
          'N_ur': 1.0196,
          'N_ncdm': 2,
          'm_ncdm': '0.01,0.05',
          'tau_reio': 0.0568,
          'omega_b': 0.02247,
          'omega_cdm': 0.11923,
          'Omega_k': 0.}
   cosmo = Class() 
   cosmo.set(params) 
   cosmo.compute() 
   forecast = forecast = make_forecast(cosmo)
   basis = np.array(['log(A_s)','N','alpha0','b','b2','bs','N2','N4','alpha2','alpha4'])
   forecast.marg_params = basis
   forecast.compute_derivatives()
   
if typ == 'lens':
   params = {'output': 'tCl lCl mPk',
          'l_max_scalars': 1000,
          'lensing': 'yes',
          'P_k_max_h/Mpc': 2.,
          'non linear':'halofit', 
          'z_pk': '0.0,6',
          'A_s': 2.10732e-9,
          'n_s': 0.96824,
          'alpha_s': 0.,
          'h': 0.6770,
          'N_ur': 1.0196,
          'N_ncdm': 2,
          'm_ncdm': '0.01,0.05',
          'tau_reio': 0.0568,
          'omega_b': 0.02247,
          'omega_cdm': 0.11923,
          'Omega_k': 0.}

   cosmo = Class() 
   cosmo.set(params) 
   cosmo.compute() 
   forecast = forecast = make_forecast(cosmo)
   basis = np.array(['log(A_s)','N','alpha0','b','b2','bs','alphax'])
   forecast.marg_params = basis
   forecast.compute_Cl_derivatives()
