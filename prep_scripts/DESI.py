# import all revelant packages
import os, sys
os.chdir('/home/noah/Berkeley/fishlss/')
sys.path.append('/home/noah/Berkeley/fishlss/')
from headers import *

Nk,Nmu,nbins,P = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4].lower() == 'true'
C,Pr,EDE,Alin = sys.argv[5].lower() == 'true',sys.argv[6].lower() == 'true',sys.argv[7].lower() == 'true',sys.argv[8].lower() == 'true'

# Using halofit to estimate the nonlinear matter power spectrum
desiParams = {
          'output': 'tCl lCl mPk',
          'modes': 's',
          'l_max_scalars': 5000,
          'lensing': 'yes',
          'P_k_max_h/Mpc': 40.,
          'non linear':'halofit', 
          'z_pk': '0.0,10',
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

desiCosmo = Class()
desiCosmo.set(desiParams)
desiCosmo.compute()

DESI = experiment(zmin=0.6, zmax=1.7, nbins=nbins, fsky=0.5, sigma_z=0.0, ELG=True)

desiCast = fisherForecast(experiment=DESI,cosmo=desiCosmo,kmin=5.e-4,\
                          kmax=1.,Nk=Nk,Nmu=Nmu,velocileptors=True,name='DESI')

print('Finished setting up '+desiCast.name)


desiCast.marg_params = np.array(['h','log(A_s)','omega_cdm','n_s','omega_b','tau_reio','b','N',\
                                 'm_ncdm','N_ur','alpha_s','b2',\
                             'bs','alpha0','alpha2','alpha4','alpha6','N2','N4',\
                                 'Omega_k','f','alpha_perp','alpha_parallel','f_NL'])

if P: 
   desiCast.compute_derivatives()
   print(desiCast.name+': calculated P(k,mu) derivatives')

desiCast.marg_params = np.array(['h','log(A_s)','omega_cdm','n_s','omega_b','tau_reio','b',\
                                 'b2','bs','alpha0','alphax','N','gamma','m_ncdm','N_ur','alpha_s','Omega_k'])
                                 
if C: 
   desiCast.compute_Cl_derivatives()
   print(desiCast.name+': calculated Cl derivatives')

desiCast.recon = True
desiCast.marg_params = np.array(['alpha_perp','alpha_parallel','b','N'])
if Pr:
   desiCast.compute_derivatives()
   print(desiCast.name+': calculated P_recon derivatives')
desiCast.recon = False

desiCast.marg_params = np.array(['fEDE'])
log10z_cs = np.linspace(1.5,6.5,20)
if EDE:
   for log10z_c in log10z_cs:
      desiCast.log10z_c = log10z_c
      desiCast.compute_derivatives()
   print(desiCast.name+': calculated EDE derivatives')
   
desiCast.marg_params = np.array(['A_lin'])
omega_lins = np.linspace(10,500,20)
if Alin:
   for omega_lin in omega_lins:
      desiCast.omega_lin = omega_lin
      desiCast.compute_derivatives()
   print(desiCast.name+': calculated Alin derivatives')