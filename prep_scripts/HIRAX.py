# import all revelant packages
import os, sys
os.chdir('/home/noah/Berkeley/fishlss/')
sys.path.append('/home/noah/Berkeley/fishlss/')
from headers import *

Nk,Nmu,nbins,P = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4].lower() == 'true'
C,Pr,EDE,Alin = sys.argv[5].lower() == 'true',sys.argv[6].lower() == 'true',sys.argv[7].lower() == 'true',sys.argv[8].lower() == 'true'

hiraxParams = {
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

hiraxCosmo = Class()
hiraxCosmo.set(hiraxParams)
hiraxCosmo.compute()

HIRAX = experiment(zmin=0.8, zmax=2.5, nbins=nbins, fsky=0.36, sigma_z=0.0, HI=True, fill_factor=1, 
                   tint=4, Ndetectors=1e3,sigv=10) 
                   
                   
hiraxCast = fisherForecast(experiment=HIRAX,cosmo=hiraxCosmo,kmin=5.e-4,\
                          kmax=1.,Nk=Nk,Nmu=Nmu,velocileptors=True,name='HIRAX')
                          
print('Finished setting up '+hiraxCast.name)

hiraxCast.marg_params = np.array(['h','log(A_s)','omega_cdm','n_s','omega_b','tau_reio','b','N',\
                                 'm_ncdm','N_ur','alpha_s','b2',\
                             'bs','alpha0','alpha2','alpha4','alpha6','N2','N4',\
                                 'Omega_k','f','alpha_perp','alpha_parallel','f_NL','Tb'])
if P: 
   hiraxCast.compute_derivatives() 
   print(hiraxCast.name+': calculated P(k,mu) derivatives')                                
                                 
hiraxCast.marg_params = np.array(['h','log(A_s)','omega_cdm','n_s','omega_b','tau_reio','b',\
                                 'b2','bs','alpha0','alphax','N','gamma','m_ncdm','N_ur','alpha_s','Omega_k'])
                                 
if C: 
   hiraxCast.compute_Cl_derivatives()   
   print(hiraxCast.name+': calculated Cl derivatives')        

hiraxCast.recon = True
hiraxCast.marg_params = np.array(['alpha_perp','alpha_parallel','b','N'])
if Pr: 
   hiraxCast.compute_derivatives()
   print(hiraxCast.name+': calculated P_recon derivatives')        
hiraxCast.recon = False

hiraxCast.marg_params = np.array(['fEDE'])
log10z_cs = np.linspace(1.5,6.5,20)
if EDE:
   for log10z_c in log10z_cs:
      hiraxCast.log10z_c = log10z_c
      hiraxCast.compute_derivatives()  
   print(hiraxCast.name+': calculated EDE derivatives')                          
   
hiraxCast.marg_params = np.array(['A_lin'])
omega_lins = np.linspace(10,500,20)
if Alin:
   for omega_lin in omega_lins:
      hiraxCast.omega_lin = omega_lin
      hiraxCast.compute_derivatives() 
   print(hiraxCast.name+': calculated Alin derivatives')                 