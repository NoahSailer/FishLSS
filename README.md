# FishLSS

Forecasting code for LSS surveys. Requires numpy, scipy, [pyFFTW](https://hgomersall.github.io/pyFFTW/), and [CLASS_EDE](https://github.com/mwt5345/class_ede): an 
extension of the Boltzmann code [CLASS](https://github.com/lesgourg/class_public) that includes an Early Dark Energy model.


FishLSS calculates derivatives of both the 3D galaxy power spectrum and CMB 
lensing auto/cross spectrum, with repsect to any CLASS_EDE parameter, as well
as several extenions:

(1) Primordial features (<img src="https://render.githubusercontent.com/render/math?math=A_\text{lin}">, <img src="https://render.githubusercontent.com/render/math?math=\omega_\text{lin}">, <img src="https://render.githubusercontent.com/render/math?math=\phi_\text{lin}">)

(2) Non-linear bias parameters (<img src="https://render.githubusercontent.com/render/math?math=b_1">, <img src="https://render.githubusercontent.com/render/math?math=b_2">, <img src="https://render.githubusercontent.com/render/math?math=b_s">) and counterterms (<img src="https://render.githubusercontent.com/render/math?math=\alpha_0">, <img src="https://render.githubusercontent.com/render/math?math=\alpha_2">, <img src="https://render.githubusercontent.com/render/math?math=\alpha_4">)

(3) Primordial non-Gaussinity (local <img src="https://render.githubusercontent.com/render/math?math=f_\text{NL}"> through its effect on the linear bias parameter)

-------

Here's an example showing how to create a forecasting object, calculate derivatives,
and generate a Fisher matrix: 
```
# import dependencies
from headers import *


# create CLASS object
params = {'output': 'mPk','P_k_max_h/Mpc': 40.,'non linear':'halofit', 
          'z_pk': '0.0,10','A_s': 2.10732e-9,'n_s': 0.96824,
          'alpha_s': 0.,'h': 0.6770, 'N_ur': 1.0196,
          'N_ncdm': 2,'m_ncdm': '0.01,0.05','tau_reio': 0.0568,
          'omega_b': 0.02247,'omega_cdm': 0.11923,'Omega_k': 0.}

cosmo = Class()
cosmo.set(params)
cosmo.compute()

# Create and experiment, this one observes LBGs from 2 < z < 5, and we split
# the sample into three z-bins
exp = experiment(zmin=2., zmax=5., nbins=3, fsky=0.34, sigma_z=0.001, LBG=True)

# Create the forecast object. 
fishcast = fisherForecast(experiment=exp,cosmo=cosmo,params=params,
                          khmin=5.e-4,khmax=1.,Nk=1000,Nmu=200,
                          velocileptors=True,name='Example')
                          
# Specify which derivatives to compute 
basis = np.array(['h','log(A_s)','n_s','omega_cdm'])
fishcast.marg_params = basis

# Compute derivatives, automatically saves to output/Example/derivatives
fishcast.compute_derivatives()

# Load derivatives and compute Fisher matrix
second_basis = np.array(['h','n_s'])
derivatives = fishcast.load_derivatives(second_basis)
F = fishcast.gen_fisher(second_basis, kmax_knl = 0.5, derivatives=derivatives)
```
