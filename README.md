[![](https://img.shields.io/badge/arXiv-2106.09713%20-red.svg)](https://arxiv.org/abs/2106.09713)

## Fisher forecasting for Large Scale Structure surveys (FishLSS)

![Fishing astro](https://github.com/NoahSailer/FishLSS/blob/master/figures/fishing_astro.jpg)

If you make use of this code please cite [(Sailer et al. 2021)](https://inspirehep.net/literature/1869110).

-------

Requirements: numpy, scipy, [pyFFTW](https://hgomersall.github.io/pyFFTW/), [velocileptors](https://github.com/sfschen/velocileptors), and [CLASS](https://github.com/lesgourg/class_public). To forecast sensitivity to Early Dark Energy one can optionally install [CLASS_EDE](https://github.com/mwt5345/class_ede). (and similarly for any other modified version of CLASS)

-------

FishLSS calculates derivatives of both the redshift-space galaxy power spectrum and CMB 
lensing auto/cross spectrum with repsect to any CLASS parameter, in addition to

(1) Bias parameters, counterterms and stochastic contributions (<img src="https://render.githubusercontent.com/render/math?math=b">, <img src="https://render.githubusercontent.com/render/math?math=b_2">, <img src="https://render.githubusercontent.com/render/math?math=b_s">, <img src="https://render.githubusercontent.com/render/math?math=\alpha_{2n}">, <img src="https://render.githubusercontent.com/render/math?math=\alpha_{x}">, <img src="https://render.githubusercontent.com/render/math?math=N_{2n}">)

(2) (linear) Primordial features (<img src="https://render.githubusercontent.com/render/math?math=A_\text{lin}">, <img src="https://render.githubusercontent.com/render/math?math=\omega_\text{lin}">, <img src="https://render.githubusercontent.com/render/math?math=\phi_\text{lin}">)

(3) Primordial non-Gaussinity (local <img src="https://render.githubusercontent.com/render/math?math=f_\text{NL}"> through its effect on scale dependent bias, assuming <img src="https://render.githubusercontent.com/render/math?math=b_\phi = 2 \delta_c(b-1)">)

-------

Below is a very quick example showing how to create a forecasting object, calculate derivatives of the redshift-space galaxy power spectrum, and generate a Fisher matrix. See notebooks/ for more examples.
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

# Define an experiment. This one observes LBGs (i.e. an idealized MegaMapper survey) 
# from 2 < z < 5, and we split the sample into three z-bins
exp = experiment(zmin=2., zmax=5., nbins=3, fsky=0.34, LBG=True)

# Create the forecast object. 
fishcast = fisherForecast(experiment=exp,cosmo=cosmo,name='Example')
                          
# Specify which derivatives to compute 
basis = np.array(['h','log(A_s)','n_s','omega_cdm','b'])
fishcast.free_params = basis

# Compute derivatives, automatically saves to output/Example/derivatives
fishcast.compute_derivatives()

# Compute (small) Fisher matrix with kmax = knl (k_{non-linear})
fisher_basis = np.array(['h','n_s'])
F = fishcast.gen_fisher(fisher_basis, kmax_knl = 1)
```
