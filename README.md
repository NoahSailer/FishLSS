[![](https://img.shields.io/badge/arXiv-2106.09713%20-red.svg)](https://arxiv.org/abs/2106.09713)

## Fisher forecasting for Large Scale Structure surveys

![Fishing astro](https://github.com/NoahSailer/FishLSS/blob/master/figures/fishing_astro.jpg)

If you make use of this code please cite [this paper](https://arxiv.org/abs/2106.09713).

-------

**Requirements:** in addition to the usual [anaconda](https://www.anaconda.com/products/distribution) stuff, ```FishLSS``` requires [pyFFTW](https://hgomersall.github.io/pyFFTW/), [velocileptors](https://github.com/sfschen/velocileptors) and [CLASS](https://github.com/lesgourg/class_public), which can be installed by running
```
pip install pyFFTW
pip install -v git+https://github.com/sfschen/velocileptors
git clone https://github.com/lesgourg/class_public
cd class_public
make clean
make
``` 

To forecast sensitivity to Early Dark Energy one can optionally install [CLASS_EDE](https://github.com/mwt5345/class_ede) instead of the vanilla ```CLASS``` (and similarly for any other modified version of ```CLASS```).

**Installation:** ```FishLSS``` is now pip-installable! Just run
```
pip install -v git+https://github.com/NoahSailer/FishLSS
```
-------

**What is ``FishLSS`` good for?**

Fisher forecasting codes compute the Fisher information matrix for a set of **observables** and model **parameters**. ``FishLSS`` is set up to model the following observables:

- the redshift-space power spectrum of any biased tracer of the CDM+baryon field 

- the post-reconstruction galaxy power spectrum

- the projected cross-correlation of galaxies with the CMB lensing convergence, the projected galaxy power spectrum, and the CMB lensing convergence power spectrum

With the exception of the CMB lensing convergence power spectrum, which is modeled with ```HaloFit```, all of these observables are modeled self-consistently using 1-loop Lagrangian perturbation theory (i.e. ```velocileptors```).

```FishLSS``` can compute the derivatives, and hence the Fisher information from the observables listed above, with respect to the following sets of parameters:

- any standard ```CLASS``` input, or any extra parameters introduced by a modified version of ```CLASS``` (e.g. the maximum amplitude of Early Dark Energy when running ```CLASS_EDE```)

- bias parameters, counterterms and stochastic contributions

- the fixed-template BAO parameters 

- (linear) primordial features 

- local primordial non-Gaussianity through its effect on scale dependent bias

-------

**Quickstart example:** below is a code snippet which creates forecasting object, calculate derivatives of the redshift-space galaxy power spectrum, and generate a Fisher matrix. See ```notebooks/``` for more detailed examples.

```
# import dependencies
import numpy as np
from classy import Class
from FishLSS.fisherForecast import fisherForecast
from FishLSS.experiment import experiment

# create CLASS object
params = {'output': 'mPk lCl','P_k_max_h/Mpc': 40.,'non linear':'halofit', 
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
basis = np.array(['h','log(A_s)','n_s','omega_cdm','b','b_2'])
fishcast.free_params = basis

# Derivatives of P(k,mu), automatically saves to output/Example/derivatives
fishcast.compute_derivatives()

# Compute (small) Fisher matrix with kmax = knl (k_{non-linear})
fisher_basis = np.array(['h','n_s','b'])
globe = 2 # the number of redshift-independent parameters
F = fishcast.gen_fisher(fisher_basis, globe, kmax_knl = 1)
# F is a (2 + 1*3) x (2 + 1*3) matrix in the basis {h,n_s,b(z1),b(z2),b(z3)}
```
