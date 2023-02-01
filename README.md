[![](https://img.shields.io/badge/arXiv-2106.09713%20-red.svg)](https://arxiv.org/abs/2106.09713)

## Fisher forecasting for Large Scale Structure surveys

![Fishing astro](https://github.com/NoahSailer/FishLSS/blob/master/figures/fishing_astro.jpg)

If you make use of this code please cite [(Sailer et al. 2021)](https://arxiv.org/abs/2106.09713).

-------

**Requirements:** the usual [anaconda](https://www.anaconda.com/products/distribution) stuff, [pyFFTW](https://hgomersall.github.io/pyFFTW/), [velocileptors](https://github.com/sfschen/velocileptors), and [CLASS](https://github.com/lesgourg/class_public). To forecast sensitivity to Early Dark Energy one can optionally use [CLASS_EDE](https://github.com/mwt5345/class_ede) instead of the vanilla CLASS (and similarly for any other modified version of CLASS).

**Installation:** ```git clone https://github.com/NoahSailer/FishLSS``` 

At the moment all code and notebooks should be run from the ```FishLSS/``` directory, sorry about that! Making the code executable from anywhere is on my todo list.

-------

**What is ``FishLSS`` good for?**

Fisher forecasting codes compute the Fisher information matrix for a set of **observables** and model **parameters**. ``FishLSS`` is set up to model the following observables:

- the redshift-space power spectrum of any biased tracer of the CDM+baryon field <img src="https://latex.codecogs.com/svg.latex?P_{gg}(k,\mu)" /> 

- the post-reconstruction galaxy power spectrum

- the projected cross-correlation of galaxies with the CMB lensing convergence <img src="https://latex.codecogs.com/svg.latex?C^{\kappa g}_\ell" />, the projected galaxy power spectrum <img src="https://latex.codecogs.com/svg.latex?C^{g g}_\ell" />, and the CMB lensing convergence power spectrum <img src="https://latex.codecogs.com/svg.latex?C^{\kappa\kappa}_\ell" />

With the exception of <img src="https://latex.codecogs.com/svg.latex?C^{\kappa\kappa}_\ell" />, which is modeled with ```HaloFit```, all of these observables are modeled self-consistently using 1-loop Lagrangian perturbation theory (i.e. ```velocileptors```).

```FishLSS``` can compute the derivatives, and hence the Fisher information from the observables listed above, with respect to the following sets of parameters:

- any standard ```CLASS``` input <img src="https://latex.codecogs.com/svg.latex?(\omega_{b},n_s,A_s,\cdots)" />, or any extra parameters introduced by a modified version of ```CLASS``` (e.g. the maximum amplitude <img src="https://latex.codecogs.com/svg.latex?f_\text{EDE}" /> of Early Dark Energy using ```CLASS_EDE```)

- Bias parameters, counterterms and stochastic contributions <img src="https://latex.codecogs.com/svg.latex?(b,\,b_2,\,b_s,\,\alpha_{2n},\,\alpha_x,\,N_{2n})" /> 

- The fixed-template BAO parameters <img src="https://latex.codecogs.com/svg.latex?(\alpha_\parallel,\alpha_\perp)" /> 

- (linear) primordial features <img src="https://latex.codecogs.com/svg.latex?(A_\text{lin},\,\omega_\text{lin},\,\phi_\text{lin})" /> 

- Local <img src="https://latex.codecogs.com/svg.latex?f_\text{NL}"/>  through its effect on scale dependent bias, assuming the tenuous <img src="https://latex.codecogs.com/svg.latex?b_\phi = 2 \delta_c(b-1)"/> relationship

-------

**Code structure**

```experiment.py``` is used to define the experiment object (redshift range, sky coverage, linear bias, number density, etc.). ```twoPoint.py``` contains all the code relevant for computing power spectra, while ```twoPointNoise.py``` contains all the code relevant for computing covariance matrices. ```fisherForecast.py``` is used to define a forecast object, compute derivatives and Fisher matrices. ```castorina.py``` contains the assumed evolution of the 21cm linear bias. ```headers.py``` is a shorthand piece of code to import everything that one needs to import in e.g. a Jupyter notebook to run ```FishLSS``` and to make aesthetically-pleasing plots (importing ```headers.py``` may cause LaTeX errors when plotting).

CMB fisher matrices, CMB lensing noise curves, and the assumed fiducial reionization history can be found in ```input/```. The directory ```bao_recon/``` contains code to compute the reconstructed power spectrum, which is wrapped in ```twoPoint.py```. 


-------

**Quickstart example:** below is a code snippet which creates forecasting object, calculate derivatives of the redshift-space galaxy power spectrum, and generate a Fisher matrix. See ```notebooks/``` for more detailed examples.

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
