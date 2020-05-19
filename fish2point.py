import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower


###############################################################
nGal = 12.       # number density of galaxies [1/Mpc^3]. From Fletcher 1946, which I'm sure is outdated.
vSurvey = 5.5e11 # volume of the survey [Mpc^3]. From http://burro.astr.cwru.edu/Academics/Astr328/HW/HW1/vol_h070_OM3_OL7.dat, assuming a redshift range 0 < z < 2.
beta = 0.8       # linear redshift distortion. This value is not physically accurate, I'm just setting it as a constant for debugging purposes.
b = 0.9          # galaxy bias. This is not accurate, again I'm just choosing a value for debugging.
minkh = 1e-4
maxkh = 0.25

def calculate_galaxy_power_spectrum(z=1., H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965, minkh=minkh, maxkh=maxkh):
   '''
   Returns the galaxy power spectrum at redshft z with the specified
   cosmological parameters. 
   '''
   print('Calculating galaxy power spectrum')
   pars = camb.CAMBparams()
   pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
   pars.InitPower.set_params(ns=ns)
   # Note non-linear corrections couples to smaller scales than you want
   pars.set_matter_power(redshifts=[z], kmax=2.0)
   # Non-Linear spectra (Halofit)
   pars.NonLinear = model.NonLinear_both
   #
   results = camb.get_results(pars)
   results.calc_power_spectra(pars)
   kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints = 200)
   return kh_nonlin, b*pk_nonlin[0]


def compute_dlnPdp(dH0, dns, k, p):
   ''' returns the derivative of the matter power
   spectrum with respect to the input parameters H0 and ns.
   In this derivative I'm treating (1+ beta * mu^2) as a 
   constant.
   '''
   print('Calculating derivatives')
   dlnP1 = np.log(calculate_galaxy_power_spectrum(H0=67.5*(1.+dH0))[1]) - np.log(p)
   dlnPdp1 = dlnP1 / (67.5*dH0)
   dlnP2 = np.log(calculate_galaxy_power_spectrum(ns=0.965*(1.+dns))[1]) - np.log(p)
   dlnPdp2 = dlnP2 / (0.965*dns)
   dlnPdp = np.array([dlnPdp1,dlnPdp2])
   return dlnPdp


def compute_Fisher_matrix(dH0=0.01,dns=0.01):
   '''
   Computes the Fisher matrix for the parameters
   H0 and ns. dH0 and dns are the fractional changes
   to the fidicual values used to estimate the logarithmic
   derivatives of the power spectrum.
   '''
   k,p = calculate_galaxy_power_spectrum()
   dlnPdp = compute_dlnPdp(dH0,dns,k,p)
   # Interpolate the derivarives for integration
   dlnPdp1 = scipy.interpolate.interp1d(k, dlnPdp[0], kind='linear', bounds_error=False, fill_value=0.)
   dlnPdp2 = scipy.interpolate.interp1d(k, dlnPdp[1], kind='linear', bounds_error=False, fill_value=0.)
   # Interpolate the effective volume for integration
   P = scipy.interpolate.interp1d(k, p, kind='linear', bounds_error=False, fill_value=0.)
   p = lambda k,mu: P(k) * (1.+beta*mu**2.)**2.
   vEff = lambda k,mu: vSurvey * ((nGal * p(k,mu)) / (nGal * p(k,mu) + 1.))**2.
   # Integrate
   integrand11 = lambda k,mu: vEff(k,mu)*(dlnPdp1(k)**2.)*(k**2.)/(8.*np.pi**2.)
   F11 = integrate.dblquad(integrand11, -1., 1., lambda mu: minkh, lambda mu: maxkh)[0] 
   integrand12 = lambda k,mu: vEff(k,mu)*dlnPdp1(k)*dlnPdp2(k)*(k**2.)/(8.*np.pi**2.)
   F12 = integrate.dblquad(integrand12, -1., 1., lambda mu: minkh, lambda mu: maxkh)[0]
   integrand22 = lambda k,mu: vEff(k,mu)*(dlnPdp2(k)**2.)*(k**2.)/(8.*np.pi**2.)
   F22 = integrate.dblquad(integrand22, -1., 1., lambda mu: minkh, lambda mu: maxkh)[0]
   # Cast into matrix form
   F = np.array([[F11,F12],
                 [F12,F22]])
   return F


F = compute_Fisher_matrix()
Finv = np.linalg.inv(F)

print('Relative error on H0:', np.sqrt(Finv[0][0]) / 67.5)
print('Relative error on ns:', np.sqrt(Finv[1][1]) / 0.965)
