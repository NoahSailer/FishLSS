from headers import *
from twoPoint import *

'''
Values and defintions from Table 3 of Wilson and White 2019.
'''
zs = np.array([2.,3.,3.8,4.9,5.9])
Muvstar = np.array([-20.60,-20.86,-20.63,-20.96,-20.91])
Muvstar = interp1d(zs, Muvstar, kind='linear', bounds_error=False, fill_value=0.)
phi = np.array([9.70,5.04,9.25,3.22,1.64])
phi = interp1d(zs, phi, kind='linear', bounds_error=False, fill_value=0.)
alpha = np.array([-1.6,-1.78,-1.57,-1.60,-1.87])
alpha = interp1d(zs, alpha, kind='linear', bounds_error=False, fill_value=0.)
muv = np.array([24.2,24.7,25.4,25.5,25.8])
muv = interp1d(zs, muv, kind='linear', bounds_error=False, fill_value=0.)


def compute_covariance_matrix(fishcast):
   '''
   Returns a square array of linear dimension Nk*Nmu. 
   '''
   z = fishcast.experiment.zmid
   prefactor = (4.*np.pi**2.) / (fishcast.dk*fishcast.dmu*fishcast.Vsurvey*fishcast.k**2.)
   if fishcast.experiment.HI: 
      pn = compute_tracer_power_spectrum(fishcast, z)(fishcast.k,fishcast.mu)
      diagonal_values = prefactor * pn**2.
      return np.diag(diagonal_values)
   if fishcast.experiment.LBG: number_density = n(fishcast)
   else: number_density = fishcast.experiment.n
   diagonal_values = prefactor * (fishcast.P_fid + 1./number_density)**2.
   return np.diag(diagonal_values)


def Muv(fishcast):
   '''
   Equation 2.6 of Wilson and White 2019.
   '''
   z = fishcast.experiment.zmid
   result = muv(z) - 5. * np.log10(fishcast.cosmo.luminosity_distance(z)*1.e5)
   result += 2.5 * np.log10(1.+z)
   return result


def n(fishcast):
   '''
   Equation 2.5 of Wilson and White 2019. Return number
   density of LBGs at redshift z in units of Mpc^3/h^3.
   '''
   z = fishcast.experiment.zmid
   result = (np.log(10.)/2.5) * phi(z)
   result *= 10.**( -0.4 * (1.+alpha(z)) * (Muv(fishcast)-Muvstar(z)) )
   result *= np.exp(-10.**(-0.4 * (Muv(fishcast)-Muvstar(z)) ) )
   return result
