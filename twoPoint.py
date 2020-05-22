from headers import *

def compute_matter_power_spectrum(experiment, cosmo):
   '''
   Computes the matter power spectrum for a given cosmology
   at redshift z. Assumes that cosmo.comute() has already been called.
   Returns a function of kh.
   '''
   z = experiment.zmid
   kk = np.logspace(-4.0,1.0,200)
   pkcb = np.array([cosmo.pk_cb(k,z) for k in kk])
   p = interp1d(kk, pkcb, kind='linear', bounds_error=False, fill_value=0.)
   return p


def compute_galaxy_power_spectrum(experiment, cosmo, b=0.9, RSD=True, Zerror=True):
   '''
   Computes the linear galaxy power spectrum assuming a linear
   bias parameter b. Returns a function of kh and mu.
   '''
   z = experiment.zmid
   pmatter = compute_matter_power_spectrum(experiment, cosmo)

   if RSD and Zerror: 
      f = cosmo.scale_independent_growth_factor_f(z)
      # convert to km/(s Mpc), this conversion might be slightly off, double check it.
      Hz = cosmo.Hubble(z)*(3.086e5)
      sigma_parallel = (3.e5)*(1.+z)*experiment.sigma_z/Hz
      p = lambda k,mu: pmatter(k) * np.exp(-(k*mu*sigma_parallel)**2.) * (b+f*mu**2.)**2.
      return p

   elif RSD and not Zerror: 
      f = cosmo.scale_independent_growth_factor_f(z)
      p = lambda k,mu: pmatter(k) * (b+f*mu**2.)**2.
      return p

   elif not RSD and Zerror: 
      # convert to km/(s Mpc), this conversion might be slightly off, double check it.
      Hz = cosmo.Hubble(z)*(3.086e5)
      sigma_parallel = (3.e5)*(1.+z)*experiment.sigma_z/Hz
      p = lambda k,mu: pmatter(k) * np.exp(-(k*mu*sigma_parallel)**2.) * (b**2.)
      return p

   else: 
      p = lambda k,mu: pmatter(k) * (b**2.)
      return p
