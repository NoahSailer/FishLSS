from headers import *

def compute_matter_power_spectrum(experiment, cosmo, z=-1.):
   '''
   Computes the matter power spectrum for a given cosmology
   at redshift z. Assumes that cosmo.comute() has already been called.
   Returns a function of k.
   '''
   z = experiment.zmid
   if z>=0: z = z
   kk = np.logspace(-4.0,1.0,200)
   pkcb = np.array([cosmo.pk_cb(k,z) for k in kk])
   p = interp1d(kk, pkcb, kind='linear', bounds_error=False, fill_value=0.)
   return p


def compute_galaxy_power_spectrum(experiment, cosmo, b=0.9, RSD=True, Zerror=True):
   '''
   Computes the linear galaxy power spectrum assuming a linear
   bias parameter b. Returns a function of k and mu.
   '''
   z = experiment.zmid
   pmatter = compute_matter_power_spectrum(experiment, cosmo)

   def compute_f(relative_step=0.01):
      z = experiment.zmid
      p_hi = compute_matter_power_spectrum(experiment,cosmo,z=z*(1.+relative_step))
      p_low = compute_matter_power_spectrum(experiment,cosmo,z=z*(1.-relative_step))
      p_fid = compute_matter_power_spectrum(experiment,cosmo,z=z)
      dPdz = lambda k: (p_hi(k) - p_low(k)) / (z*2.*relative_step)
      return lambda k: -(1.+z) * dPdz(k) / (2. * p_fid(k))

   if RSD and Zerror: 
      f = compute_f()
      # convert to km/(s Mpc), this conversion might be slightly off, double check it.
      Hz = cosmo.Hubble(z)*(3.086e5)
      sigma_parallel = (3.e5)*(1.+z)*experiment.sigma_z/Hz
      p = lambda k,mu: pmatter(k) * np.exp(-(k*mu*sigma_parallel)**2.) * (b+f(k)*mu**2.)**2.
      return p

   elif RSD and not Zerror: 
      f = compute_f()
      p = lambda k,mu: pmatter(k) * (b+f(k)*mu**2.)**2.
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
