from headers import *
from castorina import castorinaBias,castorinaPn

def compute_matter_power_spectrum(experiment, cosmo, z):
   '''
   Computes the matter power spectrum for a given cosmology
   at redshift z. Assumes that cosmo.comute() has already been called.
   Returns a function of k.
   '''
   kk = np.logspace(-4.0,1.0,200)
   pkcb = np.array([cosmo.pk_cb(k,z) for k in kk])
   p = interp1d(kk, pkcb, kind='linear', bounds_error=False, fill_value=0.)
   return p


def LBGb(z):
   '''
   Equation 2.7 of Wilson and White 2019. Returns the bias of
   LBGs.
   '''
   zs = np.array([2.,3.,3.8,4.9,5.9])
   muv = np.array([24.2,24.7,25.4,25.5,25.8])
   muv = interp1d(zs, muv, kind='linear', bounds_error=False, fill_value=0.)
   A = lambda m: -0.98*(m-25.) + 0.11
   B = lambda m: 0.12*(m-25.) + 0.17
   return A(muv(z))*(1.+z)+B(muv(z))*(1.+z)**2.


def nofl(x, hexpack=True, Nside=256, D=6):
   '''
   Adapted from https://github.com/slosar/PUMANoise.
   Returns baseline density
   '''
   # quadratic packing
   if hexpack:
      # square packing
      a,b,B,C,D=0.4847, -0.330,  1.3157,  1.5975,  6.8390
   else:
      # hexagonal packing
      a,b,B,C,D=0.56981864, -0.52741196,  0.8358006 ,  1.66354748,  7.31776875
   xn=x/(Nside*D)
   n0=(Nside/D)**2
   res=n0*(a+b*xn)/(1+B*xn**C)*np.exp(-(xn)**D)
   if (type(res)==np.ndarray): res[res<1e-10] = 1e-10
   return res


def PNoise(fishcast, z, effic=0.7, hexpack=True, Nside=256, D=6, skycoupling=0.9, Tground=300., omtcoupling=0.9, Tampl=50., ttotal=5*365*24*3600.):
   '''
   Adapted from https://github.com/slosar/PUMANoise.
   Thermal + shot noise power in Mpc^3. Thermal noise is 
   given by equation D4 in https://arxiv.org/pdf/1810.09572.
   I divide by Tb (equation B1) to convert to Mpc^3.
   Returns a function of k and mu.
   '''
   Hz = fishcast.cosmo.Hubble(z)*(3.086e5) #check this conversion, should be in (km/s/Mpc)
   Ez = fishcast.cosmo.Hubble(z)/fishcast.cosmo.Hubble(0)
   lam = 0.21 * (1+z)
   r = (1.+z) * fishcast.cosmo.angular_distance(z)
   Deff = D * np.sqrt(effic)
   FOV = (lam / Deff)**2
   y = 3e5*(1+z)**2/(1420e6*Hz)
   Ohi = 4e-4*(1+z)**0.6
   Sarea=4*np.pi*fishcast.experiment.fsky
   Ae=np.pi/4*D**2*effic
   # k dependent terms
   kperp = lambda k,mu: k*np.sqrt(1.-mu**2.)
   l = lambda k,mu: kperp(k,mu) * r * lam / (2 * np.pi)
   Nu = lambda k,mu: nofl(l(k,mu),hexpack=hexpack,Nside=Nside,D=D)*lam**2
   # temperatures
   Tb = 188e-3*(fishcast.params['h'])/Ez*Ohi*(1+z)**2
   Tsky = lambda f: 25.*(f/400.)**(-2.75) +2.75
   Tscope = Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling
   Tsys = Tsky(1420./(1+z))+Tscope
   # noise and shot power spectra
   Pn = lambda k,mu: (Tsys/Tb)**2*r**2*y*(lam**4/Ae**2)*1/(2*Nu(k,mu)*ttotal)*(Sarea/FOV)
   Pshot = castorinaPn(z)/(fishcast.params['h'])**3
   return lambda k,mu: Pn(k,mu) + Pshot


def HIb(z): return castorinaBias(z)


def compute_tracer_power_spectrum(fishcast, RSD=True, Zerror=True, Noise=True):
   '''
   Computes the power spectrum of the matter tracer assuming a linear
   bias parameter b. Returns a function of k and mu. For HI surverys
   returns the HI power spectrum + instrumental noise.
   '''
   z = fishcast.experiment.zmid  #update

   experiment = fishcast.experiment
   cosmo = fishcast.cosmo
   pmatter = compute_matter_power_spectrum(experiment, cosmo, z)

   if experiment.LBG: b = LBGb(z)
   elif experiment.HI: b = HIb(z)
   else: b = 0.9

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
      if experiment.HI and Noise: return lambda k,mu: p(k,mu) + PNoise(fishcast, z)(k,mu)
      return p

   elif RSD and not Zerror: 
      f = compute_f()
      p = lambda k,mu: pmatter(k) * (b+f(k)*mu**2.)**2.
      if experiment.HI and Noise: return lambda k,mu: p(k,mu) + PNoise(fishcast, z)(k,mu)
      return p

   elif not RSD and Zerror: 
      # convert to km/(s Mpc), this conversion might be slightly off, double check it.
      Hz = cosmo.Hubble(z)*(3.086e5)
      sigma_parallel = (3.e5)*(1.+z)*experiment.sigma_z/Hz
      p = lambda k,mu: pmatter(k) * np.exp(-(k*mu*sigma_parallel)**2.) * (b**2.)
      if experiment.HI and Noise: return lambda k,mu: p(k,mu) + PNoise(fishcast, z)(k,mu)
      return p

   else: 
      p = lambda k,mu: pmatter(k) * (b**2.)
      if experiment.HI and Noise: return lambda k,mu: p(k,mu) + PNoise(fishcast, z)(k,mu)
      return p