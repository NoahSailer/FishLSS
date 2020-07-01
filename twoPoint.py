from headers import *
from castorina import castorinaBias,castorinaPn
from LPT.moment_expansion_fftw import MomentExpansion

def compute_matter_power_spectrum(fishcast, z):
   '''
   Computes the linear matter power spectrum for a given cosmology
   at redshift z. Assumes that cosmo.comute() has already been called.
   Returns a function of k [h/Mpc].
   '''
   kk = np.logspace(np.log10(fishcast.khmin),np.log10(fishcast.khmax),fishcast.Nk)
   pmatter = np.array([fishcast.cosmo.pk_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in kk])
   return np.repeat(pmatter,fishcast.Nmu)


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
   m = 25.
   # below I assume a constant limiting maginitude m -25, could
   # in general be replaced with some m(z)
   return A(m)*(1.+z)+B(m)*(1.+z)**2.


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
   I divide by Tb (equation B1) to convert to Mpc^3/h^3.
   Returns a function of k and mu.
   '''
   Hz = fishcast.cosmo.Hubble(z)*(299792.458)
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
   kperp = lambda k,mu: fishcast.params['h']*k*np.sqrt(1.-mu**2.)
   l = lambda k,mu: kperp(k,mu) * r * lam / (2 * np.pi)
   Nu = lambda k,mu: nofl(l(k,mu),hexpack=hexpack,Nside=Nside,D=D)*lam**2
   # temperatures
   Tb = 188e-3*(fishcast.params['h'])/Ez*Ohi*(1+z)**2
   Tsky = lambda f: 25.*(f/400.)**(-2.75) +2.75
   Tscope = Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling
   Tsys = Tsky(1420./(1+z))+Tscope
   # noise and shot power spectra
   Pn = lambda k,mu: (Tsys/Tb)**2*r**2*y*(lam**4/Ae**2)*1/(2*Nu(k,mu)*ttotal)*(Sarea/FOV)*fishcast.params['h']**3.
   Pshot = castorinaPn(z)
   return lambda k,mu: Pn(k,mu) + Pshot


def HIb(z): return castorinaBias(z)


def compute_f(fishcast, z, step=0.01):
   '''
   Returns the logarithmic derivative of the linear growth rate.
   '''
   p_hi = compute_matter_power_spectrum(fishcast,z=z+step)
   p_higher = compute_matter_power_spectrum(fishcast,z=z+2.*step)
   p_fid = compute_matter_power_spectrum(fishcast,z=z)
   dPdz = (p_fid - (4./3.) * p_hi + (1./3.) * p_higher) / ((-2./3.)*step)
   return -(1.+z) * dPdz / (2. * p_fid)


def compute_tracer_power_spectrum(fishcast, z, RSD=True, Zerror=True, Noise=True, 
                                  Wiggles=True):
   '''
   Computes the power spectrum of the matter tracer assuming a linear
   bias parameter b. Returns a function of k [h/Mpc] and mu. For HI surverys
   returns the HI power spectrum + instrumental noise (thermal + shot), if desired.
   '''
   experiment = fishcast.experiment
   cosmo = fishcast.cosmo
   pmatter = compute_matter_power_spectrum(fishcast, z)
    
   if Wiggles: wiggles = lambda k: 1. + fishcast.A_lin * np.sin(fishcast.omega_lin * k + fishcast.phi_lin)
   else: wiggles = lambda k: 1. + k*0.
        
   if fishcast.velocileptors and not experiment.HI:
      D = fishcast.cosmo.scale_independent_growth_factor(z)
      f = fishcast.cosmo.scale_independent_growth_factor_f(z)
      #f = compute_f(fishcast, z)
      klin = np.logspace(np.log10(fishcast.khmin),np.log10(fishcast.khmax),fishcast.Nk)
      plin = np.array([fishcast.cosmo.pk_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in klin])
      mome = MomentExpansion(klin,plin,kmin=fishcast.khmin,kmax=fishcast.khmax,nk=fishcast.Nk)
      biases = [0.70,0.5,-0.3,0.0]
      cterms = [10.0,20.,-60.]
      stoch  = [1800.,-1000.]
      pars   = biases + cterms + stoch
      kw,pkw = mome.compute_redshift_space_power_at_mu(pars,f,fishcast.mu,reduced=True,Nmu=fishcast.Nmu)
      return pkw

   if experiment.LBG: b = LBGb(z)
   elif experiment.HI: b = HIb(z)
   else: b = fishcast.experiment.b(z)
    
   K,MU = fishcast.k,fishcast.mu
    
   if RSD and Zerror: 
      f = fishcast.cosmo.scale_independent_growth_factor_f(z)
      #f = compute_f(fishcast, z)
      Hz = cosmo.Hubble(z)*(299792.458)/fishcast.params['h']
      sigma_parallel = (3.e5)*(1.+z)*experiment.sigma_z/Hz
      p = pmatter * np.exp(-(K*MU*sigma_parallel)**2.) * (b+f*MU**2.)**2. * wiggles(K)
      if experiment.HI and Noise: return p + PNoise(fishcast, z)(K,MU)
      return p 

   elif RSD and not Zerror: 
      f = compute_f(fishcast, z)
      p = pmatter * (b+f*MU**2.)**2. * wiggles(K)
      if experiment.HI and Noise: return p + PNoise(fishcast, z)(K,MU)
      return p 

   elif not RSD and Zerror: 
      Hz = cosmo.Hubble(z)*(299792.458)
      sigma_parallel = (3.e5)*(1.+z)*experiment.sigma_z/Hz
      p = pmatter * np.exp(-(K*MU*sigma_parallel)**2.) * (b**2.) * wiggles(K)
      if experiment.HI and Noise: return p + PNoise(fishcast, z)(K,MU)
      return p

   else: 
      p = pmatter * (b**2.) * wiggles(K)
      if experiment.HI and Noise: return p + PNoise(fishcast, z)(K,MU)
      return p