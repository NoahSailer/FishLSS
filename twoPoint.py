from headers import *
from castorina import castorinaBias,castorinaPn
from LPT.moment_expansion_fftw import MomentExpansion

def compute_matter_power_spectrum(fishcast, z, linear=False):
   '''
   Computes the cdm + baryon power spectrum for a given cosmology
   at redshift z. By default returns the linear power spectrum, with
   an option to return the Halofit guess for the nonlinear power
   spectrum.
   Returns an array of length Nk*Nmu. 
   '''
   kk = np.logspace(np.log10(fishcast.khmin),np.log10(fishcast.khmax),fishcast.Nk)
   if linear: pmatter = np.array([fishcast.cosmo.pk_cb_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in kk])
   else: pmatter = np.array([fishcast.cosmo.pk_cb(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in kk])
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


def hAlphaB(z):
   '''
   From Table 2 of Merson+19.
   '''
   zs = np.array([0.9,1.,1.2,1.4,1.6,1.8,1.9])
   b = np.array([1.05,1.05,1.17,1.30,1.44,1.6,1.6])
   b_interp = interp1d(zs, b, kind='linear', bounds_error=False, fill_value=0.)
   return b_interp(z) 


def ELGb(fishcast,z):
   D = fishcast.cosmo.scale_independent_growth_factor(z)
   return 0.84/D

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


def PNoise(fishcast, z, effic=0.7, hexpack=True, Nside=256, D=6, 
           skycoupling=0.9, Tground=300., omtcoupling=0.9, Tampl=50., 
           ttotal=5*365*24*3600.):
   '''
   Adapted from https://github.com/slosar/PUMANoise.
   Thermal + shot noise power in Mpc^3/h^3. Thermal noise is 
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


def HIb(z): 
   #return castorinaBias(z)
   # Table 1 of Chen 19
   zs = np.array([2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.])
   b = np.array([1.88,2.08,2.3,2.54,2.81,3.11,3.42,3.74,4.06])
   return interp1d(zs, b, kind='linear', bounds_error=False, fill_value=0.)(z)

def HIneff(z):
   # Table 1 of Chen 19
   zs = np.array([2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.])
   n = np.array([13.,20.,28.,36.,44.,49.,53.,57.,59.])*0.001
   return interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)(z)
   


def compute_f(fishcast, z, step=0.01):
   '''
   Returns the logarithmic derivative of the linear growth rate. Calculated
   from taking a derivative of the power spectrum.
   '''
   p_hi = compute_matter_power_spectrum(fishcast,z=z+step)
   p_higher = compute_matter_power_spectrum(fishcast,z=z+2.*step)
   p_fid = compute_matter_power_spectrum(fishcast,z=z)
   dPdz = (p_fid - (4./3.) * p_hi + (1./3.) * p_higher) / ((-2./3.)*step)
   return -(1.+z) * dPdz / (2. * p_fid)


def compute_b(fishcast,z):
   if fishcast.experiment.LBG and not fishcast.experiment.custom_b: return LBGb(z)
   if fishcast.experiment.HI and not fishcast.experiment.custom_b: return HIb(z)
   if fishcast.experiment.Halpha and not fishcast.experiment.custom_b: return hAlphaB(z)
   if fishcast.experiment.ELG and not fishcast.experiment.custom_b: return ELGb(fishcast,z)
   return fishcast.experiment.b(z)


def compute_tracer_power_spectrum(fishcast, z, b=-1., bE2=0., f=-1., A_lin=-1., omega_lin=-1., phi_lin=-1.):
   '''
   Computes the nonlinear power spectrum of the matter tracer assuming a linear
   bias parameter b. Returns a function of k [h/Mpc] and mu. For HI surverys
   returns the HI power spectrum + noise (thermal + shot).
   '''
   if b == -1.: b = compute_b(fishcast,z)
   if f == -1.: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
   if A_lin == -1.: A_lin = fishcast.A_lin
   if omega_lin == -1.: omega_lin = fishcast.omega_lin
   if phi_lin == -1.: phi_lin = fishcast.phi_lin

   K,MU = fishcast.k,fishcast.mu

   Hz = fishcast.cosmo.Hubble(z)*(299792.458)/fishcast.params['h']
   sigma_parallel = (3.e5)*(1.+z)*fishcast.experiment.sigma_z/Hz

   if not fishcast.velocileptors:
      # If not using velocileptors, estimate the non-linear evolution
      # using Halofit, and approximate RSD with Kaiser.
      pmatter = compute_matter_power_spectrum(fishcast, z, linear=fishcast.linear)
      result = pmatter * np.exp(-(K*MU*sigma_parallel)**2.) * (b+f*MU**2.)**2. * (1. + A_lin * np.sin(omega_lin * K + phi_lin))
      if fishcast.experiment.HI: result += PNoise(fishcast, z)(K,MU)
      return result

   klin = np.logspace(np.log10(fishcast.khmin),np.log10(fishcast.khmax),fishcast.Nk)
   plin = np.array([fishcast.cosmo.pk_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in klin])
   plin *= (1. + A_lin * np.sin(omega_lin * klin + phi_lin))
   mome = MomentExpansion(klin,plin,kmin=fishcast.khmin,kmax=fishcast.khmax,nk=fishcast.Nk)
   b1 = b-1.
   b2 = bE2-8.*b1/21.
   bs = 2.*b1/7.
   b3 = -b1/3.
   biases = [b1,b2,bs,b3]
   cterms = [0.,0.,0.]
   stoch  = [0.,0.]
   pars   = biases + cterms + stoch
   kw,pkw = mome.compute_redshift_space_power_at_mu(pars,f,fishcast.mu,reduced=True,Nmu=fishcast.Nmu)
   del mome
   pkw *= np.exp(-(K*MU*sigma_parallel)**2.)
   if fishcast.experiment.HI: pkw += PNoise(fishcast, z)(K,MU)
   return pkw