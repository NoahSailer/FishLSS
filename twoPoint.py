from headers import *
from LPT.moment_expansion_fftw import MomentExpansion
from scipy.signal import savgol_filter
from castorina import castorinaBias,castorinaPn
from twoPointNoise import *

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


def LBGb(fishcast,z):
   '''
   Equation 2.7 of Wilson and White 2019. Returns the bias of
   LBGs.
   '''
   zs = np.array([2.,3.,3.8,4.9,5.9])
   muv = np.array([24.2,24.7,25.4,25.5,25.8])
   muv = interp1d(zs, muv, kind='linear', bounds_error=False, fill_value=0.)
   A = lambda m: -0.98*(m-25.) + 0.11
   B = lambda m: 0.12*(m-25.) + 0.17
   def b(m): return A(m)*(1.+z)+B(m)*(1.+z)**2.
   return b(24.5)
   #def b(M): 
   #   m = M + 5. * np.log10(fishcast.cosmo.luminosity_distance(z)*1.e5) - 2.5 * np.log10(1.+z)
   #   return A(m)*(1.+z)+B(m)*(1.+z)**2.
   #upper_limit = Muv(fishcast,z)
   #integrand = lambda M: (np.log(10.)/2.5) * phi(z) * 10.**( -0.4 * (1.+alpha(z)) * (M-Muvstar(z)) )*\
   #                          np.exp(-10.**(-0.4 * (M-Muvstar(z)) ) )
   #weighted_bias = lambda M: integrand(M) * b(M)
   #return scipy.integrate.quad(weighted_bias, -200, upper_limit)[0]/scipy.integrate.quad(integrand, -200, upper_limit)[0]


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


def MSEb(fishcast,z):
   D = fishcast.cosmo.scale_independent_growth_factor(z)
   D0 = fishcast.cosmo.scale_independent_growth_factor(0)
   return D0/D


def HIb(z): return castorinaBias(z)

    
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
   if fishcast.experiment.LBG and not fishcast.experiment.custom_b: return LBGb(fishcast,z)
   if fishcast.experiment.HI and not fishcast.experiment.custom_b: return HIb(z)
   if fishcast.experiment.Halpha and not fishcast.experiment.custom_b: return hAlphaB(z)
   if fishcast.experiment.ELG and not fishcast.experiment.custom_b: return ELGb(fishcast,z)
   if fishcast.experiment.Euclid and not fishcast.experiment.custom_b: return np.sqrt(1+z)
   if fishcast.experiment.MSE and not fishcast.experiment.custom_b: return MSEb(fishcast,z)
   return fishcast.experiment.b(z)


def get_smoothed_p(fishcast,z,division_factor=2.):
   '''
   Returns a power spectrum without wiggles, given by:
      P_nw = P_approx * F[P/P_approx]
   where P is the linear power spectrum, P_approx is given by Eisenstein & Hu (1998),
   and F is an SG low-pass filter.
   '''
   def Peh(k,p):
      '''
      Returns the smoothed power spectrum Eisenstein & Hu (1998).
      '''
      k = k.copy() * fishcast.params['h']
      Obh2      = fishcast.params['omega_b'] 
      Omh2      = fishcast.params['omega_b'] + fishcast.params['omega_cdm']
      f_baryon  = Obh2 / Omh2
      theta_cmb = fishcast.cosmo.T_cmb() / 2.7
      k_eq = 0.0746 * Omh2 * theta_cmb ** (-2)
      sound_horizon = fishcast.params['h'] * 44.5 * np.log(9.83/Omh2) / \
                            np.sqrt(1 + 10 * Obh2** 0.75) 
      alpha_gamma = 1 - 0.328 * np.log(431*Omh2) * f_baryon + \
                0.38* np.log(22.3*Omh2) * f_baryon ** 2
      ks = k * sound_horizon / fishcast.params['h']
      q = k / (13.41*k_eq)
      gamma_eff = Omh2 * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
      q_eff = q * Omh2 / gamma_eff
      L0 = np.log(2*np.e + 1.8 * q_eff)
      C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
      Teh = L0 / (L0 + C0 * q_eff**2)
      t_with_wiggles = np.sqrt(p/k**fishcast.params['n_s'])
      t_with_wiggles /= t_with_wiggles[0]
      return p * (Teh/t_with_wiggles)**2.
    
   klin = np.logspace(np.log10(fishcast.khmin),np.log10(fishcast.khmax),int(fishcast.Nk))
   plin = np.array([fishcast.cosmo.pk_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in klin])
   p_approx = Peh(klin,plin)
   psmooth = savgol_filter(plin/p_approx,int(fishcast.Nk/division_factor)+1-\
                                         int(fishcast.Nk/division_factor)%2, 6)*p_approx
   return psmooth


def compute_tracer_power_spectrum(fishcast, z, b=-1., bE2=0., bEs=0., alpha0=0., alpha2=0., alpha4=0.,
                                  sn=0.,sn2=0.,f=-1., A_lin=-1., omega_lin=-1., phi_lin=-1.):
   '''
   Computes the nonlinear redshift-space power spectrum [Mpc/h]^3 of the matter tracer.
   Returns an array of length Nk*Nmu. 
   '''
   bfid = compute_b(fishcast,z)
   if b == -1.: b = bfid
   if f == -1.: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
   if A_lin == -1.: A_lin = fishcast.A_lin
   if omega_lin == -1.: omega_lin = fishcast.omega_lin
   if phi_lin == -1.: phi_lin = fishcast.phi_lin
    
   # For AP effect 
   ap0 = fishcast.cosmo.angular_distance(z)*fishcast.params['h'] / fishcast.Da_fid(z)
   ap1 = fishcast.cosmo.Hubble(z)*(299792.458)/fishcast.params['h'] / fishcast.Hz_fid(z)

   K,MU = fishcast.k,fishcast.mu

   if not fishcast.velocileptors:
      # If not using velocileptors, estimate the non-linear evolution
      # using Halofit, and approximate RSD with Kaiser.
      # No AP effect here
      pmatter = compute_matter_power_spectrum(fishcast, z, linear=fishcast.linear)
      result = pmatter * (b+f*MU**2.)**2. * (1. + A_lin * np.sin(omega_lin * K + phi_lin))
      return result
    
   if fishcast.AP: 
      ap_factor = np.sqrt(ap1**2 * fishcast.mu**2 + (1-fishcast.mu**2)/ap0**2)
      kprime = fishcast.k*ap_factor
      muprime = fishcast.mu*ap1/ap_factor
      K,MU = kprime,muprime

   klin = np.logspace(np.log10(min(K)),np.log10(max(K)),fishcast.Nk)
   plin = np.array([fishcast.cosmo.pk_cb_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in klin])
   plin *= (1. + A_lin * np.sin(omega_lin * klin + phi_lin))
   if fishcast.smooth: plin = get_smoothed_p(fishcast,z)
   mome = MomentExpansion(klin,plin,kmin=min(K),kmax=max(K),nk=fishcast.Nk)
   b1 = b-1.
   b2 = bE2 + 8*(bfid-b)/21 # these factors keep b2 and bs fixed when varying b
   bs = bEs - 2*(bfid-b)/7
   alpha0_fid = 1.22 + 0.24*bfid**2*(z-5.96) 
   Hz = fishcast.cosmo.Hubble(z)*(299792.458)/fishcast.params['h']
   biases = [b1,b2,bs,0.]
   cterms = [alpha0+alpha0_fid,alpha2,alpha4]
   noise = compute_n(fishcast,z)
   if fishcast.experiment.HI: noise = noise[0]
   # The velocileptors sn2 input adds a term ~ - 2 * sn2 * (k mu)^2 to the power spectrum. To
   # correct for this strange factor of -2, I multiply my -0.5.
   stoch  = [sn,-0.5*sn2-0.5*noise*((1+z)*300/Hz)**2.]
   pars   = biases + cterms + stoch
   kw,pkw = mome.compute_redshift_space_power_at_mu(pars,f,MU,reduced=True,Nmu=fishcast.Nmu)
   del mome
   # pkw is the redshift-space non-linear power spectrum of the matter tracer
   if fishcast.AP: return ap1*pkw/ap0**2
   return pkw


#def compute_recon_power_spectrum(fishcast, z, b=-1., bE2=0., bEs=0., alpha0=0., alpha2=0., alpha4=0.,
#                                  sn=0.,sn2=0.,f=-1.):
#   bfid = compute_b(fishcast,z)
#   if b == -1.: b = bfid
#   if f == -1.: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
#    
#   # For AP effect 
#   ap0 = fishcast.cosmo.angular_distance(z)*fishcast.params['h'] / fishcast.Da_fid(z)
#   ap1 = fishcast.cosmo.Hubble(z)*(299792.458)/fishcast.params['h'] / fishcast.Hz_fid(z)
#
#   K,MU = fishcast.k,fishcast.mu
#    
#   if fishcast.AP: 
#      ap_factor = np.sqrt(ap1**2 * fishcast.mu**2 + (1-fishcast.mu**2)/ap0**2)
#      kprime = fishcast.k*ap_factor
#      muprime = fishcast.mu*ap1/ap_factor
#      K,MU = kprime,muprime
#
#   klin = np.logspace(np.log10(min(K)),np.log10(max(K)),fishcast.Nk)
#   plin = np.array([fishcast.cosmo.pk_cb_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in klin])
#    
#   zelda = Zeldovich_Recon(klin,plin,R=15)
#
#   zelda.make_pddtable(f,nu,D=1,kmin=1e-3,kmax=0.5,nk=100)
#
#   if fishcast.AP: return ap1*pkw/ap0**2
#   return pkw