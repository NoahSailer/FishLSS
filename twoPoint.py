from headers import *
from velocileptors.LPT.cleft_fftw import CLEFT
from bao_recon.zeldovich_rsd_recon_fftw import Zeldovich_Recon
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from scipy.signal import savgol_filter
from castorina import castorinaBias,castorinaPn
from twoPointNoise import *
from scipy.integrate import simps
from math import ceil
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.special import legendre

#################################################################################################
# Biases for various surveys and probes.

def LBGb(fishcast,z,m=24.5):
   '''
   Equation 2.7 of Wilson and White 2019. 
   '''
   #zs = np.array([2.,3.,3.8,4.9,5.9])
   #muv = np.array([24.2,24.7,25.4,25.5,25.8])
   #muv = interp1d(zs, muv, kind='linear', bounds_error=False, fill_value=0.)
   A = lambda m: -0.98*(m-25.) + 0.11
   B = lambda m: 0.12*(m-25.) + 0.17
   def b(m): return A(m)*(1.+z)+B(m)*(1.+z)**2.
   return b(m)

def hAlphaB(z):
   '''
   From Table 2 of Merson+19.
   '''
   zs = np.array([0.9,1.,1.2,1.4,1.6,1.8,1.9])
   b = np.array([1.05,1.05,1.17,1.30,1.44,1.6,1.6])
   b_interp = interp1d(zs, b, kind='linear', bounds_error=False, fill_value=0.)
   return b_interp(z) 

def EuclidB(z):
   '''
   From Table 3 of arxiv.org/pdf/1910.09273
   '''
   zs = np.array([0.9,1.,1.2,1.4,1.65,1.8])
   b = np.array([1.46,1.46,1.61,1.75,1.90,1.90])
   b_interp = interp1d(zs, b, kind='linear', bounds_error=False, fill_value=0.)
   return b_interp(z) 

def Romanb(z): return 1.1*z+0.3

def ELGb(fishcast,z):
   D = fishcast.cosmo_fid.scale_independent_growth_factor(z)
   return 0.84/D

def MSEb(fishcast,z):
   '''
   Constant clustering approximation.
   '''
   D = fishcast.cosmo_fid.scale_independent_growth_factor(z)
   D0 = fishcast.cosmo_fid.scale_independent_growth_factor(0)
   if z<=2.4: return D0/D
   return LBGb(fishcast,z,m=24.5)

def HIb(z): return castorinaBias(z)
    
def compute_b(fishcast,z):
   '''
   Quick way of getting the bias. 
   '''
   exp = fishcast.experiment
   custom = exp.custom_b
   if exp.LBG    and not custom: return LBGb(fishcast,z)
   if exp.HI     and not custom: return HIb(z)
   if exp.Halpha and not custom: return hAlphaB(z)
   if exp.ELG    and not custom: return ELGb(fishcast,z)
   if exp.Euclid and not custom: return EuclidB(z) 
   if exp.MSE    and not custom: return MSEb(fishcast,z)
   if exp.Roman  and not custom: return Romanb(z)
   return exp.b(z)

#################################################################################################
# scale-dependent growth rate (I'm not using this function anywhere)

def compute_f(fishcast, z, step=0.01):
   '''
   Returns the scale-dependent growth factor.
   '''
   p_hi = compute_matter_power_spectrum(fishcast,z=z+step)
   p_higher = compute_matter_power_spectrum(fishcast,z=z+2.*step)
   p_fid = compute_matter_power_spectrum(fishcast,z=z)
   dPdz = (p_fid - (4./3.) * p_hi + (1./3.) * p_higher) / ((-2./3.)*step)
   return -(1.+z) * dPdz / (2. * p_fid)

#################################################################################################
# functions for calculating P_{mm}(k,mu), P_{gg}(k,mu), P_{XY}(k), C^{XY}_\ell, and the smoothed
# power spectrum

def compute_matter_power_spectrum(fishcast, z, linear=False):
   '''
   Computes the cdm + baryon power spectrum for a given cosmology
   at redshift z. By default returns the linear power spectrum, with
   an option to return the Halofit guess for the nonlinear power
   spectrum.
   Returns an array of length Nk*Nmu. 
   '''
   kk = np.logspace(np.log10(fishcast.kmin),np.log10(fishcast.kmax),fishcast.Nk)
   if linear: pmatter = np.array([fishcast.cosmo.pk_cb_lin(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in kk])
   else: pmatter = np.array([fishcast.cosmo.pk_cb(k*fishcast.params['h'],z)*fishcast.params['h']**3. for k in kk])
   return np.repeat(pmatter,fishcast.Nmu)


def get_smoothed_p(fishcast,z,klin,plin,division_factor=2.):
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
      theta_cmb = fishcast.cosmo_fid.T_cmb() / 2.7
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

   p_approx = Peh(klin,plin)
   psmooth = savgol_filter(plin/p_approx,int(fishcast.Nk/division_factor)+1-\
                                         int(fishcast.Nk/division_factor)%2, 6)*p_approx
   return psmooth


def compute_tracer_power_spectrum(fishcast, z, b=-1., b2=-1, bs=-1, 
                                  alpha0=-1, alpha2=0, alpha4=0.,alpha6=0.,
                                  N=None,N2=-1,N4=0.,f=-1., A_lin=-1., 
                                  omega_lin=-1., phi_lin=-1.,kIR=0.2,     
                                  moments=False,bL1=None,bL2=None,bLs=None,
                                  one_loop=True):
   '''
   Computes the nonlinear redshift-space power spectrum P(k,mu) [Mpc/h]^3 
   of the matter tracer. Returns an array of length Nk*Nmu. 
   '''
   exp = fishcast.experiment
   if fishcast.recon: 
      return compute_recon_power_spectrum(fishcast,z,b=b,b2=b2,bs=bs,N=N)

   if b == -1.: b = compute_b(fishcast,z)
   if b2 == -1 and exp.b2 is not None: b2 = exp.b2(z) 
   elif b2 == -1: b2 = 8*(b-1)/21
   if bs == -1: bs = -2*(b-1)/7
   if f == -1.: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
   if A_lin == -1.: A_lin = fishcast.A_lin
   if omega_lin == -1.: omega_lin = fishcast.omega_lin
   if phi_lin == -1.: phi_lin = fishcast.phi_lin 
   if alpha0 == -1: alpha0 = 1.22 + 0.24*b**2*(z-5.96) 
   if N is None: N = 1/compute_n(fishcast,z)
   #
   noise = 1/compute_n(fishcast,z)
   if exp.HI: noise = castorinaPn(z) # only keep the shot noise piece
   sigv = exp.sigv
   Hz = fishcast.Hz_fid(z)
   if N2 == -1: N2 = -noise*((1+z)*sigv/fishcast.Hz_fid(z))**2
    
   K = fishcast.k
   MU = fishcast.mu
   h = fishcast.params['h']
   klin = np.array([K[i*fishcast.Nmu] for i in range(fishcast.Nk)])
   plin = np.array([fishcast.cosmo.pk_cb_lin(k*h,z)*h**3. for k in klin]) 
   plin *= (1. + A_lin * np.sin(omega_lin * klin + phi_lin))

   if fishcast.smooth: plin = get_smoothed_p(fishcast,z)
    
   if fishcast.linear2:
      pmatter = np.repeat(plin,fishcast.Nmu)
      result = pmatter * (b+f*MU**2.)**2.
      result /= 1 - N2*(K*MU)**2/noise 
      result += N   
      return result
    
   if fishcast.linear:
      # If not using velocileptors, use linear theory
      # and approximate RSD with Kaiser.
      pmatter = np.repeat(plin,fishcast.Nmu)
      result = pmatter * (b+f*MU**2.)**2.
      result += N+N2*(K*MU)**2+N4*(K*MU)**4
      result += pmatter*K**2*(alpha0+alpha2*MU**2+alpha4*MU**4)
      return result
   
   if bL1 is None: bL1 = b-1.
   if bL2 is None: bL2 = b2 - 8*(b-1)/21 
   if bLs is None: bLs = bs + 2*(b-1)/7
   
   biases = [bL1,bL2,bLs,0.]
   cterms = [alpha0,alpha2,alpha4,alpha6]
   stoch  = [0,N2,N4]
   pars   = biases + cterms + stoch

   lpt = LPT_RSD(klin,plin,kIR=kIR,one_loop=one_loop)
   lpt.make_pltable(f,kmin=min(klin),kmax=max(klin),nk=len(klin))
   k,p0,p2,p4 = lpt.combine_bias_terms_pkell(pars)
   if moments: return k,p0,p2,p4
   p0 = np.repeat(p0,fishcast.Nmu) 
   p2 = np.repeat(p2,fishcast.Nmu) 
   p4 = np.repeat(p4,fishcast.Nmu)
   pkmu = p0+0.5*(3*MU**2-1)*p2+0.125*(35*MU**4-30*MU**2+3)*p4 + N
   del lpt
   return pkmu 


def compute_real_space_cross_power(fishcast, X, Y, z, gamma=1., b=-1., 
                                   b2=-1,bs=-1,alpha0=-1,alphax=0,N=None):
   '''
   Wrapper function for CLEFT. Returns P_XY where X,Y = k or g
   as a function of k.
   '''
   if b == -1.: b = compute_b(fishcast,z)
   if b2 == -1: b2 = 8*(b-1)/21
   if bs == -1: bs = -2*(b-1)/7
   if alpha0 == -1: alpha0 = 1.22 + 0.24*b**2*(z-5.96) 
   if N == None: 
      N = 1/compute_n(fishcast,z)
      if fishcast.experiment.HI: N = castorinaPn(z)
   bk = (1+gamma)/2-1
   h = fishcast.params['h']
   
   K = fishcast.k
    
   klin = np.array([K[i*fishcast.Nmu] for i in range(fishcast.Nk)])
   plin = np.array([fishcast.cosmo.pk_cb_lin(k*h,z)*h**3. for k in klin])
    
    
   ######################################################################################################## 
   if fishcast.linear:
      print('IMPLEMENT STUFF HERE')
    
   if X == Y and X == 'k': 
      plin = np.array([fishcast.cosmo.pk_lin(k*h,z)*h**3. for k in klin])
      cleft = CLEFT(klin,plin)
      cleft.make_ptable(kmin=min(klin),kmax=max(klin),nk=fishcast.Nk)
      kk,pmm = cleft.combine_bias_terms_pk(0,0,0,0,0,0) 
      if z>10: pmm=np.array([fishcast.cosmo.pk(k*h,z)*h**3. for k in klin])
      return interp1d(kk, pmm, kind='linear', bounds_error=False, fill_value=0.)
    
   cleft = CLEFT(klin,plin)
   cleft.make_ptable(kmin=min(klin),kmax=max(klin),nk=fishcast.Nk)
   
   bL1 = b-1.
   bL2 = b2 - 8*(b-1)/21 
   bLs = bs + 2*(b-1)/7
    
    
   if X == Y and X == 'g':
      kk,pgg = cleft.combine_bias_terms_pk(bL1,bL2,bLs,alpha0,0,N) 
      return interp1d(kk, pgg, kind='linear', bounds_error=False, fill_value=0.) 
   
   # if neither of the above cases are true, return Pkg
   kk = cleft.pktable[:,0]
   pkg = cleft.pktable[:,1]+0.5*(bL1+bk)*cleft.pktable[:,2]+bL1*bk*cleft.pktable[:,3]+\
         0.5*bL2*cleft.pktable[:,4]+0.5*bk*bL2*cleft.pktable[:,5]+0.5*bLs*cleft.pktable[:,7]+\
         0.5*bk*bLs*cleft.pktable[:,8]+alphax*kk**2*cleft.pktable[:,11]
   return interp1d(kk, pkg, kind='linear', bounds_error=False, fill_value=0.) 
   
   
def compute_lensing_Cell(fishcast, X, Y, zmin=None, zmax=None,zmid=None,gamma=1., 
                         b=-1,b2=-1,bs=-1, alpha0=-1,alphax=0.,N=-1,
                         noise=False,Nzsteps=100,Nzeff='auto',maxDz=0.2):
   '''
   Calculates C^XY_l using the Limber approximation where X,Y = 'k' or 'g'. 
   Returns an array of length len(fishcast.ell). If X=Y=k, use CLASS with 
   HaloFit to calculate the lensing convergence. 
   ---------------------------------------------------------------------
   noise: if True returns the projected shot noise. 
   (replaces P -> 1/n)

   Nzsteps: number of integration points
   ---------------------------------------------------------------------
   '''
   if X == Y and X == 'k' and zmin is None and zmax is None: 
      lmin,lmax = int(min(fishcast.ell)),int(max(fishcast.ell))
      Cphiphi = fishcast.cosmo.raw_cl(lmax)['pp'][lmin:]
      return 0.25*fishcast.ell**2*(fishcast.ell+1)**2*Cphiphi*((1+gamma)/2)**2

   if zmin is None or zmax is None: raise Exception('Must specify zmin and zmax')
   if zmid is None: zmid = (zmin+zmax)/2
    
   b_fid = compute_b(fishcast,zmid)  
   b2_fid = 8*(b_fid-1)/21
   bs_fid = -2*(b_fid-1)/7
   alpha0_fid = 1.22 + 0.24*b_fid**2*(zmid-5.96) 
   N_fid = 1/compute_n(fishcast,zmid)
   if fishcast.experiment.HI: N_fid = castorinaPn(zmid)
        
   if b == -1: b = b_fid
   if b2 == -1: b2 = b2_fid
   if bs == -1: bs = bs_fid
   if alpha0 == -1: alpha0 = alpha0_fid
   if N == -1: N = N_fid
       
   z_star = 1098
   chi = lambda z: (1.+z)*fishcast.cosmo.angular_distance(z)*fishcast.params['h']
   def dchidz(z): 
      if z <= 0.02: return (chi(z+0.01)-chi(z))/0.01
      return (-chi(z+0.02)+8*chi(z+0.01)-8*chi(z-0.01)+chi(z-0.02))/0.12 
   chi_star = chi(z_star)  

   # CMB lensing convergence kernel
   W_k = lambda z: 1.5*(fishcast.params['omega_cdm']+fishcast.params['omega_b'])/\
                   fishcast.params['h']**2*(1/2997.92458)**2*(1+z)*chi(z)*\
                   (chi_star-chi(z))/chi_star
   
   # Galaxy kernel (arbitrary normalization)
   def nonnorm_Wg(z):
      result = fishcast.cosmo.Hubble(z)
      try:
         number_density = compute_n(fishcast,z)
      except:
         if X == Y and X == 'k': number_density = 10
         else: raise Exception('Attemped to integrate outside of \
                                specificed n(z) range')
      if fishcast.experiment.HI: number_density = castorinaPn(z)
      result *= number_density * dchidz(z) * chi(z)**2 
      return result
  
   zs = np.linspace(zmin,zmax,Nzsteps)
   chis = np.array([chi(z) for z in zs])
   Wg = np.array([nonnorm_Wg(z) for z in zs])
   Wg /= simps(Wg,x=chis)
   Wk = np.array([W_k(z) for z in zs])
   if X == Y and X == 'k': kern = Wk**2/chis**2
   elif X == Y and X == 'g': kern = Wg**2/chis**2
   else: kern = Wg*Wk/chis**2
   if Nzeff == 'auto': Nzeff = ceil((zmax-zmin)/maxDz)
   if Nzeff > Nzsteps: Nzeff = Nzsteps
   mask = [(zs < np.linspace(zmin,zmax,Nzeff+1)[1:][i])*\
           (zs >= np.linspace(zmin,zmax,Nzeff+1)[:-1][i]) for i in range(Nzeff)]
   mask[-1][-1] = True
   zeff = np.array([simps(kern*zs*m,x=chis)/simps(kern*m,x=chis) for m in mask])

   bz = lambda z: compute_b(fishcast,z) * b/b_fid
   b2z = lambda z: 8*(compute_b(fishcast,z)-1)/21 * b2/b2_fid
   bsz = lambda z: -2*(compute_b(fishcast,z)-1)/7 * bs/bs_fid
   alpha0z = lambda z: (1.22 + 0.24*compute_b(fishcast,z)**2*(z-5.96)) * alpha0/alpha0_fid
   def Nz(z):
      if X == Y and X == 'k': return 0
      if not fishcast.experiment.HI: return 1/compute_n(fishcast,z) * N/N_fid
      else: return castorinaPn(z) * N/N_fid

   # calculate P_XY 
   if not noise: P = [compute_real_space_cross_power(
                 fishcast, X, Y, zz, gamma=gamma, b=bz(zz), 
                 b2=b2z(zz),bs=bsz(zz),alpha0=alpha0z(zz),alphax=alphax,N=Nz(zz)) for zz in zeff]

   if noise: P = [fishcast.get_f_at_fixed_mu(np.ones(fishcast.Nk*fishcast.Nmu)/\
                  compute_n(fishcast, zz),0) for zz in zeff]
   P = np.array(P)

   def result(ell):
      kval = (ell+0.5)/chis
      integrands = np.array([kern*P[i](kval) for i in range(Nzeff)])*mask
      integrand = np.sum(integrands,axis=0)
      return simps(integrand,x=chis)
   
   return np.array([result(l) for l in fishcast.ell])
    

def compute_recon_power_spectrum(fishcast,z,b=-1.,b2=-1.,bs=-1.,N=None):
   '''
   Returns the reconstructed power spectrum, following Stephen's paper.
   '''
   if b == -1.: b = compute_b(fishcast,z)
   if b2 == -1: b2 = 8*(b-1)/21
   if bs == -1: bs = -2*(b-1)/7
   noise = 1/compute_n(fishcast,z)
   if fishcast.experiment.HI: noise = castorinaPn(z)
   if N is None: N = 1/compute_n(fishcast,z)
   f = fishcast.cosmo.scale_independent_growth_factor_f(z) 
    
   bL1 = b-1.
   bL2 = b2-8*(b-1)/21
   bLs = bs+2*(b-1)/7
    
   K,MU = fishcast.k,fishcast.mu
   h = fishcast.params['h']
   klin = np.logspace(np.log10(min(K)),np.log10(max(K)),fishcast.Nk)
   mulin = MU.reshape((fishcast.Nk,fishcast.Nmu))[0,:]
   plin = np.array([fishcast.cosmo.pk_cb_lin(k*h,z)*h**3. for k in klin])
    
   zelda = Zeldovich_Recon(klin,plin,R=15)

   kSparse,p0ktable,p2ktable,p4ktable = zelda.make_pltable(f,ngauss=3,kmin=min(K),kmax=max(K),nk=500,method='RecSym')
   bias_factors = np.array([1, bL1, bL1**2, bL2, bL1*bL2, bL2**2, bLs, bL1*bLs, bL2*bLs, bLs**2,0,0,0])
   p0Sparse = np.sum(p0ktable*bias_factors, axis=1)
   p2Sparse = np.sum(p2ktable*bias_factors, axis=1)
   p4Sparse = np.sum(p4ktable*bias_factors, axis=1)
   p0,p2,p4 = Spline(kSparse,p0Sparse)(klin),Spline(kSparse,p2Sparse)(klin),Spline(kSparse,p4Sparse)(klin)
   l0,l2,l4 = legendre(0),legendre(2),legendre(4)
   Pk = lambda mu: p0*l0(mu) + p2*l2(mu) + p4*l4(mu)
   result = np.array([Pk(mu) for mu in mulin]).T
   return result.flatten() + N