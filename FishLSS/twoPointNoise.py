import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from FishLSS.castorina import castorinaBias,castorinaPn
from FishLSS.input.reio_hist import Xhi
from FishLSS import __path__ as FishLSS_path


'''
Values and defintions from Table 3 of Wilson and White 2019.
'''
zs = np.array([2.,3.,3.8,4.9,5.9])
Muvstar = np.array([-20.60,-20.86,-20.63,-20.96,-20.91])
Muvstar = interp1d(zs, Muvstar, kind='linear', bounds_error=False,fill_value='extrapolate')
muv = np.array([24.2,24.7,25.4,25.5,25.8])
muv = interp1d(zs, muv, kind='linear', bounds_error=False,fill_value='extrapolate')
phi = np.array([9.70,5.04,9.25,3.22,1.64])*0.001
phi = interp1d(zs, phi, kind='linear', bounds_error=False,fill_value='extrapolate')
alpha = np.array([-1.6,-1.78,-1.57,-1.60,-1.87])
alpha = interp1d(zs, alpha, kind='linear', bounds_error=False,fill_value='extrapolate')

def compute_covariance_matrix(fishcast, zbin_index, nratio=1):
   '''
   Covariance is diagonal. Returns an array of length Nk*Nmu. 
   '''
   z = fishcast.experiment.zcenters[zbin_index]
   prefactor = (4.*np.pi**2.) / (fishcast.dk*fishcast.dmu*fishcast.Vsurvey[zbin_index]*fishcast.k**2.)
   number_density = compute_n(fishcast, z)
   # The number density is effectively reduced if there are redshift uncertainties
   Hz = fishcast.cosmo_fid.Hubble(z)*(299792.458)/fishcast.params['h']
   sigma_parallel = (3.e5)*(1.+z)*fishcast.experiment.sigma_z/Hz
   number_density = number_density * np.maximum(np.exp(-fishcast.k**2. * fishcast.mu**2. * sigma_parallel**2.),1.e-20)
   # this assumes that the fiducial exeriment doesn't have any redshift errors, come up with a more general fix
   P_fid = fishcast.P_fid[zbin_index]
   if fishcast.recon: P_fid = fishcast.P_recon_fid[zbin_index]
   if not fishcast.experiment.HI:
      C = prefactor * (P_fid-1/compute_n(fishcast, z)+1/nratio/number_density)**2.
   else: 
      # since HI noise blows up at high k, I'm trying to avoid numerical from 
      # subtracting O(1e20) from O(1e20)
      C = prefactor * P_fid**2
   return np.maximum(C,1e-50) # avoiding numerical nonsense with possible 0's


def covariance_Cls(fishcast,kmax_knl=1.,CMB='SO',fsky_CMB=0.4,fsky_intersect=None):
   '''
   Returns a covariance matrix Cov[X,Y] as a function of l. X (and Y) is in the basis
   
        X \in {k-k, k-g1, ..., k-gn, g1-g1, ..., gn-gn}   (the basis has dimension 2*n+1)
   
   where g1 is the galaxies in the first redshift bin, k-gi is the cross-correlation of
   the CMB kappa map and the galaxies in the i'th bin, and so on.
   
   This function returns a numpy.array with shape (2*n+1,2*n+1,len(l)), where l = fishcast.l
   '''
   n = fishcast.experiment.nbins
   zs = fishcast.experiment.zcenters
   zes = fishcast.experiment.zedges
   # Lensing noise
   if CMB == 'SO': 
      data = np.genfromtxt(FishLSS_path+'/input/nlkk_v3_1_0deproj0_SENS2_fsky0p4_it_lT30-3000_lP30-5000.dat')
      l,N = data[:,0],data[:,7]
   elif CMB == 'Planck': 
      data = np.genfromtxt(FishLSS_path+'/input/nlkk_planck.dat')
      l,N = data[:,0],data[:,1]
   elif CMB == 'Perfect':
      l,N = fishcast.ell, fishcast.ell*0
   else: 
      data = np.genfromtxt(FishLSS_path+'/input/S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat')
      l,N = data[:,0],data[:,7]
   
   Nkk_interp = interp1d(l,N,kind='linear',bounds_error=False, fill_value=1)
   l = fishcast.ell  ; Nkk = Nkk_interp(l)
   # Cuttoff high ell by blowing up the covariance for ell > ellmax
   chi = lambda z: (1.+z)*fishcast.cosmo_fid.angular_distance(z)*fishcast.params['h']
   ellmaxs =  np.array([kmax_knl*chi(z)/np.sqrt(fishcast.Sigma2(z)) for z in zs])
   constraint = np.ones((n,len(l)))
   idx = np.array([np.where(l)[0] >= ellmax for ellmax in ellmaxs])
   for i in range(n): constraint[i][idx[i]] *= 1e10
   # relevant fsky's
   fsky_LSS = fishcast.experiment.fsky
   if fsky_intersect is None: fsky_intersect = min(fsky_LSS,fsky_CMB)  # full-overlap by default
   # "empty" covariance matrix
   C = np.zeros((2*n+1,2*n+1,len(l)))
   # kk,kk component of covariance
   Ckk = fishcast.Ckk_fid
   C[0,0] = 2*(Ckk + Nkk)**2/(2*l+1) / fsky_CMB
   #
   for i in range(n):
      Ckgi = fishcast.Ckg_fid[i] 
      Cgigi = fishcast.Cgg_fid[i]
      # kk, kg
      C[i+1,0] = 2*(Ckk + Nkk) * Ckgi/(2*l+1)*constraint[i] / fsky_CMB
      C[0,i+1] = C[i+1,0]
      # kk, gg
      C[i+1+n,0] = 2*Ckgi**2/(2*l+1)*constraint[i] * fsky_intersect / fsky_LSS / fsky_CMB
      C[0,i+1+n] = C[i+1+n,0]
      for j in range(n):
         Ckgj = fishcast.Ckg_fid[j]
         Cgjgj = fishcast.Cgg_fid[j]
         # kgi, kgj
         C[i+1,j+1] = Ckgi*Ckgj*constraint[i]*constraint[j]
         if i == j: C[i+1,j+1] += (Ckk + Nkk)*Cgigi*constraint[i]
         C[i+1,j+1] /= (2*l+1) * fsky_intersect
         # gigi, gjgj
         if i == j: C[i+1+n,j+1+n] = 2*Cgigi**2 / (2*l+1)*constraint[i] / fsky_LSS
         # kgi, gjgj
         if i == j: 
            C[i+1,i+1+n] = 2*Cgigi*Ckgi/(2*l+1)*constraint[i] / fsky_LSS
            C[i+1+n,i+1] = C[i+1,i+1+n]
   return C
      

def compute_n(fishcast, z):
   '''
   Returns the relevant number density h^3/Mpc^3. For HI surveys
   returns an array of length Nk*Nmu, for all other surveys
   return a float.
   '''
   if fishcast.experiment.LBG and not fishcast.experiment.custom_n: return LBGn(fishcast, z)
   if fishcast.experiment.Halpha and not fishcast.experiment.custom_n: return hAlphaN(fishcast, z)
   if fishcast.experiment.ELG and not fishcast.experiment.custom_n: return ELGn(fishcast, z)
   if fishcast.experiment.HI and not fishcast.experiment.custom_n: return HIneff(fishcast,z)
   if fishcast.experiment.Euclid and not fishcast.experiment.custom_n: return Euclidn(z)
   if fishcast.experiment.MSE and not fishcast.experiment.custom_n: return MSEn(fishcast,z)
   if fishcast.experiment.Roman and not fishcast.experiment.custom_n: return Romann(fishcast,z)
   return fishcast.experiment.n(z)
    
    
def Muv(fishcast, z, m=24.5):
   '''
   Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
   '''
   result = m - 5. * np.log10(fishcast.cosmo_fid.luminosity_distance(z)*1.e5)
   result += 2.5 * np.log10(1.+z)
   return result


def muv_from_Muv(fishcast, z, M):
   '''
   Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
   '''
   result = M + 5. * np.log10(fishcast.cosmo_fid.luminosity_distance(z)*1.e5)
   result -= 2.5 * np.log10(1.+z)
   return result


def LBGn(fishcast, z, m=24.5):
   '''
   Equation 2.5 of Wilson and White 2019. Return number
   density of LBGs at redshift z in units of Mpc^3/h^3.
   '''
   upper_limit = Muv(fishcast,z,m=m)
   integrand = lambda M: (np.log(10.)/2.5) * phi(z) * 10.**( -0.4 * (1.+alpha(z)) * (M-Muvstar(z)) )*\
                             np.exp(-10.**(-0.4 * (M-Muvstar(z)) ) )
   
   return quad(integrand, -200, upper_limit)[0]


def ELGn(fishcast, z):
   zs = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65])
   dNdz = np.array([309,2269,1923,2094,1441,1353,1337,523,466,329,126])
   N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
   volume = np.array([((1.+z+0.05)*fishcast.cosmo_fid.angular_distance(z+0.05))**3. for z in zs])
   volume -= np.array([((1.+z-0.05)*fishcast.cosmo_fid.angular_distance(z-0.05))**3. for z in zs])
   volume *= 4.*np.pi*fishcast.params_fid['h']**3./3. # volume in Mpc^3/h^3
   n = list(N/volume)
   zs = np.array([0.6,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.7])
   n = [n[0]] + n
   n = n + [n[-1]]
   n = np.array(n)
   n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
   return float(n_interp(z))


def Romann(fishcast, z):
   zs = np.linspace(1.05,2.95,20)
   dNdz = np.array([6160,5907,4797,5727,5147,4530,4792,3870,2857,2277,1725,1215,1642,1615,1305,1087,850,795,847,522])
   N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
   volume = np.array([((1.+z+0.05)*fishcast.cosmo_fid.angular_distance(z+0.05))**3. for z in zs])
   volume -= np.array([((1.+z-0.05)*fishcast.cosmo_fid.angular_distance(z-0.05))**3. for z in zs])
   volume *= 4.*np.pi*fishcast.params_fid['h']**3./3. # volume in Mpc^3/h^3
   n = list(N/volume)
   zs = np.array([zs[0]] + list(zs) + [zs[-1]])
   n = np.array([n[0]] + n + [n[-1]])
   n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
   return float(n_interp(z))


def Euclidn(z):
   #'''
   #From Table 3 of https://arxiv.org/pdf/1606.00180.pdf.
   #'''
   #zs = np.array([0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])
   #n = np.array([1.25,1.92,1.83,1.68,1.51,1.35,1.20,1.00,0.80,0.58,0.38,0.35,0.21,0.11])*1e-3
   #return interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)(z)
   '''
   From Table 3 of https://arxiv.org/pdf/1910.09273.pdf
   '''
   zs = np.array([0.9,1.,1.2,1.4,1.65,1.8])
   n = np.array([6.86,6.86,5.58,4.21,2.61,2.61])*1e-4
   n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
   return n_interp(z) 


def hAlphaN(fishcast, z):
   '''
   Table 2 from Merson+17. Valid for 0.9<z<1.9.
   '''
   zs = np.array([0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])
   dNdz = np.array([10535.,8014.,4998.,3931.,3455.,2446.,2078.,1747.,1524.,1329.])
   N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
   volume = np.array([((1.+z+0.05)*fishcast.cosmo_fid.angular_distance(z+0.05))**3. for z in zs])
   volume -= np.array([((1.+z-0.05)*fishcast.cosmo_fid.angular_distance(z-0.05))**3. for z in zs])
   volume *= 4.*np.pi*fishcast.params_fid['h']**3./3. # volume in Mpc^3/h^3
   n = list(N/volume)
   zs = np.array([0.9,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.9])
   n = [n[0]] + n
   n = n + [n[-1]]
   n = np.array(n)
   n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
   return n_interp(z)


def MSEn(fishcast,z,m=24.5):
   # return ELG number density for z<2.4
   if z <= 2.4: return 1.8e-4 
   # interpolate figure 2 of https://arxiv.org/pdf/1903.03158.pdf to get efficiency
   mags   = np.array([22.75, 23.25, 23.75, 24.25])
   zs     = np.array([2.6,3.0,3.4,3.8])
   blue   = np.array([[0.619,0.846,0.994], [0.452,0.745,0.962], [0.269,0.495,0.919], [0.102,0.327,0.908]])
   orange = np.array([[0.582,0.780,0.981], [0.443,0.663,0.929], [0.256,0.481,0.849], [0.119,0.314,0.854]])
   green  = np.array([[0.606,0.805,0.919], [0.486,0.708,0.815], [0.289,0.559,0.746], [0.146,0.363,0.754]])
   red    = np.array([[0.624,0.752,0.934], [0.501,0.671,0.843], [0.334,0.552,0.689], [0.199,0.371,0.699]])
   weight = np.array([0.4,0.3,0.3])
   b,o = np.sum(blue*weight,axis=1),np.sum(orange*weight,axis=1)
   g,r = np.sum(green*weight,axis=1),np.sum(red*weight,axis=1)
   eff = np.array([b,o,r,g])
   #
   efficiency = interp2d(zs,mags,eff,kind='linear',bounds_error=False)
   #
   def integrand(M): 
      result = (np.log(10.)/2.5) * phi(z) * 10.**( -0.4 * (1.+alpha(z)) * (M-Muvstar(z)) )
      result *= np.exp(-10.**(-0.4 * (M-Muvstar(z)) ) )
      m = muv_from_Muv(fishcast,z,M)
      result *= efficiency(z,m)
      return result
   #
   n = lambda m: quad(integrand, -200, Muv(fishcast,z,m=m))[0]
   return n(m)


def nofl(x, hexpack=True, Nside=256, D=6):
   '''
   Adapted from https://github.com/slosar/PUMANoise.
   Helper function for puma_therm. Returns baseline 
   density.
   '''
   # quadratic packing
   if hexpack:
      # hexagonal packing
      a,b,c,d,e=0.56981864, -0.52741196,  0.8358006 ,  1.66354748,  7.31776875
   else:
      # square packing
      a,b,c,d,e=0.4847, -0.330,  1.3157,  1.5975,  6.8390
   xn=x/(Nside*D)
   n0=(Nside/D)**2
   res=n0*(a+b*xn)/(1+c*xn**d)*np.exp(-(xn)**e)
   return res

def get_Tb(fishcast,z):
   '''
   Returns the mean 21cm brightness temp in K. 
   If z < 6 use fitting formula from Eq. B1 of
   https://arxiv.org/pdf/1810.09572.
   '''
   if z <= 6:
       Ohi = 4e-4*(1+z)**0.6
       h = fishcast.params_fid['h']
       Ez = fishcast.cosmo_fid.Hubble(z)/fishcast.cosmo_fid.Hubble(0)
       Tb = 188e-3*h/Ez*Ohi*(1+z)**2
       return Tb
   omb = fishcast.params_fid['omega_b']
   omm = fishcast.params_fid['omega_cdm'] + omb
   result = 28e-3 * ((1+z)*0.14/10/omm)**0.5
   result *= omb/0.022
   return result * (1-Xhi(z))
    

def HI_therm(fishcast, z, effic=0.7, hexpack=True, skycoupling=0.9, 
             Tground=300., omtcoupling=0.9, Tampl=50., old=False):
   '''
   Adapted from https://github.com/slosar/PUMANoise.
   Thermal noise power in Mpc^3/h^3. Thermal noise is 
   given by equation D4 in https://arxiv.org/pdf/1810.09572.
   I divide by Tb (see get_Tb) to convert to Mpc^3/h^3.
   Returns a function of k [h/Mpc] and mu.
   '''
   D = fishcast.experiment.D
   ttotal = fishcast.experiment.tint*365*24*3600.*fishcast.experiment.fill_factor**2
   Nside = np.sqrt(fishcast.experiment.Ndetectors/fishcast.experiment.fill_factor)
   Hz = fishcast.cosmo_fid.Hubble(z)*(299792.458)/fishcast.params_fid['h'] # in h km/s/Mpc
   Ez = fishcast.cosmo_fid.Hubble(z)/fishcast.cosmo_fid.Hubble(0)
   lam = 0.211 * (1+z) 
   r = (1.+z) * fishcast.cosmo_fid.angular_distance(z)*fishcast.params_fid['h'] # in Mpc/h
   Deff = D * np.sqrt(effic) 
   FOV = (lam / Deff)**2 
   y = 3e5*(1+z)**2/(1420e6*Hz) 
   Sarea=4*np.pi*fishcast.experiment.fsky 
   Ae=np.pi/4*D**2*effic
   # k dependent terms
   kperp = lambda k,mu: k*np.sqrt(1.-mu**2.)
   l = lambda k,mu: kperp(k,mu) * r * lam / (2 * np.pi) 
   def Nu(k,mu):
      if old: return nofl(l(k,mu),hexpack=hexpack,Nside=Nside,D=D)*lam**2 
      #
      ll,pi2lnb = np.genfromtxt(FishLSS_path+'/input/baseline_bs_44_D_14.txt').T
      nofl_new = interp1d(ll,pi2lnb/2/np.pi/ll,bounds_error=False,fill_value=0)
      result = nofl_new(l(k,mu))*lam**2
      result = np.maximum(result,1e-20)
      I = np.where(l(k,mu) < D)
      result[I] = 1e-20
      return result
   # temperatures
   Tb = get_Tb(fishcast,z) 
   Tsky = lambda f: 25.*(f/400.)**(-2.75) +2.7
   Tscope = Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling
   Tsys = Tsky(1420./(1+z))+Tscope
   Pn = lambda k,mu: (Tsys/Tb)**2*r**2*y*(lam**4/Ae**2)*1/(2*Nu(k,mu)*ttotal)*(Sarea/FOV)
   return Pn


def HI_shot(z): 
   '''
   PUMA shot noise Mpc^3/h^3 from Emanuele Castorina
   for z < 6. For z > 6 assume that the shot noise
   is 0.
   '''
   if z<= 6: return castorinaPn(z)
   return 1e-10


def HIneff(fishcast,z,old=True):
   '''
   Effective number density for PUMA. Returns
   an array of length Nk*Nmu.
   '''
   therm = HI_therm(fishcast,z,old=old)(fishcast.k,fishcast.mu)
   shot = HI_shot(z)
   return 1./(therm+shot)
