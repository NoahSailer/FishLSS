from headers import *
from twoPoint import *
import twoPoint
from castorina import castorinaBias,castorinaPn
import scipy

'''
Values and defintions from Table 3 of Wilson and White 2019.
'''
zs = np.array([2.,3.,3.8,4.9,5.9])
Muvstar = np.array([-20.60,-20.86,-20.63,-20.96,-20.91])
Muvstar = interp1d(zs, Muvstar, kind='linear', bounds_error=False, fill_value=0.)
muv = np.array([24.2,24.7,25.4,25.5,25.8])
muv = interp1d(zs, muv, kind='linear', bounds_error=False, fill_value=0.)
phi = np.array([9.70,5.04,9.25,3.22,1.64])*0.001
phi = interp1d(zs, phi, kind='linear', bounds_error=False, fill_value=0.)
alpha = np.array([-1.6,-1.78,-1.57,-1.60,-1.87])
alpha = interp1d(zs, alpha, kind='linear', bounds_error=False, fill_value=0.)

def compute_covariance_matrix(fishcast, zbin_index):
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
   return prefactor * (fishcast.P_fid[zbin_index]-1/compute_n(fishcast, z)+1/number_density)**2.

def covariance_Cls(fishcast,kmax_knl=1.,CMB='SO'):
   '''
   [[Ckk , Ckg1 , Gkg2 ],
    [Ckg1, Cg1g1, 0    ],
    [Ckg2, 0    , Cg2g2]]
   '''
   n = fishcast.experiment.nbins
   zs = fishcast.experiment.zcenters
   zes = fishcast.experiment.zedges
   # Lensing noise
   if CMB == 'SO': data = np.genfromtxt('input/nlkk_v3_1_0deproj0_SENS2_fsky0p4_it_lT30-3000_lP30-5000.dat')
   else: data = np.genfromtxt('input/S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat')
   l,N = data[:,0],data[:,7]
   Nkk_interp = interp1d(l,N,kind='linear')
   l = fishcast.ell  ; Nkk = Nkk_interp(l)
   # Galaxy shot noise
   Nggs = [twoPoint.compute_lensing_Cell(fishcast,'g','g',zes[i],zes[i+1],noise=True) for i in range(n)] 
   # Create covariance object
   C = np.zeros((n+1,n+1,len(fishcast.ell)))
   # Cuttoff high ell by blowing up the covariance for ell > ellmax
   chi = lambda z: (1.+z)*fishcast.cosmo_fid.angular_distance(z)*fishcast.params['h']
   ellmaxs =  np.array([kmax_knl*chi(z)/np.sqrt(fishcast.Sigma2(z)) for z in zs])
   constraint = np.ones((n,len(l)))
   idx = np.array([np.where(l)[0] >= ellmax for ellmax in ellmaxs])
   for i in range(n): constraint[i][idx[i]] *= 1e20
   # build covariance
   C[0,0] = fishcast.Ckk_fid + Nkk
   for i in range(n):
      C[i+1,0] = fishcast.Ckg_fid[i] * constraint[i]
      C[0,i+1] = fishcast.Ckg_fid[i] * constraint[i]
      C[i+1,i+1] = (fishcast.Cgg_fid[i]) * constraint[i] #  + Nggs[i]
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
   if fishcast.experiment.MSE and not fishcast.experiment.custom_n: return MSEn(z)
   return fishcast.experiment.n(z)
    
    
def Muv(fishcast, z, m=24.5):
   '''
   Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
   '''
   result = m - 5. * np.log10(fishcast.cosmo_fid.luminosity_distance(z)*1.e5)
   result += 2.5 * np.log10(1.+z)
   return result


def LBGn(fishcast, z):
   '''
   Equation 2.5 of Wilson and White 2019. Return number
   density of LBGs at redshift z in units of Mpc^3/h^3.
   '''
   upper_limit = Muv(fishcast,z)
   integrand = lambda M: (np.log(10.)/2.5) * phi(z) * 10.**( -0.4 * (1.+alpha(z)) * (M-Muvstar(z)) )*\
                             np.exp(-10.**(-0.4 * (M-Muvstar(z)) ) )
   
   return scipy.integrate.quad(integrand, -200, upper_limit)[0]


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


def MSEn(z):
   zs = np.array([1.6,2.2,2.6,4.0])
   ns = np.array([1.8,1.8,1.1,1.1])*1e-4
   return interp1d(zs, ns, kind='linear', bounds_error=False, fill_value=0.)(z)


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
      a,b,d,d,e=0.4847, -0.330,  1.3157,  1.5975,  6.8390
   xn=x/(Nside*D)
   n0=(Nside/D)**2
   res=n0*(a+b*xn)/(1+c*xn**d)*np.exp(-(xn)**e)
   return res


def HI_therm(fishcast, z, effic=0.7, hexpack=True, Nside=256, D=6, 
           skycoupling=0.9, Tground=300., omtcoupling=0.9, Tampl=50.):
   '''
   Adapted from https://github.com/slosar/PUMANoise.
   Thermal noise power in Mpc^3/h^3. Thermal noise is 
   given by equation D4 in https://arxiv.org/pdf/1810.09572.
   I divide by Tb (equation B1) to convert to Mpc^3/h^3.
   Returns a function of k [h/Mpc] and mu.
   '''
   ttotal = fishcast.experiment.tint*365*24*3600.*fishcast.experiment.fill_factor**2
   Nside = np.sqrt(fishcast.experiment.Ndetectors/fishcast.experiment.fill_factor)
   Hz = fishcast.cosmo_fid.Hubble(z)*(299792.458)/fishcast.params_fid['h'] # in h km/s/Mpc
   Ez = fishcast.cosmo_fid.Hubble(z)/fishcast.cosmo_fid.Hubble(0)
   lam = 0.211 * (1+z) 
   r = (1.+z) * fishcast.cosmo_fid.angular_distance(z)*fishcast.params_fid['h'] # in Mpc/h
   Deff = D * np.sqrt(effic) 
   FOV = (lam / Deff)**2 
   y = 3e5*(1+z)**2/(1420e6*Hz)
   Ohi = 4e-4*(1+z)**0.6 
   Sarea=4*np.pi*fishcast.experiment.fsky 
   Ae=np.pi/4*D**2*effic
   # k dependent terms
   kperp = lambda k,mu: k*np.sqrt(1.-mu**2.)
   l = lambda k,mu: kperp(k,mu) * r * lam / (2 * np.pi) 
   def Nu(k,mu): 
      result = nofl(l(k,mu),hexpack=hexpack,Nside=Nside,D=D)*lam**2
      return np.maximum(result,1e-20)
   # temperatures
   Tb = 188e-3*(fishcast.params_fid['h'])/Ez*Ohi*(1+z)**2 
   Tsky = lambda f: 25.*(f/400.)**(-2.75) +2.7
   Tscope = Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling
   Tsys = Tsky(1420./(1+z))+Tscope
   Pn = lambda k,mu: (Tsys/Tb)**2*r**2*y*(lam**4/Ae**2)*1/(2*Nu(k,mu)*ttotal)*(Sarea/FOV)
   return Pn


def HI_shot(z): 
   '''
   PUMA shot noise Mpc^3/h^3 from Emanuele Castorina.
   '''
   return castorinaPn(z)


def HIneff(fishcast,z):
   '''
   Effective number density for PUMA. Returns
   an array of length Nk*Nmu.
   '''
   therm = HI_therm(fishcast,z)(fishcast.k,fishcast.mu)
   shot = HI_shot(z)
   return 1./(therm+shot)