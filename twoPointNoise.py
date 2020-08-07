from headers import *
from twoPoint import *

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
   Returns a square array of linear dimension Nk*Nmu. 
   '''
   z = fishcast.experiment.zcenters[zbin_index]
   prefactor = (4.*np.pi**2.) / (fishcast.dk*fishcast.dmu*fishcast.Vsurvey[zbin_index]*fishcast.k**2.)
   if fishcast.experiment.HI: 
      pn = compute_tracer_power_spectrum(fishcast, z)
      diagonal_values = prefactor * pn**2.
      return diagonal_values
   number_density = compute_n(fishcast, z)
   diagonal_values = prefactor * (fishcast.P_fid[zbin_index] + 1./number_density)**2.
   #return np.diag(diagonal_values)
   return diagonal_values


def compute_n(fishcast, z):
   if fishcast.experiment.LBG and not fishcast.experiment.custom_n: return LBGn(fishcast, z)
   if fishcast.experiment.Halpha and not fishcast.experiment.custom_n: return hAlphaN(fishcast, z)
   if fishcast.experiment.ELG and not fishcast.experiment.custom_n: return ELGn(fishcast, z)
   loc = np.where(fishcast.experiment.zcenters >= z)[0]
   if len(loc) == 0:
      if z > fishcast.experiment.zedges[-1]: 
         print('Tried to interpolate outside of specified redshift range')
         return
      else:
         return fishcast.experiment.n[-1]
   return fishcast.experiment.n[loc[0]]  

def Muv(fishcast, z):
   '''
   Equation 2.6 of Wilson and White 2019.
   '''
   result = muv(z) - 5. * np.log10(fishcast.cosmo.luminosity_distance(z)*1.e5)
   result += 2.5 * np.log10(1.+z)
   return result


def LBGn(fishcast, z):
   '''
   Equation 2.5 of Wilson and White 2019. Return number
   density of LBGs at redshift z in units of Mpc^3/h^3.
   '''
   result = (np.log(10.)/2.5) * phi(z)
   result *= 10.**( -0.4 * (1.+alpha(z)) * (Muv(fishcast,z)-Muvstar(z)) )
   result *= np.exp(-10.**(-0.4 * (Muv(fishcast,z)-Muvstar(z)) ) )
   return result


def ELGn(fishcast, z):
   zs = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65])
   dNdz = np.array([45,290,190,205,135,125,130,55,50,40,15])*10.
   N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
   volume = np.array([((1.+z+0.05)*fishcast.cosmo.angular_distance(z+0.05))**3. for z in zs])
   volume -= np.array([((1.+z-0.05)*fishcast.cosmo.angular_distance(z-0.05))**3. for z in zs])
   volume *= 4.*np.pi*fishcast.params['h']**3./3. # volume in Mpc^3/h^3
   n = list(N/volume)
   zs = np.array([0.6,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.7])
   n = [n[0]] + n
   n = n + [n[-1]]
   n = np.array(n)
   n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
   return n_interp(z)


def hAlphaN(fishcast, z):
   '''
   Table 2 from Merson+17. Valid for 0.9<z<1.9.
   '''
   zs = np.array([0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])
   dNdz = np.array([10535.,8014.,4998.,3931.,3455.,2446.,2078.,1747.,1524.,1329.])
   N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
   volume = np.array([((1.+z+0.05)*fishcast.cosmo.angular_distance(z+0.05))**3. for z in zs])
   volume -= np.array([((1.+z-0.05)*fishcast.cosmo.angular_distance(z-0.05))**3. for z in zs])
   volume *= 4.*np.pi*fishcast.params['h']**3./3. # volume in Mpc^3/h^3
   n = list(N/volume)
   zs = np.array([0.9,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.9])
   n = [n[0]] + n
   n = n + [n[-1]]
   n = np.array(n)
   n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
   return n_interp(z)