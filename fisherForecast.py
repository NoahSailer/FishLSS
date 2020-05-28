from headers import *
from twoPoint import *
from twoPointNoise import *


class fisherForecast(object):
   '''
   Capable of calculating the galaxy power spectrum and its derivatives
   with respect to cosmological parameters. Able to calculate the
   Fisher matrix for an arbitrary number of parameters.
   '''

   def __init__(self, khmin=1e-4, khmax=0.25, cosmo=None, experiment=None, Nmu=100, Nk=100, params=None, marg_params=np.array(['A_s,h'])):
      '''
      '''
      self.khmin = khmin
      self.khmax = khmax
      self.Nmu = Nmu
      self.Nk = Nk
      self.marg_params = marg_params

      self.experiment = None
      self.cosmo = None
      self.k = None  # [h/Mpc]
      self.dk = None
      self.mu = None
      self.dmu = None
      self.P_fid = None
      self.Vsurvey = None
      self.params = None

      if (cosmo is None) or (experiment is None):
         print('Attempted to create a forecast without an experiment or cosmology.')     
      else:
         self.set_experiment_and_cosmology_specific_parameters(experiment,cosmo,params)


   def set_experiment_and_cosmology_specific_parameters(self, experiment, cosmo, params):
      self.experiment = experiment
      self.cosmo = cosmo
      self.params = params
      k = np.logspace(np.log10(self.khmin),np.log10(self.khmax),self.Nk)
      dk = list(k[1:]-k[:-1])
      dk.append(dk[-1])
      dk = np.array(dk)
      mu = np.linspace(0.,1.,self.Nmu)
      dmu = list(mu[1:]-mu[:-1])
      dmu.append(dmu[-1])
      dmu = np.array(dmu)
      self.k = np.repeat(k,self.Nmu)
      self.dk = np.repeat(dk,self.Nmu)
      self.mu = np.tile(mu,self.Nk)
      self.dmu = np.tile(dmu,self.Nk)
      self.P_fid = compute_tracer_power_spectrum(self)(self.k,self.mu)
      self.Vsurvey = self.comoving_survey_volume()


   def comoving_survey_volume(self):
      '''
      Returns the comoving volume in Mpc^3/h^3 assuming
      that the universe is flat.
      '''
      zmin,zmax = self.experiment.zmin,self.experiment.zmax
      vsmall = (4*np.pi/3) * ((1.+zmin)*self.cosmo.angular_distance(zmin))**3.
      vbig = (4*np.pi/3) * ((1.+zmax)*self.cosmo.angular_distance(zmax))**3.
      return self.experiment.fsky*(vbig - vsmall)*self.params['h']**3.


   def compute_dPdp(self, param, relative_step=0.01, one_sided=False, five_point=False, analytical=True, Noise=False):
      '''
      Calculates the derivative of the galaxy power spectrum
      with respect to the input parameter around the fidicual
      cosmology and redshift. Returns a an array of length
      Nk*Nmu.

      To make the derivatives ~2 times faster, I don't 
      recompute the cosmology after taking the derivative.
      '''
      default_value = self.params[param]

      if analytical:
         if param == 'n_s': return self.P_fid * np.log(self.k*self.params['h']/0.05)
         if param == 'A_s': return self.P_fid / self.params['A_s']

      if one_sided:
         self.cosmo.set({param : default_value * (1. + relative_step)})
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self)(self.k,self.mu)
         self.cosmo.set({param : default_value * (1. + 2.*relative_step)})
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self)(self.k,self.mu)
         self.cosmo.set({param : default_value})
         return (self.P_fid - (4./3.) * P_dummy_hi + (1./3.) * P_dummy_higher) / ((-2./3.) * default_value * relative_step)

      if five_point:
         self.cosmo.set({param : default_value * (1. + relative_step)})
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self)(self.k,self.mu)
         self.cosmo.set({param : default_value * (1. + 2.*relative_step)})
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self)(self.k,self.mu)
         self.cosmo.set({param : default_value * (1. - relative_step)})
         self.cosmo.compute()
         P_dummy_low = compute_tracer_power_spectrum(self)(self.k,self.mu)
         self.cosmo.set({param : default_value * (1. - 2.*relative_step)})
         self.cosmo.compute()
         P_dummy_lower = compute_tracer_power_spectrum(self)(self.k,self.mu)
         self.cosmo.set({param : default_value})
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * default_value * relative_step)

      self.cosmo.set({param : default_value * (1. + relative_step)})
      self.cosmo.compute()
      P_dummy_hi = compute_tracer_power_spectrum(self)(self.k,self.mu)
      self.cosmo.set({param : default_value * (1. - relative_step)})
      self.cosmo.compute()
      P_dummy_low = compute_tracer_power_spectrum(self)(self.k,self.mu)
      self.cosmo.set({param : default_value})
      return (P_dummy_hi - P_dummy_low) / (2. * default_value * relative_step)      


   def compute_dPdvecp(self, relative_step=0.01):
      '''
      Calculates the derivatives of the galaxy power spectrum
      with respect to all the parameters that are being 
      margenalized over. Returns an array with dimensions (n, Nk*Nmu), 
      where n is the number of marginalized parameters.
      '''
      result = np.array([self.compute_dPdp(param=marg_param) for marg_param in self.marg_params]) 
      self.cosmo.compute()
      return result


   def get_covariance_matrix(self): return compute_covariance_matrix(self)


   def compute_Fisher_matrix(self):
      n = len(self.marg_params)
      F = np.ones((n,n))
      C = self.get_covariance_matrix()
      # Since the matrix is diagonal, inverting it is trivial
      Cinv = np.diag(1./np.diag(C))
      dPdvecp = self.compute_dPdvecp()
      for i in range(n):
         for j in range(n):
            F[i,j] = np.dot(dPdvecp[i],np.dot(Cinv,dPdvecp[j]))
      return F


   def compute_marginalized_errors(self,F=None):
      if F is None: F = self.compute_Fisher_matrix()
      Finv = np.linalg.inv(F)
      return [np.sqrt(Finv[i,i]) for i in range(len(self.marg_params))] 


   def print_marginalized_errors(self,marg_errors=None,F=None):
      if marg_errors is None: marg_errors = self.compute_marginalized_errors(F=F)
      for i in range(len(self.marg_params)):
         marg_param = self.marg_params[i]
         print('Relative error on '+marg_param+':',marg_errors[i]/self.params[marg_param])


   def pretty_plot(self,k,p,xlabel=None,ylabel=None,c='k'):
      pretty_k = [ k[i] for i in (self.Nmu*np.linspace(0,self.Nk-1,self.Nk)).astype(int) ]
      pretty_p = [ p[i] for i in (self.Nmu*np.linspace(0,self.Nk-1,self.Nk)).astype(int) ]
      plt.semilogx(pretty_k, pretty_p, c=c)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.xlim(self.khmin,self.khmax)
      plt.show()
