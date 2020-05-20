from headers import *

class fish2point(object):
   '''
   An object that contains all the fidicual cosmological parameters.
   Capable of calculating the galaxy power spectrum and its derivatives
   with respect to cosmological parameters. Able to calculate the
   Fisher matrix for an arbitrary number of parameters.
   '''

   def __init__(self, input_params=None, marg_params=np.array(['n_s','h']), 
                      khmin=1e-4, khmax=1., Nmu=int(1e2), Nk=int(1e2),
                      Vsurvey=1.3e10, n=12., sigma_z=0.01,z=1.):

      params = {
          'output': 'mPk',
          'P_k_max_h/Mpc': 20.,
          'z_pk': '0.0,10',
          'A_s': np.exp(3.040)*1e-10,
          'n_s': 0.96824,
          'h': 0.6770,
          'N_ur': 2.0328,
          'N_ncdm': 1,
          'm_ncdm': 0.06,
          'tau_reio': 0.0568,
          'omega_b': 0.022447,
          'omega_cdm': 0.11923}

      self.params = params
      if input_params is not None: self.params = input_params
      self.dummy_params = self.params.copy()
      self.marg_params_fid = [params[marg_param] for marg_param in marg_params]
      #
      self.marg_params = marg_params
      #
      self.khmin = khmin
      self.khmax = khmax
      self.Vsurvey = Vsurvey
      self.n = n                   
      self.sigma_z = sigma_z
      self.Nk = Nk
      self.Nmu = Nmu
      self.z = z
      #
      k = np.logspace(np.log10(khmin),np.log10(khmax),Nk)
      dk = list(k[1:]-k[:-1])
      dk.append(dk[-1])
      dk = np.array(dk)
      mu = np.linspace(-1.,1.,Nmu)
      dmu = list(mu[1:]-mu[:-1])
      dmu.append(dmu[-1])
      dmu = np.array(dmu)
      #
      self.k = np.repeat(k,Nmu)
      self.dk = np.repeat(dk,Nmu)
      self.mu = np.tile(mu,Nk)
      self.dmu = np.tile(dmu,Nk)
      #
      self.P_fid = self.compute_galaxy_power_spectrum()


   def params(self,params):
      self.params = params
      self.dummy_params = self.params.copy()
      self.P_fid = self.compute_galaxy_power_spectrum()


   def compute_galaxy_power_spectrum(self, b=0.9):
      '''
      Calculated the galaxy power spectrum assuming a simple linear
      bias b between the galaxy and matter overdensity fields. 
      Returns a Nk*Nmu long array (the value of P(k,mu) for
      each (k,mu) bin)
      '''
      z = self.z
      cosmo = Class()
      cosmo.set(self.dummy_params)
      cosmo.compute()
      kk = np.logspace(-4.0,1.0,200)
      pkcb = np.array([cosmo.pk_cb(k*self.params['h'],z)*self.params['h']**3 for k in kk])
      P = interp1d(kk, pkcb, kind='linear', bounds_error=False, fill_value=0.)
      f = cosmo.scale_independent_growth_factor_f(z)
      # convert to km/(s Mpc), this conversion might be slightly off, double check it.
      Hz = cosmo.Hubble(z)*(3.086e5)
      sigma_parallel = (3.e5)*(1.+z)*self.sigma_z/Hz
      p =  lambda k,mu: P(k) * np.exp(-(k*mu*sigma_parallel)**2.) * (b+f*mu**2.)**2.
      return p(self.k,self.mu)


   def compute_dPdp(self, param, b=0.9, relative_step=0.01):
      '''
      Calculates the derivative of the galaxy power spectrum
      with respect to the input param at the fidicual cosmology
      '''
      z = self.z
      self.dummy_params[param] = self.params[param]*(1.+relative_step)
      P_dummy = self.compute_galaxy_power_spectrum()
      self.dummy_params = self.params.copy()
      return (P_dummy - self.P_fid) / (self.params[param] * relative_step)      


   def compute_dPdvecp(self, z=1., b=0.8, relative_step=0.01):
      return np.array([self.compute_dPdp(param=marg_param) for marg_param in self.marg_params])   


   def compute_covariance_matrix(self):
      prefactor = (8.*np.pi**2.) / (self.dk*self.dmu*self.Vsurvey*self.k**2.)
      diagonal_values = prefactor * (self.P_fid + 1./self.n)**2.
      return np.diag(diagonal_values)


   def compute_Fisher_matrix(self):
      num_params = len(self.marg_params)
      F = np.ones((num_params,num_params))
      C = self.compute_covariance_matrix()
      # Since the matrix is diagonal, inverting it is trivial
      Cinv = np.diag(1./np.diag(C))
      dPdvecp = self.compute_dPdvecp()
      for i in range(num_params):
         for j in range(num_params):
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


   def pretty_plot(self,k,p):
      pretty_k = [ k[i] for i in (self.Nmu*np.linspace(0,self.Nk-1,self.Nk)).astype(int) ]
      pretty_p = [ p[i] for i in (self.Nmu*np.linspace(0,self.Nk-1,self.Nk)).astype(int) ]
      plt.semilogx(pretty_k, pretty_p)
      plt.xlabel(r'hMpc$^{-1}$')
      plt.xlim(self.khmin,0.25)
      plt.show()
