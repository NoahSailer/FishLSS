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
      self.P_fid = None     # fidicual power spectra at the center of each redshift bin
      self.Vsurvey = None   # comoving volume [Mpc/h]^3 in each redshift bin
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
      self.P_fid = np.array([compute_tracer_power_spectrum(self,z)(self.k,self.mu) for z in experiment.zcenters])
      self.Vsurvey = np.array([self.comov_vol(experiment.zedges[i],experiment.zedges[i+1]) for i in range(len(experiment.zedges)-1)])


   def comov_vol(self,zmin,zmax):
      '''
      Returns the comoving volume in Mpc^3/h^3 between 
      zmin and zmax assuming that the universe is flat.
      '''
      vsmall = (4*np.pi/3) * ((1.+zmin)*self.cosmo.angular_distance(zmin))**3.
      vbig = (4*np.pi/3) * ((1.+zmax)*self.cosmo.angular_distance(zmax))**3.
      return self.experiment.fsky*(vbig - vsmall)*self.params['h']**3.


   def compute_dPdp(self, param, z, relative_step=0.01, one_sided=False, five_point=False, analytical=True, Noise=False):
      '''
      Calculates the derivative of the galaxy power spectrum
      with respect to the input parameter around the fidicual
      cosmology and redshift (for the respective bin). Returns
      an array of length Nk*Nmu.
      '''
    
      P_fid = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
        
      if analytical:
         # for derivatives whose quantities are CLASS inputs, but can be computed analytically
         if param == 'n_s': return P_fid * np.log(self.params['h']*self.k/0.05)
         if param == 'A_s': return P_fid / self.params['A_s']
      
      # derivatives that aren't class inputs that can be computed analytically
      if param == 'log(A_s)' : return P_fid
      if param == 'b' : 
         pmatter = compute_matter_power_spectrum(self, z)(self.k)
         Hz = self.cosmo.Hubble(z)*(299792.458)/self.params['h']
         sigma_parallel = (3.e5)*(1.+z)*self.experiment.sigma_z/Hz
         f = compute_f(self, z)(self.k)
         b = self.experiment.b(z) 
         return pmatter * np.exp(-(self.k*self.mu*sigma_parallel)**2.) * 2. * (b+f*self.mu**2.)
      if param == 'N' : return np.ones(len(self.k))
    
      default_value = self.params[param]
        
      def set_param(value):
         self.cosmo.set({param : value})
         self.params[param] = value

      # code for numerical differentiation
      if one_sided:
         self.cosmo.compute()
         P_fid = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value * (1. + relative_step))
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value * (1. + 2.*relative_step))
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value)
         self.cosmo.compute()
         return (P_fid - (4./3.) * P_dummy_hi + (1./3.) * P_dummy_higher) / ((-2./3.) * default_value * relative_step)

      if five_point:
         set_param(default_value * (1. + relative_step))
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value * (1. + 2.*relative_step))
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value * (1. - relative_step))
         self.cosmo.compute()
         P_dummy_low = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value * (1. - 2.*relative_step))
         self.cosmo.compute()
         P_dummy_lower = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         set_param(default_value)
         self.cosmo.compute()
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * default_value * relative_step)

      set_param(default_value * (1. + relative_step))
      self.cosmo.compute()
      P_dummy_hi = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
      set_param(default_value * (1. - relative_step))
      self.cosmo.compute()
      P_dummy_low = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
      set_param(default_value)
      self.cosmo.compute()
      return (P_dummy_hi - P_dummy_low) / (2. * default_value * relative_step)      


   def compute_dPdvecp(self, z, relative_step=0.01, five_point=False):
      '''
      Calculates the derivatives of the galaxy power spectrum
      with respect to all the parameters that are being 
      margenalized over. Returns an array with dimensions (n, Nk*Nmu), 
      where n is the number of marginalized parameters.
      '''
      result = np.array([self.compute_dPdp(param=marg_param, z=z, five_point=five_point) for marg_param in self.marg_params]) 
      return result


   def get_covariance_matrix(self, zbin_index): return compute_covariance_matrix(self,zbin_index)


   def compute_Fisher_matrix_for_specific_zbin(self, zbin_index, five_point=False):
      z = self.experiment.zcenters[zbin_index]
      n = len(self.marg_params)
      F = np.ones((n,n))
      C = self.get_covariance_matrix(zbin_index)
      # Since the matrix is diagonal, inverting it is trivial
      Cinv = np.diag(1./np.diag(C))
      dPdvecp = self.compute_dPdvecp(z, five_point=five_point)
      for i in range(n):
         for j in range(n):
            F[i,j] = np.dot(dPdvecp[i],np.dot(Cinv,dPdvecp[j]))
      return F


   def compute_Fisher_matrix(self, five_point=False):
      F = self.compute_Fisher_matrix_for_specific_zbin(0,five_point=five_point)
      for i in range(1,len(self.experiment.zedges)-1):
         F += self.compute_Fisher_matrix_for_specific_zbin(i)
      return F


   def compute_marginalized_errors(self,F=None):
      if F is None: F = self.compute_Fisher_matrix()
      Finv = np.linalg.inv(F)
      return [np.sqrt(Finv[i,i]) for i in range(len(self.marg_params))] 


   def print_marginalized_errors(self,marg_errors=None,F=None):
      if marg_errors is None: marg_errors = self.compute_marginalized_errors(F=F)
      for i in range(len(self.marg_params)):
         marg_param = self.marg_params[i]
         if marg_param == 'log(A_s)': fid_val = np.log(self.params['A_s'])
         else: fid_val = self.params[marg_param]
         print('Relative error on '+marg_param+':',marg_errors[i]/fid_val)
        
        
   def get_f_at_fixed_mu(self,f,mu):
      closest_index = np.where(self.mu >= mu)[0][0]
      indices = np.array([closest_index+n*self.Nmu for n in np.linspace(0,self.Nk-1,self.Nk)])
      f_fixed = [f[i] for i in indices.astype(int)]
      k = [self.k[i] for i in indices.astype(int)]
      f = interp1d(k,f_fixed,kind='linear')
      return f


   def pretty_plot(self,k,curves,xlabel=None,ylabel=None,c=None,datalabels=None,legendtitle=None,filename=None):
      plt.figure(figsize=(14,12))
      pretty_k = [ k[i] for i in (self.Nmu*np.linspace(0,self.Nk-1,self.Nk)).astype(int) ]
      for i in range(len(curves)):
         curve = curves[i]
         pretty_p = [ curve[j] for j in (self.Nmu*np.linspace(0,self.Nk-1,self.Nk)).astype(int) ]
         plt.semilogx(pretty_k, pretty_p, c=c[i],label=datalabels[i], lw=4)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.legend(loc=0,title=legendtitle)
      plt.xlim(self.khmin,self.khmax)
      plt.tight_layout()
      if filename is not None: plt.savefig(filename+'.pdf')
      plt.show()
    
    
   def plot_error_ellipse(self,F=None,param1=None,param2=None,cs=['lightblue','blue']):
      '''
      param1 and param2 must be marginalized parameters.
      '''
      plt.figure(figsize=(14,12))
      if param1 is None or param2 is None: 
            print('parameters have not been set')
      if F is None: F = self.compute_Fisher_matrix()
      # find the 2x2 submatrix that corresponds to the two parameters of interest, which I call G
      index1 = np.where(self.marg_params == param1)[0][0]
      index2 = np.where(self.marg_params == param2)[0][0]
      G = np.array([[F[index1,index1],F[index1,index2]],[F[index1,index2],F[index2,index2]]])
      # write G = S D S^T for a diagonal matrix D
      eigenvals,S = np.linalg.eig(G)
      D = np.diag(eigenvals)
      # find the solution of the ellipse in the diagonal basis
      theta = np.linspace(0.,2.*np.pi,1000)
      for i,f in enumerate(np.array([0.167,0.434])):
         xprime = np.cos(theta) / np.sqrt(f * D[0,0])
         yprime = np.sin(theta) / np.sqrt(f * D[1,1])
         # transform back to the parameter basis
         x,y = np.dot(S, np.array([xprime,yprime]))
         x += self.params[param1]
         y += self.params[param2]
         plt.fill(x,y,c=cs[i],alpha=0.6)
      plt.xlabel(self.pretty_label(param1))
      plt.ylabel(self.pretty_label(param2))
      plt.tight_layout()
      plt.show()
        
      
   def pretty_label(self,param):
      if param == 'log(A_s)': return r'$\log(A_s)$'
      elif param == 'n_s': return r'$n_s$'
      elif param == 'h': return r'$h$'
      elif param == 'omega_b': return r'$\Omega_b$'
      elif param == 'omega_cdm': return r'$\Omega_c$'
      elif param == 'm_ncdm': return r'$\sum m_\nu$'
      elif param == 'N': return r'$N$'
      elif param == 'b': return r'$b$'
      elif param == 'A_s': return r'$A_s$'
      else: return param
    
    
   def plot_posterior_matrix(self,F=None,cs=['lightblue','blue']):
      if F is None: F = self.compute_Fisher_matrix()
      n = len(self.marg_params)
      fig, axs = plt.subplots(n, n,figsize=(n*7,n*7))#, gridspec_kw={'hspace': 0.1})
      for i in range(n):
         for j in range(n):
            param1 = self.marg_params[j]
            param2 = self.marg_params[i]  
            if i < j: 
               axs[i,j].axis('off')
            elif i == j:
               if param1 == 'log(A_s)': fid_value = np.log(self.params['A_s'])
               else: fid_value = self.params[param1]
               sigma = np.sqrt(np.linalg.inv(F)[i,i])
               domain = np.linspace(fid_value-4.*sigma,fid_value+4.*sigma,100)
               gauss = lambda x: np.exp(-(x-fid_value)**2./(2.*sigma**2.))/np.sqrt(2.*np.pi*sigma**2.)
               axs[i,j].plot(domain,gauss(domain),c='k',lw=4)
               axs[i,j].set_yticklabels([])
               axs[i,j].get_shared_x_axes().join(axs[i,j], axs[n-1,j])
               if i != n-1: axs[i,j].xaxis.set_ticklabels([])
               axs[i,j].yaxis.set_visible(False)
            else:    
               axs[i,j].get_shared_x_axes().join(axs[i,j], axs[n-1,j])
               axs[i,j].get_shared_y_axes().join(axs[i,j], axs[i,0])
               if i != n-1: axs[i,j].xaxis.set_ticklabels([])
               if j != 0: axs[i,j].yaxis.set_ticklabels([])
               G = np.array([[F[i,i],F[i,j]],[F[i,j],F[j,j]]])
               # write G = S D S^T for a diagonal matrix D
               eigenvals,S = np.linalg.eig(G)
               D = np.diag(eigenvals)
               # find the solution of the ellipse in the diagonal basis
               theta = np.linspace(0.,2.*np.pi,1000)
               for k,f in enumerate(np.array([0.167,0.434])):
                  xprime = np.cos(theta) / np.sqrt(f * D[0,0])
                  yprime = np.sin(theta) / np.sqrt(f * D[1,1])
                  # transform back to the parameter basis
                  x,y = np.dot(S, np.array([xprime,yprime]))
                  if param1 == 'log(A_s)': fid_val1 = np.log(self.params['A_s'])
                  else: fid_val1 = self.params[param1]
                  if param2 == 'log(A_s)': fid_val2 = np.log(self.params['A_s'])
                  else: fid_val2 = self.params[param2]
                  x += fid_val1
                  y += fid_val2
                  axs[i,j].fill(x,y,c=cs[k],alpha=0.6)
            if i == n-1: 
                axs[i,j].set_xlabel(self.pretty_label(param1))
            if j == 0: 
                axs[i,j].set_ylabel(self.pretty_label(param2))
      plt.subplots_adjust(wspace=0.05, hspace=0.03)
      plt.tight_layout()
      plt.show()