from headers import *
from twoPoint import *
from twoPointNoise import *


class fisherForecast(object):
   '''
   Capable of calculating the galaxy power spectrum and its derivatives
   with respect to cosmological parameters. Able to calculate the
   Fisher matrix for an arbitrary number of parameters.
   '''

   def __init__(self, khmin=1e-4, khmax=0.25, cosmo=None, experiment=None, 
                Nmu=100, Nk=100, params=None, marg_params=np.array(['A_s,h']),
                fEDE=0., log10z_c=3.56207, thetai_scf=2.83, A_lin=0., 
                omega_lin=0.01, phi_lin=np.pi/2.):
      self.khmin = khmin
      self.khmax = khmax
      self.Nmu = Nmu
      self.Nk = Nk
      self.marg_params = marg_params
      # parameters for EDE. These are redundant if fEDE has a nonzero fiducial value.
      self.fEDE = fEDE
      self.log10z_c = log10z_c
      self.thetai_scf = thetai_scf
      # parameters for primordial wiggles
      self.A_lin = A_lin 
      self.omega_lin = omega_lin
      self.phi_lin = phi_lin

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
      if 'log(A_s)' in self.marg_params and 'A_s' not in np.array(list(params.keys())):
         print('Must included A_s in params if trying to marginalize over log(A_s).')
         return
      if 'fEDE' in np.array(list(params.keys())):
         self.fEDE = params['fEDE']
         self.log10z_c = params['log10z_c']
         self.thetai_scf = params['thetai_scf']
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


   def compute_dPdp(self, param, z, relative_step=0.01, one_sided=False, five_point=False, 
                    analytical=True, Noise=False, fEDE_step=0.1):
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
      
      # derivatives that aren't CLASS inputs that can be computed analytically
      if param == 'log(A_s)' : return P_fid
      if param == 'b' or param == 'f_NL': 
         pmatter = compute_matter_power_spectrum(self, z)(self.k)
         Hz = self.cosmo.Hubble(z)*(299792.458)/self.params['h']
         sigma_parallel = (3.e5)*(1.+z)*self.experiment.sigma_z/Hz
         f = compute_f(self, z)(self.k)
         b = self.experiment.b(z) 
         if param == 'b': return pmatter * np.exp(-(self.k*self.mu*sigma_parallel)**2.) * 2. * (b+f*self.mu**2.)
         # derivative wrt f_NL. 
         D = 0.76 * self.cosmo.scale_independent_growth_factor(z) # normalized so D(a) = a in the MD era
         # hacky way of calculating the transfer function
         T = np.sqrt(pmatter/self.k**self.params['n_s'])
         T /= T[0]
         fNL_factor = 3.*1.68*(b-1.)*self.params['omega_cdm']*(100.*self.params['h'])**2.
         fNL_factor /= D * self.k**2. * T * 299792.458**2.
         return pmatter * np.exp(-(self.k*self.mu*sigma_parallel)**2.) * 2. * (b+f*self.mu**2.) * fNL_factor
         
      if param == 'N' : return np.ones(len(self.k))
    
      # derivative of early dark energy parameters (Hill+2020)
      if param == 'fEDE' and self.fEDE == 0.:
         EDE_params = {'log10z_c': self.log10z_c,'fEDE': fEDE_step,'thetai_scf': self.thetai_scf,
                       'Omega_Lambda':0.0,'Omega_fld':0,'Omega_scf':-1,
                       'n_scf':3,'CC_scf':1,'scf_tuning_index':3,
                       'scf_parameters':'1, 1, 1, 1, 1, 0.0',
                       'attractor_ic_scf':'no'}
         self.cosmo.set(EDE_params)
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         self.cosmo.set({'fEDE':2.*fEDE_step})
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self,z)(self.k,self.mu)
         self.cosmo = Class()
         self.cosmo.set(self.params)
         self.cosmo.compute()
         return (P_fid - (4./3.) * P_dummy_hi + (1./3.) * P_dummy_higher) / ((-2./3.) * fEDE_step)
      
      if (param == 'log10z_c' or param == 'thetai_scf') and self.fEDE == 0.:
         print('Attempted to marginalize over log10z_c or thetai_scf when fEDE has a fiducial value of 0.')
         return
    
      # derivatives of parameters related to primordial features (Beutler+20)
      P_fid_no_wiggles = compute_tracer_power_spectrum(self,z,Wiggles=False)(self.k,self.mu)
      if param == 'A_lin': return P_fid_no_wiggles * np.sin(self.omega_lin * self.k + self.phi_lin)
      if (param == 'omega_lin' or param == 'phi_lin') and self.A_lin == 0.:
         print('Attemped to marginalize over omega_lin or phi_lin when A_lin has a fiducial value of 0.')
         return
      if param == 'omega_lin': return P_fid_no_wiggles * self.A_lin * np.cos(self.omega_lin * self.k + self.phi_lin) * self.k
      if param == 'phi_lin': return P_fid_no_wiggles * self.A_lin * np.cos(self.omega_lin * self.k + self.phi_lin)
    
      # brute force numerical differentiation
      default_value = self.params[param]
               
      def set_param(value):
         self.cosmo.set({param : value})
         self.params[param] = value
        
      if one_sided:
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

      # defaults to a two sided derivative
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


   def non_default_fidval(self, param):
      '''
      For parameters that aren't default CLASS(-EDE) inputs.
      Returns their fiducial value. 
      '''
      if param == 'log(A_s)': return np.log(self.params['A_s'])
      if param == 'b': return self.experiment.b(self.zcenters[0]) # assumes b(z) = const.
      if param == 'N': return 1./self.experiment.n[0] # assumes N(z) = const.
      if param == 'fEDE': return 0.
      if param == 'A_lin': return 0.
      if param == 'f_NL': return 0.
      return


   def compute_marginalized_errors(self, F=None):
      if F is None: F = self.compute_Fisher_matrix()
      Finv = np.linalg.inv(F)
      return [np.sqrt(Finv[i,i]) for i in range(len(self.marg_params))] 
    

   def print_marginalized_errors(self, marg_errors=None, F=None):
      if marg_errors is None: marg_errors = self.compute_marginalized_errors(F=F)
      for i in range(len(self.marg_params)):
         marg_param = self.marg_params[i]
         print('Error on '+marg_param+':',marg_errors[i])
        
        
   def get_f_at_fixed_mu(self,f,mu):
      '''
      For a function f(k,mu), which is represented as an array of length Nmu*Nk,
      return a function f(k)
      '''
      closest_index = np.where(self.mu >= mu)[0][0]
      indices = np.array([closest_index+n*self.Nmu for n in np.linspace(0,self.Nk-1,self.Nk)])
      f_fixed = [f[i] for i in indices.astype(int)]
      k = [self.k[i] for i in indices.astype(int)]
      f = interp1d(k,f_fixed,kind='linear')
      return f
    
   
   ############################################################################################################
   ############################################################################################################
   # Functions for plotting


   def pretty_plot(self,k,curves,xlabel=None,ylabel=None,c=None,datalabels=None,legendtitle=None,filename=None):
      '''
      Plot's a function f(k,mu) at mu=0.
      '''
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
    
    
   def pretty_label(self, param):
      if param == 'log(A_s)': return r'$\log(A_s)$'
      elif param == 'n_s': return r'$n_s$'
      elif param == 'h': return r'$h$'
      elif param == 'omega_b': return r'$\Omega_b$'
      elif param == 'omega_cdm': return r'$\Omega_c$'
      elif param == 'm_ncdm': return r'$\sum m_\nu$'
      elif param == 'N': return r'$N$'
      elif param == 'b': return r'$b$'
      elif param == 'A_s': return r'$A_s$'
      elif param == 'f_NL': return r'$f_{NL}$'
      elif param == 'fEDE': return r'$f_{EDE}$'
      else: return param
    
    
   def plot_error_ellipse(self, F=None, param1=None, param2=None, cs=['lightblue','blue']):
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
           
    
   def plot_posterior_matrix(self, F=None, cs=['lightblue','blue'], param_subset=None):
      if param_subset is None: param_subset = self.marg_params
      if F is None: F = self.compute_Fisher_matrix()
      n = len(param_subset)
      fig, axs = plt.subplots(n, n,figsize=(n*7,n*7))#, gridspec_kw={'hspace': 0.1})
      for i in range(n):
         for j in range(n):
            param1 = param_subset[j]
            param2 = param_subset[i]  
            if i < j: 
               axs[i,j].axis('off')
            elif i == j:
               try: fid_value = self.params[param1]
               except KeyError: fid_value = self.non_default_fidval(param1)
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
                  try: fid_val1 = self.params[param1]
                  except KeyError: fid_val1 = self.non_default_fidval(param1)
                  try: fid_val2 = self.params[param2]
                  except KeyError: fid_val2 = self.non_default_fidval(param2)
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