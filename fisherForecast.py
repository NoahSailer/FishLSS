from headers import *
from twoPoint import *
from twoPointNoise import *
from castorina import castorinaBias,castorinaPn
from multiprocessing import Pool
from functools import partial


class fisherForecast(object):
   '''
   Capable of calculating the galaxy power spectrum and its derivatives
   with respect to cosmological parameters. Able to calculate the
   Fisher matrix for an arbitrary number of parameters.
   '''

   def __init__(self, khmin=1e-4, khmax=0.25, cosmo=None, experiment=None, 
                Nmu=50, Nk=50, params=None, marg_params=np.array(['A_s','h']),
                fEDE=0., log10z_c=3.56207, thetai_scf=2.83, A_lin=0., 
                omega_lin=0.01, phi_lin=np.pi/2., velocileptors=False,
                linear=False, name='toy_model',smooth=False):
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
      self.velocileptors = velocileptors
      self.linear = linear
      self.name = name
      self.smooth = smooth

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
      #dk.append(dk[-1])
      dk.insert(0,dk[0])
      dk = np.array(dk)
      mu = np.linspace(0.,1.,self.Nmu)
      dmu = list(mu[1:]-mu[:-1])
      dmu.append(dmu[-1])
      dmu = np.array(dmu)
      self.k = np.repeat(k,self.Nmu)
      self.dk = np.repeat(dk,self.Nmu)
      self.mu = np.tile(mu,self.Nk)
      self.dmu = np.tile(dmu,self.Nk)
      self.P_fid = np.array([compute_tracer_power_spectrum(self,z) for i,z in enumerate(experiment.zcenters)])
      self.Vsurvey = np.array([self.comov_vol(experiment.zedges[i],experiment.zedges[i+1]) \
                               for i in range(len(experiment.zedges)-1)])
        
        
   def comov_vol(self,zmin,zmax):
      '''
      Returns the comoving volume in Mpc^3/h^3 between 
      zmin and zmax assuming that the universe is flat.
      Includes the fsky of the experiment.
      '''
      vsmall = (4*np.pi/3) * ((1.+zmin)*self.cosmo.angular_distance(zmin))**3.
      vbig = (4*np.pi/3) * ((1.+zmax)*self.cosmo.angular_distance(zmax))**3.
      return self.experiment.fsky*(vbig - vsmall)*self.params['h']**3.


   def compute_dPdp(self, param, z, relative_step=-1., absolute_step=-1., 
                    one_sided=False, five_point=False):
      '''
      Calculates the derivative of the galaxy power spectrum
      with respect to the input parameter around the fidicual
      cosmology and redshift (for the respective bin). Returns
      an array of length Nk*Nmu.
      '''
      default_step = {'log(A_s)':0.1,'A_s':0.1,'omega_cdm':0.2,'n_s':0.1,'tau_reio':0.3,'m_ncdm':0.05}
        
      if relative_step == -1.: 
         try: relative_step = default_step[param]
         except: relative_step = 0.01
      if absolute_step == -1.: 
         try: absolute_step = default_step[param]
         except: absolute_step = 0.01
    
      P_fid = compute_tracer_power_spectrum(self,z)
               
      if param == 'N' : return np.ones(len(self.k))
        
      if param == 'Tb': 
         Ez = fishcast.cosmo.Hubble(z)/fishcast.cosmo.Hubble(0)
         Ohi = 4e-4*(1+z)**0.6
         Tb = 188e-3*(fishcast.params['h'])/Ez*Ohi*(1+z)**2
         return 2. * ( P_fid + castorinaPn(z)(self.k,self.mu) ) / Tb
    
      def dPdk(): 
         dP = P_fid - np.roll(P_fid,self.Nmu)
         dk = self.k - np.roll(self.k,self.Nmu)
         return dP/dk
      
      def dPdmu():
          def dPdmu_fixed_k(i): 
             dP = P_fid[self.Nmu*i:self.Nmu*(i+1)] - np.roll(P_fid[self.Nmu*i:self.Nmu*(i+1)],1)
             dmu = self.dmu[0]
             dPdmu = dP/dmu
             dPdmu[0] = 0.
             return list(dPdmu)
          result = dPdmu_fixed_k(0)
          for i in range(1,self.Nk): result = result + dPdmu_fixed_k(i)
          return np.array(result)
          
      if param == 'Da': 
         K,MU = self.k,self.mu
         result = -2.*P_fid + K*(1.-MU**2.)*dPdk() - MU*(1.-MU**2.)*dPdmu()
         return result/(self.cosmo.angular_distance(z) * self.params['h']) 
        
      if param == 'Hz':
         K,MU = self.k,self.mu
         result = P_fid - K*MU**2.*dPdk() - MU*(1.-MU**2.)*dPdmu()
         Hz = self.cosmo.Hubble(z)*(299792.458)/self.params['h']
         return result/Hz
        
      # This list is getting ugly and somewhat out of hand. This should be coded up more cleverly...
    
      if param == 'A_lin': 
         P_dummy_hi = compute_tracer_power_spectrum(self,z,A_lin=self.A_lin+absolute_step)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,A_lin=self.A_lin+2.*absolute_step)
         P_dummy_low = compute_tracer_power_spectrum(self,z,A_lin=self.A_lin-absolute_step)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,A_lin=self.A_lin-2.*absolute_step)
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * absolute_step)
 
      if (param == 'omega_lin' or param == 'phi_lin') and self.A_lin == 0.:
         print('Attemped to marginalize over omega_lin or phi_lin when A_lin has a fiducial value of 0.')
         return
    
      if param == 'omega_lin':
         P_dummy_hi = compute_tracer_power_spectrum(self,z,omega_lin=self.omega_lin*(1.+relative_step))
         P_dummy_higher = compute_tracer_power_spectrum(self,z,omega_lin=self.omega_lin*(1.+2.*relative_step))
         P_dummy_low = compute_tracer_power_spectrum(self,z,omega_lin=self.omega_lin*(1.-relative_step))
         P_dummy_lower = compute_tracer_power_spectrum(self,z,omega_lin=self.omega_lin*(1.-2.*relative_step))
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * self.omega_lin * relative_step)
            
      if param == 'phi_lin':  
         P_dummy_hi = compute_tracer_power_spectrum(self,z,phi_lin=self.phi_lin*(1.+relative_step))
         P_dummy_higher = compute_tracer_power_spectrum(self,z,phi_lin=self.phi_lin*(1.+2.*relative_step))
         P_dummy_low = compute_tracer_power_spectrum(self,z,phi_lin=self.phi_lin*(1.-relative_step))
         P_dummy_lower = compute_tracer_power_spectrum(self,z,phi_lin=self.phi_lin*(1.-2.*relative_step))
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * self.phi_lin * relative_step)
        
      if param == 'b2':
         P_dummy_hi = compute_tracer_power_spectrum(self,z,bE2=absolute_step)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,bE2=2.*absolute_step)
         P_dummy_low = compute_tracer_power_spectrum(self,z,bE2=-absolute_step)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,bE2=-2.*absolute_step)
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * absolute_step)
        
      if param == 'bs':
         P_dummy_hi = compute_tracer_power_spectrum(self,z,bEs=absolute_step)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,bEs=2.*absolute_step)
         P_dummy_low = compute_tracer_power_spectrum(self,z,bEs=-absolute_step)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,bEs=-2.*absolute_step)
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * absolute_step)
      
      if param == 'alpha0':
         P_dummy_hi = compute_tracer_power_spectrum(self,z,alpha0=absolute_step)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,alpha0=2.*absolute_step)
         P_dummy_low = compute_tracer_power_spectrum(self,z,alpha0=-absolute_step)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,alpha0=-2.*absolute_step)
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * absolute_step)
      
      if param == 'alpha2':
         # Strange factor of -1 here...
         P_dummy_hi = compute_tracer_power_spectrum(self,z,alpha2=absolute_step)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,alpha2=2.*absolute_step)
         P_dummy_low = compute_tracer_power_spectrum(self,z,alpha2=-absolute_step)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,alpha2=-2.*absolute_step)
         return -1.*(-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * absolute_step)
      
      if param == 'alpha4':
         P_dummy_hi = compute_tracer_power_spectrum(self,z,alpha4=absolute_step)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,alpha4=2.*absolute_step)
         P_dummy_low = compute_tracer_power_spectrum(self,z,alpha4=-absolute_step)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,alpha4=-2.*absolute_step)
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * absolute_step)
        
      if param == 'sn2':
         # Strange factor of -1 here...
         P_dummy_hi = compute_tracer_power_spectrum(self,z,sn2=10.)
         P_dummy_higher = compute_tracer_power_spectrum(self,z,sn2=20.)
         P_dummy_low = compute_tracer_power_spectrum(self,z,sn2=-10.)
         P_dummy_lower = compute_tracer_power_spectrum(self,z,sn2=-20.)
         return -1.*(-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * 10.)
        
      if param == 'b' or param == 'f_NL' or param == 'f_NL_eq' or param == 'f_NL_orth': 
         b_fid = compute_b(self,z)
         P_dummy_hi = compute_tracer_power_spectrum(self,z,b=b_fid*(1.+relative_step))
         P_dummy_higher = compute_tracer_power_spectrum(self,z,b=b_fid*(1.+2.*relative_step))
         P_dummy_low = compute_tracer_power_spectrum(self,z,b=b_fid*(1.-relative_step))
         P_dummy_lower = compute_tracer_power_spectrum(self,z,b=b_fid*(1.-2.*relative_step))
         dPdb = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * self.phi_lin * relative_step)
         if param == 'b': return dPdb
         # derivative wrt f_NL
         D = 0.76 * self.cosmo.scale_independent_growth_factor(z) # normalized so D(a) = a in the MD era
         # hacky way of calculating the transfer function
         pmatter = compute_matter_power_spectrum(self, z, linear=True)
         T = np.sqrt(pmatter/self.k**self.params['n_s'])
         T /= T[0]
         fNL_factor = 3.*1.68*(b_fid-1.)*(self.params['omega_cdm']/self.params['h']**2.)*100.**2.
         fNL_factor /= D * self.k**2. * T * 299792.458**2.
         if param == 'f_NL_eq': fNL_factor *= 3. * (1.*self.k)**2.
         if param == 'f_NL_orth': fNL_factor *= -3. * (1.*self.k)
         return dPdb * fNL_factor
            
      # derivative of early dark energy parameters (Hill+2020)
      if param == 'fEDE' and self.fEDE == 0.:
         EDE_params = {'log10z_c': self.log10z_c,'fEDE': absolute_step,'thetai_scf': self.thetai_scf,
                       'Omega_Lambda':0.0,'Omega_fld':0,'Omega_scf':-1,
                       'n_scf':3,'CC_scf':1,'scf_tuning_index':3,
                       'scf_parameters':'1, 1, 1, 1, 1, 0.0',
                       'attractor_ic_scf':'no'}
         self.cosmo.set(EDE_params)
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self,z)
         self.cosmo.set({'fEDE':2.*absolute_step})
         self.cosmo.compute()
         P_dummy_hi2 = compute_tracer_power_spectrum(self,z)
         self.cosmo.set({'fEDE':3.*absolute_step})
         self.cosmo.compute()
         P_dummy_hi3 = compute_tracer_power_spectrum(self,z)
         self.cosmo.set({'fEDE':4.*absolute_step})
         self.cosmo.compute()
         P_dummy_hi4 = compute_tracer_power_spectrum(self,z)
         self.cosmo = Class()
         self.cosmo.set(self.params)
         self.cosmo.compute() 
         return (-3.*P_dummy_hi4 + 16.*P_dummy_hi3 - 36.*P_dummy_hi2 + 48.*P_dummy_hi - 25.*P_fid)/(12.*absolute_step)
      
      if (param == 'log10z_c' or param == 'thetai_scf') and self.fEDE == 0.:
         print('Attempted to marginalize over log10z_c or thetai_scf when fEDE has a fiducial value of 0.')
         return
    
      result = np.zeros(len(self.k))
      
      # brute force numerical differentiation
      flag = False 
      if param == 'log(A_s)' : 
         flag = True
         param = 'A_s'  
        
      default_value = self.params[param] 
        
      if param == 'm_ncdm' and self.params['N_ncdm']>1:
         # CLASS takes a string as an input when there is more than one massless neutrino
         default_value_float = np.array(list(map(float,list(default_value.split(',')))))
         Mnu = sum(default_value_float)
         up = ','.join(list(map(str,list(default_value_float+relative_step*Mnu/self.params['N_ncdm']))))
         upup = ','.join(list(map(str,list(default_value_float+2.*relative_step*Mnu/self.params['N_ncdm']))))
         down = ','.join(list(map(str,list(default_value_float-relative_step*Mnu/self.params['N_ncdm']))))
         downdown = ','.join(list(map(str,list(default_value_float-2.*relative_step*Mnu/self.params['N_ncdm']))))
         step = Mnu*relative_step
      else:
         up = default_value * (1. + relative_step)
         if default_value == 0.: up = default_value + absolute_step
         upup = default_value * (1. + 2.*relative_step)
         if default_value == 0.: upup = default_value + 2.*absolute_step
         down = default_value * (1. - relative_step)
         if default_value == 0.: down = default_value - absolute_step
         downdown = default_value * (1. - 2.*relative_step)
         if default_value == 0.: downdown = default_value - 2.*absolute_step
         step = default_value * relative_step
         if default_value == 0.: step = absolute_step
            
    
      def set_param(value):
         self.cosmo.set({param : value})
         self.params[param] = value
        
      if one_sided:
         set_param(up)
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self,z)
         set_param(upup)
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self,z)
         set_param(default_value)
         self.cosmo.compute()
         result += (P_fid - (4./3.) * P_dummy_hi + (1./3.) * P_dummy_higher) / ((-2./3.) * step)
         if flag: result *= self.params['A_s']
         return result

      if five_point:
         set_param(up)
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(self,z)
         set_param(upup)
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(self,z)
         set_param(down)
         self.cosmo.compute()
         P_dummy_low = compute_tracer_power_spectrum(self,z)
         set_param(downdown)
         self.cosmo.compute()
         P_dummy_lower = compute_tracer_power_spectrum(self,z)
         set_param(default_value)
         self.cosmo.compute()
         result += (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
         if flag: result *= self.params['A_s']
         return result

      # defaults to a two sided derivative
      set_param(up)
      self.cosmo.compute()
      P_dummy_hi = compute_tracer_power_spectrum(self,z)
      set_param(down)
      self.cosmo.compute()
      P_dummy_low = compute_tracer_power_spectrum(self,z)
      set_param(default_value)
      self.cosmo.compute()
      result += (P_dummy_hi - P_dummy_low) / (2. * step)
      if flag: result *= self.params['A_s']
      return result
   

   def Sigma2(self, z): return sum(compute_matter_power_spectrum(self,z,linear=True)*self.dk*self.dmu) / (6.*np.pi**2.)

    
   def kmax_constraint(self, z, alpha=1.): return self.k < alpha/np.sqrt(self.Sigma2(z))
    
    
   def compute_wedge(self, z):
      if not self.experiment.HI: return self.k > 0.008 # set this to be bigger than khmin to aoid edge effects with velocileptors
      #
      kparallel = self.k*self.mu
      kperpendicular = self.k*np.sqrt(1.-self.mu**2.)
      #
      chi = (1.+z)*self.cosmo.angular_distance(z)*self.params['h'] # Mpc/h
      Hz = self.cosmo.Hubble(z)*(299792.458)/self.params['h'] # h km / s Mpc
      c = 299792.458 # km/s
      lambda21 = 0.21 * (1+z) # meters
      D_eff = 6. * np.sqrt(0.7) # meters, MAKE THIS A VARIABLE CONTROLLED BY THE EXPERIMENT
      theta_w = self.experiment.N_w * 1.22 * lambda21 / (2.*D_eff)
      #
      wedge = kparallel > chi*Hz*np.sin(theta_w)*kperpendicular/(c*(1.+z))
      kparallel_constraint = kparallel > self.experiment.kparallel_min
      return wedge*kparallel_constraint
    
    
   def compute_derivatives(self, five_point=True, parameters=None, z=None):
      if parameters is not None: 
         for i,p in enumerate(parameters):
            dPdp = self.compute_dPdp(param=p, z=z[i], five_point=five_point)
            if p == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(int(100*z[i]))+'.txt'
            elif p == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(int(100*z[i]))+'.txt'
            else: filename = p+'_'+str(int(100*z[i]))+'.txt'
            np.savetxt('output/'+self.name+'/derivatives/'+filename,dPdp)
         return
      zs = self.experiment.zcenters
      for z in zs:
         for marg_param in self.marg_params:
            dPdp = self.compute_dPdp(param=marg_param, z=z, five_point=five_point)
            if marg_param == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(int(100*z))+'.txt'
            elif marg_param == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(int(100*z))+'.txt'
            else: filename = marg_param+'_'+str(int(100*z))+'.txt'
            np.savetxt('output/'+self.name+'/derivatives/'+filename,dPdp)
            
            
   def check_derivatives(self):
      directory = 'output/'+self.name+'/derivatives'
      for root, dirs, files in os.walk(directory, topdown=False):
         for file in files:
            filename = os.path.join(directory, file)
            dPdp = np.genfromtxt(filename)
            plt.figure(figsize=(6,6))
            k = np.linspace(0.008,1.,1000)
            plt.semilogx(k,self.get_f_at_fixed_mu(dPdp,0.)(k),color='b')
            plt.semilogx(k,self.get_f_at_fixed_mu(dPdp,0.3)(k),color='g')
            plt.semilogx(k,self.get_f_at_fixed_mu(dPdp,0.7)(k),color='r')
            plt.xlabel(r'$k$ [h/Mpc]')
            plt.show()
            plt.clf()
            print(file)
                  
                
   def gen_fisher(self,basis,log10z_c=-1.,omega_lin=-1.,alpha=1.,zbin_indices=None):
      if log10z_c == -1. : log10z_c = self.log10z_c
      if omega_lin == -1. : omega_lin = self.omega_lin
      def fish(zbin_index):
         n = len(basis)
         z = self.experiment.zcenters[zbin_index]
         dPdvecp = np.array([None]*n)
         for i,param in enumerate(basis): 
            if param == 'fEDE': filename = 'fEDE_'+str(int(1000.*log10z_c))+'_'+str(int(100*z))+'.txt'
            elif param == 'A_lin': filename = 'A_lin_'+str(int(100.*omega_lin))+'_'+str(int(100*z))+'.txt'
            else: filename = param+'_'+str(int(100*z))+'.txt'
            try:
               dPdvecp[i] = np.genfromtxt('output/'+self.name+'/derivatives/'+filename)
            except:
               print('Have not calculated derivative of '+ param)
               return
         F = np.ones((n,n))
         Cinv = 1./self.get_covariance_matrix(zbin_index)
         for i in range(n):
            for j in range(n):
               F[i,j] = np.sum(dPdvecp[i]*Cinv*dPdvecp[j]*self.compute_wedge(z)*self.kmax_constraint(z,alpha))
         return F
      if zbin_indices is not None: return sum([fish(i) for i in zbin_indices])
      result = fish(0)
      for i in range(1,len(self.experiment.zedges)-1): result+=fish(i)
      return result
      

   def compute_dPdvecp(self, z, five_point=False):
      '''
      Calculates the derivatives of the galaxy power spectrum
      with respect to all the parameters that are being 
      margenalized over. Returns an array with dimensions (n, Nk*Nmu), 
      where n is the number of marginalized parameters.
      '''
      wedge = self.compute_wedge(z)
      result = []
      for marg_param in self.marg_params:
         dPdp = self.compute_dPdp(param=marg_param, z=z, five_point=five_point)
         result.append(dPdp*wedge*self.kmax_constraint(z))
      return np.array(result)


   def get_covariance_matrix(self, zbin_index): return compute_covariance_matrix(self,zbin_index)


   def compute_Fisher_matrix_for_specific_zbin(self, zbin_index, five_point=False):
      z = self.experiment.zcenters[zbin_index]
      n = len(self.marg_params)
      F = np.ones((n,n))
      C = self.get_covariance_matrix(zbin_index)
      # Since the matrix is diagonal, inverting it is trivial
      Cinv = 1./C
      dPdvecp = self.compute_dPdvecp(z, five_point=five_point)
      for i in range(n):
         for j in range(n):
            F[i,j] = np.sum(dPdvecp[i]*Cinv*dPdvecp[j])
      return F


   def compute_Fisher_matrix(self, five_point=False):
      F = self.compute_Fisher_matrix_for_specific_zbin(0,five_point=five_point)
      result = [F]
      for i in range(1,len(self.experiment.zedges)-1):
         Fsub = self.compute_Fisher_matrix_for_specific_zbin(i,five_point=five_point)
         F += Fsub
         result.append(Fsub)
      return F,np.array(result)


   def non_default_fidval(self, param):
      '''
      For parameters that aren't default CLASS(-EDE) inputs.
      Returns their fiducial value. 
      '''
      if param == 'log(A_s)': return np.log(self.params['A_s'])
      if param == 'b': return self.experiment.b(self.zcenters[0]) # assumes b(z) = const.
      if param == 'N': return 1./self.experiment.n[0] # assumes N(z) = const.
      if param == 'fEDE': return self.fEDE
      if param == 'A_lin': return self.A_lin
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
    
    
   def Nmodes(self,zmin,zmax,nbins,kpar=-1.):
    
      def G(z):
         Sigma2 = self.Sigma2(z)
         f = self.cosmo.scale_independent_growth_factor_f(z)
         kparallel = self.k*self.mu
         kperpendicular = self.k*np.sqrt(1.-self.mu**2.)
         return np.exp(-0.5 * (kperpendicular**2. + kparallel**2. * (1.+f)**2.) * Sigma2)
      def I1(z):
         f = self.cosmo.scale_independent_growth_factor_f(z)      
         K,MU,b = self.k,self.mu,compute_b(self,z)
         P_L = compute_matter_power_spectrum(self,z,linear=True) * (b+f*MU**2.)**2.
         P_F = compute_tracer_power_spectrum(self,z)
         P_F += 1./compute_n(self,z)
         integrand = ( G(z)**2. * P_L / P_F )**2. 
         integrand *= self.compute_wedge(z) 
         if kpar > 0.: integrand *= (self.k*self.mu > kpar)
         return sum(integrand * self.k**2. * self.dk * self.dmu / (2. * np.pi**2.))
    
      zedges = np.linspace(zmin,zmax,nbins+1)
      zs = (zedges[1:]+zedges[:-1])/2.
      dV = np.array([self.comov_vol(zedges[i],zedges[i+1]) for i in range(nbins)])
      I = np.array([I1(z) for z in zs])
      return sum(I*dV) 
      
    
   
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
               Finv = np.linglag.inv(F)
               Ginv = np.array([[Finv[i,i],Finv[i,j]],[Finv[i,j],Finv[j,j]]])
               G = np.linalg.inv(Ginv)
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