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
                linear=False, name='toy_model',smooth=False,AP=True):
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
      self.AP = AP

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
         self.set_experiment_and_cosmology_specific_parameters(experiment, cosmo, params)
           
        
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
    
      # convienent for AP, functions for the fiducial Da and Hz's
      redshifts = np.linspace(self.experiment.zmin,self.experiment.zmax,1000)
      Da_fid = np.array([self.cosmo.angular_distance(z) for z in redshifts])*self.params['h']
      Hz_fid = np.array([self.cosmo.Hubble(z)*(299792.458)/self.params['h'] for z in redshifts])
      self.Da_fid = interp1d(redshifts,Da_fid,kind='linear')
      self.Hz_fid = interp1d(redshifts,Hz_fid,kind='linear')
      
      self.P_fid = np.zeros((self.experiment.nbins,self.Nk*self.Nmu))
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         try: 
            p = np.genfromtxt('output/'+self.name+'/derivatives/pfid_'+str(int(100*z))+'.txt')
            self.P_fid[i] = p
         except:
            self.P_fid[i] = compute_tracer_power_spectrum(self,z)
            np.savetxt('output/'+self.name+'/derivatives/pfid_'+str(int(100*z))+'.txt',self.P_fid[i])
        
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
         Ez = self.cosmo.Hubble(z)/self.cosmo.Hubble(0)
         Ohi = 4e-4*(1+z)**0.6
         Tb = 188e-3*(self.params['h'])/Ez*Ohi*(1+z)**2
         return 2. * ( P_fid + castorinaPn(z) ) / Tb
    
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
         noise = compute_n(self,z)
         if self.experiment.HI: noise = noise[0]
         Hz = self.cosmo.Hubble(z)*(299792.458)/self.params['h']
         return 2.*noise*(1+z)*300*self.k**2*self.mu**2/Hz
        
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

    
   def kmax_constraint(self, z, kmax_knl=1.): return self.k < kmax_knl/np.sqrt(self.Sigma2(z))
    
    
   def compute_wedge(self, z):
      '''
      Returns the foreground wedge. If not an HI experiment, just returns a kmin constraint.
      The object that is returned is an array of bools of length Nk*Nmu.
      '''
      if not self.experiment.HI: return self.k > 0.003 #
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
      '''
      Calculates all the derivatives and saves them to the 
      output/forecast name/derivatives directory
      '''
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
      '''
      Plots all the derivatives in the output/forecast name/derivatives directory
      '''
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
            
            
   def load_derivatives(self, basis, log10z_c=-1.,omega_lin=-1):
      '''
      Let basis = [p1, p2, ...], and denote the centers of the
      redshift bins by z1, z2, ... This returns a matrix of the
      form:
      
      derivatives = [[dPdp1, dPdp2, ...],  (z=z1)
                     [dPdp1, dPdp2, ...],  (z=z2)
                     ...]
                     
      Where dPdpi is the derivative with respect to the ith basis
      parameter (an array of length Nk*Nmu). 
      '''
      if log10z_c == -1. : log10z_c = self.log10z_c  
      if omega_lin == -1. : omega_lin = self.omega_lin
        
      nbins = self.experiment.nbins
      derivatives = np.empty((nbins,len(basis),self.Nk*self.Nmu))
      directory = 'output/'+self.name+'/derivatives/'
        
      for zbin_index in range(nbins):
         z = self.experiment.zcenters[zbin_index]
         for i,param in enumerate(basis):
            if param == 'fEDE': filename = 'fEDE_'+str(int(1000.*log10z_c))+'_'+str(int(100*z))+'.txt'
            elif param == 'A_lin': filename = 'A_lin_'+str(int(100.*omega_lin))+'_'+str(int(100*z))+'.txt'
            else: filename = param+'_'+str(int(100*z))+'.txt'
            try:
               dPdp = np.genfromtxt(directory+filename)
            except:
               print('Have not calculated derivative of ' + param)
            derivatives[zbin_index,i] = dPdp
      return derivatives
            
                
   def gen_fisher(self,basis,log10z_c=-1.,omega_lin=-1.,kmax_knl=1.,kmax=-10.,derivatives=None,zbins=None):
      '''
      Computes an array of Fisher matrices, one for each redshift bin.
      '''
      if log10z_c == -1. : log10z_c = self.log10z_c
      if omega_lin == -1. : omega_lin = self.omega_lin
        
      if derivatives is None: derivatives = self.load_derivatives(basis,log10z_c=log10z_c,omega_lin=omega_lin)   
      if zbins is None: zbins = range(self.experiment.nbins)
         
      def fish(zbin_index):
         n = len(basis)
         F = np.zeros((n,n))
         z = self.experiment.zcenters[zbin_index]
         dPdvecp = derivatives[zbin_index]
         Cinv = 1./self.get_covariance_matrix(zbin_index)
         constraints = self.compute_wedge(z)*self.kmax_constraint(z,kmax_knl)
         if kmax > 0: constraints = self.compute_wedge(z)*(self.k<kmax)
         for i in range(n):
            for j in range(n):
               F[i,j] = np.sum(dPdvecp[i]*Cinv*dPdvecp[j]*constraints)
         return F

      result = sum([fish(i) for i in zbins])
      return result


   def get_covariance_matrix(self, zbin_index): return compute_covariance_matrix(self,zbin_index)

        
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