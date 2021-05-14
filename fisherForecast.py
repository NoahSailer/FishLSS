from headers import *
from twoPoint import *
from twoPointNoise import *
from castorina import castorinaBias,castorinaPn
from multiprocessing import Pool
from functools import partial
import os


class fisherForecast(object):
   '''
   Capable of calculating the galaxy power spectrum and its derivatives
   with respect to cosmological parameters. Able to calculate the
   Fisher matrix for an arbitrary number of parameters.
   '''
   def __init__(self, 
                kmin=5e-4, 
                kmax=1., 
                cosmo=None, 
                cosmo_fid=None,
                experiment=None, 
                Nmu=10, 
                Nk=300, 
                marg_params=np.array(['A_s','h']),
                fEDE=0., 
                log10z_c=3.56207, 
                thetai_scf=2.83, 
                xi_idr=0.,
                A_lin=0., 
                omega_lin=0.01, 
                phi_lin=np.pi/2., 
                velocileptors=False,
                linear=False, 
                name='toy_model',
                smooth=False,
                AP=True,
                recon=False,
                ell=np.arange(10,1000,1)):
        
      self.kmin = kmin
      self.kmax = kmax
      self.Nmu = Nmu
      self.Nk = Nk
      self.marg_params = marg_params
      # parameters for EDE. These are redundant if fEDE has a nonzero fiducial value.
      self.fEDE = fEDE
      self.log10z_c = log10z_c
      self.thetai_scf = thetai_scf
      #
      self.xi_idr = xi_idr
      # parameters for primordial wiggles
      self.A_lin = A_lin 
      self.omega_lin = omega_lin
      self.phi_lin = phi_lin
      #
      self.velocileptors = velocileptors
      self.linear = linear
      self.name = name
      self.smooth = smooth
      self.AP = AP
      self.recon = recon
      self.ell = ell

      self.experiment = None
      self.cosmo = None
      self.k = None  # [h/Mpc]
      self.dk = None
      self.mu = None
      self.dmu = None
      self.P_fid = None          # fidicual power spectra at the center of each redshift bin
      self.P_recon_fid = None
      self.Vsurvey = None        # comoving volume [Mpc/h]^3 in each redshift bin
      self.params = None
    
      # make directories for storing derivatives and fiducial power spectra
      o,on = 'output/','output/'+self.name+'/'
      directories = np.array([o,on,on+'/derivatives/',on+'/derivatives_Cl/',on+'/derivatives_recon/'])
      for directory in directories: 
         if not os.path.exists(directory): os.mkdir(directory)

      if (cosmo is None) or (experiment is None):
         print('Attempted to create a forecast without an experiment or cosmology.')     
      else:
         self.set_experiment_and_cosmology_specific_parameters(experiment, cosmo, cosmo_fid)
           
        
   def set_experiment_and_cosmology_specific_parameters(self, experiment, cosmo, cosmo_fid):
      self.experiment = experiment
      self.cosmo = cosmo
      params = cosmo.pars
      if cosmo_fid is None:
         cosmo_fid = Class()
         cosmo_fid.set(params)
         cosmo_fid.compute()
      self.cosmo_fid = cosmo_fid # it's useful when taking derivatives 
                                 # to have access to the fiducial cosmology
      self.params = params
      self.params_fid = cosmo_fid.pars
      if 'log(A_s)' in self.marg_params and 'A_s' not in np.array(list(params.keys())):
         print('Must include A_s in params if trying to marginalize over log(A_s).')
         return
      if 'fEDE' in np.array(list(params.keys())):
         self.fEDE = params['fEDE']
         self.log10z_c = params['log10z_c']
         self.thetai_scf = params['thetai_scf']
      k = np.logspace(np.log10(self.kmin),np.log10(self.kmax),self.Nk)
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
    
      self.Vsurvey = np.array([self.comov_vol(experiment.zedges[i],experiment.zedges[i+1]) \
                               for i in range(len(experiment.zedges)-1)])
       
      redshifts = np.linspace(self.experiment.zmin,self.experiment.zmax,1000)
      Da_fid = np.array([self.cosmo.angular_distance(z) for z in redshifts])*self.params['h']
      Hz_fid = np.array([self.cosmo.Hubble(z)*(299792.458)/self.params['h'] for z in redshifts])
      rsd_fid = np.array([self.cosmo.get_current_derived_parameters(['rs_d'])['rs_d']*self.params['h'] for z in redshifts])
      self.Da_fid = interp1d(redshifts,Da_fid,kind='linear')
      self.Hz_fid = interp1d(redshifts,Hz_fid,kind='linear')
      self.rsd_fid = interp1d(redshifts,rsd_fid,kind='linear')
      
    
      # Calculate the fiducial power spectra in each redshift bin, and save them
      self.P_fid = np.zeros((self.experiment.nbins,self.Nk*self.Nmu))
      self.P_recon_fid = np.zeros((self.experiment.nbins,self.Nk*self.Nmu))
      self.Ckk_fid = np.zeros(len(self.ell))
      self.Ckg_fid = np.zeros((self.experiment.nbins,len(self.ell)))
      self.Cgg_fid = np.zeros((self.experiment.nbins,len(self.ell)))
      self.Ngg = np.zeros((self.experiment.nbins,len(self.ell)))
        
      try: 
        Ckk = np.genfromtxt('output/'+self.name+'/derivatives_Cl/Ckk_fid.txt')
        if len(Ckk) != len(self.ell): raise Exception('')
        self.Ckk_fid = Ckk
      except: 
        zs = self.experiment.zedges
        self.Ckk_fid = compute_lensing_Cell(self,'k','k')
        np.savetxt('output/'+self.name+'/derivatives_Cl/Ckk_fid.txt',self.Ckk_fid)
        
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]
         try: 
            p = np.genfromtxt('output/'+self.name+'/derivatives/pfid_'+str(int(100*z))+'.txt')
            if len(p) != self.Nk*self.Nmu: raise Exception('')
            self.P_fid[i] = p
            p = np.genfromtxt('output/'+self.name+'/derivatives_recon/pfid_'+str(int(100*z))+'.txt')
            if len(p) != self.Nk*self.Nmu: raise Exception('')
            self.P_recon_fid[i] = p
         except:
            self.P_fid[i] = compute_tracer_power_spectrum(self,z)
            np.savetxt('output/'+self.name+'/derivatives/pfid_'+str(int(100*z))+'.txt',self.P_fid[i])
            self.recon = True
            self.P_recon_fid[i] = compute_tracer_power_spectrum(self,z)
            np.savetxt('output/'+self.name+'/derivatives_recon/pfid_'+str(int(100*z))+'.txt',self.P_recon_fid[i])
            self.recon = False
            
         try:
            filename = 'output/'+self.name+'/derivatives_Cl/Ckg_fid_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
            Ckg = np.genfromtxt(filename)
            if len(Ckg) != len(self.ell): raise Exception('')
            filename = 'output/'+self.name+'/derivatives_Cl/Cgg_fid_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
            Cgg = np.genfromtxt(filename)
            filename = 'output/'+self.name+'/derivatives_Cl/Ngg_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
            Ngg = np.genfromtxt(filename)
            self.Ckg_fid[i] = Ckg
            self.Cgg_fid[i] = Cgg
            self.Ngg[i] = Ngg
            
         except:
            self.Ckg_fid[i] = compute_lensing_Cell(self,'k','g',zmin,zmax)
            self.Cgg_fid[i] = compute_lensing_Cell(self,'g','g',zmin,zmax)
            self.Ngg[i] = compute_lensing_Cell(self,'g','g',zmin,zmax,noise=True)
            filename = 'output/'+self.name+'/derivatives_Cl/Ckg_fid_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
            np.savetxt(filename,self.Ckg_fid[i])
            filename = 'output/'+self.name+'/derivatives_Cl/Cgg_fid_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
            np.savetxt(filename,self.Cgg_fid[i])
            filename = 'output/'+self.name+'/derivatives_Cl/Ngg_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
            np.savetxt(filename,self.Ngg[i])
            
      self.kpar_cut = np.ones((self.experiment.nbins,self.Nk*self.Nmu))
      for i in range(self.experiment.nbins): 
         z = self.experiment.zcenters[i]
         self.kpar_cut[i] = self.compute_kpar_cut(z,i)
  

   def compute_kpar_cut(self,z,zindex=None):
      # Create a "k_parallel cut", removing modes whose sn2 term
      # is >= 20% of the total power
      def get_n(z):
         if self.experiment.HI: return compute_n(self,z)[0]
         return compute_n(self,z)
      sigv = self.experiment.sigv
      sn2 = (self.k*self.mu)**2*((1+z)*sigv/self.Hz_fid(z))**2/get_n(z)
      if zindex is not None: Ps = self.P_fid[zindex]
      else: Ps = compute_tracer_power_spectrum(self,z)
      idx = np.where(sn2/Ps >= 0.2) ; idx2 = np.where(Ps <= 0)
      kpar_cut = np.ones(self.Nk*self.Nmu)
      kpar_cut[idx] = 0 ; kpar_cut[idx2] = 0 
      return kpar_cut
     
        
   def comov_vol(self,zmin,zmax):
      '''
      Returns the comoving volume in Mpc^3/h^3 between 
      zmin and zmax assuming that the universe is flat.
      Includes the fsky of the experiment.
      '''
      vsmall = (4*np.pi/3) * ((1.+zmin)*self.cosmo.angular_distance(zmin))**3.
      vbig = (4*np.pi/3) * ((1.+zmax)*self.cosmo.angular_distance(zmax))**3.
      return self.experiment.fsky*(vbig - vsmall)*self.params['h']**3.


   def LegendreTrans(self,l,p):
      '''
      Returns the l'th multipole of P(k,mu), where P(k,mu) is a 
      vector of length Nk*Nmu. Returns a vector of length Nk.
      '''
      n = self.Nk ; m = self.Nmu
      mu = self.mu.reshape((n,m))[0]
      p_reshaped = p.reshape((n,m))
      result = np.zeros(n)
      for i in range(n):
         integrand = (2*l+1)*p_reshaped[i,:]*scipy.special.legendre(l)(mu)
         result[i] = scipy.integrate.simps(integrand,x=mu)
      return result


   def compute_dPdp(self, param, z, relative_step=-1., absolute_step=-1., 
                    one_sided=False, five_point=False,kwargs=None):
      '''
      Calculates the derivative of the galaxy power spectrum
      with respect to the input parameter around the fidicual
      cosmology and redshift. Returns an array of length Nk*Nmu.
      '''
    
      if param == 'N' : return np.ones(self.k.shape)
      if param == 'N2': return self.k**2*self.mu**2
      if param == 'N4': return self.k**4*self.mu**4
    
      default_step = {'tau_reio':0.3,'m_ncdm':0.05,'A_lin':0.002}
        
      if relative_step == -1.: 
         try: relative_step = default_step[param]
         except: relative_step = 0.01
      if absolute_step == -1.: 
         try: absolute_step = default_step[param]
         except: absolute_step = 0.01

      b_fid = compute_b(self,z)    
      f_fid = self.cosmo.scale_independent_growth_factor_f(z)    
      alpha0_fid = 1.22 + 0.24*b_fid**2*(z-5.96) 
      Hz = self.Hz_fid(z)
      N_fid = 1/compute_n(self,z)
      noise = 1/compute_n(self,z)
      if self.experiment.HI: noise = noise[0]
      sigv = self.experiment.sigv
      N2_fid = -noise*((1+z)*sigv/Hz)**2.
        
      if param != 'alpha0': alpha0_fid = 0 # you'd get a k^2 dPz/dtheta term for alpha0 != 0
            
      if kwargs is None: 
         kwargs = {'fishcast':self, 'z':z, 'b':b_fid, 'b2':8*(b_fid-1)/21,
                   'bs':-2*(b_fid-1)/7,'alpha0':alpha0_fid, 'alpha2':0,
                   'alpha4':0., 'alpha6':0, 'N':N_fid, 'N2':N2_fid, 'N4':0.,
                   'f':-1, 'A_lin':self.A_lin, 'omega_lin':self.omega_lin,
                   'phi_lin':self.phi_lin,'kIR':0.2}
    
      if self.experiment.HI: kwargs['N'] = noise # ignores thermal HI noise in deriavtives
    
      if param in kwargs or param == 'f_NL': 
         fNL_flag = False
         if param == 'f_NL': 
            param = 'b' ; fNL_flag = True
         if param == 'f': default_value = f_fid
         else: default_value = kwargs[param]
         up = default_value*(1+relative_step)
         if default_value == 0: up = absolute_step
         upup = default_value*(1+2*relative_step)
         if default_value == 0: upup = 2*absolute_step
         down = default_value*(1-relative_step)
         if default_value == 0: down = -absolute_step
         downdown = default_value*(1-2*relative_step)
         if default_value == 0: downdown = -2*absolute_step
         step = up - default_value
         kwargs[param] = up
         P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
         kwargs[param] = upup
         P_dummy_higher = compute_tracer_power_spectrum(**kwargs)
         kwargs[param] = down
         P_dummy_low = compute_tracer_power_spectrum(**kwargs)
         kwargs[param] = downdown
         P_dummy_lower = compute_tracer_power_spectrum(**kwargs)
         kwargs[param] = default_value
         dPdtheta = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
         if fNL_flag:
            D = 0.76 * self.cosmo.scale_independent_growth_factor(z) # normalized so D(a) = a in the MD era
            # brute force way of getting the transfer function, normalized to 1 at kmin
            pmatter = compute_matter_power_spectrum(self, z, linear=True)
            T = np.sqrt(pmatter/self.k**self.params['n_s'])
            T /= T[0]
            fNL_factor = 3.*1.68*(b_fid-1.)*((self.params['omega_cdm']+self.params['omega_b'])/\
                                              self.params['h']**2.)*100.**2.
            fNL_factor /= D * self.k**2. * T * 299792.458**2.
            dPdtheta *= fNL_factor
         return dPdtheta
       
      P_fid = compute_tracer_power_spectrum(**kwargs) 
        
      if param == 'Tb': 
         Ez = self.cosmo.Hubble(z)/self.cosmo.Hubble(0)
         Ohi = 4e-4*(1+z)**0.6
         Tb = 188e-3*(self.params['h'])/Ez*Ohi*(1+z)**2
         return 2. * ( P_fid - noise  + castorinaPn(z)) / Tb  
    
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
          
      if param == 'alpha_parallel':
         K,MU = self.k,self.mu
         return -P_fid - MU*(1-MU**2)*dPdmu() - K*MU**2*dPdk()
        
      if param == 'alpha_perp':
         K,MU = self.k,self.mu
         return -2*P_fid + MU*(1-MU**2)*dPdmu() - K*(1-MU**2)*dPdk()       
            
      def one_sided(param,step):  
         self.cosmo.set({param:step})
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
         ap0_hi = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         ap1_hi = self.cosmo.Hubble(z)*(299792.458)/self.params['h'] / self.Hz_fid(z)
         #
         self.cosmo.set({param:2.*step})
         self.cosmo.compute()
         P_dummy_hi2 = compute_tracer_power_spectrum(**kwargs)
         ap0_hi2 = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         ap1_hi2 = self.cosmo.Hubble(z)*(299792.458)/self.params['h'] / self.Hz_fid(z)
         #
         self.cosmo.set({param:3.*step})
         self.cosmo.compute()
         P_dummy_hi3 = compute_tracer_power_spectrum(**kwargs)
         ap0_hi3 = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         ap1_hi3 = self.cosmo.Hubble(z)*(299792.458)/self.params['h'] / self.Hz_fid(z)
         #
         self.cosmo.set({param:4.*step})
         self.cosmo.compute()
         P_dummy_hi4 = compute_tracer_power_spectrum(**kwargs)
         ap0_hi4 = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         ap1_hi4 = self.cosmo.Hubble(z)*(299792.458)/self.params['h'] / self.Hz_fid(z)
         #
         self.cosmo = Class()
         self.cosmo.set(self.params_fid)
         self.cosmo.compute() 
         #
         dap0dp = (-3.*ap0_hi4 + 16.*ap0_hi3 - 36.*ap0_hi2 + 48.*ap0_hi - 25.*1)/(12.*step)
         dap1dp = (-3.*ap1_hi4 + 16.*ap1_hi3 - 36.*ap1_hi2 + 48.*ap1_hi - 25.*1)/(12.*step)
         result = (-3.*P_dummy_hi4 + 16.*P_dummy_hi3 - 36.*P_dummy_hi2 + 48.*P_dummy_hi - 25.*P_fid)/(12.*step)
         K,MU = self.k,self.mu
         if self.AP: result += (dap1dp-2*dap0dp)*P_fid + MU*(1-MU**2)*(dap1dp+dap0dp)*dPdmu() +\
                                K*(dap1dp*MU**2 - dap0dp*(1-MU**2))*dPdk()
         return result
         
      # derivative of early dark energy parameters (Hill+2020)
      if param == 'fEDE' and self.fEDE == 0.:
         EDE_params = {'log10z_c': self.log10z_c,'fEDE': absolute_step,'thetai_scf': self.thetai_scf,
                       'Omega_Lambda':0.0,'Omega_fld':0,'Omega_scf':-1,
                       'n_scf':3,'CC_scf':1,'scf_tuning_index':3,
                       'scf_parameters':'1, 1, 1, 1, 1, 0.0',
                       'attractor_ic_scf':'no'}
         self.cosmo.set(EDE_params)
         return one_sided('fEDE',absolute_step)
           
      if param == 'xi_idr' and self.xi_idr == 0.: 
         idr_params = {'xi_idr':0.,'a_idm_dr':1e4,'f_idm_dr':1.}
         self.cosmo.set(idr_params)
         return one_sided('xi_idr',absolute_step)
         
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
         #self.params[param] = value
        
      if five_point:
         set_param(up)
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
         aperp_hi = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         apar_hi = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.params['h'])
         #
         set_param(upup)
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(**kwargs)
         aperp_higher = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         apar_higher = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.params['h'])
         #
         set_param(down)
         self.cosmo.compute()
         P_dummy_low = compute_tracer_power_spectrum(**kwargs)
         aperp_low = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         apar_low = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.params['h'])
         #
         set_param(downdown)
         self.cosmo.compute()
         P_dummy_lower = compute_tracer_power_spectrum(**kwargs)
         aperp_lower = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
         apar_lower = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.params['h'])
         #
         set_param(default_value)
         self.cosmo.compute()
         #
         result += (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
         daperpdp = (-aperp_higher + 8.*aperp_hi - 8.*aperp_low + aperp_lower) / (12. * step)
         dapardp = (-apar_higher + 8.*apar_hi - 8.*apar_low + apar_lower) / (12. * step)
         K,MU = self.k,self.mu
         if self.AP: result += -(dapardp+2*daperpdp)*P_fid - MU*(1-MU**2)*(dapardp-daperpdp)*dPdmu() -\
                               K*(dapardp*MU**2 + daperpdp*(1-MU**2))*dPdk()
         if flag: result *= self.params['A_s']
         return result

      # defaults to a two sided derivative
      set_param(up)
      self.cosmo.compute()
      P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
      aperp_hi = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
      apar_hi = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.params['h']) 
      #
      set_param(down)
      self.cosmo.compute()
      P_dummy_low = compute_tracer_power_spectrum(**kwargs)
      aperp_low = self.cosmo.angular_distance(z)*self.params['h'] / self.Da_fid(z) 
      apar_low = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.params['h']) 
      #
      set_param(default_value)
      self.cosmo.compute()
      result += (P_dummy_hi - P_dummy_low) / (2. * step)
      daperpdp = (aperp_hi - aperp_low) / (2. * step)
      dapardp = (apar_hi - apar_low) / (2. * step)
      K,MU = self.k,self.mu
      if self.AP: result += -(dapardp+2*daperpdp)*P_fid - MU*(1-MU**2)*(dapardp-daperpdp)*dPdmu() -\
                               K*(dapardp*MU**2 + daperpdp*(1-MU**2))*dPdk()
      if flag: result *= self.params['A_s']
      return result


   def compute_dCdp(self, param, X, Y, zmin=None, zmax=None, 
                    relative_step=-1., absolute_step=-1.):
      '''
      '''
      default_step = {'tau_reio':0.3,'m_ncdm':0.05,'A_lin':0.002}
        
      if relative_step == -1.: 
         try: relative_step = default_step[param]
         except: relative_step = 0.01
      if absolute_step == -1.: 
         try: absolute_step = default_step[param]
         except: absolute_step = 0.01
            
      if zmin is not None and zmax is not None: zmid = (zmin+zmax)/2
      else: zmid = self.experiment.zcenters[0] # Ckk, where b and stuff don't matter
            
      b_fid = compute_b(self,zmid)      
      alpha0_fid = 1.22 + 0.24*b_fid**2*(zmid-5.96) 
      noise = 1/compute_n(self,zmid)
      if self.experiment.HI: noise = noise[0]
          
      kwargs = {'fishcast':self, 'X':X, 'Y':Y, 'zmin':zmin, 'zmax':zmax,
                'zmid':zmid,'gamma':1, 'b':b_fid, 'b2':8*(b_fid-1)/21,
                'bs':-2*(b_fid-1)/7,'alpha0':alpha0_fid,'alphax':0,'N':noise}
                    
      if param in kwargs: 
         default_value = kwargs[param]
         up = default_value*(1+relative_step)
         if default_value == 0: up = absolute_step
         upup = default_value*(1+2*relative_step)
         if default_value == 0: upup = 2*absolute_step
         down = default_value*(1-relative_step)
         if default_value == 0: down = -absolute_step
         downdown = default_value*(1-2*relative_step)
         if default_value == 0: downdown = -2*absolute_step
         step = up - default_value
         kwargs[param] = up
         P_dummy_hi = compute_lensing_Cell(**kwargs)
         kwargs[param] = upup
         P_dummy_higher = compute_lensing_Cell(**kwargs)
         kwargs[param] = down
         P_dummy_low = compute_lensing_Cell(**kwargs)
         kwargs[param] = downdown
         P_dummy_lower = compute_lensing_Cell(**kwargs)
         kwargs[param] = default_value
         return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. *  step) 
    
      P_fid = compute_lensing_Cell(**kwargs)
              
      # brute force numerical differentiation
      flag = False 
      if param == 'log(A_s)': 
         flag = True
         param = 'A_s'  
        
      default_value = self.params_fid[param] 
        
      if param == 'm_ncdm' and self.params['N_ncdm']>1:
         # CLASS takes a string as an input when there is more than one massless neutrino
         default_value_float = np.array(list(map(float,list(default_value.split(',')))))
         Mnu = sum(default_value_float)
         up = ','.join(list(map(str,list(default_value_float+relative_step*Mnu/self.params['N_ncdm']))))
         upup = ','.join(list(map(str,list(default_value_float+2.*relative_step*Mnu/self.params['N_ncdm']))))
         down = ','.join(list(map(str,list(default_value_float-relative_step*Mnu/self.params['N_ncdm']))))
         downdown = ','.join(list(map(str,list(default_value_float-2.*relative_step*Mnu/self.params['N_ncdm']))))
         step = (up-down)/2
      else:
         up = default_value * (1. + relative_step)
         if default_value == 0.: up = default_value + absolute_step
         upup = default_value * (1. + 2.*relative_step)
         if default_value == 0.: upup = default_value + 2.*absolute_step
         down = default_value * (1. - relative_step)
         if default_value == 0.: down = default_value - absolute_step
         downdown = default_value * (1. - 2.*relative_step)
         if default_value == 0.: downdown = default_value - 2.*absolute_step
         step = (up-down)/2
         if default_value == 0.: step = absolute_step
            
    
      def set_param(value):
         self.cosmo.set({param : value})
         #self.params[param] = value
      
      set_param(up)
      self.cosmo.compute()
      P_dummy_hi = compute_lensing_Cell(**kwargs)
      set_param(upup)
      self.cosmo.compute()
      P_dummy_higher = compute_lensing_Cell(**kwargs)
      set_param(down)
      self.cosmo.compute()
      P_dummy_low = compute_lensing_Cell(**kwargs)
      set_param(downdown)
      self.cosmo.compute()
      P_dummy_lower = compute_lensing_Cell(**kwargs)
      set_param(default_value)
      self.cosmo.compute()
      result = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
      if flag: result *= self.params['A_s']
      return result

   
   def Sigma2(self, z): return sum(compute_matter_power_spectrum(self,z,linear=True)*self.dk*self.dmu) / (6.*np.pi**2.)

    
   def kmax_constraint(self, z, kmax_knl=1.): return self.k < kmax_knl/np.sqrt(self.Sigma2(z))
    
    
   def compute_wedge(self, z, kmin=0.003):
      '''
      Returns the foreground wedge. If not an HI experiment, just returns a kmin constraint.
      The object that is returned is an array of bools of length Nk*Nmu.
      '''
      if not self.experiment.HI: return self.k > kmin
      #
      kparallel = self.k*self.mu
      kperpendicular = self.k*np.sqrt(1.-self.mu**2.)
      #
      chi = (1.+z)*self.cosmo.angular_distance(z)*self.params['h'] # Mpc/h
      Hz = self.cosmo.Hubble(z)*(299792.458)/self.params['h'] # h km / s Mpc
      c = 299792.458 # km/s
      lambda21 = 0.21 * (1+z) # meters
      D_eff = 6. * np.sqrt(0.7) # effective dish diameter, in meters
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
            folder = '/derivatives/'
            if self.recon: folder = '/derivatives_recon/'
            np.savetxt('output/'+self.name+folder+filename,dPdp)
         return
      zs = self.experiment.zcenters
      for z in zs:
         for marg_param in self.marg_params:
            dPdp = self.compute_dPdp(param=marg_param, z=z, five_point=five_point)
            if marg_param == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(int(100*z))+'.txt'
            elif marg_param == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(int(100*z))+'.txt'
            else: filename = marg_param+'_'+str(int(100*z))+'.txt'
            folder = '/derivatives/'
            if self.recon: folder = '/derivatives_recon/'
            np.savetxt('output/'+self.name+folder+filename,dPdp)
       
    
   def compute_Cl_derivatives(self):
      '''
      Calculates the derivatives of Ckk, Ckg, and Cgg with respect to 
      each of the marg_params. 
      '''
      zs = self.experiment.zedges
      for marg_param in self.marg_params:
         #
         if marg_param != 'gamma':
            dCdp = self.compute_dCdp(marg_param, 'k', 'k')
            filename = 'Ckk_'+marg_param+'.txt'
            np.savetxt('output/'+self.name+'/derivatives_Cl/'+filename,dCdp)
         else:
            for i,z in enumerate(zs[:-1]):
               dCdp = self.compute_dCdp(marg_param, 'k', 'k',zmin=zs[i],zmax=zs[i+1])
               filename = 'Ckk_'+marg_param+'_'+str(int(100*zs[i]))+'_'+str(int(100*zs[i+1]))+'.txt'
               np.savetxt('output/'+self.name+'/derivatives_Cl/'+filename,dCdp)
         #
         for i,z in enumerate(zs[:-1]):
            dCdp = self.compute_dCdp(marg_param, 'g', 'g', zmin=zs[i], zmax=zs[i+1])
            filename = 'Cgg_'+marg_param+'_'+str(int(100*zs[i]))+'_'+str(int(100*zs[i+1]))+'.txt'
            np.savetxt('output/'+self.name+'/derivatives_Cl/'+filename,dCdp)   
            #
            dCdp = self.compute_dCdp(marg_param, 'k', 'g', zmin=zs[i], zmax=zs[i+1])
            filename = 'Ckg_'+marg_param+'_'+str(int(100*zs[i]))+'_'+str(int(100*zs[i+1]))+'.txt'
            np.savetxt('output/'+self.name+'/derivatives_Cl/'+filename,dCdp)
            
            
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
            
            
   def load_derivatives(self, basis, log10z_c=-1.,omega_lin=-1,polys=True):
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
      N = len(basis)
      if self.recon and polys: N += 15
      derivatives = np.empty((nbins,N,self.Nk*self.Nmu))
      folder = '/derivatives/'
      if self.recon: folder = '/derivatives_recon/'
      directory = 'output/'+self.name+folder
        
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
         if self.recon and polys:
            for i in range(len(basis),N):
               m = i - len(basis)         
               derivatives[zbin_index,i] = self.mu**(2*(m//5)) * self.k**(m%5)
      return derivatives
            

   def shuffle_fisher(self,F,globe,Nz=None):
      N = len(F)
      if Nz is None: Nz = self.experiment.nbins
      loc = (N-globe)//Nz
      result = np.zeros(F.shape)
      mapping = {}
      for i in range(N):
         if i<globe: 
            mapping[i] = i
         else: 
            z = i-globe 
            y = globe + (z%Nz)*loc + z//Nz
            mapping[i] = y
      for i in range(N):
         for j in range(N):
            result[i,j] = F[mapping[i],mapping[j]]
      return result


   def combine_fishers(self,Fs,globe):
      N = len(Fs)
      result = Fs[0]
      for i in range(1,N): result = self.combine_2fishers(result,Fs[i],globe)
      return result


   def combine_2fishers(self,F1,F2,globe):
      '''
      helper function for combine_fishers
      '''
      N1 = int(len(F1)-globe)
      N2 = int(len(F2)-globe)
      N = N1+N2
      F = np.zeros((N+globe,N+globe))
      for i in range(N+globe):
         for j in range(N+globe):
            if i<globe+N1 and j<globe+N1: F[i,j] = F1[i,j]
            if i<globe and j<globe: F[i,j] = F1[i,j] + F2[i,j]
            if j>=globe+N1 and i<globe: F[i,j] = F2[i,j-N1]
            if i>=globe+N1 and j<globe: F[i,j] = F2[i-N1,j]
            if i>=globe+N1 and j>=globe+N1: F[i,j] = F2[i-N1,j-N1]
      return F


   def gen_fisher(self,basis,globe,log10z_c=-1.,omega_lin=-1.,kmax_knl=1.,
                  kmin=0.003,kmax=-10.,kpar_min=-1.,derivatives=None,zbins=None,
                  polys=True):
      '''
      Computes an array of Fisher matrices, one for each redshift bin.
      '''
      if log10z_c == -1. : log10z_c = self.log10z_c
      if omega_lin == -1. : omega_lin = self.omega_lin
        
      if derivatives is None: derivatives = self.load_derivatives(basis,log10z_c=log10z_c,
                                                                  omega_lin=omega_lin,polys=polys)   
      if zbins is None: zbins = range(self.experiment.nbins)
           
      def fish(zbin_index):
         n = len(basis)
         if self.recon: n += 15
         F = np.zeros((n,n))
         z = self.experiment.zcenters[zbin_index]
         dPdvecp = derivatives[zbin_index]
         Cinv = 1./self.get_covariance_matrix(zbin_index)
         mus = self.mu.reshape(self.Nk,self.Nmu)[0] 
         ks = self.k.reshape(self.Nk,self.Nmu)[:,0] 
         constraints = self.compute_wedge(z,kmin=kmin)*self.kmax_constraint(z,kmax_knl)
         if kmax > 0: constraints = self.compute_wedge(z,kmin=kmin)*(self.k<kmax)
         if kpar_min > 0: constraints *= (self.k*self.mu>kpar_min)
         for i in range(n):
            for j in range(n):
               integrand = (dPdvecp[i]*Cinv*dPdvecp[j]*constraints) 
               integrand *= self.kpar_cut[zbin_index]
               F[i,j] = np.sum(integrand)
         return F

      fishers = [fish(zbin_index) for zbin_index in zbins]
      result = self.combine_fishers(fishers,globe)
      result = self.shuffle_fisher(result,globe,Nz=len(zbins))
      return result


   def load_lensing_derivatives(self,param,param_index,globe,zbin_index):
      '''
      Loads the derivative of (Ckk, Ckgi, Cgigi), i = 1 , 2, ..., Nz
      with respect to param.
      '''
      n = self.experiment.nbins
      zs = self.experiment.zedges
      result = np.zeros((2*n+1,len(self.ell)))
    
      # if param is a local parameter (param_index >= globe)
      # then only take derivatives wrt C^{\kappa g_m} and 
      # C^{g_m g_m}, where m = zbin_index+1
      # alphax and alphaa are always assumed to be local
      if param_index >= globe:
         m = zbin_index+1
         filename = 'Ckg_'+param+'_'+str(int(100*zs[m-1]))+'_'+str(int(100*zs[m]))+'.txt'   
         result[m] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename)
         filename = 'Cgg_'+param+'_'+str(int(100*zs[m-1]))+'_'+str(int(100*zs[m]))+'.txt' 
         result[m+n] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename)
         return result
    
      if not 'gamma' in param: 
         result[0] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/Ckk_'+param+'.txt')
      else:
         idx = int(param[-1])
         filename = 'Ckk_gamma_'+str(int(100*zs[idx-1]))+'_'+str(int(100*zs[idx]))+'.txt'
         result[0] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename) 
      for i in range(1,n+1):
         if 'gamma' in param: 
            idx = int(param[-1])
            filename = 'Ckg_gamma_'+str(int(100*zs[idx-1]))+'_'+str(int(100*zs[idx]))+'.txt'
            result[idx] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename)
            filename = 'Cgg_gamma_'+str(int(100*zs[idx-1]))+'_'+str(int(100*zs[idx]))+'.txt'
            result[idx+n] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename)
         else:
            filename = 'Ckg_'+param+'_'+str(int(100*zs[i-1]))+'_'+str(int(100*zs[i]))+'.txt'
            result[i] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename)
            filename = 'Cgg_'+param+'_'+str(int(100*zs[i-1]))+'_'+str(int(100*zs[i]))+'.txt'
            result[i+n] = np.genfromtxt('output/'+self.name+'/derivatives_Cl/'+filename)
      return result

    
   def gen_lensing_fisher(self,basis,globe,ell_min=30,ell_max=1900,kmax_knl=1,
                          CMB='SO',kk=True,only_kk=False,bins=None):
      '''
      '''
      n = len(basis)
      C = covariance_Cls(self,kmax_knl=kmax_knl,CMB=CMB)
      Nz = self.experiment.nbins
      loc = n - globe
      N = globe + loc*Nz
      Np = 1 + 2*Nz
      if bins is None: bins = np.arange(0,Nz,1)
      idx_bins = []
      if kk: idx_bins = idx_bins + [0]
      for binn in bins: idx_bins = idx_bins + [binn+1,binn+Nz+1]
        
      result = np.zeros((N,N))
      derivs = np.zeros((N,Np,len(self.ell)))
      for i in range(globe): 
         derivs[i] = self.load_lensing_derivatives(basis[i],i,globe,0)
      for i in range(globe,N):
         param_index = globe + (i-globe)//Nz
         zbin_index = (i-globe)%Nz
         derivs[i] = self.load_lensing_derivatives(basis[param_index],param_index,globe,zbin_index)
            
      start = np.where(self.ell >= ell_min)[0][0]
      end = np.where(self.ell >= ell_max)[0][0]
      for i in range(start,end):
         for j in range(N):
            for k in range(N):
               if only_kk: 
                  result[j,k] += derivs[j][0,i]*derivs[k][0,i]/C[0,0,i]
               else:
                  Cinv = np.linalg.inv(C[idx_bins,:,i][:,idx_bins])
                  A = np.dot(Cinv,derivs[j][idx_bins,i])
                  result[j,k] += np.dot(derivs[k][idx_bins,i],A)
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
      f = interp1d(k,f_fixed,kind='linear',bounds_error=False, fill_value=0.)
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
    
    
   def Nmodes_fixed_k(self,k,zmin,zmax,nbins,Deltak=0.1):
    
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
         integrand = ( G(z)**2. * P_L / P_F )**2. 
         integrand *= self.k**2. * Deltak * self.dmu / (2. * np.pi**2.)
         ks = self.k.reshape((self.Nk,self.Nmu))[:,0]
         integrand = integrand.reshape((self.Nk,self.Nmu))
         integrand = np.sum(integrand, axis=1)
         answer = interp1d(ks,integrand)    
         return answer(k)
    
      zedges = np.linspace(zmin,zmax,nbins+1)
      zs = (zedges[1:]+zedges[:-1])/2.
      dV = np.array([self.comov_vol(zedges[i],zedges[i+1]) for i in range(nbins)])
      I = np.array([I1(z) for z in zs])
      return sum(I*dV)