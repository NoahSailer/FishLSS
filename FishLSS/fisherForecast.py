import numpy as np
from scipy.special import legendre
from scipy.integrate import simps
from scipy.interpolate import interp1d
import os, json
from os.path import exists
from classy import Class

from FishLSS.castorina import castorinaBias,castorinaPn
from FishLSS.twoPoint import *
from FishLSS.twoPointNoise import *


class fisherForecast(object):
   '''
   Class for computing derivatives of the galaxy (full-shape)
   power spectrum, as well as the CMB lensing cross-/auto spectrum.
   Can build and combine Fisher matrices formed from both of these
   observables.
   '''
   def __init__(self, 
                cosmo=None, 
                cosmo_fid=None,
                experiment=None, 
                kmin=5e-4, 
                kmax=1., 
                Nmu=50, 
                Nk=500, 
                free_params=np.array(['h']),
                fEDE=0., 
                log10z_c=3.56207, 
                thetai_scf=2.83, 
                A_lin=0., 
                omega_lin=0.01, 
                phi_lin=np.pi/2., 
                A_log=0., 
                omega_log=0.01, 
                phi_log=np.pi/2., 
                velocileptors=True,
                linear=False, 
                linear2=False,
                name='toy_model',
                smooth=False,
                AP=True,
                ell=np.arange(10,1000,1),
                N2cut=0.2,
                setup=True,
                overwrite=False,
                basedir='.'):
        
      # some basic stuff for the k-mu grid  
      self.kmin = kmin
      self.kmax = kmax
      self.Nmu = Nmu
      self.Nk = Nk
      # free parameters in the forecast
      self.free_params = free_params
      # parameters for EDE. These are redundant if fEDE has a nonzero fiducial value.
      self.fEDE = fEDE
      self.log10z_c = log10z_c
      self.thetai_scf = thetai_scf
      #
      # parameters for primordial features
      self.A_lin = A_lin 
      self.omega_lin = omega_lin
      self.phi_lin = phi_lin
      self.A_log = A_log 
      self.omega_log = omega_log
      self.phi_log = phi_log
      #
      self.velocileptors = velocileptors
      self.linear = linear
      self.linear2 = linear2
      self.name = name
      self.smooth = smooth
      self.AP = AP
      self.recon = False
      self.ell = ell
      self.N2cut = N2cut
      self.basedir = basedir
      if basedir != '' and not basedir.endswith('/'):
         self.basedir = basedir + '/'
      
      # Set up the k-mu grid 
      k = np.logspace(np.log10(self.kmin),np.log10(self.kmax),self.Nk) # [h/Mpc]
      dk = list(k[1:]-k[:-1])
      dk.insert(0,dk[0])
      dk = np.array(dk)
      self.k = np.repeat(k,self.Nmu)
      self.dk = np.repeat(dk,self.Nmu)
      mu = np.linspace(0.,1.,self.Nmu)
      dmu = list(mu[1:]-mu[:-1])
      dmu.append(dmu[-1])
      dmu = np.array(dmu)
      self.mu = np.tile(mu,self.Nk)
      self.dmu = np.tile(dmu,self.Nk)
      
      self.custom_steps = {'tau_reio':0.3,'m_ncdm':0.05,'A_lin':0.002,'A_log':0.002}

      # we set these up later
      self.experiment = None     # 
      self.cosmo = None          # CLASS object
      self.P_fid = None          # fidicual power spectra at the center of each redshift bin
      self.P_recon_fid = None    # fiducial reconstructed power spectra "..."
      self.Vsurvey = None        # comoving volume [Mpc/h]^3 in each redshift bin
      self.params = None         # CLASS parameters
    
      # make directories for storing derivatives and fiducial power spectra
      o,on = self.basedir+'output/',self.basedir+'output/'+self.name+'/'
      directories = np.array([o, on, on+'/derivatives/', on+'/derivatives_Cl/', on+'/derivatives_recon/'])
      for directory in directories: 
         if not os.path.exists(directory): os.mkdir(directory)

      if (cosmo is None) or (experiment is None):
         print('Attempted to create a forecast without an experiment or cosmology.')     
      else:
         self.set_experiment_and_cosmology_specific_parameters(experiment, cosmo, cosmo_fid)
      
      self.create_json_summary()
      
      if setup or overwrite: 
         self.compute_fiducial_Pk(overwrite=overwrite)
         self.compute_fiducial_Cl(overwrite=overwrite)
         self.compute_fiducial_Precon(overwrite=overwrite)



                                     ##
                                   #### 
                                  #####  
                                    ###
                                    ###
                                    ###
                                    ###
                                    ###
                                  #######  
   
   
   #######################################################################
   #######################################################################
   ## This section of the code is used to set up the forecast object 
   ## Computes (or loads) fiducial power spectra, sets up distances, ...
   
        
   def set_experiment_and_cosmology_specific_parameters(self, experiment, cosmo, cosmo_fid):
      '''
      Set up the experiment- and cosmology-dependent pieces of the forecast
      '''
      self.experiment = experiment
      self.cosmo = cosmo
      params = cosmo.pars
      
      # it's useful when taking derivatives 
      # to have access to the fiducial cosmology
      if cosmo_fid is None:
         cosmo_fid = Class()
         cosmo_fid.set(params)
         cosmo_fid.compute()
      self.cosmo_fid = cosmo_fid 
      
      self.params = params
      self.params_fid = cosmo_fid.pars
      if 'log(A_s)' in self.free_params and 'A_s' not in np.array(list(params.keys())):
         print('Must include A_s in CLASS params if trying to take derivatives w.r.t. log(A_s).')
         return
      if 'fEDE' in np.array(list(params.keys())):
         self.fEDE = params['fEDE']
         self.log10z_c = params['log10z_c']
         self.thetai_scf = params['thetai_scf']

      # Fiducial volume in each redshift bin
      edges = experiment.zedges
      num_bins = len(experiment.zedges) - 1
      self.Vsurvey = np.array([self.comov_vol(edges[i],edges[i+1]) for i in range(num_bins)])
          
      # Fiducial distances as a function of z
      # Da = angular diameter distance [Mpc/h]
      # Hz = hubble [h km/s/Mpc]
      zs = np.linspace(0,60,1000)
      h = self.params['h']
      c = 299792.458 # speed of light in km/s
      Da_fid = np.array([self.cosmo_fid.angular_distance(z)*h for z in zs])
      Hz_fid = np.array([self.cosmo_fid.Hubble(z)*c/h for z in zs])
      # interpolate
      self.Da_fid = interp1d(zs,Da_fid,kind='linear')
      self.Hz_fid = interp1d(zs,Hz_fid,kind='linear')

        
   def compute_fiducial_Pk(self, overwrite=False):
      '''
      Either compute or load fiducial full-shape power spectra
      '''
      self.P_fid = np.zeros((self.experiment.nbins,self.Nk*self.Nmu))
        
      # Compute fiducial power spectra in each redshift bin
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]
         # P(k)
         fname = self.basedir+'output/'+self.name+'/derivatives/pfid_'+str(int(100*z))+'.txt'
         if not exists(fname) or overwrite:  
            self.P_fid[i] = compute_tracer_power_spectrum(self,z)
            np.savetxt(fname,self.P_fid[i])
         else:
            self.P_fid[i] = np.genfromtxt(fname)
            
      # setup the k_par cut  
      self.kpar_cut = np.ones((self.experiment.nbins,self.Nk*self.Nmu))
      for i in range(self.experiment.nbins): 
         z = self.experiment.zcenters[i]
         self.kpar_cut[i] = self.compute_kpar_cut(z,i)
        
        
   def compute_fiducial_Cl(self, overwrite=False):
      '''
      Either compute or load fiducial C_ells (Ckk, Ckg, Cgg)
      '''
      self.Ckk_fid = np.zeros(len(self.ell))
      self.Ckg_fid = np.zeros((self.experiment.nbins,len(self.ell)))
      self.Cgg_fid = np.zeros((self.experiment.nbins,len(self.ell)))

      # Ckk    
      fname = self.basedir+'output/'+self.name+'/derivatives_Cl/Ckk_fid.txt'
      if not exists(fname) or overwrite:  
         self.Ckk_fid = compute_lensing_Cell(self,'k','k')
         np.savetxt(fname,self.Ckk_fid)
      else:
         self.Ckk_fid = np.genfromtxt(fname)
         
      # Compute fiducial power spectra in each redshift bin
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]
            
         # Ckg
         fname = self.basedir+'output/'+self.name+'/derivatives_Cl/Ckg_fid_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
         if not exists(fname) or overwrite:
            self.Ckg_fid[i] = compute_lensing_Cell(self,'k','g',zmin,zmax)
            np.savetxt(fname,self.Ckg_fid[i])
         else: 
            self.Ckg_fid[i] = np.genfromtxt(fname)
         
         # Cgg
         fname = self.basedir+'output/'+self.name+'/derivatives_Cl/Cgg_fid_'+str(int(100*zmin))+'_'+str(int(100*zmax))+'.txt'
         if not exists(fname) or overwrite:
            self.Cgg_fid[i] = compute_lensing_Cell(self,'g','g',zmin,zmax)
            np.savetxt(fname,self.Cgg_fid[i])
         else:
            self.Cgg_fid[i] = np.genfromtxt(fname)


   def compute_fiducial_Precon(self, overwrite=False):
      '''
      Either compute or load reconstructed power spectra
      '''
      self.P_recon_fid = np.zeros((self.experiment.nbins,self.Nk*self.Nmu))
                
      # Compute fiducial power spectra in each redshift bin
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]

         # P_recon(k)
         fname = self.basedir+'output/'+self.name+'/derivatives_recon/pfid_'+str(int(100*z))+'.txt'
         if not exists(fname) or overwrite:  
            self.recon = True
            self.P_recon_fid[i] = compute_tracer_power_spectrum(self,z)
            self.recon = False
            np.savetxt(fname,self.P_recon_fid[i])
         else:
            self.P_recon_fid[i] = np.genfromtxt(fname)
          
         
   def create_json_summary(self):
      '''
      Creates and stores a .json file in output/forecast_name/
      This file summarizes some of the basic forecast details
      '''
      ze = list(self.experiment.zedges)
      zs = self.experiment.zcenters
      bs = list([float(compute_b(self,z)) for z in zs])
      if self.experiment.HI:
          ns = list([float(castorinaPn(z)) for z in zs])
      else:
          ns = list([float(compute_n(self,z)) for z in zs])
      
      data = {'Forecast name': self.name,
              'Edges of redshift bins': ze,
              'Centers of redshift bins': list(zs),
              'Linear Eulerian bias in each bin': bs,
              'Number density in each bin': ns,
              'fsky': self.experiment.fsky,
              'CLASS default parameters': self.params_fid}

      with open(self.basedir+'output/'+self.name+'/'+'summary.json', 'w') as write_file:
         json.dump(data, write_file, indent=2)
  

   def compute_kpar_cut(self,z,zindex=None):
      # Create a "k_parallel cut", removing modes whose N2 term
      # is >= 100*N2cut % of the total power
      def get_n(z):
         if self.experiment.HI: return castorinaPn(z)
         return compute_n(self,z)
      sigv = self.experiment.sigv
      sn2 = (self.k*self.mu)**2*((1+z)*sigv/self.Hz_fid(z))**2/get_n(z)
      if zindex is not None: Ps = self.P_fid[zindex]
      else: Ps = compute_tracer_power_spectrum(self,z)
      idx = np.where(sn2/Ps >= self.N2cut) ; idx2 = np.where(Ps <= 0)
      kpar_cut = np.ones(self.Nk*self.Nmu)
      kpar_cut[idx] = 0 ; kpar_cut[idx2] = 0 
      return kpar_cut

         
   def comov_vol(self,zmin,zmax):
      '''
      Returns the comoving volume in (Mpc/h)^3 between 
      zmin and zmax, assuming that the universe is flat.
      Includes the fsky of the experiment.
      '''
      chi = lambda z: (1+z)*self.cosmo.angular_distance(z)
      h = self.params_fid['h']
      vsmall = (4*np.pi/3) * chi(zmin)**3.
      vbig = (4*np.pi/3) * chi(zmax)**3.
      return self.experiment.fsky*(vbig - vsmall)*h**3.

                                      
                                  ### 
                                #######
                               #########
                               ###   ### 
                               ###   ###
                                     ###
                                    ###
                                  ###
                                 ###
                               #########
                               #########
   
   
   #######################################################################
   #######################################################################
   ## This section contains some helper-functions
   ##

   def LegendreTrans(self,l,p,mu_max=1.):
      '''
      Returns the l'th multipole of P(k,mu), where P(k,mu) is a 
      vector of length Nk*Nmu. Returns a vector of length Nk.
      '''
      n = self.Nk ; m = self.Nmu
      mu = self.mu.reshape((n,m))[0]
      p_reshaped = p.reshape((n,m))
      result = np.zeros(n)
      for i in range(n):
         integrand = (2*l+1)*p_reshaped[i,:]*legendre(l)(mu)
         result[i] = simps(integrand,x=mu)
      return result
      
      
   def LegendreTransInv(self,pls):
      '''
      Given pls = [p0,p2,p4,...], where each pn is an 
      array of length Nk, returns P(k,mu), which is 
      an array of length Nk*Nmu.
      '''
      n = len(pls)
      pls_repeat = np.zeros((n,self.Nk*self.Nmu))
      legendre_polys = pls_repeat.copy()
      for i in range(n): 
         pls_repeat = np.repeat(pls[i],self.Nmu)
         legendre_polys = legendre(2*i)(self.mu)
      return np.sum(pls_repeat*legendre_polys,axis=0)
      
      
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
      D_eff = self.experiment.D * np.sqrt(0.7) # effective dish diameter, in meters
      theta_w = self.experiment.N_w * 1.22 * lambda21 / (2.*D_eff)
      #
      wedge = kparallel > chi*Hz*np.sin(theta_w)*kperpendicular/(c*(1.+z))
      kparallel_constraint = kparallel > self.experiment.kparallel_min
      return wedge*kparallel_constraint
      
      
   def dPdk(self,P):
      '''
      Given an input array P (length Nk*Nmu), which is 
      defined on the (flattened) k-mu grid, returns 
      dPdk on that same grid. 
      
      assumes that k is log spaced
      '''
      P_reshaped = P.reshape((self.Nk,self.Nmu)) 
      P_low  = np.roll(P_reshaped,1,axis=0)
      P_low2 = np.roll(P_reshaped,2,axis=0)
      P_low3 = np.roll(P_reshaped,3,axis=0) 
      P_low4 = np.roll(P_reshaped,4,axis=0)
      P_hi   = np.roll(P_reshaped,-1,axis=0)
      P_hi2  = np.roll(P_reshaped,-2,axis=0)
      P_hi3  = np.roll(P_reshaped,-3,axis=0)
      P_hi4  = np.roll(P_reshaped,-4,axis=0)
      dP     = (-P_hi2 + 8.*P_hi - 8.*P_low + P_low2) / 12.
      dP_low = -(-3.*P_low4 + 16.*P_low3 - 36.*P_low2 + 48.*P_low - 25.*P_reshaped)/12.
      dP_hi  = (-3.*P_hi4 + 16.*P_hi3 - 36.*P_hi2 + 48.*P_hi - 25.*P_reshaped)/12.
      # correct for "edge effects"
      dP[:5,:] = dP_hi[:5,:]
      dP[-5:,:] = dP_low[-5:,:]  
      ks = self.k.reshape((self.Nk,self.Nmu))[:,0]
      dlnk = np.log(ks)[1]-np.log(ks)[0] 
      dPdk = dP.flatten()/dlnk/self.k
      return dPdk
   
   
   def dPdmu(self,P):
      '''
      Given an input array P (length Nk*Nmu), which is 
      defined on the (flattened) k-mu grid, returns 
      dPdmu on that same grid. 
      
      assumes that mu is lin spaced
      '''
      P_reshaped = P.reshape((self.Nk,self.Nmu)) 
      P_low  = np.roll(P_reshaped,1,axis=1)
      P_low2 = np.roll(P_reshaped,2,axis=1)
      P_low3 = np.roll(P_reshaped,3,axis=1) 
      P_low4 = np.roll(P_reshaped,4,axis=1)
      P_hi   = np.roll(P_reshaped,-1,axis=1)
      P_hi2  = np.roll(P_reshaped,-2,axis=1)
      P_hi3  = np.roll(P_reshaped,-3,axis=1)
      P_hi4  = np.roll(P_reshaped,-4,axis=1)
      dP     = (-P_hi2 + 8.*P_hi - 8.*P_low + P_low2) / 12.
      dP_low = -(-3.*P_low4 + 16.*P_low3 - 36.*P_low2 + 48.*P_low - 25.*P_reshaped)/12.
      dP_hi  = (-3.*P_hi4 + 16.*P_hi3 - 36.*P_hi2 + 48.*P_hi - 25.*P_reshaped)/12.
      # correct for "edge effects"
      dP[:,:5] = dP_hi[:,:5]
      dP[:,-5:] = dP_low[:,-5:] 
      dmu = self.mu[1] - self.mu[0]
      return dP.flatten()/dmu
      
      
   def get_steps(self, param, default_value, relative_step, absolute_step):
      '''
      Helper function for compute_dPdp and compute_dCdp.
      Returns the values for which the derivative 'stencil'
      should be evaluated.
      '''
      # CLASS takes a string as an input when there is more than one massive neutrino.
      # If the absolute step size is 0.02 eV, and '0.01,0.05' is the default neutrino mass,
      # then the code below sets up = '0.02,0.06', and so on.
      if param == 'm_ncdm' and self.params['N_ncdm']>1:
         default_value_float = np.array(list(map(float,list(default_value.split(',')))))
         Mnu = sum(default_value_float)
         single_step = relative_step*Mnu/self.params['N_ncdm']
         new_value = lambda stp: ','.join(list(map(str,list(default_value_float+stp))))
         up =       new_value(single_step)
         upup =     new_value(2.*single_step)
         down =     new_value(-single_step)
         downdown = new_value(-2.*single_step)
         step = Mnu*relative_step
      # If varying anything other than the neutrino mass, then the parameter
      # value is assumed to be a single float. Absolute step sizes are used 
      # when the default parameter value is zero.
      elif default_value == 0.:
         up =       default_value + absolute_step
         upup =     default_value + 2.*absolute_step
         down =     default_value - absolute_step
         downdown = default_value - 2.*absolute_step
         step = absolute_step 
      else:
         up =       default_value * (1. + relative_step)
         upup =     default_value * (1. + 2.*relative_step)
         down =     default_value * (1. - relative_step)
         downdown = default_value * (1. - 2.*relative_step)
         step = default_value * relative_step
      return up,upup,down,downdown,step
      
    
   def compute_dPdp(self, param, z, one_sided=False, five_point=False):
      '''
      Calculates the derivative of the tracer power spectrum at 
      redshift z with respect to the input parameter around the 
      fidicual cosmology. Returns an array of length Nk*Nmu.
      
      Defaults to a two sided approximation:
                   df/dx = (f(x+e) - f(x-e)) / 2 e      (1)
      where e is the step size.
      
      By default e = 0.01 * (the fiducial value of param), or 
      e = 0.01 when (the fiducial value of param) = 0. Custom 
      step sizes can be set with self.custom_steps.
      ----------------------------------------------------------
      - param (string): input parameter 
      
      - z (float): redshift 
      
      - one_sided (boolean, default False): replace Eq. (1) with
      
      df/dx = (-3 f(x+4e) + 16 f(x+3e) - 36 f(x+2e) + 48 f(x+e) - 25 f(x)) / 12 e
           
      - five_point (boolean, default False): replace Eq. (1) with
      
      df/dx = (-f(x+2e) + 8 f(x+e) - 8 f(x-e) + f(x-2e)) / 12 e
      '''
      ### These derivatives are computed analytically.
      if param == 'N' : return np.ones(self.k.shape)
      if param == 'N2' and not self.linear2: return self.k**2*self.mu**2
      if param == 'N4': return self.k**4*self.mu**4
      # Fiucial P(k,mu,z) parameters  
      
      ##########################################################################
      
      b_fid = compute_b(self,z)    
      if z < 6 and self.experiment.alpha0 is None:
         alpha0_fid = 1.22 + 0.24*b_fid**2*(z-5.96)
      elif self.experiment.alpha0 is None:
         alpha0_fid = 0.
      if self.experiment.alpha0 is not None:
         alpha0_fid = self.experiment.alpha0(z)
      Hz = self.Hz_fid(z)
      N_fid = 1/compute_n(self,z)
      noise = 1/compute_n(self,z)
      if self.experiment.HI: noise = castorinaPn(z)
      sigv = self.experiment.sigv
      N2_fid = -noise*((1+z)*sigv/Hz)**2.
      
      if param == 'f':
         f = self.cosmo_fid.scale_independent_growth_factor_f(z)
      else:
         f=-1
       
      kwargs = {'fishcast':self, 'z':z, 'b':b_fid, 'b2':8*(b_fid-1)/21,
                'bs':-2*(b_fid-1)/7,'alpha0':alpha0_fid, 'alpha2':0,
                'alpha4':0., 'alpha6':0, 'N':N_fid, 'N2':N2_fid, 'N4':0.,
                'f':f, 'A_lin':self.A_lin, 'omega_lin':self.omega_lin,
                'phi_lin':self.phi_lin,'A_log':self.A_log, 
                'omega_log':self.omega_log,'phi_log':self.phi_log,'kIR':0.2}
      # ignore thermal 21cm noise in deriavtives      
      if self.experiment.HI: kwargs['N'] = noise 
      P_fid = compute_tracer_power_spectrum(**kwargs) 
        
      if param == 'norm': return 2*P_fid  
        
      if param == 'Tb': 
         Ez = self.cosmo.Hubble(z)/self.cosmo.Hubble(0)
         Ohi = 4e-4*(1+z)**0.6
         Tb = 188e-3*(self.params['h'])/Ez*Ohi*(1+z)**2
         return 2. * ( P_fid - noise  + castorinaPn(z)) / Tb  
      
      ##########################################################################
      
      ### The remaining derivatives are calculated numerically.
      # Get the step size
      try: 
         relative_step = self.custom_steps[param]
         absolute_step = self.custom_steps[param]
      except: 
         relative_step = 0.01
         absolute_step = 0.01
      
      def set_param(value,compute=True):
         if param in kwargs: 
            kwargs[param] = value
            if not compute: return None
            return compute_tracer_power_spectrum(**kwargs), 1., 1.
         self.cosmo.set({param : value})
         self.cosmo.compute()
         h = self.params['h']
         c = 299792.458
         if compute:
            P = compute_tracer_power_spectrum(**kwargs)
            aperp = self.cosmo.angular_distance(z)*h/self.Da_fid(z)
            apar  = self.Hz_fid(z)/(self.cosmo.Hubble(z)*c/h)
            return P, aperp, apar

      def AP_effect(daperpdp,dapardp):
         K,MU = self.k,self.mu
         res = -(dapardp+2*daperpdp)*P_fid
         res -= MU*(1-MU**2)*(dapardp-daperpdp)*self.dPdmu(P_fid)
         res -= K*(dapardp*MU**2 + daperpdp*(1-MU**2))*self.dPdk(P_fid)
         return res
       
      if param == 'alpha_parallel': return AP_effect(0.,1.)
      if param == 'alpha_perp': return AP_effect(1.,0.)    
      
      def one_sided_deriv(param,step,fid=0.):  
         P_dummy_hi, aperp_hi, apar_hi = set_param(fid+step)
         P_dummy_hi2, aperp_hi2, apar_hi2 = set_param(fid+2.*step)
         P_dummy_hi3, aperp_hi3, apar_hi3 = set_param(fid+3.*step)
         P_dummy_hi4, aperp_hi4, apar_hi4 = set_param(fid+4.*step)
         self.cosmo = Class()
         self.cosmo.set(self.params_fid)
         self.cosmo.compute() 
         stencil = lambda h4,h3,h2,h1,h0: (-3.*h4+16.*h3-36.*h2+48.*h1-25.*h0)/(12.*step)
         result  = stencil(P_dummy_hi4,P_dummy_hi3,P_dummy_hi2,P_dummy_hi,P_fid)
         daperpdp = stencil(aperp_hi4,aperp_hi3,aperp_hi2,aperp_hi,1.)
         dapardp  = stencil(apar_hi4,apar_hi3,apar_hi2,apar_hi,1.)
         if self.AP: result += AP_effect(daperpdp,dapardp)  
         return result
         
      # derivative w.r.t. fEDE (early dark energy) 
      if param == 'fEDE' and self.fEDE == 0.:
         EDE_params = {'log10z_c': self.log10z_c,'fEDE': absolute_step,'thetai_scf': self.thetai_scf,
                       'Omega_Lambda':0.0,'Omega_fld':0,'Omega_scf':-1,'n_scf':3,'CC_scf':1,
                       'scf_tuning_index':3,'scf_parameters':'1, 1, 1, 1, 1, 0.0','attractor_ic_scf':'no'}
         self.cosmo.set(EDE_params)
         return one_sided_deriv('fEDE',absolute_step)
         
      # brute force numerical differentiation
      result = np.zeros(len(self.k))
      As_flag = False ; fNL_flag = False
      if param == 'log(A_s)': As_flag = True ; param = 'A_s'  
      if param == 'f_NL': fNL_flag = True ; param = 'b'
      if param in kwargs: default_value = kwargs[param]
      else: default_value = self.params_fid[param] 
      up,upup,down,downdown,step = self.get_steps(param, default_value, relative_step, absolute_step)
      
      if one_sided: return one_sided_deriv(param,step,fid=default_value)

      P_dummy_hi, aperp_hi, apar_hi = set_param(up)
      P_dummy_low, aperp_low, apar_low = set_param(down)
      if five_point:
         P_dummy_higher, aperp_higher, apar_higher = set_param(upup)
         P_dummy_lower, aperp_lower, apar_lower = set_param(downdown)
      set_param(default_value,compute=False)
      
      if five_point:
         stencil = lambda hh,h,l,ll: (-hh + 8.*h - 8.*l + ll) / (12. * step)
         result  += stencil(P_dummy_higher,P_dummy_hi,P_dummy_low,P_dummy_lower)
         daperpdp = stencil(aperp_higher,aperp_hi,aperp_low,aperp_lower)
         dapardp  = stencil(apar_higher,apar_hi,apar_low,apar_lower)
         if self.AP: result += AP_effect(daperpdp,dapardp) 
      else:
         # defaults to two-sided derivative
         stencil = lambda h,l: (h-l) / (2. * step)
         result  += stencil(P_dummy_hi,P_dummy_low)
         daperpdp = stencil(aperp_hi,aperp_low)
         dapardp  = stencil(apar_hi,apar_low)
         if self.AP: result += AP_effect(daperpdp,dapardp)

      if As_flag: result *= self.params['A_s']    
      if fNL_flag:
         D = 0.76 * self.cosmo.scale_independent_growth_factor(z) # normalized so D(a) = a in the MD era
         # brute force way of getting the transfer function, normalized to 1 at kmin
         pmatter = compute_matter_power_spectrum(self, z, linear=True)
         T = np.sqrt(pmatter/self.k**self.params['n_s']) ; T /= T[0]
         fNL_factor = 3.*1.68*(b_fid-1.)*self.cosmo.Om_m(0)*100.**2.
         fNL_factor /= D * self.k**2. * T * 299792.458**2.
         
         result *= fNL_factor
      return result


   def compute_dCdp(self, param, X, Y, zmin=None, zmax=None, 
                    relative_step=-1., absolute_step=-1., five_point=False):
      '''
      '''
      default_step = {'tau_reio':0.3,'m_ncdm':0.05,'A_lin':0.002,'A_log':0.002}
        
      if relative_step == -1.: 
         try: relative_step = default_step[param]
         except: relative_step = 0.01
      if absolute_step == -1.: 
         try: absolute_step = default_step[param]
         except: absolute_step = 0.01
            
      if zmin is not None and zmax is not None: zmid = (zmin+zmax)/2
      else: zmid = self.experiment.zcenters[0] # Ckk, where b and stuff don't matter
            
      b_fid = compute_b(self,zmid)      
      if zmid < 6:
         alpha0_fid = 1.22 + 0.24*b_fid**2*(zmid-5.96)
      else: 
         alpha0_fid = 0. 
      noise = 1/compute_n(self,zmid)
      if self.experiment.HI: noise = castorinaPn(zmid)
          
      kwargs = {'fishcast':self, 'X':X, 'Y':Y, 'zmin':zmin, 'zmax':zmax,
                'zmid':zmid,'gamma':1, 'b':b_fid, 'b2':8*(b_fid-1)/21,
                'bs':-2*(b_fid-1)/7,'alpha0':alpha0_fid,'alphax':0,'N':noise}
          
      fNL_flag = False
      if param == 'f_NL': fNL_flag = True ; param = 'b'
                
      if param in kwargs: 
         default_value = kwargs[param]
         up = default_value*(1+relative_step)
         if default_value == 0: up = absolute_step
         down = default_value*(1-relative_step)
         if default_value == 0: down = -absolute_step
         upup = default_value*(1+2*relative_step)
         if default_value == 0: upup = 2*absolute_step
         downdown = default_value*(1-2*relative_step)
         if default_value == 0: downdown = -2*absolute_step
         step = up - default_value
         kwargs[param] = up
         P_dummy_hi = compute_lensing_Cell(**kwargs)
         kwargs[param] = down
         P_dummy_low = compute_lensing_Cell(**kwargs)
         if five_point:
            kwargs[param] = upup
            P_dummy_higher = compute_lensing_Cell(**kwargs)
            kwargs[param] = downdown
            P_dummy_lower = compute_lensing_Cell(**kwargs)
         kwargs[param] = default_value
         if five_point: result = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. *  step) 
         else:          result =  (P_dummy_hi-P_dummy_low) / (2. * step)
         if fNL_flag:
            D = 0.76 * self.cosmo.scale_independent_growth_factor(zmid) # normalized so D(a) = a in the MD era
            # brute force way of getting the transfer function, normalized to 1 at kmin
            pmatter = compute_matter_power_spectrum(self, zmid, linear=True)
            T = np.sqrt(pmatter/self.k**self.params['n_s']) ; T /= T[0]
            fNL_factor = 3.*1.68*(b_fid-1.)*self.cosmo.Om_m(0)*100.**2.
            fNL_factor /= D * self.k**2. * T * 299792.458**2.
            fNL_interp = self.get_f_at_fixed_mu(fNL_factor,0.)
            chi = (1+zmid)*self.cosmo.angular_distance(zmid)*self.params_fid['h']
            fNL_ell = fNL_interp((self.ell+0.5)/chi)
            result *= fNL_ell
         return result
    
      P_fid = compute_lensing_Cell(**kwargs)
              
      # brute force numerical differentiation
      flag = False 
      if param == 'log(A_s)': flag = True ; param = 'A_s'  
        
      default_value = self.params_fid[param] 
        
      if param == 'm_ncdm' and self.params['N_ncdm']>1:
         # CLASS takes a string as an input when there is more than one massive neutrino
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
         step = (up-down)/2
         if default_value == 0.: step = absolute_step
            
    
      def set_param(value):
         self.cosmo.set({param : value})
         #self.params[param] = value
      
      set_param(up)
      self.cosmo.compute()
      P_dummy_hi = compute_lensing_Cell(**kwargs)
      set_param(down)
      self.cosmo.compute()
      P_dummy_low = compute_lensing_Cell(**kwargs)
      if five_point:
         set_param(upup)
         self.cosmo.compute()
         P_dummy_higher = compute_lensing_Cell(**kwargs)
         set_param(downdown)
         self.cosmo.compute()
         P_dummy_lower = compute_lensing_Cell(**kwargs)
      set_param(default_value)
      self.cosmo.compute()
      if five_point: result = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
      else: result = (P_dummy_hi-P_dummy_low)/(2.*step)
      if flag: result *= self.params['A_s']
      
      return result

       
   def compute_derivatives(self, five_point=True, parameters=None, z=None, overwrite=False, one_sided=False):
      '''
      Calculates all the derivatives and saves them to the 
      output/forecast name/derivatives directory
      '''
      if parameters is not None: 
         for i,p in enumerate(parameters):
            if p == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(int(100*z[i]))+'.txt'
            elif p == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(int(100*z[i]))+'.txt'
            elif p == 'A_log': filename = 'A_log_'+str(int(100.*self.omega_log))+'_'+str(int(100*z[i]))+'.txt'
            else: filename = p+'_'+str(int(100*z[i]))+'.txt'
            folder = '/derivatives/'
            if self.recon: folder = '/derivatives_recon/'
            fname = self.basedir+'output/'+self.name+folder+filename
            if not exists(fname) or overwrite: 
               dPdp = self.compute_dPdp(param=p, z=z[i], five_point=five_point, one_sided=one_sided)
               np.savetxt(fname,dPdp)
            else:
               continue
         return
      zs = self.experiment.zcenters
      for z in zs:
         for free_param in self.free_params:
            if free_param == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(int(100*z))+'.txt'
            elif free_param == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(int(100*z))+'.txt'
            elif free_param == 'A_log': filename = 'A_log_'+str(int(100.*self.omega_log))+'_'+str(int(100*z))+'.txt'
            else: filename = free_param+'_'+str(int(100*z))+'.txt'
            folder = '/derivatives/'
            if self.recon: folder = '/derivatives_recon/'
            fname = self.basedir+'output/'+self.name+folder+filename
            if not exists(fname) or overwrite:
               dPdp = self.compute_dPdp(param=free_param, z=z, five_point=five_point, one_sided=one_sided)
               np.savetxt(fname,dPdp)
            else:
               continue
       
    
   def compute_Cl_derivatives(self, five_point=True, overwrite=False):
      '''
      Calculates the derivatives of Ckk, Ckg, and Cgg with respect to 
      each of the free_params. 
      '''
      zs = self.experiment.zedges
      for free_param in self.free_params:
         #
         if free_param != 'gamma':
            filename = 'Ckk_'+free_param+'.txt'
            fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
            if not exists(fname) or overwrite: 
               dCdp = self.compute_dCdp(free_param, 'k', 'k', five_point=five_point)
               np.savetxt(fname,dCdp)
            else:
               continue
         else:
            for i,z in enumerate(zs[:-1]):
               filename = 'Ckk_'+free_param+'_'+str(int(100*zs[i]))+'_'+str(int(100*zs[i+1]))+'.txt'
               fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
               if not exists(fname) or overwrite: 
                  dCdp = self.compute_dCdp(free_param, 'k', 'k',zmin=zs[i],zmax=zs[i+1], five_point=five_point)
                  np.savetxt(fname,dCdp)
               else:
                  continue
         #
         for i,z in enumerate(zs[:-1]):
            filename = 'Cgg_'+free_param+'_'+str(int(100*zs[i]))+'_'+str(int(100*zs[i+1]))+'.txt'
            fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
            if not exists(fname) or overwrite: 
               dCdp = self.compute_dCdp(free_param, 'g', 'g', zmin=zs[i], zmax=zs[i+1], five_point=five_point)
               np.savetxt(fname,dCdp) 
            else:
               continue
            
            filename = 'Ckg_'+free_param+'_'+str(int(100*zs[i]))+'_'+str(int(100*zs[i+1]))+'.txt'
            fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
            if not exists(fname) or overwrite: 
               dCdp = self.compute_dCdp(free_param, 'k', 'g', zmin=zs[i], zmax=zs[i+1], five_point=five_point)
               np.savetxt(fname,dCdp)
            else:
               continue
            
                        
   def load_derivatives(self, basis, log10z_c=-1.,omega_lin=-1,omega_log=-1,polys=True):
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
      if omega_log == -1. : omega_log = self.omega_log
        
      nbins = self.experiment.nbins
      N = len(basis)
      if self.recon and polys: N += 15
      derivatives = np.empty((nbins,N,self.Nk*self.Nmu))
      folder = '/derivatives/'
      if self.recon: folder = '/derivatives_recon/'
      directory = self.basedir+'output/'+self.name+folder
        
      for zbin_index in range(nbins):
         z = self.experiment.zcenters[zbin_index]
         for i,param in enumerate(basis):
            if param == 'fEDE': filename = 'fEDE_'+str(int(1000.*log10z_c))+'_'+str(int(100*z))+'.txt'
            elif param == 'A_lin': filename = 'A_lin_'+str(int(100.*omega_lin))+'_'+str(int(100*z))+'.txt'
            elif param == 'A_log': filename = 'A_log_'+str(int(100.*omega_log))+'_'+str(int(100*z))+'.txt'
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


   def gen_fisher(self,basis,globe,log10z_c=-1.,omega_lin=-1.,omega_log=-1.,kmax_knl=1.,
                  kmin=0.003,kmax=-10.,kpar_min=-1.,mu_min=-1,derivatives=None,
                  zbins=None,polys=True,simpson=False,nratio=1.):
      '''
      Computes an array of Fisher matrices, one for each redshift bin.
      '''
      if log10z_c == -1. : log10z_c = self.log10z_c
      if omega_lin == -1. : omega_lin = self.omega_lin
      if omega_log == -1. : omega_log = self.omega_log
        
      if derivatives is None: derivatives = self.load_derivatives(basis,log10z_c=log10z_c,
                                                                  omega_lin=omega_lin,
                                                                  omega_log=omega_log,polys=polys)   
      if zbins is None: zbins = range(self.experiment.nbins)
           
      def fish(zbin_index):
         n = len(basis)
         if self.recon: n += 15
         F = np.zeros((n,n))
         z = self.experiment.zcenters[zbin_index]
         dPdvecp = derivatives[zbin_index]
         Cinv = 1./compute_covariance_matrix(self,zbin_index,nratio=nratio)
         mus = self.mu.reshape(self.Nk,self.Nmu)[0] 
         ks = self.k.reshape(self.Nk,self.Nmu)[:,0] 
         constraints = self.compute_wedge(z,kmin=kmin)*self.kmax_constraint(z,kmax_knl)
         if kmax > 0: constraints = self.compute_wedge(z,kmin=kmin)*(self.k<kmax)
         constraints *= (self.k > kmin)
         kpar = self.k*self.mu
         kperp = self.k*np.sqrt(1-self.mu**2)
         if kpar_min > 0: constraints *= (kpar>kpar_min)
         if mu_min > 0: constraints *= (kpar > kperp*mu_min/np.sqrt(1-mu_min**2))
         for i in range(n):
            for j in range(n):
               integrand = (dPdvecp[i]*Cinv*dPdvecp[j]*constraints) 
               integrand *= self.kpar_cut[zbin_index] # N2 < 20 % of power
               if simpson:
                  
                  ##########################################################################################
                  # this is wrong, need to factor out dk dmu
                    
                  integrand_rshpd = integrand.reshape(self.Nk,self.Nmu) 
                  partial_integrand = simps(integrand_rshpd,x=mus) # integrate over mu
                  F[i,j] = simps(partial_integrand,x=ks) # integrate over k
               else:
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
         result[m] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
         filename = 'Cgg_'+param+'_'+str(int(100*zs[m-1]))+'_'+str(int(100*zs[m]))+'.txt' 
         result[m+n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
         return result
    
      if not 'gamma' in param: 
         result[0] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/Ckk_'+param+'.txt')
      else:
         idx = int(param[-1])
         filename = 'Ckk_gamma_'+str(int(100*zs[idx-1]))+'_'+str(int(100*zs[idx]))+'.txt'
         result[0] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename) 
      for i in range(1,n+1):
         if 'gamma' in param: 
            idx = int(param[-1])
            filename = 'Ckg_gamma_'+str(int(100*zs[idx-1]))+'_'+str(int(100*zs[idx]))+'.txt'
            result[idx] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
            filename = 'Cgg_gamma_'+str(int(100*zs[idx-1]))+'_'+str(int(100*zs[idx]))+'.txt'
            result[idx+n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
         else:
            filename = 'Ckg_'+param+'_'+str(int(100*zs[i-1]))+'_'+str(int(100*zs[i]))+'.txt'
            result[i] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
            filename = 'Cgg_'+param+'_'+str(int(100*zs[i-1]))+'_'+str(int(100*zs[i]))+'.txt'
            result[i+n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
      return result

    
   def gen_lensing_fisher(self,basis,globe,ell_min=30,ell_max=1900,kmax_knl=1,
                          CMB='SO',kk=True,only_kk=False,bins=None,fsky_CMB=0.4,
                          fsky_intersect=None):
      '''
      '''
      n = len(basis)
      C = covariance_Cls(self,kmax_knl=kmax_knl,CMB=CMB,fsky_CMB=fsky_CMB,fsky_intersect=fsky_intersect)
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
    
    
   def Nmodes(self,zmin,zmax,nbins,kpar=-1.,kmax=-1,alpha0=-1,alpha2=0,linear=False,halofit=False,nratio=1.):
    
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
         if linear: P_F = P_L + 1/compute_n(self,z)/nratio
         else: P_F = compute_tracer_power_spectrum(self,z,alpha0=alpha0,alpha2=alpha2)
         if halofit: P_F = compute_matter_power_spectrum(self,z) * (b+f*MU**2.)**2. + 1/compute_n(self,z)
         integrand = ( G(z)**2 * P_L / P_F )**2. 
         integrand *= self.compute_wedge(z) 
         if kpar > 0.: integrand *= (self.k*self.mu > kpar)
         if kmax > 0.: integrand *= (self.k < kmax)
         return sum(integrand * self.k**2. * self.dk * self.dmu / (2. * np.pi**2.))
         # we are dividing by 2 pi^2 (and not 4 pi^2) since we integrate from mu = 0 to 1
    
      zedges = np.linspace(zmin,zmax,nbins+1)
      zs = (zedges[1:]+zedges[:-1])/2.
      dV = np.array([self.comov_vol(zedges[i],zedges[i+1]) for i in range(nbins)])
      I = np.array([I1(z) for z in zs])
      return sum(I*dV) 
    
    
   def Nmodes_fixed_k(self,k,zmin,zmax,nbins,Deltak=0.1,alpha0=-1,alpha2=0,kmin=0.001):
    
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
         P_F = compute_tracer_power_spectrum(self,z,alpha0=alpha0,alpha2=alpha2)
         integrand = ( G(z)**2. * P_L / P_F )**2. 
         integrand *= self.compute_wedge(z,kmin=kmin) 
         integrand *= self.k**2. * Deltak * self.dmu / (2. * np.pi**2.)
         # we are dividing by 2 pi^2 (and not 4 pi^2) since we integrate from mu = 0 to 1
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
