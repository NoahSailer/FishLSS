from headers import *

class experiment(object):
   '''
   A class to organize experiment-related parameters. Includes
   several "presets" for LSS tracers (i.e. their b(z) and n(z)).
   
   -------------------------------------------------------------
   '''
   def __init__(self, 
                zmin=0.8,             
                zmax=1.2,           
                nbins=1,            
                zedges=None,            
                fsky=0.5,           
                sigma_z=0.0,         
                n=1e-3,              
                b=1.5,              
                b2=None,              
                bs = None,
                N2 = None,
                N4 = None,
                alpha0 = None,          
                alpha2 = None,
                alpha4 = None,
                alphax = None,
                LBG=False,
                HI=False,               
                Halpha=False,           
                ELG=False,             
                Euclid=False,        
                MSE=False,              
                Roman=False,            
                custom_n=False,         
                custom_b=False,         
                pesimistic=False,      
                Ndetectors=256**2.,     
                fill_factor=0.5,       
                tint=5,             
                sigv=100,             
                D = 6,
                HI_ideal=False):      
                
      params = {'A_s': 2.10732e-9,'n_s': 0.96824,'h': 0.6770,'N_ur': 1.0196,
                'N_ncdm': 2,'m_ncdm': '0.01,0.05','tau_reio': 0.0568,
                'omega_b': 0.02247,'omega_cdm': 0.11923}
      cosmo = Class()
      cosmo.set(params)
      cosmo.compute()
      self.cosmo = cosmo
      self.k = None
      self.mu = None  
      self.LBG = LBG
      self.HI = HI
      self.Halpha = Halpha
      self.ELG = ELG
      self.Euclid = Euclid
      self.MSE = MSE
      self.Roman = Roman
      self.custom_n = custom_n
      self.custom_b = custom_b   
      self.Ndetectors = Ndetectors
      self.fill_factor = fill_factor
      self.tint = tint
      self.D = D  
      if pesimistic: 
         self.N_w = 3.
         self.kparallel_min = 0.1
      else: 
         self.N_w = 1.
         self.kparallel_min = 0.01   

      self.zmin = zmin
      self.zmax = zmax
      self.nbins = nbins
      self.zedges = np.linspace(zmin,zmax,nbins+1)
      if zedges is not None: 
         self.zedges = zedges
         self.nbins = len(zedges)-1
      self.zcenters = (self.zedges[1:]+self.zedges[:-1])/2.
      self.fsky = fsky
      self.sigma_z = sigma_z
      self.sigv = sigv
      # If the number density is not a float, assumed to be a function of z
      if not isinstance(n, float): self.cust_n = n
      else: self.custum_n = lambda z: n + 0.*z
      # If the bias is not a float, assumed to be a function of z
      if not isinstance(b, float): self.cust_b = b
      else: self.cust_b = lambda z: b + 0.*z
      # Assumed fiducial values for higher-order biases 
      if b2 is None: self.b2 = lambda z: 8.*(self.b(z)-1.)/21.
      if bs is None: self.bs = lambda z: -2.*(self.b(z)-1.)/7.
      # Fiducial value for the N2 stochastic term
      zs = np.linspace(0,100,1000)
      h = self.cosmo.pars['h']
      c = 299792.458 # speed of light in km/s
      Hz_fid = np.array([self.cosmo.Hubble(z)*c/h for z in zs])
      self.Hz_fid = interp1d(zs,Hz_fid,kind='linear')
      def noise(z):
         if self.HI: return castorinaPn(z)
         return 1./self.n(z)
      if N2 is None: self.N2 = lambda z: -noise(z)*((1.+z)*self.sigv/self.Hz_fid(z))**2. 
      # 
      if N4 is None: self.N4 = lambda z: 0.*z
      if alpha0 is None: 
         def a0(z):
            if z > 6: return 0
            return 1.22 + 0.24*self.b(z)**2.*(z-5.96)
         self.alpha0 = lambda z: a0(z)
      if alpha2 is None: self.alpha2 = lambda z: 0.*z
      if alpha4 is None: self.alpha4 = lambda z: 0.*z
      if alphax is None: self.alphax = lambda z: 0.*z
      
      
      # interpolating a table
      zs = np.array([2.,3.,3.8,4.9,5.9])
      Muvstar = np.array([-20.60,-20.86,-20.63,-20.96,-20.91])
      self.Muvstar = interp1d(zs, Muvstar, kind='linear', bounds_error=False,fill_value='extrapolate')
      muv = np.array([24.2,24.7,25.4,25.5,25.8])
      self.muv = interp1d(zs, muv, kind='linear', bounds_error=False,fill_value='extrapolate')
      phi = np.array([9.70,5.04,9.25,3.22,1.64])*0.001
      self.phi = interp1d(zs, phi, kind='linear', bounds_error=False,fill_value='extrapolate')
      alpha = np.array([-1.6,-1.78,-1.57,-1.60,-1.87])
      self.alpha = interp1d(zs, alpha, kind='linear', bounds_error=False,fill_value='extrapolate')
   
   ############################################################################
   ##
   ## Preset linear biases
   ##  
   ############################################################################
         
   def LBGb(self,z,m=24.5):
      '''
      Eq. (2.7) of Wilson and White 2019. 
      '''
      A = lambda m: -0.98*(m-25.) + 0.11
      B = lambda m: 0.12*(m-25.) + 0.17
      def b(m): return A(m)*(1.+z)+B(m)*(1.+z)**2.
      return b(m)

   def hAlphaB(self,z):
      '''
      From Table 2 of Merson+19.
      '''
      zs = np.array([0.9,1.,1.2,1.4,1.6,1.8,1.9])
      b = np.array([1.05,1.05,1.17,1.30,1.44,1.6,1.6])
      b_interp = interp1d(zs, b, kind='linear', bounds_error=False, fill_value=0.)
      return b_interp(z) 

   def EuclidB(self,z):
      '''
      From Table 3 of arxiv.org/pdf/1910.09273
      '''
      zs = np.array([0.9,1.,1.2,1.4,1.65,1.8])
      b = np.array([1.46,1.46,1.61,1.75,1.90,1.90])
      b_interp = interp1d(zs, b, kind='linear', bounds_error=False, fill_value=0.)
      return b_interp(z) 

   def Romanb(self,z): return 1.1*z+0.3

   def ELGb(self,z):
      D = self.cosmo.scale_independent_growth_factor(z)
      return 0.84/D

   def MSEb(self,z):
      '''
      Constant clustering approximation.
      '''
      D = self.cosmo.scale_independent_growth_factor(z)
      D0 = self.cosmo.scale_independent_growth_factor(0)
      if z<=2.4: return D0/D
      return self.LBGb(z,m=24.5)

   def HIb(self,z): return castorinaBias(z)
    
   def b(self,z):
      '''
      Quick way of getting the bias. This is what 
      FishLSS always calls to get the bias from
      a forecast.
      '''
      if not self.custom_b:
         if self.LBG:    return self.LBGb(z)
         if self.HI:     return self.HIb(z)
         if self.Halpha: return self.hAlphaB(z)
         if self.ELG:    return self.ELGb(z)
         if self.Euclid: return self.EuclidB(z) 
         if self.MSE:    return self.MSEb(z)
         if self.Roman:  return self.Romanb(z)
      return self.cust_b(z)

   ############################################################################  
   ##
   ## Preset number densities
   ##  
   ############################################################################

   def Muv(self, z, m=24.5):
      '''
      Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
      '''
      result = m - 5. * np.log10(self.cosmo.luminosity_distance(z)*1.e5)
      result += 2.5 * np.log10(1.+z)
      return result

   def muv_from_Muv(self, z, M):
      '''
      Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
      '''
      result = M + 5. * np.log10(self.cosmo.luminosity_distance(z)*1.e5)
      result -= 2.5 * np.log10(1.+z)
      return result

   def LBGn(self, z, m=24.5):
      '''
      Equation 2.5 of Wilson and White 2019. Return number
      density of LBGs at redshift z in units of Mpc^3/h^3.
      '''
      upper_limit = self.Muv(z,m=m)
      integrand = lambda M: (np.log(10.)/2.5) * self.phi(z) * 10.**( -0.4 * (1.+self.alpha(z)) * (M-self.Muvstar(z)) )*\
                                np.exp(-10.**(-0.4 * (M-self.Muvstar(z)) ) )
      return scipy.integrate.quad(integrand, -200, upper_limit)[0]

   def ELGn(self, z):
      zs = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65])
      dNdz = np.array([309,2269,1923,2094,1441,1353,1337,523,466,329,126])
      N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
      volume = np.array([((1.+z+0.05)*self.cosmo.angular_distance(z+0.05))**3. for z in zs])
      volume -= np.array([((1.+z-0.05)*self.cosmo.angular_distance(z-0.05))**3. for z in zs])
      volume *= 4.*np.pi*self.cosmo.pars['h']**3./3. # volume in Mpc^3/h^3
      n = list(N/volume)
      zs = np.array([0.6,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.7])
      n = [n[0]] + n
      n = n + [n[-1]]
      n = np.array(n)
      n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
      return float(n_interp(z))

   def Romann(self, z):
      zs = np.linspace(1.05,2.95,20)
      dNdz = np.array([6160,5907,4797,5727,5147,4530,4792,3870,2857,2277,1725,1215,1642,1615,1305,1087,850,795,847,522])
      N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
      volume = np.array([((1.+z+0.05)*self.cosmo.angular_distance(z+0.05))**3. for z in zs])
      volume -= np.array([((1.+z-0.05)*self.cosmo.angular_distance(z-0.05))**3. for z in zs])
      volume *= 4.*np.pi*self.cosmo.pars['h']**3./3. # volume in Mpc^3/h^3
      n = list(N/volume)
      zs = np.array([zs[0]] + list(zs) + [zs[-1]])
      n = np.array([n[0]] + n + [n[-1]])
      n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
      return float(n_interp(z))

   def Euclidn(self,z):
      '''
      From Table 3 of https://arxiv.org/pdf/1910.09273.pdf
      '''
      zs = np.array([0.9,1.,1.2,1.4,1.65,1.8])
      n = np.array([6.86,6.86,5.58,4.21,2.61,2.61])*1e-4
      n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
      return n_interp(z) 

   def hAlphaN(self, z):
      '''
      Table 2 from Merson+17. Valid for 0.9<z<1.9.
      '''
      zs = np.array([0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])
      dNdz = np.array([10535.,8014.,4998.,3931.,3455.,2446.,2078.,1747.,1524.,1329.])
      N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
      volume = np.array([((1.+z+0.05)*self.cosmo.angular_distance(z+0.05))**3. for z in zs])
      volume -= np.array([((1.+z-0.05)*self.cosmo.angular_distance(z-0.05))**3. for z in zs])
      volume *= 4.*np.pi*self.cosmo.pars['h']**3./3. # volume in Mpc^3/h^3
      n = list(N/volume)
      zs = np.array([0.9,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.9])
      n = [n[0]] + n
      n = n + [n[-1]]
      n = np.array(n)
      n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)
      return n_interp(z)


   def MSEn(self,z,m=24.5):
      # return ELG number density for z<2.4
      if z <= 2.4: return 1.8e-4 
      # interpolate figure 2 of https://arxiv.org/pdf/1903.03158.pdf to get efficiency
      mags   = np.array([22.75, 23.25, 23.75, 24.25])
      zs     = np.array([2.6,3.0,3.4,3.8])
      blue   = np.array([[0.619,0.846,0.994], [0.452,0.745,0.962], [0.269,0.495,0.919], [0.102,0.327,0.908]])
      orange = np.array([[0.582,0.780,0.981], [0.443,0.663,0.929], [0.256,0.481,0.849], [0.119,0.314,0.854]])
      green  = np.array([[0.606,0.805,0.919], [0.486,0.708,0.815], [0.289,0.559,0.746], [0.146,0.363,0.754]])
      red    = np.array([[0.624,0.752,0.934], [0.501,0.671,0.843], [0.334,0.552,0.689], [0.199,0.371,0.699]])
      weight = np.array([0.4,0.3,0.3])
      b,o = np.sum(blue*weight,axis=1),np.sum(orange*weight,axis=1)
      g,r = np.sum(green*weight,axis=1),np.sum(red*weight,axis=1)
      eff = np.array([b,o,r,g])
      #
      efficiency = interp2d(zs,mags,eff,kind='linear',bounds_error=False)
      #
      def integrand(M): 
         result = (np.log(10.)/2.5) * self.phi(z) * 10.**( -0.4 * (1.+self.alpha(z)) * (M-self.Muvstar(z)) )
         result *= np.exp(-10.**(-0.4 * (M-self.Muvstar(z)) ) )
         m = self.muv_from_Muv(z,M)
         result *= efficiency(z,m)
         return result
      #
      n = lambda m: scipy.integrate.quad(integrand, -200, self.Muv(z,m=m))[0]
      return n(m)


   def nofl(self, x, hexpack=True, Nside=256, D=6):
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
         a,b,c,d,e=0.4847, -0.330,  1.3157,  1.5975,  6.8390
      xn=x/(Nside*D)
      n0=(Nside/D)**2
      res=n0*(a+b*xn)/(1+c*xn**d)*np.exp(-(xn)**e)
      return res

   def get_Tb(self, z):
      '''
      Returns the mean 21cm brightness temp in K. 
      If z < 6 use fitting formula from Eq. B1 of
      https://arxiv.org/pdf/1810.09572.
      '''
      if z <= 6:
          Ohi = 4e-4*(1+z)**0.6
          h = self.cosmo.pars['h']
          Ez = self.cosmo.Hubble(z)/self.cosmo.Hubble(0)
          Tb = 188e-3*h/Ez*Ohi*(1+z)**2
          return Tb
      omb = self.cosmo.pars['omega_b']
      omm = self.cosmo.pars['omega_cdm'] + omb
      result = 28e-3 * ((1+z)*0.14/10/omm)**0.5
      result *= omb/0.022
      return result * (1-Xhi(z))
       

   def HI_therm(self, z, effic=0.7, hexpack=True, skycoupling=0.9, 
                Tground=300., omtcoupling=0.9, Tampl=50., old=False):
      '''
      Adapted from https://github.com/slosar/PUMANoise.
      Thermal noise power in Mpc^3/h^3. Thermal noise is 
      given by equation D4 in https://arxiv.org/pdf/1810.09572.
      I divide by Tb (see get_Tb) to convert to Mpc^3/h^3.
      Returns a function of k [h/Mpc] and mu.
      '''
      D = self.D
      ttotal = self.tint*365*24*3600.*self.fill_factor**2
      Nside = np.sqrt(self.Ndetectors/self.fill_factor)
      Hz = self.cosmo.Hubble(z)*(299792.458)/self.cosmo.pars['h'] # in h km/s/Mpc
      Ez = self.cosmo.Hubble(z)/self.cosmo.Hubble(0)
      lam = 0.211 * (1+z) 
      r = (1.+z) * self.cosmo.angular_distance(z)*self.cosmo.pars['h'] # in Mpc/h
      Deff = D * np.sqrt(effic) 
      FOV = (lam / Deff)**2 
      y = 3e5*(1+z)**2/(1420e6*Hz) 
      Sarea=4*np.pi*self.fsky 
      Ae=np.pi/4*D**2*effic
      # k dependent terms
      kperp = lambda k,mu: k*np.sqrt(1.-mu**2.)
      l = lambda k,mu: kperp(k,mu) * r * lam / (2 * np.pi) 
      def Nu(k,mu):
         if old: return nofl(l(k,mu),hexpack=hexpack,Nside=Nside,D=D)*lam**2 
         # this code was used for the baseline density in 2205.11504
         #
         #ll,pi2lnb = np.genfromtxt('input/baseline_bs_44_D_14.txt').T
         #nofl_new = interp1d(ll,pi2lnb/2/np.pi/ll,bounds_error=False,fill_value=0)
         #result = nofl_new(l(k,mu))*lam**2
         #result = np.maximum(result,1e-20)
         #I = np.where(l(k,mu) < D)
         #result[I] = 1e-20
         #return result
      # temperatures
      Tb = get_Tb(z) 
      Tsky = lambda f: 25.*(f/400.)**(-2.75) +2.7
      Tscope = Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling
      Tsys = Tsky(1420./(1+z))+Tscope
      Pn = lambda k,mu: (Tsys/Tb)**2*r**2*y*(lam**4/Ae**2)*1/(2*Nu(k,mu)*ttotal)*(Sarea/FOV)
      return Pn


   def HI_shot(self, z): 
      '''
      PUMA shot noise Mpc^3/h^3 from Emanuele Castorina
      for z < 6. For z > 6 assume that the shot noise
      is 0.
      '''
      if z<= 6: return castorinaPn(z)
      return 1e-10


   def HIneff(self,z,old=True):
      '''
      Effective number density for PUMA. Returns
      an array of length Nk*Nmu.
      '''
      shot = HI_shot(z)
      try:
         therm = HI_therm(z,old=old)(self.k,self.mu)
      except: 
         print('Need k-mu grid to include 21cm thermal noise.')
         print('Please specify (self.k, self.mu) or have a fisherForecast')
         print('object do this for you. Ignoring thermal noise for now.')
         return 1./shot
      return 1./(therm+shot)
      
   def n(self, z):
      '''
      Returns number density h^3/Mpc^3. For HI surveys
      returns an array of length Nk*Nmu, for all other surveys
      return a float.
      '''
      if not self.custom_n:
         if self.LBG:     return LBGn(z)
         if self.Halpha:  return hAlphaN(z)
         if self.ELG:     return ELGn(z)
         if self.HI:      return HIneff(z)
         if self.Euclid:  return Euclidn(z)
         if self.MSE:     return MSEn(z)
         if self.Roman:   return Romann(z)
      return self.cust_n(z)
