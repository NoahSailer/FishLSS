from headers import *
import types

class experiment(object):
   '''
   An object that contains all the information related to the experiment
   '''
   def __init__(self, 
                zmin=0.8,               # Minimum redshift of survey
                zmax=1.2,               # Maximum redshift of survey
                nbins=1,                # Number of redshift bins
                zedges=None,            # Optional: Edges of redshift bins. Default is evenly-spaced bins.
                fsky=0.5,               # Fraction of sky observed
                sigma_z=0.0,            # Redshift error sz/(1+z)
                n=12.,                  # Galaxy number density, float (constant n) or function of z
                b=1.5,                  # Galaxy bias, float (constant b) or function of z
                LBG=False,              # 
                HI=False,               # 
                Halpha=False,           # 
                ELG=False,              #
                Euclid=False,           #
                MSE=False,              # 
                custom_n=False,         #
                custom_b=False,         #
                pesimistic=False,       # HI survey: specifies the k-wedge 
                Ndetectors=256**2.,     # HI survey: number of detectors
                fill_factor=0.5,        # HI survey: the array's fill factor 
                tint=5,                 # HI survey: oberving time [years]
                sigv=100):              # comoving velocity dispersion for FoG contribution [km/s]

      self.zmin = zmin
      self.zmax = zmax
      self.nbins = nbins
      self.zedges = np.linspace(zmin,zmax,nbins+1)
      if zedges is not None: self.zedges = zedges
      self.zcenters = (self.zedges[1:]+self.zedges[:-1])/2.
      self.fsky = fsky
      self.sigma_z = sigma_z
      # If the number density is not a float, assumed to be a function of z
      if not isinstance(n, float): self.n = n
      else: self.n = lambda z: n + 0.*z
      # If the bias is not a float, assumed to be a function of z
      if not isinstance(b, float): self.b = b
      else: self.b = lambda z: b + 0.*z
      self.LBG = LBG
      self.HI = HI
      self.Halpha = Halpha
      self.ELG = ELG
      self.Euclid = Euclid
      self.MSE = MSE
      self.custom_n = custom_n
      self.custom_b = custom_b
      self.Ndetectors = Ndetectors
      self.fill_factor = fill_factor
      self.tint = tint
      self.sigv = sigv
      if pesimistic: 
         self.N_w = 3.
         self.kparallel_min = 0.1
      else: 
         self.N_w = 1.
         self.kparallel_min = 0.01
