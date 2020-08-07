from headers import *
import types

class experiment(object):
   '''
   An object that contains all the information related to the experiment
   '''

   def __init__(self, zmin=0.8, zmax=1.2, nbins=1, zedges=None, fsky=0.5, sigma_z=0.0, n=12., b=1.5, 
                LBG=False, HI=False, Halpha=False, ELG=False, custom_n=False, custom_b=False, N_w=1.5):

      self.zmin = zmin
      self.zmax = zmax
      self.nbins = nbins
      self.zedges = np.linspace(zmin,zmax,nbins+1)
      if zedges is not None: self.zedges = zedges
      self.zcenters = (self.zedges[1:]+self.zedges[:-1])/2.
      self.fsky = fsky
      self.sigma_z = sigma_z
      self.n = n*np.ones(nbins) #add option to set n to be an array
      if isinstance(b, types.LambdaType): self.b = b
      else: self.b = lambda z: b + 0.*z
      self.LBG = LBG
      self.HI = HI
      self.Halpha = Halpha
      self.ELG = ELG
      self.N_w = N_w
      self.custom_n = custom_n
      self.custom_b = custom_b