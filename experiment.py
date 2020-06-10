from headers import *

class experiment(object):
   '''
   An object that contains all the information related to the experiment
   '''

   def __init__(self, zmin=0.8, zmax=1.2, nbins=1, zedges=None, fsky=0.5, sigma_z=0.01, n=12., LBG=False, HI=False):

      self.zmin = zmin
      self.zmax = zmax
      self.nbins = nbins
      self.zedges = np.linspace(zmin,zmax,nbins+1)
      if zedges is not None: self.zedges = zedges
      self.zcenters = (self.zedges[1:]+self.zedges[:-1])/2.
      self.fsky = fsky
      self.sigma_z = sigma_z
      self.n = n*np.ones(nbins) #add option to set n to be an array
      self.LBG = LBG
      self.HI = HI
