from headers import *

class experiment(object):
   '''
   An object that contains all the information related to the experiment
   '''

   def __init__(self, zmin=0.8, zmax=1.2, fsky=0.5, n=12., sigma_z=0.01):

      self.zmin = zmin
      self.zmax = zmax
      self.zmid = np.mean((zmin,zmax))
      self.fsky = fsky
      self.n = n
      self.sigma_z = sigma_z
