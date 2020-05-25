from headers import *

def compute_covariance_matrix(fishcast):
   '''
   Return a square array of linear dimension Nk*Nmu. 
   '''
   prefactor = (4.*np.pi**2.) / (fishcast.dk*fishcast.dmu*fishcast.Vsurvey*fishcast.k**2.)
   diagonal_values = prefactor * (fishcast.P_fid + 1./fishcast.experiment.n)**2.
   return np.diag(diagonal_values)


