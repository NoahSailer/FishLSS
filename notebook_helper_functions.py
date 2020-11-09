from headers import *

def meld_fishers(Fs):
   '''
   Takes a list of fisher matrices as an input. These matrices may
   have different dimensions. However, it is assumed that the matrices
   are in the same basis (I assume a smaller matrix is a top-left submatrix
   of the full fisher matrix). This function increases the sizes of the
   smaller matrices (setting all additional components to zero), and 
   then adds all the matrices together.
   '''
   fisher_dimensions = [len(Fs[i]) for i in range(len(Fs))]
   largest_dimension = max(fisher_dimensions)
   for i in range(len(fisher_dimensions)):
      f = Fs[i]
      if len(f) < largest_dimension:
         extended_fisher = np.zeros((largest_dimension,largest_dimension))
         for j in range(len(f)):
            for k in range(len(f)):
               extended_fisher[j,k] = f[j,k]
         Fs[i] = extended_fisher
   # sum all the fisher matrices
   F = sum(Fs)
   return F


def pretty_table(Fs,basis,cosmo_params,scaling,name='',Fs2=None,HI=False):
   '''
   ASSUMES A PARTICULAR BASIS THAT IS VERY SPECIFIC.
   '''
   F = meld_fishers(Fs)
   indices = np.array([np.where(cosmo_params[i]==basis)[0][0] for i in range(6,len(cosmo_params))])
   sigmas = list(np.sqrt(np.diag(np.linalg.inv(F[:14,:14])))[:6])
   for index in indices:
      x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,index]
      if HI: x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,index]
      sigma = np.sqrt(np.linalg.inv(F[x][:,x])[-1,-1])
      sigmas += [sigma]
   sigmas = np.array(sigmas)
   sigmas *= scaling
   sigmas = np.round(sigmas,2)
   result = name
   if Fs2 is None:
      for i in range(len(sigmas)-1): result += ' $'+str(sigmas[i])+'$ &'
      result += ' $'+str(sigmas[-1]) + r'$ \\'
      return result
    
   F2 = meld_fishers(Fs2)
   sigmas2 = list(np.sqrt(np.diag(np.linalg.inv(F2[:14,:14])))[:6])
   for index in indices:
      x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,index]
      if HI: x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,index]
      sigma = np.sqrt(np.linalg.inv(F2[x][:,x])[-1,-1])
      sigmas2 += [sigma]
   sigmas2 = np.array(sigmas2)
   sigmas2 *= scaling
   sigmas2 = np.round(sigmas2,2)

   for i in range(len(sigmas)-1): result += ' $'+str(sigmas[i])+'/'+str(sigmas2[i])+'$ &'
   result += ' $'+str(sigmas[-1]) + r'/' + str(sigmas2[-1]) + r'$ \\'
   return result    


def one_param_constraints(Fs,basis,cosmo_params,HI=False):
   '''
   ASSUMES A PARTICULAR BASIS THAT IS VERY SPECIFIC.
   '''
   F = meld_fishers(Fs)
   indices = np.array([np.where(cosmo_params[i]==basis)[0][0] for i in range(6,len(cosmo_params))])
   sigmas = list(np.sqrt(np.diag(np.linalg.inv(F[:14,:14])))[:6])
   for index in indices:
      x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,index]
      if HI: x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,index]
      sigma = np.sqrt(np.linalg.inv(F[x][:,x])[-1,-1])
      sigmas += [sigma]
   sigmas = np.array(sigmas)
   return sigmas