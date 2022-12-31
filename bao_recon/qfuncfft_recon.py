import numpy as np

from bao_recon.loginterp import loginterp
from bao_recon.spherical_bessel_transform import SphericalBesselTransform



class QFuncFFT:
    '''
       Class to calculate all the functions of q, X(q), Y(q), U(q), xi(q) etc.
       Modified from velocilptors for zeldovich reconstruction.
       
       Throughout we use the ``generalized correlation function'' notation of 1603.04405.
              
       Note that one should always cut off the input power spectrum above some scale.
       I use exp(- (k/20)^2 ) but a cutoff at scales twice smaller works equivalently,
       and probably beyond that. The important thing is to keep all integrals finite.
       This is done automatically in the Zeldovich class.
       
       Currently using the numpy version of fft. The FFTW takes longer to start up and
       the resulting speedup is unnecessary in this case.
       
    '''
    def __init__(self, k, p, qv = None, pair_type='matter', shear = False, low_ring=True):

        self.shear = shear
        self.pair_type = pair_type
        
        self.k = k
        self.p = p

        if qv is None:
            self.qv = np.logspace(-5,5,2e4)
        else:
            self.qv = qv
        
        self.sph = SphericalBesselTransform(self.k, L=5, low_ring=True, fourier=True)
        
        self.setup_xiln()

    def setup_xiln(self):
        
        # Compute a bunch of generalized correlation functions
        
        if self.pair_type == 'disp x disp' or self.pair_type == 'matter':
            # this is what we need for Aij
            self.xi0m2 = self.xi_l_n(0,-2, side='right') # since this approaches constant on the left only interpolate on right
            self.xi2m2 = self.xi_l_n(2,-2)

        if self.pair_type == 'disp x bias' or self.pair_type == 'matter':
            self.xi1m1 = self.xi_l_n(1,-1)
            if self.shear:
                self.xi3m1 = self.xi_l_n(3,-1)
            
        if self.pair_type == 'matter':
            self.xi00 = self.xi_l_n(0,0)
            self.xi20 = self.xi_l_n(2,0)
            self.xi40 = self.xi_l_n(4,0)
        
    
    def xi_l_n(self, l, n, _int=None, extrap=False, qmin=1e-3, qmax=1000, side='both'):
        '''
        Calculates the generalized correlation function xi_l_n, which is xi when l = n = 0
        
        If _int is None assume integrating the power spectrum.
        '''
        if _int is None:
            integrand = self.p * self.k**n
        else:
            integrand = _int * self.k**n
        
        qs, xint =  self.sph.sph(l,integrand)

        if extrap:
            qrange = (qs > qmin) * (qs < qmax)
            return loginterp(qs[qrange],xint[qrange],side=side)(self.qv)
        else:
            return np.interp(self.qv, qs, xint)


