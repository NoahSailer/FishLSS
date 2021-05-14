import numpy as np
from bao_recon.loginterp import loginterp
import time

from scipy.interpolate import interp1d
from scipy.special import hyp2f1, hyperu, gamma

from bao_recon.spherical_bessel_transform_fftw import SphericalBesselTransform
from bao_recon.qfuncfft_recon import QFuncFFT
import pyfftw

class Zeldovich_Recon:
    '''
    Class to evaluate Zeldovich power spectra post-reconstruction.
    
    Based on the soon-to-be-available velocilptors code.
    
    '''

    def __init__(self, k, p, R = 15., cutoff=20, jn=15, N = 4000, threads=1, extrap_min = -6, extrap_max = 3, shear = True, import_wisdom=False, wisdom_file='./recon_wisdom.npy'):
    
        self.shear = shear
        
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        
        # set up integration/FFTlog grid
        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        # self up linear correlation between fields:
        Sk = np.exp(-0.5 * (R*self.kint)**2)
        Skm1 = - np.expm1(-0.5 * (R*self.kint)**2)
        
        self.plins = {}
        self.plins['mm'] = self.pint
        self.plins['dd'] = self.pint * Skm1**2
        self.plins['ds'] = -self.pint * Skm1 * Sk
        self.plins['ss'] = self.pint * Sk**2
        self.plins['dm'] = self.pint * Skm1
        self.plins['sm'] = -self.pint * Sk
        
        # ... and calculate Lagrangian correlators
        self.setup_2pts()
        
        # setup hankel transforms for power spectra
        if self.shear:
            self.num_power_components = 10
            self.num_power_components_ds = 4
        else:
            self.num_power_componens = 6
            self.num_power_components_ds = 3
            
            
        self.jn = jn
        self.threads = threads
        
        # if we choose, import fftw wisdom to speed the startup
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.sphx = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components_ds, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.sphs = SphericalBesselTransform(self.qint, L=self.jn, ncol=1, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        
        # indices for the cross
        self.ds_inds = [1,2,4,7]
        
        # misc
        self.sqrtpi = np.sqrt(np.pi)
        #self.gammas = [gamma(n) for n in range(50)] # 50 should be all we ever need...
        #self.gammahalfs = [gamma(n+0.5) for n in range(50)]
        
        
    def setup_2pts(self):
        
        # define the various power spectra and compute xi_l_n
        
        species = ['s','d','m']
        self.qfs = {}
        
        # matter
        self.qfs['mm'] = QFuncFFT(self.kint, self.plins['mm'], pair_type='matter',\
                                                    qv=self.qint,  shear=self.shear)
        # dd, ds, ss
        self.qfs['dd'] = QFuncFFT(self.kint, self.plins['dd'], pair_type='disp x disp',\
                                                    qv=self.qint,  shear=self.shear)
        self.qfs['ds'] = QFuncFFT(self.kint, self.plins['ds'], pair_type='disp x disp',\
                                                    qv=self.qint,  shear=self.shear)
        self.qfs['ss'] = QFuncFFT(self.kint, self.plins['ss'], pair_type='disp x disp',\
                                                    qv=self.qint,  shear=self.shear)
            
        # dm, sm
        self.qfs['dm'] = QFuncFFT(self.kint, self.plins['dm'], pair_type='disp x bias',\
                                                    qv=self.qint,  shear=self.shear)
        self.qfs['sm'] = QFuncFFT(self.kint, self.plins['sm'], pair_type='disp x bias',\
                                                    qv=self.qint,  shear=self.shear)

        # Now piece together the various X, Y, U, ...
        # First for the pure displacements
        pairs = ['mm','dd','ds','ss']
        
        self.Xlins = {}
        self.Ylins = {}
        self.XYlins = {}
        self.yqs = {}
        self.sigmas = {}
        
        for pair in pairs:
            a, b = pair[0], pair[1]
            self.Xlins[pair] = 2./3 * ( 0.5*self.qfs[a+a].xi0m2[0] + 0.5*self.qfs[b+b].xi0m2[0] \
                                            - self.qfs[pair].xi0m2 - self.qfs[pair].xi2m2 )
            self.Ylins[pair] = 2 * self.qfs[pair].xi2m2
            
            self.XYlins[pair] = self.Xlins[pair] + self.Ylins[pair]
            self.sigmas[pair] = self.Xlins[pair][-1]
            self.yqs[pair] = (1*self.Ylins[pair]/self.qint)
            
            
        # Now for the bias x displacmment terms
        pairs = ['mm', 'dm', 'sm']
    
        self.Ulins = {}
        self.Vs = {} # dm, sm
        self.Xs2s = {} # dm, sm
        self.Ys2s = {} # dm, sm
        
        for pair in pairs:
            a, b = pair[0], pair[1]
            self.Ulins[pair] = - self.qfs[pair].xi1m1
            
            if self.shear:
                J2 = 2.*self.qfs[pair].xi1m1/15 - 0.2*self.qfs[pair].xi3m1
                J3 = -0.2*self.qfs[pair].xi1m1 - 0.2*self.qfs[pair].xi3m1
                J4 = self.qfs[pair].xi3m1
                
                self.Vs[pair] = 4 * J2 * self.qfs['mm'].xi20
                self.Xs2s[pair] = 4 * J3**2
                self.Ys2s[pair] = 6*J2**2 + 8*J2*J3 + 4*J2*J4 + 4*J3**2 + 8*J3*J4 + 2*J4**2
                
        # ... and finally the pure bias terms, i.e. matter
        self.corlins = {'mm':self.qfs['mm'].xi00}
        if self.shear:
            self.zetas = {'mm': 2*(4*self.qfs['mm'].xi00**2/45. + 8*self.qfs['mm'].xi20**2/63. + 8*self.qfs['mm'].xi40**2/35)}
            self.chis = {'mm':4*self.qfs['mm'].xi20**2/3.}
                
                
    def setup_method_ii(self):
        '''
            Correlators for method ii.
        '''
        self.XYlin_ds0lag = self.XYlin - self.XYlin[-1]
        self.sigma_ds0lag = self.XYlin_ds0lag[-1]
    
    
    #### Define RSD Kernels #######
    
    def setup_rsd_facs(self,f,nu,D=1):
        # all the usual redshift factors
        self.f = f
        self.nu = nu
        self.D = D
        self.Kfac = np.sqrt(1+f*(2+f)*nu**2); self.Kfac2 = self.Kfac**2
        self.s = f*nu*np.sqrt(1-nu**2)/self.Kfac
        self.c = np.sqrt(1-self.s**2); self.c2 = self.c**2; self.ic2 = 1/self.c2; self.c3 = self.c**3
        self.Bfac = -0.5 * self.Kfac2 * self.Ylin * self.D**2 # this times k is "B"
        
        # for RecIso
        self.kaiser = 1 + f*nu**2
        self.Hlpower = -0.5 * f * nu * np.sqrt(1-f*nu**2) * self.Ylin
        self.Hlfac = - f*nu*np.sqrt(1-nu**2)/self.kaiser
        self.Bfac_ds = -0.5 * self.kaiser * self.Ylin
        
        
    
    def _G0_l_n(self,n,m,k):
        return gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n) * hyp2f1(0.5-n,-n,0.5-m-n,self.ic2)

    
    def _dG0dA_l_n(self,n,m,k):
        x = self.ic2
        fnm = gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n)
        ret = fnm*(2*n/self.c-2*n*self.c)*hyp2f1(0.5-n,-n,0.5-m-n,x)/(k*self.qint)
        if n > 0:
            ret += fnm*n*(n-0.5)/(0.5-m-n)*(2/self.c-2/self.c3)*hyp2f1(1.5-n,1-n,1.5-m-n,x)/(k*self.qint)
        return ret
    
    def _d2G0dA2_l_n(self,n,m,k):
        x = self.ic2
        fnm = gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n)
        ret = fnm /(k*self.qint)**2 * (1-1./x) * ( (2*m-1-4*n*(m+1))*hyp2f1(0.5-n,-n,0.5-m-n,x) \
                                                +(1-4*n**2+m*(4*n-2))*hyp2f1(1.5-n,-n,0.5-m-n,x) )
        return ret

    def _G0_l(self,l,k,nmax=10):
        powerfac = self.Bfac * k**2/self.ic2
        g0l = 0
        
        for ii in range(nmax):
            n = l+ii
            g0l += powerfac**n * self._G0_l_n(n,l,k)
        
        return g0l
    
    def _dG0dA_l(self,l,k,nmax=10):
        powerfac = self.Bfac * k**2/self.ic2
        dg0l = 0
        
        for ii in range(nmax):
            n = l+ii
            dg0l += powerfac**n * self._dG0dA_l_n(n,l,k)
        
        return dg0l
    
    
    def _d2G0dA2_l(self,l,k,nmax=10):
        powerfac = self.Bfac * k**2/self.ic2
        dg0l = 0
        
        for ii in range(nmax):
            n = l+ii
            dg0l += powerfac**n * self._d2G0dA2_l_n(n,l,k)
        
        return dg0l
        
    def _H0_l_np(self,N,M,A):
        return (-1)**(N-M) * A**(2*M-N) * gamma(M+0.5)/gamma(2*M+1)/gamma(2*M-N+1)/gamma(N-M+1)#/self.sqrtpi
        
    def _H0_l(self,N,k):
        # This can probably be sped up using an interpolation routine later
        powerfac = k**2 * self.Hlpower
        ret = 0
        for ii in np.arange(np.ceil(0.5*N),N+1):
            ret += self._H0_l_np(N,ii,powerfac)
        return ret
    
    def _K_ds_n(self,n,k,lmax=8,power=0):
        ksq = k**2
        ret = 0
        
        if power == 0 or power == 1:
            for ll in range(lmax):
                ret += self.Hlfac**ll * self._H0_l(ll,k) * hyperu(-ll,n-ll+1,-ksq*self.Bfac_ds)
        elif power == 2:
            B = ksq * self.Bfac_ds
            for ll in range(lmax):
                ret += self.Hlfac**ll * self._H0_l(ll,k) * (hyperu(-ll,n-ll+1,-B) \
                                                       + n/B*hyperu(-ll,n-ll,-B))
    
        return ret/self.sqrtpi # put the numerical factor here to avoid many evaluations
    
    
    # Now define the actual integrals
                
    def p_integrals(self, k, nmax=8):
        '''
        Pre-reconstruction and post-reconstruction 'dd' spectra.
        '''
        K = k*self.Kfac; Ksq = K**2

        expon = np.exp(-0.5*Ksq * (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * self.sigma)
            
            
        A = k*self.qint*self.c
        d2Gs = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)]
        dGs = [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0]
        G0s = [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0]
                
        G1s = [-(dGs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G2s = [-(d2Gs[ii] + A * dGs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )

        for l in range(self.jn):
            bias_integrands[0,:] = 1. * G0s[l] # za
            bias_integrands[1,:] = -2 * K * self.Ulin * G1s[l]  # b1
            bias_integrands[2,:] = self.corlin * G0s[l] - Ksq*self.Ulin**2 * G2s[l]# b1sq
            bias_integrands[3,:] = - Ksq * self.Ulin**2 * G2s[l] # b2
            bias_integrands[4,:] = (-2 * K * self.Ulin * self.corlin) * G1s[l] # b1b2
            bias_integrands[5,:] = 0.5 * self.corlin**2  * G0s[l] # b2sq
            
            if self.shear:
                bias_integrands[6,:] = -Ksq * (self.Xs2 * G0s[l] + self.Ys2 * G2s[l] )# bs
                bias_integrands[7,:] = -2*K*self.V * G1s[l] # b1bs
                bias_integrands[8,:] = self.chi  * G0s[l] # b2bs
                bias_integrands[9,:] = self.zeta * G0s[l]# bssq

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                    
                # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret

    def make_ptable(self, f, nu, D = 1, ks = None, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        
        D2 = D**2; D4 = D2**2
        
        pair = 'mm'
        self.Xlin = self.Xlins[pair] * D2
        self.Ylin = self.Ylins[pair] * D2
        self.sigma = self.sigmas[pair] * D2
        self.yq = self.yqs[pair] * D2
        self.XYlin = self.XYlins[pair] * D2
        
        self.Ulin = self.Ulins[pair] * D2
        self.corlin = self.corlins[pair] * D2
        
        if self.shear:
            self.V = self.Vs[pair] * D4
            self.Xs2 = self.Xs2s[pair] * D4; self.sigmas2 = self.Xs2[-1]
            self.Ys2 = self.Ys2s[pair] * D4
            self.chi = self.chis[pair] * D4
            self.zeta = self.zetas[pair] * D4
            
        
        self.setup_rsd_facs(f,nu)
        
        if ks is None:
            self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            self.pktable = np.zeros([len(ks), self.num_power_components+1])
            kv = np.array(ks)
        
        self.pktable[:, 0] = kv[:]; N = len(kv)
        for foo in range(N):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo])
            
            
    def make_pddtable(self, f, nu, D = 1, ks = None, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        D2 = D**2; D4 = D2**2
        
        pair = 'dd'
        self.Xlin = self.Xlins[pair] * D2
        self.Ylin = self.Ylins[pair] * D2
        self.sigma = self.sigmas[pair] * D2
        self.yq = self.yqs[pair] * D2
        self.XYlin = self.XYlins[pair] * D2
        
        self.Ulin = self.Ulins['dm'] * D2
        self.corlin = self.corlins['mm'] * D2
        
        if self.shear:
            self.V = self.Vs['dm'] * D4
            self.Xs2 = self.Xs2s['dm'] * D4; self.sigmas2 = self.Xs2[-1]
            self.Ys2 = self.Ys2s['dm'] * D4
            self.chi = self.chis['mm'] * D4
            self.zeta = self.zetas['mm'] * D4
            
        self.setup_rsd_facs(f,nu)
        
        if ks is None:
            self.pktable_dd = np.zeros([nk, self.num_power_components+1]) # one column for ks
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            self.pktable_dd = np.zeros([len(ks), self.num_power_components+1])
            kv = np.array(ks)
        
        self.pktable_dd[:, 0] = kv[:]; N = len(kv)
        for foo in range(N):
            self.pktable_dd[foo, 1:] = self.p_integrals(kv[foo])
            
        
        
    def pds_integrals_RecSym(self, k, nmax=8):
        '''
        Only a small subset of terms included for now for testing.
        '''
        K = k*self.Kfac; Ksq = K**2

        expon = np.exp(-0.5*Ksq *  (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * self.sigma)
            
            
        A = k*self.qint*self.c
        d2Gs = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)]
        dGs = [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0]
        G0s = [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0]
                
        G1s = [-(dGs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G2s = [-(d2Gs[ii] + A * dGs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        
        ret = np.zeros(self.num_power_components_ds)
        bias_integrands = np.zeros( (self.num_power_components_ds,self.N)  )
        
        for l in range(self.jn):
            bias_integrands[0,:] = 1. * G0s[l] # za
            bias_integrands[1,:] = - K * self.Ulin * G1s[l]  # b1
            bias_integrands[2,:] = - 0.5 * Ksq * self.Ulin**2 * G2s[l] # b2

            
            if self.shear:
                bias_integrands[3,:] = - 0.5 * Ksq * (self.Xs2 * G0s[l] + self.Ys2 * G2s[l])# bs

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                    
            # do FFTLog
            ktemps, bias_ffts = self.sphx.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret

    def pds_integrals_RecIso(self, k, nmax=8):
        '''
        Only a small subset of terms included for now for testing.
        '''
        ksq = k**2
        
        expon = np.exp(-0.5*ksq * self.kaiser * (self.XYlin_ds0lag - self.sigma_ds0lag))
        exponm1 = np.expm1(-0.5*ksq * self.kaiser * (self.XYlin_ds0lag - self.sigma_ds0lag))
        suppress = np.exp(-0.5*ksq * self.kaiser * self.sigma_ds0lag)
        damp_fac = np.exp(-0.25 * ksq * (self.Kfac2 * self.sigma_dd + self.sigma_ds))
        
        K0s = [ self._K_ds_n(l,k,lmax=nmax) for l in range(self.jn)  ]
        K1s = [0,] + [ k0/self.kaiser/self.yq/k for k0 in K0s]
        K2s = [ self._K_ds_n(l,k,lmax=nmax,power=2) for l in range(self.jn)  ]
        
        ret = np.zeros(self.num_power_components_ds)
        bias_integrands = np.zeros( (self.num_power_components_ds,self.N)  )
        
        for l in range(self.jn):
            bias_integrands[0,:] = 1. * K0s[l] # za
            bias_integrands[1,:] = - k * self.Ulin * K1s[l]# b1
            bias_integrands[2,:] = - 0.5 * ksq * self.Ulin**2 * K2s[l] # b2

            if self.shear:
                bias_integrands[3,:] = - 0.5 * ksq * (self.Xs2 * K0s[l] + self.Ys2 * K2s[l])# bs

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
                                                                    
            # do FFTLog
            ktemps, bias_ffts = self.sphx.sph(l, bias_integrands)
            ret += self.kaiser**l * k**l * interp1d(ktemps, bias_ffts)(k)

        return 4 * damp_fac * suppress*np.pi*ret


    def make_pdstable(self, f, nu, D = 1, ks = None, kmin = 1e-3, kmax = 3, nk = 100, method='RecSym'):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        
        '''
        D2 = D**2; D4 = D2**2
        
        pair = 'ds'
        self.Xlin = self.Xlins[pair] * D2
        self.Ylin = self.Ylins[pair] * D2
        self.sigma = self.sigmas[pair] * D2
        self.yq = self.yqs[pair] * D2
        self.XYlin = self.XYlins[pair] * D2
        
        self.sigma_dd = D2 * self.sigmas['dd']
        self.sigma_ds = D2 * self.sigmas['ss']
        
        self.Ulin = self.Ulins['sm'] * D2
        
        if self.shear:
            self.Xs2 = self.Xs2s['sm'] * D4; self.sigmas2 = self.Xs2[-1]
            self.Ys2 = self.Ys2s['sm'] * D4

        self.setup_rsd_facs(f,nu)
        self.setup_method_ii()
        
        if ks is None:
            self.pktable_ds = np.zeros([nk, self.num_power_components+1]) # one column for ks
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            self.pktable_ds = np.zeros([len(ks), self.num_power_components+1])
            kv = np.array(ks)
        
        self.pktable_ds[:, 0] = kv[:]; N = len(kv)

        if method == 'RecSym':
            for foo in range(N):
                self.pktable_ds[foo, self.ds_inds] = self.pds_integrals_RecSym(kv[foo])
        elif method == 'RecIso':
            for foo in range(N):
                self.pktable_ds[foo, self.ds_inds] = self.pds_integrals_RecIso(kv[foo])
            
    def pss_integrals(self, k, nmax=8):
        '''
        Only a small subset of terms included for now for testing.
        '''
        K = k*self.Kfac; Ksq = K**2

        expon = np.exp(-0.5*Ksq *  (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * self.sigma)
            
        A = k*self.qint*self.c
        G0s = [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0]
        
        ret = 0
        bias_integrands = np.zeros( (1,self.N)  )
        
        
        for l in range(self.jn):
            bias_integrands[0,:] = 1. * G0s[l] # za

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                    
            # do FFTLog
            ktemps, bias_ffts = self.sphs.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret

    def make_psstable(self, f, nu, D = 1, ks = None, kmin = 1e-3, kmax = 3, nk = 100, method='RecSym'):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        D2 = D**2
        
        pair = 'ss'
        self.Xlin = self.Xlins[pair] * D2
        self.Ylin = self.Ylins[pair] * D2
        self.sigma = self.sigmas[pair] * D2
        self.yq = self.yqs[pair] * D2
        self.XYlin = self.XYlins[pair] * D2
        
        self.setup_rsd_facs(f,nu)
        
        if ks is None:
            self.pktable_ss = np.zeros([nk, self.num_power_components+1]) # one column for ks
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            self.pktable_ss = np.zeros([len(ks), self.num_power_components+1])
            kv = np.array(ks)
        
        self.pktable_ss[:, 0] = kv[:]; N = len(kv)
        
        if method == 'RecSym':
            for foo in range(N):
                self.pktable_ss[foo, 1] = self.pss_integrals(kv[foo])
        elif method == 'RecIso':
            self.setup_rsd_facs(0,0)
            for foo in range(N):
                self.pktable_ss[foo, 1] = self.pss_integrals(kv[foo])

    # Compute multipoles directly

    def make_pltable(self,f, D = 1,ngauss = 3, kv = None, kmin = 1e-3, kmax = 0.5, nk = 30, nmax=8, method = 'Pre-Recon', a_perp = 1, a_par = 1):
        ''' Make a table of the monopole and quadrupole in k space.
            Using gauss legendre integration.
            With a_perp and a_par, this gives the observed (and not ``true'') multipoles.'''

        # since we are always symmetric in nu, can ignore negative values
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
        
        pknutable = np.zeros((len(nus),nk,self.num_power_components+3))
        
        
        for ii, nu in enumerate(nus_calc):
            # calculate P(k,nu) at the true coordinates, given by
            # k_true = k_apfac * kobs
            # nu_true = nu * a_perp/a_par/fac
            # Note that the integration grid on the other hand is over observed
            fac = np.sqrt(1 + nu**2 * ((a_perp/a_par)**2-1))
            k_apfac = fac / a_perp
            nu_true = nu * a_perp/a_par/fac
            if method == 'Pre-Recon':
                self.make_ptable(f,nu_true,D=D,ks=k_apfac*kv)
                pknutable[ii,:,:-3] = self.pktable[:,1:]
                
            elif method == 'RecSym' or method == 'RecIso':
                self.make_pddtable(f,nu_true,D=D,ks=k_apfac*kv)
                self.make_pdstable(f,nu_true,D=D,ks=k_apfac*kv,method=method)
                self.make_psstable(f,nu_true,D=D,ks=k_apfac*kv,method=method)
                
                pknutable[ii,:,:-3] = self.pktable_dd[:,1:] + self.pktable_ss[:,1:] \
                                        - 2 * self.pktable_ds[:,1:]
                pknutable[ii,:,-3] = self.pktable_dd[:,1]
                pknutable[ii,:,-2] = self.pktable_ds[:,1]
                pknutable[ii,:,-1] = self.pktable_ss[:,1]

        pknutable[ngauss:,:,:] = np.flip(pknutable[0:ngauss],axis=0)

        self.kv = kv

        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None,None]*pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None,None]*pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None,None]*pknutable,axis=0)

        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable

    
    # misc functions
    def export_wisdom(self, wisdom_file='./recon_wisdom.npy'):
        np.save(wisdom_file, pyfftw.export_wisdom())
