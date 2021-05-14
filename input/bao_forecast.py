#
# A Python port of the "bao_forecast.c" code of Seo & Eisenstein (2007)
# available at:
# https://www.cfa.harvard.edu/~deisenst/acousticpeak/bao_forecast.c
#
from __future__ import print_function,division


import numpy as np



def bao_forecast(nbar,sig8,Sperp,Spar,beta,mu_min=0.0):
    """  
    Returns the forecasted precision on the distance scale for a
    survey of volume 1(Gpc/h)^3.
    Inputs:
      nbar - The number density in h^3/Mpc^3
      sig8 - Real-space, linear theory clustering amplitude
      Sperp- Transverse, rms Lagrangian displacement.
      Spar - Transverse, rms Lagrangian displacement.
      beta - RSD distortion parameter
    Output:
      The Fisher matrix: {{Fdd,Fdh},{Fdh,Fhh}}
      where the elements are for D/s and Hs and the comments in
      the original C code say the errors are in percent.
    """
    # Use WMAP3 choice, OmegaM=0.24.
    # WMAP3 power spectrum, normalized to unity at k=0.2.
    Pbao = np.array([14.10, 20.19, 16.17, 11.49, 8.853, 7.641,\
                     6.631, 5.352, 4.146, 3.384, 3.028, 2.799, 2.479,\
                     2.082, 1.749, 1.551, 1.446, 1.349, 1.214, 1.065,\
                     0.9455, 0.8686, 0.8163, 0.7630, 0.6995, 0.6351,\
                     0.5821, 0.5433, 0.5120, 0.4808, 0.4477, 0.4156,\
                     0.3880, 0.3655, 0.3458, 0.3267, 0.3076, 0.2896,\
                     0.2734, 0.2593, 0.2464, 0.2342, 0.2224, 0.2112,\
                     0.2010, 0.1916, 0.1830, 0.1748, 0.1670, 0.1596])
    Nk=len(Pbao) #50
    dk=0.01 ## not changeable

    BAOpower = 2710.
    BAOsilk  = 8.38
    BAOamp   = 0.05169
    Nmu=20
    dmu=(1.-mu_min)/Nmu

    norm     = BAOamp**2/8/np.pi**2*1e9*dk*dmu
    #
    nP   = nbar*sig8**2*BAOpower	# Evaluated at k=0.2h/Mpc.
    # Set up some arrays.  k is in h/Mpc while mu is dimensionless.
    kk,mu = np.meshgrid( (np.arange(Nk)+0.5)*dk, mu_min+(np.arange(Nmu)+0.5)*dmu )
    k2,mu2= kk**2,mu**2
    Pbao  = np.outer(np.ones(Nmu),Pbao)
    Silk  = np.exp(-2*(kk*BAOsilk)**1.4)*k2
    kaiser= (1+beta*mu2)**2	# Called R(mu) in Seo & Eisenstein.
    noise = 1.0/(nP*kaiser)
    S2tot = Sperp**2*(1-mu2)+Spar**2*mu2
    kern  = Silk*np.exp(-k2*S2tot)/(Pbao+noise)**2 
    Fdd   = np.sum(kern * (1-mu2)**2)  * norm
    Fdh   = np.sum(kern * (1-mu2)*mu2) * norm
    Fhh   = np.sum(kern * mu2**2)      * norm
    return( np.array([[Fdd,Fdh],[Fdh,Fhh]]) )
    #



def growth(zz,omm=0.3):
    """    
    An approximation to the growth factor, at z, for a LCDM universe.
    """
    lna = np.linspace(-np.log(1+zz),0.0,1000)
    omz = omm/(omm + (1-omm)*np.exp(3*lna))
    fom = omz**0.545
    Dofz= np.exp( -np.trapz(fom,x=lna) )
    return(Dofz)
    #


def chiOfz(zz,omm=0.3):
    """      
    The comoving, radial distance (in Mpc/h) to redshift zz in a
    LCDM cosmology.
    """
    zp1  = np.linspace(1.0,1.0+zz,1000)
    conH = 2997.925/np.sqrt(omm*zp1**3+(1-omm))
    chi  = np.trapz(conH,x=zp1)
    return(chi)
    #



def volume(zcen,dz,area):
    """     
    The comoving volume, in (Gpc/h)^3, in the redshift range zcen-dz/2
    to zcen+dz/2, for a sky area of "area" sq. deg.
    """
    omega = area * (np.pi/180.)**2	# Steradians.
    cmax  = chiOfz(zcen+dz/2)
    cmin  = chiOfz(zcen-dz/2)
    vol   = omega/3*( cmax**3-cmin**3 )
    return(vol/1e9)
    #



def fisher(zcen,dz,area=1000.,nbar=3e-4,bias=1,s8=0.8,omm=0.3,recon=False, mu_min=0.0):
    """   
    Returns the Fisher matrix for objects in a shell of (total) width
    dz, centered on zcen.
    The values of nbar, bias, s8 are all specified at z=0.
    Also returns an estimate for the percentage error in a model where
    the transverse and radial scales are locked together.
    """
    omz  = omm*(1+zcen)**3/( omm*(1+zcen)**3+(1-omm) )
    fom  = omz**0.545
    D    = growth(zcen,omm)
    sig8 = s8 * D
    Sperp= 12.4*(sig8/0.9)*(0.758) # Mpc/h, Seo & Eisenstein (2007), p.16
    Spar = Sperp * (1+fom)
    if recon=="max":
        Sperp,Spar=0.,0.
    elif recon==True:
        Sperp,Spar = Sperp/2,Spar/2
    bb   = bias/D		   # Const clustering assumption.
    F    = bao_forecast(nbar,bb*sig8,Sperp,Spar,fom/bb,mu_min=mu_min) * volume(zcen,dz,area)
    won  = np.array([1,1])
    err  = 1.0/np.sqrt( np.dot(won,np.dot(F,won)) )
    return( (F,err) )
    #
