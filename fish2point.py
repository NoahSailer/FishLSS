import numpy as np
import matplotlib.pyplot as plt


p = 4. # get the power spectrum from velocileptors
nGal = # get the number density of galaxies, assume to be a constant
vSurvey = # volume of the survey
vEff = lambda k,mu: vSurvey * ((nGal * p(k,mu)) / (nGal * p(k,mu) + 1.))**2.

def dPdp(parms*,):
   ''' returns the derivative of the matter power
   spectrum with respect to the input parameters
   '''
   return

def Fisher():
   return


