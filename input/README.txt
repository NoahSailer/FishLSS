The following files contain the lensing convergence noise curves: 
   nlkk_v3_1_0deproj0_SENS2_fsky0p4_it_lT30-3000_lP30-5000.dat
   nlkk_planck.dat
   S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat
See twoPointNoise.py for how we load the noise curves from these files.

The remaining files are CMB Fisher matrices in the basis
   'h','log(A_s)','n_s','omega_cdm','omega_b','tau_reio','m_ncdm','N_ur','alpha_s','Omega_k' 
Planck_SO combines low-ell Planck (ell < 30) with high-ell SO (ell > 30), and so on.
