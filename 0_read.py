# reads OM10 catalog and generates figures for sanity check
# takes ~10s

# Created by Yoon Chan Taak, 2023 August 11
# github address
# Taak & Treu 2023 
from __future__ import division, print_function
import os, time, math, numpy as np

import matplotlib
import matplotlib.pyplot as plt
import om10

from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=70,Om0=0.3)

font={	'family': 'normal',
		'weight': 'bold',
		'size'	: 30}
matplotlib.rc('font',**font)

from datetime import datetime
t0=datetime.now()


#### Read OM10 catalog

db=om10.DB()

# db.lenses gives the full catalog
# db.lenses[i] gives the parameters for a specific lens
# db.lenses['LENSID'] gives the lensid for all lenses
# len(db.lenses) is 15658

'''
lensid 		id of lens
flagtype 	0 for quasar sources
nimg 		2 or 4
zlens 		redshift of lens
veldisp 	veldisp [km/s]
ellip 		ellipticity of lens
phie 		PA of ellip [deg]
gamma 		shear of lens
phig 		PA of shear [deg]
zsrc 		redshift of source
xsrc,ysrc 	position of source [arcsec]
magi_in 	original (intrinsic) i-band magnitude of source
magi 		magnitude of second/third-brightest lensed image (double/quad)
imsep 		image separation [arcsec]
ximg,yimg 	positions of images (4) [arcsec]
mag 		magnifications of images (4)
delay 		time delays between images (4) [days]
kappa 		?? (all values are zero) (4)
fstar 		?? (all values are zero) (4)
dd 			ang. diam. distance, deflector
ddlum 		luminosity distance, deflector
abmag_i 	absolute magnitude of deflector
apmag_i 	apparent magnitude of deflector
kcorr 		k-correction for deflector: m - M = 5logd - 5 + kcorr
ds 			ang. diam. distance, source
dds 		ang. diam. distance, deflector to source
sigcrit 	?? (all values are zero)
dslum 		luminosity distance, source
l_i 		?? (all values are zero)
reff 		?? (all values are zero)
reff_t 		?? (all values are zero)
'''


zd=db.lenses['ZLENS']
zs=db.lenses['ZSRC']
sig=db.lenses['VELDISP']	# [km/s]
m_s=db.lenses['MAGI_IN']	# [AB mag]
m_im=db.lenses['MAGI']
ds_l=db.lenses['DSLUM']		# [Mpc]
kcorr=db.lenses['KCORR']
imsep=db.lenses['IMSEP']	# [arcsec]
nimg=db.lenses['NIMG']
n4=len(np.where(nimg==4)[0])
print('Quad fraction for full sample: '+format(n4/len(nimg),'6.4f'))
magnif=db.lenses['MAG']

#### Calculate luminosities/magnitudes

m5100=m_s-2.5*(-0.5)*np.log10(7471/5100)
f5100_nu=3631e-26 * 10**(m5100*-0.4) # units of W/m^2/Hz
f5100_lam=f5100_nu*3e8/(5100e-10)**2 # units of W/m^2/m
lamL5100=5100e-10*f5100_lam * 4*np.pi*(ds_l*3.09e22)**2 # units of W
lbol=lamL5100*8.1*1e7 # units of erg/s, BC=8.1 (Runnoe+12)
loglbol=np.log10(lbol)

m_s_ab=m_s-5*np.log10(ds_l*1e6)+5	# absolute magnitude of source, z=0 i-filter
m_i_z2=m_s_ab-2.5*(1-0.5)*np.log10(1+2)	# absolute magnitude of source, z=2 i-filter


#### Draw figures

# Deflector population property histograms
plt.close('all')
fig=plt.figure(1,figsize=(20,10))
ax1=plt.subplot(1,2,1)
ax1.hist(zd,50)
ax1.set_xlabel(r'$z_{\rm d}$')
ax1.set_ylabel('Number of lens systems')
ax1.axis('on')

ax2=plt.subplot(1,2,2)
ax2.hist(sig,50)
ax2.set_xlabel(r'$\sigma_{\rm d}$ [km s$^{-1}$]')
ax2.set_ylabel('Number of lens systems')
ax2.axis('on')

plt.savefig('0_hist_def.pdf')
plt.close('all')

# Source population property histograms
fig=plt.figure(1,figsize=(20,10))
ax1=plt.subplot(1,2,1)
ax1.hist(zs,50)
ax1.set_xlabel(r'$z_{\rm s}$')
ax1.set_ylabel('Number of lens systems')
ax1.axis('on')

ax2=plt.subplot(1,2,2)
ax2.hist(m_s,50)
ax2.set_xlabel(r'$i$-band source magnitude [mag]')
ax2.set_ylabel('Number of lens systems')
ax2.axis('on')

plt.savefig('0_hist_src.pdf')
plt.close('all')

# Source bolometric luminosity histogram
fig=plt.figure(1,figsize=(10,10))
ax=fig.gca()
ax.hist(loglbol,20)
ax.set_xlabel(r'$\log (L_{\rm bol,s}$/erg s$^{-1}$)')
ax.set_ylabel('Number of lens systems')
ax.axis('on')
plt.savefig('0_hist_lsrc.pdf')
plt.close('all')

# Image separation histogram, logarithmic
fig=plt.figure(1,figsize=(10,10))
ax=fig.gca()
ax.hist(imsep,20)
ax.set_xlabel('Image separation [arcsec]')
ax.set_ylabel('Number of lens systems')
ax.axis('on')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('0_hist_imsep_log.pdf')
plt.close('all')


print(datetime.now()-t0)
