# conducts calculations & plots figures in TT23
# takes ~20s for k=1, ~60s for all three coefficient sets
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



# for the three sets of coefficients in Suberlak+21 Table 2
# k=1: S20 SDSS-PS1 combined
# k=2: M10 SDSS
# k=3: S20 SDSS

#for k in range(1,4): 	# use this for loop to calculate for all three coefficient sets
for k in range(1,2):	# use this for loop to calculate just for the best coefficienc set
	
	if k == 1:
		a1,a1err,a2,a2err,a3,a3err,a4,a4err=-0.476,0.008,-0.479,0.005,0.118,0.003,0.118,0.008
		b1,b1err,b2,b2err,b3,b3err,b4,b4err=2.597,0.02,0.17,0.02,0.035,0.007,0.141,0.02
		strk="S20, SDSS-PS1 combined"
	if k == 2:
		a1,a1err,a2,a2err,a3,a3err,a4,a4err=-0.486,0.012,-0.479,0.005,0.119,0.004,0.121,0.012
		b1,b1err,b2,b2err,b3,b3err,b4,b4err=2.5,0.019,0.17,0.02,0.03,0.009,0.178,0.027
		strk="M10, SDSS"
	if k == 3:
		a1,a1err,a2,a2err,a3,a3err,a4,a4err=-0.543,0.009,-0.479,0.005,0.125,0.003,0.104,0.008
		b1,b1err,b2,b2err,b3,b3err,b4,b4err=2.515,0.019,0.17,0.02,0.042,0.007,0.127,0.019
		strk="S20, SDSS"


	# calculate SF_inf & tau for each source quasar (using Eq. 2)
	print("k="+format(k)+": "+strk)
	x2=np.log10(7690/4000) 	# central wavelength for i-band
	x3=m_i_z2+23			# i-filter absolute magnitude (z=2)
	x4=(m_i_z2*-0.25+2.5)-9	# black hole mass; from Taak&Im 20
	x4err=0.3

	logsfinf = a1 + a2*x2 + a3*x3 + a4*x4
	logsfinferr = np.sqrt(a1err**2 + (a2err*x2)**2 + (a3err*x3)**2 + (a4*x4err)**2 + (a4err*x4)**2)
	logtau= b1 + b2*x2 + b3*x3 + b4*x4
	logtauerr = np.sqrt(b1err**2 + (b2err*x2)**2 + (b3err*x3)**2 + (b4*x4err)**2 + (b4err*x4)**2)
	sfinf=10**logsfinf			# 0.02-0.84 mag; SF_inf	(for x4=0) --> 0.03-0.50 mag (mbh=2.5-0.25*m_i_z2)
	tau=10**logtau			# 209-637 days			(for x4=0) --> 345-348 days (mbh=2.5-0.25*m_i_z2)
	dt=20	# expected time candence for LSST observations: 20 days (each of rizy filters)

	sf=sfinf*(1-np.exp(-dt/tau))**0.5	# 0.006-0.148 mag; actual stdev of observed magnitudes (SF)
												#	(for x4=0) --> 0.007-0.120 mag (mbh=2.5-0.25*m_i_z2)


	# calculate fraction of sources with detectable variability & quad fraction
	ct,ct4=0,0
	id_vardm=[]
	for i in range(len(zs)):
		xx=10**(0.4*(m_im[i]-23.92))
		yy=np.sqrt((0.04-0.039)*xx+0.039*xx**2)
		if yy < sf[i]:
			id_vardm.append(i)
			ct=ct+1
			if nimg[i]==4:
				ct4=ct4+1

	print('fraction of sources with SF > dm: '+format(ct/len(zs),'5.3f'))
	print('Quad fraction for SF > dm sample: '+format(ct4/ct,'6.4f'))

	### Figure 1: mag vs SF_inf, mag vs SF (Delta t), compared with LSST magnitude errors
	fig=plt.figure(1,figsize=(20,10))
	ax1=plt.subplot(1,2,1)
	ax1.scatter(m_im,sfinf,s=1)
	# simulated photometric errors from Ivezic+19
	x=np.arange(np.min(m_im),np.max(m_im),0.01)
	x2=10**(0.4*(x-23.92))
	ysq=(0.04-0.039)*x2+0.039*x2**2
	y=np.sqrt(ysq)
	ax1.plot(x,y,color='r')
	
	ax1.set_xlabel('2nd/3rd image magnitude (double/quad) [mag]')
	ax1.set_ylabel(r'SF$_{\infty}$ [mag]')
	ax1.axis('on')
	ax1.plot([17,17.5],[0.4,0.4],color='r')
	ax1.text(17.7,0.393,'LSST predicted error',fontsize=20)
	ax1.text(-.2,.95,'(a)',transform=ax1.transAxes)

	ax2=plt.subplot(1,2,2)	
	ax2.scatter(m_im,sf,s=1)
	ax2.plot(x,y,color='r')
	
	ax2.set_xlabel('2nd/3rd image magnitude (double/quad) [mag]')
	ax2.set_ylabel(r'SF ($\Delta t$ = 20 d) [mag]')
	ax2.axis('on')
	ax2.text(16.5,0.095,'SF $> \Delta$m: '+format(ct/len(zs)*100,'5.2f')+'$\%$',fontsize=20)
	ax2.text(-.2,.95,'(b)',transform=ax2.transAxes)
	
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('fig1-'+format(k)+'.png',dpi=300)
	os.system("convert fig1-"+format(k)+".png fig1-"+format(k)+".pdf")	# file size is similar, but loads faster
	os.system("rm fig1-"+format(k)+".png")
	plt.close('all')


	'''
	# plot tau vs SF_inf
	fig=plt.figure(1,figsize=(10,10))
	ax=fig.gca()
	ax.errorbar(tau,sfinf,yerr=np.log(10)*sf*logsfinferr,fmt='o')
	ax.set_xlabel(r'$\tau$ [days]')
	ax.set_ylabel(r'SF$_{\infty}$ [mag]')
	ax.axis('on')
	plt.savefig('tau-sfinf-'+format(k)+'.pdf')
	plt.close('all')

	fig=plt.figure(1,figsize=(10,10))
	ax=fig.gca()
	ax.errorbar(logtau,logsfinf,yerr=logsfinferr,fmt='o')
	ax.set_xlabel(r'log ($\tau$/days)')
	ax.set_ylabel(r'log (SF$_{\infty}$/mag)')
	ax.axis('on')
	plt.savefig('tau-sfinf-log-'+format(k)+'.pdf')
	plt.close('all')
	'''

	### Figure 2: zs vs Hb time lag
	loglag_rest=1.527+0.533*np.log10(lamL5100/1e37)	# Bentz+13 eq 2; 5.1-4500 days (restframe)
	lag=10**loglag_rest*(1+zs)			# observed lag; 6.8-20000 days

	fig=plt.figure(1,figsize=(10,10))
	ax=fig.gca()
	ax.scatter(zs,lag,s=1)
	ax.set_xlabel(r'$z_{\rm s}$')
	ax.set_ylabel(r'observed H$\beta$ time lag [days]')
	ax.set_yscale('log')
	ax.axis('on')
	plt.savefig('fig2-'+format(k)+'.png',dpi=300)
	os.system("convert fig2-"+format(k)+".png fig2-"+format(k)+".pdf")	# file size is similar, but loads faster
	os.system("rm fig2-"+format(k)+".png")
	plt.close('all')


	# brightest image magnitude
	m_im1=m_im*0
	for i in range(len(m_im1)):
		m_im1[i]=m_s[i]-2.5*np.log10(np.max(abs(magnif[i])))


	### Figure 3: scatter plots of all parameters
	paramlist=['sfinf','lag','m_im1','m_im','imsep']
	paramname=[r'SF$_\infty$ [mag]',r'observed H$\beta$ time lag [days]',r'1st img $i$-band magnitude [mag]',r'2nd/3rd $i$-band magnitude [mag]','image separation [arcsec]']
	paramscale=['linear','log','linear','linear','linear']
	npar=len(paramlist)
	
	idx1=np.where((sfinf>0.15) & (m_im1<21))	# blue
	idx2=np.where((sfinf>0.20) & (m_im1<20))	# red
	idx3=np.where((sfinf>0.20) & (m_im1<21))
	idx4=np.where((sfinf>0.15) & (m_im1<20))
	idx1_4=np.where((sfinf>0.15) & (m_im1<21) & (nimg==4))	# identical to above, but for quad fractions
	idx2_4=np.where((sfinf>0.20) & (m_im1<20) & (nimg==4))
	idx3_4=np.where((sfinf>0.20) & (m_im1<21) & (nimg==4))
	idx4_4=np.where((sfinf>0.15) & (m_im1<20) & (nimg==4))

	fig,axs=plt.subplots(npar,npar,figsize=(30,30))
	for i in range(npar):
		for j in range(npar):
			if i>j:
				exec('x1='+paramlist[i])
				exec('x2='+paramlist[j])
				axs[i,j].scatter(x2,x1,s=0.5,color='grey')
				axs[i,j].scatter(x2[idx1],x1[idx1],s=1,color='dodgerblue')
				axs[i,j].scatter(x2[idx2],x1[idx2],s=20,color='red',marker='X',linewidth=0.5)
				axs[i,j].set_xlabel(paramname[j])
				axs[i,j].set_ylabel(paramname[i])
				axs[i,j].set_xscale(paramscale[j])
				axs[i,j].set_yscale(paramscale[i])
				if i!=npar-1:
					axs[i,j].set_xlabel('')
				if j!=0:
					axs[i,j].set_ylabel('')
			if i<j:
				axs[i,j].set_axis_off()


	for i in range(npar):
		exec('x1='+paramlist[i])
		if paramscale[i]=='log':
			(counts,bins)=np.histogram(x1,bins=np.logspace(np.log10(min(x1)),np.log10(max(x1)),10))
			(counts1,bins1)=np.histogram(x1[idx1],bins=np.logspace(np.log10(min(x1)),np.log10(max(x1)),10))
			(counts2,bins2)=np.histogram(x1[idx2],bins=np.logspace(np.log10(min(x1)),np.log10(max(x1)),10))
		else:
			(counts,bins)=np.histogram(x1,10)
			(counts1,bins1)=np.histogram(x1[idx1],bins=bins)
			(counts2,bins2)=np.histogram(x1[idx2],bins=bins)
		axs[i,i].hist(bins[:-1],bins,weights=counts*0.2,color='grey')
		axs[i,i].hist(bins1[:-1],bins1,weights=counts1*0.2,color='dodgerblue')
		axs[i,i].hist(bins2[:-1],bins2,weights=counts2*0.2,color='red')
		axs[i,i].set_yscale('log')
		axs[i,i].set_xscale(paramscale[i])
		axs[i,i].set_ylabel('Number of lenses')
		if i==npar-1:
			axs[i,i].set_xlabel(paramname[i])

	plt.tight_layout()
	plt.text(0.45,0.85,r"\# of lenses per 20,000 deg$^2$ (LSST coverage)",fontsize=50, transform=fig.transFigure)
	plt.text(0.45,0.81,strk+" light curve", fontsize=50, transform=fig.transFigure)

	plt.scatter(0.48,0.785,s=100,color='grey',transform=fig.transFigure,clip_on=False)
	plt.scatter(0.48,0.755,s=100,color='dodgerblue',transform=fig.transFigure,clip_on=False)
	plt.scatter(0.48,0.725,s=100,color='red',marker='X',linewidth=0.1,transform=fig.transFigure,clip_on=False)

	plt.text(0.5,0.78,'All lensed QSOs', fontsize=50, transform=fig.transFigure, family='serif')
	plt.text(0.5,0.75,r'SF$_\infty > 0.15, i_1 < 21$', fontsize=50, transform=fig.transFigure, family='Arial')
	plt.text(0.5,0.72,r'SF$_\infty > 0.20, i_1 < 20$', fontsize=50, transform=fig.transFigure, family='Arial')

	plt.text(0.8,0.78,format(len(sfinf)/5,'7.1f'), fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')
	plt.text(0.8,0.75,format(len(idx1[0])/5,'7.1f'), fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')
	plt.text(0.8,0.72,format(len(idx2[0])/5,'7.1f'), fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')

	plt.text(0.9,0.81,r'f$_{\rm quad}$', fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')
	plt.text(0.9,0.78,format(n4/len(sfinf),'5.3f'),fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')
	plt.text(0.9,0.75,format(len(idx1_4[0])/len(idx1[0]),'5.3f'),fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')
	plt.text(0.9,0.72,format(len(idx2_4[0])/len(idx2[0]),'5.3f'),fontsize=50, transform=fig.transFigure, horizontalalignment='right', family='Arial')

	print('Quad fraction for SFinf>0.20 & i_1<21: '+format(len(idx3_4[0])/len(idx3[0]),'5.3f'))
	print('Quad fraction for SFinf>0.15 & i_1<20: '+format(len(idx4_4[0])/len(idx4[0]),'5.3f'))

	plt.savefig('fig3-'+format(k)+'-log.png',dpi=300)
	os.system("convert fig3-"+format(k)+"-log.png fig3-"+format(k)+"-log.pdf")	# file size is similar, but loads faster
	os.system("rm fig3-"+format(k)+"-log.png")
	plt.close('all')

	### Figure 4: physical parameter histogram distribution plots (only for k=1)
	if k == 1:
		fig,axs=plt.subplots(2,2,figsize=(20,20))
		# plot zd, zs, sig, m_s
		paramlist=[['zd','sig'],['zs','m_s']]
		for i in range(2):
			for j in range(2):
				exec('x1='+paramlist[i][j])
				(counts0,bins0)=np.histogram(x1,20)
				(counts1,bins1)=np.histogram(x1[idx1],bins=bins0)
				(counts2,bins2)=np.histogram(x1[idx2],bins=bins0)
				axs[i,j].hist(bins0[:-1],bins0,weights=counts0*0.2,color='grey')
				axs[i,j].hist(bins1[:-1],bins1,weights=counts1*0.2,color='dodgerblue')
				axs[i,j].hist(bins2[:-1],bins2,weights=counts2*0.2,color='red')
				axs[i,j].set_yscale('log')
				axs[i,j].set_ylabel('Number of lensed QSOs')


		axs[0,0].set_xlabel(r'$z_{\rm d}$')
		axs[0,1].set_xlabel(r'$\sigma_{\rm d}$ [km s$^{-1}$]')
		axs[1,0].set_xlabel(r'$z_{\rm s}$')
		axs[1,1].set_xlabel(r'$i$-band source magnitude [mag]')

		plt.savefig('fig4.pdf')
		plt.close('all')

	### Figure 5: reference magnitude histogram (for depth-area comparison) (only for k=1)
	if k == 1:
		fig=plt.figure(1,figsize=(10,10))
		ax=fig.gca()

		hist0_fit=np.histogram(m_im,70)
		hist0=np.histogram(m_im,20)
		hist1=np.histogram(m_im[idx1],bins=hist0[1])
		hist2=np.histogram(m_im[idx2],bins=hist0[1])
		hist3=np.histogram(m_im[id_vardm],bins=hist0[1])
		ax.hist(hist0[1][:-1],hist0[1],weights=hist0[0]*0.2,color='lightgray',fill=False)
		ax.hist(hist3[1][:-1],hist3[1],weights=hist3[0]*0.2,color='darkgray')
		ax.hist(hist1[1][:-1],hist1[1],weights=hist1[0]*0.2,color='dodgerblue')
		ax.hist(hist2[1][:-1],hist2[1],weights=hist2[0]*0.2,color='red')



		ax.plot([17,23],3.5*.4*np.array([10**(0*0.8),10**(6*0.8)]),color='darkblue') # 3.5 factor is for scaling difference between hist0_fit and hist0 due to binsize
		id_fit=min(np.where(hist0_fit[1] >21)[0])	# fit slope at refmag=21 mag
		slope2=(np.log10(hist0_fit[0][id_fit+3])-np.log10(hist0_fit[0][id_fit-3]))/(hist0_fit[1][id_fit+3]-hist0_fit[1][id_fit-3])
		ax.plot([17,23],[3.5*10**((17-hist0_fit[1][id_fit])*slope2)*hist0_fit[0][id_fit]*0.2,3.5*10**((23-hist0_fit[1][id_fit])*slope2)*hist0_fit[0][id_fit]*0.2],color='coral')
		ax.set_xlabel(r'2nd/3rd image $i$-band magnitude [mag]')
		ax.set_ylabel('Number of lens systems')
		ax.axis('on')
		ax.set_yscale('log')
		ax.set_xlim(17,23)
		ax.set_ylim(.1,1e3)
		ax.text(19.2,0.5e3,'slope=0.8',color='darkblue',fontsize=20,weight='bold')
		ax.text(17.2,2.e1,'slope='+format(slope2,'4.2f'),color='coral',fontsize=20,weight='bold')
		plt.tight_layout()
		plt.savefig('fig5.pdf')
		plt.close('all')

	if k==1:
		# Number of lensed QSOs for various depths
		depth=[20.0,21.0,22.0,23.0,23.3]
		for i in range(len(depth)):
			iddepth=np.where(m_im < depth[i])[0]
			print('Depth = '+format(depth[i],'4.1f')+': '+format(len(iddepth)/5,'6.1f')+' lensed QSOs in 20,000 deg2')

		# Number of lensed QSOs for other surveys
		surveyname=['PS1/3pi','LSST','Euclid/Deep']
		surveydepth=[21.5,24.0,24.5]-2.5*np.log10(2) # depths shown are 5sigma, need to offset to 10sigma
		surveycoverage=[3e4,2e4,40]
		surveypsf=[1.1,0.75,0.23]
		for i in range(len(surveyname)):
			idsur=np.where((m_im < surveydepth[i]) & (imsep > surveypsf[i]*2/3))
			print('# of lensed QSOs in '+surveyname[i]+': '+format(len(idsur[0])*surveycoverage[i]/1e5,'7.2f'))
			idsur2=np.where((m_im < surveydepth[i]) & (imsep > surveypsf[i]*2/3) & (sfinf > 0.2) & (m_im1 < 20))
			print('# of bright lensed QSOs w/ large variability in '+surveyname[i]+': '+format(len(idsur2[0])*surveycoverage[i]/1e5,'12.2e'))


print(datetime.now()-t0)
