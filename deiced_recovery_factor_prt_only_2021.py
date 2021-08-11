import numpy as np
from netCDF4 import Dataset
import glob
import matplotlib.pyplot as plt
import math
import glob
import scipy
from scipy import signal
from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import sys
from scipy import stats
from scipy.odr import *
from numpy import genfromtxt

experiment_name='prtonly_since201712_test1'

#this experiment uses the following flights (contents of sensors_by_flight.txt):
#Flight name,De-iced serial number, De-iced type of sensor, Non-de-iced serial number, Non-de-iced type of sensor 
#c072,19868E,plate,15480,plate
#c073,19868E,plate,15480,plate
#c074,19868E,plate,15480,plate
#c075,19868E,plate,15480,plate
#c076,19868E,plate,15480,plate
#c077,19868E,plate,15480,plate
#c078,19868E,plate,15480,plate
#c079,19868E,plate,15480,plate
#c080,19868E,plate,15480,plate
#c081,19868E,plate,15480,plate
#c082,19868E,plate,15480,plate
#c083,19868E,plate,15480,plate
#c084,19868E,plate,15480,plate
#c085,19868E,plate,15480,plate
#c087,19868E,plate,15480,plate
#c088,19868E,plate,2614,loom
#c089,19868E,plate,15480,plate
#c090,19868E,plate,15480,plate
#c091,19868E,plate,19207E,loom
#c092,19868E,plate,15480,plate
#c093,19868E,plate,19207E,loom
#c094,19868E,plate,15480,plate
#c095,19868E,plate,17004E,plate
#c096,19868E,plate,17004E,plate
#c097,19868E,plate,17004E,plate
#c098,19868E,plate,17004E,plate
#c099,19868E,plate,17004E,plate
#c100,19868E,plate,17004E,plate
#c101,19868E,plate,17004E,plate
#c102,19868E,plate,17004E,plate
#c103,19868E,plate,17004E,plate
#c104,19868E,plate,17004E,plate
#c105,19868E,plate,17004E,plate
#c106,19868E,plate,17004E,plate
#c109,19868E,plate,17004E,plate
#c110,19868E,plate,17004E,plate
#c111,19868E,plate,17004E,plate
#c122,20473E,plate,17004E,plate
#c123,20473E,plate,17004E,plate
#c124,20473E,plate,17004E,plate
#c125,20473E,plate,17004E,plate
#c126,20473E,plate,17004E,plate
#c127,20473E,plate,17004E,plate
#c169,20472E,plate,19206E,plate
#c170,20472E,plate,19206E,plate
#c171,20472E,plate,19206E,plate
#c172,20472E,plate,19206E,plate
#c173,20472E,plate,19206E,plate
#c174,20472E,plate,19206E,plate

#the maximum allowable difference between the originally reported TAT_DI_R and TAT_ND_R for the data to be considered good
max_difference_tatdi_tatnd=1.0
#maximum difference between temperature and dewpoint to be safely out of cloud
dewpoint_diff=2.0
#maximum/minimum allowable roll/aoa/aoss for the data to be considered good
max_roll=0.5
max_aoss=1.0
max_aoa=8.0
min_aoa=3.0

#what's the name of the first filename. this data will be ignored, it just helps the code to run easily (this file should be sitting in the same directory as where you run this code)
first_name='core_faam_RECOVERY_FACTOR_20160218_v004_r999_b945_1hz.nc'
flight_name=first_name[-11:-7]
#open up a first dataset 
start_fh = Dataset(first_name,'r')
wow_m = start_fh.variables['WOW_IND'][:]
lat_m = start_fh.variables['LAT_GIN'][:]
PALT_RVS_m = start_fh.variables['PALT_RVS'][:]   
TDEW_GE_m = start_fh.variables['TDEW_GE'][:]
PS_RVSM_m = start_fh.variables['PS_RVSM'][:]
Q_RVSM_m = start_fh.variables['Q_RVSM'][:]
ROLL_GIN_m = start_fh.variables['ROLL_GIN'][:]
AOA_m = start_fh.variables['AOA'][:]
AOSS_m = start_fh.variables['AOSS'][:]
ITDI_m = start_fh.variables['ITDI'][:]
ITND_m = start_fh.variables['NDTI'][:]
TAT_DI_R_m = start_fh.variables['TAT_DI_R'][:]
TAT_ND_R_m = start_fh.variables['TAT_ND_R'][:]

#open up a file that contains information about which sensor was fitted for each flight. 1=loom, 2=plate, 3=thermistor
type_data = genfromtxt('sensors_by_flight.txt', delimiter=',', dtype=str, skip_header=1)
type_flight_name=type_data[:,0]
di_sn=type_data[:,1]
di_type=type_data[:,2]
ndi_sn=type_data[:,3]
ndi_type=type_data[:,4]
 
di_type_code=np.zeros(len(ITDI_m))
ndi_type_code=np.zeros(len(ITDI_m))
di_sn_arr=np.zeros(len(ITDI_m))
ndi_sn_arr=np.zeros(len(ITDI_m))
fn_arr=np.zeros(len(ITDI_m))

#new circuit installed second half of 2017. DECADES calibration done 7th December 2017, want to use all PRT temperature data after this point
fnames = sorted(glob.glob('prtonly_sincenewcircuit/*.nc'))

#erase the data in the starter file (this isn't relevant, it's just to give something to concatenate on to
ITDI_m[:] = np.nan
ITND_m[:] = np.nan
PS_RVSM_m[:] = np.nan
Q_RVSM_m[:] = np.nan
di_type_code[:] = np.nan
ndi_type_code[:] = np.nan
wow_m[:] = np.nan
lat_m[:] = np.nan
PALT_RVS_m[:] = np.nan
TDEW_GE_m[:] = np.nan
ROLL_GIN_m[:] = np.nan
AOA_m[:] = np.nan
AOSS_m[:] = np.nan
TAT_DI_R_m[:] = np.nan
TAT_ND_R_m[:] = np.nan
di_sn_arr[:] = np.nan
ndi_sn_arr[:] = np.nan
fn_arr[:] = np.nan

counter=0

#for all the files found, open them up and concatenate the data onto the end of the existing data
for f in fnames:
    fh = Dataset(f,'r')
    flight_number=f[-10:-7]
    flight_name=f[-11:-7]
    this_flight_ditype=di_type[np.where(type_flight_name==flight_name)]
    this_flight_nditype=ndi_type[np.where(type_flight_name==flight_name)]
    this_flight_disn=di_sn[np.where(type_flight_name==flight_name)]
    this_flight_ndisn=ndi_sn[np.where(type_flight_name==flight_name)]
    
    if (this_flight_ditype=='loom'):
        di_type_code_thisflight=1
    if (this_flight_ditype=='plate'):
        di_type_code_thisflight=2    
    if (this_flight_ditype=='thermistor'):
        di_type_code_thisflight=3    
    if (this_flight_nditype=='loom'):
        ndi_type_code_thisflight=1
    if (this_flight_nditype=='plate'):
        ndi_type_code_thisflight=2    
    if (this_flight_nditype=='thermistor'):
        ndi_type_code_thisflight=3        
    
    print(flight_name)
    print('di = '+str(di_type_code_thisflight))
    print('ndi = '+str(ndi_type_code_thisflight))    

    lat_m = np.concatenate((lat_m,fh.variables['LAT_GIN'][:]))
    PALT_RVS_m = np.concatenate((PALT_RVS_m,fh.variables['PALT_RVS'][:]))   
    TAT_DI_R_m = np.concatenate((TAT_DI_R_m,fh.variables['TAT_DI_R'][:]))
    TAT_ND_R_m = np.concatenate((TAT_ND_R_m,fh.variables['TAT_ND_R'][:]))
    TDEW_GE_m = np.concatenate((TDEW_GE_m,fh.variables['TDEW_GE'][:]))
    PS_RVSM_m = np.concatenate((PS_RVSM_m,fh.variables['PS_RVSM'][:]))
    Q_RVSM_m = np.concatenate((Q_RVSM_m,fh.variables['Q_RVSM'][:]))
    ITDI_m = np.concatenate((ITDI_m,fh.variables['ITDI'][:]))
    ITND_m = np.concatenate((ITND_m,fh.variables['NDTI'][:]))
    ROLL_GIN_m = np.concatenate((ROLL_GIN_m,fh.variables['ROLL_GIN'][:]))
    AOA_m = np.concatenate((AOA_m,fh.variables['AOA'][:]))
    AOSS_m = np.concatenate((AOSS_m,fh.variables['AOSS'][:]))
    
    di_type_code_thisflight_arr=np.full(len(fh.variables['TDEW_GE'][:]), di_type_code_thisflight)     
    di_type_code = np.concatenate((di_type_code,di_type_code_thisflight_arr))  
    ndi_type_code_thisflight_arr=np.full(len(fh.variables['TDEW_GE'][:]), ndi_type_code_thisflight)     
    ndi_type_code = np.concatenate((ndi_type_code,ndi_type_code_thisflight_arr))        
    
    di_sn_thisflight_arr=np.full(len(fh.variables['TDEW_GE'][:]), this_flight_disn)     
    di_sn_arr = np.concatenate((di_sn_arr,di_sn_thisflight_arr))  
    ndi_sn_thisflight_arr=np.full(len(fh.variables['TDEW_GE'][:]), this_flight_ndisn)     
    ndi_sn_arr = np.concatenate((ndi_sn_arr,ndi_sn_thisflight_arr))   
    flight_number_thisflight_arr=np.full(len(fh.variables['TDEW_GE'][:]), flight_number)     
    fn_arr = np.concatenate((fn_arr,flight_number_thisflight_arr))         

#save copies of the data before doing anything with it
ITDI_full=ITDI_m.copy()
ITND_full=ITND_m.copy()
TAT_DI_R_full=TAT_DI_R_m.copy()
TAT_ND_R_full=TAT_ND_R_m.copy()

#get rid of masked data   
PALT_RVS = PALT_RVS_m.data  
TAT_DI_R = TAT_DI_R_m.data
TAT_ND_R = TAT_ND_R_m.data
TDEW_GE = TDEW_GE_m.data
PS_RVSM = PS_RVSM_m.data
Q_RVSM = Q_RVSM_m.data
ITDI = ITDI_m.data
ITND = ITND_m.data
ROLL_GIN = ROLL_GIN_m.data
AOA = AOA_m.data
AOSS = AOSS_m.data
PALT_RVS[PALT_RVS_m.mask] = np.nan
TAT_DI_R[TAT_DI_R_m.mask] = np.nan
TAT_ND_R[TAT_ND_R_m.mask] = np.nan
TDEW_GE[TDEW_GE_m.mask] = np.nan
PS_RVSM[PS_RVSM_m.mask] = np.nan
Q_RVSM[Q_RVSM_m.mask] = np.nan
ITDI[ITDI_m.mask] = np.nan
ITND[ITND_m.mask] = np.nan
ROLL_GIN[ROLL_GIN_m.mask] = np.nan
AOA[AOA_m.mask] = np.nan
AOSS[AOSS_m.mask] = np.nan

#make some arrays to fill
TempProfile=np.array(TAT_DI_R)
ProfileFlag=np.array(TAT_DI_R)
temp_minus_tdew=np.array(TAT_DI_R)
mach=np.array(TAT_DI_R)

#calculate mach number from static and dynamic rvsm pressures
mach=np.sqrt(5*((1+Q_RVSM/PS_RVSM)**(2/7)-1))  

#go through all the data. make a first guess as to whether the aircraft is profiling or level
for i in range (0,(len(PALT_RVS)-1)):
    ProfileFlag[-1]=-17
    if PALT_RVS[i]<=(PALT_RVS[(i+1)]-2):
        TempProfile[i]=1
    elif PALT_RVS[i]>=(PALT_RVS[(i+1)]+2):
        TempProfile[i]=2
    else:
        TempProfile[i]=0 
#go through all possible profile data that was just found and look at each half second to see if the profile was consistent. if it was, give the data a flag        
for i in range (0,(len(PALT_RVS)-1),1):
    if TempProfile[i]==1 and TempProfile[i+1]==1 and TempProfile[i+2]==1:
        ProfileFlag[i]=1
    elif TempProfile[i]==2 and TempProfile[i+1]==2 and TempProfile[i+2]==2:
        ProfileFlag[i]=2
    else:
        ProfileFlag[i]=0

#look though all the data 
for i in range (0,(len(PALT_RVS)-1)):
    #if the temperature was within dewpoint_diff K of the dewpoint (chance of cloud), give the data a flag
    if (TAT_DI_R[i]-TDEW_GE[i])<dewpoint_diff:
        ProfileFlag[i]=ProfileFlag[i]-5
    #if the aircraft was on the ground, give the data a flag
    if np.mean(Q_RVSM[i])>=50.0:
        ProfileFlag[i]=ProfileFlag[i]
    else:
        ProfileFlag[i]=-10
    #if the magnitude of the roll angle was greater than max_roll degree, give the data a flag
    if ((ROLL_GIN[i])**2.0)**0.5>=max_roll:
        ProfileFlag[i]=-37   
    #if the magnitude of the angle of side slip was greater than max_aoss, give the data a flag
    if ((AOSS[i])**2.0)**0.5>=max_aoss:
        ProfileFlag[i]=-37
    #if the angle of attack was greater than max_aoa degrees, give the data a flag
    if AOA[i]>=max_aoa:
        ProfileFlag[i]=-43
    #if the angle of attack was less than min_aoa degrees, give the data a flag
    if AOA[i]<=min_aoa:
        ProfileFlag[i]=-44
    #if the magnitude of the difference between the reported true air temperature for the de-iced and non-de-iced probes was greater than max_difference_tatdi_tatnd K, give the data a flag 
    if ((TAT_DI_R[i]-TAT_ND_R[i])**2)**0.5>max_difference_tatdi_tatnd:
        ProfileFlag[i]=-27    
    #if the indicated temperature was less than 150 K or more than 350 K, or if the mach number was greater than 0.8, give the data a flag
    if (ITDI[i] >= 350.0) | (ITND[i] >= 350.0) | (ITDI[i] <= 150.0) | (ITND[i] <= 150.0) | (mach[i] >= 0.8) | (TAT_ND_R[i] <= 150.0) :
        ProfileFlag[i]=-17
    #change all datapoints which have a non-zero flag to be nans
    if ProfileFlag[i]==0:
        ITDI[i]=ITDI[i]
        ITND[i]=ITND[i]
        mach[i]=mach[i]
        TAT_DI_R[i]=TAT_DI_R[i]
        TAT_ND_R[i]=TAT_ND_R[i]
       # gamma[i]=gamma[i]
    else:
        ITDI[i]=np.nan
        ITND[i]=np.nan
        mach[i]=np.nan
        TAT_DI_R[i]=np.nan
        TAT_ND_R[i]=np.nan
        #gamma[i]=np.nan

for i in range (0,(len(PALT_RVS)-1)):
    if (np.isnan(ITDI[i]) == True) | (np.isnan(ITND[i]) == True) | (np.isnan(mach[i]) == True) | (np.isnan(TAT_DI_R[i]) == True) | (np.isnan(TAT_ND_R[i]) == True) :
        ITDI[i]=np.nan
        ITND[i]=np.nan
        mach[i]=np.nan
        TAT_DI_R[i]=np.nan
        TAT_ND_R[i]=np.nan

#doing this because of a mix up in the recovery factor in the processing for a few flights
TAT_DI_R=ITDI/(1.0+0.2*0.9928*mach**2.0)
TAT_ND_R=ITND/(1.0+0.2*0.999*mach**2.0)

#make the last datapoint in each array a nan, because the loop might not pick it up
ITDI[-1]=np.nan
ITND[-1]=np.nan
mach[-1]=np.nan

#from rosemount 5755 ##############################################
eta_ndi=-6.0943146E-04*mach**2.0 + 1.4054157E-03*mach
sigma_eta_ndi=-3.4581190E-04*mach**2 + 5.9345748E-04*mach
###################################################################

eta_di_meas=1.0-ITDI/ITND*(1.0-eta_ndi)
#estimate of NPL temperature to resistance cal uncertainty ########
sigma_itnd_cal=7.93E-06*ITND**2.0-4.65E-03*ITND+7.00E-01
sigma_itdi_cal=7.93E-06*ITDI**2.0-4.65E-03*ITDI+7.00E-01
###################################################################


ratio=ITDI/ITND
#uncertainty in the ratio of the indicated temperatures
u_ratio=((1/ITND)**2.0*sigma_itdi_cal**2.0+(ITDI/(ITND**2.0))**2.0*sigma_itnd_cal**2.0)**0.5
mean_ratio=np.mean(ratio[~np.isnan(mach)])
stdev_ratio=np.std(ratio[~np.isnan(mach)])
llim=str(mean_ratio-stdev_ratio)
ulim=str(mean_ratio+stdev_ratio)

print('ratio is between '+llim+' and '+ulim)
print('mean ratio is '+str(mean_ratio))
print('ratio stdev is '+str(stdev_ratio))


eta_di_calc=1-mean_ratio*(1.0-eta_ndi)
#uncertainty in the calculated eta di
u_eta_di_calc=((eta_ndi-1.0)**2.0*stdev_ratio**2.0+mean_ratio**2.0*sigma_eta_ndi**2.0)**0.5

#just for doing some plots
mach_invented=np.arange(0.0,0.9,0.001)
eta_ndi_toplot=-6.0943146E-04*mach_invented**2.0 + 1.4054157E-03*mach_invented
sigma_eta_ndi_toplot=-3.4581190E-04*mach_invented**2 + 5.9345748E-04*mach_invented
eta_di_calc_toplot=1-mean_ratio*(1.0-eta_ndi_toplot)
u_eta_di_calc_toplot=((eta_ndi_toplot-1.0)**2.0*stdev_ratio**2.0+   mean_ratio**2.0*sigma_eta_ndi_toplot**2.0)**0.5

#work out what r corresponds to the etas we've worked out
r_nd=1.0-eta_ndi_toplot*(1.0+(2.0/(0.4*mach_invented**2.0)))
r_di=1.0-eta_di_calc_toplot*(1.0+(2.0/(0.4*mach_invented**2.0)))

#calculate the new TATs
tat_nd_new=ITND/((1-eta_ndi)*(1+0.2*mach**2.0))
tat_di_new=ITDI/((1-eta_di_calc)*(1+0.2*mach**2.0))

#working out the mean differences between the TATs before, and with new etas
diff_old=np.mean(TAT_DI_R[~np.isnan(mach)]-TAT_ND_R[~np.isnan(mach)])
diff_new=np.mean(tat_di_new[~np.isnan(mach)]-tat_nd_new[~np.isnan(mach)])
stdev_diff_old=np.std(TAT_DI_R[~np.isnan(mach)]-TAT_ND_R[~np.isnan(mach)])
stdev_diff_new=np.std(tat_di_new[~np.isnan(mach)]-tat_nd_new[~np.isnan(mach)])
print(diff_old)
print(diff_new)
print(stdev_diff_old)
print(stdev_diff_new)

#make a histogram of the differences between TATs
plt.plot([0.99,1.01],[0,0],'w.',label='Mean diff old = '+str(np.around(diff_old,decimals=3)),alpha=0)
plt.plot([0.99,1.01],[0,0],'w.',label='Mean diff new = '+str(np.around(diff_new,decimals=3)),alpha=0)
plt.plot([0.99,1.01],[0,0],'w.',label='Mean ratio = '+str(np.around(mean_ratio,decimals=5)),alpha=0)
plt.plot([0.99,1.01],[0,0],'w.',label='Stdev ratio = '+str(np.around(stdev_ratio,decimals=5)),alpha=0)
plt.hist(TAT_DI_R[~np.isnan(mach)]-TAT_ND_R[~np.isnan(mach)], 50, normed=True,range=(-1,1),alpha=0.2,color='grey')
plt.hist(tat_di_new[~np.isnan(mach)]-tat_nd_new[~np.isnan(mach)], 50, normed=True,range=(-1,1),alpha=0.2,color='green')
plt.xlabel('De-iced - non-de-iced true air temperature (K)')
plt.ylabel('Density')
plt.title('de-iced TAT - non-de-ice TAT')
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.savefig('hist_'+experiment_name+'.png')
plt.clf()

#histogram indicated temperature ratio
plt.plot([0.99,1.01],[0,0],'w.',label='Mean ITDI/ITND ratio = '+str(np.around(mean_ratio,decimals=5)),alpha=0)
plt.plot([0.99,1.01],[0,0],'w.',label='Stdev ratio = '+str(np.around(stdev_ratio,decimals=5)),alpha=0)
plt.hist(ratio[~np.isnan(mach)], 50, normed=True,alpha=0.8,color='grey')
plt.axis((0.995,1.005,0,1200))
plt.xlabel('ITDI/ITND')
plt.ylabel('Density')
plt.savefig('hist_indicatedT_'+experiment_name+'.png')
plt.clf()

#difference between TATs over time
plt.plot(TAT_DI_R-TAT_ND_R,'k',label='DI - NDI old')
plt.plot(tat_di_new-tat_nd_new,'g',label='DI - NDI new')
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Time')
plt.ylabel('Temperature difference (K)')
plt.savefig('di-nd_'+experiment_name+'.png')
plt.clf()

#change from old to new given by changing to new etas
plt.plot(TAT_DI_R-tat_di_new,'r',label='DI old-new')
plt.plot(TAT_ND_R-tat_nd_new,'k',label='NDI old-new')
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Time')
plt.ylabel('Temperature difference (K)')
plt.savefig('old-new_'+experiment_name+'.png')
plt.clf()


##check to see if there's any signigicant mach dependendce in the ratio of the indicated temperatures ##############################
def linear_function(data, a,b):
    return (a*data+b)
print('###################################')
popt_x, pcov_x = curve_fit(linear_function,mach[~np.isnan(mach)],ratio[~np.isnan(mach)],sigma=u_ratio[~np.isnan(mach)],absolute_sigma=True)
####################################################################################################################################
#make a plot of mach vs ratio
plt.plot(mach,ratio,'k.',label='ratio measured')
plt.plot(mach,mach*popt_x[0]+popt_x[1],'r.',label='ratio linear fit')
plt.plot(mach,mean_ratio*mach/mach,'b.',label='mean ratio')
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Mach')
plt.ylabel('ratio (ITDI/ITND)')
plt.savefig('ratio_'+experiment_name+'.png')
plt.clf()

llim_lin=str(popt_x[0]*0.3+popt_x[1])
ulim_lin=str(popt_x[0]*0.7+popt_x[1])
print('if we fitted a straight line to ratio of indicated temperatures vs. Mach, ratio would be between '+llim_lin+' (for M=0.3) and '+ulim_lin+' (for M=0.7)')

#make a plot of mach vs eta (Rosemount 5755 for NDI, new one calculated here for DI)
plt.plot(mach_invented,eta_ndi_toplot,'k',label='eta NDI')
plt.plot([5,5],[5,5],'r.',markersize=7, alpha=0.9,label='$\eta$ DI measured')
plt.plot(mach,eta_di_meas,'r.',markersize=0.3, alpha=0.1)
plt.plot(mach_invented,eta_di_calc_toplot,color='maroon',label='$\eta$ DI calculated')
plt.fill_between(mach_invented, eta_di_calc_toplot-u_eta_di_calc_toplot, eta_di_calc_toplot+u_eta_di_calc_toplot,alpha=0.1,color='red')
plt.fill_between(mach_invented, eta_ndi_toplot-sigma_eta_ndi_toplot, eta_ndi_toplot+sigma_eta_ndi_toplot,alpha=0.1,color='grey')
plt.axis((0.2,0.7,-0.002,0.005))
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Mach')
plt.ylabel('$\eta$')
plt.savefig('eta_'+experiment_name+'.png')
plt.clf()

#make a plot similar to above, but showing the various sensor type combinations in different colours
combination_code=di_type_code**2+ndi_type_code
fig = plt.figure()
ax = plt.subplot(111)
ax.fill_between(mach_invented, eta_di_calc_toplot-u_eta_di_calc_toplot, eta_di_calc_toplot+u_eta_di_calc_toplot,alpha=0.1,color='red')
ax.fill_between(mach_invented, eta_ndi_toplot-sigma_eta_ndi_toplot, eta_ndi_toplot+sigma_eta_ndi_toplot,alpha=0.1,color='grey')
ax.scatter([-5,-5],[-5,-5],s=10,c='dimgrey',alpha=0.9,label='plate,loom')
ax.scatter([-5,-5],[-5,-5],s=10,c='silver',alpha=0.9,label='plate,plate')
    
