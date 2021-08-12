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
from itertools import groupby
from lmfit import Model

##########################################################################################################################################
#This code uses Buck CR2 data from many flights to create a calibration curve and associated uncertainties for WVSS2 4252 on the flush inlet.
#Data used in this analysis was from 2020 (in the data_2020_posteffix mentioned on L74):
#core_faam_20200130_v005_r0_c214_1hz_prelim.nc
#core_faam_20200203_v005_r0_c215_1hz_prelim.nc
#core_faam_20200203_v005_r0_c216_1hz_prelim.nc
#core_faam_20200204_v005_r0_c217_1hz_prelim.nc
#core_faam_20200205_v005_r0_c218_1hz_prelim.nc
#core_faam_20200206_v005_r0_c219_1hz_prelim.nc
#core_faam_20200206_v005_r0_c220_1hz_prelim.nc
#core_faam_20200207_v005_r0_c221_1hz_prelim.nc
#core_faam_20200208_v005_r0_c222_1hz_prelim.nc
#core_faam_20200211_v005_r0_c223_1hz_prelim.nc
#core_faam_20200211_v005_r0_c224_1hz_prelim.nc
#core_faam_20200212_v005_r0_c225_1hz_prelim.nc
#core_faam_20200212_v005_r0_c226_1hz_prelim.nc
#core_faam_20200213_v005_r0_c227_1hz_prelim.nc
#core_faam_20200214_v005_r0_c228_1hz_prelim.nc
#core_faam_20200214_v005_r0_c229_1hz_prelim.nc
#core_faam_20200310_v005_r0_c230_1hz_prelim.nc
#core_faam_20200827_v005_r0_c231_1hz_prelim.nc
#core_faam_20200921_v005_r0_c232_1hz_prelim.nc
#core_faam_20200925_v005_r0_c233_1hz_prelim.nc
##########################################################################################################################################



#effix refers to the fix to a misplaced bracket in the Buck processing code, found and fixed April 2021. This code is run using data processed after the bracket was fixed

#maximum number of calibration points to look for
cal_points=250    
#how long in seconds you want to average over at the end of the chunk to create a cal point
sample_length=10
#fractional maximum allowable stdev in either buck or wvss2 vmr over the calibration chunk (the last sample_length seconds)
max_vmr_stdev=0.02

#Extract from worklog: Nov. 14, 2018, 4:04 p.m.	WVSSII 4252 fitted to flush inlet, window 6. This unit has never been fitted to the aircraft before but has been tested in lab.
#Only use data since 14th November 2018 - before that it was different units 
#Buck was last repaired January 2020, potential intermittently slight overreading issue before then, so use data since January 2020 only

#read in data from a starter file
first_name='core_faam_20200127_v005_r0_c213_1hz_prelim_new.nc'
start_fh = Dataset(first_name,'r')
TAS = start_fh.variables['TAS'][:]
wow = start_fh.variables['WOW_IND'][:]
lat = start_fh.variables['LAT_GIN'][:]
time = start_fh.variables['Time'][:]
PALT_RVS = start_fh.variables['PALT_RVS'][:]   
TAT_DI_R = start_fh.variables['TAT_DI_R'][:]
TDEW_GE = start_fh.variables['TDEW_GE'][:]
PS_RVSM = start_fh.variables['PS_RVSM'][:]
Q_RVSM = start_fh.variables['Q_RVSM'][:]
WVSS2F_VMR = start_fh.variables['WVSS2F_VMR'][:]
TDEWCR2C= start_fh.variables['TDEWCR2C'][:]
VMR_CR2= start_fh.variables['VMR_CR2'][:]
VMR_C_U= start_fh.variables['VMR_C_U'][:]
TDEW_C_U= start_fh.variables['TDEW_C_U'][:]

#get more data
fnames = glob.glob('data_2020_posteffix/*_1hz_prelim.nc')
fnames = np.sort(fnames)

for f in fnames:
    fh = Dataset(f,'r')
    time =np.concatenate((time,time[-1]+fh.variables['Time'][:]))
    PS_RVSM = np.concatenate((PS_RVSM,fh.variables['PS_RVSM'][:]))
    PALT_RVS = np.concatenate((PALT_RVS,fh.variables['PALT_RVS'][:]))
    VMR_CR2 = np.concatenate((VMR_CR2,fh.variables['VMR_CR2'][:]))
    #if there wasn't a de-iced temperature in the file, use non-de-iced instead, call it TAT_DI_R for ease from here on
    try:	
        TAT_DI_R = np.concatenate((TAT_DI_R,fh.variables['TAT_DI_R'][:]))
    except KeyError as e:
        TAT_DI_R = np.concatenate((TAT_DI_R,fh.variables['TAT_ND_R'][:]))
    try:	
        TDEWCR2C = np.concatenate((TDEWCR2C,fh.variables['TDEWCR2C'][:]))		
    except KeyError as e:
        print('no CR2 data')
        raise Exception			
    wow = np.concatenate((wow,fh.variables['WOW_IND'][:]))
    TDEW_C_U = np.concatenate((TDEW_C_U,fh.variables['TDEW_C_U'][:]))
    VMR_C_U= np.concatenate((VMR_C_U,fh.variables['VMR_C_U'][:]))
    WVSS2F_VMR = np.concatenate((WVSS2F_VMR,fh.variables['WVSS2F_VMR'][:]))
    TDEW_GE = np.concatenate((TDEW_GE,fh.variables['TDEW_GE'][:]))

max_vmr=np.nanmax(WVSS2F_VMR)

#save some copies of each array because you're about to get rid of (make NaN) quite a bit of data when you filter out profiles etc.
PS_RVSMs=1.0*PS_RVSM
TAT_DI_Rs=1.0*TAT_DI_R
WVSS2F_VMRs=1.0*WVSS2F_VMR
VMR_CR2s=1.0*VMR_CR2
TDEWCR2Cs=1.0*TDEWCR2C
VMR_C_Us=1.0*VMR_C_U

#rougly calculate relative humidity (don't use these formulae for anything other than rough)
ew_tat=np.exp(np.log(611.2)+(17.62*(TAT_DI_Rs-273.15))/(243.12+(TAT_DI_Rs-273.15)))
ew_tdew=np.exp(np.log(611.2)+(17.62*(TDEWCR2Cs-273.15))/(243.12+(TDEWCR2Cs-273.15)))    
rh=(100.0*ew_tdew/ew_tat)
    
#make the arrays to fill
TempProfile=np.array(TAT_DI_R)
ProfileFlag=np.array(TAT_DI_R)
ProfileFlag[:]=0

#go through all the data. make a first guess as to whether the aircraft is profiling or level
for i in range (0,(len(PALT_RVS)-1)):
    ProfileFlag[-1]=-17
    if PALT_RVS[i]<=(PALT_RVS[(i+1)]-2):
        TempProfile[i]=1
    elif PALT_RVS[i]>=(PALT_RVS[(i+1)]+2):
        TempProfile[i]=2
    else:
        TempProfile[i]=0 
#go through all possible profile data that was just found and look at each half second to see if the profile was consistent. if it was, give the data a flag. give it 10s to stabilise        
for i in range (0,(len(PALT_RVS)-1),1):
    if TempProfile[i]==1 and TempProfile[i+1]==1:
        ProfileFlag[i:i+10]=1
    elif TempProfile[i] and TempProfile[i+1]==2:
        ProfileFlag[i:i+10]=2
    elif ProfileFlag[i]==0:
        ProfileFlag[i]=0

#look though all the data 
for i in range (0,(len(PALT_RVS)-1)):
   #everytime the rh goes above 95%, set the next 5 minutes to be nan
    if (rh[i])>95.0:
        ProfileFlag[i:i+600]=-5
    #if the aircraft was on the ground, give the data a flag
    if wow[i]==0:
        ProfileFlag[i]=ProfileFlag[i]
    else:
        ProfileFlag[i]=-10
    #if either VMR is negative, give the data a flag
    if VMR_CR2[i] <= 0.0:
        ProfileFlag[i]=-52
    if WVSS2F_VMR[i] <= 0.0:
        ProfileFlag[i]=-53
    #give the data and the following 20 minutes a flag if this datapoint, the one before and the one after has a negative VMR (Buck or WVSS2)
    if (VMR_CR2[i-1] <= 0.0) & (VMR_CR2[i] <= 0.0) &(VMR_CR2[i+1] <= 0.0):
        ProfileFlag[i:i+1200]=-54
    if (WVSS2F_VMR[i-1] <= 0.0) & (WVSS2F_VMR[i] <= 0.0) & (WVSS2F_VMR[i+1] <= 0.0) :
        ProfileFlag[i:i+1200]=-55
    #change all datapoints which have a non-zero flag to be nans
    if ProfileFlag[i]==0:
        PS_RVSM[i]=PS_RVSM[i]
        TAT_DI_R[i]=TAT_DI_R[i]
        WVSS2F_VMR[i]=WVSS2F_VMR[i]
        VMR_CR2[i]=VMR_CR2[i]
        TDEWCR2C[i]=TDEWCR2C[i]        
        VMR_C_U[i]=VMR_C_U[i]
    else:
        PS_RVSM[i]=np.nan
        TAT_DI_R[i]=np.nan
        WVSS2F_VMR[i]=np.nan
        VMR_CR2[i]=np.nan    
        TDEWCR2C[i]=np.nan   
        VMR_C_U[i]=np.nan
        
VMR_CR2[len(PALT_RVS)-1]=np.nan
PS_RVSM[len(PALT_RVS)-1]=np.nan
TAT_DI_R[len(PALT_RVS)-1]=np.nan
WVSS2F_VMR[len(PALT_RVS)-1]=np.nan 
TDEWCR2C[len(PALT_RVS)-1]=np.nan   
VMR_C_U[len(PALT_RVS)-1]=np.nan
        
#make the arrays that will contain the calibration data
cal_wvss2_vmr=np.zeros(cal_points)      
cal_wvss2_vmr_stdev=np.zeros(cal_points)    
cal_len=np.zeros(cal_points)    
cal_buck_vmr=np.zeros(cal_points)    
cal_buck_t_stdev=np.zeros(cal_points)    
cal_buck_vmr_stdev=np.zeros(cal_points)    
cal_buck_vmr_averageunc=np.zeros(cal_points)    
cal_buck_tmirr=np.zeros(cal_points)   
cal_buck_tmirr_temp=np.zeros(cal_points)   
cal_ge_tmirr=np.zeros(cal_points) 
cal_psrvsm=np.zeros(cal_points)   
#minimun length of a chunk based on the time response of the buck at that mirror temperature, plus the time you want the calibration at the end of the chunk to be averaged over   
required_len=np.zeros(cal_points)

for i in range (0,cal_points):   
    #sort the data into chunks of good data, make a list of these chunks starting with the longest
    result_wvss2f = [list(v) for k,v in groupby(WVSS2F_VMR,np.isfinite) if k]        
    wvss2f_touse= np.array(sorted(result_wvss2f, key=lambda x: len(x))[-(1+i)])        
    index_start=np.where(WVSS2F_VMR==wvss2f_touse[0])
    index_end=np.where(WVSS2F_VMR==wvss2f_touse[-1])
    time_touse=time[index_start[0][0]:(index_end[0][0]+1)]
    buckT_touse_temp=TDEWCR2C[index_start[0][0]:(index_end[0][0]+1)]
    cal_buck_tmirr_temp[i]=np.mean(buckT_touse_temp[-(sample_length+1):-1])      
    #nasty fix to get around occassional issue of having two wvss2 measurements that are exactly the same in different places (means you can't properly find the start or end of the chunk)
    if (index_end[0][0]-index_start[0][0]) != (len(wvss2f_touse)-1):
        print('duplicate datapoint - no need to worry but ignore this chunk')
        required_len[i]=100000000000000000000.0
    else:
        #this is the [length of time the buck takes to stabilise at this mirror temperature (calculated based on calibration data of 10 K steps, so should be plenty)]+[sample length (number of seconds you want to average over at the end of the chunk)]
        required_len[i]=(-4.1545423E-04*cal_buck_tmirr_temp[i]**3 + 3.8758575E-01*cal_buck_tmirr_temp[i]**2 - 1.2046226E+02*cal_buck_tmirr_temp[i] + 1.2503438E+04)+sample_length
        
    #the chunk needs to be at least as long as the bit you're going to take as the sample (the period you average over) + settling time at that mirror temperature. 
    if len(time_touse) > required_len[i]:
        buckvmr_touse=VMR_CR2[index_start[0][0]:(index_end[0][0]+1)]
        buckvmrunc_touse=VMR_C_U[index_start[0][0]:(index_end[0][0]+1)]
        buckT_touse=TDEWCR2C[index_start[0][0]:(index_end[0][0]+1)]
        geT_touse=TDEW_GE[index_start[0][0]:(index_end[0][0]+1)]
        psrvsm_touse=PS_RVSM[index_start[0][0]:(index_end[0][0]+1)]
        #now get out the average or standard deviations of the last sample_length seconds of this chunk)
        cal_wvss2_vmr[i]=np.mean(wvss2f_touse[-(sample_length+1):-1])
        cal_wvss2_vmr_stdev[i]=np.std(wvss2f_touse[-(sample_length+1):-1])
        cal_len[i]=len(time_touse) 
        cal_buck_t_stdev[i]=np.std(buckT_touse[-(sample_length+1):-1])
        cal_buck_vmr[i]=np.mean(buckvmr_touse[-(sample_length+1):-1])
        cal_buck_vmr_stdev[i]=np.std(buckvmr_touse[-(sample_length+1):-1])
        cal_buck_vmr_averageunc[i]=np.mean(buckvmrunc_touse[-(sample_length+1):-1])
        cal_buck_tmirr[i]=np.mean(buckT_touse[-(sample_length+1):-1])
        cal_ge_tmirr[i]=np.mean(geT_touse[-(sample_length+1):-1])
        cal_psrvsm[i]=np.mean(psrvsm_touse[-(sample_length+1):-1])
        #filter out chunks where the standard deviation in either Buck or WVSS2 VMR is more than max_vmr_stdev (initially 10%) of the average VMR for this chunk
        if (cal_buck_vmr_stdev[i]/cal_buck_vmr[i]>max_vmr_stdev)    |  (cal_wvss2_vmr_stdev[i]/cal_wvss2_vmr[i]  >max_vmr_stdev) :
            cal_wvss2_vmr[i]=np.nan
            cal_buck_vmr[i]=np.nan  
            cal_buck_vmr_averageunc[i]=np.nan         
        print('******************************')
        print(i)
        print(cal_wvss2_vmr[i])
        print(cal_buck_vmr[i]) 
        print(cal_buck_vmr_stdev[i])
        print(cal_wvss2_vmr_stdev[i])
        
    else:
        cal_wvss2_vmr[i]=np.nan
        cal_wvss2_vmr_stdev[i]=np.nan
        cal_len[i]=np.nan
        cal_buck_vmr[i]=np.nan
        cal_buck_vmr_stdev[i]=np.nan
        cal_buck_vmr_averageunc[i]=np.nan
        cal_buck_tmirr[i]=np.nan
        cal_ge_tmirr[i]=np.nan
        cal_psrvsm[i]=np.nan      
        
    #reset arrays
    result_wvss2f=np.nan
    wvss2f_touse=np.nan
    index_start=np.nan
    index_end=np.nan
    time_touse=np.nan
    buckvmr_touse=np.nan
    buckvmrunc_touse=np.nan
    buckT_touse=np.nan
    geT_touse=np.nan
    
#plot of WVSS2 uncalibrated vs Buck VMR
plt.plot(WVSS2F_VMR,VMR_CR2,'.',label='Flight data')
plt.plot(cal_wvss2_vmr,cal_buck_vmr,'r.',label='Calibration points')
plt.plot([0.0,np.nanmax(WVSS2F_VMR)],[0.0,np.nanmax(WVSS2F_VMR),],'k',label='1:1')
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Buck CR2 VMR (ppm)')
plt.savefig('wvss2_vs_buck_flightdata_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()

#for the calibration data, get rid of the NaNs and sort it by WVSS2 VMR (makes for easier plotting)
unsorted_xdata = cal_wvss2_vmr[~np.isnan(cal_wvss2_vmr)]
unsorted_ydata = cal_buck_vmr[~np.isnan(cal_wvss2_vmr)]
#the uncertainty in the Buck average for each calibration point is the RSS of [average unceratinty in the Buck VMR across sample_length (halved because of a coverage factor of 2 in VMR_C_U for data processed at this time)] and [standard deviation in that average]
unsorted_unc=((0.5*cal_buck_vmr_averageunc[~np.isnan(cal_wvss2_vmr)])**2.0+(cal_buck_vmr_stdev[~np.isnan(cal_wvss2_vmr)])**2.0)**0.5
unsorted_yweights=1.0/unsorted_unc
arr1inds = unsorted_xdata.argsort()
xdata = unsorted_xdata[arr1inds[::-1]]
ydata = unsorted_ydata[arr1inds[::-1]]
unc = unsorted_unc[arr1inds[::-1]]
yweights = unsorted_yweights[arr1inds[::-1]]

#do a cubic fit to the calibration data, using lmfit (because it calculates the uncertainty in the predicted model for us
def func(t, a, b, c, d):
    return a + b*t + c*t**2.0 +d*t**3.0
# build Model
hmodel = Model(func)
# create lmfit Parameters. Note that you really must provide initial values.
params = hmodel.make_params(a=0.0, b=1.0, c=0.0, d=0.0)
# do fit, print resulting parameters
result = hmodel.fit(ydata, params, t=xdata,weights=yweights)
print(result.fit_report())
# calculate the (1 sigma) uncertainty in the predicted model
dprod = result.eval_uncertainty(result.params, sigma=1)  



#Unceratinties to include: fit (i.e. uncertainty that the true fit to the data lies within this bound), residuals, buck uncertainty

############### sigma_f ##################################################################################################################################  
#uncertatinty due to fit, call it sigma_f
sigma_f=1.0*dprod
def func_quintic(x, aaa, bbb, ccc, ddd,eee,fff):
    return aaa + bbb*x + ccc*x**2.0 +ddd*x**3.0+eee*x**4.0+fff*x**5.0
#do a quitic fit to sigma_f so you can easily describe it as a formula
popt_quintic_sigma_f,pcov_quintic_sigma_f = curve_fit(func_quintic, xdata,sigma_f)
############### sigma_f ##################################################################################################################################  

############### sigma_r ##################################################################################################################################  
#some of the uncertainty in the calibrated WVSS2 VMR can be quantified from the residuals to the fit. The residuals are generally smaller in magnitude at the lower end of VMR, but larger as a fraction of the measured VMR for that calibration point. So, fit a curve to the maximum fractional residual across the range
residuals=result.best_fit-ydata
residuals_magnitude=np.abs(residuals)
fractional_residuals=(result.best_fit-ydata)/ydata
fractional_residuals_magnitude=np.abs((result.best_fit-ydata)/ydata)
max_ur_arr_x=np.empty(25)
max_ur_arr=np.empty(25)
max_ur_arr_x[:]=np.nan
max_ur_arr[:]=np.nan
#find the maximum fractional residual magnitude for each 1000 VMR - this is what you'll fit the sigma_r curve to
for i in range (0,25):    
    max_ur_arr_x[i]=float(i)*1000.0+500.0
    try:
        max_ur_arr[i]=np.nanmax(fractional_residuals_magnitude[np.where((xdata>(max_ur_arr_x[i]-500.0))&(xdata<(max_ur_arr_x[i]+500.0)))])
    except ValueError:  #raised if `y` is empty.
        pass
#the shape of this data is well represented by a power law (increasing at smaller values)
def func_power(x, pp,qq):
    return pp*x**qq
popt_power,pcov_power = curve_fit(func_power, max_ur_arr_x[~np.isnan(max_ur_arr)],max_ur_arr[~np.isnan(max_ur_arr)])
sigma_r_fractional_power=(popt_power[0]*xdata**popt_power[1])
sigma_r=sigma_r_fractional_power*xdata
plt.plot(xdata,fractional_residuals_magnitude,'ko',label='Fractional residual for each calibration point')    
plt.plot(max_ur_arr_x,max_ur_arr,'go',label='Local maximum fractional residual')
plt.plot(xdata,sigma_r_fractional_power,'b-',label='Power law fit to maximum fractional residual')
plt.axis((-1000,26000,-0.01,0.3))
plt.legend(fontsize=10,loc=1,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Fractional residual')
plt.savefig('residuals_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()
############### sigma_r ################################################################################################################################## 

############### sigma_b ################################################################################################################################## 
#this is looking at the uncertainty in each buck measurement. VMR_C_U isn't a fixed function of vmr because there are lots of components. the below allows us to work out the minimum buck unceratinty at each datapoint - this has to contribute to the unceratinnty in calibrated WVSS2
min_u_0=np.nanmin(VMR_C_U[np.where(VMR_CR2[~np.isnan(VMR_CR2)]<1000.0)])
min_u_5000=np.nanmin(VMR_C_U[~np.isnan(VMR_CR2)][np.where(VMR_CR2[~np.isnan(VMR_CR2)]>5000.0)])
min_u_10000=np.nanmin(VMR_C_U[~np.isnan(VMR_CR2)][np.where(VMR_CR2[~np.isnan(VMR_CR2)]>10000.0)])
min_u_15000=np.nanmin(VMR_C_U[~np.isnan(VMR_CR2)][np.where(VMR_CR2[~np.isnan(VMR_CR2)]>15000.0)])
min_u_20000=np.nanmin(VMR_C_U[~np.isnan(VMR_CR2)][np.where(VMR_CR2[~np.isnan(VMR_CR2)]>20000.0)])
min_u_arr=[min_u_0,min_u_5000,min_u_10000,min_u_15000,min_u_20000]
min_u_arr_x=[0.0,5000.0,10000.0,15000.0,20000.0]
#the shape of this data is well represented by a straight line
def func_linear(x, aa, bb):
    return aa + bb*x 
invented_x=np.linspace(0, max_vmr,10000 )
popt_linear,pcov_linear = curve_fit(func_linear, min_u_arr_x,min_u_arr)#, sigma=unc, absolute_sigma=True)
#half it because VMR_C_U is the k=2 uncertainty
sigma_b=0.5*(popt_linear[0]+popt_linear[1]*xdata)
plt.plot(VMR_CR2[~np.isnan(VMR_CR2)], VMR_C_U[~np.isnan(VMR_CR2)],'r.',markersize=0.1,alpha=0.8)
plt.plot([-9999,99999],[-9999,99999],'r.',label='Flight data')
#plt.plot(min_u_arr_x,min_u_arr,'go')
plt.plot(invented_x, func_linear(invented_x, *popt_linear), 'b-',  label='Fit to minimum Buck CR2 uncertainty')
plt.axis((-1000,26000,-10,1000))
plt.legend(fontsize=10,loc=1,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Buck CR2 VMR (ppm)')
plt.ylabel('Uncertainty in Buck CR2 VMR (ppm)')
plt.savefig('buck_unc_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()
############### sigma_b ################################################################################################################################## 

#this gives you the uncerainty at each calibration point, same thing as u_wvss2_invented_x but with different xs, so you can plot it nicely
sigma_total=(sigma_f**2.0+sigma_r**2.0+sigma_b**2.0)**0.5



#invent lots of WVSS2 VMRs so you can use them as x values to plot the curves you've made
invented_x=np.linspace(0, max_vmr+20000,20000 )
calibrated_wvss2_inventedx=result.params['a'].value + result.params['b'].value*invented_x + result.params['c'].value*invented_x**2.0 +result.params['d'].value*invented_x**3.0  
sigma_b_inventedx=0.5*(popt_linear[0]+popt_linear[1]*invented_x)
sigma_f_inventedx=(popt_quintic_sigma_f[0]+popt_quintic_sigma_f[1]*invented_x+popt_quintic_sigma_f[2]*invented_x**2.0+popt_quintic_sigma_f[3]*invented_x**3.0+popt_quintic_sigma_f[4]*invented_x**4.0+popt_quintic_sigma_f[5]*invented_x**5.0)
sigma_r_inventedx=invented_x*(popt_power[0]*invented_x**popt_power[1])
u_wvss2_inventedx=(sigma_b_inventedx**2.0+sigma_f_inventedx**2.0+sigma_r_inventedx**2.0)**0.5
#make a plot to show the sources of unceratinty
plt.plot(invented_x,sigma_f_inventedx,'g-',label='Uncertainty in fit')
plt.plot(invented_x,sigma_b_inventedx,'b-',label='Buck CR2 minimum uncertainty')
plt.plot(invented_x,sigma_r_inventedx,'r-',label='Fit residuals')
plt.plot(invented_x,u_wvss2_inventedx,'k-',label='Combined uncertainty')
plt.axis((-1000,26000,-10,800))
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Uncertainty VMR (ppm)')
plt.savefig('wvss2_cal_uncertainties_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()

#calculate unceratinties for each flight data point
sigma_f_data=(popt_quintic_sigma_f[0]+popt_quintic_sigma_f[1]*WVSS2F_VMRs+popt_quintic_sigma_f[2]*WVSS2F_VMRs**2.0+popt_quintic_sigma_f[3]*WVSS2F_VMRs**3.0+popt_quintic_sigma_f[4]*WVSS2F_VMRs**4.0+popt_quintic_sigma_f[5]*WVSS2F_VMRs**5.0)
sigma_b_data=0.5*(popt_linear[0]+popt_linear[1]*WVSS2F_VMRs)
sigma_r_data=WVSS2F_VMRs*(popt_power[0]*WVSS2F_VMRs**popt_power[1])
print('********************')
print('popt_quintic_sigma_f')
print(popt_quintic_sigma_f)
print('popt_linear')
print(popt_linear)
print('popt_power')
print(popt_power)

#the combined uncertainty is the RSS of the fit, Buck and residual unceratinties
u_wvss2c=(sigma_f_data**2.0+sigma_r_data**2.0+sigma_b_data**2.0)**0.5
u_wvss2c_expanded=2.0*u_wvss2c
calibrated_wvss2=result.params['a'].value + result.params['b'].value*WVSS2F_VMRs + result.params['c'].value*WVSS2F_VMRs**2.0 +result.params['d'].value*WVSS2F_VMRs**3.0  
print('result.params[a].value') 
print(result.params['a'].value)
print('result.params[b].value')
print(result.params['b'].value)
print('result.params[c].value')
print(result.params['c'].value)
print('result.params[d].value')
print(result.params['d'].value)
fractional_uncertainty=u_wvss2c/calibrated_wvss2
plt.plot(calibrated_wvss2,fractional_uncertainty,'.')
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('fractional uncertainty')
plt.savefig('wvss2_calibration_fracunc'+str(sample_length)+'_posteffix.png')
plt.clf()

calibrated_wvss2_goodonly=result.params['a'].value + result.params['b'].value*WVSS2F_VMR + result.params['c'].value*WVSS2F_VMR**2.0 +result.params['d'].value*WVSS2F_VMR**3.0  
#plot all the flight data, highlighting the good data (the bits that weren't NaNed by ProfileFlag at the start) 
plt.plot(time,VMR_CR2s, 'blue')
plt.plot(time,WVSS2F_VMRs, 'red')
plt.fill_between(time,
                 calibrated_wvss2-u_wvss2c,
                 calibrated_wvss2+u_wvss2c,
                 color="#EF5008",
                 label='uncertainty band of fit')    
plt.fill_between(time,
                 VMR_CR2s-0.5*VMR_C_U,
                 VMR_CR2s+0.5*VMR_C_U,
                 color="#EF1008",
                 label='uncertainty band of fit')                    
plt.plot(time,calibrated_wvss2, 'black')
plt.plot(time,calibrated_wvss2_goodonly, 'yellow')
plt.show()
plt.clf()

#plot the calibration data and the fit, with error bars and a band to show 1 sigma combined uncerainty. first for low VMR, then for high VMR
plt.errorbar(xdata, ydata, yerr=unc,fmt='.', label='Calibration points')
plt.plot(xdata, result.best_fit, 'k--', label='Cubic fit')
plt.fill_between(invented_x,
                 calibrated_wvss2_inventedx-u_wvss2_inventedx,
                 calibrated_wvss2_inventedx+u_wvss2_inventedx,
                 color="#EF5008",alpha=0.5,
                 label='Combined uncertainty')                     
plt.legend(loc='upper left')
plt.axis((-200,13000,-200,13000))
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Buck CR2 VMR (ppm)')
plt.savefig('wvss2_calibration_curve_low_vmr_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()
plt.errorbar(xdata, ydata, yerr=unc,fmt='.', label='Calibration points')
plt.plot(xdata, result.best_fit, 'k--', label='Cubic fit')
plt.fill_between(invented_x,
                 calibrated_wvss2_inventedx-u_wvss2_inventedx,
                 calibrated_wvss2_inventedx+u_wvss2_inventedx,
                 color="#EF5008",alpha=0.5,
                 label='Combined uncertainty')                     
plt.axis((11000,25000,11000,25000))
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Buck CR2 VMR (ppm)')
plt.savefig('wvss2_calibration_curve_high_vmr_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()

#plot the calibration data and the fit, with error bars and a band to show 1 sigma combined uncerainty. first for low VMR, then for high VMR, then for very high
plt.errorbar(xdata, ydata, yerr=unc,fmt='.', label='Calibration points')
plt.plot(xdata, result.best_fit, 'k--', label='Cubic fit')
plt.fill_between(invented_x,
                 calibrated_wvss2_inventedx-u_wvss2_inventedx,
                 calibrated_wvss2_inventedx+u_wvss2_inventedx,
                 color="#EF5008",alpha=0.5,
                 label='Combined uncertainty')                     
plt.legend(loc='upper left')
plt.axis((-200,13000,-200,13000))
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Buck CR2 VMR (ppm)')
plt.savefig('wvss2_calibration_curve_low_vmr_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()
plt.errorbar(xdata, ydata, yerr=unc,fmt='.', label='Calibration points')
plt.plot(xdata, result.best_fit, 'k--', label='Cubic fit')
plt.fill_between(invented_x,
                 calibrated_wvss2_inventedx-u_wvss2_inventedx,
                 calibrated_wvss2_inventedx+u_wvss2_inventedx,
                 color="#EF5008",alpha=0.5,
                 label='Combined uncertainty')                     
plt.axis((20000,35000,20000,35000))
plt.legend(fontsize=10,loc=2,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('WVSS2 VMR (ppm)')
plt.ylabel('Buck CR2 VMR (ppm)')
plt.savefig('wvss2_calibration_curve_veryhigh_vmr_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()

#plot to show the difference in VMR between the Buck and the WVSS2, before and after this WVSS2 calibration
diff_old=VMR_CR2-WVSS2F_VMR
diff_new=VMR_CR2-calibrated_wvss2_goodonly
plt.plot(VMR_CR2,diff_old,'b.',markersize=0.4)
plt.plot(VMR_CR2,diff_new,'g.',markersize=0.4)
plt.plot([-99999,99999],[-99999,99999],'b.',label='Difference without calibration')
plt.plot([-99999,99999],[-99999,99999],'g.',label='Difference after calibration')
plt.plot([-9999,99999],[0,0],'k')
plt.axis((-200,25000,-2000,2000))
plt.legend(fontsize=10,loc=3,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Buck CR2 VMR (ppm)')
plt.ylabel('Buck CR2 VMR - WVSS2 VMR (ppm)')
plt.savefig('diff_vmr_sl'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()
diff_old_fractional=(VMR_CR2-WVSS2F_VMR)/WVSS2F_VMR
diff_new_fractional=(VMR_CR2-calibrated_wvss2_goodonly)/calibrated_wvss2_goodonly
plt.plot(VMR_CR2,diff_old_fractional,'b.',markersize=0.4)
plt.plot(VMR_CR2,diff_new_fractional,'g.',markersize=0.4)
plt.plot([99999,99999],[99999,99999],'b.',label='Difference without calibration')
plt.plot([99999,99999],[99999,99999],'g.',label='Difference after calibration')
plt.fill_between(invented_x,
                 -u_wvss2_inventedx/invented_x,
                 u_wvss2_inventedx/invented_x,
                 color="green",alpha=0.3,label='Uncertainty in calibrated WVSS2 VMR (as fraction)')  
plt.plot([-9999,99999],[0,0],'k')
plt.axis((50,1000,-2,2))
plt.xscale('log')
plt.legend(fontsize=10,loc=4,markerfirst=False,fancybox=False,framealpha=0)
plt.xlabel('Buck CR2 VMR (ppm)')
plt.ylabel('(Buck CR2 VMR - WVSS2 VMR )/ WVSS2 VMR ')
plt.savefig('diff_vmr_sl_fractionallog'+str(sample_length)+'_posteffix.png')
plt.show()
plt.clf()

#output from posteffix (effix relating to the missing bracket in the calculation of the enhancement factor in the Buck data processing, fixed April 2021):
#********************
#popt_quintic_sigma_f
#[ 1.60610563e+00  6.92615739e-03 -1.81841518e-06  3.31182678e-10
# -2.14777675e-14  4.68728542e-19]
#popt_linear
#[6.73899570e+00 4.54789904e-03]
#popt_power
#[ 8.30487713 -0.60291293]
#result.params[a].value
#24.579821051318337
#result.params[b].value
#0.9569840340817223
#result.params[c].value
#1.1619968131941573e-05
#result.params[d].value
#-3.6889287391309806e-10
#sigma_f_data=(popt_quintic_sigma_f[0]+popt_quintic_sigma_f[1]*WVSS2F_VMRs+popt_quintic_sigma_f[2]*WVSS2F_VMRs**2.0+popt_quintic_sigma_f[3]*WVSS2F_VMRs**3.0+popt_quintic_sigma_f[4]*WVSS2F_VMRs**4.0+popt_quintic_sigma_f[5]*WVSS2F_VMRs**5.0)
#sigma_b_data=0.5*(popt_linear[0]+popt_linear[1]*WVSS2F_VMRs)
#sigma_r_data=WVSS2F_VMRs*(popt_power[0]*WVSS2F_VMRs**popt_power[1])
#u_wvss2c=(sigma_f_data**2.0+sigma_r_data**2.0+sigma_b_data**2.0)**0.5
#calibrated_wvss2=result.params['a'].value + result.params['b'].value*WVSS2F_VMRs + result.params['c'].value*WVSS2F_VMRs**2.0 +result.params['d'].value*WVSS2F_VMRs**3.0  

#DON'T APPLY THE CALIBRATION ABOVE 23000 PPM


