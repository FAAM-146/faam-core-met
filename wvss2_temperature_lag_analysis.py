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
from scipy.signal import correlate
from scipy import signal, fftpack
from itertools import groupby
from numpy import genfromtxt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.interpolate import interp1d

#############################################################################################################
#This code looks at many flights to find the average lag in seconds between WVSS2 and temperature data
#It tests to see if this lag is different for plate-type PRTs and loom-type PRTs
#############################################################################################################

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
#ssm stands for seconds since midnight
#how many flights will we look at
number_of_flights=95
#length of the chunks you want to split the data into, in wvss2 datapoints (these are usually about 2.3s apart)
segment_length_wdatapoints=50
#segment length is the number of datapoints multiplied by the average time difference between consecutive WVSS2 datapoints
segment_length_seconds_apprx=segment_length_wdatapoints*2.359190732115133
#if the maximum correlation coefficient found when looking at up to 10s of lag is greater than max_cc, count this chunk's lag measurement as good
max_cc=0.7

#set up some arrays to fill
lags=np.zeros((number_of_flights,1000))
lags[:]=np.nan
lengths=np.zeros((number_of_flights,1000))
lengths[:]=np.nan
cc_arr=np.zeros((number_of_flights,1000))
cc_arr[:]=np.nan
lags_noend_full=np.zeros((number_of_flights,1000))
lags_noend_full[:]=np.nan
cc_arr_nointerp=np.zeros((number_of_flights,1000))
lags_full=np.zeros((number_of_flights,1000))
lags_cc_pass=np.zeros((number_of_flights,1000))
cc_arr_nointerp[:]=np.nan
lags_full[:]=np.nan
lags_cc_pass[:]=np.nan
use_array=np.zeros((number_of_flights))
use_array[:]=np.nan
type_array=np.zeros((number_of_flights))
type_array[:]=np.nan
gc=0

#read in some flight data (full rate FAAM core data netcdfs and corresponding raw WVSS2 data files)
for n in range (0,number_of_flights,1):
    if n == 0:
        first_name=glob.glob('*_c124.nc')[0]
        first_name_wvss2='wvss2_data/c124_wvss2.csv'
        use='ND' 
        type='plate'     
    if n == 1:
        first_name=glob.glob('*_c125.nc')[0]
        first_name_wvss2='wvss2_data/c125_wvss2.csv'
        use='ND' 
        type='plate' 
    if n == 2:
        first_name=glob.glob('*_c126.nc')[0]
        first_name_wvss2='wvss2_data/c126_wvss2.csv'
        use='ND' 
        type='plate' 
    if n == 3:
        first_name=glob.glob('*_c127.nc')[0]
        first_name_wvss2='wvss2_data/c127_wvss2.csv'
        use='ND' 
        type='plate'         
    if n == 4:
        first_name=glob.glob('*128.nc')[0]
        first_name_wvss2='wvss2_data/c128_wvss2.csv'
        use='ND'
        type='loom'
    if n == 5:
        first_name=glob.glob('*_c129.nc')[0]
        first_name_wvss2='wvss2_data/c129_wvss2.csv'
        use='ND'
        type='loom'
    if n == 6:
        first_name=glob.glob('*_c130.nc')[0]
        first_name_wvss2='wvss2_data/c130_wvss2.csv'
        use='ND'
        type='loom'
    if n == 7:
        first_name=glob.glob('*_c131.nc')[0]
        first_name_wvss2='wvss2_data/c131_wvss2.csv'
        use='ND'
        type='loom'
    if n == 8:
        first_name=glob.glob('*_c132.nc')[0]
        first_name_wvss2='wvss2_data/c132_wvss2.csv'
        use='ND'
        type='loom'
    if n == 9:
        first_name=glob.glob('*_c133.nc')[0]
        first_name_wvss2='wvss2_data/c133_wvss2.csv'
        use='ND'
        type='loom'
    if n == 10:
        first_name=glob.glob('*_c134.nc')[0]
        first_name_wvss2='wvss2_data/c134_wvss2.csv'
        use='ND' 
        type='loom'
    if n == 11:
        first_name=glob.glob('*_c135.nc')[0]
        first_name_wvss2='wvss2_data/c135_wvss2.csv'
        use='DI'  
        type='plate'        
    if n == 12:
        first_name=glob.glob('*_c136.nc')[0]
        first_name_wvss2='wvss2_data/c136_wvss2.csv'
        use='DI' 
        type='plate'        
    if n == 13:
        first_name=glob.glob('*_c137.nc')[0]
        first_name_wvss2='wvss2_data/c137_wvss2.csv'
        use='DI'  
        type='plate' 
    if n == 14:
        first_name=glob.glob('*_c138.nc')[0]
        first_name_wvss2='wvss2_data/c138_wvss2.csv'
        use='DI'  
        type='plate'          
    if n == 15:
        first_name=glob.glob('*_c139.nc')[0]
        first_name_wvss2='wvss2_data/c139_wvss2.csv'
        use='DI' 
        type='plate'        
    if n == 16:
        first_name=glob.glob('*_c140.nc')[0]
        first_name_wvss2='wvss2_data/c140_wvss2.csv'
        use='DI'   
        type='plate'
    if n == 17:
        first_name=glob.glob('*_c141.nc')[0]
        first_name_wvss2='wvss2_data/c141_wvss2.csv'
        use='DI'   
        type='plate'
    if n == 18:
        first_name=glob.glob('*_c142.nc')[0]
        first_name_wvss2='wvss2_data/c142_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 19:
        first_name=glob.glob('*_c144.nc')[0]
        first_name_wvss2='wvss2_data/c144_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 20:
        first_name=glob.glob('*_c145.nc')[0]
        first_name_wvss2='wvss2_data/c145_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 21:
        first_name=glob.glob('*_c146.nc')[0]
        first_name_wvss2='wvss2_data/c146_wvss2.csv'
        use='DI' 
        type='plate'        
    if n == 22:
        first_name=glob.glob('*_c147.nc')[0]
        first_name_wvss2='wvss2_data/c147_wvss2.csv'
        use='ND'  
        type='plate'
    if n == 23:
        first_name=glob.glob('*_c149.nc')[0]
        first_name_wvss2='wvss2_data/c149_wvss2.csv'
        use='ND'    
        type='plate'
    if n == 24:
        first_name=glob.glob('*_c150.nc')[0]
        first_name_wvss2='wvss2_data/c150_wvss2.csv'
        use='ND'   
        type='plate' 
    if n == 26:
        first_name=glob.glob('*_c152.nc')[0]
        first_name_wvss2='wvss2_data/c152_wvss2.csv'
        use='DI'
        type='plate'
    if n == 27:
        first_name=glob.glob('*_c153.nc')[0]
        first_name_wvss2='wvss2_data/c153_wvss2.csv'
        use='DI'
        type='plate'
    if n == 28:
        first_name=glob.glob('*_c154.nc')[0]
        first_name_wvss2='wvss2_data/c154_wvss2.csv'
        use='DI'
        type='plate'
    if n == 29:
        first_name=glob.glob('*_c155.nc')[0]
        first_name_wvss2='wvss2_data/c155_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 30:
        first_name=glob.glob('*_c156.nc')[0]
        first_name_wvss2='wvss2_data/c156_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 31:
        first_name=glob.glob('*_c157.nc')[0]
        first_name_wvss2='wvss2_data/c157_wvss2.csv'
        use='DI'
        type='loom'
    if n == 32:
        first_name=glob.glob('*_c158.nc')[0]
        first_name_wvss2='wvss2_data/c158_wvss2.csv'
        use='DI'
        type='loom'
    if n == 33:
        first_name=glob.glob('*_c159.nc')[0]
        first_name_wvss2='wvss2_data/c159_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 34:
        first_name=glob.glob('*_c160.nc')[0]
        first_name_wvss2='wvss2_data/c160_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 35:
        first_name=glob.glob('*_c161.nc')[0]
        first_name_wvss2='wvss2_data/c161_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 36:
        first_name=glob.glob('*_c162.nc')[0]
        first_name_wvss2='wvss2_data/c162_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 37:
        first_name=glob.glob('*_c163.nc')[0]
        first_name_wvss2='wvss2_data/c163_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 38:
        first_name=glob.glob('*_c164.nc')[0]
        first_name_wvss2='wvss2_data/c164_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 39:
        first_name=glob.glob('*_c165.nc')[0]
        first_name_wvss2='wvss2_data/c165_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 40:
        first_name=glob.glob('*_c166.nc')[0]
        first_name_wvss2='wvss2_data/c166_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 41:
        first_name=glob.glob('*_c167.nc')[0]
        first_name_wvss2='wvss2_data/c167_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 42:
        first_name=glob.glob('*_c168.nc')[0]
        first_name_wvss2='wvss2_data/c168_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 43:
        first_name=glob.glob('*_c169.nc')[0]
        first_name_wvss2='wvss2_data/c169_wvss2.csv'
        use='ND' 
        type='plate' 
    if n == 44:
        first_name=glob.glob('*_c170.nc')[0]
        first_name_wvss2='wvss2_data/c170_wvss2.csv'
        use='ND' 
        type='plate'  
    if n == 47:
        first_name=glob.glob('*_c173.nc')[0]
        first_name_wvss2='wvss2_data/c173_wvss2.csv'
        use='ND' 
        type='plate' 
    if n == 48:
        first_name=glob.glob('*_c174.nc')[0]
        first_name_wvss2='wvss2_data/c174_wvss2.csv'
        use='ND' 
        type='plate' 
    if n == 49:
        first_name=glob.glob('*_c175.nc')[0]
        first_name_wvss2='wvss2_data/c175_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 50:
        first_name=glob.glob('*_c176.nc')[0]
        first_name_wvss2='wvss2_data/c176_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 51:
        first_name=glob.glob('*_c177.nc')[0]
        first_name_wvss2='wvss2_data/c177_wvss2.csv'
        use='DI' 
        type='plate'       
    if n == 52:
        first_name=glob.glob('*_c178.nc')[0]
        first_name_wvss2='wvss2_data/c178_wvss2.csv'
        use='DI'
        type='plate' 
    if n == 53:
        first_name=glob.glob('*_c179.nc')[0]
        first_name_wvss2='wvss2_data/c179_wvss2.csv'
        use='DI'
        type='plate'
    if n == 54:
        first_name=glob.glob('*_c180.nc')[0]
        first_name_wvss2='wvss2_data/c180_wvss2.csv'
        use='DI'    
        type='plate'        
    if n == 55:
        first_name=glob.glob('*_c181.nc')[0]
        first_name_wvss2='wvss2_data/c181_wvss2.csv'
        use='DI' 
        type='plate'         
    if n == 56:
        first_name=glob.glob('*_c182.nc')[0]
        first_name_wvss2='wvss2_data/c182_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 57:
        first_name=glob.glob('*_c183.nc')[0]
        first_name_wvss2='wvss2_data/c183_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 58:
        first_name=glob.glob('*_c184.nc')[0]
        first_name_wvss2='wvss2_data/c184_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 59:
        first_name=glob.glob('*_c185.nc')[0]
        first_name_wvss2='wvss2_data/c185_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 60:
        first_name=glob.glob('*_c186.nc')[0]
        first_name_wvss2='wvss2_data/c186_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 61:
        first_name=glob.glob('*_c187.nc')[0]
        first_name_wvss2='wvss2_data/c187_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 62:
        first_name=glob.glob('*_c188.nc')[0]
        first_name_wvss2='wvss2_data/c188_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 63:
        first_name=glob.glob('*_c189.nc')[0]
        first_name_wvss2='wvss2_data/c189_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 64:
        first_name=glob.glob('*_c190.nc')[0]
        first_name_wvss2='wvss2_data/c190_wvss2.csv'
        use='DI' 
        type='plate'         
    if n == 65:
        first_name=glob.glob('*_c191.nc')[0]
        first_name_wvss2='wvss2_data/c191_wvss2.csv'
        use='DI'
        type='plate'
    if n == 67:
        first_name=glob.glob('*_c193.nc')[0]
        first_name_wvss2='wvss2_data/c193_wvss2.csv'
        use='DI'      
        type='plate'  
    if n == 68:
        first_name=glob.glob('*_c195.nc')[0]
        first_name_wvss2='wvss2_data/c195_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 69:
        first_name=glob.glob('*_c196.nc')[0]
        first_name_wvss2='wvss2_data/c196_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 70:
        first_name=glob.glob('*_c198.nc')[0]
        first_name_wvss2='wvss2_data/c198_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 71:
        first_name=glob.glob('*_c199.nc')[0]
        first_name_wvss2='wvss2_data/c199_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 72:
        first_name=glob.glob('*_c200.nc')[0]
        first_name_wvss2='wvss2_data/c200_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 73:
        first_name=glob.glob('*_c201.nc')[0]
        first_name_wvss2='wvss2_data/c201_wvss2.csv'
        use='DI'
        type='plate'
    if n == 74:
        first_name=glob.glob('*_c202.nc')[0]
        first_name_wvss2='wvss2_data/c202_wvss2.csv'
        use='DI'     
        type='plate'        
    if n == 75:
        first_name=glob.glob('*_c203.nc')[0]
        first_name_wvss2='wvss2_data/c203_wvss2.csv'        
        use='DI'
        type='plate'
    if n == 76:
        first_name=glob.glob('*_c204.nc')[0]
        first_name_wvss2='wvss2_data/c204_wvss2.csv'
        use='DI' 
        type='plate'         
    if n == 77:
        first_name=glob.glob('*_c205.nc')[0]
        first_name_wvss2='wvss2_data/c205_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 79:
        first_name=glob.glob('*_c207.nc')[0]
        first_name_wvss2='wvss2_data/c207_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 80:
        first_name=glob.glob('*_c208.nc')[0]
        first_name_wvss2='wvss2_data/c208_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 81:
        first_name=glob.glob('*_c209.nc')[0]
        first_name_wvss2='wvss2_data/c209_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 82:
        first_name=glob.glob('*_c210.nc')[0]
        first_name_wvss2='wvss2_data/c210_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 83:
        first_name=glob.glob('*_c211.nc')[0]
        first_name_wvss2='wvss2_data/c211_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 84:
        first_name=glob.glob('*_c212.nc')[0]
        first_name_wvss2='wvss2_data/c212_wvss2.csv'
        use='DI' 
        type='plate'  
    if n == 85:
        first_name=glob.glob('*_c213.nc')[0]
        first_name_wvss2='wvss2_data/c213_wvss2.csv'
        use='DI' 
        type='plate'          
    if n == 86:
        first_name=glob.glob('*_c214.nc')[0]
        first_name_wvss2='wvss2_data/c214_wvss2.csv'
        use='DI'
        type='plate'
    if n == 87:
        first_name=glob.glob('*_c215.nc')[0]
        first_name_wvss2='wvss2_data/c215_wvss2.csv'
        use='DI' 
        type='plate'        
    if n == 88:
        first_name=glob.glob('*_c216.nc')[0]
        first_name_wvss2='wvss2_data/c216_wvss2.csv'        
        use='DI'
        type='plate'
    if n == 89:
        first_name=glob.glob('*_c217.nc')[0]
        first_name_wvss2='wvss2_data/c217_wvss2.csv'
        use='DI' 
        type='plate' 
    if n == 90:
        first_name=glob.glob('*_c218.nc')[0]
        first_name_wvss2='wvss2_data/c218_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 91:
        first_name=glob.glob('*_c219.nc')[0]
        first_name_wvss2='wvss2_data/c219_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 92:
        first_name=glob.glob('*_c220.nc')[0]
        first_name_wvss2='wvss2_data/c220_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 93:
        first_name=glob.glob('*_c221.nc')[0]
        first_name_wvss2='wvss2_data/c221_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 94:
        first_name=glob.glob('*_c222.nc')[0]
        first_name_wvss2='wvss2_data/c222_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 78:
        first_name=glob.glob('*_c223.nc')[0]
        first_name_wvss2='wvss2_data/c223_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 66:
        first_name=glob.glob('*_c224.nc')[0]
        first_name_wvss2='wvss2_data/c224_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 46:
        first_name=glob.glob('*_c225.nc')[0]
        first_name_wvss2='wvss2_data/c225_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 45:
        first_name=glob.glob('*_c226.nc')[0]
        first_name_wvss2='wvss2_data/c226_wvss2.csv'
        use='DI' 
        type='plate'
    if n == 25:
        first_name=glob.glob('*_c227.nc')[0]
        first_name_wvss2='wvss2_data/c227_wvss2.csv'
        use='DI' 
        type='plate'

    if use=='ND':
        use_array[n]=0
    else:
        use_array[n]=1
        
    print(type)
    if type=='loom':
        type_array[n]=0
    elif type=='plate':
        type_array[n]=1
    
    print(first_name_wvss2)    
    wvss2_raw_data = genfromtxt(first_name_wvss2, delimiter=',', dtype=str, skip_header=1)
    start_fh = Dataset(first_name,'r')
    wow_ini = start_fh.variables['WOW_IND'][:]
    time = start_fh.variables['Time'][:]
    PS_RVSM = start_fh.variables['PS_RVSM'][:]
    PALT_RVS_ini = start_fh.variables['PALT_RVS'][:]
    ROLL_GIN_ini = start_fh.variables['ROLL_GIN'][:]
    AOSS_ini = start_fh.variables['AOSS'][:]
    W_C_ini = start_fh.variables['W_C'][:]
    AOA_ini = start_fh.variables['AOA'][:]
    TDEW_GE_ini = start_fh.variables['TDEW_GE'][:]
    WVSS2F_VMR_ini = start_fh.variables['WVSS2F_VMR'][:]
    if use=='DI':
        TAT_DI_R = start_fh.variables['TAT_DI_R'][:]
    if use=='ND':
        TAT_DI_R = start_fh.variables['TAT_ND_R'][:]   
    TAT_DI_R_copy=TAT_DI_R.copy()
    for k in range (0,(len(PALT_RVS_ini)-1)):
        if wow_ini[k,0]==0:
            TAT_DI_R[k,:]=TAT_DI_R[k,:]
        else:
            TAT_DI_R[k,:]=np.nan

    #reshape the data to make it easier to work with
    temperature=np.ravel(TAT_DI_R)
    pressure_core=np.ravel(PS_RVSM)   
    PALT_RVS_unravelled = np.ravel(PALT_RVS_ini)
    ROLL_GIN_unravelled = np.ravel(ROLL_GIN_ini)
    W_C_unravelled = np.ravel(W_C_ini)
    AOSS_unravelled = np.ravel(AOSS_ini)
    TDEW_GE_unravelled = np.ravel(TDEW_GE_ini)
    AOA_unravelled = np.ravel(AOA_ini)
    wow_unravelled = np.ravel(wow_ini)
    time_core=np.zeros(len(temperature))
    time_core[0]=time[0]
    for i in range (1,len(temperature),1):
            time_core[i]=time_core[i-1]+0.03125
    #read in the relevant wvss2 data            
    wvss2_ssm=np.zeros(len(wvss2_raw_data[:,1]))
    wvss2_vmr_raw=np.zeros(len(wvss2_raw_data[:,1]))
    wvss2_vmr_raw_nonan=np.zeros(len(wvss2_raw_data[:,1]))
    wvss2_ssm_nonan=np.zeros(len(wvss2_raw_data[:,1]))
    wvss2_pres=np.zeros(len(wvss2_raw_data[:,1]))
    wvss2_ssm[:]=np.nan
    wvss2_vmr_raw[:]=np.nan
    wvss2_vmr_raw_nonan[:]=np.nan
    wvss2_ssm_nonan[:]=np.nan
    wvss2_pres[:]=np.nan
    wvss2_datetime=wvss2_raw_data[:,0]
    for i in range (0,len(wvss2_vmr_raw+1),1):
        wvss2_time=wvss2_datetime[i][-12:]
        wvss2_hour=float(wvss2_datetime[i][-12:-10])
        wvss2_minute=float(wvss2_datetime[i][-9:-7])
        wvss2_second=float(wvss2_datetime[i][-6:])
        wvss2_ssm_nonan[i]=wvss2_hour*60.0*60.0+wvss2_minute*60.0+wvss2_second
        wvss2_vmr_raw_nonan[i]=float(wvss2_raw_data[i,1])
        wvss2_pres[i]=float(wvss2_raw_data[i,2])
        wow_index=find_nearest_index(time, wvss2_ssm_nonan[i])
        if wow_ini[wow_index]==0:
            wvss2_vmr_raw[i]=float(wvss2_raw_data[i,1])
            wvss2_ssm[i]=wvss2_hour*60.0*60.0+wvss2_minute*60.0+wvss2_second
        else:
            wvss2_vmr_raw[i]=np.nan
            wvss2_ssm[i]=np.nan 
     
    #these are placeholder calibration coefficients (not that the WVSS2 calibration will make any difference to the lag analysis anyway)
    wvss2_cal_coeffs=np.array([0.0,1.0,0.0,0.0])
    wvss2_vmr_calibrated=wvss2_cal_coeffs[0]+wvss2_cal_coeffs[1]*wvss2_vmr_raw+wvss2_cal_coeffs[2]*wvss2_vmr_raw**2.0+wvss2_cal_coeffs[3]*wvss2_vmr_raw**3.0
    wvss2_ssm_copy=wvss2_ssm.copy()
    wvss2_vmr_calibrated_copy=wvss2_vmr_calibrated.copy()    
    length_of_flight_seconds=int(len(temperature[~np.isnan(temperature)])/32)
    #find out how many segments there'll be in this flight from the length of flight and the length of the segments you want
    #-1 to be on the safe side
    number_of_segments=int(length_of_flight_seconds/segment_length_seconds_apprx)-1
    correlation=np.zeros((320,number_of_segments))
    correlation[:]=np.nan

    #this is where the lag analysis happens. For each segment, first find the wvss2 data and nearest temperature data
    for m in range (1,number_of_segments,1):
        ssm_touse= wvss2_ssm_copy[~np.isnan(wvss2_ssm_copy)][segment_length_wdatapoints*m:segment_length_wdatapoints*m+segment_length_wdatapoints]
        wvss2_touse= wvss2_vmr_calibrated_copy[~np.isnan(wvss2_vmr_calibrated_copy)][segment_length_wdatapoints*m:segment_length_wdatapoints*m+segment_length_wdatapoints]
        temperature_touse=np.zeros((len(ssm_touse)))
        temperature_gradient_touse=np.zeros((len(ssm_touse)))
        core_times_matching_ssm_touse=np.zeros((len(ssm_touse)))
        core_indices_matching_ssm_touse=np.zeros((len(ssm_touse)),dtype=int)
        
        for kk in range (0,(len(ssm_touse)),1):
            core_times_matching_ssm_touse[kk]=time_core[find_nearest_index(time_core, ssm_touse[kk])]
            core_indices_matching_ssm_touse[kk]=int(find_nearest_index(time_core, ssm_touse[kk]))

        #temperature measurements to use should be averages of 64 datapoints leading up to nearest temperature measuremet to ssm_touse[0] - lag
        #the position of the maximum of correlation is the delay to wvss2 in [1/32] s 
        #try up to 10 s delay to wvss2 by moving the temperature gradient you average over to the left, by 1/32s each try
        vmr_gradient=np.abs(np.gradient(wvss2_touse))
        for jj in range (0,320,1):
            for ii in range (0,(len(ssm_touse)),1):
                temperature_touse[ii]=np.mean(temperature[(core_indices_matching_ssm_touse[ii]-64-jj):(core_indices_matching_ssm_touse[ii]-jj)])           
            temperature_gradient_touse=np.abs(np.gradient(temperature_touse))
            #correlation for this try:
            correlation[jj,m]=np.corrcoef(vmr_gradient, temperature_gradient_touse)[0,1]  

        lag_wvss2_1over32 = np.argmax(correlation[:,m])

        print('lag for this chunk is '+str(lag_wvss2_1over32/32.))
        lag_wvss2_seconds=lag_wvss2_1over32/32.0  
        if correlation[lag_wvss2_1over32,m] >max_cc:
            lags[n,m]=lag_wvss2_seconds  
        else:   
            lags[n,m]=np.nan      
            
all_lags=np.ravel(lags)
all_plate_lags=np.ravel(lags[np.where(type_array ==1.0)])
all_loom_lags=np.ravel(lags[np.where(type_array ==0.0)])

plt.hist(all_plate_lags[~np.isnan(all_plate_lags)], 50, normed=True,alpha=0.8,color='grey')
plt.axis((0,5,0,1.2))
plt.xlabel('Lag (plates, s)')
plt.ylabel('Density')
plt.savefig('hist_lags_platess_v5_'+str(segment_length_wdatapoints)+'_'+str(max_cc)+'test.png')
plt.clf()

plt.hist(all_loom_lags[~np.isnan(all_loom_lags)], 50, normed=True,alpha=0.8,color='grey')
plt.axis((0,5,0,1.2))
plt.xlabel('Lag (looms, s)')
plt.ylabel('Density')
plt.savefig('hist_lags_looms_v5_'+str(segment_length_wdatapoints)+'_'+str(max_cc)+'test.png')
plt.clf()

plt.hist(all_lags[~np.isnan(all_lags)], 50, normed=True,alpha=0.8,color='grey')
plt.axis((0,5,0,1.2))
plt.xlabel('Lag (all, s)')
plt.ylabel('Density')
plt.savefig('hist_lags_all_v5_'+str(segment_length_wdatapoints)+'_'+str(max_cc)+'test.png')
plt.clf()

print(len(all_loom_lags[~np.isnan(all_loom_lags)]))
print(len(all_plate_lags[~np.isnan(all_plate_lags)]))
print(len(all_lags[~np.isnan(all_lags)]))        
print(np.mean(all_loom_lags[~np.isnan(all_loom_lags)]))
print(np.mean(all_plate_lags[~np.isnan(all_plate_lags)]))
print(np.mean(all_lags[~np.isnan(all_lags)]))
print(np.std(all_loom_lags[~np.isnan(all_loom_lags)]))
print(np.std(all_plate_lags[~np.isnan(all_plate_lags)]))
print(np.std(all_lags[~np.isnan(all_lags)]))


#output for max_cc=0.7,segment_length_wdatapoints=50
#277
#1349
#1626
#2.001692238267148
#1.6332468495181616
#1.696013991389914
#0.8843822905980818
#1.5691805046017877
#1.4816482216756812


        
 