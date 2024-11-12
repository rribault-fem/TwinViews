import pandas as pd

# nopn interractive backend to avoid matplotlib openning new windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfilt
import seaborn as sns

def ButterFilter(data : pd.Series, freq : float = 10, filter_type : str =  'hp', fs:float = 200):
    """
    The Butterworth filter used for filtering data
    data: the dataset as a pd.Series 
    freq: the cutting frequency, as an array wtih [lp, hp] is bandpass 
    filter type: "lp" low pass, "hp": high pass, "bp": pass band filter 
    fs: sampling frequency of the data 
    """
    sos = butter(50, freq, filter_type, fs=fs, output='sos')
    data_filt = sosfilt(sos, data)
    data_filt = pd.Series(data_filt, index=data.index ) 
    return(data_filt)

def FilterData(data : pd.Series, lf:int, hf:int, fs:int):
    lf_val = ButterFilter(data, freq = lf, filter_type = 'lp', fs = fs)
    wf_val = ButterFilter(data, freq = [lf,hf], filter_type = 'bp', fs = fs)
    hf_val = ButterFilter(data, freq = hf, filter_type = 'hp', fs = fs)
    return(lf_val, wf_val, hf_val)


def XarrStats(xarr:xr.Dataset, var:str, lf:float = 1/16.5 , hf:float = 1/0.52, sampling: str = '1H'):
    #generate simulation variable name 
    simu_var = 'simu_'+var
    #generate UTC time index
    utc = []
    for time in xarr['time']:
        global_time = xarr.sel(time = time)['time'].values + pd.to_timedelta(xarr.sel(time = time)['time_sensor'], unit = 's').round('ms').values
        utc.append(global_time)
    utc = np.concatenate(utc)       
    #Convert to pd.Series
    simu_Series = pd.Series(data = np.concatenate(xarr[simu_var]), index = utc, name = simu_var)
    sensor_Series = pd.Series(data = np.concatenate(xarr[var]), index = utc, name = var).dropna()
    #sampling frequencies
    fs_sensor = 1/ sensor_Series.index.diff()[2].total_seconds()
    fs_simu = 1/ simu_Series.index.diff()[2].total_seconds()
    hf_sensor = 0.45*fs_sensor if hf > fs_sensor else hf

    
    #frequency filtering 
    lf_simu_val, wf_simu_val, hf_simu_val = FilterData(simu_Series, lf, hf, fs_simu) 
    lf_sensor_val, wf_sensor_val, hf_sensor_val = FilterData(sensor_Series, lf, hf_sensor,fs_sensor) 
    
    #envrionmental conditions
    Hs = xarr.hm0.values[:,0] if 'hm0' in xarr.data_vars else np.nan
    Tp = xarr.tp.values[:,0] if 'tp' in xarr.data_vars else np.nan 
    wspd = xarr.AI_WindSpeed.values[:,0] if 'AI_WindSpeed' in xarr.data_vars else np.nan  
    wdir= xarr.WindDir.values[:,0] if 'WindDir' in xarr.data_vars else np.nan 
    CurrDir = xarr.CurrDir.values[:,0] if 'CurrDir' in xarr.data_vars else np.nan  
    CurrSpeed= xarr.CurrSpeed.values[:,0] if 'CurrSpeed' in xarr.data_vars else np.nan   
    
    env_dataframe = pd.DataFrame(data = {'Hs':Hs,'Tp':Tp,\
                                         'wspd':wspd, 'wdir':wdir, \
                                         'CurrDir':CurrDir, 'CurrSpeed':CurrSpeed}, index = xarr.time.values)
    env_dataframe = env_dataframe.resample(sampling).ffill()
    
    #generate the full frequency 
    #@ reviewer: do we wanna add the channel name to the dataframe collumns? 
    data = pd.DataFrame(data = {'simu': simu_Series, 'sensor': sensor_Series,\
                                'simu_lf': lf_simu_val, 'sensor_lf': lf_sensor_val,\
                                'simu_wf': wf_simu_val, 'sensor_wf': wf_sensor_val,\
                                'simu_hf': hf_simu_val, 'sensor_hf': hf_sensor_val,\
                                })
    data = data.resample(sampling).mean()
    
    return(pd.merge(data, env_dataframe, left_index=True, right_index=True, how='outer'))
    

def EnhancedBoxPlot(stats: pd.DataFrame, channel: str):
    
    cols = stats.columns
    cols = cols.drop(['Hs', 'Tp', 'wspd', 'wdir', 'CurrDir','CurrSpeed'])
    
    fig, ax = plt.subplots()
    green_diamond = dict(markerfacecolor='g', marker='D')
    bp = ax.boxplot(stats[['simu', 'sensor', 'simu_lf', 'sensor_lf', 'simu_wf', 'sensor_wf', 'simu_hf', 'sensor_hf']].values, patch_artist=True, flierprops=green_diamond)
    _ = plt.xticks([1,2,3,4,5,6,7,8],cols, rotation = 45)
    
    
    colors = ['tab:blue','tab:orange','tab:blue','tab:orange','tab:blue','tab:orange','tab:blue','tab:orange']
    
    boxplot = stats.boxplot(column = ['simu', 'sensor', 'simu_lf', 'sensor_lf', 'simu_wf', 'sensor_wf', 'simu_hf', 'sensor_hf'], 
                       ax = ax, patch_artist=True, flierprops=green_diamond, return_type='dict')
    
    _ = plt.xticks([1,2,3,4,5,6,7,8],cols, rotation = 45)
    
    for box, color in zip(boxplot['boxes'], colors):
        box.set(color=color, linewidth=2)
    """    
    for whisker, color in zip(boxplot['whiskers'], colors * 2):  # Whiskers are doubled
        whisker.set(color=color, linewidth=1.5)
    for cap, color in zip(boxplot['caps'], colors * 2):  # Caps are doubled
        cap.set(color=color, linewidth=1.5)
    for median, color in zip(boxplot['medians'], colors):
        median.set(color=color, linewidth=2)
    
    """
    plt.title(channel + '\n boxplot by frequency interval', fontsize = 20)
    plt.ylabel(channel, fontsize = 16)
    
    return(fig, ax)

def ScatterStats(xarr:xr.Dataset, var:str, sampling: str = '1s', x:str = 'n/a', max_pitch = 1, grp_thresphod = 5):
    simu_var = 'simu_'+var
    #generate UTC time index
    utc = []
    for time in xarr['time']:
        global_time = xarr.sel(time = time)['time'].values + pd.to_timedelta(xarr.sel(time = time)['time_sensor'], unit = 's').round('ms').values
        utc.append(global_time)
    utc = np.concatenate(utc)       
       
    """
    df= xarr.to_dataframe().reset_index()
    df.index = df['time'] + pd.to_timedelta(df['time_sensor'], unit = 's').round('L')
    df = df[df.apply(lambda x: x.isnumeric())]
    df2 = df.ffill()

    df2 = df2.resample('1s').mean()
    df = df[df['V_GridRealPowerLog'] > 100000].dropna()
    df['V_GridRealPowerLog'].head(20)
    """
    
    #Convert to pd.Series
    simu_Series = pd.Series(data = np.concatenate(xarr[simu_var]), index = utc, name = simu_var)
    sensor_Series = pd.Series(data = np.concatenate(xarr[var]), index = utc, name = var).dropna()
    sensor_wspd = pd.Series(data = np.concatenate(xarr['AI_WindSpeed']), index = utc, name = var).dropna()
    simu_wspd = pd.Series(data = np.concatenate(xarr['simu_AI_WindSpeed']), index = utc, name = var).dropna()
    sensor_wdir = pd.Series(data = np.concatenate(xarr['V_ST_TrueWindDir']), index = utc, name = var).dropna()
    rotor_rpm = pd.Series(data = np.concatenate(xarr['V_RotorRpm']), index = utc, name = var).dropna()
    simu_rotor_rpm = pd.Series(data = np.concatenate(xarr['simu_V_RotorRpm']), index = utc, name = var).dropna()
    sensor_rotor_pwr = pd.Series(data = np.concatenate(xarr['V_GridRealPowerLog']), index = utc, name = var).dropna()
    simu_rotor_pwr = pd.Series(data = np.concatenate(xarr['simu_V_GridRealPowerLog']), index = utc, name = var).dropna()
    sensor_rotor_pitch = pd.Series(data = np.concatenate(xarr['V_PitchAngle']), index = utc, name = var).dropna()
    simu_rotor_pitch = pd.Series(data = np.concatenate(xarr['simu_V_PitchAngle']), index = utc, name = var).dropna()
     
    
    #envrionmental conditions
    Hs = xarr.hm0.values[:,0] if 'hm0' in xarr.data_vars else np.nan
    Tp = xarr.tp.values[:,0] if 'tp' in xarr.data_vars else np.nan 
    wspd = xarr.AI_WindSpeed.values[:,0] if 'AI_WindSpeed' in xarr.data_vars else np.nan  
    wdir= xarr.WindDir.values[:,0] if 'WindDir' in xarr.data_vars else np.nan 
    CurrDir = xarr.CurrDir.values[:,0] if 'CurrDir' in xarr.data_vars else np.nan  
    CurrSpeed= xarr.CurrSpeed.values[:,0] if 'CurrSpeed' in xarr.data_vars else np.nan   

    data = pd.DataFrame(data = {'simu': simu_Series, 'sensor': sensor_Series,
                                'AI_WindSpeed':sensor_wspd, 'V_ST_TrueWindDir':sensor_wdir,
                                'simu_AI_WindSpeed': simu_wspd,
                                'V_RotorRpm':rotor_rpm,'simu_V_RotorRpm':simu_rotor_rpm,
                                'V_PitchAngle':sensor_rotor_pitch ,'simu_V_PitchAngle':simu_rotor_pitch,
                                'V_GridRealPowerLog':sensor_rotor_pwr,'simu_V_GridRealPowerLog':simu_rotor_pwr}).resample(sampling).mean()
    
    env_dataframe = pd.DataFrame(data = {'Hs':Hs,'Tp':Tp,\
                                         'wspd':wspd, 'wdir':wdir, \
                                         'CurrDir':CurrDir, 'CurrSpeed':CurrSpeed}, index = xarr.time.values)
    env_dataframe = env_dataframe.resample(sampling).ffill()
    
    #merge data and data cleaning 
    merged = pd.merge(data, env_dataframe, left_index=True, right_index=True, how='outer')
    #cleaning SCADA data 
    merged =  merged[merged['V_GridRealPowerLog'] > 100000]
    pitch_condition = merged['V_PitchAngle'] < max_pitch
    power_condition = merged['V_GridRealPowerLog'] > 0.95* 2000000
    merged = merged[pitch_condition | power_condition]
    
    #Cleaning SIMA data 
    merged =  merged[merged['simu_V_GridRealPowerLog'] > 100000]   
    pitch_condition = merged['simu_V_PitchAngle'] < max_pitch
    power_condition = merged['simu_V_GridRealPowerLog'] > 0.9* 2000000
    merged = merged[pitch_condition | power_condition]
  
    
    merged ['AI_WindSpeed'] = merged['AI_WindSpeed'].round(1)
    merged ['simu_AI_WindSpeed'] = merged['simu_AI_WindSpeed'].round(1)
    
    #average power production 
    mean_merged = merged.groupby(by=['AI_WindSpeed']).mean()
    std_merged = merged.groupby(by=['AI_WindSpeed']).std()
    simu_mean_merged = merged.groupby(by=['simu_AI_WindSpeed']).mean()
    simu_std_merged = merged.groupby(by=['simu_AI_WindSpeed']).std()
    
     
      
    fig, ax = plt.subplots(2,3, figsize = (36*0.3,24*0.3))
    #scatter mesures
    ax[0,0].scatter(merged['AI_WindSpeed'].values, merged['V_GridRealPowerLog'].values, c = merged['V_RotorRpm'].values, alpha = 0.1)
    ax[0,0].plot(mean_merged['V_GridRealPowerLog'])
    ax[0,0].fill_between(mean_merged.index, mean_merged['V_GridRealPowerLog']- std_merged['V_GridRealPowerLog'], mean_merged['V_GridRealPowerLog']+ std_merged['V_GridRealPowerLog'], alpha=0.7)
    #scatter simu
    ax[0,1].scatter(merged['simu_AI_WindSpeed'].values, merged['simu_V_GridRealPowerLog'].values, c = merged['simu_V_RotorRpm'].values, alpha = 0.1)
    ax[0,1].plot(simu_mean_merged['simu_V_GridRealPowerLog'], color = 'tab:orange')
    ax[0,1].fill_between(simu_mean_merged.index, simu_mean_merged['simu_V_GridRealPowerLog']- simu_std_merged['simu_V_GridRealPowerLog'], simu_mean_merged['simu_V_GridRealPowerLog']+ simu_std_merged['simu_V_GridRealPowerLog'], alpha=0.7, color = 'tab:orange')
    # simu vs measures wspd vs power 
    ax[0,2].plot(mean_merged['V_GridRealPowerLog'], color = 'tab:blue', label = 'SCADA')
    ax[0,2].fill_between(mean_merged.index, mean_merged['V_GridRealPowerLog']- std_merged['V_GridRealPowerLog'], mean_merged['V_GridRealPowerLog']+ std_merged['V_GridRealPowerLog'], alpha=0.7, color = 'tab:blue')
    ax[0,2].plot(simu_mean_merged['simu_V_GridRealPowerLog'], color = 'tab:orange', label = 'SIMA')
    ax[0,2].fill_between(simu_mean_merged.index, simu_mean_merged['simu_V_GridRealPowerLog']- simu_std_merged['simu_V_GridRealPowerLog'], simu_mean_merged['simu_V_GridRealPowerLog']+ simu_std_merged['simu_V_GridRealPowerLog'], alpha=0.7, color = 'tab:orange')
    
    
    ax[0,0].set_xlabel('Wind speed [m/s]')
    ax[0,1].set_xlabel('Wind speed [m/s]')
    ax[0,2].set_xlabel('Wind speed [m/s]')    

    ax[0,0].set_ylabel('Power production [W]')
    ax[0,1].set_ylabel('Power production [W]')
    ax[0,2].set_ylabel('Power production [W]')

    ax[0,0].set_title('Scada')
    ax[0,1].set_title('SIMA')
    ax[0,2].set_title('Scada vs SIMA')
    ax[0,2].legend()
    
    #Cp et TSR 
    merged = merged[merged['V_PitchAngle'] < max_pitch]
    merged = merged[merged['simu_V_PitchAngle'] < max_pitch]

    merged['V_Rotorrads'] = merged['V_RotorRpm'] * 2*np.pi / 60
    merged['TSR'] = 41*merged['V_Rotorrads'] / merged['AI_WindSpeed']
    merged ['TSR'] = merged['TSR'].round(1)
    merged['Cp'] = (2*merged['V_GridRealPowerLog']) / (1.29*5281*merged['AI_WindSpeed']**3)
    
    merged['simu_V_Rotorrads'] = merged['simu_V_RotorRpm'] * 2*np.pi / 60
    merged['simu_TSR'] = 41*merged['simu_V_Rotorrads'] / merged['simu_AI_WindSpeed']
    merged ['simu_TSR'] = merged['simu_TSR'].round(1)
    merged['simu_Cp'] = (2*merged['simu_V_GridRealPowerLog']) / (1.29*5281*merged['simu_AI_WindSpeed']**3)
    
    tsr_group = merged.groupby(by=['TSR'])
    tsr_group = tsr_group.filter(lambda x: len(x) >= grp_thresphod)
    
    mean_tsr = tsr_group.groupby(by=['TSR']).mean()
    std_tsr = tsr_group.groupby(by=['TSR']).std()
    #std_tsr = merged.groupby(by=['TSR']).std()
    
    simu_tsr_group = merged.groupby(by=['simu_TSR'])
    simu_tsr_group = simu_tsr_group.filter(lambda x: len(x) >= grp_thresphod)

    simu_mean_tsr = simu_tsr_group.groupby(by=['simu_TSR']).mean()
    simu_std_tsr = simu_tsr_group.groupby(by=['simu_TSR']).std()   

    merged.plot.scatter('TSR', 'Cp', c = 'V_PitchAngle', alpha = 0.1, ax = ax[1,0])
    ax[1,0].plot(mean_tsr['Cp'])
    ax[1,0].fill_between(mean_tsr.index, mean_tsr['Cp']- std_tsr['Cp'], mean_tsr['Cp']+ std_tsr['Cp'], alpha=0.7)
    #scatter simu
    merged.plot.scatter('simu_TSR', 'simu_Cp', c = 'V_PitchAngle', alpha = 0.1, ax = ax[1,1])
    ax[1,1].plot(simu_mean_tsr['simu_Cp'], color = 'tab:orange')
    ax[1,1].fill_between(simu_mean_tsr.index, simu_mean_tsr['simu_Cp']- simu_std_tsr['simu_Cp'], simu_mean_tsr['simu_Cp']+ simu_std_tsr['simu_Cp'], alpha=0.7, color = 'tab:orange')
    # overlay 
    ax[1,2].plot(mean_tsr['Cp'], label = 'SCADA')
    ax[1,2].fill_between(mean_tsr.index, mean_tsr['Cp']- std_tsr['Cp'], mean_tsr['Cp']+ std_tsr['Cp'], alpha=0.7)
    ax[1,2].plot(simu_mean_tsr['simu_Cp'], color = 'tab:orange', label = 'SIMA')
    ax[1,2].fill_between(simu_mean_tsr.index, simu_mean_tsr['simu_Cp']- simu_std_tsr['simu_Cp'], simu_mean_tsr['simu_Cp']+ simu_std_tsr['simu_Cp'], alpha=0.7, color = 'tab:orange')
     
    
    ax[1,0].set_xlabel('TSR [-]')
    ax[1,1].set_xlabel('TSR [-]')
    ax[1,2].set_xlabel('TSR [-]')    

    ax[1,0].set_ylabel('Cp [-]')
    ax[1,1].set_ylabel('Cp [-]')
    ax[1,2].set_ylabel('Cp [-]')

    ax[1,0].set_title('Scada')
    ax[1,1].set_title('SIMA')
    ax[1,2].set_title('Scada vs SIMA')
    ax[1,2].legend()    
    
    return fig

def PairGridGenerator(xarray,  abs_val = True):
    #convert to dataframe
    dataframe = xarray.to_dataframe()

    #generates delta variables 
    dataframe['Delta_mean']    = (dataframe.Simu_mean - dataframe.Sensor_mean) #/ dataframe.Sensor_mean
    dataframe['Delta_mean_lf'] = (dataframe.Simu_mean_lf - dataframe.Sensor_mean_lf)#/ dataframe.Sensor_mean_lf
    dataframe['Delta_mean_wf'] = (dataframe.Simu_mean_wf - dataframe.Sensor_mean_wf)#/ dataframe.Sensor_mean_wf
    dataframe['Delta_mean_hf'] = (dataframe.Simu_mean_hf - dataframe.Sensor_mean_hf)#/ dataframe.Sensor_mean_hf

    dataframe['Delta_std']    = (dataframe.Simu_std - dataframe.Sensor_std) #/ dataframe.Sensor_std
    dataframe['Delta_std_lf'] = (dataframe.Simu_std_lf - dataframe.Sensor_std_lf)# / dataframe.Sensor_std_lf
    dataframe['Delta_std_wf'] = (dataframe.Simu_std_wf - dataframe.Sensor_std_wf)# / dataframe.Sensor_std_wf
    dataframe['Delta_std_hf'] = (dataframe.Simu_std_hf - dataframe.Sensor_std_hf) #/ dataframe.Sensor_std_hf

    dataframe['Delta_CV']    = dataframe['Delta_std'] / dataframe['Delta_mean']
    dataframe['Delta_CV_lf'] = dataframe['Delta_std_lf'] / dataframe['Delta_mean_lf']
    dataframe['Delta_CV_wf'] = dataframe['Delta_std_wf'] / dataframe['Delta_mean_wf']
    dataframe['Delta_CV_hf'] = dataframe['Delta_std_hf'] / dataframe['Delta_mean_hf']

    if abs_val == True:
        var_lst = ['Delta_mean','Delta_mean_lf','Delta_mean_wf','Delta_mean_hf','Delta_std','Delta_std_lf','Delta_std_wf','Delta_std_hf']
        for var in var_lst:
            dataframe[var] = np.abs(dataframe[var])

    #now plot mean values 
    x_vars = ["hs", "tp", "cur", "mag10"]
    y_vars = ['Delta_mean','Delta_mean_lf','Delta_mean_wf','Delta_mean_hf']

    g = sns.PairGrid(dataframe, x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot)

    #nice plot 
    xlabels = ['hs','tp','current','max10']
    ylabels = [r'$\Delta$ mean',r'$\Delta$ mean $_{lf}$',r'$\Delta$ mean $_{wf}$',r'$\Delta$ mean $_{hf}$']
    for i in range(len(xlabels)):
        g.axes[0,i].xaxis.set_label_text(xlabels[i])
    for i in range(len(ylabels)):
        g.axes[i,0].yaxis.set_label_text(ylabels[i])
    plt.suptitle('Error on: '+xarray.attrs['variable']+' average value', fontsize = 20)

    
    dataframe_no_nan = dataframe.fillna(0)
    i = 0
    for x in x_vars:
        j = 0 
        for y in y_vars:
                fig = plt.figure()
                corr = plt.xcorr(dataframe_no_nan[x], dataframe_no_nan[y])
                plt.close(fig)
                #print('Correlation between:',x,'and',y,':',np.nanmax(corr[1]))
                g.axes[j,i].set_title('cross corr.= '+str(np.round(np.nanmax(corr[1]),3)), fontsize = 10)
                j = j+1
        i = i+1 
    plt.tight_layout()

    #now plot standard deviation 
    x_vars = ["hs", "tp", "cur", "mag10"]
    y_vars = ['Delta_std','Delta_std_lf','Delta_std_wf','Delta_std_hf']
    plt.figure()
    g = sns.PairGrid(dataframe, x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot)
    #nice plot 
    xlabels = ['hs','tp','current','max10']
    ylabels = [r'$\Delta$ std',r'$\Delta$ std $_{lf}$',r'$\Delta$ std $_{wf}$',r'$\Delta$ std $_{hf}$']
    for i in range(len(xlabels)):
        g.axes[0,i].xaxis.set_label_text(xlabels[i])
    for i in range(len(ylabels)):
        g.axes[i,0].yaxis.set_label_text(ylabels[i])
    plt.suptitle('Error on: '+xarray.attrs['variable']+' standard deviation', fontsize = 20)

    i = 0 
    for x in x_vars:
        j = 0 
        for y in y_vars:
                fig = plt.figure()
                corr = plt.xcorr(dataframe_no_nan[x], dataframe_no_nan[y])
                plt.close(fig)
                #print('Correlation between:',x,'and',y,':',np.nanmax(corr[1]))
                g.axes[j,i].set_title('cross corr. = '+str(np.round(np.nanmax(corr[1]),3)), fontsize = 10)
                j = j+1
        i = i+1 
    plt.tight_layout()

    #now plot variation coefficient 
    x_vars = ["hs", "tp", "cur", "mag10"]
    y_vars = ['Delta_CV','Delta_CV_lf','Delta_CV_wf','Delta_CV_hf']
    plt.figure()
    g = sns.PairGrid(dataframe, x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot)
    #nice plot 
    xlabels = ['hs','tp','current','max10']
    ylabels = [r'$\Delta$ cov',r'$\Delta$ cov $_{lf}$',r'$\Delta$ cov $_{wf}$',r'$\Delta$ cov $_{hf}$']
    for i in range(len(xlabels)):
        g.axes[0,i].xaxis.set_label_text(xlabels[i])
    for i in range(len(ylabels)):
        g.axes[i,0].yaxis.set_label_text(ylabels[i])
    plt.suptitle('Error on: '+xarray.attrs['variable']+' coefficient of variation', fontsize = 20)

    i = 0 
    for x in x_vars:
        j = 0 
        for y in y_vars:
                fig = plt.figure()
                corr = plt.xcorr(dataframe_no_nan[x], dataframe_no_nan[y])
                plt.close(fig)
                #print('Correlation between:',x,'and',y,':',np.nanmax(corr[1]))
                g.axes[j,i].set_title('cross corr. = '+str(np.round(np.nanmax(corr[1]),3)), fontsize = 10)
                j = j+1
        i = i+1 
    plt.tight_layout()



if __name__ == '__main__':
    DemoSATH = load_simu_sensor_xr_dataset()
    #frequency statistics 
    stats = XarrStats(DemoSATH,'V_MRU_Heading')
    EnhancedBoxPlot(stats,'V_MRU_Heading')

    #power production scatterplots 
    DemoSATH = load_simu_sensor_xr_dataset(year = '2023', month = '12', day = '04')
    ScatterStats(DemoSATH, 'V_GridRealPowerLog', sampling = '10 min')