#!/usr/bin/env python
"""Provides classes for the detection, classification and analysis of dentate spikes (DSs).
'signal' defines functions for reading, resampling and plotting the local field potential (LFP).
'detection' defines functions to detect DSs.
'classification' defines the CSDbC and WFbC methods, and functions to compute classification metrics (e.g., precision and recall).
'analysis' defines functions for computing the CSD, the DS width and the DS rate relative to delta power.
"""

__author__ = "Rodrigo Santiago"
__copyright__ = "Copyright 2023, Tort Lab"
__credits__ = "Rodrigo Santiago, Adriano Tort, Vitor Lopes-dos-Santos, Cesar Renno-Costa, Diego Laplagne"
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Rodrigo Santiago"
__email__ = "rsantiago@neuro.ufrn.br"
__status__ = "Production"


class signal():
    
    def getLFP(session_name, orderByXML=True, doubleLinear=True, probe=1):
        """Reads LFP data from EEG file based on XML metadata.
        This particular function was used for loading the LFP data published
        by Yuta Senzai and György Buzsáki. Data available at 
        https://buzsakilab.nyumc.org/datasets/SenzaiY/YutaMouse21/YutaMouse21-140611/.
        
        Parameters
        ----------
        session_name : string
            The name/number that identifies the recording session.
        orderByXML : bool, default=True
            If True (default), defines the channel order according to the XML file.
            If False, channels are sorted in ascending order according to their numbering.
        doubleLinear : bool, default=True
            Defines whether the probe comprises one linear array (False) or two (True).
        probe : int, default=1
            The number of the linear electrode array (1 or 2). The first array is the
            default option.
        
        Returns
        -------
        LFP : ndarray
            The LFP matrix (data samples by channel)
        srate : float
            The sampling rate (in samples per second)
        """
        import numpy as np
        import xml.etree.ElementTree as ET

        # gets parameters (XML file)
        filepath = session_name + ".xml"
        tree     = ET.parse(filepath)
        root     = tree.getroot()

        for child in root.findall('fieldPotentials'):
            srate = float(child.find('lfpSamplingRate').text) # sampling rate

        for child in root.findall('acquisitionSystem'):
            nCh = int(child.find('nChannels').text) # total number of channels

        # gets LFP
        inputfilename = session_name+'.eeg'

        with open(inputfilename,'rb') as fid:
            data_array = np.fromfile(fid,np.int16)

        LFP = np.reshape(data_array,(int(len(data_array)/nCh),nCh)).transpose()

        # gets channel order according to the anatomical description in the XML file
        if orderByXML:
            ch_order = [int(channel.text) for child in root.findall('anatomicalDescription')
                        for child2 in child.find('channelGroups')
                        for channel in child2.findall('channel')]

        # chooses one of the double linear arrays of the probe
        if doubleLinear:
            nCh      = int(len(LFP)/2)            # number of channels in one linear probe
            ch_order = ch_order[probe-1::2][:nCh] # new channel order
            LFP      = LFP[ch_order]              # LFP with reordered channels

        return LFP, srate
    
    def resampleLFP(LFP, srate, new_srate, njobs=-1):
        """Resamples LFP based on a new sampling rate. First, the function checks
        whether the input LFP data is 1D (single channel) or 2D (multiple channels)
        and then resamples it accordingly. If it's 1D, it resamples the entire signal.
        If it's 2D, it resamples each channel in parallel using the joblib library for
        improved performance. Finally, it calculates and prints the elapsed time for
        the processing in minutes.

        Parameters
        ----------
        LFP : ndarray
            LFP matrix (channels by samples)
        srate : float
            Current sampling rate.
        new_srate : float
            New sampling rate.
        njobs: int, default=-1
            Number of jobs for parallel processing when LFP data contains more than one
            channel. The default value (-1) recruits all available processor cores.
        
        Returns
        -------
        LFP_resampled : ndarray
            Resampled LFP matrix.
        """
        import numpy as np
        from time import time
        from joblib import Parallel, delayed
        from scipy.signal import resample

        tic = time() # Record the current time as 'tic' for measuring elapsed time later
        
        # Check the dimensionality of the input LFP data
        if len(np.squeeze(LFP).shape) == 1:
            # If LFP is 1D (a single channel), perform resampling
            new_length    = int(len(LFP)/(srate/new_srate)) # Calculate the new length for resampling
            LFP_resampled = resample(LFP,new_length)        # Resample the LFP data to the new length

        elif len(np.squeeze(LFP).shape) == 2:
            # If LFP is 2D (multiple channels), perform resampling for each channel in parallel
            new_length    = int(len(LFP[0])/(srate/new_srate)) # Calculate the new length for resampling
            # Use Parallel and delayed for parallel processing of resampling across channels
            LFP_resampled = np.reshape(Parallel(n_jobs=njobs)
                                       (delayed(resample)(LFP[ch],new_length) 
                                        for ch in range(len(LFP))),(len(LFP),new_length))
        
        toc = time() # Record the current time as 'toc' after the processing is done
        # Calculate and print the elapsed time in minutes
        print("Processing elapsed time: "+str(np.round((toc-tic)/60.,2))+" min.")

        return LFP_resampled
    
    def plotLFP(time_vector, LFP, startT, windowT, lfpshift=1.1, srate=1e3, linew=1.25):
        """Plots LFP signals.
        
        Parameters
        ----------
        time_vector : ndarray
            Array with all time points.
        LFP : ndarray
            LFP matrix (channels by samples)
        startT : float,int
            Start time for plotting (in seconds).
        windowT : float, int
            Window size (in seconds).
        lfpshift : float, default=1.1
            Vertical distance between channels (in mV).
        srate : float, default=1000.0
            Sampling rate (samples per second).
        linew: float, default=1.25
            Plotting line width.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # indexes of observation time interval
        startID  = int(startT*srate)  # start time index
        windowID = int(windowT*srate) # number of indexes that give the size of the time window

        if len(np.squeeze(LFP).shape) == 1:
            plt.plot(time_vector[startID:startID+windowID],
                     LFP[startID:startID+windowID],'k',linewidth=linew)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.autoscale(enable=True, axis='y', tight=True)
            plt.xticks(np.linspace(startT,startT+windowT,9),size=15)
            plt.yticks(size=15)
            plt.xlabel('Time (s)',size=15)
            plt.ylabel('Amplitude (mV)',size=15)
            plt.tight_layout();

        else:
            nCh = len(LFP) # number of channels
            for ch in range(nCh):
                plt.plot(time_vector[startID:startID+windowID],
                         LFP[ch,startID:startID+windowID]-lfpshift*ch,'k',linewidth=linew)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.autoscale(enable=True, axis='y', tight=True)
            plt.xticks(np.linspace(startT,startT+windowT,9),size=15)
            plt.yticks(np.linspace(0,-lfpshift*ch,nCh),np.arange(nCh)+1,size=15)
            plt.xlabel('Time (s)',size=16)
            plt.ylabel('Channel',size=16)
            plt.tight_layout();
            
class analysis():
            
    def runCSD(LFPs):
        """Computes the current source density (CSD) for the local field potentials
        (LFPs) given. The CSDs of the extreme channels are null (zero).
        
        Parameters
        ----------
        LFPs : ndarray
            LFP matrix (channels by samples)
        
        Returns
        -------
        CSD : ndarray
            Current source density matrix.
        """
        import numpy as np

        nCh = len(LFPs) # number of channels
        
        # CSD matrix initialization
        if len(np.shape(LFPs)) == 1:
            CSD = np.zeros(nCh)
        else:
            nSamples = np.size(LFPs,1)
            CSD = np.zeros((nCh,nSamples))
        
        # CSD matrix computation
        for chi in range(1,nCh-1):
            CSD[chi] = -(LFPs[chi-1]-2*LFPs[chi]+LFPs[chi+1])

        return CSD

    def CSDmatrix(LFP, DS_ind, srate=1e3, csd_win_t=0.4):
        """Computes the current source density (CSD) for the local field potentials
        (LFPs) given based on dentate spike indexes. The CSDs of the extreme channels
        are null (zero).
        
        Parameters
        ----------
        LFPs : ndarray
            LFP matrix (channels by samples)
        DS_ind : ndarray
            Indexes of the detected events in the LFP signal.
        srate : float, default=1000.0
            Sampling rate (samples per second).
        csd_win_t : float, default=0.4
            Size of the time window centered on the DS peak used to compute the CSD
            (in seconds).
        
        Returns
        -------
        CSD : ndarray
            Current source density matrix.
        csd_t : ndarray
            Time vector for plotting the CSD.
        csd_ch_list : ndarray
            Channel numbers for plotting the CSD.
        """
        import numpy as np

        nCh          = len(LFP)             # number of channels
        csd_win_nInd = int(srate*csd_win_t) # number of indices of the time window
        csd_ch_list  = np.arange(nCh)       # list of channel numbers
        
        # CSD matrix initialization
        CSD = np.zeros((len(DS_ind),len(csd_ch_list),csd_win_nInd+1))
        
        # CSD time vector (in ms)
        csd_t = np.linspace(-csd_win_t/2,csd_win_t/2,csd_win_nInd+1)*1e3
        
        # CSD matrix computation for each DS
        for csd_DS_ind,DSindex in enumerate(DS_ind):
            csd_win_ind = np.linspace(DSindex-(csd_win_nInd/2),DSindex+(csd_win_nInd/2),
                                      csd_win_nInd+1,dtype=int)
            for ch in csd_ch_list[1:-1]:
                V_ch_above = np.array(LFP[ch-1][csd_win_ind])
                V_ch_below = np.array(LFP[ch+1][csd_win_ind])
                V_ch       = np.array(LFP[ch][csd_win_ind])
                CSD[csd_DS_ind,ch]   = 2*V_ch - V_ch_above - V_ch_below
                    
        CSD = np.squeeze(CSD)

        return CSD, csd_t, csd_ch_list
    
    def plotDSCSD(csd_t, csd_ch_list, CSD, title, t_waveform, DS_indexes, LFP, srate=1e3, colormap='RdBu_r', lenInTitle=True, amp_m=0.6, v_min=-1., v_max=1., ch_shift=1, winsize=400):
        """Plots the the LFP of each channel and the CSD triggered at the DS peak.
        
        Parameters
        ----------
        csd_t : ndarray
            Time vector for plotting the CSD.
        csd_ch_list : ndarray
            Channel numbers for plotting the CSD.
        CSD : ndarray
            Current source density matrix.
        title : str
            The title of the plot.
        t_waveform : ndarray
            Time vector of the DS waveform (in milliseconds from DS peak).
        DS_indexes : ndarray
            Indexes of the detected events in the LFP signal.
        LFP : ndarray
            LFP matrix (channels by samples)
        srate : float, default=1000.0
            Sampling rate (samples per second).
        colormap : str, default='RdBu_r'
            Matplotlib colormap name.
        lenInTitle : bool, default=True
            If True, the number of events is displayed in the second line of the
            title.
        amp_m : float, default=0.6
            Amplitude multiplier for plotting LFPs.
        v_min, v_max : float, default=-1., 1.
            Equivalent to Matplotlib 'vmin' and 'vmax', they define the data range
            that the colormap covers.
        ch_shift : int, default=1
            Sets the offset for the position of the 'y' tick labels.
        winsize : int or float, default=400
            Size of the time window centered on the DS peak used to compute the CSD
            (in milliseconds).
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np
        
        nCh = len(LFP) # number of channels

        yticklabels = [(str(digit+ch_shift)) for digit in np.arange(nCh)] # channel labels
        divnorm     = colors.TwoSlopeNorm(vmin=v_min,vcenter=0.,vmax=v_max) # colorbar levels
        
        # CSD plotting with the colorbar centered at zero.
        plt.contourf(csd_t,csd_ch_list,CSD,cmap=colormap,levels=100,norm=divnorm)
        plt.yticks(np.arange(nCh),yticklabels,size=13)
        plt.xticks(size=13)
        plt.xlabel('Time from DS peak (ms)',size=14)
        plt.ylabel('Channel',size=14)
        if lenInTitle: plt.title(title+'\n(#'+str(len(DS_indexes))+')',size=14)
        else: plt.title(title,size=14)
        plt.xlim(-winsize/2,winsize/2)
        plt.ylim(0,nCh-1)
        plt.colorbar();
        plt.gca().invert_yaxis();

        ind_win_ms = (t_waveform>=-winsize/2)*(t_waveform<=winsize/2) # time window indexes
        n_half_ind = int((winsize/2)*srate/1e3)                       # number of indexes of the half window
        
        # LFP plotting
        for ch in range(nCh):
            LFP_DS_ch = [LFP[ch,DS_index-n_half_ind:DS_index+n_half_ind+1] for DS_index in DS_indexes]
            plt.plot(t_waveform[ind_win_ms], np.mean(np.array(LFP_DS_ch),axis=0)*(-amp_m)+(ch), 'k');
        
    def plotConcCSD(CSD, title, colormap='RdBu_r', t_win=0, srate=1e3, vrange=None, colorbar=False):
        """Plots the CSD profiles of each event in a concatenated way.
        
        Parameters
        ----------
        CSD : ndarray
            Current source density matrix.
        title : str
            The title of the plot.
        colormap : str, default='RdBu_r'
            Matplotlib colormap name.
        t_win : int or float, default=0
            Time window around DS peak for each CSD (in milliseconds).
        srate : float, default=1000.0
            Sampling rate (samples per second).
        vrange : list, tuple or None. default=None
            Defines the value range of the colorbar, i.e., the Matplotlib parameter
            'halfrange'. The first element is a string defining the measure to be
            applied: 'percentile' or 'std' (standard deviation). The second element
            is the related value. If None, the range is [-3, 3].
        colorbar : bool, default=False
            If True, displays the colorbar.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np
        from sklearn.preprocessing import minmax_scale

        t_win      = int(t_win*srate/1e3)     # time window around DS peak (in number of samples)
        nCh        = len(CSD[1])              # number of channelzs
        middle_ind = int(np.shape(CSD)[-1]/2) # window center index
        
        # Concatenated CSDs
        CSD_stack  = np.hstack(CSD[:,:,middle_ind-int(t_win/2):middle_ind+int(t_win/2)+1])
        
        # Plot
        if vrange == None:
            plt.contourf(CSD_stack,cmap=colormap,levels=100,norm=colors.CenteredNorm(),vmin=-3,vmax=3)
        elif vrange[0] == 'percentile':
            plt.contourf(CSD_stack, cmap=colormap, levels=100,
                         norm=colors.CenteredNorm(halfrange=np.percentile(CSD_stack,vrange[1])))
        elif vrange[0] == 'std':
            plt.contourf(CSD_stack, cmap=colormap, levels=100,
                         norm=colors.CenteredNorm(halfrange=vrange[1]*np.std(CSD_stack)))
        
        plt.yticks(np.arange(nCh),np.arange(nCh)+1,size=14)
        plt.xticks((int(t_win/2),np.arange(int(t_win/2),(t_win+1)*len(CSD),(t_win+1))[-1]),
                   (1,len(CSD)),size=14)
        plt.xlabel('Event',size=15)
        plt.ylabel('Channel',size=15)
        plt.title(title,size=16)
        plt.ylim(0,nCh-1)
        if colorbar: plt.colorbar();
        plt.gca().invert_yaxis();
    
    def DSpikeWidths(t_waveform, DS_waveforms, start_range=[-15,-5], end_range=[5,15]):
        """Computes the width of the mean dentate spike (DS) waveform based on its
        second derivative. The width consists of the temporal distance between the
        second upward concavities before and after the DS peak. The concavities are
        identified from the positive values of the second derivative of the mean
        waveform: the start width limit corresponds to the timestamp of the local
        maximum of the second derivative between −15 and −5 ms, while the end width
        limit is defined as the local maximum between 5 and 15 ms.

        Parameters
        ----------
        t_waveform : ndarray
            Time vector of the DS waveform (in milliseconds from DS peak).
        DS_waveforms : ndarray
            Waveforms of all dentate spikes of all groups.
        start_range : list or ndarray, default==[-15,-5]
            Tuple containing the time range (in milliseconds from DS peak) for
            detecting the start width limit.
        end_range : list or ndarray, default=[5,15]
            Tuple containing the time range (in milliseconds from DS peak) for
            detecting the end width limit.
        
        Returns
        -------
        DS_width_all : ndarray
            Widths of the mean dentate spikes of each group/channel (in milliseconds).
        DS_mean_dd_all : ndarray
            Second derivatives of the mean waveforms of the the DS of each group/channel.
        DS_peak_mask_all : ndarray
            Boolean arrays as the index masks of the time array indicating the
            limits of the DS widths.
        """
        import numpy as np
        from scipy.interpolate import interp1d
        from scipy.signal import find_peaks

        DS_width_all     = [] # DS widths per channel/group
        DS_mean_dd_all   = [] # 2nd derivatives of the mean DS waveforms
        DS_peak_mask_all = [] # index masks of the selected peaks

        t_waveform_new = np.linspace(-200,200,1601) # new time vector (new sampling rate: 4 kHz)
        
        dt = t_waveform_new[1]-t_waveform_new[0] # spacing between values (sampling period)

        if len(np.shape(DS_waveforms)) == 2: # for one channel/group
            DS_waveforms = [DS_waveforms]

        for wf in DS_waveforms:
            if type(np.mean(wf,axis=0))==np.ndarray and len(wf)>0:
                DS_mean    = np.mean(wf,axis=0)                          # mean DS waveform
                f          = interp1d(t_waveform, DS_mean, kind='cubic') # interpolation function
                DS_mean    = f(t_waveform_new)                           # use interpolation function returned by `interp1d`
                DS_mean_dd = np.gradient(np.gradient(DS_mean,dt),dt)     # second derivative of the interpolated mean waveform
                
                # Find peaks of the 2nd derivative signal within the start range (−15 to −5 ms by default)
                id_peaks,_ = find_peaks(DS_mean_dd[(t_waveform_new>=start_range[0])*(t_waveform_new<=start_range[1])], height=0)
                # Index of the highest peak
                id_peak = np.argmax(DS_mean_dd[(t_waveform_new>=start_range[0])*(t_waveform_new<=start_range[1])][id_peaks])
                # Start width limit (in ms)
                t_peak1 = t_waveform_new[id_peaks[id_peak]+len(DS_mean_dd[t_waveform_new<start_range[0]])]

                # Find peaks of the 2nd derivative signal within the end range (5 to 15 ms by default)
                id_peaks,_ = find_peaks(DS_mean_dd[(t_waveform_new>=end_range[0])*(t_waveform_new<=end_range[1])],height=0)
                # Index of the highest peak
                id_peak = np.argmax(DS_mean_dd[(t_waveform_new>=end_range[0])*(t_waveform_new<=end_range[1])][id_peaks])
                # End width limit (in ms)
                t_peak2 = t_waveform_new[id_peaks[id_peak]+len(DS_mean_dd[t_waveform_new<end_range[0]])]

                peak_mask = (t_waveform_new==t_peak1)+(t_waveform_new==t_peak2) # index mask of the width limits

                DS_width_all.append(t_peak2-t_peak1)
                DS_mean_dd_all.append(DS_mean_dd)
                DS_peak_mask_all.append(peak_mask)
            else:
                DS_width_all.append(np.nan)
                DS_mean_dd_all.append(np.array([np.nan]*len(t_waveform_new)))
                DS_peak_mask_all.append(np.array([False]*len(t_waveform_new)))
                
        # Transform list into array and suppress the first dimension
        DS_width_all     = np.squeeze(DS_width_all)
        DS_mean_dd_all   = np.squeeze(DS_mean_dd_all)
        DS_peak_mask_all = np.squeeze(DS_peak_mask_all)

        return DS_width_all, DS_mean_dd_all, DS_peak_mask_all


    def plotDSclass(DS_classes, DS_type1, DS_type2, t_waveform):
        """Plots the mean waveform of each DS type as well as the respective
        second derivative indicating the width limits.
        
        Parameters
        ----------
        DS_classes : ndarray
            Integer array with DS classes (zeros and ones).
        DS_type1 : ndarray
            DS1 waveforms.
        DS_type2 : list or ndarray
            DS2 waveforms.
        t_waveform : ndarray
            Time vector of the DS waveform (in milliseconds from DS peak).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        
        # Mean waveforms
        DS1_mean = np.mean(DS_type1,axis=0)
        DS2_mean = np.mean(DS_type2,axis=0)
        # Waveform standard deviations 
        DS1_std = np.std(DS_type1,axis=0)
        DS2_std = np.std(DS_type2,axis=0)

        plt.figure(figsize=(11,9))

        plt.subplot(221) ### DS1 waveform (mean and standard deviation)
        lower_bound = DS1_mean - DS1_std
        upper_bound = DS1_mean + DS1_std
        if type(DS1_mean) == np.ndarray:
            plt.fill_between(t_waveform,lower_bound,upper_bound,color='C2',alpha=0.3)
            plt.plot(t_waveform,DS1_mean,'C2',lw=3)
            plt.title('DS1 (#'+str(len(DS_type1))+')',size=15)
        else:
            plt.title('DS1 (#0)',size=15)
        plt.autoscale(enable=True,axis='x',tight=True)
        plt.xlabel('Time from DS peak (ms)',size=14)
        plt.ylabel('Amplitude (mV)',size=14)
        plt.xticks(size=13)
        plt.yticks(size=13)
        plt.grid(linestyle='dotted')

        plt.subplot(222) ### DS2 waveform (mean and standard deviation)
        lower_bound = DS2_mean - DS2_std
        upper_bound = DS2_mean + DS2_std
        if type(DS2_mean) == np.ndarray:
            plt.fill_between(t_waveform,lower_bound,upper_bound,color='C3',alpha=0.3)
            plt.plot(t_waveform,DS2_mean,'C3',lw=3)
            plt.title('DS2 (#'+str(len(DS_type2))+')',size=15)
        else:
            plt.title('DS2 (#0)',size=15)
        plt.autoscale(enable=True,axis='x',tight=True)
        plt.xlabel('Time from DS peak (ms)',size=14)
        plt.ylabel('Amplitude (mV)',size=14)
        plt.xticks(size=13)
        plt.yticks(size=13)
        plt.grid(linestyle='dotted')

        xnew = np.linspace(-200,200,1601) # new time vector (new sampling rate: 4 kHz)

        plt.subplot(223) ### Mean DS1 and DS2 waveforms (resampled)
        # Interpolation to quadruple sampling rate
        if type(DS2_mean) == np.ndarray:
            f2 = interp1d(t_waveform, DS2_mean, kind='cubic')
            plt.plot(xnew,f2(xnew),'C3',lw=3,label='DS2')
        if type(DS1_mean) == np.ndarray:
            f1 = interp1d(t_waveform, DS1_mean, kind='cubic')
            plt.plot(xnew,f1(xnew),'C2',lw=3,label='DS1')
        plt.title('Mean DS waveforms',size=15)
        plt.autoscale(enable=True,axis='x',tight=True)
        plt.xlabel('Time from DS peak (ms)',size=14)
        plt.ylabel('Amplitude (mV)',size=14)
        plt.xticks(size=13)
        plt.yticks(size=13)
        plt.legend(fontsize=13)
        plt.grid(linestyle='dotted')

        plt.subplot(224) ### 2nd derivatives and DS width limits based on resampled mean waveforms
        if type(DS2_mean) == np.ndarray:
            DS2_width,DS2_mean_dd,DS2_p_mask = analysis.DSpikeWidths(t_waveform,DS_type2)
            plt.plot(xnew,DS2_mean_dd,'C3',lw=3)
            for t_peak in xnew[DS2_p_mask]: plt.axvline(t_peak,ls='--',c='C3',lw=2)
        else:
            DS2_width = 0
            DS2_p_mask = np.zeros(len(t_waveform),dtype=bool)
        if type(DS1_mean) == np.ndarray:
            DS1_width,DS1_mean_dd,DS1_p_mask = analysis.DSpikeWidths(t_waveform,DS_type1)
            plt.plot(xnew,DS1_mean_dd,'C2',lw=3)
            for t_peak in xnew[DS1_p_mask]: plt.axvline(t_peak,ls='dotted',c='C2',lw=3)
        else:
            DS1_width = 0
            DS1_p_mask = np.zeros(len(t_waveform),dtype=bool)
        plt.title('DS widths (ms): '+str(DS1_width)+', '+str(DS2_width),size=15)
        plt.autoscale(enable=True,axis='x',tight=True)
        plt.xlabel('Time from DS peak (ms)',size=14)
        plt.ylabel('2nd derivative (mV/ms²)',size=14)
        plt.xticks(size=13)
        plt.yticks(size=13)
        plt.xlim(-20,20)
        plt.grid(linestyle='dotted')

        plt.tight_layout();
        
    def DSRateDeltaPower(IMF_delta, LFP, epoch_min, srate, delta_range, ch, ref_ch=None, detectDSperBin=False,thrs=0.9):
        """Computes the number of DSs and the relative delta power per one-minute window.
        
        Parameters
        ----------
        IMF_delta : ndarray
            Delta component of the LFP.
        LFP : ndarray
            LFP matrix (channels by samples).
        epoch_min : float, int
            Bin size (in minutes).
        srate : float
            Sampling rate (samples per second).
        delta_range : list of floats
            Tuple indicating the delta frequency range (in Hz).
        ch : int
            Target channel index of the LFP matrix.
        ref_ch : int or None, default=None
            Reference channel index of the LFP matrix. If None, no reference channel
            is selected.
        detectDSperBin : bool, default=False
            If True, DSs are detected per time bin. The fault value is False, i.e.,
            DSs are detected considering the whole LFP.
        thrs : float, default=0.9
            Threshold that defines the "weak" and "strong" delta waves.
            The relative delta power above thrs defines the "strong" delta.
        
        Returns
        -------
        t_delta : ndarray
            Time vector for plotting DS rate and delta relative power.
            Each point correspond to one "epoch_min" bin.
        rel_delta_power : ndarray
            Relative delta power per time bin.
        n_DSs : ndarray
            Number of DSs per time bin.
        PSD_all : ndarray
            Power spectral density for each time bin.
        PSD_high : ndarray
            Power spectral density for each time bin of strong delta.
        PSD_low : ndarray
            Power spectral density for each time bin of weak delta.
        F : ndarray
            Power spectral density frequencies.
        """
        import numpy as np
        from scipy.signal import welch
        from time import time

        tic = time() # measure the elapsed processing time

        nsamples  = int(epoch_min*60*srate)    # number of samples 
        n_epochs  = int(len(LFP[ch])/nsamples) # number of epochs

        # Parameters for Power Spectrum Density
        windowlen = 4.*srate     # window length
        overlap   = windowlen/2. # overlap length
        nfft      = 2.**14       # number of FFTs

        t_delta = np.linspace(epoch_min,epoch_min*n_epochs,n_epochs)

        n_DSs           = [] # number of DSs per time bin
        rel_delta_power = [] # relative delta power per time bin
        PSD_all         = [] # All PSDs per time bin
        PSD_high        = [] # PSDs of high delta per time bin
        PSD_low         = [] # PSDs of low delta per time bin

        if not(detectDSperBin): # DS detection considering the whole LFP
            DS_ind,_,_,_,_ = detection.detectDS(LFP,ch,refch=ref_ch)
        
        # Compute power spectral density (PSD) of delta component
        for epoch_id in range(n_epochs):
            X_delta     = IMF_delta[int(epoch_id*nsamples):int((epoch_id+1)*nsamples)]
            F,PSD_delta = welch(X_delta,fs=srate,nperseg=windowlen,noverlap=overlap,nfft=nfft)
            PSD_all.append(PSD_delta)

            # Relative delta power
            delta_idx = (F>=delta_range[0])*(F<=delta_range[1])        # delta range indexes 
            rel_power = np.sum(PSD_delta[delta_idx])/np.sum(PSD_delta) # relative delta power
            rel_delta_power.append(rel_power)

            if rel_power > thrs:
                PSD_high.append(PSD_delta) # PSD of strong delta
            else:
                PSD_low.append(PSD_delta)  # PSD of weak delta
            
            if detectDSperBin: # DS detection per time bin
                DS_ind,_,_,_,_ = detection.detectDS(LFP[:, int(epoch_id*nsamples):int((epoch_id+1)*nsamples)], ch, refch=ref_ch)
                n_DSs.append(len(DS_ind))
            else: # Number of DSs considering the whole LFP
                n_DSs.append(np.sum((DS_ind>=int(epoch_id*nsamples))*(DS_ind<int((epoch_id+1)*nsamples))))

        # tranforming lists into arrays
        rel_delta_power = np.array(rel_delta_power)
        n_DSs           = np.array(n_DSs)
        PSD_all         = np.array(PSD_all)
        PSD_high        = np.array(PSD_high)
        PSD_low         = np.array(PSD_low)

        toc      = time()
        elapsedT = (toc-tic)/60. # elapsed processing time (in minutes)
        print("Processing elapsed time: "+str(np.round(elapsedT,2))+" min.")

        return t_delta, rel_delta_power, n_DSs, PSD_all, PSD_high, PSD_low, F
    
    def DSperRelDelta(rel_delta_power, n_DSs, PSD_all, bin_ranges):
        """Computes the number of DSs, medians and quartiles per time bin of
        weak ans strong delta waves.
        
        Parameters
        ----------
        rel_delta_power : ndarray
            Relative delta power per time bin.
        n_DSs : ndarray
            Number of DSs per time bin.
        PSD_all : ndarray
            Power spectral density for each time bin.
        bin_ranges : float
            Ranges of weak and strong delta. The first line corresponds to
            the relative power range for weak delta, while the second line
            contais the relative power range for strong delta.
        
        Returns
        -------
        n_DSs_per_ratio_bin : list of ndarrays
            Number of DSs per bins of weak and strong delta.
        quartile1 : list of floats
            First quartiles of DS rate for weak and strong delta.
        quartile3 : list of floats
            Third quartiles of DS rate for weak and strong delta.
        medians : list of floats
            Medians of DS rate for weak and strong delta.
        """
        import numpy as np

        ### Number of DSs per range of relative delta power ###
        n_DSs_per_ratio_bin = []
        for bin_range in bin_ranges:
            ind = (rel_delta_power > bin_range[0])*(rel_delta_power <= bin_range[1])
            n_DSs_per_ratio_bin.append(n_DSs[ind])

        quartile1 = [] # First quartiles of DS rate for weak and strong delta.
        quartile3 = [] # Third quartiles of DS rate for weak and strong delta.
        medians   = [] # Medians of DS rate for weak and strong delta.
        for i in range(len(n_DSs_per_ratio_bin)):
            if len(n_DSs_per_ratio_bin[i]):
                q1,m,q3 = np.percentile(n_DSs_per_ratio_bin[i],[25,50,75])
                quartile1.append(q1)
                quartile3.append(q3)
                medians.append(m)
            else:
                quartile1.append(0)
                quartile3.append(0)
                medians.append(0)
                n_DSs_per_ratio_bin[i] = np.array([0])

        return n_DSs_per_ratio_bin, quartile1, quartile3, medians
            
class detection():
    
    def detectDS(lfp, ch, refch=0, srate=1e3, winsize=400, min_dist=50, lowf=1., highf=200., thrs_level=7, offsetRange=[-10,10], filtOutliers=True):
        """Detects dentate spikes (DSs) in the target channel of the LFP matrix.
        
        Parameters
        ----------
        lfp : ndarray
            LFP matrix (channels by samples).
        ch : int
            Target channel index.
        refch : int, default=0
            Reference channel index.
        srate : float, default=1000.0
            Current sampling rate.
        winsize : int, default=400
            Window size of the DS waveform (in milliseconds).
        min_dist: int, default=50
            Minimum distance between neighboring events (in milliseconds).
        lowf : float, default=1.
            Low cutoff frequency of the band-pass filter.
        highf : float, default=200.
            High cutoff frequency of the band-pass filter.
        thrs_level : int or float, default=7
            Level of the DS detection threshold.
        offsetRange : list of floats or ints or None, default=[-10,10]
            Time range used for peak offset correction (in milliseconds).
            Consider zero as the peak instant. If None, this step is ignored.
        filtOutliers : bool, default=True
            If True, the events whose peak amplitudes are out of the
            Tukey's fences of the distribution are ignored.
        
        Returns
        -------
        DS_ind : ndarray
            Indexes of the detected events in the LFP signal.
        DS_waveforms : ndarray
            Waveforms of the detected events.
        t_waveform : ndarray
            Time vector of the DS waveform (in milliseconds).
        offset : ndarray
            If offsetRange is defined, it returns the offsets applied to each
            event (in milliseconds).
        DS_waveforms_ : ndarray
            Waveforms of the detected events before the peak offset correction
            is applied. If If offsetRange is None, DS_waveforms_ will be equal
            to DS_waveforms.
        """
        import numpy as np
        from scipy.signal import butter,filtfilt,find_peaks

        # Minimum distance between DSs (in number of samples).
        n_samples_distance = int(min_dist*srate/1e3)

        # Computes the half-window size around the peak, considering the full window size (winsize) in ms
        halfwin_ms = winsize/2                 # half size of the waveform time window (ms)
        halfwin    = int(srate*halfwin_ms/1e3) # half size of the waveform time window (#samples)

        # Parameters of the IIR filter:
        order = 4        # filter order
        nyq   = .5*srate # Nyquist frequency (Hz)

        # Transfer function polynomials of the IIR filter (Butterworth type):
        b,a = butter(order,[lowf/nyq,highf/nyq],'bandpass')

        # Filtered signal (1-200 Hz) of the LFP in the target channel (DS channel) subtracted 
        # by the LFP of a reference channel, avoiding peaks due to artifacts.
        if type(refch)==int:
            lfp_ = filtfilt(b,a,lfp[ch,:]-lfp[refch,:])
        else:
            lfp_ = filtfilt(b,a,lfp[ch,:])

        # DSs will be chosen as peaks of the previous signal between the lower and upper
        # thresholds. The last one serving to again avoid peaks due to artifacts.
        thrs  = np.median(np.abs(lfp_))*thrs_level

        # The last index to be considered in the time-series, avoiding selecting DS
        # waveforms that would exceed the end of the signal.
        lastID = len(lfp_)-halfwin-1

        # Peak detection
        DS_indexes = find_peaks(lfp_[halfwin:lastID], height=thrs, distance=n_samples_distance)
        DS_ind     = np.array(DS_indexes[0]+halfwin)

        # DS waveforms extracted from the raw LFP and centered on previously detected peaks
        DS_waveforms = [lfp[ch,ind-halfwin:ind+halfwin+1] for ind in DS_ind]
        DS_waveforms = np.squeeze(DS_waveforms)

        DS_waveforms_ = DS_waveforms

        # Time vector of the DS waveform (in ms and relative to DS peak)
        t_waveform = np.linspace(-halfwin_ms,halfwin_ms,int(winsize*srate/1e3+1))

        # Peak offset correction due to filtering
        if offsetRange and len(np.shape(DS_waveforms))>=2:
            ind_win   = (t_waveform>=offsetRange[0])*(t_waveform<=offsetRange[1])
            t_central = t_waveform[ind_win]

            offset = np.array([np.argmax(DS_waveforms[ds,ind_win])-int(int(len(t_central)/2)*srate/1e3) 
                               for ds in range(len(DS_waveforms))])

            DS_ind       = DS_ind+offset
            DS_waveforms = [lfp[ch,ind-halfwin:ind+halfwin+1] for ind in DS_ind]
            DS_waveforms = np.squeeze(DS_waveforms)
        else:
            offset = [0]

        # Outlier filtering based on the amplitude distribution of DS peaks
        if filtOutliers and len(np.shape(DS_waveforms))==2:
            DS_peaks          = DS_waveforms[:,t_waveform==0]      # peak distribution
            p25,m,p75         = np.percentile(DS_peaks,[25,50,75]) # median and IQR
            DS_peak_thrs_high = p75+1.5*(p75-p25)                  # upper threshold
            DS_peak_thrs_low  = p25-1.5*(p75-p25)                  # lower threshold
            
            # outlier excluder index mask
            DS_indmask = np.concatenate((DS_peaks>DS_peak_thrs_low)*(DS_peaks<DS_peak_thrs_high)) 
            
            DS_ind        = DS_ind[DS_indmask]        # DS index array without outliers
            DS_waveforms  = DS_waveforms[DS_indmask]  # DS waveform array without outliers
            DS_waveforms_ = DS_waveforms_[DS_indmask] # DS waveform array without outliers
            if len(offset)>1:
                offset = offset[DS_indmask]

        return DS_ind, DS_waveforms, t_waveform, offset, DS_waveforms_

class classification():
    
    def CSDbC(LFPs, DS_indexes, n_PCs=1):
        """Classifies dentate spikes into types 1 or 2 based on the current
        source density of the dentate gyrus laminar profile obtained through
        a linear probe.
        
        Parameters
        ----------
        LFPs : ndarray
            Matrix of channels by local field potentials.
        DS_indexes : ndarray
            Indexes of dentate spike peaks detected in 'LFPs' matrix.
        n_PCs : int, float or 'mle', default=1
            Number of components to keep.
            if n_components is not set all components are kept::

                n_components == min(n_samples, n_features)

            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components.

            If ``svd_solver == 'arpack'``, the number of components must be
            strictly less than the minimum of n_features and n_samples.

            Hence, the None case results in::

                n_components == min(n_samples, n_features) - 1
        
        Returns
        -------
        class_prob : ndarray
            Matrix of probabilities of each event belonging to one of the two
            classes (DS types).
        DS_classes : ndarray
            Array containing the classes of each event. "0" corresponds to DS
            type 1, and "1" corresponds to DS type 2.
        DS1_ind : ndarray
            DS type 1 indexes.
        DS2_ind : ndarray
            DS type 2 indexes.
        """
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        
        nCh              = len(LFPs)                       # Number of channels
        dentateSpikes    = np.zeros((len(DS_indexes),nCh)) # DS peak array
        dentateSpikesCSD = np.zeros((len(DS_indexes),nCh)) # CSD array at DS peaks

        for di,dx in enumerate(DS_indexes):                         # For each DS index:
            dentateSpikes[di] = LFPs[:,dx]                          # store amplitude samples at DS peak
            dentateSpikesCSD[di] = analysis.runCSD(dentateSpikes[di]) # compute CSD at DS peak

        pca   = PCA(n_components=n_PCs)             # Principal Component Analysis model
        trans = pca.fit_transform(dentateSpikesCSD) # Transformed DS peak matrix by PCA
        
        # 2-component Gaussian Mixture Model
        clf = GaussianMixture(n_components=2, random_state=0, covariance_type='full')
        clf.fit(trans) # GMM model fitted with transformed matrix by PCA

        dsClassProb = clf.predict_proba(trans) # GMM class probabilities
        nclass      = np.size(dsClassProb,1)   # Number of classes (clusters)
        
        mainsink = np.zeros(nclass,dtype=int)  # Array for storing the channels with the main current sinks

        for ci in range(nclass): # For each class:
            prob         = dsClassProb[:,ci]                          # probabilities for such class
            meancsd      = np.mean(dentateSpikesCSD[prob>0.5],axis=0) # mean CSD for DSs in this class (prob > 0.5)
            mainsink[ci] = np.argmin(meancsd[:np.argmax(meancsd)])    # main sink before main source
            
        # If the main sinks are in the same position when analyzing from top to bottom,
        # re-evaluate from bottom to top.
        if list(mainsink).count(mainsink[0]) == len(mainsink):
            for ci in range(nclass):
                prob         = dsClassProb[:,ci]                                # probabilities for such class
                meancsd      = np.mean(dentateSpikesCSD[prob>0.5],axis=0)[::-1] # upside-down mean CSD for DSs in this class
                mainsink[ci] = np.argmin(meancsd[:np.argmax(meancsd)])          # main sink before main source

        order      = np.argsort(mainsink) # order of the indexes of the main current sinks (DS1 before DS2)
        class_prob = dsClassProb[:,order] # class probabilities matrix in the righ order (DS1 before DS2)
        
        DS_classes = np.argmax(class_prob,axis=1)               # DS classes (0 -> DS1, 1 -> DS2)
        DS1_ind    = DS_indexes[(DS_classes*-1+1).astype(bool)] # DS1 indexes
        DS2_ind    = DS_indexes[DS_classes.astype(bool)]        # DS2 indexes

        return class_prob, DS_classes, DS1_ind, DS2_ind
    
    def secondDerivAnalysis(t_waveform, DS_waveforms, DS1_waveforms, DS2_waveforms, DS_classes, t_range=[-10,10], thrs=0.06):
        """Performs the DS single type checking.
        
        To assess whether the two DS groups represented distinct populations,
        we calculate a dissimilarity index (DI) based on the second derivative
        of their mean waveforms. Specifically, the DI is defined as the mean
        absolute difference between their scaled second derivatives (lying
        between 0 and 1) within a 20-ms window centered at the DS peak. If DI
        is equal to or lower than the empirically chosen threshold of 0.06
        (adjustable), it indicates that the DSs belong to the same type.
        
        The type of the single DS cluster is then determined through a voting
        process based on the width metrics and their corresponding boundary
        values. Type 1 is voted on if the DS width exceeds 19 ms, if the start
        width limit occur before −9.5 ms from the peak, or if the end width
        limit occur after 9.5 ms. Otherwise, votes are for type 2. In this way,
        DSs are classified as the type that received at least two votes out of
        three.
        
        Parameters
        ----------
        t_waveform : ndarray
            Time vector of the DS waveform (in milliseconds).
        DS_waveforms : ndarray
            Waveforms of all dentate spikes.
        DS1_waveforms : ndarray
            Waveforms of type-1 detate spikes.
        DS2_waveforms : ndarray
            Waveforms of type-2 detate spikes.
        DS_classes : ndarray
            Array containing the classes of each event. "0" corresponds to DS
            type 1, and "1" corresponds to DS type 2.
        t_range : list or ndarray, default=[-10,10]
            List with the initial and final values of the time window (in ms)
            used for the second derivative analysis of the mean waveforms.
        thrs : float, default=0.06
            Threshold of the dissimilarity index used to identify single type
            clusters.
        
        Returns
        -------
        DS1_waveforms : ndarray
            Waveforms of type-1 detate spikes.
        DS2_waveforms : ndarray
            Waveforms of type-2 detate spikes.
        DS_classes : ndarray
            Array containing the classes of each event. "0" corresponds to DS
            type 1, and "1" corresponds to DS type 2.
        DS1_unique_cluster : int
            If "1", all dentate spikes were identified as type 1.
        DS2_unique_cluster : int
            If "1", all dentate spikes were identified as type 2.
        """
        import numpy as np
        from sklearn.preprocessing import minmax_scale

        win_ind     = (t_waveform >= t_range[0])*(t_waveform <= t_range[1]) # time window of interest
        DS1_mean    = np.mean(DS1_waveforms,axis=0)                         # mean DS1 waveform
        dt          = (t_waveform[1]-t_waveform[0])                         # spacing between values (sampling period).
        DS1_mean_d  = np.gradient(DS1_mean[win_ind],dt)                     # 1st derivative of the mean DS1 waveform
        DS1_mean_dd = minmax_scale(np.gradient(DS1_mean_d,dt))              # scaled 2nd derivative of the mean DS1 waveform

    
        if DS2_waveforms.any():
            DS2_mean    = np.mean(DS2_waveforms,axis=0)            # mean DS2 waveform
            DS2_mean_d  = np.gradient(DS2_mean[win_ind],dt)        # 1st derivative of the mean DS2 waveform
            DS2_mean_dd = minmax_scale(np.gradient(DS2_mean_d,dt)) # scaled 2nd derivative of the mean DS2 waveform
            
            # Absolute amplitude difference between putative DS1 and DS2 mean waveforms
            abs_amp_diff = np.mean(np.abs(DS1_mean_dd-DS2_mean_dd))

            DS1_unique_cluster = 0
            DS2_unique_cluster = 0
            
            # DS single type checking
            if np.round(abs_amp_diff,2) <= thrs:
                
                DS_width, _, DS_pmask = analysis.DSpikeWidths(t_waveform,DS_waveforms) # DS width metrics
                
                vote_count            = 0                          # vote count for DS1
                t_waveform_new        = np.linspace(-200,200,1601) # adjusted time array for comparing DS width limits 
                
                # Voting
                if DS_width[0] > 19:                      # if DS width > 19 ms
                    vote_count += 1
                if t_waveform_new[DS_pmask[0]][0] < -9.5: # if start width limit < -9.5 ms
                    vote_count += 1
                if t_waveform_new[DS_pmask[0]][1] > 9.5:  # if end width limit > 9.5 ms
                    vote_count += 1
                if vote_count > 1:                        # if there are at least 2 votes, all events are DS1
                    DS1_unique_cluster += 1
                    DS1_waveforms = DS_waveforms
                    DS2_waveforms = np.zeros(len(t_waveform))
                    DS_classes    = np.zeros(len(DS_classes),dtype=int)
                else:                                     # for less than 2 votes, all events are DS2
                    DS2_unique_cluster += 1
                    DS2_waveforms = DS_waveforms
                    DS1_waveforms = np.zeros(len(t_waveform))
                    DS_classes    = np.ones(len(DS_classes),dtype=int)
        else:
            DS1_unique_cluster = 1
            DS2_unique_cluster = 0

        return DS1_waveforms, DS2_waveforms, DS_classes, DS1_unique_cluster, DS2_unique_cluster
    
    def WFbC(X_DS, X_feat, singleTypeCheck=True, srate=1e3, thrsd=0.06, init_state=0):
        """Waveform-based classification of dentate spikes.
        
        Parameters
        ----------
        X_DS : ndarray
            All full dentate spike waveforms.
        X_feat : ndarray
            Features of the dentate spike waveforms used for classification.
        singleTypeCheck : bool, default=True
            Determines if the DS single type checking will be performed.
        srate : float, default=1000.0
            Sampling rate (samples per second).
        thrsd : float, default=0.06
            Threshold of the dissimilarity index used to identify single type
            clusters.
        init_state : int, RandomState instance or None, default=0
            Controls the random seed given to the method chosen to initialize
            the GMM parameters. In addition, it controls the generation of
            random samples from the fitted distribution. Pass an int for
            reproducible output across multiple function calls.
        
        Returns
        -------
        DS_classes : ndarray
            Array containing the classes of each event. "0" corresponds to DS
            type 1, and "1" corresponds to DS type 2.
        DS_type1 : ndarray
            Waveforms of type-1 detate spikes.
        DS_type2 : ndarray
            Waveforms of type-2 detate spikes.
        DS1_uniquecluster : int or None
            If "1", all dentate spikes were identified as type 1.
        DS2_uniquecluster : int or None
            If "1", all dentate spikes were identified as type 2.
        """
        import numpy as np
        from sklearn.mixture import GaussianMixture
        
        # Copies of arrys with DS waveforms and features
        X_DS_original   = X_DS.copy()
        X_feat_original = X_feat.copy()

        # Classification via a 2-component Gaussian Mixture Model
        cluster    = GaussianMixture(n_components=2,random_state=init_state).fit_predict(X_feat)
        DS_classes = cluster # DS classes

        DS_0 = []
        DS_1 = []
        for ind,DS_class in enumerate(DS_classes):
            if DS_class == 0:
                DS_0.append(X_DS[ind]) # Waveforms of cluster 0
            else:
                DS_1.append(X_DS[ind]) # Waveforms of cluster 1

        DS_0              = np.array(DS_0)
        DS_1              = np.array(DS_1)
        DS_type1,DS_type2 = DS_0,DS_1

        win_size     = len(X_DS[0])                    # waveform window size
        ind_central  = int(win_size/2)                 # central index of the waveform window
        ind_10ms     = ind_central + int(10*srate/1e3) # index at 10 ms
        ind_50ms     = ind_central + int(50*srate/1e3) # index at 50 ms
        halfwin_size = int((win_size/2)*(srate/1e3))   # half size of the waveform window
        
        # DS-type assignment based on the difference between the mean waveforms from 10 to 50 ms after the peak.
        if np.argmin((np.sum(np.mean(DS_0,axis=0)[ind_10ms:ind_50ms]),np.sum(np.mean(DS_1,axis=0)[ind_10ms:ind_50ms]))) == 0:
            DS_type1,DS_type2 = DS_1,DS_0
            DS_classes        = abs(DS_classes-1)
            
        ### Identification of very small groups as putative artifacts. ###
        proportion = np.sum(DS_classes)/len(DS_classes)
        artifact   = False
        
        # Reclassifies the larger group until there are 2 groups with sizes between 5 and 95% of the total remaining events
        while (proportion > 0.95) or (proportion < 0.05):
            artifact = True

            if (proportion > 0.95):
                DS_classes[DS_classes==0] = 2 # number "2" indentifies the artifact class
            elif (proportion < 0.05):
                DS_classes[DS_classes==1] = 2

            X_DS   = X_DS_original[DS_classes!=2]
            X_feat = X_feat_original[DS_classes!=2]

            cluster     = GaussianMixture(n_components=2, random_state=init_state).fit_predict(X_feat)
            DS_classes_ = cluster
            DS_0 = []
            DS_1 = []
            for ind,DS_class in enumerate(DS_classes_):
                if DS_class == 0:
                    DS_0.append(X_DS[ind])
                else:
                    DS_1.append(X_DS[ind])

            DS_0              = np.array(DS_0)
            DS_1              = np.array(DS_1)
            DS_type1,DS_type2 = DS_0,DS_1

    
            if np.argmin((np.sum(np.mean(DS_0,axis=0)[ind_10ms:ind_50ms]),np.sum(np.mean(DS_1,axis=0)[ind_10ms:ind_50ms]))) == 0:
                DS_type1,DS_type2 = DS_1,DS_0
                DS_classes_       = abs(DS_classes_-1)

            DS_classes[DS_classes!=2] = DS_classes_
            proportion = np.sum(DS_classes_)/len(DS_classes_)

        # Single type checking
        if singleTypeCheck:
            t_waveform = np.linspace(-halfwin_size,halfwin_size,win_size)

            if artifact:
                DS_type1, DS_type2, DS_classes_, DS1_uniquecluster, DS2_uniquecluster = classification.secondDerivAnalysis(t_waveform, X_DS, DS_type1, DS_type2, DS_classes_, thrs=thrsd)
                DS_classes[DS_classes!=2] = DS_classes_
            else:
                DS_type1, DS_type2, DS_classes, DS1_uniquecluster, DS2_uniquecluster = classification.secondDerivAnalysis(t_waveform, X_DS, DS_type1, DS_type2, DS_classes, thrs=thrsd)
        else:
            DS1_uniquecluster = None
            DS2_uniquecluster = None

        return DS_classes, DS_type1, DS_type2, DS1_uniquecluster, DS2_uniquecluster
    
    def cm_metrics(cf):
        """Computes the classification metrics precision and recall for each
        class (DS type).
        
        Parameters
        ----------
        cf : ndarray
            Confusion matrix.
        
        Returns
        -------
        precision_DS1 : float
            Precision for dentate spike type 1.
        precision_DS2 : float
            Precision for dentate spike type 2.
        recall_DS1 : float
            Recall for dentate spike type 1.
        recall_DS2 : float
            Recall for dentate spike type 2.
        """
        precision_DS1 = cf[0,0]/(cf[0,0]+cf[1,0]) # precision for DS1
        precision_DS2 = cf[1,1]/(cf[1,1]+cf[0,1]) # precision for DS2
        recall_DS1    = cf[0,0]/(cf[0,0]+cf[0,1]) # recall for DS1
        recall_DS2    = cf[1,1]/(cf[1,1]+cf[1,0]) # recall for DS2
        
        return precision_DS1, precision_DS2, recall_DS1, recall_DS2
    
    def plotCM(cf, c_map='binary', v_min=0, v_max=1, sp=111, title=None):
        """Plots a confusion matrix.
        
        Parameters
        ----------
        cf : ndarray
            Confusion matrix.
        c_map : str, default='binary'
            Matplotlib colormap.
        v_min, v_max : float, default=0, 1
            Equivalent to Matplotlib 'vmin' and 'vmax', they define the data
            range that the colormap covers.
        sp : int or 3-tuple, default=111
            Matplotlib code for subplot.
        title : str or None, default=None
            Plot title. If None, no title is displayed.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = plt.subplot(sp)
        im = ax.matshow(cf, cmap=c_map, vmin=v_min, vmax=v_max)
        for i in range(2):
            for j in range(2):
                c = np.round(cf[j,i],2)
                if c > v_max/2.: plt.text(i, j, str(c), va='center', ha='center', size=18, color='w')
                else: plt.text(i, j, str(c), va='center', ha='center', size=18, color='k')
        plt.xticks([0,1],['DS1','DS2'],size=14)
        plt.yticks([0,1],['DS1','DS2'],size=14,rotation=90,va='center')
        plt.xlabel('WFbC',size=15)
        plt.ylabel('CSDbC',size=15)
        if title: plt.title(title,size=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im,cax=cax)
        cbar.ax.tick_params(labelsize=13)
        ax.tick_params(bottom=True,top=False)
        ax.xaxis.set_ticks_position('bottom');