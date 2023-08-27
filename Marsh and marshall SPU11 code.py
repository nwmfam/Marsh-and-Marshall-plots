# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:25:15 2022

@author: beefc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:30:29 2022

@author: beefc
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from matplotlib.collections import LineCollection
import math

def load_file_and_return_data_array(filename,time_interval, cols_to_use):
    """This does what it says"""
    data = np.loadtxt(filename, dtype = float, skiprows=2, usecols=cols_to_use)
    data = data
    time = np.arange(0,len(data)*time_interval,time_interval)
    """This will create the time array"""
    return data,time

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def create_displacement_data(data,dt):
    
    acceleration = data
    # acceleration = signal.detrend(acceleration) 
    velocity = scipy.integrate.cumtrapz(acceleration,dx = dt)
    velocity = signal.detrend(velocity)
    disp = integrate.cumtrapz(velocity,dx = dt)
    disp = signal.detrend(disp)
    velocity = np.append(0,velocity)
    disp = np.append(0,disp)
    disp = np.append(0,disp)
    """turns m to cm"""
    disp = disp*100
       
    return disp

def filter_data(data,upper_freq,lower_freq,fps):
    
    filtered_data = butter_highpass_filter(data,lower_freq,fps)
    filtered_data = butter_lowpass_filter(filtered_data,upper_freq,fps)
    
    return filtered_data

def plot_acc(data1,data2,time,start_plot,end_plot,upper_freq,lower_freq,title):
    
    "Start and stop times - change these to suit"
    start = start_plot
    end = end_plot
    
    start_array = np.ones(len(time))*start
    end_array = np.ones(len(time))*end
    s_list = abs(time - start_array)
    e_list = abs(time - end_array)
    "Start and end indexes for plotting"
    start_index = list(s_list).index(min(s_list))
    end_index = list(e_list).index(min(e_list))  
    plt.plot(time[start_index:end_index],data1[start_index:end_index],label = 'SE sensor')
    plt.plot(time[start_index:end_index],data2[start_index:end_index],label = 'NW Sensor')
    plt.title("{} filtered ACC data".format(title))
    plt.legend()
    plt.ylim(-30,30)
    plt.ylabel("Acceleration m/s/s")
    plt.xlabel("time (s)")
    plt.grid()
    # plt.savefig('{1}_{2}_{0}_acceleration.png'.format(title,start,end))
    plt.show()
    
def plot_acc_single(data1,time,start_plot,end_plot,upper_freq,lower_freq,title):
    
    "Start and stop times - change these to suit"
    start = start_plot
    end = end_plot
    
    start_array = np.ones(len(time))*start
    end_array = np.ones(len(time))*end
    s_list = abs(time - start_array)
    e_list = abs(time - end_array)
    "Start and end indexes for plotting"
    start_index = list(s_list).index(min(s_list))
    end_index = list(e_list).index(min(e_list))  
    plt.plot(time[start_index:end_index],data1[start_index:end_index],label = 'sensor acceleration')
    plt.title("{} filtered ACC data".format(title))
    plt.legend()
    plt.ylim(-30,30)
    plt.ylabel("Acceleration m/s/s")
    plt.xlabel("time (s)")
    plt.grid()
    # plt.savefig('{1}_{2}_{0}_acceleration.png'.format(title,start,end))
    plt.show()
    
def plot_acc_ground(data1,time,start_plot,end_plot,upper_frequency,lower_frequency,title,acc_lim):
    
    "Start and stop times - change these to suit"
    start = start_plot
    end = end_plot
    
    start_array = np.ones(len(time))*start
    end_array = np.ones(len(time))*end
    s_list = abs(time - start_array)
    e_list = abs(time - end_array)
    "Start and end indexes for plotting"
    start_index = list(s_list).index(min(s_list))
    end_index = list(e_list).index(min(e_list))  
    plt.plot(time[start_index:end_index],data1[start_index:end_index],label = 'Ground Acceleration')
    plt.title("{} Ground acceleration".format(title))
    plt.legend()
    plt.ylim(-30,30)
    plt.ylabel("Acceleration $\mathregular{m/s^{2}}$")
    plt.xlabel("time (s)")
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.grid()
    # plt.savefig('{1}_{2}_{0}_ground_acceleration.png'.format(title,start,end))
    plt.show()
    
def plot_acc_top(data1,time,start_plot,end_plot,upper_frequency,lower_frequency,title,acc_lim):
    
    "Start and stop times - change these to suit"
    start = start_plot
    end = end_plot
    
    start_array = np.ones(len(time))*start
    end_array = np.ones(len(time))*end
    s_list = abs(time - start_array)
    e_list = abs(time - end_array)
    "Start and end indexes for plotting"
    start_index = list(s_list).index(min(s_list))
    end_index = list(e_list).index(min(e_list))  
    plt.plot(time[start_index:end_index],data1[start_index:end_index],label = 'Roof Acceleration',color = "red")
    plt.title("Roof acceleration")
    # plt.legend()
    plt.margins(x=0,y=0)
    plt.ylim(-30,30)
    plt.ylabel("Acceleration $\mathregular{m/s^{2}}$")
    plt.xlabel("Time (s)")
    # plt.gca().axes.xaxis.set_ticklabels([])
    # plt.gca().axes.yaxis.set_ticklabels([])
    plt.grid()
    plt.savefig('Roof acceleration.png')
    plt.show()
    
def plot_disp(DISP_relative,time,start_plot,end_plot,upper_frequency,lower_frequency,title,disp_lim):
    
    "Start and stop times - change these to suit"
    start = start_plot
    end = end_plot
    
    start_array = np.ones(len(time))*start
    end_array = np.ones(len(time))*end
    s_list = abs(time - start_array)
    e_list = abs(time - end_array)
  
    "Start and end indexes for plotting"
    start_index = list(s_list).index(min(s_list))
    end_index = list(e_list).index(min(e_list))
    plt.plot(time[start_index:end_index],DISP_relative[start_index:end_index],label = 'Relative_displacement {}'.format(title))
    plt.title("Relative displacement")
    # plt.legend()
    plt.margins(x=0,y=0)
    plt.ylim(-disp_lim,disp_lim)
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    
    # plt.axis('off')
    # plt.ax.set_xticks(5)
    plt.ylabel("Displacement (cm)")
    plt.xlabel("Time (s)")
    plt.grid()
    
    plt.savefig('Displacement.png')
    plt.show()
    
def hysteresis(disp,acc,start,end,time,title,ylim,xlim):
    
    "Plots Hysteresis loops"
    print(start)
    print(end)
    start_array = np.ones(len(time))*start
    end_array = np.ones(len(time))*end
    s_list = abs(time - start_array)
    e_list = abs(time - end_array)
  
    "Start and end indexes for plotting"
    start_index = list(s_list).index(min(s_list))
    end_index = list(e_list).index(min(e_list))
    
    x = disp[start_index:end_index]
    y = -acc[start_index:end_index]/9.81
    time = time[start_index:end_index]  #  Defining how colour changes
    
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
   
    fig, ax = plt.subplots()
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(time)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # plt.colorbar(line)
    # ax.set_xlim(x.min(), x.max())
    # ax.set_ylim(y.min(), y.max())
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.grid()
    # plt.title("{0} Acceleration displacement plot".format(title))
    plt.ylabel("Acceleration $m/s^{2}$")
    plt.ylim(-ylim/9.81,ylim/9.81)
    print(ylim/9.81)
    print(xlim)
    plt.xlabel("Total Drift (%)")
    plt.xlim(-xlim,xlim)
    plt.savefig('Hysteresis.png'.format(title,start,end))
    plt.show()
    
def find_changes_in_acc_dir(acc,time):
    
    intercept_times = []
    
    for i in range(len(acc)-1):
        if acc[i] >=0:
            if acc[i+1]<0:
                time_to_record = (time[i]+time[i+1])/2
                intercept_times = np.append(time_to_record,intercept_times)
        elif acc[i] <0:
            if acc[i+1]>=0:
                time_to_record = (time[i]+time[i+1])/2
                intercept_times = np.append(time_to_record,intercept_times)
    return intercept_times

def find_changing_time_period(intercept_times):
    
    time_period_of_building = []
    at_time = []
    for i in range(1,len(intercept_times)-1,1):
        t_p = -(intercept_times[i+1]-intercept_times[i-1])
        time = intercept_times[i]
        time_period_of_building = np.append(t_p,time_period_of_building)
        at_time = np.append(time,at_time)
        
    return time_period_of_building,at_time

def identify_the_half_cycles(acc,time):
    """This finds the intercepts and their identifier"""
    
    intercept_no = []
    for i in range(len(acc)-1):
        if acc[i]>=0:
            if acc[i+1]<0:
                intercept = i
                intercept_no = np.append(intercept,intercept_no)
        if acc[i]<0:
            if acc[i+1]>=0:
                intercept = i
                intercept_no = np.append(intercept,intercept_no)
    intercept_no = np.flip(intercept_no)
    
    return intercept_no

def averaging_period(period,intercept_times):
    
    avg_no = 3 #number of period values to get a rolling average - needs to be odd number
    interval = (avg_no//2)+1
    avg_period = []
    time_point = []
    for i in range(interval,len(period),1):
        avg_period.append(np.mean(period[i-interval:i+interval]))
        time_point.append(intercept_times[i])
    time_point = np.flip(time_point)
    
        
    return avg_period,time_point
        
        
def print_period(t_p_of_build,at_time,start_plot,end_plot,title):
        
    plt.plot(at_time,t_p_of_build,label ='Period estimated from cycle length',color = 'dodgerblue')
    plt.title('period during EQ {}'.format(title))
    plt.ylabel("Period (s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.ylim([0,3])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.grid()
    # plt.savefig('{1}_{2}_{0}_period.png'.format(title,start_plot,end_plot))
    plt.show()
    
def print_average_period(t_p_of_build,at_time,start_plot,end_plot,title):
        
    plt.plot(at_time,t_p_of_build,label ='Period estimated from cycle length',color = 'dodgerblue')
    plt.title('average period during EQ {}'.format(title))
    plt.ylabel("Period (s)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.ylim([0,3])
    # plt.savefig('{1}_{2}_{0}_average_period.png'.format(title,start_plot,end_plot))
    plt.show()
    
def determine_frequency(dt):
        
    lower_frequency = 0.25
        
    if dt < 0.02:
        upper_frequency = 25
        
    else:
        frequency_of_recording = 1/dt/2 - 1
        upper_frequency = math.floor(frequency_of_recording)
    return upper_frequency, lower_frequency

def determine_start_and_end_plot(disp, time, threshold):
    
    max_val = 0
    for i in range(len(disp)):
        if np.abs(disp[i])>max_val:
            max_val = disp[i]
    
    for j in range(len(disp)):
        
        if np.abs(disp[j])>= max_val*threshold:
            start_plot = time[j]
            break
    for ii in range(len(disp)):
        
        if np.abs(disp[ii])>= max_val*threshold:
            end_plot = time[ii]
            
    start_plot = np.round(start_plot)
    end_plot = np.round(end_plot)
            
    return start_plot, end_plot

def determine_limits_to_show_in_plot(data,standard):
    
    max_val = standard
    
    for i in range(len(data)):
        if np.abs(data[i])>max_val:
            max_val = np.abs(data[i])
            
    limit = math.ceil(max_val)
    return limit


   
                    

  
def zoomed_in_plots(failures_to_return,ACC_top_filtered,fundamental_acceleration,time,upper_frequency,lower_frequency,title,acc_lim,DISP_relative,disp_lim): 
    
    time_range = 8
    
    for i in range(len(failures_to_return)):
        start_plot = np.round(failures_to_return[i]-time_range/2)
        end_plot = np.round(failures_to_return[i]+time_range/2)
    
        plot_acc_top(ACC_top_filtered,time,start_plot,end_plot,upper_frequency,lower_frequency,title,acc_lim)
        """This plots the relative damage"""
        plot_disp(DISP_relative,time,start_plot,end_plot,upper_frequency,lower_frequency,title,disp_lim)
        """This plots the hysteresis loops"""
        hysteresis(DISP_relative,fundamental_acceleration,start_plot,end_plot,time,title,acc_lim,disp_lim)

def polynomial(period,time,start_plot,end_plot,title,orderxx):
    
    """This fits a polynomial to the period plot and plots it"""
    
    poly_period = np.polyfit(time,period,orderxx)
    p = np.poly1d(poly_period)
    plt.plot(time,p(time),label = "Period from {} order polynomial".format(orderxx))
    plt.ylabel("Period (s)")
    plt.grid()
    plt.xlabel("time (s)")
    plt.legend
    plt.plot(time,period,label = "Period from cycle length")
    plt.ylim(0,3)
    plt.title("Period vs time {} ".format(title))
    plt.show()
    
    start = p(time)[0]
    end = p(time)[-1]
    
    percent = (end/start-1)*100
               
    print("Period change is {}%".format(percent))
    
def find_time_of_peak_displacement(DISP,time):
    
    MAX = 0
    for i in range(len(DISP)):
        if np.abs(DISP[i])>= MAX:
            MAX = np.abs(DISP[i])
            time_of_max = time[i]
            
    return MAX,time_of_max
    
def show_avg_period_either_side_of_max_displacement(period,at_time,t_of_max,ss,ee):
    
    for i in range(len(period)):
        if at_time[i]<= t_of_max:
            point_to_go_to = i
            
    period_before_max = period[0:point_to_go_to+1]
    start_plot = at_time[0]
    end_plot = at_time[point_to_go_to]
    at_time_before_max = at_time[0:point_to_go_to+1]
    period_after_max = period[point_to_go_to:-1]
    at_time_after_max = at_time[point_to_go_to:-1]
    
    average_period_before_max = np.mean(period_before_max)
    average_period_after_max = np.mean(period_after_max)
    
    average_period_before = []
    average_period_after = []
    
    for j in range(len(at_time_before_max)):
        
        average_period_before = np.append(average_period_before_max,average_period_before)
        
    for jj in range(len(at_time_after_max)):
        
        average_period_after = np.append(average_period_after_max,average_period_after)
        
    plt.plot(at_time,period,label ='Period estimated from cycle length',color = 'dodgerblue')
    plt.plot(at_time_before_max,average_period_before,color = "red",label = "Mean period before peak displacement")
    plt.plot(at_time_after_max,average_period_after,color = "green",label = "Mean period after peak displacement",linestyle = "dashed")
    plt.title('Period plot')
    plt.ylabel("Period (s)")
    plt.xlabel("Time (s)")
    plt.margins(x=0)
    plt.xlim(ss,ee)
    plt.legend()
    # plt.gca().axes.yaxis.set_ticklabels([])
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.grid()
    plt.ylim([0,2.5])
    plt.savefig('Period_Plot.png')
    plt.show()
    
    start = average_period_before_max
    end =  average_period_after_max
    
    percent = (end/start-1)*100
               
    print("Period change is {:.1f}%".format(percent))
    print("Initial period is {:.2f}s".format(start))
    print("Final period is {:.2f}s".format(end))
    
    
def main():
    
        
    test_no = 1
    build_heights = [0,32.9,32.9,20,20,32.9,2.4,2.4,2.4,25.4,25.4,25.4,25.4,25.4,25.4,27.5,27.5,27.5,27.5,27.5,27.5,10.4,10.4,10.4,1.1,1.1,1.8,1.8,2.9,2.9,2.9,2.9,2.9,2.9,2.9]
    """This script runs everything that needs be done
    changes can be made here and parameters adjusted as required,
    all other functions references in this main function should remain the same"""
    """There is an index for each data set on the excel document"""
    """#############################################################"""
    """CHANGE THIS TO LOAD A NEW DATA SET"""
    filename = "Test {}.txt".format(test_no)
    """#############################################################"""
    
    """This is the time interval of the recorded data"""
    time_interval = np.loadtxt(filename,dtype = float, skiprows = 1, usecols = 0, max_rows = 1)
    dt = time_interval
    textfile = open(filename, "r")
    """Title of the dataset used"""
    title = textfile.readline().strip()
    textfile.close()
    """Loading the acceleration of the top of the test/building"""
    cols_to_use = 1
    data, time = load_file_and_return_data_array(filename, time_interval, cols_to_use)
    """Loading the acceleration data for the ground"""
    cols_to_use = 0
    data_g, time_g = load_file_and_return_data_array(filename, time_interval, cols_to_use)
    
    ###FILTERING THE DATA BEGINS HERE FPS IS "FRAMES PER SECOND" UPPER FREQUENCY AND LOWER FREQUNECY IS ALMOST
    ###ALWAYS 0.25 HZ AND 25 HZ RESPECTIVELY UNLESS FOR SOME REASON THAT DOES NOT FIT THE DATA PROVIDED (IE 
    ### IT ALREADY IS 25 HZ###
    ##########################################################################################################
    """Determines the frequencies to be used for the bandpass filter, default is 25 hz to 0.25 hz 
    unless this is not possible given the time interval of the dataset"""
    
    upper_frequency, lower_frequency = determine_frequency(time_interval)
    fps = 1/dt
    
    """Filtering the acceleration data to remove the noise"""
    
    ACC_top_filtered = filter_data(data,upper_frequency,lower_frequency,fps)
    ACC_ground_filtered = filter_data(data_g,upper_frequency,lower_frequency,fps)
    
    """CREATING DISPLACEMENT DATA"""
    
    DISP_top = create_displacement_data(ACC_top_filtered,dt)
    DISP_ground = create_displacement_data(ACC_ground_filtered,dt)
    
    DISP_relative = (DISP_top-DISP_ground)
###This is probably all that was important for purely filtering, most of the functions requires are in the top 100 lines of code"""

    """This calculates the plot boundaries to use, xlims, ylims, and time ranges for effective plots"""
    
    """tHIS threshold is mostly 0.2 meaning the printed plots will not plot until acc exceeds 20% of the peak"""
    
    threshold = 0.2

    
    start_plot, end_plot = determine_start_and_end_plot(ACC_top_filtered, time, threshold)
    standard = 10
    acc_lim = determine_limits_to_show_in_plot(ACC_top_filtered,standard)
    standard = 2
    disp_lim = determine_limits_to_show_in_plot(DISP_relative,standard)
    
    """Plotting the neccessary plots here"""
    
    """Overall accelleration records"""
    title = title
    """This plots the ground acceleration"""
    # plot_acc_ground(ACC_ground_filtered,time,start_plot,end_plot,upper_frequency,lower_frequency,title,acc_lim)
    """This plots the roof acceleration"""
    plot_acc_top(ACC_top_filtered,time,start_plot,end_plot,upper_frequency,lower_frequency,title,acc_lim)
    """This plots the relative damage"""
    plot_disp(DISP_relative,time,start_plot,end_plot,upper_frequency,lower_frequency,title,disp_lim)
    """This plots the hysteresis loops"""
    # hysteresis(DISP_relative,ACC_top_filtered,start_plot,end_plot,time,title,acc_lim,disp_lim)
    """Plotting periods"""
    
    max_disp, t_of_max = find_time_of_peak_displacement(DISP_relative,time)
    
    hysteresis(DISP_relative,ACC_top_filtered,t_of_max - 3,t_of_max + 3,time,title,acc_lim,disp_lim)
    
    # print("The peak displacement is {0}cm and occurs at t = {1}s".format(max_disp,t_of_max))
    
    data_to_use = DISP_relative[np.int64(start_plot/dt):np.int64(end_plot/dt)]
    t_to_use = time[np.int64(start_plot/dt):np.int64(end_plot/dt)]
    intercept_times = find_changes_in_acc_dir(data_to_use,t_to_use)
    
    intercept_times = list(intercept_times)
      
    
    t_p_of_build,at_time = find_changing_time_period(intercept_times)
    avg_period, time_point = averaging_period(t_p_of_build,intercept_times)
    # print_average_period(avg_period,time_point,start_plot,end_plot,title)   
    # print_period(t_p_of_build,at_time,start_plot,end_plot,title) 
    # order = 2
    # polynomial(t_p_of_build,at_time,start_plot,end_plot,title,order)
    
    # show_avg_period_either_side_of_max_displacement(t_p_of_build,at_time,t_of_max,start_plot,end_plot)
    show_avg_period_either_side_of_max_displacement(avg_period,time_point,t_of_max,start_plot,end_plot)
    print("The peak displacement is {0:.0f}cm and occurs at t = {1}s".format(max_disp*build_heights[test_no],t_of_max))
    print("The peak drift ratio is {0:.1f}% and occurs at t = {1}s".format(max_disp,t_of_max))
    print(title)

    # polynomial(t_p_of_build,at_time,start_plot,end_plot)
    
    # polynomial(avg_period,time_point,start_plot,end_plot)
    # return max_disp
    print(t_p_of_build)
    print(intercept_times)
main()

# maxdrift = 0
# for i in range(33):
    # main(i+1)
    # drift = main(i)
    # if drift>maxdrift:
    #     maxdrift = drift
# print(maxdrift)
    
    
