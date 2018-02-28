#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script provides utility functions for computing the bias metrics.

Created on Thu Jul 20 20:02:57 2017

@author: emilywall
"""
import csv
import json
import math
import sys
import os
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from pprint import pprint
import seaborn as sns

# define some important variable
color_map = { 'none': '#7f7f7f', 'Un-Assign': '#7f7f7f', 'Point Guard': '#1f77b4', 'Shooting Guard': '#ff7f0e', 'Small Forward': '#2ca02c', 'Power Forward': '#d62728', 'Center': '#9467bd' }
num_to_pos_map = { 0: 'Center', 1: 'Shooting Guard', 2: 'Point Guard', 3: 'Small Forward', 4: 'Power Forward', 5: 'none', 6: 'Un-Assign' }
pos_to_num_map = { 'Center': 0, 'Shooting Guard': 1, 'Point Guard': 2, 'Small Forward': 3, 'Power Forward': 4, 'none': 5, 'Un-Assign': 6 }
pos_names = sorted(pos_to_num_map.items(), key = operator.itemgetter(1))
pos_names = [x[0] for x in pos_names]

int_to_num_map = { 'hover': 0, 'drag': 1, 'double_click': 2, 'click': 3, 'set_attribute_weight_vector_drag': 4, 'set_attribute_weight_vector_calc': 5, 'set_attribute_weight_vector_select': 6, 'category_click': 7, 'category_double_click': 8, 'help_hover': 9 }
interaction_names = sorted(int_to_num_map.items(), key = operator.itemgetter(1))
interaction_names = [x[0] for x in interaction_names]

attrs_all = ['Player', 'Player Anonymized', 'Team', 'Position', 'Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Field Goals Att.', 'Avg. Field Goals Made', 'Avg. Free Throws Att.', 'Avg. Free Throws Made', 'Avg. Minutes', 'Avg. Personal Fouls', 'Avg. Points', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds', 'Avg. Turnovers', 'Games Played', 'Height (Inches)', 'Weight (Pounds)', '+/-']
attrs = attrs_all[4 : -1]
window_methods = ['all', 'fixed', 'classification_v1', 'classification_v2', 'category_v1', 'category_v2']
window_method = 'all' # options include 'all', 'fixed', 'classification_v1', 'classification_v2', 'category_v1', and 'category_v2'
marks = 'categories' # options include 'classifications' and 'categories'
metric_names = ['data_point_coverage', 'data_point_distribution', 'attribute_coverage', 'attribute_distribution', 'attribute_weight_coverage', 'attribute_weight_distribution']
verbose = True
to_filter = False
hover_thresh = 0.250

base_dir = '/Users/emilywall/git/bias_eval/'
sub_dir = 'real_data_filtered/'
directory = base_dir + sub_dir
data_directory = base_dir + 'data/'
data_file_name = 'bball_top100_decimal.csv'
plot_directory = '../' + sub_dir + 'plots/' + window_method + '/'

all_participants = [1506460542091, 1506629987658, 1507820577674, 1507828088222, 1508856756204, 1508339795840, 1508441006909, 1508778224670, 1509482115747, 1509568819048]
cond_var = [1506460542091, 1506629987658, 1507820577674, 1507828088222, 1508856756204]
cond_size = [1508339795840, 1508441006909, 1508778224670, 1509482115747, 1509568819048]


# read in the original data file
def read_data(directory, file_name):
    dataset = []
    attr_value_map = dict()
    
    with open(directory + file_name, 'rb') as data_file:
        reader = csv.reader(data_file)
        first_line = True
        for row in reader:
            if (not first_line): # don't use the header
                cur_player = bball_player(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22])
                dataset.append(cur_player)
            first_line = False
    
    attrs = dataset[0].get_map().keys()
    attrs.remove('Name')
    
    for i in range(0, len(attrs)):
        cur_attr = attrs[i]
        cur_distr = []
        for j in range(0, len(dataset)):
            cur_val = float(dataset[j].get_map()[cur_attr])
            cur_distr.append(cur_val)
        cur_distr.sort()
        
        attr_value_map[cur_attr] = { 'min': np.amin(cur_distr),
                              'max': np.amax(cur_distr),
                              'mean': np.mean(cur_distr),
                              'variance': np.var(cur_distr),
                              'distribution': cur_distr,
                              'dataType': 'numeric' }
    return dataset, attr_value_map
    
# recreate the set of logs from the file
def recreate_logs(directory, file_name):
    all_logs = json.loads(open(directory + file_name).read())
    filtered_logs = []
    attr_logs = []
    item_logs = []
    help_logs = []
    cat_logs = []

    filtered_hovers = 0
    filtered_clicks = 0
    hover_distr = []
    for i in range(0, len(all_logs)):
        cur_log = all_logs[i]
        if (to_filter):
            if (cur_log['customLogInfo']['eventType'].lower() == 'hover'):
                if (cur_log['customLogInfo']['elapsedTime'] < hover_thresh): 
                    filtered_hovers += 1
                    continue
                else: 
                    hover_distr.append(cur_log['customLogInfo']['elapsedTime'])
                    filtered_logs.append(cur_log)
            elif (cur_log['customLogInfo']['eventType'].lower() == 'click'):
                filtered_clicks += 1
                continue
            else: 
                filtered_logs.append(cur_log)
            
        if ('attribute' in cur_log['eventName'].lower()): 
            attr_logs.append(cur_log)
        elif ('item' in cur_log['eventName'].lower()):
            item_logs.append(cur_log)
        elif ('help' in cur_log['eventName'].lower()):
            help_logs.append(cur_log)
        elif ('category' in cur_log['eventName'].lower()):
            cat_logs.append(cur_log)
        else: 
            print '***error: unknown log', cur_log
            
    if (to_filter):
        hover_distr = sorted(hover_distr)
        print 'filtered out ', filtered_hovers, ' hovers less than ', hover_thresh, ' s; ', len(hover_distr), ' hovers remaining'
        print 'filtered out ', filtered_clicks, ' clicks'
        all_logs = filtered_logs
        #print 'distribution', hover_distr
        #sns.distplot(hover_distr, hist=False, rug=True);
    
    #print file_name, ': attribute (', len(attr_logs), ') + item (', len(item_logs), ') + help (', len(help_logs), ') + category (', len(cat_logs), ') = ', len(all_logs), ' total logs'
    return all_logs, attr_logs, item_logs, help_logs, cat_logs 

# get the subset of logs that happen in the given time frame having 
# the given interaction types
#   time arg can be a Date object; returns all logs that occurred since 'time'
#   time arg can be an integer; returns the last 'time' logs
def get_log_subset(logs, time, interaction_types):
    log_subset = []
    if (not isinstance(time, datetime) and math.isnan(time)):
        time = len(logs)

    if (isinstance(time, datetime)):
        for i in range(0, len(logs)):
            cur_log = logs[i]
            cur_time = datetime.strptime(cur_log['eventTimeStamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
            cur_event_type = cur_log['customLogInfo']['eventType']
            if (cur_time >= time and (len(interaction_types) == 0 or cur_event_type in interaction_types)):
                log_subset.append(logs[i])
    else:
        if (time > len(logs)): 
            time = len(logs)
        num_logs = 0
        i = len(logs) - 1
        while (i >= 0 and num_logs < time):
            cur_log = logs[i]
            cur_event_type = cur_log['customLogInfo']['eventType']
            if (len(interaction_types) == 0 or cur_event_type in interaction_types):
                log_subset.append(logs[i])
                num_logs += 1
            i = i - 1

    return log_subset
    
# get the subset of logs based on the windowing method
def get_logs_by_window_method(window_method, all_logs, item_logs, attr_logs, i, rolling_dist, label_indices, cat_indices, prev_decision):
    attr_log_subset = []
    item_log_subset = []

    if (window_method == 'fixed' and rolling_dist > 0):
        if (i <= len(item_logs)):
            item_log_subset = get_item_log_subset(all_logs[i - rolling_dist : i])
            #item_log_subset = item_logs[i - rolling_dist : i]
        if (i <= len(attr_logs)):
            attr_log_subset = get_attr_log_subset(all_logs[i - rolling_dist : i])
            #attr_log_subset = attr_logs[i - rolling_dist : i]
    elif (window_method == 'classification_v1'):
        if (i in label_indices):
            if (prev_decision == i):
                if (is_attr_log(all_logs[i])):
                    attr_log_subset = [all_logs[i]]
                elif (is_item_log(all_logs[i])): 
                    item_log_subset = [all_logs[i]]
            else: 
                item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
                attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
            prev_decision = i
    elif (window_method == 'classification_v2'):
        if (i in label_indices):
            if (is_attr_log(all_logs[i])):
                attr_log_subset = [all_logs[i]]
            elif (is_item_log(all_logs[i])): 
                item_log_subset = [all_logs[i]]
            prev_decision = i
        else: 
            item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
            attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
    elif (window_method == 'category_v1'):
        if (i in cat_indices):
            if (prev_decision == i):
                if (is_attr_log(all_logs[i])):
                    attr_log_subset = [all_logs[i]]
                elif (is_item_log(all_logs[i])): 
                    item_log_subset = [all_logs[i]]
            else: 
                item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
                attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
            prev_decision = i
    elif (window_method == 'category_v2'):
        if (i in cat_indices):
            if (is_attr_log(all_logs[i])):
                attr_log_subset = [all_logs[i]]
            elif (is_item_log(all_logs[i])): 
                item_log_subset = [all_logs[i]]
            prev_decision = i
        else: 
            item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
            attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
    else: 
        if (i <= len(item_logs)):
            item_log_subset = item_logs[0 : i]
        if (i <= len(attr_logs)):
            attr_log_subset = attr_logs[0 : i]
    
    return attr_log_subset, item_log_subset, prev_decision
    
# get only the attribute logs from the subset of logs
def get_attr_log_subset(logs):
    attr_logs = []
    for i in range(0, len(logs)):
        cur_log = logs[i]
        if ('attribute' in cur_log['eventName'].lower()): 
            attr_logs.append(cur_log)
            
    return attr_logs
    
# get only the item logs from the subset of logs
def get_item_log_subset(logs):
    item_logs = []
    for i in range(0, len(logs)):
        cur_log = logs[i]
        if ('item' in cur_log['eventName'].lower()): 
            item_logs.append(cur_log)
            
    return item_logs
    
# determine if the given log is an attribute log
def is_attr_log(log): 
    if ('attribute' in log['eventName'].lower()): 
        return True
    else: 
        return False
        
# determine if the given log is an item log
def is_item_log(log): 
    if ('item' in log['eventName'].lower()): 
        return True
    else: 
        return False
    
# separate out the set of logs by data item
def get_logs_by_item(logs):
    log_subsets = dict()
    
    for i in range(0, len(logs)):
        cur_log = logs[i]
        cur_data = cur_log['dataItem']
        cur_queue = []
        if (cur_data['Name'][7 : 10] in log_subsets.keys()): 
            cur_queue = log_subsets[cur_data['Name'][7 : 10]]
        cur_queue.append(cur_log)
        log_subsets[cur_data['Name'][7 : 10]] = cur_queue

    return log_subsets
  
# get the expected value from the markov chain
def get_markov_expected_value(N, k):
    try: 
        num = math.pow(N, k) - math.pow((N - 1), k)
        denom = math.pow(N, (k - 1))
        return num / float(denom)
    except OverflowError: 
        print '** Warning: overflow computing markov expected value with N = ' + str(N) + ' and k = ' + str(k)
        return float(N)
    
# get the quantile that the given value belongs to
def get_quantile(quantile_list, value):
    for i in range(0, len(quantile_list)):
        quant_val = quantile_list[i]
        if (i == 0):
            if (value <= quant_val):
                return quant_val
        else:
            if (value <= quant_val and value > quantile_list[i - 1]):
                return quant_val
    return -1;
    
# get the final classification of data points as well as lists of decisions
# classifications is a dictionary where (key, value) = (player id, classification)
# decisions_labels is of the form (index, user classification, actual classification, player id)
# decisions_cat is of the form (index, category)
# decisions_help is of the form (index, 'help')
def get_classifications_and_decisions(all_logs, dataset): 
    classification = dict()
    decisions_labels = []
    decisions_cat = []
    decisions_help = []

    for i in range(0, len(all_logs)):
        cur_log = all_logs[i]
            
        info = cur_log['customLogInfo']
        if ('classification' in info):
            cur_player = cur_log['dataItem']['Name'].replace('Player ', '')
            cur_class = info['classification']
            if (cur_class != 'none'):
                actual_class = get_bball_player(dataset, cur_player).get_full_map()['Position']
                classification[cur_player] = cur_class
                decisions_labels.append((i, cur_class, actual_class, cur_player))
        elif ('category' in cur_log['eventName'].lower()):
            decisions_cat.append((i, cur_log['dataItem'].replace('category_', '')))
        elif ('help' in cur_log['eventName'].lower()):
            decisions_help.append((i, 'help'))
    return classification, decisions_labels, decisions_cat, decisions_help
    
# forward fill the list to remove default -1 values
def forward_fill(arr):
    last_val = arr[0]
    for i in range(0, len(arr)):
        if (arr[i] == -1): 
            arr[i] = last_val
        else: 
            last_val = arr[i]
    return arr
    
# modify array to remove initial -1's then forward fill through other -1's
def remove_defaults(arr, first_decision):
    if (first_decision > -1):
        arr[0 : first_decision] = [0] * first_decision
    arr = forward_fill(arr)
    return arr

# turn the metrics' information into matrices that can be visualized as heat maps
def get_metric_matrices(directory, file_name, dataset):
    # emily todo: fill this in
    # get a list of data points to make sure matrices use same indexing
    data_pts = []
    for i in range(0, len(dataset)):
        cur_player = dataset[i]
        data_pts.append(cur_player.get_map()['Name'])
    data_pts.sort()
        
    # now go through the metrics
    metric_map = dict()
    bias_logs = json.loads(open(directory + file_name).read())
    for i in range(0, len(bias_logs)):
        cur_log = bias_logs[i]['bias_metrics']
        
        for metric_type in cur_log:
            cur_metric = cur_log[metric_type]
            if (metric_type not in metric_map):
                if ('data_point' in metric_type):
                    metric_map[metric_type] = []
                else:
                    metric_map[metric_type] = {}
                    
            if ('data_point_coverage' in metric_type): 
                #print 'dpc'
                visited_pts = cur_metric['info']['visited']
                cur_iter = [0] * len(data_pts)
                for j in range(0, len(visited_pts)):
                    cur_iter[data_pts.index(visited_pts[j])] = 1
                metric_map[metric_type].append(cur_iter)
                    
            elif ('data_point_distribution' in metric_type):
                #print 'dpd'
                visited_pts = cur_metric['info']['distribution_vector'].keys()
                cur_iter = [0] * len(data_pts)
                for j in range(0, len(visited_pts)):
                    cur_iter[data_pts.index(visited_pts[j])] = cur_metric['info']['distribution_vector'][visited_pts[j]]['observed']
                metric_map[metric_type].append(cur_iter)
                
            elif ('attribute_coverage' in metric_type):
                #print 'ac'
                attributes = cur_metric['info']['attribute_vector'].keys()
                for j in range(0, len(attributes)):
                    quantiles = cur_metric['info']['attribute_vector'][attributes[j]]['quantiles']
                    quantiles.sort()
                    if (attributes[j] not in metric_map[metric_type]): 
                        metric_map[metric_type][attributes[j]] = []
                    cur_iter = [0] * cur_metric['info']['attribute_vector'][attributes[j]]['number_of_quantiles']
                    for k in range(0, len(quantiles)):
                        truth_val = cur_metric['info']['attribute_vector'][attributes[j]]['quantile_coverage'][str(quantiles[k])]
                        if (truth_val == True): 
                            cur_iter[k] = 1
                        else: 
                            cur_iter[k] = 0
                    metric_map[metric_type][attributes[j]].append(cur_iter)
                
            '''elif ('attribute_distribution' in metric_type):
                print 'ad'
                
            elif ('attribute_weight_coverage' in metric_type):
                print 'awc'
                
            elif ('attribute_weight_distribution' in metric_type):
                print 'awd'
                
            else: 
                print '** Warning: unrecognized metric type'''
            
    return metric_map
        
# plot the metric over time marking the times where 
# (1) points were classified if marks == 'classifications', or 
# (2) categories were clicked if marks == 'categories' 
def plot_metric(x_values, metric_values, title, x_label, y_label, directory, file_name, decisions, marks, fig_num, verbose):
    if (verbose): 
        print 'Plotting', title
        
    plt.figure(num = fig_num, figsize = (15, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    sub_plot = plt.subplot(1, 1, 1)
    x_values = np.array(x_values)
    metric_values = np.array(metric_values)
    plt.plot(x_values, metric_values, c = '#000000')
    
    if (marks == 'classifications'):
        for i in range(0, len(decisions)):
            tup = decisions[i]
            if (tup[1] == tup[2]): 
                line_width = 4
            else: 
                line_width = 2
            sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = line_width, zorder = 0, clip_on = False)
    elif (marks == 'categories'):
        for i in range(0, len(decisions)): 
            tup = decisions[i]
            sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = 2, zorder = 0, clip_on = False)
    
    # label, save, and clear
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# plot multiple metrics over time in subplots of one figure marking the times where 
# (1) points were classified if marks == 'classifications', or 
# (2) categories were clicked if marks == 'categories' 
def plot_metric_with_subplot(x_values, metric_values, titles, x_label, y_label, directory, file_name, decisions, marks, fig_num, verbose):
    if (verbose): 
        print 'Plotting Subplots'

    plt.figure(num = fig_num, figsize = (15, 60), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(0, len(x_values)):
        cur_x = np.array(x_values[i])
        cur_y = np.array(metric_values[i])
        sub_plot = plt.subplot(len(x_values), 1, i + 1)
        plt.plot(cur_x, cur_y, c = '#000000')
            
        # axis labels and title
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(titles[i])
        
        if (marks == 'classifications'):
            for i in range(0, len(decisions)):
                tup = decisions[i]
                if (tup[1] == tup[2]): 
                    line_width = 4
                else: 
                    line_width = 2
                sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = line_width, zorder = 0, clip_on = False)
        elif (marks == 'categories'):
            for i in range(0, len(decisions)): 
                tup = decisions[i]
                sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = 2, zorder = 0, clip_on = False)
         
    plt.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# plot the metric over time as a heat map style vis, marking the times where 
# (1) points were classified if marks == 'classifications', or 
# (2) categories were clicked if marks == 'categories' 
def plot_metric_heat_map(matrix, title, x_label, y_label, directory, file_name, fig_num):
    if (verbose): # emily todo: fill this in
        print 'Plotting', title
    
    plt.figure(num = fig_num, figsize = (len(matrix), len(matrix[0])), dpi = 60, facecolor = 'w', edgecolor = 'k')
    plt.subplot(1, 1, 1)
    matrix = np.array(matrix)
    heatmap = plt.pcolor(matrix, cmap = 'Blues')
    
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%d' % matrix[y, x], horizontalalignment = 'center', verticalalignment = 'center')
    
    plt.colorbar(heatmap)
    plt.gca().invert_yaxis()
    #plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation = 'vertical')
    #plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    '''plt.xlabel(x_label, fontsize = 200)
    plt.ylabel(y_label, fontsize = 200)
    plt.title(title, fontsize = 200)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)'''
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    font = { 'family': 'normal', 'weight': 'bold', 'size': 200 }
    matplotlib.rc('font', **font)

    #plt.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# compute the average / max / last bias metric value over time for the given participants
# comp_type can be 'avg', 'max', or 'last'
def avg_bias_values(ids, comp_type):
    all_users_bias = dict()
    all_users_summary = dict()
    
    for i in range(0, len(ids)):
        cur_user = str(ids[i])
        
        user_window_methods_bias = dict()
        user_window_methods_summary = dict()
        
        # separate computation for each windowing method
        for j in range(0, len(window_methods)):
            wm = window_methods[j]
            
            # initialize a bunch of dictionaries to store shit
            user_window_methods_bias[wm] = dict()
            user_window_methods_summary[wm] = dict()
            
            user_window_methods_bias[wm][metric_names[0]] = []
            user_window_methods_bias[wm][metric_names[1]] = []
            user_window_methods_bias[wm][metric_names[2]] = dict()
            user_window_methods_bias[wm][metric_names[3]] = dict()
            user_window_methods_bias[wm][metric_names[4]] = dict()
            user_window_methods_bias[wm][metric_names[5]] = dict()
            
            # attribute and attribute weight metrics gotta have another layer of depth per attribute
            user_window_methods_summary[wm][metric_names[2]] = dict()
            user_window_methods_summary[wm][metric_names[3]] = dict()
            user_window_methods_summary[wm][metric_names[4]] = dict()
            user_window_methods_summary[wm][metric_names[5]] = dict()
            for k in range(0, len(attrs)):
                ak = attrs[k]
                user_window_methods_bias[wm][metric_names[2]][ak] = []
                user_window_methods_bias[wm][metric_names[3]][ak] = []
                user_window_methods_bias[wm][metric_names[4]][ak] = []
                user_window_methods_bias[wm][metric_names[5]][ak] = []
            
            cur_file_name = directory + 'user_' + cur_user + '/logs/bias_' + wm + '_' + cur_user + '.json'
            cur_file = json.loads(open(cur_file_name).read())
            
            # go through each computed bias value in time
            for k in range(0, len(cur_file)):
                if (metric_names[0] in cur_file[k]['bias_metrics']):
                    user_window_methods_bias[wm][metric_names[0]].append(cur_file[k]['bias_metrics'][metric_names[0]]['metric_level'])
                if (metric_names[1] in cur_file[k]['bias_metrics']):
                    user_window_methods_bias[wm][metric_names[1]].append(cur_file[k]['bias_metrics'][metric_names[1]]['metric_level'])
                for l in range(0, len(attrs)):
                    al = attrs[l]
                    if (metric_names[2] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[2]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[2]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[3] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[3]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[3]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[4] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[4]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[4]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[5] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[5]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[5]]['info']['attribute_vector'][al]['metric_level'])
            
            # compute summary: average, max, or last
            if (comp_type == 'avg'):
                user_window_methods_summary[wm][metric_names[0]] = np.mean(user_window_methods_bias[wm][metric_names[0]])
                user_window_methods_summary[wm][metric_names[1]] = np.mean(user_window_methods_bias[wm][metric_names[1]])
                for k in range(0, len(attrs)):
                    ak = attrs[k]
                    user_window_methods_summary[wm][metric_names[2]][ak] = np.mean(user_window_methods_bias[wm][metric_names[2]][ak])
                    user_window_methods_summary[wm][metric_names[3]][ak] = np.mean(user_window_methods_bias[wm][metric_names[3]][ak])
                    user_window_methods_summary[wm][metric_names[4]][ak] = np.mean(user_window_methods_bias[wm][metric_names[4]][ak])
                    user_window_methods_summary[wm][metric_names[5]][ak] = np.mean(user_window_methods_bias[wm][metric_names[5]][ak])
            elif (comp_type == 'max'):
                user_window_methods_summary[wm][metric_names[0]] = np.amax(user_window_methods_bias[wm][metric_names[0]])
                user_window_methods_summary[wm][metric_names[1]] = np.amax(user_window_methods_bias[wm][metric_names[1]])
                for k in range(0, len(attrs)):
                    user_window_methods_summary[wm][metric_names[2]][ak] = np.amax(user_window_methods_bias[wm][metric_names[2]][ak])
                    user_window_methods_summary[wm][metric_names[3]][ak] = np.amax(user_window_methods_bias[wm][metric_names[3]][ak])
                    user_window_methods_summary[wm][metric_names[4]][ak] = np.amax(user_window_methods_bias[wm][metric_names[4]][ak])
                    user_window_methods_summary[wm][metric_names[5]][ak] = np.amax(user_window_methods_bias[wm][metric_names[5]][ak])
            else: # comp_type == 'last'
                user_window_methods_summary[wm][metric_names[0]] = user_window_methods_bias[wm][metric_names[0]][-1]
                user_window_methods_summary[wm][metric_names[1]] = user_window_methods_bias[wm][metric_names[1]][-1]
                for k in range(0, len(attrs)):
                    user_window_methods_summary[wm][metric_names[2]][ak] = user_window_methods_bias[wm][metric_names[2]][ak][-1]
                    user_window_methods_summary[wm][metric_names[3]][ak] = user_window_methods_bias[wm][metric_names[3]][ak][-1]
                    user_window_methods_summary[wm][metric_names[4]][ak] = user_window_methods_bias[wm][metric_names[4]][ak][-1]
                    user_window_methods_summary[wm][metric_names[5]][ak] = user_window_methods_bias[wm][metric_names[5]][ak][-1]
            
        all_users_bias[cur_user] = user_window_methods_bias
        all_users_summary[cur_user] = user_window_methods_summary
    
    # now summarize it for participants in each condition
    results_1 = dict()
    results_2 = dict()
    for i in range(0, len(window_methods)):
        wm = window_methods[i]
        results_1[wm] = dict()
        results_2[wm] = dict()
        
        results_1[wm][metric_names[0]] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[0]], all_users_summary[str(cond_size[1])][wm][metric_names[0]], all_users_summary[str(cond_size[2])][wm][metric_names[0]], all_users_summary[str(cond_size[3])][wm][metric_names[0]], all_users_summary[str(cond_size[4])][wm][metric_names[0]]])
        results_2[wm][metric_names[0]] = np.mean([all_users_summary[str(cond_var[0])][wm][metric_names[0]], all_users_summary[str(cond_var[1])][wm][metric_names[0]], all_users_summary[str(cond_var[2])][wm][metric_names[0]], all_users_summary[str(cond_var[3])][wm][metric_names[0]], all_users_summary[str(cond_var[4])][wm][metric_names[0]]])
        results_1[wm][metric_names[1]] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[1]], all_users_summary[str(cond_size[1])][wm][metric_names[1]], all_users_summary[str(cond_size[2])][wm][metric_names[1]], all_users_summary[str(cond_size[3])][wm][metric_names[1]], all_users_summary[str(cond_size[4])][wm][metric_names[1]]])
        results_2[wm][metric_names[1]] = np.mean([all_users_summary[str(cond_var[0])][wm][metric_names[1]], all_users_summary[str(cond_var[1])][wm][metric_names[1]], all_users_summary[str(cond_var[2])][wm][metric_names[1]], all_users_summary[str(cond_var[3])][wm][metric_names[1]], all_users_summary[str(cond_var[4])][wm][metric_names[1]]])
        
        results_1[wm][metric_names[2]] = dict()
        results_2[wm][metric_names[2]] = dict()
        results_1[wm][metric_names[3]] = dict()
        results_2[wm][metric_names[3]] = dict()
        results_1[wm][metric_names[4]] = dict()
        results_2[wm][metric_names[4]] = dict()
        results_1[wm][metric_names[5]] = dict()
        results_2[wm][metric_names[5]] = dict()
        
        for j in range(0, len(attrs)):
            aj = attrs[j]
            results_1[wm][metric_names[2]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[2]][aj]])
            results_2[wm][metric_names[2]][aj] = np.mean([all_users_summary[str(cond_var[0])][wm][metric_names[2]][aj], all_users_summary[str(cond_var[1])][wm][metric_names[2]][aj], all_users_summary[str(cond_var[2])][wm][metric_names[2]][aj], all_users_summary[str(cond_var[3])][wm][metric_names[2]][aj], all_users_summary[str(cond_var[4])][wm][metric_names[2]][aj]])
            results_1[wm][metric_names[3]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[3]][aj]])
            results_2[wm][metric_names[3]][aj] = np.mean([all_users_summary[str(cond_var[0])][wm][metric_names[3]][aj], all_users_summary[str(cond_var[1])][wm][metric_names[3]][aj], all_users_summary[str(cond_var[2])][wm][metric_names[3]][aj], all_users_summary[str(cond_var[3])][wm][metric_names[3]][aj], all_users_summary[str(cond_var[4])][wm][metric_names[3]][aj]])
            results_1[wm][metric_names[4]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[4]][aj]])
            results_2[wm][metric_names[4]][aj] = np.mean([all_users_summary[str(cond_var[0])][wm][metric_names[4]][aj], all_users_summary[str(cond_var[1])][wm][metric_names[4]][aj], all_users_summary[str(cond_var[2])][wm][metric_names[4]][aj], all_users_summary[str(cond_var[3])][wm][metric_names[4]][aj], all_users_summary[str(cond_var[4])][wm][metric_names[4]][aj]])
            results_1[wm][metric_names[5]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[5]][aj]])
            results_2[wm][metric_names[5]][aj] = np.mean([all_users_summary[str(cond_var[0])][wm][metric_names[5]][aj], all_users_summary[str(cond_var[1])][wm][metric_names[5]][aj], all_users_summary[str(cond_var[2])][wm][metric_names[5]][aj], all_users_summary[str(cond_var[3])][wm][metric_names[5]][aj], all_users_summary[str(cond_var[4])][wm][metric_names[5]][aj]])
            
    results = dict()
    results['condition_1_size'] = results_1
    results['condition_2_varying'] = results_2
    #print results
    return results #emily TODO write this to a file # emily TODO debug this for max and last
    
# representation of a basketball player
class bball_player:
    def __init__(self, player, player_anon, team, pos, rand, avg_3p_att, avg_3p_made, avg_ast, avg_blks, avg_fg_att, avg_fg_made, avg_ft_att, avg_ft_made, avg_min, avg_pf, avg_pts, avg_or, avg_st, avg_tr, avg_to, games, height, weight):
        self.player = player
        self.player_anon = player_anon
        self.team = team
        self.pos = pos
        self.rand = rand
        self.avg_3p_att = avg_3p_att
        self.avg_3p_made = avg_3p_made
        self.avg_ast = avg_ast
        self.avg_blks = avg_blks
        self.avg_fg_att = avg_fg_att
        self.avg_fg_made = avg_fg_made
        self.avg_ft_att = avg_ft_att
        self.avg_ft_made = avg_ft_made
        self.avg_min = avg_min
        self.avg_pf = avg_pf
        self.avg_pts = avg_pts
        self.avg_or = avg_or
        self.avg_st = avg_st
        self.avg_tr = avg_tr
        self.avg_to = avg_to
        self.games = games
        self.height = height
        self.weight = weight
        
    def get_map(self):
        return {'Avg. 3-Pointers Att.': self.avg_3p_att, 'Avg. 3-Pointers Made': self.avg_3p_made, 'Avg. Assists': self.avg_ast, 'Avg. Blocks': self.avg_blks, 'Avg. Field Goals Att.': self.avg_fg_att, 'Avg. Field Goals Made': self.avg_fg_made, 'Avg. Free Throws Att.': self.avg_ft_att, 'Avg. Free Throws Made': self.avg_ft_made, 'Avg. Minutes': self.avg_min, 'Avg. Personal Fouls': self.avg_pf, 'Avg. Points': self.avg_pts, 'Avg. Offensive Rebounds': self.avg_or, 'Avg. Steals': self.avg_st, 'Avg. Total Rebounds': self.avg_tr, 'Avg. Turnovers': self.avg_to, 'Games Played': self.games, 'Height (Inches)': self.height, 'Weight (Pounds)': self.weight, 'Name': self.player_anon, 'Rand': self.rand}
                
    def get_full_map(self):
        return {'Avg. 3-Pointers Att.': self.avg_3p_att, 'Avg. 3-Pointers Made': self.avg_3p_made, 'Avg. Assists': self.avg_ast, 'Avg. Blocks': self.avg_blks, 'Avg. Field Goals Att.': self.avg_fg_att, 'Avg. Field Goals Made': self.avg_fg_made, 'Avg. Free Throws Att.': self.avg_ft_att, 'Avg. Free Throws Made': self.avg_ft_made, 'Avg. Minutes': self.avg_min, 'Avg. Personal Fouls': self.avg_pf, 'Avg. Points': self.avg_pts, 'Avg. Offensive Rebounds': self.avg_or, 'Avg. Steals': self.avg_st, 'Avg. Total Rebounds': self.avg_tr, 'Avg. Turnovers': self.avg_to, 'Games Played': self.games, 'Height (Inches)': self.height, 'Weight (Pounds)': self.weight, 'Name': self.player_anon, 'Team': self.team, 'Position': self.pos, 'Name (Real)': self.player, 'Rand': self.rand}
    
# get the bball player by the name attribute
def get_bball_player(players, name):
    for i in range(0, len(players)):
        if (players[i].get_map()['Name'] == name): 
            return players[i]

    print '*** Unable to locate player', name
    return -1

# this block is for testing util functions
if __name__ == '__main__':
    '''
    print directory
    dataset, attr_value_map = read_data('/Users/emilywall/git/bias_eval/data/', 'bball_top100_decimal.csv')
    metric_matrix = get_metric_matrices(directory, 'user_1506629987658/logs/bias_category_v1_1506629987658.json', dataset)
    #print metric_matrix['attribute_coverage']
    plot_metric_heat_map(metric_matrix['data_point_coverage'], 'Data Point Coverage', 'Data Point', 'Time (Interactions)', directory + 'user_1506629987658/plots/', 'matrix_test5.png', 1)
    #plot_metric_heat_map(metric_matrix['data_point_distribution'], 'Data Point Distribution', 'Data Point', 'Time (Interactions)', directory + 'user_1506629987658/plots/', 'matrix_test5.png', 1)
    #plot_metric_heat_map(metric_matrix['attribute_coverage']['Weight (Pounds)'], 'Attribute Coverage (Weight)', 'Quantile', 'Time (Interactions)', directory + 'user_1506629987658/plots/', 'matrix_test5.png', 1)
    # emily: this isn't saving the fig?
    # emily remove this after testing
    '''
    avg_bias_values(all_participants, 'avg')