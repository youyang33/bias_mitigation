#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:36:49 2017

This script is used to compute bias metrics from a log file.
The results will be re-written to a file and plotted, after 
which separate analysis scripts can be applied to the 
resulting bias levels.

@author: emilywall
"""

import numpy as np
import bias_util
import math
import json
import sys
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
from pprint import pprint
from scipy.stats import chisquare
from scipy.stats import ks_2samp

class bias(object):

    # initialize object
    def __init__(self, directory, in_file_name, out_file_name, data_directory, data_file_name):
        self.directory = directory
        self.in_file_name = in_file_name
        self.out_file_name = out_file_name
        
        self.data_directory = data_directory
        self.data_file_name = data_file_name
        
        self.dpc_logs = []
        self.dpd_logs = []
        self.ac_logs = []
        self.ad_logs = []
        self.awc_logs = []
        self.awd_logs = []
        
        self.dataset, self.attr_value_map = bias_util.read_data(data_directory, data_file_name)
        self.all_logs, self.attr_logs, self.item_logs, self.help_logs, self.cat_logs = bias_util.recreate_logs(directory, in_file_name)

    # get the interaction logs
    def get_logs(self): 
        return self.all_logs
    
    # get the item interaction logs
    def get_item_logs(self):
        return self.item_logs
        
    # get the attribute interaction logs
    def get_attr_logs(self):
        return self.attr_logs
        
    # get the dataset
    def get_dataset(self):
        return self.dataset
        
    # simulate computing bias metrics through time; write the results to a file
    # window_method 'all' computes bias metrics at each time step using full interaction history up to that point
    # window_method 'fixed' computes bias metrics at each time step using fixed rolling window size
    # window_method 'classification_v1' computes bias metrics at each time step where a classification was made using the interactions since the previous classification
    # window_method 'classification_v2' computes bias metrics at each time step using the interactions since the previous classification
    # window_method 'category_v1' computes bias metrics at each time step where one of the categories was clicked using the interactions since the last category was clicked
    # window_method 'category_v2' computes bias metrics at each time step using the interactions since the last category was clicked
    def simulate_bias_computation(self, plot_directory, time, interaction_types, num_quantiles, min_weight, max_weight, window_method, rolling_dist, marks, fig_num, verbose):
        dpc_metric = [-1] * len(self.all_logs)
        dpd_metric = [-1] * len(self.all_logs)
        ac_metric = dict()
        ad_metric = dict()
        awc_metric = dict()
        awd_metric = dict()
        for key in self.attr_value_map.keys(): 
            ac_metric[key] = [-1] * len(self.all_logs)
            ad_metric[key] = [-1] * len(self.all_logs)
            awc_metric[key] = [-1] * len(self.all_logs)
            awd_metric[key] = [-1] * len(self.all_logs)
        
        # classifications is a map from (id, final classification)
        # decisions_labels is a list of tuples (log index, user classification, actual classification)
        
        classifications, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(self.all_logs, self.dataset)
        label_indices = [int(tup[0]) for tup in decisions_labels]
        if (label_indices[len(label_indices) - 1] != len(self.all_logs) - 1):
            label_indices.append(len(self.all_logs) - 1)
        cat_indices = [int(tup[0]) for tup in decisions_cat]
        if (len(cat_indices) > 0 and cat_indices[len(cat_indices) - 1] != len(self.all_logs) - 1): 
            cat_indices.append(len(self.all_logs) - 1)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        f_out = open(self.directory + self.out_file_name, 'w+')
        f_out.write('[')
        last_line = None
        last_written_line = None
        
        if (not window_method in bias_util.window_methods):
            print '**Error: Invalid window method.'
            sys.exit(0)
        elif (window_method == 'fixed' and rolling_dist > 0):
            start_iter = rolling_dist
        elif (window_method == 'classification_v1' and len(decisions_labels) > 0):
            start_iter = decisions_labels[0][0]
        elif (window_method == 'classification_v2' and len(decisions_labels) > 0):
            start_iter = 0
        else: 
            start_iter = 1
            
        prev_decision = 0
            
            
        # iterate through the logs
        for i in range(start_iter, len(self.all_logs)):
            if (verbose):
                print 'Interaction', i
            if (last_line != None):
                f_out.write(last_line + ',')
                last_written_line = last_line
                last_line = None
            line_contents = []

            # figure out which set of logs to use for this iteration based on the window method
            attr_log_set, item_log_set, prev_decision = bias_util.get_logs_by_window_method(window_method, self.all_logs, self.item_logs, self.attr_logs, i, rolling_dist, label_indices, cat_indices, prev_decision)
            if (window_method == 'classification_v1' and i not in label_indices): 
                continue
            if (window_method == 'category_v1' and i not in cat_indices):
                continue
            
            # compute the metrics that use data item logs
            if (len(item_log_set) > 0):
                cur_dpc = self.compute_data_point_coverage(item_log_set, time, interaction_types, verbose)
                self.dpc_logs.append(cur_dpc)
                if (cur_dpc is None):
                    dpc_metric[i] = -1
                else:
                    dpc_metric[i] = cur_dpc['metric_level']
                line_contents.append({'metric': 'data_point_coverage', 'log': cur_dpc})
                
                cur_dpd = self.compute_data_point_distribution(item_log_set, time, interaction_types, verbose)
                self.dpd_logs.append(cur_dpd)
                if (cur_dpc is None):
                    dpd_metric[i] = -1
                else:
                    dpd_metric[i] = cur_dpd['metric_level']
                line_contents.append({'metric': 'data_point_distribution', 'log': cur_dpd})
                
                cur_ac = self.compute_attribute_coverage(item_log_set, time, interaction_types, num_quantiles, verbose)
                self.ac_logs.append(cur_ac)
                if (cur_ac is not None):
                    for attribute in cur_ac['info']['attribute_vector'].keys():
                        cur_metric = cur_ac['info']['attribute_vector'][attribute]['metric_level']
                        ac_metric[attribute][i] = cur_metric
                line_contents.append({'metric': 'attribute_coverage', 'log': cur_ac})
                
                cur_ad = self.compute_attribute_distribution(item_log_set, time, interaction_types, verbose)
                self.ad_logs.append(cur_ad)
                if (cur_ad is not None):
                    for attribute in cur_ad['info']['attribute_vector'].keys():
                        cur_metric = cur_ad['info']['attribute_vector'][attribute]['metric_level']
                        ad_metric[attribute][i] = cur_metric
                line_contents.append({'metric': 'attribute_distribution', 'log': cur_ad})
                    
            # compute the metrics that use attribute logs
            if (len(attr_log_set) > 0):
                cur_awc = self.compute_attribute_weight_coverage(attr_log_set, time, interaction_types, num_quantiles, min_weight, max_weight, verbose)
                self.awc_logs.append(cur_awc)
                if (cur_awc is not None):
                    for attribute in cur_awc['info']['attribute_vector'].keys():
                        cur_metric = cur_awc['info']['attribute_vector'][attribute]['metric_level']
                        awc_metric[attribute][i] = cur_metric
                line_contents.append({'metric': 'attribute_weight_coverage', 'log': cur_awc})
                
                cur_awd = self.compute_attribute_weight_distribution(attr_log_set, time, interaction_types, min_weight, max_weight, verbose)
                self.awd_logs.append(cur_awd)
                if (cur_awd is not None):
                    for attribute in cur_awd['info']['attribute_vector'].keys():
                        cur_metric = cur_awd['info']['attribute_vector'][attribute]['metric_level']
                        awd_metric[attribute][i] = cur_metric
                line_contents.append({'metric': 'attribute_weight_distribution', 'log': cur_awd})
                    
            if (len(line_contents) > 0):
                line = ''
                for j in range(0, len(line_contents)):
                    cur_log_info = line_contents[j]
                    line = line + json.dumps(cur_log_info['metric']) + ':' + json.dumps(cur_log_info['log'])
                    if (j != len(line_contents) - 1):
                        line = line + ','
                        
                num_interactions = i
                if (window_method == 'fixed' and rolling_dist > -1): 
                    num_interactions = rolling_dist
                elif (window_method == 'classification_v1' and label_indices.index(i) > 0):
                    num_interactions = i - label_indices[label_indices.index(i) - 1]
                elif (window_method == 'classification_v2'):
                    num_interactions = i - prev_decision + 1
                last_line = '{"computing_at_interaction":"' + str(i) + '/' + str(len(self.all_logs)) + '","num_interactions":' + str(num_interactions) + ',"window_method":"' + window_method + '",'
                if (window_method == 'fixed'):
                    last_line += '"rolling_distance":' + str(rolling_dist) + ','
                if (window_method == 'classification_v1' or window_method == 'classification_v2'):
                    last_line += '"label_indices":' + str(label_indices) + ','
                last_line += '"bias_metrics":{' + line + '}}'
        if (last_line == None):
            # go back and remove the comma from the end of the file
            f_out.close()
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            file = open(self.directory + self.out_file_name, 'r+')
            file.seek(0, os.SEEK_END)
            pos = file.tell() - 1
            while pos > 0 and file.read(1) != '\n':
                pos -= 1
                file.seek(pos, os.SEEK_SET)
            if pos > 0:
                file.seek(pos, os.SEEK_SET)
                file.truncate()
            file.close()
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            f_out = open(self.directory + self.out_file_name, 'a')
            f_out.write(last_written_line[0 : len(last_written_line)])
        else: 
            f_out.write(last_line)  
        f_out.write(']')
        f_out.close()
        
        # get the average computed value for each metric
        avg_values = dict()
        avg_values['DPC'] = np.mean(np.array([x for x in dpc_metric if x > -1]).astype(np.float))
        avg_values['DPD'] = np.mean(np.array([x for x in dpd_metric if x > -1]).astype(np.float))
        for key in self.attr_value_map.keys():
            avg_values['AC_' + key.replace(' ', '')] = np.mean(np.array([x for x in ac_metric[key] if x > -1]).astype(np.float))
            avg_values['AD_' + key.replace(' ', '')] = np.mean(np.array([x for x in ad_metric[key] if x > -1]).astype(np.float))
            avg_values['AWC_' + key.replace(' ', '')] = np.mean(np.array([x for x in awc_metric[key] if x > -1]).astype(np.float))
            avg_values['AWD_' + key.replace(' ', '')] = np.mean(np.array([x for x in awd_metric[key] if x > -1]).astype(np.float))
        
#        if (verbose):
#            print '**** For interaction logs: ', self.in_file_name
#            print '> Average values: ', avg_values
        
        # fill in the -1 default values in the metric arrays
        first_decision_point = dpc_metric.index(filter(lambda x : x != -1, dpc_metric)[0])
        dpc_metric = bias_util.remove_defaults(dpc_metric, first_decision_point)
        dpd_metric = bias_util.remove_defaults(dpd_metric, first_decision_point)
        for attribute in self.attr_value_map.keys():
            ac_metric[attribute] = bias_util.remove_defaults(ac_metric[attribute], first_decision_point)
            ad_metric[attribute] = bias_util.remove_defaults(ad_metric[attribute], first_decision_point)
            awc_metric[attribute] = bias_util.remove_defaults(awc_metric[attribute], first_decision_point)
            awd_metric[attribute] = bias_util.remove_defaults(awd_metric[attribute], first_decision_point)
        
        # plot all of the results
        x_label = 'Interactions'
        y_label = 'Metric Value'
        x_values = np.arange(1, len(self.all_logs) + 1)
        user_id = self.in_file_name.replace('interactions_', '').replace('.json', '')
        marker_decisions = []
        if (marks == 'classifications'):
            marker_decisions = decisions_labels
        elif (marks == 'categories'):
            marker_decisions = decisions_cat
        
        # plot the data point metrics
        bias_util.plot_metric(x_values, dpc_metric, 'DPC - Pilot', x_label, y_label, plot_directory, user_id + '_dpc.png', marker_decisions, marks, fig_num, verbose)
        bias_util.plot_metric(x_values, dpd_metric, 'DPD - Pilot', x_label, y_label, plot_directory, user_id + '_dpd.png', marker_decisions, marks, fig_num + 1, verbose)
        
        # plot one series of subplots plots for each attribute and attribute weight metric
        attributes = ac_metric.keys()
        attr_x_values = []
        attr_y_values = []
        titles = []
        for attribute in attributes:
            attr_x_values.append(x_values)
            attr_y_values.append(ac_metric[attribute])
            titles.append('AC (' + attribute + ') - Pilot')
        bias_util.plot_metric_with_subplot(attr_x_values, attr_y_values, titles, x_label, y_label, plot_directory, user_id + '_ac' + '.png', marker_decisions, marks, fig_num + 2, verbose)
            
        attr_y_values = []
        titles = []
        for attribute in attributes:
            attr_y_values.append(ad_metric[attribute])
            titles.append('AD (' + attribute + ') - Pilot')
        bias_util.plot_metric_with_subplot(attr_x_values, attr_y_values, titles, x_label, y_label, plot_directory, user_id + '_ad' + '.png', marker_decisions, marks, fig_num + 3, verbose)
            
        attr_y_values = []
        titles = []
        for attribute in attributes:
            attr_y_values.append(awc_metric[attribute])
            titles.append('AWC (' + attribute + ') - Pilot')
        bias_util.plot_metric_with_subplot(attr_x_values, attr_y_values, titles, x_label, y_label, plot_directory, user_id + '_awc' + '.png', marker_decisions, marks, fig_num + 4, verbose)
            
        attr_y_values = []
        titles = []
        for attribute in attributes:
            attr_y_values.append(awd_metric[attribute])
            titles.append('AWD (' + attribute + ') - Pilot')
        bias_util.plot_metric_with_subplot(attr_x_values, attr_y_values, titles, x_label, y_label, plot_directory, user_id + '_awd' + '.png', marker_decisions, marks, fig_num + 5, verbose)
            
        # plot individual plots for each attribute and attribute weight metric per attribute
#        for attribute in ac_metric.keys():
#            bias_util.plot_metric(x_values, ac_metric[attribute], 'Attribute Coverage (' + attribute + ') - Pilot', x_label, y_label, plot_directory, user_id + '_ac_' + attribute + '.png', marker_decisions, marks, verbose)
#            bias_util.plot_metric(x_values, ad_metric[attribute], 'Attribute Distribution (' + attribute + ') - Pilot', x_label, y_label, plot_directory, user_id + '_ad_' + attribute + '.png', marker_decisions, marks, verbose)
#            bias_util.plot_metric(x_values, awc_metric[attribute], 'Attribute Weight Coverage (' + attribute + ') - Pilot', x_label, y_label, plot_directory, user_id + '_awc_' + attribute + '.png', marker_decisions, marks, verbose)
#            bias_util.plot_metric(x_values, awd_metric[attribute], 'Attribute Weight Distribution (' + attribute + ') - Pilot', x_label, y_label, plot_directory, user_id + '_awd_' + attribute + '.png', marker_decisions, marks, verbose)
        
    # compute the data point coverage metric
    def compute_data_point_coverage(self, log_set, time, interaction_types, verbose):
        log_subset = bias_util.get_log_subset(log_set, time, interaction_types)
        
        current_log = dict()
        current_log['bias_type'] = 'data_point_coverage'
        current_log['current_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        current_log['number_of_logs'] = len(log_subset)
        current_log['interaction_types'] = interaction_types
        current_log['time_window'] = time
        current_log_info = dict()

        # no interactions
        if (len(log_subset) == 0): 
            return
        
        unique_ids = set()
        for i in range(0, len(log_subset)):
            cur_log = log_subset[i]
            cur_id = cur_log['dataItem']['Name'][7 : 10]
            unique_ids.add(cur_id)
            
        num_unique_expected = bias_util.get_markov_expected_value(len(self.dataset), len(log_subset))
        percent_unique = len(unique_ids) / num_unique_expected

        current_log_info['visited'] = list(unique_ids)
        current_log_info['covered_data'] = len(unique_ids)
        current_log_info['expected_covered_data'] = num_unique_expected
        if (len(log_subset) == 0): # 100% unique if no interactions
            percent_unique = 1
        current_log_info['percentage'] = percent_unique
        current_log['info'] = current_log_info

        # lower percent of unique interactions -> higher level of bias
        metric_level = 1.0 - min(1, percent_unique)
        current_log['metric_level'] = metric_level

        if (verbose): 
            print '- data point coverage metric: ', metric_level
        #pprint(current_log)
        return current_log
    
    # compute the data point distribution metric
    def compute_data_point_distribution(self, log_set, time, interaction_types, verbose):
        orig_interaction_subset = bias_util.get_log_subset(log_set, time, interaction_types)
        interaction_subset_by_data = bias_util.get_logs_by_item(bias_util.get_log_subset(log_set, time, interaction_types))
        
        current_log = dict()
        current_log['bias_type'] = 'data_point_distribution'
        current_log['current_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        current_log['number_of_logs'] = len(orig_interaction_subset)
        current_log['interaction_types'] = interaction_types
        current_log['time_window'] = time
        current_log_info = dict()
        current_log_info['distribution_vector'] = dict()

        # no interactions
        if (len(orig_interaction_subset) == 0):
            return

        # compare observed and expected number of interactions for each data point
        max_obs = 0
        expected = 1.0 * len(orig_interaction_subset) / len(self.dataset)
        chi_sq = 0
        obs_array = []
        exp_array = []
        for i in range(0, len(self.dataset)):
            cur_data = self.dataset[i]
            observed = 0
                
            if (cur_data.player_anon in interaction_subset_by_data.keys()):
                observed = len(interaction_subset_by_data[cur_data.player_anon])
            
            obs_array.append(observed)
            exp_array.append(expected)
                
            sq_diff = math.pow(observed - expected, 2) / expected
            if (observed > max_obs): 
                max_obs = observed
            current_log_info['distribution_vector'][cur_data.player_anon] = { 'data_item': cur_data.player_anon, 'observed': observed, 'expected': expected, 'diff': sq_diff }
            chi_sq = chi_sq + sq_diff

        deg_free = len(self.dataset) - 1
        res = chisquare(obs_array, f_exp = exp_array)
        prob = res[1]
        current_log_info['chi_squared'] = chi_sq
        current_log_info['degrees_of_freedom'] = deg_free
        current_log_info['max_observed_interactions'] = max_obs
        current_log['info'] = current_log_info
        current_log['metric_level'] = prob
        
        if (verbose): 
            print '- data point distribution metric: ', prob
        #pprint(current_log)
        return current_log
        
    # compute the attribute coverage metric
    def compute_attribute_coverage(self, log_set, time, interaction_types, num_quantiles, verbose):
        log_subset = bias_util.get_log_subset(log_set, time, interaction_types)
 
        current_log = dict()
        current_log['bias_type'] = 'attribute_coverage'
        current_log['current_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        current_log['number_of_logs'] = len(log_subset)
        current_log['interaction_types'] = interaction_types
        current_log['time_window'] = time
        current_log_info = dict()
        current_log_info['attribute_vector'] = dict()
        
        # no interactions
        if (len(log_subset) == 0):
            return
 
        # compare interactions to quantiles
        max_metric_value = 0
        for attribute in self.attr_value_map.keys():
            full_dist = self.attr_value_map[attribute]['distribution']
            
            # make sure it's numeric (ignoring categorical for now)
            if (self.attr_value_map[attribute]['dataType'] == 'numeric'):
                quantiles = dict()
                quantile_list = []
                for i in range(0, num_quantiles):
                    if (i != num_quantiles - 1):
                        quant_val = full_dist[int(math.floor((i + 1) * len(self.dataset) / num_quantiles) - 1)]
                    else:
                        quant_val = full_dist[len(full_dist) - 1]
                    quantile_list.append(quant_val)
                    quantiles[quant_val] = 0
 
                # figure out distribution of interactions
                for i in range(0, len(log_subset)):
                    if (attribute != 'Rand'):
                        cur_val = float(log_subset[i]['dataItem'][attribute])
                    else: 
                        cur_val = float(bias_util.get_bball_player(self.dataset, log_subset[i]['dataItem']['Name'].replace('Player ', '')).get_map()['Rand'])
                    # figure out which quantile it belongs to
                    which_quantile = bias_util.get_quantile(quantile_list, cur_val)
                    quantiles[which_quantile] += 1
 
                current_log_info['attribute_vector'][attribute] = dict()
                current_log_info['attribute_vector'][attribute]['quantiles'] = quantile_list
                current_log_info['attribute_vector'][attribute]['quantile_coverage'] = dict()
                covered_quantiles = 0
                for i in range(0, len(quantile_list)):
                    quant_val = quantile_list[i]
                    if (quantiles[quant_val] > 0):
                        covered_quantiles += 1
                        current_log_info['attribute_vector'][attribute]['quantile_coverage'][quant_val] = True
                    else:
                        current_log_info['attribute_vector'][attribute]['quantile_coverage'][quant_val] = False
 
                expected_covered_quantiles = bias_util.get_markov_expected_value(num_quantiles, len(log_subset))
                percent_unique = covered_quantiles / expected_covered_quantiles; 
                current_log_info['attribute_vector'][attribute]['number_of_quantiles'] = num_quantiles; 
                current_log_info['attribute_vector'][attribute]['covered_quantiles'] = covered_quantiles; 
                current_log_info['attribute_vector'][attribute]['expected_covered_quantiles'] = expected_covered_quantiles; 
                if (len(log_subset) == 0): # 100% unique if no interactions
                    percent_unique = 1
                current_log_info['attribute_vector'][attribute]['percentage'] = percent_unique
                # lower percent of unique interactions -> higher level of bias
                metric_val = 1.0 - min(1, percent_unique)
                if (metric_val > max_metric_value):
                    max_metric_value = metric_val
                current_log_info['attribute_vector'][attribute]['metric_level'] = metric_val
 
        current_log['info'] = current_log_info
        # in this case, the metric level is the max metric value over all the attributes
        current_log['metric_level'] = max_metric_value
        
        if (verbose): 
            print '- attribute coverage metric: ', max_metric_value
        #pprint(current_log)
        return current_log
     
    # compute the attribute distribution metric
    def compute_attribute_distribution(self, log_set, time, interaction_types, verbose):
        log_subset = bias_util.get_log_subset(log_set, time, interaction_types)

        current_log = dict()
        current_log['bias_type'] = 'attribute_distribution'
        current_log['current_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        current_log['number_of_logs'] = len(log_subset)
        current_log['interaction_types'] = interaction_types
        current_log['time_window'] = time
        current_log_info = dict()
        current_log_info['attribute_vector'] = dict()

        # no interactions
        if (len(log_subset) == 0):
            return
        
        # compare interactions to full distribution of each attribute
        max_metric_value = 0
        for attribute in self.attr_value_map.keys():
            full_dist = self.attr_value_map[attribute]['distribution']
            # make sure it's numeric (ignoring categorical for now)
            if (self.attr_value_map[attribute]['dataType'] == 'numeric'):
                # figure out distribution of interactions
                int_dist = []
                for i in range(0, len(log_subset)):
                    if (attribute != 'Rand'):
                        cur_val = float(log_subset[i]['dataItem'][attribute])
                    else: 
                        cur_val = float(bias_util.get_bball_player(self.dataset, log_subset[i]['dataItem']['Name'].replace('Player ', '')).get_map()['Rand'])
                    
                    int_dist.append(cur_val)
                int_dist.sort()
                KS = ks_2samp(full_dist, int_dist)
                if (1 - float(KS[1]) > max_metric_value):
                    max_metric_value = 1 - float(KS[1])

                current_log_info['attribute_vector'][attribute] = dict()
                current_log_info['attribute_vector'][attribute]['ks'] = KS[0]
                current_log_info['attribute_vector'][attribute]['metric_level'] = 1 - KS[1]

        current_log['info'] = current_log_info
        current_log['metric_level'] = max_metric_value
         
        if (verbose): 
            print '- attribute distribution metric: ', max_metric_value
        #pprint(current_log)
        return current_log
        
    # compute the attribute weight coverage metric
    def compute_attribute_weight_coverage(self, log_set, time, interaction_types, num_quantiles, min_weight, max_weight, verbose):
        weight_vector_subset = bias_util.get_log_subset(log_set, time, [])

        current_log = dict()
        current_log['bias_type'] = 'attribute_weight_coverage'
        current_log['current_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        current_log['number_of_logs'] = len(weight_vector_subset)
        current_log['interaction_types'] = interaction_types
        current_log['time_window'] = time
        current_log_info = dict()
        current_log_info['attribute_vector'] = dict()
  
        # no interactions
        if (len(weight_vector_subset) == 0):
            return

        quantile_map = dict() # for counting number of weights that occur in each quantile
        change_map = dict() # for counting number of times each attribute's weight actually changes with each new vector
        for attribute in self.attr_value_map.keys():
            quantile_map[attribute] = dict()
            change_map[attribute] = 1

        # define quantiles
        quantile_list = []
        for i in range(0, num_quantiles):
            quant_val = max_weight
            if (i != num_quantiles - 1):
                quant_val = float(min_weight + (i + 1) * (max_weight - min_weight) / float(num_quantiles))

            quantile_list.append(quant_val)
            for attribute in self.attr_value_map.keys():
                quantile_map[attribute][quant_val] = 0

        # count quantiles interacted with for all the weight vectors
        # only counts when the weight actually changed
        for i in range(0, len(weight_vector_subset)):
            cur_weight_vector = weight_vector_subset[i]
            old_vector = cur_weight_vector['oldWeight']
            new_vector = cur_weight_vector['newWeight']

            for attribute in self.attr_value_map.keys():
                if (attribute == 'Rand'):
                    continue # skip rand here -- weight doesn't make sense for random variable
                # count which quantile the weight falls in
                # if it's the first weight vector, check old vector and new vector
                if (i == 0):
                    attr_weight = old_vector[attribute]
                    # figure out which quantile it belongs to
                    which_quantile = bias_util.get_quantile(quantile_list, attr_weight)
                    quantile_map[attribute][which_quantile] += 1

                # for all other weight vectors, check if the weight actually changed
                if (old_vector[attribute] != new_vector[attribute]):
                    change_map[attribute] += 1
                    attr_weight = new_vector[attribute]
                    # figure out which quantile it belongs to
                    which_quantile = bias_util.get_quantile(quantile_list, attr_weight)
                    quantile_map[attribute][which_quantile] += 1

        # compute metric values
        max_metric_value = 0
        for attribute in self.attr_value_map.keys():
            if (attribute == 'Rand'):
                continue # skip rand here -- weight doesn't make sense for random variable
            current_log_info['attribute_vector'][attribute] = dict()
            current_log_info['attribute_vector'][attribute]['quantiles'] = quantile_list
            current_log_info['attribute_vector'][attribute]['quantile_coverage'] = dict()
            covered_quantiles = 0
            for i in range(0, len(quantile_list)):
                quant_val = quantile_list[i]
                if (quantile_map[attribute][quant_val] > 0):
                    covered_quantiles  += 1 
                    current_log_info['attribute_vector'][attribute]['quantile_coverage'][quant_val] = True
                else: 
                    current_log_info['attribute_vector'][attribute]['quantile_coverage'][quant_val] = False

            expected_covered_quantiles = bias_util.get_markov_expected_value(num_quantiles, change_map[attribute])
            percent_unique = covered_quantiles / expected_covered_quantiles
            current_log_info['attribute_vector'][attribute]['number_of_quantiles'] = num_quantiles
            current_log_info['attribute_vector'][attribute]['covered_quantiles'] = covered_quantiles
            current_log_info['attribute_vector'][attribute]['expected_covered_quantiles'] = expected_covered_quantiles
            current_log_info['attribute_vector'][attribute]['percentage'] = percent_unique
            # lower percent of unique interactions -> higher level of bias
            metric_val = 1.0 - min(1, percent_unique)
            if (metric_val > max_metric_value): 
                max_metric_value = metric_val;
            current_log_info['attribute_vector'][attribute]['metric_level'] = metric_val

        current_log['info'] = current_log_info
        # in this case, the metric level is the max metric value over all the attributes
        current_log['metric_level'] = max_metric_value
        self.awc_logs.append(current_log)
         
        if (verbose): 
            print '- attribute weight coverage metric: ', max_metric_value
        #pprint(current_log)
        return current_log
    
    # compute the attribute weight distribution metric
    def compute_attribute_weight_distribution(self, log_set, time, interaction_types, min_weight, max_weight, verbose):
        weight_vector_subset = bias_util.get_log_subset(log_set, time, [])

        current_log = dict()
        current_log['bias_type'] = 'attribute_weight_distribution'
        current_log['current_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        current_log['number_of_logs'] = len(weight_vector_subset)
        current_log['interaction_types'] = interaction_types
        current_log['time_window'] = time
        current_log_info = dict()
        current_log_info['attribute_vector'] = dict()

        # no interactions
        if (len(weight_vector_subset) == 0):
            return

        # 0 if no interactions
        if (len(weight_vector_subset) == 0):
            current_log['info'] = current_log_info
            current_log['metric_level'] = 0
            return current_log

        # exponential distribution sampled N+1 times between 0 and (max weight - min weight)
        exp_distr = []
        for i in range(0, len(self.dataset) + 1):
            x_val = i * abs(max_weight - min_weight) / len(self.dataset)
            exp_distr.append(math.exp(-1 * x_val))

        # compute the distributions of delta weight (change in weight)
        weight_distr = dict()
        for attribute in self.attr_value_map.keys():
            if (attribute == 'Rand'):
                continue # skip rand here -- weight doesn't make sense for random variable
            cur_distr = []
            for i in range(0, len(weight_vector_subset)):
                cur_distr.append(abs(weight_vector_subset[i]['newWeight'][attribute] - weight_vector_subset[i]['oldWeight'][attribute]))
            weight_distr[attribute] = cur_distr

        # compare delta weight distributions to exponential distribution
        max_metric_value = 0
        for attribute in self.attr_value_map.keys():
            if (attribute == 'Rand'):
                continue # skip rand here -- weight doesn't make sense for random variable
            KS = ks_2samp(exp_distr, weight_distr[attribute])
            if (1 - float(KS[1]) > max_metric_value):
                max_metric_value = 1 - float(KS[1])

            current_log_info['attribute_vector'][attribute] = dict()
            current_log_info['attribute_vector'][attribute]['ks'] = KS[0]
            current_log_info['attribute_vector'][attribute]['metric_level'] = 1 - KS[1]

        current_log['info'] = current_log_info
        current_log['metric_level'] = max_metric_value
        self.awd_logs.append(current_log)
         
        if (verbose): 
            print '- attribute weight distribution metric: ', max_metric_value
        #pprint(current_log)
        return current_log
       
# this block is for computing bias metrics
if __name__ == '__main__':
    all_users = [f[5 :] for f in listdir(bias_util.directory) if ('user_' in f and not isfile(join(bias_util.directory, f)))]
    #all_users = ['1509568819048']
    for i in range(0, len(all_users)):
        cur_user = all_users[i]
        cur_dir = bias_util.directory + 'user_' + cur_user + '/'
        cur_file = 'interactions_' + cur_user + '.json'
        interaction_types = [] # ['set_attribute_weight_vector_init', 'category_click', 'set_attribute_weight_vector_select', 'drag', 'set_attribute_weight_vector_calc', 'double_click', 'help_hover', 'category_double_click', 'set_attribute_weight_vector_drag'] # everything except hover and click
        num_quantiles = 4
        min_weight = -1.0
        max_weight = 1.0
        rolling_dist = 10
        plot_svm = True
        
        for j in range(0, len(bias_util.window_methods)):
            window_method = bias_util.window_methods[j]
            print '**** Window Method:', window_method
            cur_bias = bias(cur_dir + 'logs/', cur_file, cur_file.replace('interactions', 'bias_' + window_method), bias_util.data_directory, bias_util.data_file_name)
            time = len(cur_bias.get_logs())
            fig_num = 2
            cur_bias.simulate_bias_computation(cur_dir + 'plots/' + window_method + '/', time, interaction_types, num_quantiles, min_weight, max_weight, window_method, rolling_dist, bias_util.marks, fig_num, bias_util.verbose) # fig_num 13 - 18
            print '------------------------------------------------------'