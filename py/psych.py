#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script is used to analyze classification data from 
the bias eval user study. It produces values against
which to compare the bias interaction metrics. 

Created on Mon Aug 21 00:13:06 2017

@author: emilywall
"""

import numpy as np
import matplotlib.pyplot as plt
import bias_util
import json
import copy
import sys
import os
import operator
from os import listdir
from os.path import isfile, isdir, join
from sklearn import svm
from sklearn import tree
import graphviz
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# train an SVM model using the given data and labels
# return the model weights and the classes
def get_svm_weights(data, labels): 
    data = np.array(data)
    labels = np.array(labels)
    clf = svm.SVC(kernel = 'linear') #(kernel = 'linear', C = .1)
    clf.fit(data, labels)
    results = clf.coef_
    weights = []
    for i in range(0, len(results)):
        weights.append(normalize_weights(list(results[i])))
    classes = clf.classes_
    
    return weights, classes
    
# normalize the set of weights as if all are positive, then add negative factors at the end
def normalize_weights(notNormedWeights):
    factors = []
    for i in range(len(notNormedWeights)):
        if notNormedWeights[i] < 0:
            notNormedWeights[i] *= -1
            factors.append(-1)
        else:
            factors.append(1)
    s = sum(notNormedWeights)
    pos_weights = [(r / s) for r in notNormedWeights]
    weights = [pos_weights[i] * factors[i] for i in range(len(factors))]
    return weights
    
# run SVM and write the results to the given directory
def write_svm_results(directory, file_name, log_file, to_plot, fig_num, verbose): 
    if (verbose):
        print 'Writing and Plotting SVM Data: ', file_name
        
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    logs = item_logs
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(logs, dataset)
    all_data = dict()
    all_data['classifications'] = classification
    
    x_data = []
    y_data = []
    data = []   

    features = bias_util.get_bball_player(dataset, list(classification.keys())[0]).get_map().keys()
    features.remove('Name')
    features = sorted(features)
        
    for key in classification.keys():
        cur_player = bias_util.get_bball_player(dataset, key)
        cur_map = cur_player.get_map()
        cur_map['*Classification'] = classification[key]
        data.append(cur_map)
        
        cur_x = []
        for i in range(0, len(features)):
            cur_x.append(cur_map[features[i]])
        cur_x = [float(x) for x in cur_x]
        x_data.append(cur_x)
        y_data.append(bias_util.pos_to_num_map[classification[key]])
        
    svm_weights, svm_classes = get_svm_weights(x_data, y_data)
    weights_map = dict()
    i = 0
    for j in range(0, len(svm_classes)):
        for k in range(j + 1, len(svm_classes)):
            key = bias_util.num_to_pos_map[j] + ' - ' + bias_util.num_to_pos_map[k]
            value = svm_weights[i]
            weights_map[key] = value
            i += 1
    
    all_data['features'] = features
    all_data['weights'] = weights_map
    all_data['classifications'] = data

    if not os.path.exists(directory):
        os.makedirs(directory)
    f_out = open(directory + file_name, 'w+')
    f_out.write('{')
    f_out.write('"features":' + json.dumps(all_data['features']) + ',')
    f_out.write('"weights":' + json.dumps(all_data['weights']) + ',')
    f_out.write('"classifications":' + json.dumps(all_data['classifications']))
    f_out.write('}')
    f_out.close()
    
    if (to_plot == True):
        for key in weights_map.keys():
            plot_svm(features, weights_map[key], 'SVM Feature Weights: ' + key, 'Feature', 'Weight', directory.replace('/logs/', '/plots/'), file_name.replace('.json', '.png').replace('svm', 'svm_' + key), fig_num, verbose)
            fig_num += 1
            
    return svm_weights
    
# plot the weights from SVM as a bar chart
def plot_svm(features, weights, title, x_label, y_label, directory, file_name, fig_num, verbose):
    if (verbose): 
        print 'plotting', title
    
    plt.figure(num = fig_num, figsize = (5, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    plt.subplot(1, 1, 1)
    
    # create the plot
    y_pos = np.arange(len(features))
    plt.bar(y_pos, weights, align = 'center', alpha = 0.5)
    plt.xticks(y_pos, features, rotation = 'vertical')
    plt.tight_layout()
        
    # label, save, and clear
    #plt.xlabel(x_label)
    #plt.ylabel(y_label)
    plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# compute the centroid data point given the list of data points
def compute_centroid(data):
    centroid = dict()
    attributes = data[0].keys()
    for attr in attributes:
        if (attr == 'Name' or attr == 'Name (Real)' or attr == 'Team' or attr == 'Position'): 
            continue
        else:
            avg = 0
            for i in range(0, len(data)):
                avg += float(data[i][attr])
            avg /= len(data)
            centroid[attr] = avg
        
    return centroid
    
# get the identification-confusion matrix from the final classification
# row (y) is user-defined label, col (x) is actual label
def get_id_confusion_matrix(logs, dataset):
    id_confusion = np.zeros((7, 7))
    all_data = dict()
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(logs, dataset)
    
    for i in range(0, len(dataset)):
        cur_data = dataset[i].get_full_map()
        cur_id = cur_data['Name'].replace('Player ', '')
        actual_pos = cur_data['Position']
        if (cur_id in classification): 
            user_pos = classification[cur_id]
            id_confusion[bias_util.pos_to_num_map[user_pos], bias_util.pos_to_num_map[actual_pos]] += 1
            key = 'user:' + user_pos + ',actual:' + actual_pos
            if key in all_data: 
                all_data[key].append(cur_data)
            else: 
                all_data[key] = [cur_data]
                
    return id_confusion, bias_util.pos_to_num_map, all_data
    
# write the identification-confusion matrix to a json file
def write_id_confusion_matrix(directory, file_name, log_file, fig_num, verbose):
    if (verbose):
        print 'Writing and Plotting ID-Confusion Matrix Data: ', file_name
    
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    logs = item_logs
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    id_confusion, pos_to_num_map, all_data = get_id_confusion_matrix(logs, dataset)
  
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_out = open(directory + file_name, 'w+')
    summary = dict()
    summary['rows (y)'] = 'user'
    summary['cols (x)'] = 'actual'
    summary['position_indices'] = pos_to_num_map
    summary['centroids'] = dict()
    summary['centroids']['user_centroids'] = dict()
    summary['centroids']['actual_centroids'] = dict()

    # separate out user labels and actual labels
    user_labels = dict()
    actual_labels = dict()
    for key in all_data.keys(): 
        cur_user_label = key[5 : key.index(',')]
        cur_actual_label = key[key.index('actual') + 7 : ]
        cur_data_point = copy.deepcopy(all_data[key])

        if (cur_user_label in user_labels.keys()):
            user_labels[cur_user_label] += cur_data_point
        else: 
            user_labels[cur_user_label] = cur_data_point
        if (cur_actual_label in actual_labels.keys()):
            actual_labels[cur_actual_label] += cur_data_point
        else: 
            actual_labels[cur_actual_label] = cur_data_point

    # compute centroids
    for key in user_labels.keys():
        summary['centroids']['user_centroids'][key] = compute_centroid(user_labels[key])
        #if (verbose):
        #    print 'User Centroid', key, summary['centroids']['user_centroids'][key]
    for key in actual_labels.keys():
        summary['centroids']['actual_centroids'][key] = compute_centroid(actual_labels[key])
        #if (verbose):
        #    print 'Actual Centroid', key, summary['centroids']['actual_centroids'][key]

    # get total accuracy
    num_correct = 0
    total_classifications = 0
    for i in range(0, len(id_confusion)): 
        for j in range(0, len(id_confusion[i])):
            total_classifications += id_confusion[i][j]
            if (i == j): 
                num_correct += id_confusion[i][j]
    
    summary['total_accuracy'] = str(num_correct) + '/' + str(total_classifications)
    summary['matrix'] = id_confusion.tolist()
    f_out.write('{')
    f_out.write('"summary":' + json.dumps(summary) + ',')
    f_out.write('"all_data":' + json.dumps(all_data))
    f_out.write('}')
    f_out.close()
    
    # plot the matrix
    labels = [bias_util.num_to_pos_map[0], bias_util.num_to_pos_map[1], bias_util.num_to_pos_map[2], bias_util.num_to_pos_map[3], bias_util.num_to_pos_map[4]]
    plot_id_conf_matrix(id_confusion.tolist(), labels, directory.replace('/logs/', '/plots/'), file_name.replace('.json', '.png'), fig_num)
    
# plot the identification-confusion matrix
def plot_id_conf_matrix(matrix, labels, directory, file_name, fig_num):
    plt.figure(num = fig_num, figsize = (5, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    plt.subplot(1, 1, 1)
    matrix = np.array(matrix)
    matrix = matrix[0 : 5, 0 : 5]
    heatmap = plt.pcolor(matrix, cmap = 'Blues')
    
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%d' % matrix[y, x], horizontalalignment = 'center', verticalalignment = 'center')
    
    plt.colorbar(heatmap)
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation = 'vertical')
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.xlabel('Actual Category')
    plt.ylabel('User Category')
    plt.title('Identification-Confusion Matrix')
    plt.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# write the classification accuracy over time to a file
def write_classification_accuracy(directory, file_name, log_file, fig_num, verbose):
    print 'Writing and Plotting Accuracy Over Time: ', file_name
    
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    logs = item_logs
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(logs, dataset)
    
    total_labeled = 0
    total_correct = 0
    decision_points = np.arange(1, len(all_logs) + 1)
    accuracy = [-1] * len(all_logs)
    current_labels = dict()
    correct_labels = dict()
    
    for i in range(0, len(decisions_labels)):
        cur = decisions_labels[i]
        cur_id = cur[3]
        correct_labels[cur_id] = cur[2]
        
        if ((cur_id not in current_labels and cur[1] != 'Un-Assign') or (cur_id in current_labels and current_labels[cur_id] == 'Un-Assign' and cur[1] != 'Un-Assign')):
            total_labeled += 1
        elif (cur_id in current_labels and cur[1] == 'Un-Assign' and current_labels[cur_id] != 'Un-Assign'):
            total_labeled -= 1
            
        if (cur_id not in current_labels and cur[1] == correct_labels[cur_id]): 
            total_correct += 1
        elif (cur_id in current_labels and current_labels[cur_id] != correct_labels[cur_id] and cur[1] == correct_labels[cur_id]):
            total_correct += 1
            
        if (cur_id in current_labels and current_labels[cur_id] == correct_labels[cur_id] and cur[1] != correct_labels[cur_id]):
            total_correct -= 1
            
        if (total_labeled != 0):
            accuracy[cur[0]] = total_correct / float(total_labeled)
        else: 
            accuracy[cur[0]] = 0
        current_labels[cur_id] = cur[1]
    if (len(decisions_labels) < 1):
        first_decision = -1
    else: 
        first_decision = decisions_labels[0][0]
    accuracy = bias_util.remove_defaults(accuracy, first_decision)
    accuracy = bias_util.forward_fill(accuracy)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_out = open(directory + file_name, 'w+')
    f_out.write('[')
    for i in range(0, len(decisions_labels)):
        f_out.write('{')
        f_out.write('"interaction_number":"' + str(decisions_labels[i][0]) + '",')
        f_out.write('"data_point":"' + str(decisions_labels[i][3]) + '",')
        f_out.write('"actual_class":"' + str(decisions_labels[i][2]) + '",')
        f_out.write('"user_class":"' + str(decisions_labels[i][1]) + '",')
        f_out.write('"current_accuracy":"' + str(accuracy[decisions_labels[i][0]]) + '"')
        f_out.write('}')
        if (i != len(decisions_labels) - 1): 
            f_out.write(',')
    f_out.write(']')
    f_out.close()

    plot_classification_accuracy(decision_points, accuracy, 'Accuracy Over Time', 'Interactions', 'Accuracy', directory.replace('/logs/', '/plots/'), file_name.replace('.json', '.png'), decisions_labels, fig_num, verbose)
    
# plot the classification accuracy over time
def plot_classification_accuracy(x_values, y_values, title, x_label, y_label, directory, file_name, decisions, fig_num, verbose):    
    plt.figure(num = fig_num, figsize = (15, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    sub_plot = plt.subplot(1, 1, 1)
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    plt.plot(x_values, y_values, c = '#000000')
             
    for i in range(0, len(decisions)): 
        tup = decisions[i]
        sub_plot.axvline(x = tup[0], c = bias_util.color_map[tup[1]], linewidth = 2, zorder = 0, clip_on = False)
    
    # label, save, and clear
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# get the similarity of two metric sequences by using dynamic time warping 
# input sequences are just lists of numbers; function needs to zip them with list indices
def get_dtw_similarity(y1, y2):
    x1 = np.arange(1, len(y1) + 1)
    x2 = np.arange(1, len(y2) + 1)
    
    seq1 = np.array(zip(x1, y1))
    seq2 = np.array(zip(x2, y2))
    
    distance, path = fastdtw(seq1, seq2, dist = euclidean)
    return distance, path
    
# perform dynamic time warping between users with the metrics
def dynamic_time_warping(verbose):
    if (verbose):
        print '** Creating DTW Map'
    all_users = [f.replace('user_', '').replace('.json', '') for f in listdir(bias_util.directory) if ('user' in f and isdir(join(bias_util.directory, f)))]
    all_metrics = ['data_point_coverage', 'data_point_distribution']
    for i in range(0, len(bias_util.attrs)):
        all_metrics.append('attribute_coverage_' + bias_util.attrs[i])
        all_metrics.append('attribute_distribution_' + bias_util.attrs[i])
        all_metrics.append('attribute_weight_coverage_' + bias_util.attrs[i])
        all_metrics.append('attribute_weight_distribution_' + bias_util.attrs[i])

    dtw_metric_map = dict()
    
    # read each user's bias metrics for each window method and each metric
    # compare to each other via DTW
    for i in range(0, len(bias_util.window_methods)):
        window_method = bias_util.window_methods[i]
        dtw_metric_map[window_method] = dict()
      
        if (verbose):
            print '    Window Method:', window_method
            
        for j in range(0, len(all_users)):
            cur_user = all_users[j]
            cur_file_name = bias_util.directory + 'user_' + cur_user + '/logs/bias_' + window_method + '_' + cur_user + '.json'
            cur_file = json.loads(open(cur_file_name).read())
            
            dtw_metric_map[window_method][cur_user] = dict()
            for k in range(0, len(all_metrics)):
                cur_metric = all_metrics[k]
                dtw_metric_map[window_method][cur_user][cur_metric] = []
            
            for l in range(0, len(cur_file)):
                for k in range(0, len(all_metrics)):
                    cur_metric = all_metrics[k]
                    metric_vals = dtw_metric_map[window_method][cur_user][cur_metric] 
                    if ('data_point' in cur_metric and cur_metric in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics'][cur_metric]['metric_level'])
                    elif ('attribute_coverage' in cur_metric and 'attribute_coverage' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_coverage']['info']['attribute_vector'][cur_metric.replace('attribute_coverage_', '')]['metric_level'])
                    elif ('attribute_distribution' in cur_metric and 'attribute_distribution' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_distribution']['info']['attribute_vector'][cur_metric.replace('attribute_distribution_', '')]['metric_level'])
                    elif ('attribute_weight_coverage' in cur_metric and 'attribute_weight_coverage' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_weight_coverage']['info']['attribute_vector'][cur_metric.replace('attribute_weight_coverage_', '')]['metric_level'])
                    elif ('attribute_weight_distribution' in cur_metric and 'attribute_weight_distribution' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_weight_distribution']['info']['attribute_vector'][cur_metric.replace('attribute_weight_distribution_', '')]['metric_level'])
                    dtw_metric_map[window_method][cur_user][cur_metric] = metric_vals
                    
    return dtw_metric_map, all_users, all_metrics

# run dynamic time warping and write the results to a file
def write_dtw(out_dir, verbose):
    dtw_metric_map, all_users, all_metrics = dynamic_time_warping(verbose)
    #f_out = open(out_dir + 'dtw_info.json', 'w+')
    #f_out.write('{')
    #f_out.write(json.dumps(dtw_metric_map))
    #f_out.write('}')
    #f_out.close()
    
    if (verbose):
        print '** DTW Map Done'
        print '** Writing Files'

    for i in range(0, len(bias_util.window_methods)):
        window_method = bias_util.window_methods[i]
        f_out = open(out_dir + 'post_hoc_' + window_method + '.csv', 'w+')
        if (verbose):
            print '    Writing File:', out_dir + 'post_hoc_' + window_method + '.csv'
            
        header = 'user1,user2,'
        for j in range(0, len(all_metrics)):
            header += all_metrics[j]
            if (j != len(all_metrics) - 1):
                header += ','
        f_out.write(header + '\n')
        
        for a in range(0, len(all_users)): 
            user1 = all_users[a]
            for b in range(a + 1, len(all_users)): 
                user2 = all_users[b]
                line = user1 + ',' + user2 + ','
                for j in range(0, len(all_metrics)):
                    cur_metric = all_metrics[j]
                    y1 = dtw_metric_map[window_method][user1][cur_metric]
                    y2 = dtw_metric_map[window_method][user2][cur_metric]
                    dist, path = get_dtw_similarity(y1, y2)
                    if (verbose):
                        print '        DTW ', cur_metric, ' Dist:', user1, '--', user2, '=', dist
                    line += str(dist)
                    if (j != len(all_metrics) - 1):
                        line += ','
                f_out.write(line + '\n')
                    
        f_out.close()
        
# predict the order of position classifications
# if error: make sure clicks aren't filtered!
def classification_prediction(directories, log_files, user_ids, out_dir, verbose):
    class_counts = dict()
    if (not isinstance(directories, (tuple, list))):
        X, Y, class_counts = one_classification_prediction(directories, log_files, user_ids, class_counts, verbose)
        out_file = 'graphviz_test.dot'# Emily temp remove_' + user_ids + '.dot'
        if (verbose): 
            print 'Classification Prediction: ', user_ids
            print 'counts', class_counts
    else: 
        X = []
        Y = []
        out_file = 'graphviz_test.dot' # emily temp
        for i in range(0, len(directories)):
            if (verbose): 
                print 'Classification Prediction: ', user_ids[i]
            x, y, class_counts = one_classification_prediction(directories[i], log_files[i], user_ids[i], class_counts, verbose)
            X = X + x
            Y = Y + y
            print directories[i]
            
    class_names = bias_util.pos_names[:]
    class_names_trimmed = []
    for key in class_names:
        if (key in class_counts):
            class_names_trimmed.append(key)
            
    print 'classes (', len(class_names_trimmed), '): ', class_names_trimmed
     
    feature_names = ['prev_class']
    out_dir = '/Users/emilywall/git/bias_eval/py/'# emily temp
    decision_tree(X, Y, out_dir, out_file, feature_names, class_names_trimmed, verbose)
            
# predict the order of classification given the current classification using a decision tree for a single user
def one_classification_prediction(directory, log_file, user_id, class_counts, verbose):
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(all_logs, dataset)
    
    X = []
    Y = []
    
    # decisions_labels of the form (index, user classification, actual classification, player id)
    prev = -1
    cur = -1
    for i in range(0, len(decisions_labels)):     
        prev = cur
        cur = decisions_labels[i][1]
        
        if (not cur in class_counts):
            class_counts[cur] = 1
        else: 
            class_counts[cur] += 1
            
        if (prev != -1 and cur != -1):# and prev != cur): # create a training instance
            # comment prev != cur condition to allow repetitions
            X.append([bias_util.pos_to_num_map[prev]])
            Y.append(bias_util.pos_to_num_map[cur])
            
    return X, Y, class_counts
        
# compile the training data from potentially multiple users to predict interactions
def interaction_prediction(directories, log_files, user_ids, out_dir, verbose):
    int_counts = dict()
    if (not isinstance(directories, (tuple, list))):
        if (verbose): 
            print 'Interaction Prediction: ', user_ids
        X, Y, int_counts = one_interaction_prediction(directories, log_files, user_ids, int_counts, verbose)
        out_file = 'graphviz_test.dot'# Emily temp remove_' + user_ids + '.dot'
    else: 
        X = []
        Y = []
        out_file = 'graphviz_test.dot' # emily temp
        for i in range(0, len(directories)):
            if (verbose): 
                print 'Interaction Prediction: ', user_ids[i]
            x, y, int_counts = one_interaction_prediction(directories[i], log_files[i], user_ids[i], int_counts, verbose)
            X = X + x
            Y = Y + y
            print directories[i]
        
    class_names = bias_util.interaction_names[:]
    class_names_trimmed = []
    for key in class_names:
        if (key in int_counts):
            class_names_trimmed.append(key)
    
    feature_names = ['prev_interaction']
    out_dir = '/Users/emilywall/git/bias_eval/py/'# emily temp
    decision_tree(X, Y, out_dir, out_file, feature_names, class_names_trimmed, verbose)
    
# predict the next interaction mode given the current interaction mode using a decision tree for a single user
def one_interaction_prediction(directory, log_file, user_id, int_counts, verbose): 
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    X = []
    Y = []
    prev = -1
    cur = -1
    int_to_skip = ['set_attribute_weight_vector_init']
    
    for i in range(0, len(all_logs)):
        prev = cur
        cur = all_logs[i]['customLogInfo']['eventType']
        
        if (cur in int_to_skip or prev in int_to_skip):
            continue # skip this interaction
            
        if (not cur in int_counts):
            int_counts[cur] = 1
        else: 
            int_counts[cur] += 1
                
        if (prev != -1 and cur != -1):# and prev != cur): # create a training instance
            # comment prev != cur condition to allow repetitions
            X.append([bias_util.int_to_num_map[prev]])
            Y.append(bias_util.int_to_num_map[cur])
            
    return X, Y, int_counts
            
    
# create a decision tree with the given training data
def decision_tree(X, Y, directory, file_name, feature_names, class_names, verbose):
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X, Y)
    if (feature_names != -1 and class_names != -1):
        dot_data = tree.export_graphviz(classifier, out_file = directory + file_name, feature_names = feature_names, class_names = class_names) #emily: file name should end with user's id + .dot 
    else: 
        dot_data = tree.export_graphviz(classifier, out_file = directory + file_name) #emily: file name should end with user's id + .dot 
    graph = graphviz.Source(dot_data) 
    graph.render('graphviz_test', view = True)
    if (verbose):
        print '  Done: Decision Tree ', file_name
        
# do some of the post-hoc analyses on the data
if __name__ == '__main__': 
    '''
    fig_num = 2
    # generate id-confusion matrices and svm results
    all_users = [f[5 :] for f in listdir(bias_util.directory) if ('user_' in f and not isfile(join(bias_util.directory, f)))]
    user_ids = []
    directories = []
    log_files = []
    for i in range(0, len(all_users)):
        cur_user = all_users[i]
        cur_dir = bias_util.directory + 'user_' + cur_user + '/'
        cur_file = 'interactions_' + cur_user + '.json'
        
        user_ids.append(cur_user)
        directories.append(cur_dir + 'logs/')
        log_files.append(cur_file)
        
        #interaction_prediction(cur_dir + 'logs/', 'interactions_' + cur_user + '.json', cur_user, bias_util.directory + 'analysis/', bias_util.verbose) # emily
        #classification_prediction(cur_dir + 'logs/', 'interactions_' + cur_user + '.json', bias_util.directory + 'analysis/', cur_user, bias_util.verbose)
        #break
        
        #write_id_confusion_matrix(cur_dir + 'logs/', cur_file.replace('interactions', 'id-conf'), 'interactions_' + cur_user + '.json', fig_num, bias_util.verbose) # fig_num = 2
        #write_classification_accuracy(cur_dir + 'logs/', cur_file.replace('interactions', 'accuracy'), 'interactions_' + cur_user + '.json', fig_num + 1, bias_util.verbose) # fig_num = 3
        #write_svm_results(cur_dir + 'logs/', cur_file.replace('interactions', 'svm'), 'interactions_' + cur_user + '.json', plot_svm, fig_num + 2, bias_util.verbose) # fig_num = 4 - 12
    '''
    
    
    # create decision trees to predict the labeling order differences between the two different conditions
    '''
    directories_var = [bias_util.directory + 'user_' + str(cur_user) + '/' + 'logs/' for cur_user in bias_util.cond_var]
    log_files_var = ['interactions_' + str(cur_user) + '.json' for cur_user in bias_util.cond_var]
    #classification_prediction(directories_var, log_files_var, bias_util.cond_var, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    #interaction_prediction(directories_var, log_files_var, bias_util.cond_var, bias_util.directory + 'analysis/', bias_util.verbose) # emily

    directories_size = [bias_util.directory + 'user_' + str(cur_user) + '/' + 'logs/' for cur_user in bias_util.cond_size]
    log_files_size = ['interactions_' + str(cur_user) + '.json' for cur_user in bias_util.cond_size]
    classification_prediction(directories_size, log_files_size, bias_util.cond_size, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    #interaction_prediction(directories_size, log_files_size, bias_util.cond_size, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    '''
    
    
    #interaction_prediction(directories, log_files, user_ids, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    #classification_prediction(directories, log_files, user_ids, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    
    
    #out_dir = bias_util.directory + 'analysis/'
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
        
    # write the .csv file with the all DTW info    
    #write_dtw(out_dir, bias_util.verbose)