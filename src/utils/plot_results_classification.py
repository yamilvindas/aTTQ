#!/usr/bin/env python3
"""
    Plot the results of the experiment

    Options:
    --------
    results_file: str
        Path to the final results file containing the final results over
        the repetitions
"""

import os
import math
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, top_k_accuracy_score
from scipy.stats import median_abs_deviation as mad
from scipy.spatial.distance import jensenshannon

def plotMeanLoss(results_dict, plot_curves=False):
    """
        Plot the mean loss of the experiment over the different repetitions.

        Argument:
        ---------
        results_dict: dict
            Dictionary containing the results of the experiments for each repetition (obtained using
            the code /hits_signal_learning/src/Experiments/feature_comparison/training_model_one_feature_2D_CNN.py)
    """
    # Determining the mean and std values of the loss per epoch
    train_loss_vals_per_repetition = []
    val_loss_vals_per_repetition = []
    test_loss_vals_per_repetition = []
    for repetition_nb in results_dict:
        train_loss_vals_per_repetition.append(results_dict[repetition_nb]['Loss']['Train'])
        if ('Val' in results_dict[repetition_nb]['Loss']):
            val_loss_vals_per_repetition.append(results_dict[repetition_nb]['Loss']['Val'])
        test_loss_vals_per_repetition.append(results_dict[repetition_nb]['Loss']['Test'])
    mean_train_loss_vals = np.mean(train_loss_vals_per_repetition, axis=0) # axis=0 to average according to the repetitions
    std_train_loss_vals = np.std(train_loss_vals_per_repetition, axis=0) # axis=0 to average according to the repetitions
    if (len(val_loss_vals_per_repetition) > 0):
        mean_val_loss_vals = np.mean(val_loss_vals_per_repetition, axis=0) # axis=0 to average according to the repetitions
        std_val_loss_vals = np.std(val_loss_vals_per_repetition, axis=0) # axis=0 to average according to the repetitions
    mean_test_loss_vals = np.mean(test_loss_vals_per_repetition, axis=0) # axis=0 to average according to the repetitions
    std_test_loss_vals = np.std(test_loss_vals_per_repetition, axis=0) # axis=0 to average according to the repetitions

    # Plot
    if (plot_curves):
        epochs = list(range(len(mean_train_loss_vals)))
        plt.errorbar(x=epochs, y=mean_train_loss_vals, yerr=std_train_loss_vals, label='Train')
        if (len(val_loss_vals_per_repetition) > 0):
            plt.errorbar(x=epochs, y=mean_val_loss_vals, yerr=std_val_loss_vals, label='Val')
        plt.errorbar(x=epochs, y=mean_test_loss_vals, yerr=std_test_loss_vals, label='Test')
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

def plotMeanMetrics(results_dict, metric_type, selected_epoch, last_epochs_use=10, plot_curves=False, plot_val_metrics=True, plot_sparsity_rate=False):
    """
    Plot the mean accuracies of the experiment over the different repetitions.

        Argument:
        ---------
        results_dict: dict
            Dictionary containing the results of the experiments for each repetition (obtained using
            the code /hits_signal_learning/src/Experiments/feature_comparison/training_model_one_feature_2D_CNN.py)
        metric_type: str
            Metric to compute: MCC, F1-Score, ArtAccuracy, GeneralAccuracy,
            GEAccuracy, SEAccuracy
        acc_type: str
            Metric to plot: 'GeneralAccuracy', 'F1-Score', 'MCC'
        selected_epoch: int
            Particular epoch to compute the different metrics
        last_epochs_use: int
            Last epochs to use to compute the mean metrics.
        plot_val_metrics: bool
            True if want to compute the val metrics and plot it. If there are
            no validation metrics, this argument should be False
        plot_sparsity_rate: bool
            True if want to plot the sparsity rate over the epochs
    """
    # Determining the accuracies per epoch
    if (plot_val_metrics):
        metrics = {'Train': [], 'Val': [], 'Test': []}
    else:
        metrics = {'Train': [], 'Test': []}
    sparsity_rates = []
    for data_split_type in list(metrics.keys()):
        for repetition_nb in results_dict:
            tmp_metrics = []
            tmp_sparsity_rates = []
            if (data_split_type in results_dict[repetition_nb]['Predictions']):
                nb_epochs = len(results_dict[repetition_nb]['Predictions'][data_split_type]['TrueLabels'])
                for epoch in range(nb_epochs):
                    # Sparsity rate
                    if (data_split_type == 'Train'):
                        if ('SparsityRatePerEpoch' in results_dict[repetition_nb]):
                            tmp_sparsity_rates.append(results_dict[repetition_nb]['SparsityRatePerEpoch'][epoch])
                        else:
                            tmp_sparsity_rates.append(0)

                    # Metric
                    y_true = results_dict[repetition_nb]['Predictions'][data_split_type]['TrueLabels'][epoch]
                    y_pred = results_dict[repetition_nb]['Predictions'][data_split_type]['PredictedLabels'][epoch]
                    if (metric_type.lower() == 'generalaccuracy'):
                        metric = accuracy_score(y_true, y_pred)
                    elif (metric_type.lower() == 'f1-score'):
                        # metric = f1_score(y_true, y_pred, average='micro')
                        metric = f1_score(y_true, y_pred, average='macro')
                    elif (metric_type.lower() == 'mcc'):
                        metric = matthews_corrcoef(y_true, y_pred)
                    else:
                        raise ValueError('Metric {} is not valid'.format(metric_type))
                    tmp_metrics.append(metric)
                metrics[data_split_type].append(tmp_metrics)
                if (data_split_type == 'Train'):
                    sparsity_rates.append(tmp_sparsity_rates)

    # Determining the mean and std values of the accuracy per epoch
    mean_train_metrics, std_train_metrics = np.mean(metrics['Train'], axis=0), np.std(metrics['Train'], axis=0)
    metrics_last_epochs = computeMeanMetricsLastEpochs(metrics, sparsity_rates, last_epochs_use)
    if (plot_val_metrics):
        if (len(metrics['Val']) > 0):
            # Val metrics
            mean_val_metrics, std_val_metrics = np.mean(metrics['Val'], axis=0), np.std(metrics['Val'], axis=0)
            median_val_metrics, mad_val_metrics = np.median(metrics['Val'], axis=0), mad(metrics['Val'], axis=0)
            max_val_metric_epoch = np.argmax(mean_val_metrics)

            # Test metrics
            mean_test_metrics, std_test_metrics = np.mean(metrics['Test'], axis=0), np.std(metrics['Test'], axis=0)
            median_test_metrics, mad_test_metrics = np.median(metrics['Test'], axis=0), mad(metrics['Test'], axis=0)
            print("\tTest {} at epoch {}: {} +- {} (Sparsity Rate: {} +- {})".format(metric_type, selected_epoch, mean_test_metrics[selected_epoch]*100, std_test_metrics[selected_epoch]*100, mean_sparsity_rates[selected_epoch]*100, std_sparsity_rates[selected_epoch]*100))
            print("\tMEAN Test {} over the last {} epochs: {} +- {}".format(metric_type, last_epochs_use, metrics_last_epochs['Test']['Mean']*100, metrics_last_epochs['Test']['Std']*100))
            print("\tMEDIAN Test {} over the last {} epochs: {} +- {}\n".format(metric_type, last_epochs_use, metrics_last_epochs['Test']['Median']*100, metrics_last_epochs['Test']['MaD']*100))
        else:
            mean_test_metrics, std_test_metrics = np.mean(metrics['Test'], axis=0), np.std(metrics['Test'], axis=0)
            median_test_metrics, mad_test_metrics = np.median(metrics['Test'], axis=0), mad(metrics['Test'], axis=0)
            mean_sparsity_rates, std_sparsity_rates = np.mean(sparsity_rates, axis=0), np.std(sparsity_rates, axis=0)
            print("\tTest {} at epoch {}: {} +- {} (Sparsity Rate: {} +- {})".format(metric_type, selected_epoch, mean_test_metrics[selected_epoch]*100, std_test_metrics[selected_epoch]*100, mean_sparsity_rates[selected_epoch]*100, std_sparsity_rates[selected_epoch]*100))
            max_test_metric_epoch = np.argmax(mean_test_metrics)
            print("\tMAX test {}: {} +- {} at epoch {} (Sparsity Rate: {} +- {})".format(metric_type, mean_test_metrics[max_test_metric_epoch]*100, std_test_metrics[max_test_metric_epoch]*100, max_test_metric_epoch, mean_sparsity_rates[max_test_metric_epoch]*100, std_sparsity_rates[max_test_metric_epoch]*100))
            print("\tMEAN Test {} over the last {} epochs: {} +- {} (Sparsity Rate: {} +- {})".format(metric_type, last_epochs_use, metrics_last_epochs['Test']['Mean']*100, metrics_last_epochs['Test']['Std']*100, metrics_last_epochs['SparsityRates']['Mean']*100, metrics_last_epochs['SparsityRates']['Std']*100))
            print("\tMEDIAN Test {} over the last {} epochs: {} +- {} (Sparsity Rate: {} +- {})\n".format(metric_type, last_epochs_use, metrics_last_epochs['Test']['Median']*100, metrics_last_epochs['Test']['MaD']*100, metrics_last_epochs['SparsityRates']['Median']*100, metrics_last_epochs['SparsityRates']['MaD']*100))
    else:
        mean_test_metrics, std_test_metrics = np.mean(metrics['Test'], axis=0), np.std(metrics['Test'], axis=0)
        median_test_metrics, mad_test_metrics = np.median(metrics['Test'], axis=0), mad(metrics['Test'], axis=0)
        mean_sparsity_rates, std_sparsity_rates = np.mean(sparsity_rates, axis=0), np.std(sparsity_rates, axis=0)
        print('\n')
        print("\tTest {} at epoch {}: {} +- {} (Sparsity Rate: {} +- {})".format(metric_type, selected_epoch, mean_test_metrics[selected_epoch]*100, std_test_metrics[selected_epoch]*100, mean_sparsity_rates[selected_epoch]*100, std_sparsity_rates[selected_epoch]*100))
        max_test_metric_epoch = np.argmax(mean_test_metrics)
        print("\tMAX test {}: {} +- {} at epoch {} (Sparsity Rate: {} +- {})".format(metric_type, mean_test_metrics[max_test_metric_epoch]*100, std_test_metrics[max_test_metric_epoch]*100, max_test_metric_epoch, mean_sparsity_rates[max_test_metric_epoch]*100, std_sparsity_rates[max_test_metric_epoch]*100))
        print("\tMEAN Test {} over the last {} epochs: {} +- {} (Sparsity Rate: {} +- {})".format(metric_type, last_epochs_use, metrics_last_epochs['Test']['Mean']*100, metrics_last_epochs['Test']['Std']*100, metrics_last_epochs['SparsityRates']['Mean']*100, metrics_last_epochs['SparsityRates']['Std']*100))
        print("\tMEDIAN Test {} over the last {} epochs: {} +- {} (Sparsity Rate: {} +- {})\n".format(metric_type, last_epochs_use, metrics_last_epochs['Test']['Median']*100, metrics_last_epochs['Test']['MaD']*100, metrics_last_epochs['SparsityRates']['Median']*100, metrics_last_epochs['SparsityRates']['MaD']*100))

    # Plot
    if (plot_curves):
        epochs = list(range(len(mean_train_metrics)))
        plt.errorbar(x=epochs, y=mean_train_metrics, yerr=std_train_metrics, label='Train')
        if ('Val' in metrics):
            if (len(metrics['Val']) > 0):
                plt.errorbar(x=epochs, y=mean_val_metrics, yerr=std_val_metrics, label='Val')
        plt.errorbar(x=epochs, y=mean_test_metrics, yerr=std_test_metrics, label='Test')
        plt.title(metric_type)
        plt.xlabel("Epoch")
        plt.ylabel(metric_type)
        plt.legend()
        plt.show()
    if (plot_sparsity_rate):
        epochs = list(range(len(mean_train_metrics)))
        plt.errorbar(x=epochs, y=mean_sparsity_rates, yerr=std_sparsity_rates, label='Sparsity Rate')
        plt.title("Sparsity Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity Rate")
        plt.legend()
        plt.show()

    # Return a test metrics
    return mean_test_metrics[max_test_metric_epoch]*100, metrics_last_epochs['Test']['Mean']*100, sparsity_rates


def computeMeanMetricsLastEpochs(metrics, sparsity_rates, last_epochs_use):
    """
        Computes the last mean metrics over the last last_epochs_use epochs.
        This means that, if last_epochs_use = 10, then we are going to compute
        the mean metrics using the prediction of ALL those epochs.

        Arguments:
        ----------
        metrics: dict
            Dictionary with at least two keys: Train and Test.
            The values are list corresponding to the metrics values over the
            epochs at a fixed repetition. It means that the metrics train value
            at epoch at epoch ep and repetition rep is metrics['Train'][rep][epoch]
        sparsity_rates: list
            List with the sparsity rates over the epochs for the different repetitions.
        last_epochs_use: int
            Last epochs to use to compute the mean metrics.
    """
    # Getting the metrics over the last last_epochs_use
    train_metrics_last_epochs = []
    test_metrics_last_epochs = []
    sparsity_rates_last_epochs = []
    nb_repetitions = len(metrics['Train'])
    for rep_nb in range(nb_repetitions):
        for last_epoch in range(1, last_epochs_use+1):
            train_metrics_last_epochs.append(metrics['Train'][rep_nb][-last_epoch])
            test_metrics_last_epochs.append(metrics['Test'][rep_nb][-last_epoch])
            sparsity_rates_last_epochs.append(sparsity_rates[rep_nb][-last_epoch])
    train_mean_metrics_last_epochs, train_std_metrics_last_epochs = np.mean(train_metrics_last_epochs), np.std(train_metrics_last_epochs)
    train_median_metrics_last_epochs, train_mad_metrics_last_epochs = np.median(train_metrics_last_epochs), mad(train_metrics_last_epochs)
    test_mean_metrics_last_epochs, test_std_metrics_last_epochs = np.mean(test_metrics_last_epochs), np.std(test_metrics_last_epochs)
    test_median_metrics_last_epochs, test_mad_metrics_last_epochs = np.median(test_metrics_last_epochs), mad(test_metrics_last_epochs)
    mean_sparsity_rates_last_epochs, std_sparsity_rates_last_epochs = np.mean(sparsity_rates_last_epochs), np.std(sparsity_rates_last_epochs)
    median_sparsity_rates_last_epochs, mad_sparsity_rates_last_epochs = np.median(sparsity_rates_last_epochs), mad(sparsity_rates_last_epochs)
    return {
                'Train': {
                            'Mean': train_mean_metrics_last_epochs,
                            'Std': train_std_metrics_last_epochs,
                            'Median': train_median_metrics_last_epochs,
                            'MaD': train_mad_metrics_last_epochs
                          },
                'Test': {
                            'Mean': test_mean_metrics_last_epochs,
                            'Std': test_std_metrics_last_epochs,
                            'Median': test_median_metrics_last_epochs,
                            'MaD': test_mad_metrics_last_epochs
                          },
                'SparsityRates': {
                                    'Mean': mean_sparsity_rates_last_epochs,
                                    'Std': std_sparsity_rates_last_epochs,
                                    'Median': median_sparsity_rates_last_epochs,
                                    'MaD': mad_sparsity_rates_last_epochs
                                 }
            }

def load_results_data(results_file=None, results_folder=None):
    """
        Load the results of an experiment into a dictionary.

        Arguments:
        ----------
        results_file: str
            Path to a pth file containing the results of an experiment.
        results_folder: str
            Path to a folder with pth files containing the results of an
            experiment over different repetitions. The files used to
            get the results for each repetition must start with the
            prefix 'results_exp-'

        Returns:
        --------
        results_dict: dict
            Dictionary containing the results of the experiment over the
            different repetitions and epochs. The structure of this dictionary
            is the following:
            -results_dict is a dict where the keys are the IDs of the repetitions.
            -results_dict[id_repetition] is a dict with two keys 'Loss' and 'Predictions'.
            -results_dict[id_repetition]['Loss'] and results_dict[id_repetition]['Predictions']
            are also dictionaries with three keys: 'Train', 'Val', and 'Test'.
            -results_dict[id_repetition]['Loss']['Train'] is a list of lenght the
            number of epochs used during training. Each element correspond to the
            loss function at the given epoch.
            -results_dict[id_repetition]['Predictions']['Train'] is a dict with
            two keys: 'TrueLabels' and 'PredictedLabels'.
            -results_dict[id_repetition]['Predictions']['Train']['TrueLabels'] is
            a list of lenght the number of epochs used during training. Each
            element correspond to a list of size the number of training samples,
            and each element corresponds to the true label of a given sample.
            -results_dict[id_repetition]['Predictions']['Train']['PredictedLabels'] is
            a list of lenght the number of epochs used during training. Each
            element correspond to a list of size the number of training samples,
            and each element corresponds to the predicted label of a given sample.
    """
    # Verifying that at least one of the arguments is not None
    assert (results_file != None) or (results_folder != None)

    # Loading the data
    if (results_file is not None):
        # Loading the data into a dict
        with open(results_file, "rb") as fp:   # Unpickling
            results_dict = pickle.load(fp)

        # Putting the results dictionary in the right format if it is not yet under
        # this format
        if ('Loss' in results_dict):
            tmp_results_dict = {}
            tmp_results_dict[0] = results_dict
            results_dict = tmp_results_dict

    else: # In this case, the given argument by the user was the results metrics folder
        results_dict = {}
        rep_nb = 0
        for file_name in os.listdir(results_folder):
            if (os.path.isfile(results_folder+'/'+file_name)) and \
            ('results_exp-' in file_name) and ('rep-' in file_name):
            # ('results_exp-' in file_name) and ('rep-3' in file_name):
               # Loading the file in temporary dict
               with open(results_folder+'/'+file_name, "rb") as fp:   # Unpickling
                   tmp_results_dict = pickle.load(fp)
               # Adding the results to the final results dict
               results_dict[rep_nb] = tmp_results_dict
               rep_nb += 1

    return results_dict

def load_results_data_single_repetition(results_file=None):
    """
        Load the results of ONE REPETITION of an experiment into a dictionary.

        Arguments:
        ----------
        results_file: str
            Path to a pth file containing the results of one repetition of
            an experiment.

        Returns:
        --------
        results_dict: dict
            Dictionary containing the results of the experiment over the
            different repetitions and epochs. In order to be able to use the
            other functions in this code, the structure of this dictionary
            is the following:
            -results_dict is a dict where the keys are the IDs of the repetitions.
            IN THIS CASE, THERE IS ONLY ONE KEY, 0, CORRESPONDING TO THE ONLY
            REPETITION THAT HAS BEEN DONE.
            -results_dict[id_repetition] is a dict with two keys 'Loss' and 'Predictions'.
            -results_dict[id_repetition]['Loss'] and results_dict[id_repetition]['Predictions']
            are also dictionaries with three keys: 'Train', 'Val', and 'Test'.
            -results_dict[id_repetition]['Loss']['Train'] is a list of lenght the
            number of epochs used during training. Each element correspond to the
            loss function at the given epoch.
            -results_dict[id_repetition]['Predictions']['Train'] is a dict with
            two keys: 'TrueLabels' and 'PredictedLabels'.
            -results_dict[id_repetition]['Predictions']['Train']['TrueLabels'] is
            a list of lenght the number of epochs used during training. Each
            element correspond to a list of size the number of training samples,
            and each element corresponds to the true label of a given sample.
            -results_dict[id_repetition]['Predictions']['Train']['PredictedLabels'] is
            a list of lenght the number of epochs used during training. Each
            element correspond to a list of size the number of training samples,
            and each element corresponds to the predicted label of a given sample.
    """
    # Verifying that at least one of the arguments is not None
    assert (results_file != None)

    # Loading the data
    results_dict = {}
    rep_nb = 0
    # Loading the file in temporary dict
    with open(results_file, "rb") as fp:   # Unpickling
        tmp_results_dict = pickle.load(fp)
    # Adding the results to the final results dict
    results_dict[rep_nb] = tmp_results_dict

    return results_dict


#==============================================================================#
#==============================================================================#

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--results_file', help="File containing the metrics of the experiment", type=str)
    ap.add_argument('--results_folder', help="Folder containing the files of the results of the experiment", type=str)
    ap.add_argument('--selected_epoch', default=0, help="Particular epoch to compute the different metrics", type=int)
    ap.add_argument('--last_epochs_use', default=1, help="Last epochs to use to compute the mean metrics", type=int)
    ap.add_argument('--plot_curves', default='False', help="Boolean to plot the loss and metrics curves", type=str)
    ap.add_argument('--plot_val_metrics', default='False', help="True if wanted to compute the val metrics and plot it. If there are no validation metrics, this argument should be False", type=str)
    ap.add_argument('--plot_sparsity_rate', default='False', help="True if want to plot the sparsity rate over the epochs", type=str)
    ap.add_argument('--plot_loss', default='True', help="True if want to plot the loss curves", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_file = args['results_file']
    results_folder = args['results_folder']
    if (results_file is None) and (results_folder is None):
        raise Exception("One of the two options --results_file or --results_folder has to be specified")
    elif (results_file is not None) and (results_folder is not None):
        raise Exception("The two options --results_file and --results_folder are mutually incompatible")
    selected_epoch = args['selected_epoch']
    last_epochs_use = args['last_epochs_use']
    plot_curves = args['plot_curves']
    if (plot_curves.lower() == 'true'):
        plot_curves = True
    else:
        plot_curves = False
    plot_val_metrics = args['plot_val_metrics']
    if (plot_val_metrics.lower() == 'true'):
        plot_val_metrics = True
    else:
        plot_val_metrics = False
    plot_sparsity_rate = args['plot_sparsity_rate']
    if (plot_sparsity_rate.lower() == 'true'):
        plot_sparsity_rate = True
    else:
        plot_sparsity_rate = False
    plot_loss = args['plot_loss']
    if (plot_loss.lower() == 'true'):
        plot_loss = True
    else:
        plot_loss = False

    #==========================================================================#
    # Loading the data
    if (results_file is not None):
        if ('final' not in results_file):
            results_dict = load_results_data_single_repetition(results_file)
        else:
            results_dict = load_results_data(results_file, results_folder)
    else:
        results_dict = load_results_data(results_file, results_folder)


    #==========================================================================#
    # Sparsity rate
    if ('SparsityRate' in results_dict[0]):
        sparsity_rates = [results_dict[i]['SparsityRate'] for i in results_dict]
        print("\n\n==========> Sparsity rate of: {} +- {}".format(np.mean(sparsity_rates)*100, np.std(sparsity_rates)*100))

    #==========================================================================#
    # Plotting the loss
    if ('Loss' in results_dict[0]) and (plot_loss):
        plotMeanLoss(results_dict, plot_curves=plot_curves)

    # Plotting the MCC
    plotMeanMetrics(results_dict, metric_type='mcc', selected_epoch=selected_epoch, last_epochs_use=last_epochs_use, plot_curves=plot_curves, plot_val_metrics=plot_val_metrics, plot_sparsity_rate=plot_sparsity_rate)

    # Plotting the F1 Score
    # plotMeanMetrics(results_dict, metric_type='F1-Score', selected_epoch=selected_epoch, last_epochs_use=last_epochs_use, plot_curves=plot_curves, plot_val_metrics=plot_val_metrics)

    # Plotting the accuracy
    # plotMeanMetrics(results_dict, metric_type='GeneralAccuracy', selected_epoch=selected_epoch, last_epochs_use=last_epochs_use, plot_curves=plot_curves, plot_val_metrics=plot_val_metrics)



if __name__=='__main__':
    main()
