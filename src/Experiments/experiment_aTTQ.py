#!/usr/bin/env python3
"""
    Compress a pre-trained model using TTQ.

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
        This files are usually located in /hits_signal_learning/parameters_files/model_compression/
"""
import os
import json
import shutil
import pickle
import argparse
from tqdm import tqdm

import random

import numpy as np
from math import floor
import matplotlib as mpl
from datetime import datetime

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.nn.utils.prune as prune
from pydub import AudioSegment

from labml_nn.optimizers import noam

from src.Experiments.experiment_TTQ import Experiment as ExperimentTTQ

from src.utils.GCE import GeneralizedCrossEntropy
from src.utils.model_compression import approx_weights, approx_weights_fc

from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN # Network used for training
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import TransformerClassifierMultichannelCNN

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#
class Experiment(ExperimentTTQ):
    def __init__(self, parameters_exp):
        """
            Compress a pre-trained model using TTQ.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment:
                    * exp_id: str, name of the experiment.
                    * feature_type
        """
        # Parent constructor
        super().__init__(parameters_exp)

        # Threshold hyper-parameter for TTQ
        if ('t_min' not in parameters_exp):
            parameters_exp['t_min'] = -1
        self.t_min = parameters_exp['t_min']
        if ('t_max' not in parameters_exp):
            parameters_exp['t_max'] = 1
        self.t_max = parameters_exp['t_max']

        # Parameters of the exp
        self.parameters_exp = parameters_exp


    # Quantization function
    def quantize(self, kernel, w_p, w_n):
        """
        Function based on: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
        ATTENTION: it is not the same function as we change the method to quantize
        the weights.

        Return quantized weights of a layer.
        Only possible values of quantized weights are: {zero, w_p, -w_n}.
        """
        # Our proposed approach
        delta_min = kernel.mean() + self.t_min*kernel.std()
        delta_max = kernel.mean() + self.t_max*kernel.std()
        a = (kernel > delta_max).float()
        b = (kernel < delta_min).float()
        return w_p*a + w_n*b


    # Gradients computation
    def get_grads(self, kernel_grad, kernel, w_p, w_n):
        """
        Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
        ATTENTION: it is not the same function as we change the method to quantize
        the weights.

        Arguments:
            kernel_grad: gradient with respect to quantized kernel.
            kernel: corresponding full precision kernel.
            w_p, w_n: scaling factors.
        Returns:
            1. gradient for the full precision kernel.
            2. gradient for w_p.
            3. gradient for w_n.
        """
        # Our proposed approach
        delta_min = kernel.mean() + self.t_min*kernel.std()
        delta_max = kernel.mean() + self.t_max*kernel.std()
        # masks
        a = (kernel > delta_max).float()
        b = (kernel < delta_min).float()
        c = torch.ones(kernel.size()).to(self.device) - a - b
        # scaled kernel grad and grads for scaling factors (w_p, w_n)
        return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
            (a*kernel_grad).sum(), (b*kernel_grad).sum()

    def gridSearch(self):
        """
            Does a grid search for some hyper-parameters
        """
        # Defining the values of the parameters to test
        # VALUES TO STUDY THE INFLUENCE OF X AND Y
        lr_values = [self.lr]
        t_min_values = [-5, -4, -3, 0, 3, 4, 5]
        t_max_values = [-5, -4, -3, 0, 3, 4, 5]

        # Iterating over the different values of the hyper-parameters
        base_results_folder = self.results_folder
        for lr in lr_values:
            for t_min in t_min_values:
                for t_max in t_max_values:
                    # Updating the hyper-paramet of the experiment
                    self.lr = lr
                    self.t_min = t_min
                    self.t_max = t_max

                    # Creating the datasets folder
                    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
                    os.mkdir(base_results_folder + '/LR-{}_TMIN-{}_TMAX-{}_{}/'.format(self.lr, self.t_min, self.t_max, current_datetime))
                    os.mkdir(base_results_folder + '/LR-{}_TMIN-{}_TMAX-{}_{}/model/'.format(self.lr, self.t_min, self.t_max, current_datetime))
                    os.mkdir(base_results_folder + '/LR-{}_TMIN-{}_TMAX-{}_{}/metrics/'.format(self.lr, self.t_min, self.t_max, current_datetime))
                    self.results_folder = base_results_folder + '/LR-{}_TMIN-{}_TMAX-{}_{}/'.format(self.lr, self.t_min, self.t_max, current_datetime)

                    # Training
                    self.holdout_train()

        self.results_folder = base_results_folder

#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    seed = 42
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_parameters_file = "../../parameters_files/MNIST/mnist_aTTQ.json"
    ap.add_argument('--parameters_file', default=default_parameters_file, help="Parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file) as jf:
        parameters_exp = json.load(jf)

    # Grid search parameter in the parameters file
    if ('doGridSearch' not in parameters_exp):
        parameters_exp['doGridSearch'] = False
    doGridSearch = parameters_exp['doGridSearch']

    #==========================================================================#
    # Creating an instance of the experiment
    exp = Experiment(parameters_exp)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    resultsFolder = '../../results/' + parameters_exp['exp_id'] + '_' + current_datetime
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.mkdir(resultsFolder)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))

    # Creating directories for the trained models, the training and testing metrics
    # and the parameters of the model (i.e. the training parameters and the network
    # architecture)
    if (not doGridSearch):
        os.mkdir(resultsFolder + '/model/')
        os.mkdir(resultsFolder + '/metrics/')
    os.mkdir(resultsFolder + '/params_exp/')

    # Normalizing the dataset
    exp.compute_dataset_mean_std()
    exp.normalize_dataset()

    # Balancing the classes
    exp.balance_classes_loss()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

    # Evalauting the method
    if (not doGridSearch):
        # Doing holdout evaluation
        exp.holdout_train()
    else:
        # Doing grid search
        exp.gridSearch()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

    # Saving the python file containing the network architecture
    if (parameters_exp['model_type'].lower() == '2dcnn'):
        if (parameters_exp['model_to_use'].lower() == 'timefrequency2dcnn'):
            shutil.copy2('../Models/CNNs/time_frequency_simple_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
        elif (parameters_exp['model_to_use'].lower() == 'mnist2dcnn'):
            shutil.copy2('../Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
        else:
            raise ValueError('2D CNN {} is not valid'.format(parameters_exp['model_to_use']))

    elif (parameters_exp['model_type'].lower() == 'transformer'):
        if (parameters_exp['model_to_use'].lower() == 'rawaudiomultichannelcnn'):
            shutil.copy2('../Models/Transformers/Transformer_Encoder_RawAudioMultiChannelCNN.py', resultsFolder + '/params_exp/network_architecture.py')
        else:
            raise ValueError("Transformer type {} is not valid".format(parameters_exp['model_to_use']))
    else:
        raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))
    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
