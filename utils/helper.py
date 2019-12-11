import os
import json
import torch
import numpy as np
from scipy.stats import norm as dist_model

def cal_mean(lis):
    return sum(lis)/len(lis)


def parse_json(params_path):
    """
    parse params config from json file
    @param params_path: params config path
    @return: params config dictionary
    """
    with open (params_path) as f:
        params = json.load(f)
        return params


def fit(prob_pos_X):
    """
    caculate probability mean and standard deviation from the output probability distribution of model
    @param prob_pos_X: the output probability distribution  of model
    @return: mean and standard deviation of the probability distribution
    """
    prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std


def cal_thresh(pred_prob,labels):
    """
    calculate threshold from predicted probability
    @param pred_prob: the predicted probability of model
    @param labels: target label
    @return: list of mean and standard deviation probability of each class
    """
    mu_stds = []
    for i in range(19):
        pos_mu, pos_std = fit(pred_prob[labels==i, i])
        mu_stds.append([pos_mu, pos_std])
    return mu_stds


def convert_output_to_class(preds,mu_stds,use_thresh=True,scale=1.0):
    """
    convert the output of model to predicted class
    @param preds: the output of the model
    @param mu_stds: list of mean and standard deviation probability of each class
    @param scale:
    @return: the output predicted class
    """
    preds_prob = torch.sigmoid(preds) # convert logits to probability with sigmoid
    max_class = torch.argmax(preds_prob,dim=-1).numpy().tolist() # get class with the largest probability
    max_prob = torch.max(preds_prob,dim=-1).values.detach().numpy().tolist() # get the max value of probability
    pred_class = [] # predicted class
    for i in range(len(max_prob)): # loop each output of the model
        max_class_one = max_class[i] # get class with the largest probability
        threshold = max(0.5, 1. - scale * mu_stds[max_class_one][1]) if use_thresh is True else 0.5 # find threshold for the predicted class
        print(threshold)
        if max_prob[i] >= threshold: # if the max value of probability greater than threshold
            pred_class.append(max_class[i]) # append the max class
        else:
            pred_class.append(-1) # append unseen class
    return pred_class