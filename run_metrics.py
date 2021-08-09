#!/usr/bin/env python

import os
import argparse
import numpy as np

from utils.data.fileio import _read_root
from utils.data.tools import  _get_variable_names
from utils.data.preprocess import _build_new_variables
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

def roc_input(table,var,label_sig,label_bkg):
    scores_sig = np.zeros(table[var].shape[0])
    scores_bkg = np.zeros(table[var].shape[0])
    scores_sig = table[var][(table[label_sig] == 1)]
    scores_bkg = table[var][(table[label_bkg] == 1)]
    predict = np.concatenate((scores_sig,scores_bkg),axis=None)
    siglabels = np.ones(scores_sig.shape)
    bkglabels = np.zeros(scores_bkg.shape)
    truth = np.concatenate((siglabels,bkglabels),axis=None)
    return truth, predict

def plot_response(table, scores, label_sig, label_bkg, name):
    for score_label,score_name in scores.items():
        plt.clf()
        bins=100
        var = table[score_name]
        data = [var[(table[label_sig['label']] == 1)], 
                var[(table[label_bkg['label']] == 1)]]
        labels = [label_sig['legend'],
                  label_bkg['legend']]
        for j in range(0,len(data)):
            plt.hist(data[j],bins,log=False,histtype='step',density=True,label=labels[j],fill=False,range=(-1.,1.))
        plt.legend(loc='best')
        plt.xlim(0,1)
        plt.xlabel('%s Response'%score_label)
        plt.ylabel('Number of events (normalized)')
        plt.title('NeuralNet applied to test samples')
        plt.savefig("plots/%s_%s_disc.pdf"%(score_label,name))
        plt.yscale('log')
        plt.savefig("plots/%s_%s_disc_log.pdf"%(score_label,name))
        plt.yscale('linear')

def plot_roc(table, scores, label_sig, label_bkg, name):
    plt.clf()
    for score_label,score_name in scores.items():
        truth, predict =  roc_input(table,score_name,label_sig['label'],label_bkg['label'])
        fpr, tpr, threshold = roc_curve(truth, predict)
        plt.plot(fpr, tpr, lw=2.5, label=r"{}, AUC = {:.1f}%".format(score_label,auc(fpr,tpr)*100))
    plt.legend(loc='lower right')
    plt.ylabel(r'Tagging efficiency %s'%label_sig['legend']) 
    plt.xlabel(r'Mistagging rate %s'%label_bkg['legend'])
    plt.savefig("plots/roc_%s.pdf"%name)

def main(args):
    label_bkg = [{'legend': 'QCD',
                  'label':  'fj_isQCD'}]
    label_sig = [{'legend': 'H(WW)4q',
                  'label':  'label_H_WW_qqqq'}]
    scores = {'ParticleNetv0': 'score_label_H_WW_qqqq',
              'DeepAK8':'score_deepBoosted_Hqqq'}
    funcs = {'score_deepBoosted_Hqqq': 'pfDeepBoostedJetTags_probHqqqq/(pfDeepBoostedJetTags_probHqqqq+pfDeepBoostedJetTags_probQCDb+pfDeepBoostedJetTags_probQCDbb+pfDeepBoostedJetTags_probQCDc+pfDeepBoostedJetTags_probQCDcc+pfDeepBoostedJetTags_probQCDothers)'
         }
    loadbranches = set()
    for k in scores.values():
        if k in funcs.keys():
            expr = funcs[k]
            loadbranches.update(_get_variable_names(expr))
        else:
            loadbranches.add(k)
    for k in label_bkg: loadbranches.add(k['label'])
    for k in label_sig: loadbranches.add(k['label'])

    table = _read_root(args.input, loadbranches)
    _build_new_variables(table, {k: v for k,v in funcs.items() if k in scores.values()})

    for sig in label_sig:
        plot_roc(table, scores, sig, label_bkg[0], args.name)
        plot_response(table, scores, sig, label_bkg[0], args.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('--name', help='name ROC')
    args = parser.parse_args()
    main(args)
