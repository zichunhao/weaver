#!/usr/bin/env python

import os
import argparse
import numpy as np

from utils.data.fileio import _read_root
from utils.data.tools import  _get_variable_names
from utils.data.preprocess import _build_new_variables
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def plot_loss(indir,outdir,name):
    loss_vals_training = np.load('%s/loss_vals_training.npy'%indir)
    loss_vals_validation = np.load('%s/loss_vals_validation.npy'%indir)
    loss_std_training = np.load('%s/loss_std_training.npy'%indir)
    loss_std_validation = np.load('%s/loss_std_validation.npy'%indir)
    epochs = np.array(range(len(loss_vals_training)))
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, loss_vals_training, label='Training')
    ax.plot(epochs, loss_vals_validation, label='Validation', color = 'green')
    leg = ax.legend(loc='upper right', title=name, borderpad=1, frameon=False, fontsize=16)
    leg._legend_box.align = "right" 
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch') 
    ax.set_xlim(0,np.max(epochs))
    #ax.set_xlim(5,np.max(epochs))
    f.savefig('%s/Loss_%s.png'%(outdir,indir.replace('/','')))
    f.savefig('%s/Loss_%s.pdf'%(outdir,indir.replace('/','')))
    plt.clf()

def plot_accuracy(indir,outdir,name):
    acc_vals_validation = np.load('%s/acc_vals_validation.npy'%indir)
    epochs = np.array(range(len(acc_vals_validation)))
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, acc_vals_validation, label='Validation', color = 'green')
    leg = ax.legend(loc='upper right', title=name, borderpad=1, frameon=False, fontsize=16)
    leg._legend_box.align = "right"
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_xlim(0,np.max(epochs))
    ax.set_ylim(0.8,0.95)
    f.savefig('%s/Acc_%s.png'%(outdir,indir.replace('/','')))
    plt.clf()

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

def plot_features(table, scores, label_sig, label_bkg, name, features=['fj_sdmass','fj_pt']):
    labels = {'fj_sdmass': r'$m_{SD}$',
              'fj_pt': r'$p_{T}$',
              'orig_fj_sdmass': r'$m_{SD}$',
              'orig_fj_pt': r'$p_{T}$',
          }
    feature_range = {'fj_sdmass': [30,250],
                     'fj_pt': [200,2500],
                     'orig_fj_sdmass': [30,250],
                     'orig_fj_pt': [200,2500],
                 }

    def computePercentiles(data,percentiles):
        #stepsize = 0.05
        mincut = 0.
        #maxcut = 1.
        #percentiles = np.arange(stepsize, 1., stepsize)
        tmp = np.quantile(data,np.array(percentiles))
        tmpl = [mincut]
        for x in tmp: tmpl.append(x)
        #tmpl.append(maxcut)
        perc = [0.]
        for x in percentiles: perc.append(x)
        return perc,tmpl

    for score_name,score_label in scores.items():
        bkg = (table[label_bkg['label']] == 1)
        var = table[score_name][bkg]
        percentiles = [0.05,0.1,0.2,0.3]
        per,cuts = computePercentiles(table[score_name][bkg],percentiles)
        print(cuts)
        for k in features:
            fig, ax = plt.subplots(figsize=(10,10))
            bins = 25
            for i,cut in enumerate(cuts):
                ax.hist(table[k][bkg][var>cut], bins=bins, lw=2, density=True, range=feature_range[k],
                        histtype='step',label='{0}% mistag-rate'.format(per[i]*100))
            ax.legend(loc='best')
            ax.set_xlabel(labels[k]+' (GeV)')
            ax.set_ylabel('Number of events (normalized)')
            ax.set_title('%s dependence'%labels[k])
            plt.savefig("%s_%s.pdf"%(k,score_label))
            
def plot_response(table, scores, label_sig, label_bkg, name):
    for score_name,score_label in scores.items():
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
        plt.savefig("%s_%s_disc.pdf"%(score_label,label_sig['label']))
        #plt.yscale('log')
        #plt.savefig("%s_%s_disc_log.pdf"%(score_label,label_sig['label']))
        #plt.yscale('linear')

def plot_roc(table, scores, label_sig, label_bkg, name):
    plt.clf()
    for score_name,score_label in scores.items():
        truth, predict =  roc_input(table,score_name,label_sig['label'],label_bkg['label'])
        fpr, tpr, threshold = roc_curve(truth, predict)
        plt.plot(fpr, tpr, lw=2.5, label=r"{}, AUC = {:.1f}%".format(score_label,auc(fpr,tpr)*100))
    plt.legend(loc='lower right')
    plt.ylabel(r'Tagging efficiency %s'%label_sig['legend']) 
    plt.xlabel(r'Mistagging rate %s'%label_bkg['legend'])
    #plt.xscale('log')
    plt.savefig("roc_%s.pdf"%label_sig['label'])
    #plt.xscale('linear')    

def main(args):
    label_bkg = [{'legend': 'QCD',
                  'label':  'fj_isQCD'}]
    label_sig = [{'legend': 'H(WW)4q',
                  'label':  'label_H_WW_qqqq',
                  'scores': 'H4q'
                  },
                 {'legend': 'H(WW)lnuqq',
                  'label':  'label_H_WW_lnuqq',
                  'scores': 'Hlnuqq'
                  },
             ]
    scores = {'score_label_H_WW_qqqq': 'ParticleNet_H4q',
              'score_label_H_WW_lnuqq': 'ParticleNet_Hlnuqq',
              'score_deepBoosted_Hqqqq': 'DeepBoosted_H4q',
              #'pfDeepBoostedDiscriminatorsJetTags_H4qvsQCD': 'DeepBoosted_H4qvsQCD',
              #'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_H4qvsQCD': 'DeepBoosted_H4qvsQCD_MD',
              #'orig_pfDeepBoostedDiscriminatorsJetTags_H4qvsQCD': 'DeepBoosted_H4qvsQCD',
              #'orig_pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_H4qvsQCD': 'DeepBoosted_H4qvsQCD_MD',
    }
    funcs = {'score_deepBoosted_Hqqqq': 'pfDeepBoostedJetTags_probHqqqq/(pfDeepBoostedJetTags_probHqqqq+pfDeepBoostedJetTags_probQCDb+pfDeepBoostedJetTags_probQCDbb+pfDeepBoostedJetTags_probQCDc+pfDeepBoostedJetTags_probQCDcc+pfDeepBoostedJetTags_probQCDothers)',
             'score_deepBoosted_Hqqqq': 'orig_pfDeepBoostedJetTags_probHqqqq/(orig_pfDeepBoostedJetTags_probHqqqq+orig_pfDeepBoostedJetTags_probQCDb+orig_pfDeepBoostedJetTags_probQCDbb+orig_pfDeepBoostedJetTags_probQCDc+orig_pfDeepBoostedJetTags_probQCDcc+orig_pfDeepBoostedJetTags_probQCDothers)',
         }

    # inputfiles should have same shape
    inputfiles = args.input.split(',')
    names = args.name.split(',')
    
    loadbranches = {}
    #lfeatures = ['fj_sdmass','fj_pt']
    lfeatures = ['orig_fj_sdmass','orig_fj_pt']
    for n,i in enumerate(names):
        loadbranches[i] = set()
        for k,kk in scores.items():
            if n>0 and 'DeepBoosted' in kk: continue
            if k in funcs.keys(): loadbranches[i].update(_get_variable_names(funcs[k]))
            else: loadbranches[i].add(k)
        if n==0:
            for k in label_bkg: loadbranches[i].add(k['label'])
            for k in label_sig: loadbranches[i].add(k['label'])
            for k in lfeatures: loadbranches[i].add(k)

    for n,i in enumerate(names):
        table = _read_root(inputfiles[n], loadbranches[i])
        if n==0 and 'DeepBoosted' in kk:
            _build_new_variables(table, {k: v for k,v in funcs.items() if k in scores.keys()})
        if n==0: 
            newtable = table
            #for k in table:
            #    if 'score_label' in table:
            #        newtable[k+'_'+i] = table[k]
        else:
            for k in table: 
                newtable[k+'_'+i] = table[k]
                score = scores[k]
                scores[k+'_'+i] = score+'_'+i

    cwd=os.getcwd()
    odir = 'plots/%s/'%(args.tag)
    os.system('mkdir -p %s'%odir)
    os.chdir(odir)
    
    for sig in label_sig:
        newscores = {}
        for k,kk in scores.items():
            if sig['scores'] in kk:
                newscores[k] = kk
        plot_roc(newtable, newscores, sig, label_bkg[0], args.name+sig['label'])
        plot_response(newtable, newscores, sig, label_bkg[0], args.name+sig['label'])
        plot_features(newtable, newscores, sig, label_bkg[0], args.name+sig['label'], lfeatures)

    os.chdir(cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('--name', help='name ROC')
    parser.add_argument('--tag', help='folder tag')
    parser.add_argument('--idir', help='idir')
    parser.add_argument('--odir', help='odir')
    args = parser.parse_args()
    main(args)
    plot_loss(args.idir,args.odir,args.name)
    plot_accuracy(args.idir,args.odir,args.name)
