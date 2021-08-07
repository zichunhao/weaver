#!/usr/bin/env python

import os
import argparse
import numpy as np
import uproot
from coffea import hist

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def plot_loss(args,name,indir=None):
    if indir:
        # save the loss in a numpy file
        loss_vals_training = np.load('%s/loss_vals_training.npy'%indir)
        loss_vals_validation = np.load('%s/loss_vals_validation.npy'%indir)
    else:
        # or load the loss array manually
        loss_vals_training = np.array([0.001,0.0002])
        loss_vals_validation = np.array([0.001,0.0002])        
    epochs = np.array(range(len(loss_vals_training)))
    
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, loss_vals_training, label='Training')
    ax.plot(epochs, loss_vals_validation, label='Validation', color = 'green')
    leg = ax.legend(loc='upper right', title=name, borderpad=1, frameon=False, fontsize=16)
    leg._legend_box.align = "right" 
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch') 
    ax.set_xlim(0,np.max(epochs))
    f.savefig('%s/Loss_%s.pdf'%(args.odir,indir.replace('/','')))
    plt.clf()

def plot_accuracy(args,name,indir=None):
    if indir:
        acc_vals_validation = np.load('%s/acc_vals_validation.npy'%indir)
    else:
        acc_vals_validation =  np.array([0.001,0.0002])
    epochs = np.array(range(len(acc_vals_validation)))
    
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, acc_vals_validation, label='Validation', color = 'green')
    leg = ax.legend(loc='upper right', title=name, borderpad=1, frameon=False, fontsize=16)
    leg._legend_box.align = "right"
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_xlim(0,np.max(epochs))
    ax.set_ylim(0.8,0.95)
    f.savefig('%s/Acc_%s.pdf'%(args.odir,indir.replace('/','')))
    plt.clf()

# return input by classes (signal and background)
def roc_input(events,var,label_sig,label_bkg):
    scores_sig = events[var][(events[label_sig] == 1)].to_numpy()
    scores_bkg = events[var][(events[label_bkg] == 1)].to_numpy()
    predict = np.concatenate((scores_sig,scores_bkg),axis=None)
    siglabels = np.ones(scores_sig.shape)
    bkglabels = np.zeros(scores_bkg.shape)
    truth = np.concatenate((siglabels,bkglabels),axis=None)
    return truth, predict

# get roc for a table with given scores, a label for signal, and one for background
def get_roc(events, scores, label_sig, label_bkg):
    fprs = {}
    tprs = {}
    for score_label,score_name in scores.items():
        truth, predict =  roc_input(events,score_name,label_sig,label_bkg)
        fprs[score_label], tprs[score_label], threshold = roc_curve(truth, predict)
    return fprs, tprs

# plot roc
def plot_roc(args, label_sig, label_bkg, fprs, tprs):
    fig, axs = plt.subplots(1,1,figsize=(16,16))
    
    def get_round(x_effs,y_effs,to_get=[0.01,0.02,0.03]):
        effs = []
        for eff in to_get:
            for i,f in enumerate(x_effs):
                if round(f,2) == eff:
                    effs.append(y_effs[i])
                    # print signal efficiencies
                    # print(round(f,2),y_effs[i])
                    break
        return effs
    
    markers = ['v','^','o','s']
    for k,it in fprs.items():
        axs.plot(tprs[k], fprs[k], lw=2.5, label=r"{}, AUC = {:.1f}%".format(k,auc(fprs[k],tprs[k])*100))
        y_effs = [0.01,0.02,0.03]
        x_effs = get_round(fprs[k],tprs[k],y_effs)
        #print(tprs[k])
        #print(y_effs)
        axs.scatter(x_effs,y_effs,marker=markers[0],label=k)

    axs.legend(loc='upper left')
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    axs.set_xlabel(r'Tagging efficiency %s'%label_sig['legend'])
    axs.set_ylabel(r'Mistagging rate %s'%label_bkg['legend'])
    #fig.savefig("%s/roc_%s.pdf"%(args.odir,label_sig['label']))
    axs.set_yscale('log')
    fig.savefig("%s/roc_%s_ylog.pdf"%(args.odir,label_sig['label']))
    axs.set_yscale('linear')

# plot validation 
def plot_validation(args,hist_val,vars_to_plot,label):
    for density in [True,False]:
        fig, axs = plt.subplots(1,len(vars_to_plot), figsize=(len(vars_to_plot)*8,8))
        for i,m in enumerate(vars_to_plot):
            if(len(vars_to_plot)==1):
                axs_1 = axs
            else:
                axs_1 = axs[i]
            x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process',m}])
            # print for debugging
            # print(x.values())
            hist.plot1d(x,ax=axs_1,overlay="process",density=density)
            axs_1.set_ylabel('Jets')
        fig.tight_layout()
        if density:
            fig.savefig("%s/%s_density.pdf"%(args.odir,label))
        else:
            fig.savefig("%s/%s.pdf"%(args.odir,label))

# plot score after selection on variables 
# i.e. how does the score look when cutting on e.g. pt, gmass
def plot_score_aftercut(args,hist_val,vars_to_corr,bins,processes,label):
    print('plot_score_aftercut')
    density = True
    for proc in processes:
        fig, axs = plt.subplots(1,len(vars_to_corr), figsize=(len(vars_to_corr)*8,8))
        for i,m in enumerate(vars_to_corr):
            if(len(vars_to_corr)==1):
                axs_1 = axs
            else:
                axs_1 = axs[i]
            x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process','score',m}]).integrate("process",proc)
            print(x,m)
            for j,b in enumerate(bins):
                y = x.integrate(m, slice(60+100))
                print(y)
                print(y.values())
                if j==0:
                    hist.plot1d(y,ax=axs_1,density=True)
                else:
                    hist.plot1d(y,ax=axs_1,density=True,clear=False)
        fig.tight_layout()
        fig.savefig("%s/%s_scores_%s_density.pdf"%(args.odir,proc,label))

# plot how variables look after a cut on the scores
def plot_var_aftercut(args,hist_val,vars_to_corr,bins,processes,label):
    density = True
    for proc in processes:
        fig, axs = plt.subplots(1,len(vars_to_corr), figsize=(len(vars_to_corr)*8,8))
        fig.tight_layout()
        fig.savefig("%s/%s_scores_%s_density.pdf"%(args.odir,proc,label))

    
def main(args):
    # labels here
    label_dict = {
        'qcd':{'legend': 'QCD',
               'label':  'fj_QCD_label'},
               # 'label':  'fj_isQCD'}, # for old training datasets
        'qcd_b': {'legend': 'QCDb',
                  'label':  'fj_isQCDb'},
        'qcd_bb': {'legend': 'QCDbb',
                   'label':  'fj_isQCDbb'},
        'qcd_c': {'legend': 'QCDc',
                  'label':  'fj_isQCDc'},
        'qcd_cc': {'legend': 'QCDcc',
                   'label':  'fj_isQCDcc'},
        'qcd_lep': {'legend': 'QCDlep',
                    'label':  'fj_isQCDlep'},
        'qcd_lep': {'legend': 'QCDlep',
                    'label':  'fj_isQCDlep'},
        
        'top':{'legend': 'Top',
                   'label':  'fj_isTop_label'},
        'top_lep':{'legend': 'Top lep',
                   'label':  'fj_isToplep'},
        'top_merged': {'legend': 'Top merged',
                       'label': 'fj_isTop_merged'},
        'top_semimerged': {'legend': 'Top semi-merged',
                           'label': 'fj_isTop_semimerged'},
        'top_lepmerged':{'legend': 'Top merged lepton',
                         'label': 'fj_isToplep_merged'},
        
        'hww_4q': {'legend': 'H(WW) all-had',
                   'label': 'fj_H_WW_4q'},
        'hww_4q_merged':{'legend': 'H(WW) 4q',
                         'label': 'fj_H_WW_4q_4q'},
        'hww_3q_merged':{'legend': 'H(WW) 3q',
	                 'label': 'fj_H_WW_4q_3q'},
        'hbb':{'legend': 'H(bb)',
               'label': 'fj_H_bb'},
        
        'hww_elenuqq':{'legend': 'H(WW) ele',
                       'label':  'fj_H_WW_elenuqq'},
        'hww_munuqq':{'legend': 'H(WW) mu',
                      'label':  'fj_H_WW_munuqq'},
        'hww_taunuqq':{'legend': 'H(WW) tau merged',
                       'label':  'fj_isHWW_taunuqq_merged'},
        
        'hww_munuqq_merged':{'legend': 'H(WW) mu merged',
                             'label': 'fj_isHWW_munuqq_merged'},
        'hww_munuqq_semimerged': {'legend': 'H(WW) mu semi-merged',
                                  'label': 'fj_isHWW_munuqq_semimerged'},
        'hww_elenuqq_merged':{'legend': 'H(WW) ele merged',
                              'label': 'fj_isHWW_elenuqq_merged'},
        'hww_elenuqq_semimerged': {'legend': 'H(WW) ele semi-merged',
                                   'label': 'fj_isHWW_elenuqq_semimerged'},
        'hww_taunuqq_merged':{'legend': 'H(WW) tau merged',
                              'label': 'fj_isHWW_taunuqq_merged'},
        'hww_taunuqq_semimerged': {'legend': 'H(WW) tau semi-merged',
                                   'label': 'fj_isHWW_taunuqq_semimerged'},
    }

    signals = args.signals.split(',')
    backgrounds = args.bkgs.split(',')

    if len(signals)!=len(backgrounds):
        print('Number of signals should be the same as backgrounds!')
        exit

    for i,signal in enumerate(signals):
        bkg = backgrounds[i]
        bkglabel = label_dict[bkg]['label']
        siglabel = label_dict[signal]['label']

        # default branches
        branches = ['fj_pt','fj_msoftdrop','fj_genRes_mass']
        #branches = ['fj_pt','fj_msoftdrop','fj_genH_mass'] # for old training datasets
        branches += [siglabel]
        branches += ['score_%s'%siglabel,'score_%s'%bkglabel]
        # possibly can add older taggers
        # TODO: add labels for these old taggers
        # branches += ["fj_deepTagMD_H4qvsQCD","fj_deepTag_HvsQCD","fj_PN_XbbvsQCD"]

        # add selection
        mask = "(fj_pt<1200)"
        
        ifile = uproot.open(args.ifile)["Events"]
        isqcd_separate = False
        if 'qcd' in bkg:
            try:
                #ibranches = branches + ["fj_QCD_label"]
                ibranches = branches + ["fj_isQCD"]
                print('List of branches to read ',ibranches)
                events = ifile.arrays(ibranches,mask)
            except:
                ibranches = branches + ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]
                ibranches.remove('score_%s'%bkglabel)
                ibranches.extend(['score_fj_isQCDb','score_fj_isQCDbb','score_fj_isQCDc','score_fj_isQCDcc','score_fj_isQCDlep','score_fj_isQCDothers'])
                print('List of branches to read ',ibranches)
                events = ifile.arrays(ibranches,mask)
                events["fj_QCD_label"] = (events["fj_isQCDb"]==1) | (events["fj_isQCDbb"]==1) | (events["fj_isQCDc"]==1) | (events["fj_isQCDcc"]==1) | (events["fj_isQCDcc"]==1)
                isqcd_separate = True
                print('Added fj_QCD_label to ttree')
        elif 'top' in bkg:
            ibranches = branches + ["fj_Top_label"]
            events = ifile.arrays(ibranches)
        else:
            print('not known background')
            
        # compute scores:
        #  we expect all scores to sum up to 1, e.g. given two signals in the event (signal 1 and 2) and one background process (background 1):
        #  score_signal_1 + score_signal_2 + score_background_1 = 1
        #  then nn_signal_1 = score_signal_1 / (score_signal_1 + score_background_1) = score_signal_1 / (1 - score_signal_2)
        score_name = "%s_score"%args.name
        if isqcd_separate:
            events[score_name] = events['score_%s'%siglabel] / (events['score_%s'%siglabel] + events['score_fj_isQCDb'] + events['score_fj_isQCDbb'] + events['score_fj_isQCDc'] + events['score_fj_isQCDcc'] + events['score_fj_isQCDlep'] + events['score_fj_isQCDothers'])
        else:
            events[score_name] = events['score_%s'%siglabel] / (events['score_%s'%siglabel] + events['score_%s'%bkglabel])

        # define scores_dict for which to compute rocs
        scores_dict = {
            args.name: score_name,
        }
        
        # get roc
        # TODO: for now only get roc curve for the main score
        fprs, tprs = get_roc(events, scores_dict, siglabel, bkglabel)

        # define and fill coffea histograms
        # TODO: add other scores here if needed
        hist_features = hist.Hist("Jets",
                                  hist.Cat("process", "Process"),
                                  hist.Bin("msd", "fj msoftdrop [GeV]", 60, 0, 260),
                                  hist.Bin("pt", r"fj $p_T$ [GeV]", 70, 200, 1200),
                                  hist.Bin("score", "Tagger score", 70, 0, 1),
                                  hist.Bin("gmass", "gen Res mass [GeV]", 42, 50, 260),
        )

        # define processes
        # add mh125 and mh!=125 as different processes so that we can see dependence
        processes = [bkg]
        if 'hww' in signal or 'hbb' in signal:
            processes.append('%s-mh125'%signal)
            processes.append('%s-mhflat'%signal)

        # loop over processes
        legends = {}
        for proc in processes:
            p = proc.split('-')[0]
            legend = label_dict[p]['legend']
            if 'mh125' in proc:
                mask_proc = (events[label_dict[p]['label']]==1) & (events["fj_genRes_mass"]==125)
                legend += ' mH125'
            elif 'mhflat' in proc:
                mask_proc = (events[label_dict[p]['label']]==1) & (events["fj_genRes_mass"]!=125)
                legend += ' mHflat'
            else:
                mask_proc = (events[label_dict[p]['label']]==1)
            legends[proc] = legend
            # check if events with that mask are not zero
            if len(events["fj_msoftdrop"][mask_proc])==0:
                processes.remove(proc)
                continue
            # print legends
            print('legend ',legend)
            hist_features.fill(process=proc,
                               msd = events["fj_msoftdrop"][mask_proc],
                               pt = events["fj_pt"][mask_proc],
                               score = events[score_name][mask_proc],
                               gmass = events["fj_genRes_mass"][mask_proc],
            )

        # plot features for this signal and background combination (i.e. all the processes) 
        vars_to_plot = ["pt","msd","score"]
        plt_label = "validation_%svs%s"%(siglabel,bkglabel)
        plot_validation(args,hist_features,vars_to_plot,plt_label)

        # plot how the score looks after cuts on variables
        vars_to_corr = ["gmass"]
        bins_to_corr = [[50,55]]
        #proc_to_corr = [legends['%s-mhflat'%signal]]
        proc_to_corr = ['%s-mhflat'%signal]
        plt_label = "%svs%s"%(siglabel,bkglabel)
        print('plot score')
        plot_score_aftercut(args,hist_features,vars_to_corr,bins_to_corr,proc_to_corr,plt_label)
        
        # plot roc
        plot_roc(args, label_dict[signal], label_dict[bkg], fprs, tprs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifile', help='input file(s)')
    parser.add_argument('--odir', required=True, help="output dir")
    parser.add_argument('--name', help='name of the model(s)')
    parser.add_argument('--signals', default='hww_4q_merged', help='signals')
    parser.add_argument('--bkgs', default='qcd', help='backgrounds') 
    args = parser.parse_args()

    import os
    os.system('mkdir -p %s'%args.odir)
    
    main(args)
