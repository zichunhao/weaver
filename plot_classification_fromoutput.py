#!/usr/bin/env python

import os
import argparse
import numpy as np
import uproot
from coffea import hist
import awkward as ak

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

mass_branch = 'fj_genRes_mass'
#mass_branch = 'fj_genH_mass'
        
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
def roc_input(events,var,label_sig,label_bkg,weight_hist=None,bins=None,sig_mask=None,bkg_mask=None):
    mask_sig = (events[label_sig] == 1)
    mask_bkg = (events[label_bkg] == 1)
    if sig_mask is not None:
        mask_sig = (mask_sig) & (sig_mask)
    if bkg_mask is not None:
        mask_bkg = (mask_bkg) & (bkg_mask)
    scores_sig = events[var][mask_sig].to_numpy()
    scores_bkg = events[var][mask_bkg].to_numpy()
    predict = np.concatenate((scores_sig,scores_bkg),axis=None)
    siglabels = np.ones(scores_sig.shape)
    bkglabels = np.zeros(scores_bkg.shape)
    truth = np.concatenate((siglabels,bkglabels),axis=None)

    weight = None
    if weight_hist is not None:
        weight_sig = weight_hist[np.digitize(events['fj_pt'][mask_sig].to_numpy(), bins)-1]
        weight_bkg = np.ones(scores_bkg.shape)
        weight = np.concatenate((weight_sig,weight_bkg),axis=None)

    return truth, predict, weight

# get roc for a table with given scores, a label for signal, and one for background
def get_roc(events, score_name, label_sig, label_bkg, weight_hist=None,bins=None,sig_mask=None,bkg_mask=None):
    truth, predict, weight =  roc_input(events,score_name,label_sig,label_bkg, weight_hist,bins,sig_mask,bkg_mask)
    fprs, tprs, threshold = roc_curve(truth, predict, sample_weight=weight)
    return fprs, tprs

# plot roc
def plot_roc(args, label_sig, label_bkg, fprs, tprs, label):
    fig, axs = plt.subplots(1,1,figsize=(16,16))
    
    def get_round(x_effs,y_effs,to_get=[0.01,0.02,0.03]):
        effs = []
        for eff in to_get:
            for i,f in enumerate(x_effs):
                if round(f,2) == eff:
                    effs.append(y_effs[i])
                    break
        return effs

    def get_intersections(x_effs, y_effs, to_get=0.01):
        x_eff = 0
        for i,f in enumerate(y_effs):
            if f >= to_get:
                x_eff = x_effs[i]
                break
        return x_eff
    
    ik = 0
    markers = ['v','^','o','s','p','P','h']
    for k,it in fprs.items():
        leg = k.replace('_score','')
        axs.plot(tprs[k], fprs[k], lw=2.5, label=r"{}, AUC = {:.1f}%".format(leg,auc(fprs[k],tprs[k])*100))
        y_eff = 0.01
        x_eff = get_intersections(tprs[k], fprs[k],y_eff)
        axs.hlines(y=y_eff, xmin=0.00001, xmax=0.99999, linewidth=1, color='dimgrey',linestyle="dashed")
        axs.vlines(x=x_eff, ymin=0.00001, ymax=y_eff, linewidth=1, color='dimgrey',linestyle="dashed")
        #y_effs = [0.01,0.02,0.03]
        #x_effs = get_round(fprs[k],tprs[k],y_effs)
        #print(tprs[k],k)
        #print(y_effs)
        #axs.scatter(x_effs,y_effs,s=75,marker=markers[ik],label=leg)
        ik+=1
        
    axs.legend(loc='upper left')
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    axs.set_xlabel(r'Tagging efficiency %s'%label_sig['legend'])
    axs.set_ylabel(r'Mistagging rate %s'%label_bkg['legend'])
    axs.set_ylim(0.0001,1)
    axs.set_xlim(0.0001,1)
    axs.set_yscale('log')
    fig.savefig("%s/roc_%s_ylog.pdf"%(args.odir,label))
    axs.set_yscale('linear')

# plot rocs for different cuts on mH and pt
def plot_roc_by_var(args,vars_to_corr,bin_ranges,bin_widths,events,score_name,sig,bkg,mh=False):
    i = 0
    fig, axs = plt.subplots(1,len(vars_to_corr.keys()),figsize=(8*len(vars_to_corr.keys()),8))
    for var,varname in vars_to_corr.items():
        fprs = {}; tprs = {}
        legends = []
        for j,b in enumerate(bin_ranges[i]):
            bi = b
            bf = b+bin_widths[i]
            tag = "%i"%bi
            if not mh:
                output = events[(events[bkg['label']]==1) | ((events[varname]>=bi) & (events[varname]<=bf) & (events[mass_branch]!=125))]
                # mask_sig = (events[varname]>=bi) & (events[varname]<=bf) & (events[mass_branch]!=125)
            else:
                # mask_sig = (events[mass_branch]==125)
                output = events[(events[bkg['label']]==1) | ((events[varname]>=bi) & (events[varname]<=bf) & (events[mass_branch]==125))]
            fprs[tag], tprs[tag] = get_roc(output, score_name, sig['label'], bkg['label'])
            legends.append('%s %i-%i GeV'%(var,bi,bf))

        # now plot
        if(len(vars_to_corr.keys())==1):
            axs_1 = axs
        else:
            axs_1 = axs[i]
            
        ik=0
        for k,it in fprs.items():
            axs_1.plot(tprs[k], fprs[k], lw=2.5, label=r"{}, AUC = {:.1f}%".format(legends[ik],auc(fprs[k],tprs[k])*100))
            ik+=1
        axs_1.legend(loc='upper left')
        axs_1.grid(which='minor', alpha=0.2)
        axs_1.grid(which='major', alpha=0.5)
        axs_1.set_xlabel(r'Tagging efficiency %s'%sig['legend'])
        axs_1.set_ylabel(r'Mistagging rate %s'%bkg['legend'])
        axs_1.set_ylim(0.0001,1)
        axs_1.set_yscale('log')

        i+=1

    if not mh:
        fig.savefig("%s/rocs_by_var_%s_ylog.pdf"%(args.odir,sig['label']))
    else:
        fig.savefig("%s/rocs_by_var_%s_mh125_ylog.pdf"%(args.odir,sig['label']))
        
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
            # print hist values for debugging  (in case hist is empty)
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
def plot_score_aftercut(args,hist_val,vars_to_corr,bin_ranges,bin_widths,processes,label):
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
            legends = []
            for j,b in enumerate(bin_ranges[i]):
                # print histogram identifiers for debugging
                #print(x.identifiers(m, overflow='all'))
                y = x.integrate(m, slice(b,b+bin_widths[i]))
                legends.append('%s %i-%i GeV'%(m,b,b+bin_widths[i]))
                #print(b,b+bin_widths[i])
                #print(y.values())
                #print(x.values())
                if j==0:
                    hist.plot1d(y,ax=axs_1,density=True)
                else:
                    hist.plot1d(y,ax=axs_1,density=True,clear=False)
            axs_1.set_ylabel('Jets')
            axs_1.legend(legends,title=m)
        fig.tight_layout()
        fig.savefig("%s/%s_scores_%s_density.pdf"%(args.odir,proc,label))

# compute percentiles
"""
i.e. the cuts that we should make on the tagger score so that we obtain this efficiency in our process after the cut                                                                           
uses np.quantile function                                                                                             
"""
def computePercentiles(data, percentiles):
    mincut = 0.
    tmp = np.quantile(data, np.array(percentiles))
    tmpl = [mincut]
    for x in tmp:
        tmpl.append(x)
    perc = [0.]
    for x in percentiles:
        perc.append(x)
    return perc, tmpl

# plot how variables look after a cut on the scores 
def plot_var_aftercut(args,hist_val,vars_to_plot,processes,label,cuts,percentiles):
    print('plot variable after cut')
    for density in [True,False]:
        for proc in processes:
            fig, axs = plt.subplots(1,len(vars_to_plot), figsize=(len(vars_to_plot)*8,8))
            for var in vars_to_plot:
                if(len(vars_to_plot)==1):
                    axs_1 = axs
                else:
                    axs_1 = axs[i]
                x = hist_val.sum(*[ax for ax in hist_val.axes() if ax.name not in {'process',var,'score'}])
                x = x.integrate('process',proc)
                legends = []
                # now cut on the score
                if not density:
                    cuts.pop(0)
                    percentiles.pop(0)
                #print('cuts on score ',cuts)
                for i,cut in enumerate(cuts):
                    cut = round(cut,2)
                    #if i==len(cuts)-1:
                    #print(slice(cut,1))
                    y = x.integrate('score',slice(cut,1))
                    legends.append('%s '%(percentiles[i]))
                    if i==0:
                        hist.plot1d(y,ax=axs_1,density=density)
                    else:
                        hist.plot1d(y,ax=axs_1,density=density,clear=False)
                axs_1.set_ylabel('Jets')
                axs_1.legend(legends,title='Bkg quantile')
            fig.tight_layout()
            if density:
                fig.savefig("%s/%s_scoresculpting_density.pdf"%(args.odir,label))
            else:
                fig.savefig("%s/%s_scoresculpting.pdf"%(args.odir,label))
    
def main(args):
    
    # label dictionary
    label_dict = {
        'qcd_old': {'legend': 'QCD',
                    'label':  'fj_isQCD'},
        'qcd':{'legend': 'QCD',
               'label':  'fj_QCD_label'},
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

    # get signals and backgrounds
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
        branches = ['fj_pt','fj_msoftdrop']
        branches += [mass_branch]
        branches += [siglabel]
        branches += ['score_%s'%siglabel,'score_%s'%bkglabel]
        # add older taggers
        if signal=="hww_4q" or signal=="hww_4q_merged" or signal=="hww_3q_merged":
            # branches += ["fj_deepTagMD_H4qvsQCD","fj_deepTag_HvsQCD"] # values are always -1000?
            branches += ["fj_PN_H4qvsQCD"] # when included in observers?

        # add this only if nProngs is saved
        add_nProngs = False
        #add_nProngs = True
        if add_nProngs:
            branches += ["fj_nProngs"]

        # add selection (add selection such that QCD only has gen resonance mass < 0)
        mask = "(fj_pt<1200) &"\
            "(fj_pt>300) &"\
            "((((fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1)) & (fj_genRes_mass<0)) |"\
            "((%s==1) & (fj_genRes_mass>0) ) )"%siglabel
        if bkg=='qcd_old':
            mask = "(fj_pt<1200) & (fj_pt>300) & (((fj_isQCD==1) & (%s<0)) | ((%s==1) & (%s>0) ) )"%(mass_branch,siglabel,mass_branch)
        
        ifile = uproot.open(args.ifile)["Events"]
        isqcd_separate = False
        if 'qcd' in bkg:
            if bkg=='qcd':
                ibranches = branches + ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]
                ibranches.remove('score_%s'%bkglabel)
                ibranches.extend(['score_fj_isQCDb','score_fj_isQCDbb','score_fj_isQCDc','score_fj_isQCDcc','score_fj_isQCDlep','score_fj_isQCDothers'])
                # print(ibranches)
                events = ifile.arrays(ibranches,mask)
                events_fj_QCD_label_TrueFalse = ((events["fj_isQCDb"]==1) | (events["fj_isQCDbb"]==1) | (events["fj_isQCDc"]==1) | (events["fj_isQCDcc"]==1) | (events["fj_isQCDlep"]==1) | (events["fj_isQCDothers"]==1))
                events["fj_QCD_label"] = ak.values_astype(events_fj_QCD_label_TrueFalse, int)
                isqcd_separate = True
                print('Added fj_QCD_label to ttree')
            else:
                #ibranches = branches + ["fj_QCD_label"]
                print(branches)
                ibranches = branches + ["fj_isQCD"]
                events = ifile.arrays(ibranches,mask)
        elif 'top' in bkg:
            ibranches = branches + ["fj_Top_label"]
            events = ifile.arrays(ibranches)
        else:
            print('not known background')
        print('List of branches read ',ibranches)
            
        # compute scores:
        """
          we expect all scores to sum up to 1, e.g. given two signals in the event (signal 1 and 2) and one background process (background 1):
          score_signal_1 + score_signal_2 + score_background_1 = 1
          then nn_signal_1 = score_signal_1 / (score_signal_1 + score_background_1) = score_signal_1 / (1 - score_signal_2)
        """
        score_name = "%s_score"%args.name
        if isqcd_separate:
            events[score_name] = events['score_%s'%siglabel] / (events['score_%s'%siglabel] + events['score_fj_isQCDb'] + events['score_fj_isQCDbb'] + events['score_fj_isQCDc'] + events['score_fj_isQCDcc'] + events['score_fj_isQCDlep'] + events['score_fj_isQCDothers'])
        else:
            events[score_name] = events['score_%s'%siglabel] / (events['score_%s'%siglabel] + events['score_%s'%bkglabel])
        
        # define and fill coffea histograms
        hist_features = hist.Hist("Jets",
                                  hist.Cat("process", "Process"),
                                  hist.Bin("msd", r"fj msoftdrop [GeV]", 60, 30, 420), 
                                  hist.Bin("pt", r"fj $p_T$ [GeV]", 50, 200, 1200), # bins of 20
                                  hist.Bin("score", r"Tagger score", 100, 0, 1),
                                  )
        hist_gmass = hist.Hist("genmass",
                               hist.Cat("process", "Process"),
                               hist.Bin("score", r"Tagger score", 70, 0, 1),
                               hist.Bin("pt", r"fj $p_T$ [GeV]", 50, 200, 1200), # bins of 20
                               hist.Bin("mH", r"gen Res mass [GeV]", 42, 50, 260), # bins of 5
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
                mask_proc = (events[label_dict[p]['label']]==1) & (events[mass_branch]==125)
                legend += ' mh125'
            elif 'mhflat' in proc:
                # beware of mRes = 175....
                mask_proc = (events[label_dict[p]['label']]==1) & (events[mass_branch]!=125)
                legend += ' mhflat'
            else:
                mask_proc = (events[label_dict[p]['label']]==1)
            legends[proc] = legend
            # check if events with that mask are not zero
            if len(events["fj_msoftdrop"][mask_proc])==0:
                processes.remove(proc)
                continue
            
            # print legends
            # print('legend ',legend)

            # fill the features histogram
            hist_features.fill(process=proc,
                               msd = events["fj_msoftdrop"][mask_proc],
                               pt = events["fj_pt"][mask_proc],
                               score = events[score_name][mask_proc],
            )

            # only fill the gen mass histogram for signal
            if signal in proc:
                hist_gmass.fill(process=proc,
                                mH = events[mass_branch][mask_proc],
                                score = events[score_name][mask_proc],
                                pt = events["fj_pt"][mask_proc],
                )


        # get pt histograms
        # define log bins as: np.round(np.exp(np.linspace(np.log(MIN), np.log(MAX), NUM_BINS))).astype('int').tolist()
        #ptbins = [200, 239, 286, 342, 409, 489, 585, 699, 836, 1000, 2500]
        ptbins = [200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500]
        mask_proc_sigmh125 = (events[label_dict[signal]['label']]==1) & (events[mass_branch]==125)
        mask_proc_sig = (events[label_dict[signal]['label']]==1) & (events[mass_branch]!=125)
        pthistsigmh, bin_edges = np.histogram(events["fj_pt"][mask_proc_sigmh125].to_numpy(), bins=ptbins)
        pthistsig, bin_edges = np.histogram(events["fj_pt"][mask_proc_sig].to_numpy(), bins=ptbins)
        pthist = pthistsigmh/pthistsig

        # plot weight histogram
        weight_sig = pthist[np.digitize(events['fj_pt'][mask_proc_sig].to_numpy(), ptbins)-1]
        fig, axs = plt.subplots(1,1)
        axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,histtype='step')
        axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,weights=weight_sig,histtype='step')
        axs.set_xlabel('pT (GeV)')
        axs.legend(['Unweighted','Weighted'])
        fig.savefig('%s/ptweights_%s.pdf'%(args.odir,label_dict[signal]['label']))
        fig, axs = plt.subplots(1,1)
        axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,histtype='step',density=True)
        axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,weights=weight_sig,histtype='step',density=True)
        axs.set_xlabel('pT (GeV)')
        axs.legend(['Unweighted','Weighted'])
        fig.savefig('%s/ptweights_%s_density.pdf'%(args.odir,label_dict[signal]['label']))
                
        # masks
        mask_flat = (events[mass_branch] != 125)
        mask_mh125 = (events[mass_branch] == 125)
        mask_proc_mh120130 = (events[mass_branch]>=120) & (events[mass_branch]<=130) & (events["fj_pt"]<=600) & mask_flat

        # get ROC
        fprs = {}
        tprs = {}
        fprs[score_name], tprs[score_name] = get_roc(events, score_name, siglabel, bkglabel)
        
        # get ROC for flat sample with pt weights
        fprs[score_name+'flat_weight'], tprs[score_name+'flat_weight'] = get_roc(events, score_name, siglabel, bkglabel, weight_hist=pthist, bins=ptbins,sig_mask=mask_flat)
        fprs[score_name+'flat'], tprs[score_name+'flat'] = get_roc(events, score_name, siglabel, bkglabel, weight_hist=None, bins=None,sig_mask=mask_flat)

        # get ROC for flat sample with mass around mass of higgs
        fprs[score_name+'flat_mh120130-pt200600'], tprs[score_name+'flat_mh120130-pt200600'] = get_roc(events, score_name, siglabel, bkglabel, weight_hist=None, bins=None,sig_mask=mask_proc_mh120130)
        fprs[score_name+'flat_mh120130-pt200600-weight'], tprs[score_name+'flat_mh120130-pt200600-weight'] = get_roc(events, score_name, siglabel, bkglabel, weight_hist=pthist, bins=ptbins,sig_mask=mask_proc_mh120130)

        # get ROC for Particle Net if 3q/4q
        if signal=="hww_4q" or signal=="hww_4q_merged" or signal=="hww_3q_merged":
            #fprs['DeepAK8_H4q_MD_flat'], tprs['DeepAK8_H4q_MD_flat'] = get_roc(events, "fj_deepTagMD_H4qvsQCD", siglabel, bkglabel,sig_mask=mask_flat)
            #fprs['DeepAK8_H_flat'], tprs['DeepAK8_H_flat'] = get_roc(events, "fj_deepTag_HvsQCD", siglabel, bkglabel,sig_mask=mask_flat)
            fprs['PN_H4q_flat'], tprs['PN_H4q_flat'] = get_roc(events, "fj_PN_H4qvsQCD", siglabel, bkglabel,sig_mask=mask_flat)
            
            # get ROCs for score but for 3q/4q independently
            if add_nProngs:
                mask_proc_3q = (events["fj_nProngs"] == 3) & mask_flat
                mask_proc_4q = (events["fj_nProngs"] == 4) & mask_flat
                fprs[score_name+'_flat_3q'], tprs[score_name+'_flat_3q'] = get_roc(events, score_name, siglabel, bkglabel, weight_hist=None, bins=None,sig_mask=mask_proc_3q)
                fprs[score_name+'_flat_4q'], tprs[score_name+'_flat_4q'] = get_roc(events, score_name, siglabel, bkglabel, weight_hist=None, bins=None,sig_mask=mask_proc_4q)
             
        # plot ROCs
        plot_roc(args, label_dict[signal], label_dict[bkg], fprs, tprs, label=label_dict[signal]['label'])
        
        # plot features for this signal and background combination (i.e. all the processes) 
        vars_to_plot = ["pt","msd","score"]
        plt_label = "validation_%svs%s"%(siglabel,bkglabel)
        plot_validation(args,hist_features,vars_to_plot,plt_label)

        # plot how the score looks after cuts on variables
        vars_to_corr = ["mH","pt"]
        bin_ranges = [list(range(60,240,20)),list(range(200,1200,200))]
        bin_widths = [10,200]
        proc_to_corr = ['%s-mhflat'%signal]
        plt_label = "%svs%s"%(siglabel,bkglabel)
        # plot_score_aftercut(args,hist_gmass,vars_to_corr,bin_ranges,bin_widths,proc_to_corr,plt_label)

        # plot roc for different cuts on mH and pt
        vars_to_corr = {"mH":mass_branch,
                        "pt":"fj_pt"}
        plot_roc_by_var(args,vars_to_corr,bin_ranges,bin_widths,events,score_name,label_dict[signal],label_dict[bkg])

        # accumulate rocs for summary
        fprs_summary = {}; tprs_summary = {}
        fprs_summary['flat'] = fprs[score_name+'flat']; tprs_summary['flat'] = tprs[score_name+'flat']
        fprs_summary['flat-mhptSM'] = fprs[score_name+'flat_mh120130-pt200600-weight']; tprs_summary['flat-mhptSM'] = tprs[score_name+'flat_mh120130-pt200600-weight']
        if signal=="hww_4q" or signal=="hww_4q_merged" or signal=="hww_3q_merged":
            fprs_summary['flat-PN4q'] = fprs['PN_H4q_flat']; tprs_summary['flat-PN4q'] = tprs['PN_H4q_flat']
        
        # plot roc for mh=125
        fprs = {} # reset fprs
        mask_mh125_100150 = mask_mh125 & (events["fj_msoftdrop"] >= 100) & (events["fj_msoftdrop"] <=150)
        mask_100150 = (events["fj_msoftdrop"] >= 100) & (events["fj_msoftdrop"] <=150)
        fprs[score_name+'_mh125'], tprs[score_name+'_mh125'] = get_roc(events,  score_name, siglabel, bkglabel, weight_hist=None, bins=None, sig_mask=mask_mh125)
        # fprs[score_name+'_mh125_msd100-150'], tprs[score_name+'_mh125_msd100-150'] = get_roc(events,  score_name, siglabel, bkglabel, weight_hist=None, bins=None, sig_mask=mask_mh125_100150, bkg_mask=mask_100150)
        if signal=="hww_4q" or signal=="hww_4q_merged" or signal=="hww_3q_merged":
            fprs['PN_H4q_mh125-nonMD'], tprs['PN_H4q_mh125-nonMD'] = get_roc(events, "fj_PN_H4qvsQCD", siglabel, bkglabel, sig_mask=mask_mh125)
            fprs['PN_H4q_mh125_msd100-150'], tprs['PN_H4q_mh125_msd100-150'] = get_roc(events, "fj_PN_H4qvsQCD", siglabel, bkglabel, sig_mask=mask_mh125_100150, bkg_mask=mask_100150)
            if add_nProngs:
                mask_proc_3q = (events["fj_nProngs"] == 3) & mask_mh125
                mask_proc_4q = (events["fj_nProngs"] == 4) & mask_mh125
                fprs[score_name+'_mh125_4q'], tprs[score_name+'_mh125_4q'] = get_roc(events,  score_name, siglabel, bkglabel, weight_hist=None, bins=None, sig_mask=mask_proc_4q)
                fprs[score_name+'_mh125_3q'], tprs[score_name+'_mh125_3q'] = get_roc(events,  score_name, siglabel, bkglabel, weight_hist=None, bins=None, sig_mask=mask_proc_3q)
                fprs['PN_H4q_mh125-nonMD_4q'], tprs['PN_H4q_mh125-nonMD_4q'] = get_roc(events, "fj_PN_H4qvsQCD", siglabel, bkglabel, sig_mask=mask_proc_4q)
                fprs['PN_H4q_mh125-nonMD_3q'], tprs['PN_H4q_mh125-nonMD_3q'] = get_roc(events, "fj_PN_H4qvsQCD", siglabel, bkglabel, sig_mask=mask_proc_3q)                                
        plot_roc(args, label_dict[signal], label_dict[bkg], fprs, tprs, label=label_dict[signal]['label']+'_mh125_all')

        # more rocs for summary
        fprs_summary['SM'] = fprs[score_name+'_mh125']; tprs_summary['SM'] = tprs[score_name+'_mh125']
        if signal=="hww_4q" or signal=="hww_4q_merged" or signal=="hww_3q_merged":
            fprs_summary['SM-PN4q-msd100-150'] = fprs['PN_H4q_mh125_msd100-150']; tprs_summary['SM-PN4q-msd100-150'] = tprs['PN_H4q_mh125_msd100-150']

        # plot roc summary
        plot_roc(args, label_dict[signal], label_dict[bkg], fprs_summary, tprs_summary, label=label_dict[signal]['label']+'_summary')
        
        # xcheck plot for mh=125
        vars_to_corr = {"mH":mass_branch,
                        "pt":"fj_pt"}
        bin_ranges = [[125,125],list(range(200,1200,200))]
        bin_widths = [4,200]
        plot_roc_by_var(args,vars_to_corr,bin_ranges,bin_widths,events,score_name,label_dict[signal],label_dict[bkg],True)
        
        # plot how variables look after cut on classifier (tagger score)
        vars_to_corr = ["msd"]
        proc_to_corr = [bkg]
        plt_label = "aftercuts_%svs%s"%(siglabel,bkglabel)
        # first compute percentiles on bkg (or maybe other process?)
        percentiles, cuts = computePercentiles(events[score_name][(events[bkglabel]==1)].to_numpy(), [0.97, 0.99, 0.995])
        plot_var_aftercut(args,hist_features,vars_to_corr,proc_to_corr,plt_label,cuts,percentiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifile', help='input file(s)')
    parser.add_argument('--odir', required=True, help="output dir")
    parser.add_argument('--name', required=True, help='name of the model(s)')
    parser.add_argument('--signals', default='hww_4q_merged', help='signals')
    parser.add_argument('--bkgs', default='qcd', help='backgrounds (if qcd_label then assume that you only have one qcd label)') 
    args = parser.parse_args()

    import os
    os.system('mkdir -p %s'%args.odir)
    
    main(args)
