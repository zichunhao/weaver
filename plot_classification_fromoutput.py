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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from labels import label_dict
from plot_utils import *

def fill_hists(args,events,bkg,signal,score):
    pt = "fj_pt"
    eta = "fj_eta"
    if args.dnn:
        msd = "fj_sdmass"
        oldpn = "pfParticleNetDiscriminatorsJetTags_H4qvsQCD"
    else:
        msd = "fj_msoftdrop"
        oldpn = "fj_PN_H4qvsQCD"

    mass = None
    if args.mbranch:
        mass = args.mbranch

    hist_features = hist.Hist("Jets",
                              hist.Cat("process", "Process"),
                              hist.Bin("msd", r"fj msoftdrop [GeV]", 40, 30, 260),
                              hist.Bin("pt", r"fj $p_T$ [GeV]", 50, 200, 1200), # bins of 20
                              hist.Bin("score", r"Tagger score", 100, 0, 1),
                              #hist.Bin("pn_score", r"Old Tagger score", 100, 0, 1),
                              )
    hist_gmass = None
    if args.mbranch:
        hist_gmass = hist.Hist("genmass",
                               hist.Cat("process", "Process"),
                               hist.Bin("score", r"Tagger score", 70, 0, 1),
                               hist.Bin("pt", r"fj $p_T$ [GeV]", 50, 200, 1200), # bins of 20
                               hist.Bin("mH", r"gen Res mass [GeV]", 42, 50, 260), # bins of 5
                               )

    processes = [bkg]
    if args.mbranch:
        processes.append(f'{signal}-mh125')
        processes.append(f'{signal}-mhflat')
    else:
        processes.append(signal)
    legends = {}
    for proc in processes:
        p = proc.split('-')[0]
        legend = label_dict[p]['legend']
        if 'mh125' in proc:
            mask_proc = (events[label_dict[p]['label']]==1) & (events[mass]==125)
            legend += ' mh125'
        elif 'mhflat' in proc:
            mask_proc = (events[label_dict[p]['label']]==1) & (events[mass]!=125)
            legend += ' mhflat'
        else:
            mask_proc = (events[label_dict[p]['label']]==1)
            legends[proc] = legend

        # check if events with that mask are not zero
        if len(events[msd][mask_proc])==0:
            processes.remove(proc)
            continue

        # fill histogram
        hist_features.fill(process=proc,
                           msd = events[msd][mask_proc],
                           pt = events[pt][mask_proc],
                           score = events[score][mask_proc],
                           #pn_score = events[oldpn][mask_proc],
                           )
        if signal in proc and hist_gmass:
            hist_gmass.fill(process=proc,
                            mH = events[mass][mask_proc],
                            score = events[score][mask_proc],
                            pt = events[pt][mask_proc],
                            )

    return hist_features,hist_gmass

def get_score(score,bkg,events,siglabel,bkglabel):
    """
    We expect all scores to sum up to 1, e.g. given two signals in the event (signal 1 and 2) and one background process (background 1)
      score_signal_1 + score_signal_2 + score_background_1 = 1
    Then nn_signal_1 = score_signal_1 / (score_signal_1 + score_background_1) = score_signal_1 / (1 - score_signal_2)
    """
    if bkg=='qcd':
        events[score] = events[f'score_{siglabel}'] / (events[f'score_{siglabel}'] + events['score_fj_isQCDb'] + events['score_fj_isQCDbb'] + events['score_fj_isQCDc'] + events['score_fj_isQCDcc'] + \
                                                       events['score_fj_isQCDlep'] + events['score_fj_isQCDothers'])
    elif bkg=='qcd_dnn':
        events[score] = events[f'score_{siglabel}'] / (events[f'score_{siglabel}'] + events['score_label_QCD_b'] + events['score_label_QCD_bb'] + events['score_label_QCD_c'] + \
                                                       events[f'score_label_QCD_cc'] + events['score_label_QCD_others'])
    else:
        events[score] = events[f'score_{siglabel}'] / (events[f'score_{siglabel}'] + events[f'score_{bkglabel}'])


def get_weights_mh125(args,events,signal,mask_flat,mask_mh125):
    """
    Get pt weights between 125 and !=125 higgs samples
    """
    mask_proc_sigmh125 = (events[label_dict[signal]['label']]==1) & mask_mh125
    mask_proc_sig = (events[label_dict[signal]['label']]==1) & mask_flat

    # get pt bins
    # log bins as: np.round(np.exp(np.linspace(np.log(MIN), np.log(MAX), NUM_BINS))).astype('int').tolist()
    ptbins = [200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500]
    pthistsigmh, bin_edges = np.histogram(events["fj_pt"][mask_proc_sigmh125].to_numpy(), bins=ptbins)
    pthistsig, bin_edges = np.histogram(events["fj_pt"][mask_proc_sig].to_numpy(), bins=ptbins)
    pthist = pthistsigmh/pthistsig

    # now plot
    weight_sig = pthist[np.digitize(events['fj_pt'][mask_proc_sig].to_numpy(), ptbins)-1]
    fig, axs = plt.subplots(1,1)
    axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,histtype='step')
    axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,weights=weight_sig,histtype='step')
    axs.set_xlabel('pT (GeV)')
    axs.legend(['Unweighted','Weighted'])
    fig.savefig(f'%s/ptweights_%s.pdf'%(args.odir,label_dict[signal]['label']))
    fig.savefig(f'%s/ptweights_%s.png'%(args.odir,label_dict[signal]['label']))

    fig, axs = plt.subplots(1,1)
    axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,histtype='step',density=True)
    axs.hist(events["fj_pt"][mask_proc_sig],bins=ptbins,weights=weight_sig,histtype='step',density=True)
    axs.set_xlabel('pT (GeV)')
    axs.legend(['Unweighted','Weighted'])
    fig.savefig(f'%s/ptweights_%s_density.pdf'%(args.odir,label_dict[signal]['label']))
    fig.savefig(f'%s/ptweights_%s_density.png'%(args.odir,label_dict[signal]['label']))

    return pthist

def get_branches(args,bkg,siglabel,bkglabel):
    pt = "fj_pt"
    eta = "fj_eta"
    if args.dnn:
        msd = "fj_sdmass"
        oldpn = "pfParticleNetDiscriminatorsJetTags_H4qvsQCD"
    else:
        msd = "fj_msoftdrop"
        oldpn = "fj_PN_H4qvsQCD"

    branches = [pt,msd,eta] #,oldpn]

    # mask
    mask = f"({pt}>200) & ({pt}<1200)"

    # mass branch
    mass = None
    if args.mbranch:
        mass = args.mbranch
        branches.append(mass)

    if args.nprongs:
        branches.append("fj_nProngs")

    # score branch for signal
    branches.append(siglabel)
    branches.append(f'score_{siglabel}')

    # score branch for background
    if bkg=='qcd':
        branches.extend(["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"])
        branches.extend(['score_fj_isQCDb','score_fj_isQCDbb','score_fj_isQCDc','score_fj_isQCDcc','score_fj_isQCDlep','score_fj_isQCDothers'])
        if args.mbranch:
            mask += f"& ( (((fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1)) & ({mass}<0)) | "\
                f"(({siglabel}==1) & ({mass}>0)) )"
    elif bkg=='qcd_dnn':
        branches.extend(["label_QCD_b","label_QCD_bb","label_QCD_c","label_QCD_cc","label_QCD_others"])
        branches.extend(['score_label_QCD_b','score_label_QCD_bb','score_label_QCD_c','score_label_QCD_cc','score_label_QCD_others'])
    else:
        branches.append(bkglabel)
        branches.extend([f'score_{bkglabel}'])
        mask += f"& ( ({bkglabel}==1) | ({siglabel}==1) )"

    return branches,mask

def main(args):
    # get signals and backgrounds processes
    signals = args.signals.split(',')
    backgrounds = args.bkgs.split(',')
    if len(signals)!=len(backgrounds):
        print('Number of signals should be the same as backgrounds!')
        exit

    pt = "fj_pt"
    eta = "fj_eta"
    if args.dnn:
        msd = "fj_sdmass"
        oldpn = "pfParticleNetDiscriminatorsJetTags_H4qvsQCD"
    else:
        msd = "fj_msoftdrop"
        oldpn = "fj_PN_H4qvsQCD"

    # for each (signal,bkg) process build a roc curve
    fp_tp_all = {}
    for i,sig in enumerate(signals):
        odir = args.odir + sig
        os.system(f'mkdir -p {odir}')
        bkg = backgrounds[i]
        bkglabel = label_dict[bkg]['label']
        siglabel = label_dict[sig]['label']
        bkglegend = label_dict[bkg]['legend']
        siglegend = label_dict[sig]['legend']

        # get branches to read
        branches,mask = get_branches(args,bkg,siglabel,bkglabel)
        # print('List of branches read ',branches)

        # open file
        ifile = uproot.open(args.ifile)["Events"]
        events = ifile.arrays(branches,mask)
        if bkg=='qcd':
            events["fj_QCD_label"] = events["fj_isQCDb"] | events["fj_isQCDbb"] | events["fj_isQCDc"] | events["fj_isQCDcc"] | events["fj_isQCDlep"] | events["fj_isQCDothers"]
        if bkg=='qcd_dnn':
            events["fj_QCD_label"] = events["label_QCD_b"] | events["label_QCD_bb"] | events["label_QCD_c"] | events["label_QCD_cc"] | events["label_QCD_others"]

        # compute scores
        score = f"{args.name}_score"
        get_score(score,bkg,events,siglabel,bkglabel)

        # get ROCs
        # by default apply pT and msd selection
        ptcut = r'%s $p_T$:[400-600] GeV, $|\eta|<2.4$'%args.jet
        msdcut = r'%s $m_{SD}$:[90-140] GeV'%args.jet
        roc_mask = (events[pt] > 400) & (events[pt] < 600) & (events[msd] > 90) & (events[msd] < 140) & (abs(events["fj_eta"])<2.4)
        roc_mask_num = roc_mask
        roc_mask_den = (events[pt] > 400) & (events[pt] < 600) & (abs(events["fj_eta"])<2.4)

        fp_tp = {}
        fp_tp[score]     = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask)
        #fp_tp['PNnonMD'] = get_roc(events, oldpn, siglabel, bkglabel, roc_mask=roc_mask)
        #fp_tp['PNnonMD-ratio'] = get_roc(events, oldpn, siglabel, bkglabel, ratio=True, mask_num=roc_mask_num, mask_den=roc_mask_den)
        #fp_tp['PNnonMD-nomass'] = get_roc(events, oldpn, siglabel, bkglabel, roc_mask=roc_mask_den)

        #np.savetxt("../xcheckdawei/precutmass_ROC_ParticleNetV01_official-Cristina.csv", np.vstack((fp_tp['PNnonMD-nomass'][1], fp_tp['PNnonMD-nomass'][0])).T,delimiter=',')
        #np.savetxt("../xcheckdawei/postcutmass_ROC_ParticleNetV01_official-Cristina.csv", np.vstack((fp_tp['PNnonMD-ratio'][1], fp_tp['PNnonMD-ratio'][0])).T,delimiter=',')

        if args.mbranch:
            mask_flat = (events[args.mbranch] != 125)
            mask_mh125 = (events[args.mbranch] == 125)
            mask_proc_mh120130 = (events[args.mbranch]>=120) & (events[args.mbranch]<=130) & mask_flat

            # get pt weights between flat sample and mh125 sample
            pthist = get_weights_mh125(args,events,sig,mask_flat,mask_mh125)

            # for flat sample
            fp_tp[r'$m_H$:[50-230]']               = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=mask_flat)
            ptbins = [200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500]
            fp_tp[r'$m_H$:[50-230]\times p_T^SM']  = get_roc(events, score, siglabel, bkglabel, weight_hist=pthist, bins=ptbins, roc_mask=roc_mask, sig_mask=mask_flat)
            fp_tp[r'$m_H$:[120,130]\times p_T^SM'] = get_roc(events, score, siglabel, bkglabel, weight_hist=pthist, bins=ptbins,roc_mask=roc_mask, sig_mask=mask_proc_mh120130)
            #fp_tp[r'$m_H$:[50-230] PNnonMD']       = get_roc(events, oldpn, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=mask_flat)

            # for mh125
            fp_tp[r'$m_H$:125']         = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=mask_mh125)
            #fp_tp[r'$m_H$:125 PNnonMD'] = get_roc(events, oldpn, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=mask_mh125)

            if args.nprongs:
                fp_tp[r'$m_H$:[50-230] 3prongs']  = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=(mask_flat & (events["fj_nProngs"] == 3)) )
                fp_tp[r'$m_H$:[50-230] 4prongs']  = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=(mask_flat & (events["fj_nProngs"] == 4)) )
                fp_tp[r'$m_H$:125 3prongs'] = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=(mask_mh125 & (events["fj_nProngs"] == 3)) )
                fp_tp[r'$m_H$:125 4prongs'] = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=(mask_mh125 & (events["fj_nProngs"] == 4)) )
        else:
            if args.nprongs:
                fp_tp['3prongs'] = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=(events["fj_nProngs"] == 3))
                fp_tp['4prongs'] = get_roc(events, score, siglabel, bkglabel, roc_mask=roc_mask, sig_mask=(events["fj_nProngs"] == 4))

        # plot ROC
        title = f"ROC Curve of {siglegend} vs {bkglegend}"
        label = f"{siglabel}vs{bkglabel}"
        plot_roc(odir, siglegend, bkglegend, fp_tp, label=label+"_pn", title=title, pkeys=[score],ptcut=ptcut, msdcut=msdcut)
        plot_roc(odir, siglegend, bkglegend, fp_tp, label=label+"_pnnonmd", title=title, pkeys=[score,'PNnonMD'],ptcut=ptcut, msdcut=msdcut)
        plot_roc(odir, siglegend, bkglegend, fp_tp, label=label+"_summary", title=title,
                 pkeys=[score,r'$m_H$:125',r'$m_H$:[50-230]'],
                 ptcut=ptcut, msdcut=msdcut)

        # plot ROC for different cuts on mH and pT
        vars_to_corr = {r"$p_T$": "fj_pt"}
        bin_ranges = [list(range(200, 1200, 200))]
        bin_widths = [200]*len(bin_ranges)
        if args.mbranch:
            vars_to_corr[r"$m_H$"] = args.mbranch
            bin_ranges += [list(range(60, 240, 20))]
            bin_widths = [10]*len(bin_ranges)
            #plot_roc_by_var(odir, vars_to_corr, bin_ranges, bin_widths, events, score, label_dict[sig], label_dict[bkg], label+"_mH125", title=r"%s $m_H=125$ GeV"%title,sig_mask=mask_mh125)
        plot_roc_by_var(odir, vars_to_corr, bin_ranges, bin_widths, events, score, label_dict[sig], label_dict[bkg], label+"_allmH", title=title)

        # fill histograms
        hist_features,hist_gmass = fill_hists(args,events,bkg,sig,score)

        # plot features
        #plot_1d(odir,hist_features,["pt","msd","score"],label+"_validation")

        # compute percentiles on bkg
        percentiles, cuts = computePercentiles(events[score][(events[bkglabel]==1)].to_numpy(), [0.97, 0.99, 0.995])
        # plot how variables look after cut on tagger score
        plot_var_aftercut(odir,hist_features,["msd"],[bkg],label+"_msdsculpt",cuts,percentiles)

        fp_tp_all[sig] = fp_tp[score]

    # plot summary ROC
    plot_roc(args.odir, "HWW", "QCD", fp_tp_all, label="allsig_summary", title=f"ROC Curve of HWW vs {bkglegend}", ptcut=ptcut, msdcut=msdcut)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifile', help='input file(s)')
    parser.add_argument('--odir', required=True, help="output dir")
    parser.add_argument('--name', required=True, help='name of the model(s)')
    parser.add_argument('--mbranch', default=None, help='mass branch name')
    parser.add_argument('--jet', default="AK15", help='jet type')
    parser.add_argument('--dnn', action='store_true', default=False, help='is dnn tuple?')
    parser.add_argument('--nprongs', action='store_true', default=False, help='is nprongs branch saved?')
    parser.add_argument('--signals', default='hww_4q_merged', help='signals')
    parser.add_argument('--bkgs', default='qcd', help='backgrounds (if qcd_label then assume that you only have one qcd label)')
    args = parser.parse_args()

    import os
    os.system('mkdir -p %s'%args.odir)

    main(args)
