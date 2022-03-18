#!/usr/bin/env python

import os
import numpy as np
import uproot
import hist as hist2
import awkward as ak

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from labels import label_dict
from plot_utils import *

class PlotOutput:
    def __init__(self,
                 ifile,
                 name,
                 odir,
                 jet,
                 sig,
                 bkg,
                 isigfile = None,
                 dnn = False,
                 verbose = False,
                 oldpn = False,
                 mbranch = None,
                 nprongs = None,
                 ):
        self.ifile = ifile
        self.dnn = dnn
        self.odir = odir
        self.score = f"{name}_score"
        self.verbose = verbose
        
        self.pt = "fj_pt"
        self.eta = "fj_eta"
        self.msd = "fj_msoftdrop" if not self.dnn else "fj_sdmass"
        self.mbranch = mbranch
        self.nprongs = "fj_nProngs" if nprongs else None
        if oldpn:
            self.oldpn = "pfParticleNetDiscriminatorsJetTags_H4qvsQCD" if self.dnn else "fj_PN_H4qvsQCD"
        else:
            self.oldpn = None

        self.ptrange = [400,600]
        self.msdrange = [60,150]
        self.jet = jet

        self.sigfile = isigfile
        
        self.sig = sig
        self.bkg = bkg
        self.bkglabel = label_dict[bkg]["label"]
        self.siglabel = label_dict[sig]["label"]
        self.bkglegend = label_dict[bkg]["legend"]
        self.siglegend = label_dict[sig]["legend"]

        self.events = self.get_events()
        self.get_masks()
        self.hists = self.fill_hists()

    def get_events(self):
        branches = [self.pt, self.msd, self.eta]
        if self.oldpn:
            branches.append(self.oldpn)
        if self.mbranch:
            branches.append(self.mbranch)
        if self.nprongs:
            branches.append(self.nprongs)
            
        # score for signal
        branches.append(self.siglabel)
        branches.append(f"score_{self.siglabel}")
        
        # score for background
        if self.bkg == "qcd":
            qcdlabels = ["fj_isQCDb", "fj_isQCDbb", "fj_isQCDc", "fj_isQCDcc", "fj_isQCDlep", "fj_isQCDothers"]
            branches.extend(qcdlabels)
            branches.extend([f"score_{qcdlabel}" for qcdlabel in qcdlabels])
        elif self.bkg == "qcd_dnn":        
            qcdlabels = ["label_QCD_b", "label_QCD_bb", "label_QCD_c", "label_QCD_cc", "label_QCD_others"]
            branches.extend(qcdlabels)
            branches.extend([f"score_{qcdlabel}" for qcdlabel in qcdlabels])
        else:
            branches.append(self.bkglabel)
            branches.extend([f"score_{self.bkglabel}"])
            
        if self.verbose:
            print('List of branches read ',branches)
            
        # pt mask
        mask = f"({self.pt}>200) & ({self.pt}<1500)"
        if self.mbranch:
            mask += (
                f"& ( (((fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1)) & ({self.mbranch}<=0)) | "
                f"(({self.siglabel}==1) & ({self.mbranch}>0)) )"
            )
            
        # open file
        if self.sigfile:
            events = uproot.concatenate({self.ifile:"Events",self.sigfile:"Events"},branches,cut=mask)
        else:
            events = uproot.open(self.ifile)["Events"].arrays(branches, mask)

        # make sure that the are flat samples
        if ~ak.any((events[self.siglabel]==1) & (events[self.mbranch]>0) & (events[self.mbranch]!=125)):
            print('No mbranch')
            self.mbranch = None

        # bkg label
        if self.bkg == "qcd":
            events["fj_QCD_label"] = (
                events["fj_isQCDb"]
                | events["fj_isQCDbb"]
                | events["fj_isQCDc"]
                | events["fj_isQCDcc"]
		| events["fj_isQCDlep"]
                | events["fj_isQCDothers"]
            )
        elif self.bkg == "qcd_dnn":
            events["fj_QCD_label"] = (
                events["label_QCD_b"]
                | events["label_QCD_bb"]
		| events["label_QCD_c"]
                | events["label_QCD_cc"]
                | events["label_QCD_others"]
            )
        if self.verbose:
            print(f'Bkg any {self.bkglabel}: ',ak.any(events[self.bkglabel]==1))

        # get score
        """
        We expect all scores to sum up to 1, e.g. given two signals in the event (signal 1 and 2) and one background process (background 1)
        score_signal_1 + score_signal_2 + score_background_1 = 1
        Then nn_signal_1 = score_signal_1 / (score_signal_1 + score_background_1) = score_signal_1 / (1 - score_signal_2)
        """
        if self.bkg == "qcd":
            print(f'Score defined as score_{self.siglabel}/(score_{self.siglabel}+score_QCDs*)')
            score_branch = events[f"score_{self.siglabel}"] / (
                events[f"score_{self.siglabel}"]
                + events["score_fj_isQCDb"]
                + events["score_fj_isQCDbb"]
                + events["score_fj_isQCDc"]
                + events["score_fj_isQCDcc"]
                + events["score_fj_isQCDlep"]
                + events["score_fj_isQCDothers"]
            )
        elif self.bkg == "qcd_dnn":
            print(f'Score defined as score_{self.siglabel}/(score_{self.siglabel}+score_QCDs* - nolep)')
            score_branch = events[f"score_{self.siglabel}"] / (
                events[f"score_{self.siglabel}"]
                + events["score_label_QCD_b"]
                + events["score_label_QCD_bb"]
                + events["score_label_QCD_c"]
                + events[f"score_label_QCD_cc"]
                + events["score_label_QCD_others"]
            )
        else:
            print(f'Score defined as score_{self.siglabel}/(score_{self.siglabel}+score_{self.bkglabel}')
            score_branch = events[f"score_{self.siglabel}"] / (
                events[f"score_{self.siglabel}"] + events[f"score_{self.bkglabel}"]
            )
        events[self.score] = score_branch
        if self.verbose:
            print(self.score,events[self.score])
            
        return events

    def get_masks(self):
        self.roc_mask_nomass = (
            (self.events[self.pt] > self.ptrange[0])
            & (self.events[self.pt] < self.ptrange[1])
            & (abs(self.events[self.eta]) < 2.4)
        )
        self.roc_mask = self.roc_mask_nomass & (self.events[self.msd] > self.msdrange[0]) & (self.events[self.msd] < self.msdrange[1])
        if self.mbranch:
            self.mask_flat = self.events[self.mbranch] != 125
            self.mask_mh125 = self.events[self.mbranch] == 125
            self.mask_proc_mh120130 = (
                (self.events[self.mbranch] >= 120) & (self.events[self.mbranch] <= 130) & self.mask_flat
            )
            self.mask_proc_sigmh125 = (self.events[self.siglabel]==1) & self.mask_mh125
            self.mask_proc_sig = (self.events[self.siglabel]==1) & self.mask_flat
        else:
            print('masks for no mbranch')
            self.mask_flat =  np.zeros(len(self.events), dtype='bool')
            self.mask_mh125 = np.zeros(len(self.events), dtype='bool')
            self.mask_proc_mh120130 = np.zeros(len(self.events), dtype='bool')
            self.mask_proc_sigmh125 = np.zeros(len(self.events), dtype='bool')
            self.mask_proc_sig = np.zeros(len(self.events), dtype='bool')
            
        if self.nprongs:
            self.mask3p = self.events[self.nprongs] == 3
            self.mask4p = self.events[self.nprongs] == 4
            
    def get_weights(self):
        # log bins as: np.round(np.exp(np.linspace(np.log(MIN), np.log(MAX), NUM_BINS))).astype('int').tolist()
        ptbins = [200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500]
        pthistsigmh, bin_edges = np.histogram(
            self.events[self.pt][self.mask_proc_sigmh125].to_numpy(), bins=ptbins
        )
        ptflat = self.events[self.pt][self.mask_proc_sig]
        pthistsig, bin_edges = np.histogram(ptflat.to_numpy(), bins=ptbins)
        pthist = pthistsigmh / pthistsig
        weight_sig = pthist[np.digitize(self.events[self.pt][self.mask_proc_sig].to_numpy(), ptbins) - 1]
        
        # plots
        for density in [True]: #False]:
            fig, axs = plt.subplots(1, 1)
            axs.hist(ptflat, bins=ptbins, histtype="step", density=density)
            axs.hist(ptflat, bins=ptbins, weights=weight_sig, histtype="step",density=density)
            axs.set_xlabel("pT (GeV)")
            axs.legend(["Unweighted", "Weighted"])
            tag = self.siglabel + "_density" if density else self.siglabel
            fig.savefig(f"%s/ptweights_%s.pdf" % (self.odir, tag))
            fig.savefig(f"%s/ptweights_%s.png" % (self.odir, tag))
        
        return pthist,ptbins,weight_sig
    
    def roc(self, score, option=None, sigmask=None):
        if option=="ratio":
            roc = get_roc(self.events, score, self.siglabel, self.bkglabel, ratio=True, mask_num=self.roc_mask, mask_den=self.roc_mask_nomass)
        elif option=="nomass":
            roc = get_roc(self.events, score, self.siglabel, self.bkglabel, roc_mask=self.roc_mask_nomass)
        elif option=="sigmask":
            roc = get_roc(self.events, score, self.siglabel, self.bkglabel, roc_mask=self.roc_mask, sig_mask=sigmask)
        elif option=="ptweight":
            if self.mbranch and ak.any(self.mask_proc_sigmh125):
                pthist,ptbins,weight_sig = self.get_weights()
                roc = get_roc(self.events, score, self.siglabel, self.bkglabel, weight_hist=pthist, bins=ptbins, roc_mask=self.roc_mask, sig_mask=sigmask)
        else:
            roc = get_roc(self.events, score, self.siglabel, self.bkglabel, roc_mask=self.roc_mask)
        return roc

    def plot(self, fp_tp, tag, keys):
        title = r"%s vs %s"%(self.siglegend,self.bkglegend)
        label = f"{self.siglabel}vs{self.bkglabel}"
        ptcut = r"%s $p_T$:[%s-%s] GeV, $|\eta|<2.4$"%(self.jet,self.ptrange[0],self.ptrange[1])
        msdcut = r"%s $m_{SD}$:[%s-%s] GeV" %(self.jet,self.msdrange[0],self.msdrange[1])
        plot_roc(self.odir,self.siglegend,self.bkglegend,fp_tp,
                 label=label+"_"+tag,
                 title=title,
                 pkeys=keys,
                 ptcut=ptcut,
                 msdcut=msdcut,
                 )

    def fill_hists(self):
        hists = {}
        hists["features"] = hist2.Hist(
            hist2.axis.StrCategory([], name='process', growth=True),
            hist2.axis.Regular(40, 30, 260, name='msd', label=r'fj msoftdrop [GeV]'),
            hist2.axis.Regular(50, 200, 1200, name='pt', label=r'fj $p_T$ [GeV]'),
            hist2.axis.Regular(100, 0, 1, name='score', label=r'Tagger score'),
        )
        if self.mbranch:
            hists["genmass"] = hist2.Hist(
                hist2.axis.StrCategory([], name='process', growth=True),
                hist2.axis.Regular(100, 0, 1, name='score', label=r'Tagger score'),
                hist2.axis.Regular(50, 200, 1200, name='pt', label=r'fj $p_T$ [GeV]'),
                hist2.axis.Regular(46, 30, 260, name='mh', label=r'gen Res mass [GeV]'),
            )
            
        processes = [self.bkg]
        if self.mbranch:
            processes.append(f"{self.sig}-mh125")
            processes.append(f"{self.sig}-mhflat")
        else:
            processes.append(self.sig)
            
        legends = {}
        for proc in processes:
            p = proc.split("-")[0]
            legend = label_dict[p]["legend"]
            if "mh125" in proc:
                mask_proc = (self.events[label_dict[p]["label"]] == 1) & (self.events[self.mbranch] == 125)
                legend += " mh125"
            elif "mhflat" in proc:
                mask_proc = (self.events[label_dict[p]["label"]] == 1) & (self.events[self.mbranch] != 125)
                legend += " mhflat"
            else:
                mask_proc = self.events[label_dict[p]["label"]] == 1
                legends[proc] = legend
                
            # check if events with that mask are not zero
            if len(self.events[self.msd][mask_proc]) == 0:
                processes.remove(proc)
                continue
        
            # fill histogram
            hists["features"].fill(
                process=proc,
                msd=self.events[self.msd][mask_proc],
                pt=self.events[self.pt][mask_proc],
                score=self.events[self.score][mask_proc],
            )
            if self.sig in proc and "genmass" in hists.keys():
                hists["genmass"].fill(
                    process=proc,
                    mh=self.events[self.mbranch][mask_proc],
                    score=self.events[self.score][mask_proc],
                    pt=self.events[self.pt][mask_proc],
            )
        return hists
