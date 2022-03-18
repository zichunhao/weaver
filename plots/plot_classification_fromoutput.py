#!/usr/bin/env python

import os
import argparse
import numpy as np
import uproot
import hist
import awkward as ak

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from PlotOutput import PlotOutput

from labels import label_dict
from plot_utils import *

def get_rocs(p,args,jsig=0,signame=""):
    fp_tp = {}
    score = p.score
    fp_tp[score+signame] = p.roc(score)
    fp_tp[score+signame+"-ratio"] = p.roc(score,"ratio")
    if args.oldpn and jsig==0:
        fp_tp['PNnonMD'] = p.roc(args.oldpn)
        fp_tp['PNnonMD-ratio'] = p.roc(args.oldpn,"ratio")
        fp_tp['PNnonMD-nomasscut'] = p.roc(args.oldpn,"nomass")

    if args.mbranch:
        # for flat sample
        if ak.any(p.mask_flat):
            fp_tp[r"$m_H$ flat"] = p.roc(score,"sigmask",p.mask_flat)
            if ak.any(p.mask_proc_sigmh125):
                fp_tp[r"$m_H$ flat $\times p_T^SM$"] = p.roc(score,"ptweight",p.mask_flat)
            if args.oldpn:
                fp_tp[r"$m_H$ flat PNnonMD"] = p.roc(args.oldpn,"ptweight",p.mask_flat)
        if ak.any(p.mask_proc_mh120130) and ak.any(p.mask_proc_sigmh125):
            fp_tp[r"$m_H:[120,130]$ GeV $\times p_T^SM$"] = p.roc(score,"ptweight",p.mask_proc_mh120130)
        if args.nprongs:
            fp_tp[r"$m_H$ flat 3prongs"] = p.roc(score,"sigmask",(p.mask_flat & p.mask3p))
            fp_tp[r"$m_H$ flat 4prongs"] = p.roc(score,"sigmask",(p.mask_flat & p.mask4p))
        # for mh125
        if ak.any(p.mask_mh125):
            fp_tp[r"$m_H$:125"] = p.roc(score,"sigmask",p.mask_mh125)
        if args.oldpn:
            fp_tp[r'$m_H$:125 PNnonMD'] = p.roc(args.oldpn,"sigmask",p.mask_mh125)
        if args.nprongs:
            fp_tp[r"$m_H$:125 3prongs"] = p.roc(score,"sigmask",(p.mask_mh125 & p.mask3p))
            fp_tp[r"$m_H$:125 4prongs"] = p.roc(score,"sigmask",(p.mask_mh125 & p.mask4p))
            
    else:
        if args.nprongs:
            fp_tp["3prongs"] = p.roc(score,"sigmask",p.mask3p)
            fp_tp["4prongs"] = p.roc(score,"sigmask",p.mask4p)
            
    return fp_tp

def get_rocs_by_var(odir,mbranch,sigmask=None):
    vars_to_corr = {r"$p_T$": "fj_pt"}
    bin_ranges = [list(range(200, 1000, 200))]
    bin_widths = [200] * len(bin_ranges)
    if mbranch:
        vars_to_corr[r"$m_H$"] = args.mbranch
        bin_ranges += [list(range(60, 240, 20))]
        bin_widths = [10] * len(bin_ranges)
    
def main(args):
    signals = args.signals.split(",")
    backgrounds = args.bkgs.split(",")
    if len(signals) != len(backgrounds):
        print("Number of signals should be the same as backgrounds!")
        exit
        
    fp_tp_all = {}
    for i, sig in enumerate(signals):
        bkg = backgrounds[i]
        
        odir = args.odir + sig
        os.system(f"mkdir -p {odir}")

        sigfiles = None
        if args.isig:
            sigfiles = args.isig.split(',')
        else:
            sigfiles = [None]
            
        fp_tp_sigfiles = {}
        keys_sigfiles = []
        keys_ratio = []
        fillmh = False
        for j, sigfile in enumerate(sigfiles):
            p = PlotOutput(
                args.ifile,
                args.name,
                odir,
                args.jet,
                sig,
                bkg,
                sigfile,
                args.dnn,
                args.verbose,
                args.oldpn,
                args.mbranch,
                args.nprongs,
            )                        
            
            # ROCS
            signame = ""
            if args.isig:
                signame = " "+args.isignames.split(',')[j]
                
            fp_tp = get_rocs(p,args,signame)
            if args.isig:
                for key,item in fp_tp.items():
                    fp_tp_sigfiles[key+signame] = item
                    if key==p.score:
                        keys_sigfiles.append(key+signame)
                        if j==0:
                            fp_tp_all[sig] = [p.score]
                    elif key==p.score+"-ratio":
                        keys_ratio.append(key+signame)
                    elif "m_H" in key and p.mbranch and not fillmh:
                        keys_sigfiles.append(key+signame)
                        fillmh = True
                    else:
                        print('not plotting ',key)
            else:
                fp_tp_sigfiles = fp_tp
                keys_sigfiles = [p.score]
                keys_ratio = [p.score+"-ratio"]
                fp_tp_all[sig] = [p.score]
                if p.mbranch:
                    keys_sigfiles += [r"$m_H$ flat",r"$m_H$:125"]
                if args.oldpn:
                    keys_sigfiles += ["PNnonMD"]

        if args.verbose:
            print(fp_tp_sigfiles.keys())
            
        # plot
        p.plot(fp_tp_sigfiles, "summary", keys_sigfiles)
        p.plot(fp_tp_sigfiles, "ratio", keys_ratio)

        # plot mass decorrelation

        # plot for different mH and pT cuts
        
    if len(signals)>1:
        # plot summary ROC for all signal classes and first isig file
        plot_roc(
            args.odir,
            "HWW",
            "QCD",
            fp_tp_all,
            label="allsig_summary",
            title=f"HWW vs {bkglegend}",
            ptcut="",
            msdcut="",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", help="input file with bkg and possibly signal")
    parser.add_argument("--isig", default=None, help="signal files - split by ,")
    parser.add_argument("--isignames", default=None, help="signal tags - split by ,")
    parser.add_argument("--odir", required=True, help="output dir")
    parser.add_argument("--name", required=True, help="name of the model(s)")
    parser.add_argument("--mbranch", default=None, help="mass branch name")
    parser.add_argument("--jet", default="AK15", help="jet type")
    parser.add_argument("--dnn", action="store_true", default=False, help="is dnn tuple?")
    parser.add_argument("--oldpn", action="store_true", default=False, help="is oldpn branch saved?")
    parser.add_argument("-v", action="store_true", default=False, help="verbose", dest="verbose")
    parser.add_argument("--nprongs", action="store_true", default=False, help="is nprongs branch saved?")
    parser.add_argument("--signals", default="hww_4q_merged", help="signals")
    parser.add_argument(
        "--bkgs",
        default="qcd",
        help="backgrounds (if qcd_label then assume that you only have one qcd label)",
    )
    args = parser.parse_args()

    main(args)
