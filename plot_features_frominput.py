import uproot
import awkward as ak
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep

import argparse

from coffea import hist

hists_dict = {
    'fj_prop': hist.Hist("fj",
                         hist.Cat("process", "Process"),
                         hist.Bin("gmass", r"gen Res mass [GeV]", 42, 50, 260),
                         hist.Bin("msd", r"fj msoftdrop [GeV]", 50, 20, 260),
                         hist.Bin("pt", r"fj pT [GeV]", 60, 300, 1500),
                         hist.Bin("mass", r"fj mass [GeV]", 30, 50, 200),
                         ),
    'h_gen': hist.Hist("h",
                       hist.Cat("process", "Process"),
                       hist.Bin("gmass", r"gen Res mass [GeV]", 42, 50, 260),
                       hist.Bin("gpt_over_mass", "gen H pT/mass", 30, 0, 25),
                       hist.Bin("gpt", r" gen H pt [GeV]", 30, 200, 1600),
                       ),
    'fj_dr': hist.Hist("dr",
                       hist.Cat("process", "Process"),
                       hist.Bin("gmass", r"gen Res mass [GeV]", 42, 50, 260),
                       hist.Bin("dr_W", r"dR(fj,W)", 25, 0, 2.5),
                       hist.Bin("dr_Wstar", r"dR(fj,W*)", 25, 0, 2.5),
                       hist.Bin("min_dr_Wdau", r"min dR(fj,4qs)", 30, 0, 1.0),
                       hist.Bin("max_dr_Wdau", r"max dR(fj,4qs)", 45, 0, 2.5)
                       ),
    'w_gen': hist.Hist("w",
                       hist.Cat("process", "Process"),
                       hist.Bin("gmass", r"gen Res mass [GeV]", 42, 50, 260),
                       hist.Bin("wmass", r"gen W mass [GeV]", 30, 10, 110),
                       hist.Bin("wsmass", r"gen W* mass [GeV]", 30, 10, 100),
                       hist.Bin("wpt", r"gen W pT [GeV]", 30, 50, 1000),
                       hist.Bin("wspt", r"gen W* pT [GeV]", 30, 50, 1000),
                       ),
    'labels_qcd': hist.Hist("labels",
                            hist.Cat("process", "Process"),
                            hist.Bin("QCDb", "QCD b label", 2, 0, 2),
                            hist.Bin("QCDbb", "QCD bb label", 2, 0, 2),
                            hist.Bin("QCDc", "QCD c label", 2, 0, 2),
                            hist.Bin("QCDcc", "QCD cc label", 2, 0, 2),
                            hist.Bin("QCDlep", "QCD lep label", 2, 0, 2),
                            hist.Bin("QCDothers", "QCD others label", 2, 0, 2),
                            ),
    'labels_ww': hist.Hist("labels",
                           hist.Cat("process", "Process"),
                           hist.Bin("gmass", r"gen Res mass [GeV]", 42, 50, 260),
                           hist.Bin("ww_4q", "WW 4q label", 2, 0, 2),
                           hist.Bin("ww_3q", "WW 3q label", 2, 0, 2),
                           hist.Bin("ww_elenuqq_merged", "WW ele merged label", 2, 0, 2),
                           hist.Bin("ww_elenuqq_semimerged", "WW ele semi-merged label", 2, 0, 2),
                           hist.Bin("ww_taunuqq_merged", "WW tau merged label", 2, 0, 2),
                           hist.Bin("ww_taunuqq_semimerged", "WW tau semi-merged label", 2, 0, 2),
                           ),
    'target': hist.Hist("mass",
                        hist.Cat("process", "Process"),
                        hist.Bin("gmass", r"gen Res mass [GeV]", 42, 50, 260),
                        hist.Bin("genmsd", r"fj gen msoftdrop [GeV]", 60, 0, 260),
                        hist.Bin("targetmass", r"target mass [GeV]", 70, 0, 260),
                        ),
}


def plot_by_mass(h, vars_to_plot, plabel, args, by_proc=[]):
    #masses = [120]
    masses = list(range(50, 200, 20))
    masses.remove(70)
    masses.remove(110)
    masses.remove(150)

    invmasses = list(reversed(range(50, 200, 20)))
    #invmasses += [120]
    invmasses.remove(70)
    invmasses.remove(110)
    invmasses.remove(150)

    if "gmass" in vars_to_plot:
        vars_to_plot.remove("gmass")

    if len(by_proc) == 0:
        by_proc = [1]
        plotlabel = plabel
    else:
        plotlabel = plabel + 'all'

    for density in [True, False]:
        fig, axs = plt.subplots(len(by_proc), len(vars_to_plot), figsize=(len(vars_to_plot)*7, len(by_proc)*7))

        for k, proc in enumerate(by_proc):
            if proc != 1:
                y = h.integrate('process', proc)
            else:
                y = h

            for i, var in enumerate(vars_to_plot):

                if len(by_proc) == 1:
                    ax_var = axs[i]
                else:
                    ax_var = axs[k, i]
                legs = []

                # choose order of masses to plot
                lmasses = masses
                if var in ['dr_W', 'dr_Wstar', 'min_dr_Wdau', 'max_dr_Wdau', 'msd', 'mass']:
                    lmasses = masses
                if var in ['wsmass', 'gpt_over_mass'] or proc in ['HWW3q']:
                    lmasses = invmasses

                for j, m in enumerate(lmasses):
                    if m == 120 or m == 125:
                        x = y.sum(*[ax for ax in y.axes() if ax.name not in {'gmass', var}]).integrate('gmass', m)
                    else:
                        x = y.sum(*[ax for ax in y.axes() if ax.name not in {'gmass', var}]).integrate('gmass', slice(m, m+10))
                    if j == 0:
                        hist.plot1d(x, ax=ax_var, density=density)
                    else:
                        hist.plot1d(x, ax=ax_var, clear=False, density=density)
                    legs.append('mH=%i GeV' % m)

                leg = ax_var.legend(legs)
                ax_var.set_ylabel('Jets')
                ax_var.set_title(proc)
        if density:
            fig.savefig("%s/%s_by_mh_density.pdf" % (args.odir, plotlabel))
        else:
            fig.savefig("%s/%s_by_mh.pdf" % (args.odir, plotlabel))


def plot_by_process(h, vars_to_plot, label, args):
    for density in [True, False]:
        fig, axs = plt.subplots(1, len(vars_to_plot), figsize=(len(vars_to_plot)*7, 7))
        for i, var in enumerate(vars_to_plot):
            #print(h.sum(*[ax for ax in h.axes() if ax.name not in {'process',var}]).identifiers("process", overflow='all'))
            x = h.sum(*[ax for ax in h.axes() if ax.name not in {'process', var}])
            hist.plot1d(x, ax=axs[i], overlay="process", density=density)
            axs[i].set_ylabel('Jets')
        fig.tight_layout()
        if density:
            fig.savefig("%s/%s_density.pdf" % (args.odir, label))
        else:
            fig.savefig("%s/%s.pdf" % (args.odir, label))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',  required=True, help='samples to load (qcd,hh,grav,bulk,hww,hwwlnuqq,tthad,ttlep,wjets)')
    parser.add_argument('--test', action='store_true', default=False, help='test w few files')
    parser.add_argument('--mh', action='store_true', default=False, help='add plots by mh')
    parser.add_argument('--selection', type=str, default='train', help='selection')
    parser.add_argument('--hists', choices=['fj_prop', 'h_gen', "fj_dr", "w_gen", "labels_ww", "labels_qcd", "target"], help='possible histograms to fill', required=True)
    parser.add_argument('--odir',  required=True, help='plots directory')
    parser.add_argument('--data-dir', required=True, help='directory for train and test data files')
    args = parser.parse_args()

    import os
    os.system('mkdir -p %s' % args.odir)

    ifiles = {
        "grav": os.path.join(args.data_dir, "train/GravitonToHHToWWWW*/*.root:Events"),
        "hh": os.path.join(args.data_dir, "test/*HH*cHHH1*/*.root:Events"),
        "qcd": os.path.join(args.data_dir, "train/QCD*/*.root:Events"),
        "bulk": os.path.join(args.data_dir, "train/Bulk*/*.root:Events"),
        "hww": os.path.join(args.data_dir, "test/*HToWW_*/*.root:Events"),
        "hwwlnuqq": os.path.join(args.data_dir, "test/*HToWWTo*/*.root:Events"),
        "tthad": os.path.join(args.data_dir, "test/*TTToHad*/*.root:Events"),
        "ttlep": os.path.join(args.data_dir, "test/*TTToSemi*/*.root:Events"),
        "wjets": os.path.join(args.data_dir, "test/*WJets*/*.root:Events"),
    }
    # use few files if testing script
    if args.test:
        ifiles = {
            "grav": os.path.join(args.data_dir, "train/GravitonToHHToWWWW/nano_mc2017_87_Skim.root:Events"),
            "hh": os.path.join(args.data_dir, "test/GluGluToHHTo4V_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/nano_mc2017_10_Skim.root:Events"),
            "qcd": os.path.join(args.data_dir, "train/QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/nano_mc2017_1-101_Skim.root:Events"),
            "bulk": os.path.join(args.data_dir, "train/BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part1_TuneCP5_13TeV-madgraph_pythia8/nano_mc2017_81_Skim.root:Events"),
        }

    # define branches to load
    sig_branches = ["fj_H_WW_4q", "fj_H_WW_elenuqq", "fj_H_WW_munuqq", "fj_H_WW_taunuqq"]
    qcd_branches = ["fj_isQCDb", "fj_isQCDbb", "fj_isQCDc", "fj_isQCDcc", "fj_isQCDlep", "fj_isQCDothers"]
    hqq_branches = ["fj_H_bb", "fj_H_cc", "fj_H_qq"]
    branches = ["fj_pt",
                "fj_msoftdrop",
                "fj_mass",
                "fj_genjetmsd",
                "fj_genRes_mass",
                "fj_genRes_pt",
                "fj_maxdR_HWW_daus",
                "fj_nProngs",
                ]
    branches += sig_branches
    branches += qcd_branches

    branches += ["fj_isW","fj_isWlep","fj_isTop","fj_isToplep"]
    branches += ["fj_dR_W", "fj_dR_Wstar", "fj_mindR_HWW_daus",
                 "fj_genW_pt", "fj_genWstar_pt", "fj_genW_mass", "fj_genWstar_mass"]

    # define selection
    if args.selection == "regression":
        mask = "(fj_pt>200) & (fj_pt<2500) & "\
            "( ( ( (fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1) ) & "\
            "(fj_genRes_mass<0) ) | "\
            "(fj_isTop == 1) | (fj_isToplep==1) | "\
            "( ( (fj_H_WW_4q==1) | (fj_H_WW_elenuqq==1) | (fj_H_WW_munuqq==1) | (fj_H_WW_taunuqq==1) ) & "\
            "(fj_maxdR_HWW_daus<2.0) & (fj_nProngs>1) & (fj_genRes_mass>0) ) )"
    else:
        mask = "(fj_pt>200) & (fj_pt<2500) & "\
            "( ( ( (fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1) ) & "\
            "(fj_genRes_mass<0) ) | "\
            "( ( (fj_H_WW_4q==1) | (fj_H_WW_elenuqq==1) | (fj_H_WW_munuqq==1) | (fj_H_WW_taunuqq==1) ) & "\
            "(fj_maxdR_HWW_daus<2.0) & (fj_nProngs>1) & (fj_genRes_mass>0) ) )"

    mask = "(fj_pt>200) & (fj_pt<2500) & (fj_msoftdrop>=30) & (fj_msoftdrop<260)" # relaxed mask
    mask = "(fj_pt>200) & (fj_pt<2500)" # basic mask
    #mask = "(fj_pt>200) & (fj_pt<2500) & ((fj_genRes_mass<0) | (fj_genRes_pt/fj_genRes_mass<=5))" #ptmass mask
    mask = "(fj_pt>200) & (fj_pt<2500) & ((fj_genRes_mass<0) | ((fj_genW_mass>40) & (fj_genWstar_mass<50)))"
    
    # define processes
    # sig_processes = ["fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_WW_taunuqq"]
    sig_processes = ["fj_isHWW_elenuqq_merged", "fj_isHWW_elenuqq_semimerged",
                     "fj_isHWW_munuqq_merged", "fj_isHWW_munuqq_semimerged",  # only load electrons since that should be the same as muons
                     "fj_isHWW_taunuqq_merged", "fj_isHWW_taunuqq_semimerged",
                     "fj_H_WW_4q_3q", "fj_H_WW_4q_4q"]
    qcd_processes = ["fj_isQCDb", "fj_isQCDbb", "fj_isQCDc", "fj_isQCDcc", "fj_isQCDlep", "fj_isQCDothers"]

    proc_dict = {
        "qcd": {"QCDb": "fj_isQCDb",
                "QCDbb": "fj_isQCDbb",
                "QCDc": "fj_isQCDc",
                "QCDcc": "fj_isQCDcc",
                "QCDlep": "fj_isQCDlep",
                "QCDothers": "fj_isQCDothers",
                },
        "hww": {"HWW4q": "fj_H_WW_4q",
                "HWWelenuqq": "fj_H_WW_elenuqq",
                "HWWmunuqq": "fj_H_WW_munuqq",
                "HWWtaunuqq": "fj_H_WW_taunuqq",
                },
        "hww_all": {"HWW4q": "fj_H_WW_4q_4q",
                    "HWW3q": "fj_H_WW_4q_3q",
                    "HWWele-merged": "fj_isHWW_elenuqq_merged",
                    #"HWWele-semi": "fj_isHWW_elenuqq_semimerged",
                    #"HWWmu-merged": "fj_isHWW_munuqq_merged",
                    #"HWWmu-semi": "fj_isHWW_munuqq_semimerged",
                    "HWWtau-merged": "fj_isHWW_taunuqq_merged",
                    "HWWtau-semi": "fj_isHWW_taunuqq_semimerged",
                    },
        "tthad":{
            "had": "fj_isTop",
            #"hadw": "fj_isW",
        },
        "ttlep":{
            "lep": "fj_isToplep",
            #"lepwl": "fj_isWlep",
            #"lepw": "fj_isW",
        },
        "wjets":{
            "wlep": "fj_isWlep",
        }
    }

    #proc_dict["grav"]  = proc_dict["hww"]
    #proc_dict["hh"]  = proc_dict["hww"]
    
    proc_dict["grav"] = proc_dict["hww_all"]
    proc_dict["hh"] = proc_dict["hww_all"]
    proc_dict["hww"] = proc_dict["hww_all"]
    proc_dict["hwwlnuqq"] = proc_dict["hww_all"]

    # try all
    #proc_dict["qcd"] = {"qcd": "fj_isQCDb"}
    #proc_dict["grav"] = {"grav": "fj_H_WW_4q"}
    #proc_dict["hww"] = {"ww": "fj_H_WW_4q"}
    #proc_dict["hh"] =  {"hh": "fj_H_WW_4q"}
    #proc_dict["hwwlnuqq"] =  {"wwlnu": "fj_H_WW_elenuqq"}

    # define which histograms to fill
    hist_to_fill = args.hists
    hists = {hist_to_fill: hists_dict[hist_to_fill]}

    # open events
    for sample in args.samples.split(','):
        events = uproot.iterate(ifiles[sample], branches, mask)

        for ev in events:
            # need to define new variables for all labels:
            ev["fj_H_WW_4q_4q"] = (ev["fj_H_WW_4q"] == 1) & (ev["fj_nProngs"] == 4)
            ev["fj_H_WW_4q_3q"] = (ev["fj_H_WW_4q"] == 1) & (ev["fj_nProngs"] == 3)
            ev["fj_isHWW_elenuqq_merged"] = (ev["fj_H_WW_elenuqq"] == 1) & (ev["fj_nProngs"] == 4)
            ev["fj_isHWW_elenuqq_semimerged"] = (ev["fj_H_WW_elenuqq"] == 1) & ((ev["fj_nProngs"] == 3) | (ev["fj_nProngs"] == 2))
            ev["fj_isHWW_munuqq_merged"] = (ev["fj_H_WW_munuqq"] == 1) & (ev["fj_nProngs"] == 4)
            ev["fj_isHWW_munuqq_semimerged"] = (ev["fj_H_WW_munuqq"] == 1) & ((ev["fj_nProngs"] == 3) | (ev["fj_nProngs"] == 2))
            ev["fj_isHWW_taunuqq_merged"] = (ev["fj_H_WW_taunuqq"] == 1) & (ev["fj_nProngs"] == 4)
            ev["fj_isHWW_taunuqq_semimerged"] = (ev["fj_H_WW_taunuqq"] == 1) & ((ev["fj_nProngs"] == 3) | (ev["fj_nProngs"] == 2))
            ev["target_mass"] = np.maximum(1, np.where(ev["fj_genRes_mass"] > 0, ev["fj_genRes_mass"], ev["fj_genjetmsd"]))

            # fill histograms
            for proc, label in proc_dict[sample].items():
                # apply label mask to all feature arrays
                mask_proc = (ev[label] == 1)
                # no mask on labels
                #mask_proc = (ev[label] == 1) | (ev[label] == 0)
                for k, h in hists.items():
                    if k == "fj_prop":
                        hists[k].fill(process=proc,
                                      gmass=ev.fj_genRes_mass[mask_proc],
                                      msd=ev.fj_msoftdrop[mask_proc],
                                      pt=ev.fj_pt[mask_proc],
                                      mass=ev.fj_mass[mask_proc],
                                      )
                    elif k == "h_gen":
                        hists[k].fill(process=proc,
                                      gmass=ev.fj_genRes_mass[mask_proc],
                                      gpt_over_mass=ev.fj_genRes_pt[mask_proc]/ev.fj_genRes_mass[mask_proc],
                                      gpt=ev.fj_genRes_pt[mask_proc],
                                      )
                    elif k == "fj_dr":
                        hists[k].fill(process=proc,
                                      gmass=ev.fj_genRes_mass[mask_proc],
                                      dr_W=ev.fj_dR_W[mask_proc],
                                      dr_Wstar=ev.fj_dR_Wstar[mask_proc],
                                      min_dr_Wdau=ev.fj_mindR_HWW_daus[mask_proc],
                                      max_dr_Wdau=ev.fj_maxdR_HWW_daus[mask_proc],
                                      )
                    elif k == 'w_gen':
                        hists[k].fill(process=proc,
                                      gmass=ev.fj_genRes_mass[mask_proc],
                                      wmass=ev.fj_genW_mass[mask_proc],
                                      wsmass=ev.fj_genWstar_mass[mask_proc],
                                      wpt=ev.fj_genW_pt[mask_proc],
                                      wspt=ev.fj_genWstar_pt[mask_proc],
                                      )
                    elif k == 'labels_qcd':
                        hists[k].fill(process=proc,
                                      QCDb=ev.fj_isQCDb[mask_proc],
                                      QCDbb=ev.fj_isQCDbb[mask_proc],
                                      QCDc=ev.fj_isQCDc[mask_proc],
                                      QCDcc=ev.fj_isQCDcc[mask_proc],
                                      QCDlep=ev.fj_isQCDlep[mask_proc],
                                      QCDothers=ev.fj_isQCDothers[mask_proc],
                                      )
                    elif k == "labels_ww":
                        hists[k].fill(process=proc,
                                      gmass=ev.fj_genRes_mass[mask_proc],
                                      ww_4q=ev.fj_H_WW_4q_4q[mask_proc],
                                      ww_3q=ev.fj_H_WW_4q_3q[mask_proc],
                                      ww_elenuqq_merged=ev.fj_isHWW_elenuqq_merged[mask_proc],
                                      ww_elenuqq_semimerged=ev.fj_isHWW_elenuqq_semimerged[mask_proc],
                                      ww_taunuqq_merged=ev.fj_isHWW_taunuqq_merged[mask_proc],
                                      ww_taunuqq_semimerged=ev.fj_isHWW_taunuqq_semimerged[mask_proc],
                                      )
                    elif k == "target":
                        hists[k].fill(process=proc,
                                      gmass=ev.fj_genRes_mass[mask_proc],
                                      genmsd=ev.fj_genjetmsd[mask_proc],
                                      targetmass=ev.target_mass[mask_proc],
                                      )

    # now plot
    # we can either plot by process
    var_dict = {
        "fj_prop": ["msd", "gmass", "pt"],
        "h_gen": ["gpt_over_mass", "gpt"],
        "fj_dr": ["dr_W", "dr_Wstar", "min_dr_Wdau", "max_dr_Wdau"],
        "w_gen": ["wmass", "wsmass", "wpt", "wspt"],
        "labels_ww": ["ww_4q", "ww_3q", "ww_elenuqq_merged", "ww_elenuqq_semimerged", "ww_taunuqq_merged", "ww_taunuqq_semimerged"],
        "labels_qcd": ["QCDb","QCDbb","QCDc","QCDcc","QCDlep","QCDothers"],
        "target": ["gmass","genmsd","targetmass"],
    }

    # plot feature by process (label)
    for k, h in hists.items():
        plot_by_process(h, var_dict[k], k, args)

    # or plot by gen mass value
    if args.mh:
        for k, h in hists.items():
            # plot by mass but sum all of the processes
            plot_by_mass(h, var_dict[k], k, args)

            # plot by mass for these different processes
            plot_by_mass(h, var_dict[k], k, args, by_proc=["HWW4q", "HWW3q", "HWWele-merged", "HWWtau-merged"])
