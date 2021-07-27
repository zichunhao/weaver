# Import Statements
import numpy as np
import uproot
import matplotlib.pyplot as plt

import mplhep as hep
hep.style.use("CMS")

# load output file
branches = ["fj_pt","fj_genjetmsd",
            "fj_genRes_mass",
            "output",
            "target_mass",
            ]

qcd_cats = ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]
sig_cats = ["fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_WW_taunuqq"]

branches += qcd_cats
branches += sig_cats

output = uproot.open('/storage/af/user/cmantill/training/weaver/output/3_v04_ak15_regression_hwwQCD_Jul21_ep19.root:Events').arrays(branches)

# define histograms
from coffea import hist
genresmass_range = [-99] + list(range(0,200,5))
hist_mass = hist.Hist("mass", 
                      hist.Cat("process", "Process"), 
                      hist.Bin("genmsd", r"fj GEN msoftdrop [GeV]", 60, 0, 260),
                      hist.Bin("targetmass", r"Target mass [GeV]", 70, 0, 260),
                      hist.Bin("genresmass", r"GEN mass [GeV]", genresmass_range),
                      hist.Bin("outputmass", r"Regressed mass", 60, 0, 260),
)
                        
proc_dict = {"QCDb": "fj_isQCDb",
             "QCDbb": "fj_isQCDbb",
             "QCDc": "fj_isQCDc",
             "QCDcc": "fj_isQCDcc",
             "QCDlep": "fj_isQCDlep",
             "QCDothers": "fj_isQCDothers",
             #"HWW4q": "fj_H_WW_4q",
             #"HWWelenuqq": "fj_H_WW_elenuqq",
             #"HWWmunuqq": "fj_H_WW_munuqq",
             #"HWWtaunuqq":  "fj_H_WW_taunuqq",
}

print(output)

legends = []
# loop over processes
for proc,label in proc_dict.items():
    # fill histogram
    mask_proc = output[label]==1
    # add mass H 125 for signal
    if "HWW" in label:
        mask_proc = mask_proc & (output["fj_genRes_mass"]==125)
    # apply mask
    hist_mass.fill(process=proc,
                   genmsd = output["fj_genjetmsd"][mask_proc],
                   targetmass = output["target_mass"][mask_proc],
                   genresmass = output["fj_genRes_mass"][mask_proc],
                   outputmass = output["output"][mask_proc],
        )


# make mass plots
mass_to_plot = ["genmsd","targetmass","genresmass","outputmass"]
for m in mass_to_plot:
    fig, ax = plt.subplots(1,1)
    hist.plot1d(hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process',m}]),ax=ax,overlay="process")
    ax.set_ylabel('Jets')
    fig.savefig("%s.png"%m, bbox_inches='tight')
