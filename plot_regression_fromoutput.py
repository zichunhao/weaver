# Import Statements
import numpy as np
import uproot
import matplotlib.pyplot as plt

import mplhep as hep
hep.style.use("CMS")

# branches to load
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
# process dictionary
proc_dict = {
    "qcd":{"QCDb": "fj_isQCDb",
           "QCDbb": "fj_isQCDbb",
           "QCDc": "fj_isQCDc",
           "QCDcc": "fj_isQCDcc",
           "QCDlep": "fj_isQCDlep",
           "QCDothers": "fj_isQCDothers",
    }
    "sig":{"HWW4q": "fj_H_WW_4q",
           "HWWelenuqq": "fj_H_WW_elenuqq",
           "HWWmunuqq": "fj_H_WW_munuqq",
           "HWWtaunuqq":  "fj_H_WW_taunuqq",
    }
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', default='qcd,sig', help='processes to plot (need to be separated by commas and need to be in proc_dict')
    parser.add_argument('--ifile', help="file from inference", required=True)
    args = parser.parse_args()

    output = uproot.open(args.ifile)

    to_process = args.process.split(',')

    legends = []
    # loop over processes
    for p in to_process:
        # define histogram for each big process (e.g. QCD or signal)
        from coffea import hist
        genresmass_range = [-99] + list(range(0,200,5))
        hist_mass = hist.Hist("mass",
                              hist.Cat("process", "Process"),
                              hist.Bin("genmsd", r"fj GEN msoftdrop [GeV]", 60, 0, 260),
                              hist.Bin("targetmass", r"Target mass [GeV]", 70, 0, 260),
                              hist.Bin("genresmass", r"GEN mass [GeV]", genresmass_range),
                              hist.Bin("outputmass", r"Regressed mass", 60, 0, 260),
        )
        
        hist_ratio = hist.Hist("ratio",
                               hist.Cat("process", "Process"),
                               hist.Bin("outputratio", r"Regressed mass/Target mass", 60, 0, 2),
        )
        
        for proc,label in proc_dict[p].items():
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
            hist_ratio.fill(process=proc,
                        outputratio = output["output"][mask_proc]/output["target_mass"][mask_proc],
            )
            
        # make mass plots
        mass_to_plot = ["genmsd","targetmass","genresmass","outputmass"]
        fig, axs = plt.subplots(len(mass_to_plot),1)
        for i,m in enumerate(mass_to_plot):
            hist.plot1d(hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process',m}]),ax=axs[i],overlay="process")
        axs[i].set_ylabel('Jets')
        fig.tight_layout()
        fig.savefig("mass_%s.png"%p)

        ratio_to_plot = ["outputratio"]
        fig, axs = plt.subplots(len(ratio_to_plot),1)
        for i,m in enumerate(ratio_to_plot):
            hist.plot1d(hist_ratio.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process',m}]),ax=axs[i],overlay="process")
        axs[i].set_ylabel('Jets')
        fig.tight_layout()
        fig.savefig("ratio_%s.png"%p)
