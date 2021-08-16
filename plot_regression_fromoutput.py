# Import Statements
import numpy as np
import uproot
import argparse
import matplotlib.pyplot as plt

import mplhep as hep
hep.style.use("CMS")

branches = ["fj_pt","fj_genjetmsd",
            "fj_genRes_mass",
            "output",
            "target_mass",
            "fj_msoftdrop",
            ]

qcd_cats = ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]
sig_cats = ["fj_H_bb", "fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_WW_taunuqq"]

branches += qcd_cats
branches += sig_cats

# process dictionary
proc_dict = {
    "qcd":{"QCDb": "fj_isQCDb",
           "QCDbb": "fj_isQCDbb",
           "QCDc": "fj_isQCDc",
           "QCDcc": "fj_isQCDcc",
           "QCDlep": "fj_isQCDlep",
           "QCDothers": "fj_isQCDothers",
    },
    "sig":{"HWW4q": "fj_H_WW_4q",
           "HWWelenuqq": "fj_H_WW_elenuqq",
           "HWWmunuqq": "fj_H_WW_munuqq",
           "HWWtaunuqq":  "fj_H_WW_taunuqq",
           "Hbb": "fj_H_bb"
    }
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', default='qcd,sig', help='processes to plot (need to be separated by commas and need to be in proc_dict')
    parser.add_argument('--ifile', help="file from inference", required=True)
    parser.add_argument('--odir', required=True, help="output dir")
    args = parser.parse_args()

    import os
    os.system('mkdir -p %s'%args.odir)
    
    output = uproot.open(args.ifile)["Events"].arrays(branches)

    to_process = args.process.split(',')

    legends = []
    # loop over processes
    for p in to_process:
        # define histogram for each big process (e.g. QCD or signal)
        from coffea import hist
        genresmass_range = [-99] + list(range(0,200,5))
        hist_mass = hist.Hist("mass",
                              hist.Cat("process", "Process"),
                              hist.Bin("genmsd", r"fj GEN msoftdrop [GeV]", 70, 0, 460),
                              hist.Bin("targetmass", r"Target mass [GeV]", 70, 0, 460),
                              hist.Bin("genresmass", r"GEN mass [GeV]", genresmass_range),
                              hist.Bin("outputmass", r"Regressed mass", 60, 0, 460),
                              hist.Bin("msd", r"fj msoftdrop [GeV]", 60, 0, 460),
        )
        
        hist_ratio = hist.Hist("ratio",
                               hist.Cat("process", "Process"),
                               hist.Bin("outputratio", r"Regressed mass/Target mass", 60, 0, 2),
        )
        
        for proc,label in proc_dict[p].items():
            # fill histogram
            mask_proc = (output[label]==1)
            # add mass H 125 for signal
            if "HWW" in label:
                mask_proc = mask_proc & (output["fj_genRes_mass"]==125)
            # apply mask
            hist_mass.fill(process=proc,
                           genmsd = output["fj_genjetmsd"][mask_proc],
                           targetmass = output["target_mass"][mask_proc],
                           genresmass = output["fj_genRes_mass"][mask_proc],
                           outputmass = output["output"][mask_proc],
                           msd = output["fj_msoftdrop"][mask_proc],
            )
            hist_ratio.fill(process=proc,
                        outputratio = output["output"][mask_proc]/output["target_mass"][mask_proc],
            )
            
        # make mass plots
        mass_to_plot = ["genmsd","targetmass","genresmass","outputmass"]
        fig, axs = plt.subplots(1,len(mass_to_plot), figsize=(len(mass_to_plot)*8,8))
        for i,m in enumerate(mass_to_plot):
            hist.plot1d(hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process',m}]),ax=axs[i],overlay="process")
            axs[i].set_ylabel('Jets')
        fig.tight_layout()
        fig.savefig("%s/mass_%s.pdf"%(args.odir,p))

        # plot both mass and msoftdrop
        fig, axs = plt.subplots(1,1, figsize=(8,8))
        omass = hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'outputmass'}])
        smass = hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'msd'}])

        hist.plot1d(omass,ax=axs)
        hist.plot1d(smass,ax=axs,clear=False)
        axs.legend(['Regressed','Softdrop'])
        fig.tight_layout()
        fig.savefig("%s/comparemass_%s.pdf"%(args.odir,p))

        # for mh125
        if p=="sig":
            fig, axs = plt.subplots(1,1, figsize=(8,8))
            omass = hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'outputmass','genresmass'}])
            smass = hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'msd','genresmass'}])
            omass = omass.integrate('genresmass',slice(125,130))
            smass = smass.integrate('genresmass',slice(125,130))
            print(omass,smass)
            hist.plot1d(omass,ax=axs)
            hist.plot1d(smass,ax=axs,clear=False)
            axs.legend(['Regressed','Softdrop'])
            fig.tight_layout()
            fig.savefig("%s/comparemass_%s_mh125.pdf"%(args.odir,p))
            
        """
        # plot ratio
        ratio_to_plot = ["outputratio"]
        fig, axs = plt.subplots(1,len(ratio_to_plot), figsize=(len(ratio_to_plot)*8,8))
        for i,m in enumerate(ratio_to_plot):
            axs_1 = axs
            if len(ratio_to_plot)>1:
                axs_1 = axs[i]
            hist.plot1d(hist_ratio.sum(*[ax for ax in hist_ratio.axes() if ax.name not in {'process',m}]),ax=axs_1,overlay="process")
            axs_1.set_ylabel('Jets')
        fig.tight_layout()
        fig.savefig("%s/ratio_%s.pdf"%(args.odir,p))
        """
