# Import Statements
import numpy as np
import uproot
import matplotlib.pyplot as plt

import mplhep as hep
hep.style.use("CMS")


# load Data files
branches = ["fj_pt","fj_genjetmsd",
            "fj_genRes_mass",
            "fj_maxdR_HWW_daus"]

qcd_cats = ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]
sig_cats = ["fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_WW_taunuqq"]

branches += qcd_cats
branches += sig_cats

mask_train = "(fj_pt>200) & (fj_pt<2500) & (fj_genjetmsd<260) &"\
    "( ( ( (fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1) ) & "\
    "(fj_genRes_mass<0) ) | "\
    "( ( (fj_H_WW_4q==1) | (fj_H_WW_elenuqq==1) | (fj_H_WW_munuqq==1) | (fj_H_WW_taunuqq==1) ) & "\
    "(fj_maxdR_HWW_daus<2.0) ) )"

events_QCD = uproot.iterate('/data/shared/cmantill/training/ak15_Jul9/train/QCD*/*.root:Events', branches, mask_train)
events_graviton = uproot.iterate('/data/shared/cmantill/training/ak15_Jul9/train/Graviton*/*.root:Events', branches, mask_train)

# use these instead to test only one file and e.g the plotting
#events_QCD = uproot.iterate('/data/shared/cmantill/training/ak15_Jul9/train/QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/nano_mc2017_1-101_Skim.root:Events', branches) # mask_train)
#events_graviton = uproot.iterate('/data/shared/cmantill/training/ak15_Jul9/train/GravitonToHHToWWWW/nano_mc2017_87_Skim.root:Events', branches, mask_train)

# define histograms so that we can fill them when we iterate over the list of root files
from coffea import hist
genresmass_range = [-99] + list(range(0,200,5))
hist_mass = hist.Hist("mass", 
                      hist.Cat("process", "Process"), 
                      hist.Bin("genmsd", r"fj gen msoftdrop [GeV]", 60, 0, 260),
                      hist.Bin("targetmass", r"target mass [GeV]", 70, 0, 260),
                      hist.Bin("genresmass", r"gen Res mass [GeV]", genresmass_range),
)

hist_labels = hist.Hist("labels",
                        hist.Cat("process", "Process"),
                        hist.Bin("QCDb", "QCD b label", 2, 0, 1),
                        hist.Bin("QCDbb", "QCD bb label", 2, 0, 1),
			hist.Bin("QCDc", "QCD c label", 2, 0, 1),
                        hist.Bin("QCDcc", "QCD cc label", 2, 0, 1),
                        hist.Bin("QCDlep", "QCD lep label", 2, 0, 1),
                        hist.Bin("QCDothers", "QCD others label", 2, 0, 1),
)

hist_fj = hist.Hist("fj",
                    hist.Cat("process", "Process"),
                    hist.Bin("pt", "Jet pT [GeV]", 50, 200, 1500),
                    hist.Bin("msoftdrop", "Jet mSD [GeV]", 50, 0, 250),
)

                        
proc_dict = {"QCD": events_QCD,
             "Grav": events_graviton,
}

legends = []
# loop over processes
for proc in ["QCD","Grav"]:
    # loop over each chunck (e.g. in QCD list or in GraV events)
    for ev in proc_dict[proc]:
        # define new variables
        target_mass = np.maximum(1, np.where(ev["fj_genRes_mass"]>0, ev["fj_genRes_mass"], ev["fj_genjetmsd"]))

        # fill histogram
        hist_mass.fill(process=proc,
                       genmsd = ev["fj_genjetmsd"],
                       targetmass = target_mass,
                       genresmass = ev["fj_genRes_mass"],
        )

        # uncomment if you want to see if the arrays are filled
        #print(target_mass)
        #print(ev["fj_genjetmsd"])
        #print(ev["fj_genRes_mass"])
        
        # fill other histograms here:
        # hist_labels.fill()
        # hist_fj.fill()

# make plots of each parameter for QCD and graviton
# use coffea histogram plotting

# target mass
fig, ax = plt.subplots(1,1)
# sum all the histogram axis except for the variable we want to plot (targetmass) and the process tag (process)
hist.plot1d(hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process','targetmass'}]),ax=ax,overlay="process")
ax.set_ylabel('Jets')
fig.savefig("target_mass.png", bbox_inches='tight')

# gen msd
fig, ax = plt.subplots(1,1)
hist.plot1d(hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process','genmsd'}]),ax=ax,overlay="process")
ax.set_ylabel('Jets')
fig.savefig("gen_msd.png", bbox_inches='tight')

# gen Res mass
fig, ax = plt.subplots(1,1)
hist.plot1d(hist_mass.sum(*[ax for ax in hist_mass.axes() if ax.name not in {'process','genresmass'}]),ax=ax,overlay="process")
ax.set_ylabel('Jets')
fig.savefig("gen_resmass.png", bbox_inches='tight')
