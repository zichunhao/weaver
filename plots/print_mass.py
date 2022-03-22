import uproot3
import awkward0 as ak
import numpy as np

from utils.data.fileio import _read_files

filelist = ["/data/shared/cmantill/training/ak15_Jul9/train/GravitonToHHToWWWW_part1/nano_mc2017_76_Skim.root",
            "/data/shared/cmantill/training/ak15_Jul9/train/QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/nano_mc2017_1-17_Skim.root",
]

branches = ["fj_genRes_mass",
            "fj_genjetmsd",
            "fj_nProngs",
]

table = _read_files(filelist, branches, None, True, treename="Events")

print(table["fj_genRes_mass"])
print(table["fj_genjetmsd"])
fj_nProngs = table["fj_nProngs"]
fj_genjetmsd = table["fj_genjetmsd"]
fj_genRes_mass =  table["fj_genRes_mass"]

print(np.maximum(1,np.where(fj_nProngs==0, fj_genjetmsd, fj_genRes_mass)))
#np.maximum(1, np.where(fj_nProngs==0, fj_genjetmsd, fj_genRes_mass))

from utils.data.tools import _eval_expr
expr = "np.maximum(1,np.where(fj_nProngs==0, fj_genjetmsd, fj_genRes_mass))"
table["mass"] = _eval_expr(expr, table)
print(table["mass"])
