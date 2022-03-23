import uproot
import awkward as ak
import os

plot_dir = "/hwwtaggervol/plots/cm/sample_checks/Mar22/"
sample_dir = "/hwwtaggervol/training/ak15_Mar15/"
branch = "Events"
samples = {
    # label     : [name,         selector to get subset of files (~20 each)]
    "JHU_HHbbWW": ["jhu_HHbbWW", ""],
    #"JHU_HH4W": ["GluGluToBulkGravitonToHHTo4W_JHUGen_M-2500_narrow", ""],
    #"HHbbVV": ["GluGluToHHTobbVV_node_cHHH1", ""],
    #"QCD": ["QCD", ""],
}
os.system(f"mkdir -p {plot_dir}")

for sample, (dir, sel) in samples.items():
    masks = {}
    nums = {}
    events = uproot.iterate(f"{sample_dir}/*/{sel}*.root:{branch}",["fj_pt","fj_H_VV_4q","fj_nProngs"])
    for ev in events:
        masks["all"] = (events["fj_pt"] > 0)
        if "all" in nums.keys():
            nums["all"] += ak.sum(masks["all"])
        else:
            nums["all"] = ak.sum(masks["all"])

    num = nums['all']
    print(f"{sample}: num events {num}")
