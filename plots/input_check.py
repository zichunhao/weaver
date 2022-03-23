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
    for tdir in ["train","test"]:
        print(f"{sample_dir}/{tdir}/{dir}*/{sel}*.root:{branch}")
        try:
            events = uproot.iterate(f"{sample_dir}/{tdir}/{dir}*/{sel}*.root:{branch}",["fj_pt","fj_H_VV_4q"])
            print(events)
            for ev in events:
                masks["all"] = (ev["fj_pt"] > 0)
                if "all" in nums.keys():
                    nums["all"] += ak.sum(masks["all"])
                else:
                    nums["all"] = ak.sum(masks["all"])
        except:
            print(f"No files in {tdir} for {sample}")

    if "all" in nums.keys():
        num = nums['all']
        print(f"{sample}: num events {num}")
