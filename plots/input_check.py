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
    nums = {}
    for tdir in ["train","test"]:
        print(f"{sample_dir}/{tdir}/{dir}*/{sel}*.root:{branch}")
        ibranches = ["fj_pt"]
        if "HH" in sample:
            ibranches += ["fj_H_VV_4q","fj_nprongs"]
        if "QCD" in sample:
            ibranches += ["fj_isQCDb",
                          "fj_isQCDbb",
                          "fj_isQCDc",
                          "fj_isQCDcc",
                          "fj_isQCDlep",
                          "fj_isQCDothers"]
        try:
            events = uproot.iterate(f"{sample_dir}/{tdir}/{dir}*/{sel}*.root:{branch}",ibranches)
            for ev in events:
                masks = {}
                masks["all"] = (ev["fj_pt"] > 0)
                print(masks)
                if "HH" in sample:
                    masks["4q_all"] = masks["all"] & (ev["fj_H_VV_4q"] == 1)
                    masks["4q"] = masks["4q_all"] & (ev["fj_nprongs"] == 4)
                    masks["3q"] = masks["3q_all"] & (ev["fj_nprongs"] == 3)
                print(masks)
                if "QCD" in sample:
                    masks["qcdb"] = masks["all"] & (ev["fj_isQCDb"]==1)
                    masks["qcdbb"] = masks["all"] & (ev["fj_isQCDbb"]==1)
                    masks["qcdc"] = masks["all"] & (ev["fj_isQCDc"]==1)
                    masks["qcdcc"] = masks["all"] & (ev["fj_isQCDcc"]==1)
                    masks["qcdlep"] = masks["all"] & (ev["fj_isQCDlep"]==1)
                    masks["qcdoth"] = masks["all"] & (ev["fj_isQCDothers"]==1)
                    masks["all_tagged"] =  masks["qcdb"] | masks["qcdbb"] | masks["qcdc"] | masks["qcdcc"] | masks["qcdlep"] | masks["qcdoth"]
                for m,mask in masks.items():
                    print(m,ak.sum(mask))
                    if m in nums.keys():
                        nums[m] += ak.sum(mask)
                    else:
                        nums[m] = ak.sum(mask)
        except:
            print(f"No files in {tdir} for {sample}")

    if "all" in nums.keys():
        num = nums['all']
        print(f"{sample}: num events {num}")
