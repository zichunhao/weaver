import uproot
import awkward as ak
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import jetnet
from copy import copy
import mplhep as hep
from tqdm import tqdm
from matplotlib import colors
from PyPDF2 import PdfFileMerger

# plot_dir = "../plots/sample_checks/"
# sample_dir = "../sample_data/"

plot_dir = "/hwwtaggervol/plots/sample_checks/Mar11_HWW/"
sample_dir = "/hwwtaggervol/training/ak15_Feb14/test/"

os.system(f"mkdir -p {plot_dir}")

samples = {
    # label     : [name,         selector to get subset of files (~20 each)]
    "JHU_HHbbWW": ["jhu_HHbbWW", "miniaod_20ul_51"],
    "JHU_HH4W": ["GluGluToBulkGravitonToHHTo4W_JHUGen_M-2500_narrow", "miniaod_20ul_35914-9"],
    "HHbbVV": ["GluGluToHHTobbVV_node_cHHH1_TuneCP5_13TeV-powheg-pythia8", "nano_mc2017_1-"],
    "DNN_JHU_HHbbWW": ["jhu_HHbbWW_DNN", "miniaod_20ul_52"],
    "DNN_JHU_HHbbWW_AK15": ["jhu_HHbbWW_DNN_ak15", "miniaod_52"],
    # "HWW": ["GluGluHToWWTo4q_M-125_TuneCP5_13TeV-powheg-pythia8", "nano_mc2017_1??_"],
}

colours = {
    "JHU_HHbbWW": "blue",
    "JHU_HH4W": "orange",
    "HHbbVV": "green",
    "DNN_JHU_HHbbWW": "red",
    "DNN_JHU_HHbbWW_AK15": "pink",
    "HWW": "purple",
}

events_dict = {}

for sample, (dir, sel) in samples.items():
    print(sample)
    if sample not in events_dict:
        branch = "deepntuplizer/tree" if "DNN" in sample else "Events"
        events_dict[sample] = uproot.concatenate(f"{sample_dir}/{dir}/{sel}*.root:{branch}")


sample = "JHU_HHbbWW"
events_dict[sample].fields

file = uproot.open("/hwwtaggervol/training/ak15_Feb14/test//jhu_HHbbWW_DNN/miniaod_20ul_51.root")
file = uproot.open("../sample_data/jhu_HHbbWW/miniaod_20ul_5001_Skim.root:Events")
file.keys()

file["deepntuplizer/tree"].keys()

pfcand_masks = {}
sv_masks = {}

# n pfcands, n svs
for sample, (dir, sel) in samples.items():
    pfcand_masks[sample] = events_dict[sample]["pfcand_pt_log_nopuppi"] != 0
    sv_masks[sample] = events_dict[sample]["sv_pt_log"] != 0
    events_dict[sample]["npfcands"] = np.sum(pfcand_masks[sample], axis=1)
    events_dict[sample]["nsvs"] = np.sum(sv_masks[sample], axis=1)


# get pT and WW_4q mask
pt_min = 300
pt_max = 400

masks = {}

for sample, events in events_dict.items():
    ww4q_label = "label_H_ww4q" if "DNN" in sample else "fj_H_WW_4q"
    masks[sample] = ak.values_astype(
        (events["fj_pt"] > pt_min) * (events["fj_pt"] < pt_max) * (events[ww4q_label]), bool
    )

masks

# features to plot and plot ranges
features = {
    "npfcands": [0, 100],
    "nsvs": [0, 10],
    "pfcand_pt_log_nopuppi": [-2, 5],
    "pfcand_e_log_nopuppi": [-2, 5],
    "pfcand_etarel": [-3.5, 3.5],
    "pfcand_phirel": [-3.5, 3.5],
    "pfcand_isEl": [0, 1],
    "pfcand_isMu": [0, 1],
    "pfcand_isGamma": [0, 1],
    "pfcand_isChargedHad": [0, 1],
    "pfcand_isNeutralHad": [0, 1],
    "pfcand_abseta": [0, 2.5],
    "pfcand_charge": [-1, 1],
    "pfcand_VTX_ass": [0, 7],
    "pfcand_lostInnerHits": [-1, 2],
    "pfcand_normchi2": [0, 10],
    "pfcand_quality": [0, 5],
    "pfcand_dz": [-15, 15],
    "pfcand_dzsig": [-50, 50],
    "pfcand_dxy": [-2, 2],
    "pfcand_dxysig": [-40, 40],
    "sv_pt_log": [-0.5, 6],
    "sv_mass": [0, 4],
    "sv_etarel": [-0.8, 0.8],
    "sv_phirel": [-0.8, 0.8],
    "sv_abseta": [0, 3],
    "sv_ntracks": [0, 7],
    "sv_normchi2": [0, 10],
    "sv_dxy": [0, 7],
    "sv_dxysig": [0, 30],
    "sv_d3d": [0, 5],
    "sv_d3dsig": [0, 20],
    "fj_dR_W": [0, 1.5],
    "fj_genW_pt": [0, 100],
    "fj_genW_eta": [-3.5, 3.5],
    "fj_genW_phi": [-3.5, 3.5],
    "fj_genW_mass": [0, 100],
    "fj_dR_Wstar": [0, 1.5],
    "fj_genWstar_pt": [0, 100],
    "fj_genWstar_eta": [-3.5, 3.5],
    "fj_genWstar_phi": [-3.5, 3.5],
    "fj_genWstar_mass": [0, 100],
    "fj_mindR_HWW_daus": [0, 4],
    "fj_maxdR_HWW_daus": [0, 4],
    "fj_maxdR_Hbb_daus": [0, 4],
}

# with open("../models/pyg_ef_ul_cw_8_2_preprocess.json") as f:
#     tagger_vars = json.load(f)

merger_inputs = PdfFileMerger()
merger_ws = PdfFileMerger()

for var, bins in features.items():
    print(var)
    for sample, events in events_dict.items():
        if "DNN" in sample and var.startswith("fj"):
            continue
        vals = events[var][masks[sample]]
        if not (var.startswith("n") or var.startswith("fj")):
            feat_mask = pfcand_masks[sample] if var.startswith("pfcand") else sv_masks[sample]
            vals = ak.flatten(vals[feat_mask[masks[sample]]])

        _ = plt.hist(
            vals,
            bins=np.linspace(
                bins[0], bins[1], 15 if (var.startswith("sv") or var.startswith("fj")) else 25
            ),
            histtype="step",
            density=True,
            label=sample,
            color=colours[sample],
        )
    plt.legend()
    plt.xlabel(var)
    plt.savefig(f"{plot_dir}/{var}.pdf")
    plt.show()

    if var.startswith("fj"):
        merger_ws.append(f"{plot_dir}/{var}.pdf")
    else:
        merger_inputs.append(f"{plot_dir}/{var}.pdf")

merger_inputs.write(f"{plot_dir}/input_feature_plots.pdf")
merger_inputs.close()

merger_ws.write(f"{plot_dir}/w_feature_plots.pdf")
merger_ws.close()


# jet images

average_images = {}

cm = copy(plt.cm.jet)
cm.set_under(color="white")

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)


maxR = 3
im_size = 31
ave_im_size = 25
ave_maxR = 3


def del_phi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def event_to_image(event, maxR, im_size):
    num_parts = np.sum(event["pfcand_etarel"] != 0)
    return jetnet.utils.to_image(
        np.vstack(
            (
                event["pfcand_etarel"].to_numpy()[:num_parts],
                event["pfcand_phirel"].to_numpy()[:num_parts],
                np.exp(event["pfcand_pt_log_nopuppi"].to_numpy()[:num_parts]),
            )
        ).T,
        im_size,
        maxR=maxR,
    ).T


num_images = 6

fig, axes = plt.subplots(
    nrows=len(samples),
    ncols=num_images + 1,
    figsize=((num_images + 1) * 10, len(samples) * 10),
    gridspec_kw={"wspace": 0.25, "hspace": 0},
)

vmin = 1e-2
vmax = 200

for j, sample in enumerate(samples.keys()):
    events = events_dict[sample]
    events = events[masks[sample]]

    annotate_label = "PKU's" if "DNN" in sample else "Our"

    axes[j][0].annotate(
        f"{sample}\n{annotate_label} PFNanoNTuples",
        xy=(0, -1),
        xytext=(-axes[j][0].yaxis.labelpad - 15, 0),
        xycoords=axes[j][0].yaxis.label,
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=48,
    )

    for i in range(num_images):
        im = axes[j][i].imshow(
            event_to_image(events[i], maxR=maxR, im_size=im_size),
            cmap=cm,
            interpolation="nearest",
            extent=[-maxR, maxR, -maxR, maxR],
            norm=colors.LogNorm(vmin, vmax),
        )
        if "DNN" not in sample:
            # plot Ws
            axes[j][i].plot(
                del_phi(events[i]["fj_genW_phi"], events[i]["fj_phi"]),
                events[i]["fj_genW_eta"] - events[i]["fj_eta"],
                "+",
                color="brown",
                ms=30,
                mew=2,
            )
            axes[j][i].plot(
                del_phi(events[i]["fj_genWstar_phi"], events[i]["fj_phi"]),
                events[i]["fj_genWstar_eta"] - events[i]["fj_eta"],
                "b+",
                ms=30,
                mew=2,
            )
        axes[j][i].tick_params(which="both", bottom=False, top=False, left=False, right=False)
        axes[j][i].set_xlabel("$\phi^{rel}$")
        axes[j][i].set_ylabel("$\eta^{rel}$")

    # average jet image
    if sample not in average_images:
        num_ave_ims = 200
        ave_im = np.zeros((ave_im_size, ave_im_size))
        for i in tqdm(range(num_ave_ims)):
            ave_im += event_to_image(events[i], maxR=ave_maxR, im_size=ave_im_size)

        ave_im /= num_ave_ims
        average_images[sample] = ave_im

    im = axes[j][-1].imshow(
        average_images[sample],
        cmap=plt.cm.jet,
        interpolation="nearest",
        extent=[-ave_maxR, ave_maxR, -ave_maxR, ave_maxR],
        norm=colors.LogNorm(vmin, vmax),
    )
    axes[j][-1].set_title("Average Jet Image", pad=5)
    axes[j][-1].tick_params(which="both", bottom=False, top=False, left=False, right=False)
    axes[j][-1].set_xlabel("$\phi^{rel}$")
    axes[j][-1].set_ylabel("$\eta^{rel}$")

    cbar = fig.colorbar(im, ax=axes[j].ravel().tolist(), fraction=0.007)
    cbar.set_label("$p_T$ (GeV)")

# fig.tight_layout()
plt.savefig(f"{plot_dir}/jet_images.pdf", bbox_inches="tight")
plt.show()
