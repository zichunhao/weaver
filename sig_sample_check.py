import uproot
import awkward as ak
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from jetnet.utils import to_image
from copy import copy
import mplhep as hep
from tqdm import tqdm
from matplotlib import colors

#
# plot_dir = "../plots/sample_checks/"
# sample_dir = "../sample_data/"


plot_dir = "/hwwtaggervol/plots/sample_checks/Mar9/"
sample_dir = "/hwwtaggervol/training/ak15_Feb14/test/"

os.system(f"mkdir -p {plot_dir}")

samples = {
    "JHU_HHbbWW": ["jhu_HHbbWW", "miniaod_20ul_51"],
    "JHU_HH4W": ["GluGluToBulkGravitonToHHTo4W_JHUGen_M-2500_narrow", "miniaod_20ul_35914-9"],
    "HHbbVV": ["GluGluToHHTobbVV_node_cHHH1_TuneCP5_13TeV-powheg-pythia8", "nano_mc2017_1-"],
}

events_dict = {}

for sample, (dir, sel) in samples.items():
    print(sample)
    events_dict[sample] = uproot.concatenate(f"{sample_dir}/{dir}/{sel}*.root:Events")
    events_dict[sample]["npfcands"] = np.sum(
        events_dict[sample]["pfcand_etarel"] != 0, axis=1, keepdims=True
    )

for sample, (dir, sel) in samples.items():
    events_dict[sample]["npfcands"] = np.sum(
        events_dict[sample]["pfcand_etarel"] != 0, axis=1, keepdims=True
    )
    events_dict[sample]["nsvs"] = np.sum(events_dict[sample]["sv_mass"] != 0, axis=1, keepdims=True)

# ak.values_astype(masks[sample], bool)
# events_dict[sample][masks[sample]]
events_dict[sample].fields


events_dict[sample]["fj_H_WW_4q"]
events_dict[sample]["fj_genW_pt"]
events_dict[sample]["fj_genWstar_pt"]
events_dict[sample]["fj_genWstar_eta"]
events_dict[sample]["fj_eta"]

pt_min = 300
pt_max = 400

masks = {}

for sample, events in events_dict.items():
    masks[sample] = ak.values_astype(
        (events["fj_pt"] > pt_min) * (events["fj_pt"] < pt_max) * (events["fj_H_WW_4q"]), bool
    )

features = {
    "npfcands": [0, 100],
    "nsvs": [0, 10],
    "pfcand_pt_log_nopuppi": [-2, 5],
    "pfcand_e_log_nopuppi": [-2, 5],
    "pfcand_etarel": [-2, 2],
    "pfcand_phirel": [-2, 2],
    "pfcand_isEl": [0, 1],
    "pfcand_isMu": [0, 1],
    "pfcand_isGamma": [0, 1],
    "pfcand_isChargedHad": [0, 1],
    "pfcand_isNeutralHad": [0, 1],
    "pfcand_abseta": [0, 2.5],
    "pfcand_charge": [-1, 1],
    "pfcand_VTX_ass": [0, 7],
    "pfcand_lostInnerHits": [-1, 2],
    "pfcand_normchi2": [0, 100],
    "pfcand_quality": [0, 5],
    "pfcand_dz": [-15, 15],
    "pfcand_dzsig": [-100, 100],
    "pfcand_dxy": [-2, 2],
    "pfcand_dxysig": [-50, 50],
    "sv_pt_log": [-1, 5],
    "sv_mass": [0, 3],
    "sv_etarel": [-0.5, 0.5],
    "sv_phirel": [-0.4, 0.4],
    "sv_abseta": [0, 3],
    "sv_ntracks": [0, 7],
    "sv_normchi2": [0, 10],
    "sv_dxy": [0, 7],
    "sv_dxysig": [0, 20],
    "sv_d3d": [0, 5],
    "sv_d3dsig": [0, 20],
}

#
# # tagger_vars["sv_features"]["var_names"]
#
# with open("../models/pyg_ef_ul_cw_8_2_preprocess.json") as f:
#     tagger_vars = json.load(f)
#
# tagger_vars

for var, bins in features.items():
    print(var)
    for sample, events in events_dict.items():
        _ = plt.hist(
            ak.flatten(events[var][masks[sample]]),
            bins=np.linspace(bins[0], bins[1], 31),
            histtype="step",
            density=True,
            label=sample,
        )
    plt.legend()
    plt.xlabel(var)
    plt.savefig(f"{plot_dir}/{var}.pdf")
    plt.close()


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
    return to_image(
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
    figsize=(74, 30),
    gridspec_kw={"wspace": 0.25, "hspace": 0},
)

vmin = 1e-2
vmax = 200

for j, (sample, events) in enumerate(events_dict.items()):
    events = events[masks[sample]]

    axes[j][0].annotate(
        sample,
        xy=(0, -1),
        xytext=(-axes[j][0].yaxis.labelpad - 15, 0),
        xycoords=axes[j][0].yaxis.label,
        textcoords="offset points",
        ha="right",
        va="center",
    )

    for i in range(num_images):
        im = axes[j][i].imshow(
            event_to_image(events[i], maxR=maxR, im_size=im_size),
            cmap=cm,
            interpolation="nearest",
            extent=[-maxR, maxR, -maxR, maxR],
            norm=colors.LogNorm(vmin, vmax),
        )
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
        # axes[j][i].plot(
        #     events[i]["fj_genW_phi"],  # - events[i]["fj_phi"],
        #     events[i]["fj_genW_eta"],  # - events[i]["fj_eta"],
        #     "+",
        #     color="brown",
        #     ms=30,
        #     mew=2,
        # )
        # axes[j][i].plot(
        #     events[i]["fj_genWstar_phi"],  # - events[i]["fj_phi"],
        #     events[i]["fj_genWstar_eta"],  # - events[i]["fj_eta"],
        #     "b+",
        #     ms=30,
        #     mew=2,
        # )
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


# sample = "JHU_HH4W"
# events = events_dict[sample][masks[sample]]
#
# i = 0
#
# plt.imshow(
#     event_to_image(events[i], maxR=maxR, im_size=im_size),
#     cmap=cm,
#     interpolation="nearest",
#     extent=[-maxR, maxR, -maxR, maxR],
#     norm=colors.LogNorm(vmin, vmax),
# )
# # plot Ws
# plt.plot(
#     events[i]["fj_genW_phi"] - events[i]["fj_phi"],
#     events[i]["fj_genW_eta"] - events[i]["fj_eta"],
#     "+",
#     color="black",
#     ms=30,
#     mew=2,
# )
# plt.plot(
#     events[i]["fj_genWstar_phi"] - events[i]["fj_phi"],
#     events[i]["fj_genWstar_eta"] - events[i]["fj_eta"],
#     "b+",
#     ms=30,
#     mew=2,
# )
