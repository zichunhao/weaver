import uproot
import awkward as ak
import numpy as np
import json
import os
import subprocess

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector as cvector

from copy import copy
import mplhep as hep
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors


NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
    "FatJetAK15_nConstituents",
    "JetPFCandsAK15",
)
NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
NanoAODSchema.mixins["PFCands"] = "PFCand"


def to_image(
    jet: np.ndarray, im_size: int, mask: np.ndarray = None, maxR: float = 1.0
) -> np.ndarray:
    """
    Convert a single jet into a 2D ``im_size`` x ``im_size`` array.
    Args:
        jet (np.ndarray): array of a single jet of shape ``[num_particles, num_features]``
          with features in order ``[eta, phi, pt]``.
        im_size (int): number of pixels per row and column.
        mask (np.ndarray): optional binary array of masks of shape ``[num_particles]``.
        maxR (float): max radius of the jet. Defaults to 1.0.
    Returns:
        np.ndarray: 2D array of shape ``[im_size, im_size]``.
    """
    assert len(jet.shape) == 2, "jets dimensions are incorrect"
    assert jet.shape[-1] >= 3, "particle feature format is incorrect"

    bins = np.linspace(-maxR, maxR, im_size + 1)
    binned_eta = np.digitize(jet[:, 0], bins) - 1
    binned_phi = np.digitize(jet[:, 1], bins) - 1
    pt = jet[:, 2]

    if mask is not None:
        assert len(mask.shape) == 1 and mask.shape[0] == jet.shape[0], "mask format incorrect"
        mask = mask.astype(int)
        pt *= mask

    jet_image = np.zeros((im_size, im_size))

    for eta, phi, pt in zip(binned_eta, binned_phi, pt):
        if eta >= 0 and eta < im_size and phi >= 0 and phi < im_size:
            jet_image[phi, eta] += pt

    return jet_image


plot_dir = "nobackup/plots_pfnano_v1/"
sample_dir = "/store/user/lpcpfnano/cmantill/v2_2/2017/HWWPrivate/"

os.system(f"mkdir -p {plot_dir}")

samples = {
    # label     : [name,         selector to get subset of files (~20 each)]
    "JHU_HHbbWW": ["jhu_HHbbWW", "miniaod_20ul_52"],
    "JHU_HH4W": ["GluGluToBulkGravitonToHHTo4W_JHUGen_M-2500_narrow", "miniaod_20ul_35914-9"],
    # "HHbbVV": ["GluGluToHHTobbVV_node_cHHH1_TuneCP5_13TeV-powheg-pythia8", "nano_mc2017_1-"],
}


eosbase = "root://cmseos.fnal.gov/"

pt_min = 300
pt_max = 400

events_dict = {}

# sample = "JHU_HHbbWW"
# miniaoddir, sel = "jhu_HHbbWW", "miniaod_20ul_5179"

for sample, (miniaoddir, sel) in samples.items():
    fileset = (
        subprocess.check_output(f"eos {eosbase} ls {sample_dir}/{miniaoddir}/", shell=True)
        .decode("utf-8")
        .split("\n")[:-1]
    )
    files = [f"{eosbase}{sample_dir}{miniaoddir}/{f}:Events" for f in fileset if sel in f]
    print(files)
    open_files = uproot.concatenate(files)
    events = NanoEventsFactory.from_root(
        open_files, schemaclass=NanoAODSchema, entry_stop=10000
    ).events()
    events_dict[sample] = events


jet_feats = {}

for sample, events in events_dict.items():
    print(sample)

    # sample = "JHU_HH4W"
    # events = events_dict[sample]

    jet_feats[sample] = {}
    # select jet - match it to H(VV)(4q)
    dR = 1
    jet_idx = 0

    jet_col = ak.pad_none(events.FatJetAK15, 2, axis=1)[:, jet_idx : jet_idx + 1]
    pfcands_col = events.FatJetAK15PFCands

    Z_PDGID = 23
    W_PDGID = 24
    HIGGS_PDGID = 25
    HIGGS_FLAGS = ["fromHardProcess", "isLastCopy"]
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(HIGGS_FLAGS)
    ]
    higgs = higgs[ak.all(abs(higgs.children.pdgId) == W_PDGID, axis=2)]

    j_matched, j_dR = higgs.nearest(jet_col, axis=1, return_metric=True, threshold=dR)

    idx_sel = pfcands_col.jetIdx == jet_idx
    jet_pfcands = events.PFCands[pfcands_col.pFCandsIdx[idx_sel]]

    mask_matched = ak.fill_none(
        (j_dR < dR) & (j_matched.pt > pt_min) & (j_matched.pt < pt_max), False
    )
    jets = ak.pad_none(j_matched[mask_matched], 1, axis=1)

    mask_matched = ak.values_astype(
        ak.sum(
            ak.fill_none((j_dR < dR) & (j_matched.pt > pt_min) & (j_matched.pt < pt_max), False),
            axis=1,
        ),
        bool,
    )

    eta_sign = ak.values_astype(jet_pfcands.eta > 0, int) * 2 - 1
    jet_feats[sample]["pfcand_etarel"] = (eta_sign * (jet_pfcands.eta - ak.flatten(jets.eta)))[
        mask_matched
    ]
    jet_feats[sample]["pfcand_phirel"] = jet_pfcands.delta_phi(ak.flatten(jets))[mask_matched]
    jet_feats[sample]["pfcand_pt_log_nopuppi"] = np.log(jet_pfcands.pt)[mask_matched]


average_images = {}


def plot_images(plot_dir, events_dict, samples):
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

    def event_to_image(events, i, maxR, im_size):
        num_parts = np.sum(events["pfcand_etarel"][i] != 0)
        return to_image(
            np.vstack(
                (
                    events["pfcand_etarel"][i].to_numpy()[:num_parts],
                    events["pfcand_phirel"][i].to_numpy()[:num_parts],
                    np.exp(events["pfcand_pt_log_nopuppi"][i].to_numpy()[:num_parts]),
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
        axes[j][0].annotate(
            f"{sample}\nPFNano",
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
                event_to_image(events, i, maxR=maxR, im_size=im_size),
                cmap=cm,
                interpolation="nearest",
                extent=[-maxR, maxR, -maxR, maxR],
                norm=colors.LogNorm(vmin, vmax),
            )

        # average jet image
        if sample not in average_images:
            num_ave_ims = 50
            ave_im = np.zeros((ave_im_size, ave_im_size))
            for i in tqdm(range(num_ave_ims)):
                ave_im += event_to_image(events, i, maxR=ave_maxR, im_size=ave_im_size)

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


plot_images(plot_dir, jet_feats, samples)
