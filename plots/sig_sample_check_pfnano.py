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


plot_dir = "nobackup/plots_pfnano_costhetastar/"
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
    files = [f"{eosbase}{sample_dir}{miniaoddir}/{f}" for f in fileset if sel in f]
    print(files)
    # open_files = uproot.concatenate(files[0])
    events = NanoEventsFactory.from_root(
        files[0], schemaclass=NanoAODSchema, entry_stop=10000
    ).events()
    events_dict[sample] = events


events_dict


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
#
#
# from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray
# from typing import Union
#
# from coffea.nanoevents.methods import vector
#
# from energyflow import EFPSet
#
# ak.behavior.update(vector.behavior)
#
#
# B_PDGID = 5
# Z_PDGID = 23
# W_PDGID = 24
# HIGGS_PDGID = 25
# ELE_PDGID = 11
# MU_PDGID = 13
# TAU_PDGID = 15
# B_PDGID = 5
#
# GEN_FLAGS = ["fromHardProcess", "isLastCopy"]
#
#
# def get_pid_mask(
#     genparts: GenParticleArray,
#     pdgids: Union[int, list],
#     ax: int = 2,
#     byall: bool = True,
#     use_abs: bool = True,
# ) -> ak.Array:
#     """
#     Get selection mask for gen particles matching any of the pdgIds in ``pdgids``.
#     If ``byall``, checks all particles along axis ``ax`` match.
#     """
#     if use_abs:
#         gen_pdgids = abs(genparts.pdgId)
#     else:
#         gen_pdgids = genparts.pdgId
#
#     if type(pdgids) == list:
#         mask = gen_pdgids == pdgids[0]
#         for pdgid in pdgids[1:]:
#             mask = mask | (gen_pdgids == pdgid)
#     else:
#         mask = gen_pdgids == pdgids
#
#     return ak.all(mask, axis=ax) if byall else mask
#
#
# sample = "JHU_HHbbWW"
# events = events_dict[sample]
#
# events.fields
#
# genparts = events.GenPart
#
#
# higgs = genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]
#
# children_mask = get_pid_mask(higgs.children, [W_PDGID, Z_PDGID], byall=False, use_abs=False)
# higgs_children_mask = get_pid_mask(higgs.children, [W_PDGID, Z_PDGID], ax=2, byall=True)
#
#
# higgs_children = higgs.children[children_mask]
#
# higgs = ak.flatten(higgs[higgs_children_mask])
# higgs_vs = ak.flatten(ak.flatten(higgs_children, axis=2))
#
#
# hvecs = ak.zip(
#     {
#         "pt": higgs.pt,
#         "eta": higgs.eta,
#         "phi": higgs.phi,
#         "mass": higgs.mass,
#     },
#     with_name="PtEtaPhiMLorentzVector",
# )
#
# hvecs.boostvec
#
# hvvecs = ak.zip(
#     {
#         "pt": higgs_vs.pt,
#         "eta": higgs_vs.eta,
#         "phi": higgs_vs.phi,
#         "mass": higgs_vs.mass,
#     },
#     with_name="PtEtaPhiMLorentzVector",
# )
#
# hvecs.boost(-hvvecs.boostvec)
#
# costhetastar = np.cos(self.tvector(v1_inH.x, v1_inH.y, v1_inH.z).theta)
# out["hvv_costhetastar"] = pad_val(costhetastar, -10)
#
#
#
#
# higgs_vs.phi
#
# higgs_vs.boost(higgs)
#
#
# hcboost = higgs_children.boost(higgs)
#
#
# v = higgs_children[0][1][0]
# h = higgs[0][1]
# v.boost(h)
#
# hcboost[1]
#
# [0][0]
#
#
# higgs_children
#
#
# children_mass = matched_higgs_children.mass
#
# if "VV" in decays:
#     # select only VV children
#
#     # select lower mass child as V* and higher as V
#     v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=2, keepdims=True)])
#     v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=2, keepdims=True)])
#
#     genVars["fj_dR_V"] = fatjets.delta_r(v)
#     genVars["fj_dR_Vstar"] = fatjets.delta_r(v_star)
#
#     # select event only if VV are within jet radius
#     matched_Vs_mask = ak.any(fatjets.delta_r(v) < jet_dR, axis=1) & ak.any(
#         fatjets.delta_r(v_star) < jet_dR, axis=1
#     )
#
#     # I think this will find all VV daughters - not just the ones from the Higgs matched to the fatjet? (e.g. for HH4W it'll find all 8 daughters?)
#     # daughter_mask = get_pid_mask(genparts.distinctParent, [W_PDGID, Z_PDGID], ax=1, byall=False)
#     # daughters = genparts[daughter_mask & genparts.hasFlags(GEN_FLAGS)]
#
#     # get VV daughters
#     daughters = ak.flatten(ak.flatten(matched_higgs_children.distinctChildren, axis=2), axis=2)
#     daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
#     daughters_pdgId = abs(daughters.pdgId)
#
#     nprongs = ak.sum(fatjets.delta_r(daughters) < jet_dR, axis=1)
#
#     # why checking pT > 0?
#     decay = (
#         # 2 quarks * 1
#         (ak.sum(daughters_pdgId <= B_PDGID, axis=1) == 2) * 1
#         # 1 electron * 3
#         + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
#         # 1 muon * 5
#         + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
#         # 1 tau * 7
#         + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
#         # 4 quarks * 11
#         + (ak.sum(daughters_pdgId <= B_PDGID, axis=1) == 4) * 11
#     )
#
#     matched_mask = matched_higgs_mask & matched_Vs_mask
#
#     genVVars = {f"fj_genV_{key}": ak.fill_none(v[var], -99999) for (var, key) in P4.items()}
#     genVstarVars = {
#         f"fj_genVstar_{key}": ak.fill_none(v_star[var], -99999) for (var, key) in P4.items()
#     }
#     genLabelVars = {
#         "fj_nprongs": nprongs,
#         "fj_H_VV_4q": to_label(decay == 11),
#         "fj_H_VV_elenuqq": to_label(decay == 4),
#         "fj_H_VV_munuqq": to_label(decay == 6),
#         "fj_H_VV_taunuqq": to_label(decay == 8),
#         "fj_H_VV_unmatched": to_label(~matched_mask),
#     }
#     genVars = {**genVars, **genVVars, **genVstarVars, **genLabelVars}
#
#
# hvv_vec = self.cvector(hboson_vv.pt, hboson_vv.eta, hboson_vv.phi, hboson_vv.mass)
# v1_vec = self.cvector(
#     vboson_onshell.pt, vboson_onshell.eta, vboson_onshell.phi, vboson_onshell.mass
# )
# v2_vec = self.cvector(
#     vboson_offshell.pt, vboson_offshell.eta, vboson_offshell.phi, vboson_offshell.mass
# )
#
# v1_vec_1 = self.cvector(
#     vboson_onshell.children[:, 0].pt,
#     vboson_onshell.children[:, 0].eta,
#     vboson_onshell.children[:, 0].phi,
#     vboson_onshell.children[:, 0].mass,
# )
# v1_vec_2 = self.cvector(
#     vboson_onshell.children[:, 1].pt,
#     vboson_onshell.children[:, 1].eta,
#     vboson_onshell.children[:, 1].phi,
#     vboson_onshell.children[:, 1].mass,
# )
# v2_vec_1 = self.cvector(
#     vboson_offshell.children[:, 0].pt,
#     vboson_offshell.children[:, 0].eta,
#     vboson_offshell.children[:, 0].phi,
#     vboson_offshell.children[:, 0].mass,
# )
# v2_vec_2 = self.cvector(
#     vboson_offshell.children[:, 1].pt,
#     vboson_offshell.children[:, 1].eta,
#     vboson_offshell.children[:, 1].phi,
#     vboson_offshell.children[:, 1].mass,
# )
#
# # angles
# # cos theta* (Angle between H and W)
# # phi1, costheta1, costheta2, phi
# boostH = -(hvv_vec.boostvec)
# v1_inH = v1_vec.boost(boostH)[:, 0]
# v2_inH = v2_vec.boost(boostH)[:, 0]
# costhetastar = np.cos(self.tvector(v1_inH.x, v1_inH.y, v1_inH.z).theta)
# out["hvv_costhetastar"] = pad_val(costhetastar, -10)
