import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

events = uproot.open("../inferences/03_15_pyg_ef_nn_cw_8_2.root:Events").arrays()

events

events.fields


scores = [
    "fj_isQCDb",
    "score_fj_isQCDb",
    "fj_isQCDbb",
    "score_fj_isQCDbb",
    "fj_isQCDc",
    "score_fj_isQCDc",
    "fj_isQCDcc",
    "score_fj_isQCDcc",
    "fj_isQCDlep",
    "score_fj_isQCDlep",
    "fj_isQCDothers",
    "score_fj_isQCDothers",
    "fj_H_VV_4q_3q",
    "score_fj_H_VV_4q_3q",
    "fj_H_VV_4q_4q",
    "score_fj_H_VV_4q_4q",
]


for score in scores:
    if score.startswith("score"):
        plt.hist(events[score], label=score, histtype="step", bins=np.linspace(0, 1, 101))

plt.legend()


for score in scores:
    if not score.startswith("score"):
        plt.hist(events[score], label=score, histtype="step", bins=np.linspace(0, 1, 101))

plt.legend()
