"""
Checks kinematics, resonances etc. particularly related to the gen Ws

Author: Raghav Kansal
"""

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({"font.size": 16})

plot_dir = "../../plots/inference_analysis/03_31_ak8/"

os.system(f"mkdir -p {plot_dir}")

samples = ["bulkg_hflat", "bulkg_hsm", "HHbbVV", "qcd"]

events_dict = {}

for sample in samples:
    print(sample)
    events_dict[sample] = uproot.open(f"../../inferences/03_31_ak8/{sample}.root:Events").arrays()


sigs = ["bulkg_hflat", "HHbbVV"]
for sig in sigs:
    events = events_dict[sig]

    plt.figure()
    _ = plt.hist(events["fj_mass"], np.linspace(0, 250, 100), histtype="step", label=f"All {sig}")
    _ = plt.hist(
        events["fj_mass"][events["fj_H_VV_4q_3q"] == 1],
        np.linspace(0, 250, 100),
        histtype="step",
        label="3-pronged",
    )
    _ = plt.hist(
        events["fj_mass"][events["fj_H_VV_4q_4q"] == 1],
        np.linspace(0, 250, 100),
        histtype="step",
        label="4-pronged",
    )
    plt.legend()
    plt.xlabel("Jet Mass (GeV)")
    plt.ylabel("# Jets (GeV)")
    plt.savefig(f"{plot_dir}/{sig}_fj_mass.pdf")

sig = "bulkg_hflat"
events = events_dict[sig]

mass_range = list(range(30, 251, 10))

plt.figure(figsize=(12, 12))
for i in range(len(mass_range) // 3):
    mmin = mass_range[i * 3]
    mmax = mass_range[i * 3 + 3]
    _ = plt.hist(
        events["fj_mass"][(events["fj_genRes_mass"] >= mmin) * (events["fj_genRes_mass"] < mmax)],
        np.linspace(0, 300, 100),
        histtype="step",
        label=f"{mmin} ≤ mH < {mmax} GeV",
    )
plt.legend()
plt.xlabel("Jet Mass (GeV)")
plt.ylabel("# Jets (GeV)")
plt.savefig(f"{plot_dir}/{sig}_fj_mass_by_mh.pdf")

for j in range(2):
    plt.figure(figsize=(12, 12))
    for i in range(len(mass_range) // 3):
        mmin = mass_range[i * 3]
        mmax = mass_range[i * 3 + 3]
        _ = plt.hist(
            events[f"fj_subjet{j+1}_mass"][
                (events["fj_genRes_mass"] >= mmin) * (events["fj_genRes_mass"] < mmax)
            ],
            np.linspace(0, 200 / (j + 1), 100),
            histtype="step",
            label=f"{mmin} ≤ mH < {mmax} GeV",
        )
    plt.legend()
    plt.xlabel(f"Subjet {j + 1} Mass (GeV)")
    plt.ylabel("# Jets (GeV)")
    plt.savefig(f"{plot_dir}/{sig}_fj_subjet{j+1}_mass_by_mh.pdf")

# plt.hist(events["fj_genRes_mass"], np.linspace(0, 250, 100), histtype="step")


sig = "HHbbVV"
events = events_dict[sig]
for i in range(2):
    _ = plt.hist(
        events[f"fj_subjet{i + 1}_mass"],
        np.linspace(0, 100, 100),
        histtype="step",
        label=f"All {sig} Subjet {i + 1}",
    )
    _ = plt.hist(
        events[f"fj_subjet{i + 1}_mass"][events["fj_H_VV_4q_3q"] == 1],
        np.linspace(0, 100, 100),
        histtype="step",
        label=f"3-pronged Subjet {i + 1}",
    )
    _ = plt.hist(
        events[f"fj_subjet{i + 1}_mass"][events["fj_H_VV_4q_4q"] == 1],
        np.linspace(0, 100, 100),
        histtype="step",
        label=f"4-pronged Subjet {i + 1}",
    )
plt.legend()
plt.xlabel("Subjet Mass (GeV)")
plt.ylabel("# Subjets (GeV)")
plt.savefig(f"{plot_dir}/{sig}_subjet_mass.pdf")


plt.hist(events["score_fj_H_VV_4q_4q"])

sig = "qcd"
events = events_dict[sig]

for qc in ["3q", "4q"]:
    for i in range(2):
        plt.figure(figsize=(12, 12))
        _ = plt.hist(
            events[f"fj_subjet{i + 1}_mass"],
            np.linspace(0, 100, 100),
            histtype="step",
            label=f"All {sig} Subjet {i + 1}",
            density=True,
        )
        cuts = [0.8, 0.95]
        for cut in cuts:
            _ = plt.hist(
                events[f"fj_subjet{i + 1}_mass"][events[f"score_fj_H_VV_4q_{qc}"] > cut],
                np.linspace(0, 100, 100),
                histtype="step",
                label=f"{sig} {qc} score > {cut} Subjet {i + 1}",
                density=True,
            )

        plt.legend()
        plt.xlabel("Subjet Mass (GeV)")
        plt.title(f"QCD Subjet {i + 1} with {qc} cuts")
        # plt.ylabel("# Subjets (GeV)")
        plt.savefig(f"{plot_dir}/{sig}_subjet{i + 1}_mass_{qc}_cuts.pdf")


events.fields
