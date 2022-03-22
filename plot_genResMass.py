import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


file = "../inferences/03_15_pyg_ef_nn_cw_8_2.root"

events = uproot.open(f"{file}:Events").arrays()

plt.title("Gen Resonance Distribution")
_ = plt.hist(
    events["fj_genRes_mass"][(events["fj_genRes_mass"] > 0) * (events["fj_genRes_mass"] != 125)],
    np.linspace(0, 250, 101),
    histtype="step",
)
plt.xlabel("Gen Resonance Mass (GeV)")
plt.ylabel("# Jets")
plt.savefig("../plots/genresmass.pdf")
