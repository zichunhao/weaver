## Plotting

- For plotting with one single file (bkg and signal merged):
```
python plot_classification_fromoutput.py \
       --ifile ../output/03_15_pyg_ef_nn_cw_8_2.root \
       --odir plots/03_15_pyg_ef_nn_cw_8_2/ \
       --name PNpyg \
       --mbranch fj_genRes_mass \
       --jet AK15 \
       --signals hww_4q_merged \
       --bkgs qcd \
       -v
```

- For plotting with different sample names:
```
python plot_classification_fromoutput.py \
       --ifile ../output/03_15_pyg_ef_nn_cw_8_2_sep_samples/qcd.root \
       --odir plots/03_15_pyg_ef_nn_cw_8_2_samples/ \
       --name PNpyg \
       --mbranch fj_genRes_mass \
       --jet AK15 \
       --signals hww_4q_merged \
       --bkgs qcd \
       -v \
       --isig ../output/03_15_pyg_ef_nn_cw_8_2_sep_samples/jhu_HHbbWW.root,../output/03_15_pyg_ef_nn_cw_8_2_sep_samples/bulkg_hflat.root,../output/03_15_pyg_ef_nn_cw_8_2_sep_samples/HHbbVV.root \
       --isignames JHUbbWW,bulkGFlat,PythiaHHbbVV
```