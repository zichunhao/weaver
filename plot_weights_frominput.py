import numpy as np
import uproot
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
import awkward

import mplhep as hep
hep.style.use("CMS")

# functions to make weights
def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)

# reproduce weight maker function
def make_weights(table,reweight_branches,reweight_bins,reweight_classes,reweight_method,class_weights,reweight_threshold):
    x_var, y_var = reweight_branches
    x_bins, y_bins = reweight_bins
    reweight_discard_under_overflow = False
    if not reweight_discard_under_overflow:
        # clip variables to be within bin ranges
        x_min, x_max = min(x_bins), max(x_bins)
        y_min, y_max = min(y_bins), max(y_bins)
        print(f'Clipping `{x_var}` to [{x_min}, {x_max}] to compute the shapes for reweighting.')
        print(f'Clipping `{y_var}` to [{y_min}, {y_max}] to compute the shapes for reweighting.')
        print(x_var,table[x_var])
        table[x_var] = np.clip(table[x_var].to_numpy(), min(x_bins), max(x_bins))
        table[y_var] = np.clip(table[y_var].to_numpy(), min(y_bins), max(y_bins))
    print('Using %d events to make weights', len(table[x_var]))

    sum_evts = 0
    max_weight = 0.9
    raw_hists = {}
    class_events = {}
    result = {}
    for label in reweight_classes:
        print('Making weight for class %s ',label)
        pos = (table[label] == 1)
        print('Number of jets for this class %d',len(table[x_var][pos]))
        x = table[x_var][pos]
        y = table[y_var][pos]
        
        hist, _, _ = np.histogram2d(x, y, bins=reweight_bins)
        print('%s:\n %s', label, str(hist.astype('int64')))
        sum_evts += hist.sum()
        raw_hists[label] = hist.astype('float32')
        result[label] = hist.astype('float32')
    if sum_evts != len(table[x_var]):
        print('Only %d (out of %d) events actually used in the reweighting. '
              'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
              '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
              sum_evts, len(table[x_var]))

    if reweight_method == 'flat':
        for label, classwgt in zip(reweight_classes, class_weights):
            hist = result[label]
            threshold_ = np.median(hist[hist > 0]) * 0.01
            nonzero_vals = hist[hist > threshold_]
            min_val, med_val = np.min(nonzero_vals), np.median(hist)  # not really used
            ref_val = np.percentile(nonzero_vals, reweight_threshold)
            print('label:%s, median=%f, min=%f, ref=%f, ref/min=%f' %
                  (label, med_val, min_val, ref_val, ref_val / min_val))
            # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
            wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
            result[label] = wgt
            # divide by classwgt here will effective increase the weight later
            class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
    elif self._data_config.reweight_method == 'ref':
        # use class 0 as the reference
        hist_ref = raw_hists[self._data_config.reweight_classes[0]]
        for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
            # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
            ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
            upper = np.percentile(ratio[ratio > 0], 100 - self._data_config.reweight_threshold)
            wgt = np.clip(ratio / upper, 0, 1)  # -> [0,1]
            result[label] = wgt
            # divide by classwgt here will effective increase the weight later
            class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
            
    # ''equalize'' all classes
    # multiply by max_weight (<1) to add some randomness in the sampling
    min_nevt = min(class_events.values()) * max_weight
    for label in reweight_classes:
        class_wgt = float(min_nevt) / class_events[label]
        result[label] *= class_wgt

    return result

# build weights from reweight histograms
def _build_weights(table, reweight_hists, reweight_branches, reweight_bins):
    x_var, y_var = reweight_branches
    x_bins, y_bins = reweight_bins
    rwgt_sel = None
    reweight_discard_under_overflow = False
    if reweight_discard_under_overflow:
        rwgt_sel = (table[x_var] >= min(x_bins)) & (table[x_var] <= max(x_bins)) & \
            (table[y_var] >= min(y_bins)) & (table[y_var] <= max(y_bins))
    # init w/ wgt=0: events not belonging to any class in `reweight_classes` will get a weight of 0 at the end
    wgt = np.zeros(len(table[x_var]), dtype='float32')
    sum_evts = 0
    for label, hist in reweight_hists.items():
        pos = table[label] == 1
        if rwgt_sel is not None:
            pos &= rwgt_sel
        try:
            rwgt_x_vals = table[x_var][pos].to_numpy()
            rwgt_y_vals = table[y_var][pos].to_numpy()
        except:
            rwgt_x_vals = table[x_var][pos]
            rwgt_y_vals = table[y_var][pos]
        x_indices = np.clip(np.digitize(
            rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
        y_indices = np.clip(np.digitize(
            rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
        wgt[pos] = hist[x_indices, y_indices]
        try:
            sum_evts += pos.to_numpy().sum()
        except:
            sum_evts += pos.sum()
    if sum_evts != len(table[x_var]):
        print(
            'Not all selected events used in the reweighting. '
            'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
            '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
        )
        table["weight"] = wgt


# get indices
def _get_reweight_indices(weights, up_sample=True, max_resample=10, weight_scale=1):
    all_indices = np.arange(len(weights))
    randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights))
    keep_flags = randwgt < weights
    if not up_sample:
        keep_indices = all_indices[keep_flags]
    else:
        n_repeats = len(weights) // max(1, int(keep_flags.sum()))
        if n_repeats > max_resample:
            n_repeats = max_resample
        all_indices = np.repeat(np.arange(len(weights)), n_repeats)
        randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights) * n_repeats)
        keep_indices = all_indices[randwgt < np.repeat(weights, n_repeats)]
    return keep_indices.copy()

# make plots
def make_plots(table,cats,reweight_bins,args):
    # define bins and output labels for each variable
    # [plotlabel,bins,xlabel]
    if args.regression:
        var_dict = {
            "fj_pt": ["pt",reweight_bins[0],r"$p_{T}$ [GeV]"],
            "fj_genjetmsd": ["genjetmsd",reweight_bins[1],r"GEN $m_{SD}$ [GeV]"],
            "target_mass": ["targetmass",reweight_bins[1],r"Target mass [GeV]"],
            "fj_genRes_mass": ["genmass",reweight_bins[1],r"Resonance mass [GeV]"],
        }
    else:
        var_dict = {
            "fj_pt": ["pt",reweight_bins[0],r"$p_{T}$ [GeV]"],
            "fj_msoftdrop": ["msoftdrop",reweight_bins[1],r"$m_{SD}$ [GeV]"],
        }

    for catlabel,cat in cats.items():
        for density in [True,False]:
            iax = 0
            fig, axs = plt.subplots(len(var_dict.keys()), 2)
            for var,varprop in var_dict.items():
                plotlabel = varprop[0]
                bins = varprop[1]
                xlabel = varprop[2]

                for label, hist in reweight_hists.items():
                    # print the labels to plot?
                    #print(label,cat)
                    if label in cat:
                        try:
                            x = table[var][table[label]==1].to_numpy()
                        except:
                            x = table[var][table[label]==1]
                        axs[iax,0].hist(x, bins=varprop[1], histtype='step', density=density, label=label)
                axs[iax,0].set_xlabel(xlabel)
                axs[iax,0].set_ylabel(r'Events')
                axs[iax,0].legend(prop={'size': 6})
                
                for label, hist in reweight_hists.items():
                    if label in cat:
                        try:
                            x = table[var][table[label]==1].to_numpy()
                        except:
                            x = table[var][table[label]==1]
                        axs[iax,1].hist(x, bins=varprop[1], histtype='step', density=density, label=label, weights=table['weight'][table[label]==1])
                axs[iax,1].set_xlabel(xlabel)
                axs[iax,1].set_ylabel(r'Weighted Events')
                axs[iax,1].legend(prop={'size': 6})

                iax+=1
            fig.tight_layout()
            if density:
                fig.savefig("%s/weights_%s.pdf"%(args.odir,catlabel))
            else:
                fig.savefig("%s/weights_%s_all.pdf"%(args.odir,catlabel))

def make_2d_plots(table,reweight_bins,args):
    if args.regression:
        var_dict = [
            ["fj_pt","pt",reweight_bins[0],r"$p_{T}$ [GeV]"],
            ["fj_genjetmsd","genjetmsd",reweight_bins[1],r"GEN $m_{SD}$ [GeV]"],
        ]
    else:
        var_dict = [
            ["fj_pt","pt",reweight_bins[0],r"$p_{T}$ [GeV]"],
            ["fj_msoftdrop","msoftdrop",reweight_bins[1],r"$m_{SD}$ [GeV]"],
        ]

    for catlabel,cat in cats.items():
        for density in [True,False]:
            iax = 0
            fig, axs = plt.subplots(2, len(cat), figsize=(len(cat)*8,16))

            var1 = var_dict[0][0]
            var2 = var_dict[1][0]
            
            for label, hist in reweight_hists.items():        
                if label in cat:
                    try:
                        x = table[var1][table[label]==1].to_numpy()
                        y = table[var2][table[label]==1].to_numpy()
                    except:
                        x = table[var1][table[label]==1]
                        y = table[var2][table[label]==1]

                    try:
                        axs_0 = axs[0,iax]
                        axs_1 = axs[1,iax]
                    except:
                        axs_0 = axs[0]
                        axs_1 = axs[1]
                    mp = axs_0.hist2d(x,y, bins=[reweight_bins[0],reweight_bins[1]], density=density, label=label, norm=colors.LogNorm())
                    axs_0.set_xlabel(var_dict[0][3])
                    axs_0.set_ylabel(var_dict[1][3])
                    cbar_0=fig.colorbar(mp[3],ax=axs_0) # hist2d returns counts, xedges, yedges, im - we want im
                    cbar_0.set_label('Jets')
                    axs_0.set_title(label)

                    mp = axs_1.hist2d(x,y, bins=[reweight_bins[0],reweight_bins[1]], density=density, label=label, weights=table['weight'][table[label]==1], norm=colors.LogNorm())
                    axs_1.set_xlabel(var_dict[0][3])
                    axs_1.set_ylabel(var_dict[1][3])
                    cbar_1=fig.colorbar(mp[3],ax=axs_1)
                    cbar_1.set_label('Weighted Jets')
                    axs_1.set_title(label)
                    iax+=1
                    
            fig.tight_layout()
            if density:
                fig.savefig("%s/2dweights_%s_density.pdf"%(args.odir,catlabel))
            else:
                fig.savefig("%s/2dweights_%s_all.pdf"%(args.odir,catlabel))
                
def create_table(events,branches):
    from collections import defaultdict
    table = defaultdict(list)
    for ev in events:
        for name in branches:
            table[name].append(ev[name])
    table = {name:_concat(arrs) for name, arrs in table.items()}
    return table
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--regression', action='store_true', default=False, help='regression mode')
    parser.add_argument('--weights', action='store_true', default=False, help='make weights')
    parser.add_argument('--test', action='store_true', default=False, help='plot from the test directory')
    parser.add_argument('--data-dir', required=True, help='directory for train and test data files')
    parser.add_argument('--odir', required=True, help="output dir")
    parser.add_argument('--config', default=None, help="data config")
    args = parser.parse_args()

    import os
    os.system('mkdir -p %s'%args.odir)
    
    # load config if needed
    if args.config:
        data_config_file = args.config

        from utils.data.config import DataConfig
        data_config = DataConfig.load(data_config_file, load_observers=False)
        reweight_branches = data_config.reweight_branches
        reweight_bins = data_config.reweight_bins
        reweight_hists = data_config.reweight_hists
        mask_train = data_config.selection
        mask_test = data_config.test_time_selection
    else:
        if not args.weights:
            print('Need to load config if you do not want to remake weights!')
            exit
        else:
            data_config_file = None
            
        # define selection
        if args.regression:
            mask_train = "(fj_pt>200) & (fj_pt<2500) & (fj_genjetmsd<260) &"\
                "( ( ( (fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1) ) & "\
                "(fj_genRes_mass<0) ) | "\
                "(fj_isTop == 1) | (fj_isToplep==1) | "\
                "( ( (fj_H_WW_4q==1) | (fj_H_WW_elenuqq==1) | (fj_H_WW_munuqq==1) | (fj_H_WW_taunuqq==1) ) & "\
                "(fj_maxdR_HWW_daus<2.0) & (fj_nProngs>1) & (fj_genRes_mass>0) ) )"
            mask_test = mask_train
        else:
            # CMS: here I changed msoftdrop cut to 260!
            mask_train = "(fj_pt>200) & (fj_pt<2500) & (fj_msoftdrop>=30) & (fj_msoftdrop<260) & "\
                "( ( ( (fj_isQCDb==1) | (fj_isQCDbb==1) | (fj_isQCDc==1) | (fj_isQCDcc==1) | (fj_isQCDlep==1) | (fj_isQCDothers==1) ) & "\
                "(fj_genRes_mass<0) ) | "\
                "(fj_isTop == 1) | (fj_isToplep==1) | "\
                "( ( (fj_H_WW_4q==1) | (fj_H_WW_elenuqq==1) | (fj_H_WW_munuqq==1) | (fj_H_WW_taunuqq==1) ) & "\
                "(fj_maxdR_HWW_daus<2.0) & (fj_nProngs>1) & (fj_genRes_mass>0) ) )"
            mask_test = mask_train
        
    if args.test:
        mask = mask_test
    else:
        mask = mask_train

    # define branches:
    branches = ["fj_pt","fj_msoftdrop","fj_genjetmsd","fj_genRes_mass","fj_maxdR_HWW_daus","fj_nProngs"]
    branches += ["fj_isTop","fj_isToplep"]
    branches += ["fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_WW_taunuqq"]
    branches += ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]

    qcd_cats = ["fj_isQCDb","fj_isQCDbb","fj_isQCDc","fj_isQCDcc","fj_isQCDlep","fj_isQCDothers"]
    top_cats = ["fj_isTop_merged","fj_isTop_semimerged","fj_isToplep_merged"]
    if args.regression:
        sig_cats = ["fj_H_WW_4q","fj_H_WW_elenuqq","fj_H_WW_munuqq","fj_H_WW_taunuqq"]
    else:
        sig_cats = ["fj_isHWW_elenuqq_merged","fj_isHWW_elenuqq_semimerged","fj_isHWW_munuqq_merged","fj_isHWW_munuqq_semimerged",
                    "fj_isHWW_taunuqq_merged","fj_isHWW_taunuqq_semimerged",
                    "fj_H_WW_4q_3q","fj_H_WW_4q_4q"]
        
    # define events
    if args.regression:
        events = uproot.iterate([os.path.join(args.data_dir, 'train/QCD*/*.root:Events'),
                                 os.path.join(args.data_dir, 'train/Grav*/*.root:Events')], branches, mask)
    else:
        events = uproot.iterate([os.path.join(args.data_dir, 'train/QCD*/*.root:Events'),
                                 os.path.join(args.data_dir, 'train/Grav*/*.root:Events'),
                                 os.path.join(args.data_dir, 'train/TT*/*.root:Events')], branches, mask)
        # uncomment this if for testing with one file
        # events =  uproot.iterate([os.path.join(args.data_dir, 'train/QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/nano_mc2017_1-179_Skim.root:Events'),
        #                           os.path.join(args.data_dir, 'train/GravitonToHHToWWWW/nano_mc2017_4_Skim.root:Events'),
        #                           os.path.join(args.data_dir, 'train/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/nano_mc2017_84_Skim.root:Events'),   
        #                          os.path.join(args.data_dir, 'train/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/nano_mc2017_93_Skim.root:Events')],branches,mask)
        
    # create table
    table = create_table(events,branches)
        
    # build new variables
    table["target_mass"] = np.maximum(1, np.where(table["fj_genRes_mass"]>0, table["fj_genRes_mass"], table["fj_genjetmsd"]))

    if not args.regression:
        table["fj_isTop_merged"] = (table["fj_isTop"]==1) & (table["fj_nProngs"]==3)
        table["fj_isTop_semimerged"] = (table["fj_isTop"]==1) & (table["fj_nProngs"]==2)
        table["fj_isToplep_merged"] = (table["fj_isToplep"]==1) & (table["fj_nProngs"]>=2)
        table["fj_isHWW_elenuqq_merged"] = (table["fj_H_WW_elenuqq"]==1) & (table["fj_nProngs"]==4)
        table["fj_isHWW_elenuqq_semimerged"] = (table["fj_H_WW_elenuqq"]==1) & ((table["fj_nProngs"]==3) | (table["fj_nProngs"]==2))
        table["fj_isHWW_munuqq_merged"] = (table["fj_H_WW_munuqq"]==1) & (table["fj_nProngs"]==4)
        table["fj_isHWW_munuqq_semimerged"] = (table["fj_H_WW_munuqq"]==1) & ((table["fj_nProngs"]==3) | (table["fj_nProngs"]==2))
        table["fj_isHWW_taunuqq_merged"] = (table["fj_H_WW_taunuqq"] ==1) & (table["fj_nProngs"]==4)
        table["fj_isHWW_taunuqq_semimerged"] = (table["fj_H_WW_taunuqq"] ==1) & ((table["fj_nProngs"]==3) | (table["fj_nProngs"]==2))
        table["fj_H_WW_4q_3q"] = (table["fj_H_WW_4q"]==1) & (table["fj_nProngs"]==3)
        table["fj_H_WW_4q_4q"] = (table["fj_H_WW_4q"]==1) & (table["fj_nProngs"]==4)

        table["fj_QCD_label"] = (table["fj_isQCDb"]==1) | (table["fj_isQCDbb"]==1) | (table["fj_isQCDc"]==1) | (table["fj_isQCDcc"]==1) | (table["fj_isQCDlep"]==1) | (table["fj_isQCDothers"]==1)
        table["fj_Top_label"] = (table["fj_isTop"]==1) | (table["fj_isToplep"]==1)
        
    if not args.config:
        print('define reweighting')
        # define the reweighting (you need to define this if you do not load the config)
        if args.regression:
            reweight_branches = ["fj_pt","fj_genjetmsd"]
            reweight_bins = ([200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500],
                             [-10000, 10000])
            reweight_classes = ["fj_isQCDb", "fj_isQCDbb", "fj_isQCDc", "fj_isQCDcc", "fj_isQCDlep", "fj_isQCDothers", "fj_H_WW_4q", "fj_H_WW_elenuqq", "fj_H_WW_munuqq", "fj_H_WW_taunuqq"]
            class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            reweight_threshold = 0.1
            reweight_method = "flat"
            
            # reweight_bins = ([200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500],
            #                  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260])
            # reweight_threshold = 15

        else:
            reweight_branches = ["fj_pt","fj_msoftdrop"]
            reweight_bins = ([200, 251, 316, 398, 501, 630, 793, 997, 1255, 1579, 1987, 2500],
                             [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260])
            reweight_classes = ["fj_QCD_label",
                                #"fj_Top_label",
                                "fj_isHWW_elenuqq_merged", "fj_isHWW_munuqq_merged",
                                #"fj_H_WW_4q_3q", "fj_H_WW_4q_4q", "fj_isHWW_elenuqq_merged", "fj_isHWW_elenuqq_semimerged",
                                #"fj_isHWW_munuqq_merged", "fj_isHWW_munuqq_semimerged", "fj_isHWW_taunuqq_merged", "fj_isHWW_taunuqq_semimerged"
            ]
            class_weights = [1,
                             1,1,
                             #1,
                             #0.125, 0.125, 0.125, 0.125,
                             #0.125, 0.125, 0.125, 0.125
            ]
            #reweight_threshold = 10
            reweight_threshold = 0.1
            #reweight_method = "flat"
            reweight_method = "ref"

        reweight_hists = make_weights(table,reweight_branches,reweight_bins,reweight_classes,reweight_method,class_weights,reweight_threshold)

    # build weights
    _build_weights(table, reweight_hists, reweight_branches, reweight_bins)

    #indices = _get_reweight_indices(table["weight"])
    #print(indices)

    # define categories to plot
    cats = {
        #'sig': sig_cats,
        'sig': ["fj_isHWW_elenuqq_merged", "fj_isHWW_munuqq_merged"] # for the signals only config
        }
    cats['qcd'] =  qcd_cats
    #cats['qcd'] = ["fj_QCD_label"] # uncomment for when using one single label
    
    if not args.regression:
        cats['top'] = top_cats
        #cats['top'] = ["fj_Top_label"]  # uncomment for when using one single label  
    
    # make plots
    make_plots(table,cats,reweight_bins,args)

    # 2d plots of weights
    make_2d_plots(table,reweight_bins,args)
