import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint
from sklearn import metrics

def summarize_df(data, g1_idx, g2_idx):
    """
    Returns a data frame with all statistics useful in the analysis.
    g1_index - position of an index of the first group in data's columns (e.g. healthy)
    g2_index - position of an index of the second group in data's columns (e.g. CFS)
    """
    
    lower_bounds = {}
    upper_bounds = {}
    p_values = {}
    test_stats = {}
    g1_prop_above = {}
    g2_prop_above = {}
    auc = {}
    gini = {}
    
    sequences = []
    antibodies = []
    strains = []
    peptide_ids = []
    names = []

    groups = []

    for j in range(data.shape[0]):

        row = data.iloc[j]

        dt = row[2:5]
        strains.append(', '.join(dt[dt == 'x'].index))

        groups.append(set(dt[dt == 'x'].index))
        
        sequences.append(row[1])
        antibodies.append(row[0][-4:])
        peptide_ids.append('EBNA1_' + antibodies[-1])
        names.append(peptide_ids[-1] + ' (' + strains[-1] + ')')

        g1 = np.array(row[g1_idx:g2_idx])
        g2 = np.array(row[g2_idx:])

        row = sorted(list(row[g1_idx:]))

        lower_bounds[j] = []
        upper_bounds[j] = []
        p_values[j] = []
        test_stats[j] = []
        g1_prop_above[j] = []
        g2_prop_above[j] = []
        auc[j] = []
        gini[j] = []

        y = np.array([1]*g1.shape[0]+[0]*g2.shape[0])

        sm = (g1.shape[0] + g2.shape[0])
        gini_basic = 1 - (g1.shape[0]/sm)**2 - (g2.shape[0]/sm)**2

        for i in range(len(row)-1):
            threshold = row[i]
            g1_above = (g1 > threshold).sum()
            g1_below = g1.shape[0] - g1_above
            g2_above = (g2 > threshold).sum()
            g2_below = g2.shape[0] - g2_above

            # chi square test
            test = chi2_contingency(np.array([[g1_above, g1_below], [g2_above, g2_below]]))

            # max AUC
            pred = np.concatenate([np.where(g1 > threshold, 1, 0), np.where(g2 > threshold, 1, 0)])
            fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
            auc_tmp = metrics.auc(fpr, tpr)
            auc[j].append(max(auc_tmp, 1-auc_tmp))

            # Gini index
            sm_above = g1_above + g2_above
            gini_above = 1 - (g1_above/sm_above)**2 - (g2_above/sm_above)**2
            sm_below = g1_below + g2_below
            gini_below = 1 - (g1_below/sm_below)**2 - (g2_below/sm_below)**2
            gini_current = (sm_above/sm)*gini_above + (sm_below/sm)*gini_below
            gini[j].append(gini_basic - gini_current)

            lower_bounds[j].append(threshold)
            upper_bounds[j].append(row[i+1])
            p_values[j].append(test[1])
            test_stats[j].append(test[0])

            g1_prop_above[j].append(g1_above/g1.shape[0])
            g2_prop_above[j].append(g2_above/g2.shape[0])

    best_p_values = []
    best_test_stats = []
    best_test_lower_bounds = []
    best_test_upper_bounds = []
    best_test_g1_prop_above = []
    best_test_g2_prop_above = []

    best_auc = []
    best_auc_lower_bounds = []
    best_auc_upper_bounds = []
    best_auc_g1_prop_above = []
    best_auc_g2_prop_above = []

    best_gini = []
    best_gini_lower_bounds = []
    best_gini_upper_bounds = []
    best_gini_g1_prop_above = []
    best_gini_g2_prop_above = []

    best_prop_g1_min = []
    best_prop_g1_max = []
    best_prop_g2_min = []
    best_prop_g2_max = []

    for j in range(data.shape[0]):
        index_min_p = min(range(len(p_values[j])), key=p_values[j].__getitem__)
        best_p_values.append(p_values[j][index_min_p])
        best_test_stats.append(test_stats[j][index_min_p])
        best_test_lower_bounds.append(lower_bounds[j][index_min_p])
        best_test_upper_bounds.append(upper_bounds[j][index_min_p])
        best_test_g1_prop_above.append(g1_prop_above[j][index_min_p])
        best_test_g2_prop_above.append(g2_prop_above[j][index_min_p])

        prop_g1 = proportion_confint(np.round(best_test_g1_prop_above[-1]*g1.shape[0], 0), g1.shape[0], alpha=0.05, method='normal')
        best_prop_g1_min.append(prop_g1[0])
        best_prop_g1_max.append(prop_g1[1])
        prop_g2 = proportion_confint(np.round(best_test_g2_prop_above[-1]*g2.shape[0], 0), g2.shape[0], alpha=0.05, method='normal')
        best_prop_g2_min.append(prop_g2[0])
        best_prop_g2_max.append(prop_g2[1])

        index_max_gini = max(range(len(gini[j])), key=gini[j].__getitem__)
        best_gini.append(gini[j][index_max_gini])
        best_gini_lower_bounds.append(lower_bounds[j][index_max_gini])
        best_gini_upper_bounds.append(upper_bounds[j][index_max_gini])
        best_gini_g1_prop_above.append(g1_prop_above[j][index_max_gini])
        best_gini_g2_prop_above.append(g2_prop_above[j][index_max_gini])

        index_max_auc = max(range(len(auc[j])), key=auc[j].__getitem__)
        best_auc.append(auc[j][index_max_auc])
        best_auc_lower_bounds.append(lower_bounds[j][index_max_auc])
        best_auc_upper_bounds.append(upper_bounds[j][index_max_auc])
        best_auc_g1_prop_above.append(g1_prop_above[j][index_max_auc])
        best_auc_g2_prop_above.append(g2_prop_above[j][index_max_auc])


    summary = pd.DataFrame({'peptide_id':peptide_ids, 'strains':strains, 'sequence':sequences, 'antibody':antibodies, 
                'test_lower_bound':best_test_lower_bounds, 'test_upper_bound':best_test_upper_bounds, 'min_p_value': best_p_values, 
                'test_g1_prop_above': best_test_g1_prop_above, 'test_g2_prop_above': best_test_g2_prop_above, 'max_auc':best_auc, 
                'auc_lower_bound': best_auc_lower_bounds, 'auc_upper_bound': best_auc_upper_bounds, 'auc_g1_prop_above': best_auc_g1_prop_above,
                'auc_g2_prop_above': best_auc_g2_prop_above, 'max_gini':best_gini, 'gini_lower_bound':best_gini_lower_bounds, 
                'gini_upper_bound': best_gini_upper_bounds, 'gini_g1_prop_above': best_gini_g1_prop_above, 
                'gini_g2_prop_above': best_gini_g2_prop_above, 'proportion_min_g1': best_prop_g1_min, 'proportion_max_g1': best_prop_g1_max,
                'proportion_min_g2': best_prop_g2_min, 'proportion_max_g2': best_prop_g2_max, 'groups':groups})
    
    summary['name'] = summary.peptide_id + ' (' + summary.strains + ')'

    return summary, lower_bounds, p_values, auc, gini

def plot_signals_p_value(smr, all_lower_bounds, all_p_values):
    """
    Plots a panel with signals obtained from chi-squared test.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    all_p_values - dictionary with p-values for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15))
    plot_names = list(smr.peptide_id)
    for i in range(6):
        for j in range(3):
            if 3*i+j < 16:
                axs[i, j].plot(np.log10(np.array(all_lower_bounds[3*i+j])), -np.log10(np.array(all_p_values[3*i+j])))
                axs[i, j].set_title(plot_names[3*i+j])
                axs[i, j].set_xlabel(r'log$_{10}$(threshold)')
                axs[i, j].set_ylabel(r'-log$_{10}$(p-value)')
                axs[i, j].set_ylim([0, 4])
                axs[i, j].vlines(np.log10(smr.test_lower_bound[3*i+j]), ymin = 0, ymax = 4, color = 'red', linestyles = 'dotted')
                axs[i, j].axhline(-np.log10(0.05), ls = '--', color = 'red')
            else:
                axs[i, j].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_signals_auc(smr, all_lower_bounds, auc):
    """
    Plots a panel with signals obtained from maximizing AUC.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    auc - dictionary with AUC for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15))
    plot_names = list(smr.peptide_id)
    for i in range(6):
        for j in range(3):
            if 3*i+j < 16:
                axs[i, j].plot(np.log10(np.array(all_lower_bounds[3*i+j])), auc[3*i+j])
                axs[i, j].set_title(plot_names[3*i+j])
                axs[i, j].set_xlabel(r'log$_{10}$(threshold)')
                axs[i, j].set_ylabel('AUC')
                axs[i, j].set_ylim([0.5, 1])
                axs[i, j].vlines(np.log10(smr.auc_lower_bound[3*i+j]), ymin = 0.5, ymax = 1, color = 'red', linestyles = 'dotted')
                axs[i, j].axhline(smr.max_auc.max(), ls = '--', color = 'red')
            else:
                axs[i, j].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_signals_gini(smr, all_lower_bounds, gini):
    """
    Plots a panel with signals obtained from maximizing AUC.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    gini - dictionary with AUC for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15))
    plot_names = list(smr.peptide_id)
    for i in range(6):
        for j in range(3):
            if 3*i+j < 16:
                axs[i, j].plot(np.log10(np.array(all_lower_bounds[3*i+j])), gini[3*i+j])
                axs[i, j].set_title(plot_names[3*i+j])
                axs[i, j].set_xlabel(r'log$_{10}$(threshold)')
                axs[i, j].set_ylabel('Gini index decrease')
                axs[i, j].set_ylim([0, 0.1])
                axs[i, j].vlines(np.log10(smr.gini_lower_bound[3*i+j]), ymin = 0, ymax = 0.1, color = 'red', linestyles = 'dotted')
                axs[i, j].axhline(smr.max_gini.max(), ls = '--', color = 'red')
            else:
                axs[i, j].set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_conf_int(smr, g1_name, g2_name, adjust = False):
    """
    Plots confidence interval for proportion of cases above thresholds.
    g1_name - name of the first group (e.g. healthy)
    g2_name - name of the second group (e.g. CFS)
    """
    summary = smr.copy()
    summary.loc[[1,3,5,15],'antibody'] = summary.loc[[1,3,5,15],'antibody'] + '_'

    ants = list(summary.antibody.apply(lambda x: x[:-1] if '_' in x else x).astype(int))

    summary_g1 = summary.copy()
    summary_g1['type'] = g1_name
    summary_g1['prop'] = summary_g1.test_g1_prop_above
    summary_g1['prop_max'] = summary_g1.proportion_max_g1
    summary_g1['prop_min'] = summary_g1.proportion_min_g1
    summary_g2 = summary.copy()
    summary_g2['type'] = g2_name
    summary_g2['prop'] = summary_g2.test_g2_prop_above
    summary_g2['prop_max'] = summary_g2.proportion_max_g2
    summary_g2['prop_min'] = summary_g2.proportion_min_g2
    summary_g2['antibody'] = summary_g2['antibody'] + '?'
    summary = pd.concat([summary_g1, summary_g2])

    summary = summary.sort_values('antibody')
    fig = plt.figure(figsize=(16, 8))
    ax = sns.scatterplot(data =summary, x = 'antibody', y = 'prop', hue = 'type', 
                        palette = {g1_name:'blue', g2_name:'red'})
    ax.legend(title='')
    yerr = (summary.prop_max - summary.prop_min)/2
    plt.errorbar(x=summary.antibody.iloc[::2], y=summary.prop.iloc[::2], yerr=yerr.iloc[::2], fmt='none', c= 'blue', capsize=5)
    plt.errorbar(x=summary.antibody.iloc[1::2], y=summary.prop.iloc[1::2], yerr=yerr.iloc[1::2], fmt='none', c= 'red', capsize=5)
    plt.ylabel('% above')
    plt.xlabel('antibody')
    plt.xticks([])
    for i in range(16):
        if i != 15:
            ax.axvline(2*i+1.5, ls = '--', color = 'black')
        if not adjust:
            plt.text(2*i+0.25, -0.13, ants[i])
        else:
            plt.text(2*i+0.25, 0.22, ants[i])
    plt.show()

def plot_correction(groups, adj_p_values, correction_type, strain, strain_seq, strain_stop, strain_target):
    """
    Plots p-values for each antibody with a chosen correction.
    groups - set of strains for each peptide
    adj_p_values - list with adjusted p-values
    correction_type - list with applied corrections
    strain - list of strains
    strain_seq - list with sequences for each correction
    strain_stop - list with ending points of each antibody
    strain_target - starting point of the target
    """
    tmp = [i for i,g in enumerate(groups) if strain in g]
    fig, ax = plt.subplots(figsize=(16, 6))
    for i in range(len(strain_stop)):
        p = plt.plot(list(range(strain_stop[i]-15, strain_stop[i])), (adj_p_values[tmp[i]],)*15)
        col = p[0].get_color()
        plt.plot((strain_stop[i]-15,)*5, np.linspace(adj_p_values[tmp[i]]-0.02, adj_p_values[tmp[i]]+0.02, 5), col)
        plt.plot((strain_stop[i]-1,)*5, np.linspace(adj_p_values[tmp[i]]-0.02, adj_p_values[tmp[i]]+0.02, 5), col)
    plt.xticks(range(len(strain_seq)), strain_seq)
    plt.ylabel(r'-log$_{10}$(adj. p-value)')
    plt.title(f'{strain} ({correction_type} correction)')
    plt.axhline(-np.log10(0.05), ls = '--', color = 'red')
    [i.set_color("red") for i in plt.gca().get_xticklabels()[strain_target:(strain_target+15)]]
    plt.axvspan(strain_target, strain_target+14, facecolor='grey', alpha=0.2)
    ax.set_ylim([-0.2, 3])
    plt.show()