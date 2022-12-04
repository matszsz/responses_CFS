import pandas as pd
import numpy as np
import matplotlib
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
    g1_obs_above = {}
    g1_obs_below = {}
    g2_obs_above = {}
    g2_obs_below = {}
    roc01 = {}
    youden = {}
    sen_spe_dist = {}
    sen = {}
    spe = {}
    y_plot = {}
    x_plot = {}
    
    sequences = []
    antibodies = []
    strains = []
    peptide_ids = []
    names = []
    directions = []

    AUC_TOT = []

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
        roc01[j] = []
        youden[j] = []
        sen_spe_dist[j] = []
        sen[j] = []
        spe[j] = []

        g1_obs_above[j] = []
        g2_obs_above[j] = []
        g1_obs_below[j] = []
        g2_obs_below[j] = []

        directions.append('below')

        y = np.array([1]*g1.shape[0]+[0]*g2.shape[0])

        sm = (g1.shape[0] + g2.shape[0])
        gini_basic = 1 - (g1.shape[0]/sm)**2 - (g2.shape[0]/sm)**2

        for i in range(len(row)-1):
            threshold = row[i]
            g1_above = (g1 > threshold).sum()
            g1_below = g1.shape[0] - g1_above
            g2_above = (g2 > threshold).sum()
            g2_below = g2.shape[0] - g2_above

            g1_obs_below[j].append(g1_below)
            g2_obs_below[j].append(g2_below)
            g1_obs_above[j].append(g1_above)
            g2_obs_above[j].append(g2_above)

            # chi square test
            test = chi2_contingency(np.array([[g1_above, g1_below], [g2_above, g2_below]]))

            # max AUC
            pred = np.concatenate([np.where(g1 > threshold, 1, 0), np.where(g2 > threshold, 1, 0)])
            fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
            auc_tmp = metrics.auc(fpr, tpr)
            auc[j].append(max(auc_tmp, 1-auc_tmp))

            sensitivity = g1_below/(g1_above+g1_below)
            specificity = 1 - g2_below/((g2_above+g2_below))

            sen[j].append(sensitivity)
            spe[j].append(specificity)

            # ROC01
            roc01[j].append(np.sqrt((1-sensitivity)**2 + (1-specificity)**2))

            # Youden's J-index
            youden[j].append(sensitivity + specificity - 1)

            # distance
            sen_spe_dist[j].append(np.abs(sensitivity-specificity))

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

        xxx = 1-np.array([1] + spe[j] + [0])
        yyy = np.array([0] + sen[j] + [1])

        auc_total = metrics.auc(xxx, yyy)

        if auc_total < 0.5:
            sen[j] = []
            spe[j] = []
            roc01[j] = []
            youden[j] = []
            sen_spe_dist[j] = []
            for i in range(len(row)-1):
                threshold = row[i]
                g1_above = (g1 > threshold).sum()
                g1_below = g1.shape[0] - g1_above
                g2_above = (g2 > threshold).sum()
                g2_below = g2.shape[0] - g2_above

                sensitivity = g1_above/(g1_above+g1_below)
                specificity = 1 - g2_above/((g2_above+g2_below))

                sen[j].append(sensitivity)
                spe[j].append(specificity)

                # ROC01
                roc01[j].append(np.sqrt((1-sensitivity)**2 + (1-specificity)**2))

                # Youden's J-index
                youden[j].append(sensitivity + specificity - 1)

                # distance
                sen_spe_dist[j].append(np.abs(sensitivity-specificity))

            xxx = 1-np.array([0] + spe[j] + [1])
            yyy = np.array([1] + sen[j] + [0])
            auc_total = metrics.auc(xxx, yyy)

            directions[-1] = 'above'

        AUC_TOT.append(auc_total)
        y_plot[j] = yyy
        x_plot[j] = xxx

    best_p_values = []
    best_test_stats = []
    best_test_lower_bounds = []
    best_test_upper_bounds = []
    best_test_g1_prop_above = []
    best_test_g2_prop_above = []
    best_test_sen = []
    best_test_spe = []

    best_auc = []
    best_auc_lower_bounds = []
    best_auc_upper_bounds = []
    best_auc_g1_prop_above = []
    best_auc_g2_prop_above = []
    best_auc_p_value = []
    best_auc_sen = []
    best_auc_spe = []

    best_gini = []
    best_gini_lower_bounds = []
    best_gini_upper_bounds = []
    best_gini_g1_prop_above = []
    best_gini_g2_prop_above = []
    best_gini_p_value = []
    best_gini_sen = []
    best_gini_spe = []

    best_roc01 = []
    best_roc01_lower_bounds = []
    best_roc01_upper_bounds = []
    best_roc01_g1_prop_above = []
    best_roc01_g2_prop_above = []
    best_roc01_p_value = []
    best_roc01_sen = []
    best_roc01_spe = []

    best_youden = []
    best_youden_lower_bounds = []
    best_youden_upper_bounds = []
    best_youden_g1_prop_above = []
    best_youden_g2_prop_above = []
    best_youden_p_value = []
    best_youden_sen = []
    best_youden_spe = []

    best_dist = []
    best_dist_lower_bounds = []
    best_dist_upper_bounds = []
    best_dist_g1_prop_above = []
    best_dist_g2_prop_above = []
    best_dist_p_value = []
    best_dist_sen = []
    best_dist_spe = []

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
        best_test_sen.append(sen[j][index_min_p])
        best_test_spe.append(spe[j][index_min_p])

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
        best_gini_p_value.append(p_values[j][index_max_gini])
        best_gini_sen.append(sen[j][index_max_gini])
        best_gini_spe.append(spe[j][index_max_gini])

        index_min_roc01 = min(range(len(roc01[j])), key=roc01[j].__getitem__)
        best_roc01.append(roc01[j][index_min_roc01])
        best_roc01_lower_bounds.append(lower_bounds[j][index_min_roc01])
        best_roc01_upper_bounds.append(upper_bounds[j][index_min_roc01])
        best_roc01_g1_prop_above.append(g1_prop_above[j][index_min_roc01])
        best_roc01_g2_prop_above.append(g2_prop_above[j][index_min_roc01])
        best_roc01_p_value.append(p_values[j][index_min_roc01])
        best_roc01_sen.append(sen[j][index_min_roc01])
        best_roc01_spe.append(spe[j][index_min_roc01])

        index_max_youden = max(range(len(youden[j])), key=youden[j].__getitem__)
        best_youden.append(youden[j][index_max_youden])
        best_youden_lower_bounds.append(lower_bounds[j][index_max_youden])
        best_youden_upper_bounds.append(upper_bounds[j][index_max_youden])
        best_youden_g1_prop_above.append(g1_prop_above[j][index_max_youden])
        best_youden_g2_prop_above.append(g2_prop_above[j][index_max_youden])
        best_youden_p_value.append(p_values[j][index_max_youden])
        best_youden_sen.append(sen[j][index_max_youden])
        best_youden_spe.append(spe[j][index_max_youden])

        index_min_dist = min(range(len(sen_spe_dist[j])), key=sen_spe_dist[j].__getitem__)
        best_dist.append(sen_spe_dist[j][index_min_dist])
        best_dist_lower_bounds.append(lower_bounds[j][index_min_dist])
        best_dist_upper_bounds.append(upper_bounds[j][index_min_dist])
        best_dist_g1_prop_above.append(g1_prop_above[j][index_min_dist])
        best_dist_g2_prop_above.append(g2_prop_above[j][index_min_dist])
        best_dist_p_value.append(p_values[j][index_min_dist])
        best_dist_sen.append(sen[j][index_min_dist])
        best_dist_spe.append(spe[j][index_min_dist])

        index_max_auc = max(range(len(auc[j])), key=auc[j].__getitem__)
        best_auc.append(auc[j][index_max_auc])
        best_auc_lower_bounds.append(lower_bounds[j][index_max_auc])
        best_auc_upper_bounds.append(upper_bounds[j][index_max_auc])
        best_auc_g1_prop_above.append(g1_prop_above[j][index_max_auc])
        best_auc_g2_prop_above.append(g2_prop_above[j][index_max_auc])
        best_auc_p_value.append(p_values[j][index_max_auc])
        best_auc_sen.append(sen[j][index_max_auc])
        best_auc_spe.append(spe[j][index_max_auc])


    summary = pd.DataFrame({'peptide_id':peptide_ids, 'strains':strains, 'sequence':sequences, 'antibody':antibodies, 
                'test_lower_bound':best_test_lower_bounds, 'test_upper_bound':best_test_upper_bounds, 'min_p_value': best_p_values, 
                'test_g1_prop_above': best_test_g1_prop_above, 'test_g2_prop_above': best_test_g2_prop_above, 
                'test_sen': best_test_sen, 'test_spe': best_test_spe, 'max_auc':best_auc, 
                'auc_lower_bound': best_auc_lower_bounds, 'auc_upper_bound': best_auc_upper_bounds, 'auc_p_value': best_auc_p_value, 
                'auc_sen': best_auc_sen, 'auc_spe': best_auc_spe, 'max_gini': best_gini, 'gini_p_value': best_gini_p_value,
                'gini_lower_bound':best_gini_lower_bounds, 
                'gini_upper_bound': best_gini_upper_bounds, 'gini_sen': best_gini_sen, 'gini_spe': best_gini_spe,
                'min_roc01':best_roc01, 'roc01_p_value': best_roc01_p_value, 'roc01_sen': best_roc01_sen, 'roc01_spe': best_roc01_spe,
                'roc01_lower_bound': best_roc01_lower_bounds, 'roc01_upper_bound': best_roc01_upper_bounds,
                'max_youden': best_youden, 'youden_lower_bound': best_youden_lower_bounds, 'youden_upper_bound': best_youden_upper_bounds,
                'youden_p_value': best_youden_p_value, 'youden_sen': best_youden_sen, 'youden_spe': best_youden_spe,
                'min_dist': best_youden, 'dist_lower_bound': best_dist_lower_bounds, 'dist_upper_bound': best_dist_upper_bounds,
                'dist_p_value': best_dist_p_value, 'dist_sen': best_dist_sen, 'dist_spe': best_dist_spe,
                'proportion_min_g1': best_prop_g1_min, 'proportion_max_g1': best_prop_g1_max,'proportion_min_g2': best_prop_g2_min, 
                'proportion_max_g2': best_prop_g2_max, 'groups':groups, 'AUC_total': AUC_TOT, 'direction': directions})
    
    summary['name'] = summary.peptide_id + ' (' + summary.strains + ')'

    return summary, lower_bounds, p_values, auc, gini, x_plot, y_plot, youden, roc01

def summarize_threshold(smr):

    n = smr.shape[0]

    methods = ['Chi-square', 'ROC01' ,'Youden', '|Sen-Spe|', 'Gini', 'AUC (single thr.)']
    names = ['test_lower_bound', 'roc01_lower_bound', 'youden_lower_bound', 'dist_lower_bound', 'gini_lower_bound', 'auc_lower_bound']
    bounds = smr[names]
    b = []
    for i in range(smr.shape[1]):
        for j in range(bounds.shape[0]):
            b.append(bounds.iloc[i, j])
    
    return pd.DataFrame({'Method': methods*n, 'Bounds': b})


def plot_AUC(smr, x_plot, y_plot):
    """
    Plots a panel with AUC.
    smr - data frame with all useful statistics
    x_plot - dict with 1-specifiticy values
    y_plot - dict with sensitivity values
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,25), dpi = 300)
    plot_names = list(smr.name)
    for i in range(6):
        for j in range(3):
            if 3*i+j < 16:
                axs[i, j].step(x_plot[3*i+j], y_plot[3*i+j], label = f'AUC={np.round(smr.AUC_total[3*i+j], 3)}')
                axs[i, j].plot((0, 1), (0, 1), '--', color = 'black')
                np.random.seed(0)
                sd = 0.003
                axs[i, j].scatter(x = 1-smr.roc01_spe[3*i+j]+np.random.normal(0, sd), y = smr.roc01_sen[3*i+j]+np.random.normal(0, sd), color = 'red', label = 'ROC01')
                axs[i, j].scatter(x = 1-smr.youden_spe[3*i+j]+np.random.normal(0, sd), y = smr.youden_sen[3*i+j]+np.random.normal(0, sd), color = 'green', label = 'Youden J-index')
                axs[i, j].scatter(x = 1-smr.dist_spe[3*i+j]+np.random.normal(0, sd), y = smr.dist_sen[3*i+j]+np.random.normal(0, sd), color = 'blue', label = '|Sen-Spe|')
                axs[i, j].set_title(plot_names[3*i+j])
                axs[i, j].set_ylim([0, 1])
                axs[i, j].set_xlim([0, 1])
                axs[i, j].set_ylabel('sensitivity')
                axs[i, j].set_xlabel('1 - specificity')
                axs[i, j].legend(handlelength=1, loc = 'lower right')
            else:
                axs[i, j].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_signals_p_value(smr, all_lower_bounds, all_p_values):
    """
    Plots a panel with signals obtained from chi-squared test.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    all_p_values - dictionary with p-values for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15), dpi = 300)
    plot_names = list(smr.name)
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

def plot_signals_youden(smr, all_lower_bounds, youden):
    """
    Plots a panel with signals maximizing Youden J-index.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    youden - dictionary with Youden J-index for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15), dpi = 300)
    plot_names = list(smr.name)
    for i in range(6):
        for j in range(3):
            if 3*i+j < 16:
                axs[i, j].plot(np.log10(np.array(all_lower_bounds[3*i+j])), youden[3*i+j])
                axs[i, j].set_title(plot_names[3*i+j])
                axs[i, j].set_xlabel(r'log$_{10}$(threshold)')
                axs[i, j].set_ylabel("Youden's J-statistic")
                axs[i, j].set_ylim([0, 0.35])
                axs[i, j].vlines(np.log10(smr.youden_lower_bound[3*i+j]), ymin = 0, ymax = 0.35, color = 'red', linestyles = 'dotted')
                axs[i, j].axhline(smr.max_youden.max(), ls = '--', color = 'red')
            else:
                axs[i, j].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_signals_roc01(smr, all_lower_bounds, roc01):
    """
    Plots a panel with signals maximizing ROC01.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    roc01 - dictionary with ROC01 for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15), dpi = 300)
    plot_names = list(smr.name)
    for i in range(6):
        for j in range(3):
            if 3*i+j < 16:
                axs[i, j].plot(np.log10(np.array(all_lower_bounds[3*i+j])), -np.log10(np.array(roc01[3*i+j])))
                axs[i, j].set_title(plot_names[3*i+j])
                axs[i, j].set_xlabel(r'log$_{10}$(threshold)')
                axs[i, j].set_ylabel(r'-log$_{10}$(ROC01)')
                axs[i, j].set_ylim([0, 0.35])
                axs[i, j].vlines(np.log10(smr.roc01_lower_bound[3*i+j]), ymin = 0, ymax = 0.35, color = 'red', linestyles = 'dotted')
                axs[i, j].axhline(-np.log10(smr.min_roc01.min()), ls = '--', color = 'red')
            else:
                axs[i, j].set_visible(False)
    fig.tight_layout()
    plt.show()

def plot_signals_p_value(smr, all_lower_bounds, all_p_values):
    """
    Plots a panel with signals obtained from chi-squared test.
    smr - data frame with all useful statistics
    all_lower_bounds - dictionary with all possible thresholds
    all_p_values - dictionary with p-values for each threshold
    """
    fig, axs = plt.subplots(6, 3, figsize = (15,15), dpi = 300)
    plot_names = list(smr.name)
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
    fig, axs = plt.subplots(6, 3, figsize = (15,15), dpi = 300)
    plot_names = list(smr.name)
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
    fig, axs = plt.subplots(6, 3, figsize = (15,15), dpi = 300)
    plot_names = list(smr.name)
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
    fig, ax = plt.subplots(figsize=(16, 6), dpi = 300)
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