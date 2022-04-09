import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from evaluate_clf_performance import sign_difference_test


def run(save_folders, domains, target_metric, type_of_analysis):
    title = {'anxiety': 'Recovery from Anxiety Disorders', 'any_disorder': 'Recovery from all Common Mental Disorders',
             'multiclass': 'Multiclass Classification'}
    subtitle_labels = {0: 'A', 1: 'B', 2: 'C'}
    fig1, ax1 = plt.subplots(1, 2, sharey=True, figsize=(20, 8))
    ax1 = ax1.ravel()
    # fig1.delaxes(ax1[-1])
    fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(20, 15))
    fig3, ax3 = plt.subplots(2, 2, sharex=True, figsize=(11, 11))
    ax3 = ax3.ravel()
    fig3.delaxes(ax3[-1])
    fig4, ax4 = plt.subplots(1, 3, sharex=True, figsize=(22, 12))
    for i, save_folder in enumerate(save_folders):
        df_metrics, p_vals = load_perf_data(domains, save_folder, target_metric[i])
        df_correct_pred = load_pred_data(save_folder, domains)
        df_feat_imp, plot_order = load_feat_imp(save_folder, type_of_analysis[i])
        df_significant = compare_domains(df_metrics, domains, target_metric[i])

        if i == 0:
            ylabel = ''
            xlabel = ''
            legend = False
        else:
            ylabel = ''
            xlabel = 'Patient trajectory'
            legend = True
        make_figure1(df_metrics, p_vals,  x_axis=target_metric[i], y_axis='Domains',
                     title=title[type_of_analysis[i]], ax=ax1[i], ylabel=ylabel, sublabel=subtitle_labels[i])
        make_figure2(df_correct_pred, x_axis='Patient trajectory', y_axis='Correctly classified [%]', hue='Domains',
                     xlabel=xlabel, ax=ax2[i], type_of_analysis=type_of_analysis[i], legend=legend)
        make_figure3(df_significant, type_of_analysis=type_of_analysis[i], ax=ax3[i],
                     domain_labels=('IA', 'IIA', 'IIIA', 'IVA', 'VA', 'IA+IIA+IIIA\nIVA+VA'))
        make_figure4(df_feat_imp, type_of_analysis=type_of_analysis[i], ax=ax4[i], x_axis='Feature importance [a.u.]',
                     y_axis='var_name', order=plot_order)

    fig1.tight_layout()
    # fig2.tight_layout()
    # fig3.tight_layout()
    # fig4.tight_layout()
    fig1.savefig('figures/figure1_performance.png', dpi=600)
    fig1.savefig('figures/figure1_performance.pdf')
    fig2.savefig('figures/correct_classification.png', dpi=600)
    fig3.savefig('figures/comparison_domains.png', dpi=600)
    fig4.savefig('figures/feature_importances.png', dpi=600)


def load_feat_imp(save_folder, type_of_analysis):
    df, var_names, _ = get_significant_features(save_folder, type_of_analysis=type_of_analysis)

    data = []
    labels = []
    for var in var_names:
        data.append(df.loc[var, :].values)
        labels.append([var] * data[-1].size)

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    df_all = pd.DataFrame(data=np.column_stack((data, labels)), columns=['Feature importance [a.u.]', 'var_name'])
    df_all['Feature importance [a.u.]'] = df_all['Feature importance [a.u.]'].astype(np.float)
    plot_order = df_all.groupby(['var_name']).mean().sort_values('Feature importance [a.u.]', ascending=False).index
    plot_order = plot_order.values
    return df_all, plot_order


def get_significant_features(save_folder, file_importance='IA_IIA_IIIA_IVA_VA_var_importance.csv',
                             file_perm_importance='permuted_variable_importances_domains_all.csv',
                             data_folder='/data/pzhutovsky/NESDA_anxiety',
                             consistency_threshold=50, type_of_analysis='any_anxiety'):
    df = pd.read_csv(osp.join(save_folder, file_importance), index_col=0)
    var_names = df.index.values
    df_perm = pd.read_csv(osp.join(save_folder, file_perm_importance), index_col=0)
    df_perm.drop('true_feature_importances', axis=1, inplace=True)
    n_cv = df.shape[1]
    p_values = np.zeros((var_names.size, n_cv))
    for i_var, var_name in enumerate(var_names):
        null_dist = df_perm.loc[var_name, :].values

        for i_cv in range(n_cv):
            true_val = df.loc[var_name, 'cv_{}'.format(i_cv + 1)]
            p_values[i_var, i_cv] = permutation_test(val_true=true_val, val_perm=null_dist)
    # multiple testing correction
    reject_all = np.zeros_like(p_values, dtype=bool)
    for i_cv in range(n_cv):
        p_values_cv = p_values[:, i_cv]
        reject_all[:, i_cv], _, _, _ = multipletests(p_values_cv, alpha=0.05, method='fdr_tsbh')
    reject_H0 = reject_all.sum(axis=1) > consistency_threshold
    var_names = var_names[reject_H0]
    df_desc = pd.read_csv(osp.join(data_folder, 'nesda_variable_descriptions.csv'), index_col=0, encoding='ISO-8859-1')
    df_desc = df_desc.loc[var_names].copy()
    df_desc.loc[var_names, 'selection_frequency'] = reject_all.sum(axis=1)[reject_H0]
    df_desc = df_desc.sort_values(by='selection_frequency', ascending=False)
    df_desc.to_csv('figures/consistently_significant_variables_{}.csv'.format(type_of_analysis))
    return df, var_names, df_desc


def load_pred_data(save_folder, domains):
    df_dim1 = pd.read_csv(osp.join(save_folder, 'prediction_correctly_classified_subj_max_domain1.csv'), header=[0, 1],
                          index_col=0)
    df_dim5 = pd.read_csv(osp.join(save_folder, 'prediction_correctly_classified_subj_max_domain5.csv'), header=[0, 1],
                          index_col=0)
    df_all = pd.concat((df_dim1, df_dim5), axis=1)
    idx = pd.IndexSlice
    df_mean = df_all.loc[:, idx[:, 'Mean']]
    categories = df_mean.index.values

    data = []
    labels = []
    doms = []
    for dom in domains:
        data.append(df_mean[dom].values)
        labels.append(categories)
        doms.append([dom] * len(data[-1]))
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    doms = np.concatenate(doms)

    data = np.column_stack((data, labels, doms))
    df = pd.DataFrame(data=data, columns=['Correctly classified [%]', 'Patient trajectory', 'Domains'])
    return df


def compare_domains(df_perf, domains, target_metric, alpha=0.05):
    df_sign = pd.DataFrame(columns=domains, index=domains)
    alpha /= (len(domains) * len(domains)) - len(domains)
    for dom_compare in domains:
        id_dom = df_perf.Domains == dom_compare
        metric_to_compare = df_perf.loc[id_dom, target_metric].values
        dom_others = np.setdiff1d(domains, dom_compare)

        for dom_other in dom_others:
            id_other_dom = df_perf.Domains == dom_other
            metric_other = df_perf.loc[id_other_dom, target_metric].values
            _, p_value = sign_difference_test(metric_to_compare, metric_other, n_perm=10000)

            if p_value < alpha:
                df_sign.loc[dom_compare, dom_other] = 1
            else:
                df_sign.loc[dom_compare, dom_other] = 0
    return df_sign.astype(np.float)


def make_figure1(df_metrics, p_vals, ax, x_axis, y_axis, title, ylabel, sublabel):
    # to_replace = {'IA_IIA_IIIA_IVA_VA': 'IA+IIA+IIIA\nIVA+VA'}
    to_replace = {'IA': 'Clinical', 'IIA': 'Psychological', 'IIIA': 'Sociodemographic',
                  'IVA': 'Biological', 'VA': 'Lifestyle', 'IA_IIA_IIIA_IVA_VA': 'Combination'}
    df_metrics[y_axis].replace(to_replace=to_replace, inplace=True)
    placement = {'IA': 0, 'IIA': 1, 'IIIA': 2, 'IVA': 3, 'VA': 4, 'IA_IIA_IIIA_IVA_VA': 5}
    sns.boxplot(x=x_axis, y=y_axis, data=df_metrics, color='0.8', medianprops={'color': 'k'}, linewidth=2, ax=ax,
                boxprops={'edgecolor': 'k'}, capprops={'color': 'k'}, whiskerprops={'linestyle': ':'})
    # sns.swarmplot(x=x_axis, y=y_axis, data=df_metrics, color='k', ax=ax)

    x_place = 0.86
    for key, val in p_vals.items():
        y_place = placement[key]
        ax.text(x_place, y_place + 0.11, '*', fontsize=26, ha='center', va='center', fontweight='bold')

    ax.set_xticks([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9])
    ax.tick_params(axis='y', labelsize=18, size=0)
    ax.tick_params(axis='x', labelsize=18, size=10)
    ax.axvline(0.5, 0, 1, lw=2, ls='--', c='0.5')
    ax.set_xlim((0.36, 0.88))
    ax.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
    ax.grid(True, axis='x')
    if x_axis == 'auc_one_vs_rest':
        x_axis = 'AUC'
    ax.set_xlabel(x_axis, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.text(0.04, 1.03, sublabel, fontsize=20, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, weight='bold')

    sns.despine(trim=True, ax=ax, offset=20, left=True)
    ax.set_title(title, fontsize=20)


def make_figure2(df_correct_pred, x_axis, y_axis, hue, xlabel, ax, type_of_analysis, legend):
    sns.barplot(x=x_axis, y=y_axis, hue=hue, data=df_correct_pred, ax=ax, ci=None)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='x', labelrotation=20)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=18, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.legend([])
    ax.grid(True, axis='y')
    ax.set_ylim([0, 88])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(y_axis, fontsize=20)
    ax.set_title('Predicting *{}* at follow-up'.format(type_of_analysis), fontsize=20)


def make_figure3(df_significant, type_of_analysis, ax, domain_labels):
    sns.heatmap(df_significant, annot=False, cbar=False, ax=ax, xticklabels=domain_labels, yticklabels=domain_labels,
                linewidths=0.5)
    ax.set_title('Comparison between domains\n(*{}* at follow-up)'.format(type_of_analysis), fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='y', labelrotation=0)
    ax.tick_params(axis='x', labelrotation=30)


def make_figure4(df_feat_imp, type_of_analysis, ax, x_axis, y_axis, order):
    sns.barplot(x=x_axis, y=y_axis, data=df_feat_imp, ci='sd', ax=ax, color='c', order=order)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=14)
    title_string = 'Predicting *{}* at follow-up'.format(type_of_analysis)
    ax.set_title(title_string, fontsize=20)
    ax.set_xlabel(x_axis, fontsize=20)


def load_perf_data(domains, save_folder, target_metric):
    data_dict = {}
    p_vals = {}
    for domain in domains:
        df_metric = pd.read_csv(osp.join(save_folder, '{}_performance_test.csv'.format(domain)))[target_metric]
        data_dict[domain] = df_metric

        if osp.exists(osp.join(save_folder, '{}_perf_permutations.csv'.format(domain))):
            metric_true = df_metric.mean()
            metric_perm = pd.read_csv(osp.join(save_folder,
                                               '{}_perf_permutations.csv'.format(domain)))[target_metric].values
            p_vals[domain] = permutation_test(metric_true, metric_perm)
    metric_all = np.concatenate([np.column_stack((val.values, [key] * val.size)) for key, val in data_dict.items()],
                                axis=0)
    df_metric_all = pd.DataFrame(data=metric_all, columns=[target_metric, 'Domains'])
    df_metric_all[target_metric] = df_metric_all[target_metric].astype(np.float)
    return df_metric_all, p_vals


def permutation_test(val_true, val_perm):
    return (np.sum(val_perm >= val_true) + 1) / (val_perm.size + 1)


if __name__ == '__main__':
    # SAVE_FOLDERS = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_anxiety',
    #                 '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_disorder',
    #                 '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_multiclass']
    SAVE_FOLDERS = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_anxiety',
                    '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_disorder']
    # analysis_type = ['anxiety', 'any_disorder', 'multiclass']
    analysis_type = ['anxiety', 'any_disorder']
    # TARGET_METRIC = ['AUC', 'AUC', 'auc_one_vs_rest']
    TARGET_METRIC = ['AUC', 'AUC']
    DOMAINS = ['IA', 'IIA', 'IIIA', 'IVA', 'VA', 'IA_IIA_IIIA_IVA_VA']

    run(SAVE_FOLDERS, DOMAINS, target_metric=TARGET_METRIC, type_of_analysis=analysis_type)
