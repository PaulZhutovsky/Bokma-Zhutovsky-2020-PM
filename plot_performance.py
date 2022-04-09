import seaborn as sns
import numpy as np
import pandas as pd
from scipy.special import binom
import os.path as osp
from itertools import combinations
from ml_classification import DOMAINS
import matplotlib.pyplot as plt


FOLDER_RESULTS = 'rfc'
METRIC_TO_USE = "AUC"


def load_data(folder_results=FOLDER_RESULTS, target_metric=METRIC_TO_USE):
    combination_length = [1, 2, 3, 4, 5]
    n_domains = []
    plot_order = []
    df_results = pd.DataFrame(columns=[target_metric, 'domain_used', 'number_of_combinations'])
    for i_comb, comb_len in enumerate(combination_length):
        if i_comb == 0:
            n_domains.append(binom(len(DOMAINS), comb_len))
        else:
            n_domains.append(binom(len(DOMAINS), comb_len) + n_domains[i_comb - 1])
        for i_dom, dom in enumerate(combinations(DOMAINS, comb_len)):

            dom = '_'.join(dom)
            save_pattern = osp.join(folder_results, dom + '_performance_test.csv')

            dom_metric = pd.read_csv(save_pattern)[target_metric].values
            data = np.column_stack((dom_metric, [dom] * dom_metric.size, [i_comb + 1] * dom_metric.size))
            df = pd.DataFrame(data=data, columns=[target_metric, 'domain_used', 'number_of_combinations'])
            df_results = df_results.append(df, ignore_index=True)
            plot_order.append(dom)
    df_results[target_metric] = df_results[target_metric].astype(np.float)
    df_results.domain_used = df_results.domain_used.astype(str)
    df_results.number_of_combinations = df_results.number_of_combinations.astype(np.int)
    return df_results, plot_order, n_domains


def make_feature_imp_plot(folder_results=FOLDER_RESULTS, max_keep=0.5):
    """
    We will only consider the case of all domains
    :param folder_results:
    :return:
    """
    df = pd.read_csv(osp.join(folder_results, 'IA_IIA_IIIA_IVA_VA_var_importance.csv'))
    cv_col = [col for col in df.columns if col.startswith('cv')]
    df_cv_mean = df[cv_col].mean(axis=1)
    df_cv_std = df[cv_col].std(axis=1)
    df = pd.concat((df.var_name, df_cv_mean, df_cv_std), keys=['Variable', 'Avg. Feature Importance',
                                                               'feat_imp_sd'], axis=1)
    df = df.sort_values(by='Avg. Feature Importance', ascending=False)
    id_keep = df['Avg. Feature Importance'].cumsum() < max_keep
    df = df.loc[id_keep]
    plt.figure(figsize=(11, 18))
    sns.barplot(x='Avg. Feature Importance', y='Variable', data=df, palette='Blues_r', **{'xerr': df.feat_imp_sd})
    plt.title('Random Forest classifier trained on all 5 domains: Feature Importances', fontsize=18)
    sns.despine()
    plt.tight_layout()
    plt.savefig(osp.join(folder_results, 'feature_importances.png'), dpi=500)


def make_plot(df_results, plot_order, n_domains, folder_results=FOLDER_RESULTS, target_metric=METRIC_TO_USE):
    plt.figure(figsize=(15, 8))

    ax = sns.boxplot(x='domain_used', y=target_metric, data=df_results, order=plot_order, color = '0.75', linewidth=2,
                     medianprops={'color': 'crimson', 'linewidth': 2, 'linestyle': '-'},
                     boxprops = {'edgecolor': 'k', 'lw': 2}, width = 0.8, capprops = {'color': 'k'})
    sns.swarmplot(x='domain_used', y=target_metric, data=df_results, order=plot_order, color='k', alpha=0.5, ax=ax)
    plt.xticks(rotation=45)
    plt.axhline(0.5, 0, 1, lw=2, c='r')

    for n in n_domains:
        plt.axvline(n - 0.5, 0, 1, lw=2, c='k', ls='--')
    plt.ylim([0.4, 0.8])
    plt.title(folder_results, fontsize=18)
    ax.grid(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(osp.join(folder_results, 'performance.png'), dpi=500)


def run(folder_results=FOLDER_RESULTS, target_metric=METRIC_TO_USE):
    df_results, plot_order, n_domains = load_data(folder_results=folder_results, target_metric=target_metric)
    make_plot(df_results, plot_order, n_domains, folder_results=folder_results, target_metric=target_metric)
    make_feature_imp_plot(folder_results=folder_results)


if __name__ == '__main__':
    plt.close('all')
    folder_results = ['rfsc_invd_scores_300trees_rsearch_30_simpleSearch']
    target_metric = ['AUC']
    for i, res in enumerate(folder_results):
        print(res)
        run(folder_results=res, target_metric=target_metric[i])
