import os
from itertools import product

from IPython.display import display
from ipywidgets import *
import ipywidgets as widgets

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


"""
loading data to get started
"""
def load_data(remove_undersample=True):
    scores = pd.read_csv(
        "data/d_scores_complete.csv", index_col=list(range(9))
    )
    scores = scores.reset_index('st')
    scores['st'] = scores['st'].map(lambda s: 'cv_mean' if s == 'mean' else s)
    scores['st'] = scores['st'].map(lambda s: 'cv_std' if s == 'std' else s)
    scores = scores.set_index('st', append=True)
    scores = scores.unstack("st")
    scores.columns.names = ["metric", "st"]
    s = scores.reset_index(['dataset', 'feat_range'])
    s['dataset[feat_range]'] = s['dataset'] + '[' + s['feat_range'] + ']'
    s = (
        s
        .set_index(['dataset[feat_range]'], append=True)
        .drop(['dataset', 'feat_range'], axis=1)
    )
    scores = s.reorder_levels(['dataset[feat_range]'] + list(s.index.names)[:-1])
    
    # scores = scores.reset_index()
    # scores = scores[scores.undersample == False]
    # scores = scores.drop('undersample', axis=1)
    # nidx = [t[0] for t in scores.columns[:6]]
    # scores = scores.set_index(nidx)

    feature_sets = pd.read_csv(
        "data/d_feature_sets.csv", index_col=list(range(9))
    )  # actually, both flat and ncv
    feature_sets.columns.name = "feature"
    feature_cols = list(feature_sets.columns)
    feature_sets["actual.n_features"] = feature_sets.sum(axis=1)
    feature_sets = feature_sets[["actual.n_features"] + feature_cols]

    """
    Joins and wrangling...
    """
    fss = feature_sets.reset_index(['dataset', 'feat_range', 'src', 'fold'])
    idx = list(fss.index.names)

    # create / remap cols
    fss['dataset[feat_range]'] = fss['dataset'] + '[' + fss['feat_range'] + ']'
    fss.rename(columns={'src':'_src'}, inplace=True)
    fss['src'] = fss['_src'].map({'ncv':'outer', 'refit':'ncv', 'flat':'flat'})

    bidx = (fss.src != 'flat')
    fss.loc[bidx, 'src'] = fss.loc[bidx, 'src'] + '_' + fss.loc[bidx, 'fold']

    idx = ['dataset[feat_range]'] + idx + ['src']
    fss = (
        fss
        .drop(['dataset', 'feat_range', 'fold', '_src'], axis=1)
        .reset_index()
        .set_index(idx)
    )

    anf = fss[['actual.n_features']]
    anf.columns = pd.MultiIndex.from_tuples([('n_features[actual]', '')])

    scores = scores.join(anf)  # MIND YOU, n_features only for as_cv_step = True

    feature_sets = fss.iloc[:, 1:]
    del fss, idx, bidx, anf
#     cv_results_ = pd.read_csv("data/d_cv_results_.csv", index_col=list(range(9)))

    if remove_undersample:

        s = scores.reset_index()
        s = s.loc[(s.undersample == False)]
        assert True not in s.undersample.unique()
        s = s.drop('undersample', axis=1)
        nidx = [col[0] for col in s.columns[:6]]
        s = s.set_index(nidx)
        scores = s

        fs = feature_sets.reset_index()
        fs = fs.loc[(fs.undersample == False)]
        assert True not in fs.undersample.unique()
        fs = fs.drop('undersample', axis=1)
        nidx = list(fs.columns[:6])
        fs = fs.set_index(nidx)
        feature_sets = fs

#         cvr = cv_results_.reset_index('undersample')
#         cvr = cvr.loc[cvr.undersample == False]
#         cvr.drop('undersample', axis=1, inplace=True)
#         cv_results_ = cvr

    sidx = list(scores.index.names)
    sidx.remove('estimator'); sidx.remove('src')
    sidx.append('algorithm'); sidx.append('model')
    scores = (
        scores.reset_index()
        .rename({'estimator':'algorithm', 'src':'model'}, axis=1)
        .set_index(sidx)
    )
    
    fsidx = list(feature_sets.index.names)
    fsidx.remove('estimator'); fsidx.remove('src')
    fsidx.append('algorithm'); fsidx.append('model')
    feature_sets = feature_sets.reset_index()\
        .rename({'estimator':'algorithm', 'src':'model'}, axis=1).set_index(fsidx)

    return scores, feature_sets#, cv_results_



# ----------------------------------------- tabular-based plotting
def prepare_axes(**kwargs):
    
    if 'darkmode' in kwargs:
        if kwargs['darkmode']:
            plt.style.use('dark_background')

    if 'figsize' in kwargs.keys():
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig, ax = plt.subplots()

    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
        del kwargs['title']

    return fig, ax, kwargs

def sns_barplot(**kwargs):
    sns.set_theme(style='whitegrid', font='monospace', font_scale=1.1)
    fig, ax, kwargs = prepare_axes(**kwargs)
    defaults = {'ci':95, 'palette':'pastel', 'alpha':.9}
    kwargs = {**defaults, **kwargs}
    return sns.barplot(ax=ax, **kwargs)


def sns_boxplot(**kwargs):
    sns.set_theme(style='whitegrid', font='monospace', font_scale=1.1)
    fig, ax, kwargs = prepare_axes(**kwargs)
    defaults = {'palette':'pastel'}
    kwargs = {**defaults, **kwargs}
    return sns.boxplot(ax=ax, **kwargs)


def sns_stripplot(**kwargs):
    sns.set_theme(style='whitegrid', font='monospace', font_scale=1.1)
    fig, ax, kwargs = prepare_axes(**kwargs)
    if 'ci' in kwargs:
        del kwargs['ci']
    # Show each observation with a scatterplot
    defaults = {'dodge':True, 'alpha':.8, 'zorder':1}
    kwargs = {**defaults, **kwargs}
    return sns.stripplot(**kwargs)





"""
Ordering
"""
order_dataset_fs = list(
    product(
        ["rectal", "abus", "pancreatic"],
        ["lasso", "rfe_lasso", "kbest_mi", "boruta_rf", "boruta_lightgbm"],
    )
)
order_fs_cvmode = [
    f"{i}[{j}]"
    for i, j in product(
        ["lasso", "rfe_lasso", "kbest_mi", "boruta_rf", "boruta_lightgbm"],
        ["flat", "nested_mean"],
    )
]
order_dataset_frange = [
    "rectal[3-15-1]",
    "abus[3-15-1]",
    "pancreatic[3-15-1]",
    "pancreatic[5-40-5]",
]
order_fs = ["lasso", "rfe_lasso", "kbest_mi", "boruta_rf", "boruta_lightgbm"]

