import os
from itertools import product, combinations_with_replacement

from IPython.display import display
import ipywidgets as widgets

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt; plt.rcParams['font.family'] = 'monospace'
import matplotlib as mpl; mpl.rcParams['figure.dpi'] = 100


from seaborn import displot, lmplot
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression


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
    # cv_results_ = pd.read_csv("data/d_cv_results_.csv", index_col=list(range(9)))

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

        # cvr = cv_results_.reset_index('undersample')
        # cvr = cvr.loc[cvr.undersample == False]
        # cvr.drop('undersample', axis=1, inplace=True)
        # cv_results_ = cvr

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


# ----------------------------------------- ordering of keys in plots
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
order_dataset_feat_range = [
    "rectal[3-15-1]",
    "abus[3-15-1]",
    "pancreatic[3-15-1]",
    "pancreatic[5-40-5]",
]
order_fs_method = [
    'lasso', 'rfe_lasso', 'kbest_mi', 'boruta_rf', 'boruta_lightgbm'
]
order_cvmode = ['flat', 'nested_mean']
order_dataset   = ['rectal', 'abus', 'pancreatic']
order_fs_method_cvmode = [f'{i}[{j}]'for i, j in product(order_fs_method, order_cvmode)]
order_dataset_feat_range__fs_method = list(product(
    order_dataset_feat_range,
    order_fs_method
))
order_algorithm = ['nb', 'knn', 'rf', 'svm', 'mlp', 'gbc', 'lr']



# ----------------------------------------- ipywidgets plotting
from scipy.stats import shapiro, ttest_ind, ttest_rel, wilcoxon

scores, feature_sets = load_data()

# multi-plotting
dataset_selector = widgets.SelectMultiple(
    options=order_dataset_feat_range,
    value=order_dataset_feat_range,
    description='Select datasets',
    disabled=False
)

multi_controls = dict(
    x = ['dataset[feat_range]', 'dataset[feat_range][fs_method]', 'fs_method', 'as_cv_step', 'incl_filtered', 'algorithm', 'undersample'],
    hue = ['fs_method', 'dataset[feat_range]', 'as_cv_step', 'incl_filtered', 'algorithm', 'undersample'],
    as_cv_step    = ['both', True, False],
    incl_filtered = ['both', True, False],
    
    metric = ['kappa', 'roc_auc', 'n_features[actual]'],
    stat   = ['test', 'cv_mean', 'diff', 'cv_std', 'ncv_val', 'ncv_diff'],
    model    = ['all', 'flat vs. nested', 'flat', 'ncv'],
    dataset = dataset_selector,

    plotter = [sns_barplot, sns_boxplot, sns_stripplot],
    rotate_xticks = False
)

def multi_plot(
    x, hue, plotter,
    as_cv_step='both', incl_filtered='both',
    metric='kappa', stat='test', model='all', dataset=order_dataset_feat_range,
    rotate_xticks=False
):

    global d, testies
    d = scores

    def hl_significant(val):
        '''
        highlight the maximum in a Series yellow.
        '''
        color = 'background-color: #33cc9e' if val < .05 else ''
        return color

    # metric and stat selection off the start
    if metric != 'n_features[actual]':
        d = d[(metric, stat)].reset_index().droplevel(1, axis=1)
    else:
        d = d[metric].reset_index()
    # proper ordering for easy reading
    d = (
        d.set_index(['dataset[feat_range]', 'fs_method'])
        .loc[order_dataset_feat_range__fs_method]
    )
    
    # selecte datasets....
    d = d.loc[list(dataset)].reset_index()
    
    
    # slice data & add to the title
    condition = np.array([True] * d.shape[0])
    axes_title = f'{metric} :: {stat}'
    if model == 'flat':
        axes_title += ' :: FlatCV'
    elif model == 'ncv':
        axes_title += ' :: NestedCV'
    axes_title += '\n'

    if as_cv_step != 'both' and x != 'as_cv_step':
        condition = ((condition) & (d['as_cv_step'] == as_cv_step))

        axes_title += f'\n{"feature selection mode: "}'
        axes_title += (
            'as pipeline cv-step' if as_cv_step
            else 'as preprocessing step'
        )

    if incl_filtered != 'both' and x != 'incl_filtered':
        condition = ((condition) & (d['incl_filtered'] == incl_filtered))
        
        axes_title += f'\n{"initial feature set: "}'
        axes_title += (
            'original features' if not incl_filtered
            else 'original + filtered features'
        )
    
    if len(dataset) != 4 and x != 'dataset[feat_range]':        
        axes_title += f'\ndatasets: {[da for da in dataset]}'

    d = d[condition]

    if model == 'flat vs. nested':

        # get a new slicing strategy: aggregate nested into mean
        df = d[d.model == 'flat']
        dn = d[d.model != 'flat'].groupby(list(d.columns[:-2]))[[metric]].mean().reset_index()
        dn['model'] = 'nested_mean'
        d = df.append(dn).rename(columns={'model': 'cvmode'})

        # append cvmode (created just above) to the hue categories
        hue_ = f'{hue}[cvmode]'
        d[hue_] = d[hue].astype(str) + '[' + d['cvmode'] + ']'

        ax = plotter(
            data=d,
            x=x,
            y=metric,
            hue=hue_,
            figsize=(10, 4),
            hue_order=order_fs_method_cvmode if hue == 'fs_method' else None
        ); ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), title=hue_)
        # ax.set_title(axes_title)
        if rotate_xticks:
            ax.tick_params(axis='x', labelrotation=90)

        def testies(group):
            group = group.set_index('cvmode')[metric]
            return pd.Series({
                'shapiro_flat': shapiro(group['flat']).pvalue,
                'shapiro_nested_mean': shapiro(group['nested_mean']).pvalue,
                'ttest_rel': ttest_rel(
                    group['flat'], group['nested_mean'],
                    alternative='greater'
                ).pvalue,
                'wilcoxon': wilcoxon(
                    group['flat'], group['nested_mean'],
                    alternative='greater'
                ).pvalue
            })

        tests = d.groupby([x, hue]).apply(testies).round(6)
        if x == 'dataset[feat_range]':
            ox = order_dataset_feat_range
            if hue == 'fs_method':
                oh = order_fs_method
                return tests.loc[list(product(ox, oh))].style.applymap(hl_significant)
        elif x == 'fs_method':
            ox = order_fs_method
            if hue == 'dataset[feat_range]':
                oh = order_dataset_feat_range
                return tests.loc[list(product(ox, oh))].style.applymap(hl_significant)

        tests = tests.dropna(how='all').dropna(how='all', axis=1)
        return tests.style.applymap(hl_significant)

    else:

        if model == 'flat':
            d = d[d.model == 'flat']
        elif model == 'ncv':
            d = d[d.model != 'flat']

        hue_order = order_fs_method if hue == 'fs_method' else None
        hue_order = order_dataset_feat_range if hue == 'dataset[feat_range]' else hue_order
            
        ax = plotter(
            data=d,
            x=x,
            y=metric,
            hue=hue,
            figsize=(10, 4),
            hue_order=hue_order
        )
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), title=hue)
        
        # ax.get_legend().remove()
        ax.set_title(axes_title)
        
        if rotate_xticks:
            ax.tick_params(axis='x', labelrotation=90)
        
        def test_hue_differences(df, hue, metric):
            cats = df[hue].unique()
            g = df.set_index(hue)
            wilcox_df = pd.DataFrame(index=cats, columns=cats)
            for cat1, cat2 in filter(lambda t: t[0] != t[1], combinations_with_replacement(cats, 2)):     
                wilcox_df.loc[cat1, cat2] = wilcoxon(g.loc[cat1, metric], g.loc[cat2, metric]).pvalue
            return wilcox_df
        
        testies = d.groupby(x).apply(test_hue_differences, hue, metric).round(6)
        if x == 'dataset[feat_range]':
            testies = testies.loc[set(order_dataset_feat_range).intersection(dataset)]
        testies = testies.dropna(how='all').dropna(how='all', axis=1)

        pd.set_option("display.precision", 6)
        display(testies)


results_plots_params = {
    'fig2': dict(x='dataset[feat_range]', hue='fs_method', as_cv_step='both', incl_filtered='both', plotter=sns_barplot),
    
    'fig3a': dict(x='dataset[feat_range]', hue='incl_filtered', plotter=sns_boxplot),
    'fig3b': dict(x='fs_method', hue='incl_filtered', dataset=['rectal[3-15-1]'], plotter=sns_boxplot),
    'fig3c': dict(x='fs_method', hue='incl_filtered', dataset=['abus[3-15-1]'], plotter=sns_boxplot),
    'fig3d': dict(x='fs_method', hue='incl_filtered', dataset=['pancreatic[3-15-1]', 'pancreatic[5-40-5]'], plotter=sns_boxplot),
    
    'fig4a': dict(x='dataset[feat_range]', hue='as_cv_step', plotter=sns_boxplot),
    'fig4b': dict(x='fs_method', hue='as_cv_step', dataset=['rectal[3-15-1]'], incl_filtered=False, plotter=sns_boxplot),
    'fig4c': dict(x='fs_method', hue='as_cv_step', dataset=['abus[3-15-1]'], plotter=sns_boxplot),
    'fig4d': dict(x='fs_method', hue='as_cv_step', dataset=['pancreatic[3-15-1]', 'pancreatic[5-40-5]'], plotter=sns_boxplot),
    
    'fig6': dict(x='dataset[feat_range]', hue='fs_method', plotter=sns_barplot, model='flat vs. nested', stat='test'),
    'fig7': dict(x='dataset[feat_range]', hue='fs_method', plotter=sns_barplot, model='flat vs. nested', stat='cv_mean')
}


def figure_5():

    d = (
        scores
            .set_index('n_features[actual]', append=True)[[('kappa', 'test'), ('roc_auc', 'test')]]
            .droplevel(1, 1).reset_index()
            .set_index(['dataset[feat_range]', 'fs_method'])#.loc[order_dataset_feat_range__fs_method]
            .reset_index().rename(columns={'dataset[feat_range]':'ds', 'fs_method':'fsm'})
    )

    fgrid = lmplot(
        data = d.set_index(['ds', 'fsm']).loc[order_dataset_feat_range__fs_method].reset_index(),
        x = 'n_features[actual]',
        row = 'fsm', col = 'ds', legend=False,
        y = 'kappa',
        hue='as_cv_step',
        aspect = 1.7, height=3,
        ci=None
    )


    def f_test(group):
        x = group[['n_features[actual]']]
        y = group['kappa']
        lr = LinearRegression().fit(x, y)
        r2 = lr.score(x, y)
        coef = lr.coef_[0]
        
        f, pval = f_regression(x, y)
        f = f[0]; pval = pval[0]
        
        return pd.Series([coef, r2, f, pval], index=['coef', 'R2', 'F', 'pval'])


    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_significant = s < .05
        return ['background-color: #33cc9e' if v else '' for v in is_significant]

    gb = d.set_index('as_cv_step').loc[True].groupby(['ds', 'fsm'])[['n_features[actual]', 'kappa']]
    f = gb.apply(f_test)#.loc[order_dataset_feat_range__fs_method]
    f.loc[:, 'pval'] = f['pval'].round(6)
    return f.loc[f.pval < .05]#.style.apply(highlight_max, subset='pval')



def table_6(metric='kappa'):
    cmetric = metric
    experimental_dims = scores.index.names[:-1]

    # ncv
    ncv = scores[[(cmetric, 'ncv_val')]]
    gb = ncv.groupby(experimental_dims)
    ncv = gb.agg(['mean', 'std']).droplevel(1, axis=1)
    ncv.columns.names = ['metric', 'st']
    ncv['performance_estimator'] = 'nestedcv'
    ncv = (
        ncv
        .set_index('performance_estimator', append=True)
        .droplevel('metric', axis=1)
        .unstack('performance_estimator')
        .swaplevel(axis=1)
    )

    # flt
    flt = scores[[(cmetric, 'cv_mean'), (cmetric, 'cv_std')]].reset_index('model')
    flt = (
        flt.loc[flt['model'] == 'flat']
        .rename(columns={'model':'performance_estimator'})
        .set_index('performance_estimator', append=True)
        .unstack('performance_estimator')
        .droplevel('metric', axis=1)
        .rename(lambda s: s.replace('cv_', ''), axis=1)
        .rename(lambda s: s.replace('flat', 'flatcv'), axis=1)
        .swaplevel(axis=1)
    )
    experimental_dims = experimental_dims[:-1]

    def ranker(group):
        src = group.columns.levels[0][0]
        g = (
            group
            .sort_values([(src, st) for st in ['mean', 'std']], ascending=[False, True])
        )
        g[(src, 'rank')] = pd.Series(np.arange(1, g.shape[0]+1), index=g.index)
        g = g.reset_index('algorithm').set_index('algorithm')
        return g
        
    ncv = ncv.groupby(experimental_dims).apply(ranker)
    flt = flt.groupby(experimental_dims).apply(ranker)
    ranked = flt.join(ncv)

    from scipy.stats import kendalltau

    def equal_first_place(group):
        global a
        a = group.reset_index('algorithm')
        res = a.loc[a['flatcv'] == 1, 'algorithm'] \
            == a.loc[a['nestedcv'] == 1, 'algorithm']
        res = res.values[0]
        return res

    def choices(group):
        a = group.reset_index('algorithm')
        return a.loc[a['nestedcv'] == 1, 'algorithm'].values[0], a.loc[a['flatcv'] == 1, 'algorithm'].values[0]

    def equal_top_set(group, topn=3):
        pass

    def rank_analyzer(group):
        kt = kendalltau(group['flatcv'], group['nestedcv'])
        efp = equal_first_place(group)
        ncv_choice, flatcv_choice = choices(group)
        return pd.Series(
            [kt.correlation, kt.pvalue, efp, ncv_choice, flatcv_choice],
            index=['tau', 'pval', 'equal_first_place', 'ncv_choice', 'flatcv_choice']
        )


    gb = ranked.swaplevel(axis=1)['rank'].groupby(experimental_dims)
    ranked_analyzed = gb.apply(rank_analyzer)

    return ranked, ranked_analyzed


# ---------------------------------------------------- table 7
def table_7(metric='kappa', topn=5):

    def filterer(group, metric=('kappa', 'test'), topn=topn):
        global g
        g = group
        g = group.sort_values(by=metric, ascending=False).head(topn).droplevel(0)
        return g[[(m, n) for m, n in product((metric[0],), ('test', 'cv_mean', 'ncv_val'))]].join(g['n_features[actual]'])

    _, ranked_analyzed = table_6(metric)

    gb = scores.groupby('dataset[feat_range]')
    top = gb.apply(filterer).loc[order_dataset_feat_range]
    top.join(ranked_analyzed[['flatcv_choice', 'ncv_choice']]).loc[order_dataset_feat_range]\
        .reset_index(['algorithm', 'model'])#.drop('n_features[actual]', axis=1)
    return top



# ----------------------------------------------------------------------------- heatmaps
heatmap_options = dict(
    x   = ['algorithm', 'fs_method', 'as_cv_step', 'incl_filtered', 'undersample'],
    y = ['fs_method', 'as_cv_step', 'incl_filtered', 'algorithm', 'undersample'],
    as_cv_step    = [True, 'both', False],
    incl_filtered = [False, 'both', True],
    metric = ['kappa', 'roc_auc'],
    stat   = ['test', 'mean', 'diff', 'std', 'ncv_val', 'ncv_diff'],
    algorithm    = ['all', 'flat', 'ncv'],
    dataset = order_dataset_feat_range
)

def heatmap_plot(x, y, as_cv_step, incl_filtered, metric, stat, algorithm, dataset):
    global d
    

    def hl_significant(val):
        '''
        highlight the maximum in a Series yellow.
        '''
        color = 'background-color: #33cc9e' if val < .05 else ''
        return color


    # metric and stat selection off the start
    d = scores[(metric, stat)].reset_index().droplevel(1, axis=1)
    # proper ordering for easy reading
    d = (
        d.set_index(['dataset[feat_range]', 'fs_method'])
        .loc[order_dataset_feat_range__fs_method].reset_index()
    )    
    
    # slice data & add to the title
    condition = np.array([True] * d.shape[0])
    axes_title = f'{metric} :: {stat}'
    if algorithm == 'flat':
        axes_title += ' :: FlatCV'
    elif algorithm == 'ncv':
        axes_title += ' :: NestedCV'
    axes_title += '\n'
    sufftitle = ''
    
    if as_cv_step != 'both' and x != 'as_cv_step':
        condition = ((condition) & (d['as_cv_step'] == as_cv_step))

        axes_title += f'\n{"feature selection mode: "}'
        axes_title += (
            'as pipeline cv-step' if as_cv_step
            else 'as preprocessing step'
        )
        sufftitle += f'\n{"feature selection mode: "}'
        sufftitle += (
            'as pipeline cv-step' if as_cv_step
            else 'as preprocessing step'
        )

    if incl_filtered != 'both' and x != 'incl_filtered':
        condition = ((condition) & (d['incl_filtered'] == incl_filtered))
        
        axes_title += f'\n{"initial feature set: "}'
        axes_title += (
            'original features' if not incl_filtered
            else 'original + filtered features'
        )
        sufftitle += f'\n{"initial feature set: "}'
        sufftitle += (
            'original features' if not incl_filtered
            else 'original + filtered features'
        )

    d = d[condition]
    
    global plotting_data
    plotting_data = None

    if algorithm == 'flat':
        d = d[d.algorithm == 'flat']
    elif algorithm == 'ncv':
        d = d[d.algorithm != 'flat']


    plotting_data = pd.pivot_table(d, index=['dataset[feat_range]', x], columns=y, values=metric, aggfunc='median')

    # misc ordering
    if y == 'fs_method':
        plotting_data = plotting_data[order_fs_method]

    if x == 'algorithm':
        o = list(product(order_dataset_feat_range, order_algorithm))
        plotting_data = plotting_data.loc[o]
    
    ny, nx = plotting_data.shape  # number of entries in each axis
    fig, axes = plt.subplots(figsize=(.3*ny, 3 + .3*nx), sharey=True)
    # fig.suptitle(axes_title)

    sns.heatmap(
        plotting_data.loc[dataset].transpose(),
        cmap=sns.cubehelix_palette(n_colors=50, reverse=True),
        annot=True,
        fmt='.3f',  # this rounds the values :'(
        ax=axes,
        linewidth=.3,
        linecolor='white'
    )
    axes.set_title(f'{dataset}{sufftitle}')
    plt.yticks(rotation=0)
    fig.tight_layout()



# ------------------------------------------------------------------ radiomic signatures

rs_plot_options = dict(
    dataset=order_dataset_feat_range,
    algorithm=widgets.SelectMultiple(
        options=['ncv', 'flat', 'outer'], rows=3, value=['ncv', 'flat']
    ),
    top_features=widgets.IntSlider(value=3, min=1, max=7)
)

def rs_plot(dataset, algorithm, top_features=3):

    def topf(df, topn=3):
        return df.transpose().iloc[1:, 0].nlargest(topn)

    """
    fsr: calculating percentages ('feature sets relative')
    """
    fsr = (
        feature_sets.reset_index()
    )
    # selecting the source...
    bidx = fsr.model.map(lambda s: any([sr in s for sr in algorithm]))
    fsr = (
        # slicing to selected sources
        fsr.loc[bidx]
        # as_cv_step is only True
        # and we want the percentages from across all algorithms and selected sources
        .drop(['as_cv_step', 'algorithm', 'algorithm'], axis=1)
        # percentage calculation
        .groupby(['dataset[feat_range]', 'fs_method', 'incl_filtered'])
        .agg(lambda df: df.count() / df.shape[0])
        # reordering and reindexing
        .reset_index(['incl_filtered'])
        .loc[order_dataset_feat_range__fs_method]
        .set_index(['incl_filtered'], append=True)
    )
    
    """
    tops: plotting data for the heatmaps
    """
    tops = fsr.groupby(['dataset[feat_range]', 'fs_method', 'incl_filtered']).apply(topf).unstack('feature')
    tops = tops.loc[dataset]
    tops = tops.loc[:, (tops.notna().any())].transpose().sort_index()[[
        'lasso', 'rfe_lasso', 'kbest_mi', 'boruta_rf', 'boruta_lightgbm'
    ]]
    
    original_filter = lambda s: 'original' in s
    filtered_filter = lambda s: 'original' not in s

    cols = list(product((False, True), ('lasso', 'rfe_lasso', 'kbest_mi', 'boruta_rf', 'boruta_lightgbm')))

    tops = (
        tops
        .loc[
            sorted(list(filter(filtered_filter, tops.index)))
            + sorted(list(filter(original_filter, tops.index)))
        ]
        .swaplevel(axis=1)[cols]
    )

    tops.columns = tops.columns.set_levels(tops.columns.levels[0].map(lambda b: '[orgnl+fltr]' if b else '[orgnl]'), level=0)
    tops.columns = tops.columns.set_levels(tops.columns.levels[1].map(lambda s: f'{s}'), level=1)
    tops.columns.names = ['initial feature set ', ' feature selection method']
    
    nx, ny = tops.shape  # number of entries in each axis
    fig, ax = plt.subplots(figsize=(.9*ny, 2 + .4*nx))
    title_cv_mode = ''; cvm_map = {'flat': 'FlatCV', 'ncv':'NestedCV', 'outer':'outer'}
    for source in algorithm:
        title_cv_mode += f' {cvm_map[source]}'
    ax.set_title(f'Top 3 from {dataset} :: {title_cv_mode}')
    sns.set_style('darkgrid')
    return sns.heatmap(
        tops,
        cmap=sns.cubehelix_palette(n_colors=50, reverse=True),
        annot=True,
        fmt='.3f',  # this rounds the values :'(
        ax=ax,
        linewidth=.3
    )    
