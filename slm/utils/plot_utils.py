from typing import Dict, Optional
import io
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy import interpolate


def fig2img(plt_fig):
    buf = io.BytesIO()
    plt_fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img  # Image object

##################################################
FONTSIZE = 18
FIG_DPI = 300

x = np.linspace(-1.5, 1, 100)
y = np.linspace(-0.5, 2, 100)
X, Y = np.meshgrid(x, y)
##################################################

def lineplot(data, save_to=None, xlabel=None, ylabel=None, figsize=None):
    figsize = figsize if figsize else (50, 10)
    fig, _ = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        return save_to
    return fig

def boxplot(data, save_to=None, xlabel=None, ylabel=None, figsize=None):
    figsize = figsize if figsize else (10, 10)
    fig, _ = plt.subplots(1, 1, figsize=figsize)
    ax = sns.boxplot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        return save_to
    return fig

def heatmap(data, save_to=None):
    print(f">>> Plotting heatmap in 1D space. Image save to {save_to}")
    fig = plt.figure(figsize=(15, 3))
    sns.heatmap(data, cmap="mako", annot=False)
    
    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        return save_to
    return fig


def scatterplot_2d(
    data_dict: Dict, 
    save_to: str = None,  
    ref_key: str = 'target',
    xlabel: str = 'tIC1',
    ylabel: str = 'tIC2',
    n_max_point: int = 1000,
    pop_ref: bool = False,
    xylim_key: bool = 'PDB_clusters', 
    plot_kde: bool = False,
    density_mapping: Optional[Dict] = None,
    remarks: str = None,
):
    # configure min max
    if xylim_key and xylim_key in data_dict:
        xylim = data_dict.pop(xylim_key)
        # plot
        x_max = max(xylim[:,0]) 
        x_min = min(xylim[:,0]) 
        y_max = max(xylim[:,1]) 
        y_min = min(xylim[:,1])
    else:
        xylim = None
        x_max = max(data_dict[ref_key][:,0]) 
        x_min = min(data_dict[ref_key][:,0]) 
        y_max = max(data_dict[ref_key][:,1]) 
        y_min = min(data_dict[ref_key][:,1])
        
    # Add margin.
    x_min -= (x_max - x_min)/5.0
    x_max += (x_max - x_min)/5.0
    y_min -= (y_max - y_min)/5.0
    y_max += (y_max - y_min)/5.0
    
    # Remove reference data to save time.
    if pop_ref:
        data_dict.pop(ref_key)
        
    # Configure subplots.
    plot_n_row = len(data_dict) // 5 if len(data_dict) > 5 else 1  # at most 6 columns
    plot_n_columns = len(data_dict) // plot_n_row if len(data_dict) > 5 else len(data_dict)
    plot_n_row += 1 
    fig = plt.figure(figsize=(6 * plot_n_columns, plot_n_row * 6))

    i = 0
    for k, v in data_dict.items():
        i += 1 
        plt.subplot(plot_n_row, plot_n_columns, i)
        
        if k != ref_key and v.shape[0] > n_max_point:    # subsample for visualize
            # idx = np.random.choice(v.shape[0], n_max_point, replace=False)
            # equal stride
            idx = np.arange(0, v.shape[0], v.shape[0]//n_max_point)
            v = v[idx]
        
        if v.shape[0] < v.shape[1]:
            print(f"Warning: {k} has more dimensions than samples, using uniform density.")
            density = np.ones_like(v[:,0])
            density /= density.sum()
        else:
            cov = np.transpose(v)
            density = gaussian_kde(cov)(cov)
        
        # Optional precomputed density mapping.
        if density_mapping and k in density_mapping:
            density = density_mapping[k]

        plt.scatter(v[:, 0], v[:,1], s=10, alpha=0.7, c=density, cmap="mako_r", vmin=-0.05, vmax=0.40)
        # sns.scatterplot(x=v[:, 0], y=v[:,1], s=10, alpha=0.7, c=density, cmap="mako_r", vmin=-0.05, vmax=0.40)
        
        if plot_kde:
            sns.kdeplot(x=data_dict[ref_key][:, 0], y=data_dict[ref_key][:,1])    # landscape
        
        if xylim is not None:
            plt.scatter(xylim[:,0], xylim[:,1], s=40, marker="o", c="none", edgecolors="tab:red")   # cluster centers
        
        plt.xlabel(xlabel, fontsize=FONTSIZE, fontfamily="sans-serif")
        if (i-1) % plot_n_columns == 0:
            plt.ylabel(ylabel, fontsize=FONTSIZE, fontfamily="sans-serif")
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        title = k
        if remarks and k in remarks:
            title += f" {remarks[k]}"
        plt.title(title, fontsize=FONTSIZE, fontfamily="sans-serif")
                
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        return save_to
    return fig


def scatterplot_apo(x, y, save_to=None, xlabel=None, ylabel=None, regplot=False):
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("Invalid input data for scatter plot.")
        
    fig = plt.figure(figsize=(10, 8))
    if regplot:
        sns.regplot(x=x, y=y, color='steelblue', scatter_kws={'s': 10, 'alpha': 0.8, 'edgecolor': 'k'})
    else:     
        # Create scatter plot
        sns.scatterplot(x=x, y=y, color='steelblue', alpha=0.8, edgecolor='k')

        # Add reference line
        grid_x = np.linspace(0, 1, 100)
        plt.plot(grid_x, grid_x / 2 + 0.5, color='red', linestyle='--')

    # Set plot title and axis labels
    xlabel = xlabel if xlabel else "TM_conf1/conf2"
    ylabel = ylabel if ylabel else "TM_ensemble"
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
        
    # Set plot limits and ticks
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        return save_to
    return fig
