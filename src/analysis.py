"""
Functionality for visualization and interpretability.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_feature_channel_weights(model):
    with torch.no_grad():
        w = model.depthwise_conv.weight.detach().cpu().numpy()
    return w.squeeze(3).squeeze(1)

def get_channel_link_matrix(model):
    W_fc = get_feature_channel_weights(model) # (F2, C)
    W_ch = W_fc.T
    return np.corrcoef(W_ch)

def plot_channel_link_matrix(corr, brain_regions, title, tick_step=5, save_path=None):
    C = len(brain_regions)
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(title)
    
    ticks = np.arange(0, C, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([brain_regions[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([brain_regions[i] for i in ticks], fontsize=7)
    
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_cv_accuracy_over_epochs(fold_histories, metric="test_acc", title=None, save_path=None):
    all_acc = []
    for fid in fold_histories:
        acc = np.asarray(fold_histories[fid][metric], dtype=float)
        all_acc.append(acc)

    max_epochs = max(len(a) for a in all_acc)
    acc_mat = np.full((len(all_acc), max_epochs), np.nan)
    for i, a in enumerate(all_acc):
        acc_mat[i, :len(a)] = a

    plt.figure(figsize=(8, 5))
    for i, a in enumerate(all_acc):
        ep = np.arange(1, len(a) + 1)
        plt.plot(ep, a, alpha=0.3)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    if title:
        plt.title(title)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()