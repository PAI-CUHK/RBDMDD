import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 

POSITIVE_LABEL = 'MDDRBD' 
NEGATIVE_LABEL = 'MDDonly'

def plot_feature_heatmap_style_c(X, y, feature_names, output_dir):
    
    combined_df = X.copy()
    combined_df['label'] = y
    
    sorted_df = combined_df.sort_values(by='label')
    X_sorted = sorted_df.drop('label', axis=1)
    y_sorted = sorted_df['label'].values
    
   
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X_sorted), 
                          columns=X_sorted.columns, index=X_sorted.index)
    X_plot = X_norm.T 
    
   
    n_neg = (y_sorted == 0).sum()
    n_pos = (y_sorted == 1).sum()
    total_samples = len(y_sorted)
    n_features = len(feature_names)

   
    thresholds = X.median(axis=0)
    
    prev_neg_list = [] 
    prev_pos_list = [] 
    
    X_neg = X[y == 0]
    X_pos = X[y == 1]
    
    
    for feat in feature_names:
        thresh = thresholds[feat]
        
        
        count_neg = (X_neg[feat] > thresh).sum()
        ratio_neg = count_neg / n_neg if n_neg > 0 else 0
        prev_neg_list.append(ratio_neg)
        
        
        count_pos = (X_pos[feat] > thresh).sum()
        ratio_pos = count_pos / n_pos if n_pos > 0 else 0
        prev_pos_list.append(ratio_pos)

    
    fig = plt.figure(figsize=(24, 12)) 
    
   
    gs = gridspec.GridSpec(2, 3, width_ratios=[6, 40, 1], height_ratios=[1, 20], 
                           wspace=0.05, hspace=0.01)
    
   
    ax_prev = plt.subplot(gs[1, 0])
    
    
    ax_top = plt.subplot(gs[0, 1])
    ax_main = plt.subplot(gs[1, 1])
    
    
    cbar_ax = plt.subplot(gs[1, 2])

   
    y_pos = np.arange(n_features) + 0.5 
    bar_height = 0.8
    
   
    ax_prev.invert_yaxis() 
    
   
    ax_prev.barh(y_pos, prev_neg_list, height=bar_height, color="#C84663", 
                 align='center', label=f'{NEGATIVE_LABEL}', edgecolor='none')
    
   
    
   
    ax_prev.cla() 
    ax_prev.invert_yaxis()
    
    
    ax_prev.barh(y_pos, prev_neg_list, height=bar_height/2, left=0, align='edge', 
                 color="#C84663", edgecolor='none')
    
    
    ax_prev.barh(y_pos, prev_neg_list, height=bar_height, left=0, 
                 color="#C84663", edgecolor='none') 
    
    ax_prev.barh(y_pos, prev_pos_list, height=bar_height, left=1.2, 
                 color="#4E6A89", edgecolor='none') 
    
    ax_prev.set_ylim(n_features, 0) 
    ax_prev.set_xlim(0, 2.3)
    
    ax_prev.set_yticks(y_pos)
    ax_prev.set_yticklabels(feature_names, fontsize=11, fontweight='bold')
    
    
    ax_prev.spines['top'].set_visible(False)
    ax_prev.spines['right'].set_visible(False)
    ax_prev.spines['bottom'].set_visible(False)
    ax_prev.spines['left'].set_visible(False)
    ax_prev.set_xticks([])
    
    
    # ax_prev.text(0.5, -0.5, "MDDonly", ha='center', va='bottom', fontsize=10, color="#C84663", fontweight='bold')
    # ax_prev.text(1.7, -0.5, "MDDRBD", ha='center', va='bottom', fontsize=10, color="#7F8C8D", fontweight='bold')
    ax_prev.set_title("Prevalence", fontsize=12, pad=20)


    
    color_neg = "#C84663" 
    color_pos = "#4E6A89" 

    ax_top.barh([0], [n_neg], left=[0], color=color_neg, height=1, edgecolor='none', align='center')
    ax_top.barh([0], [n_pos], left=[n_neg], color=color_pos, height=1, edgecolor='none', align='center')
    ax_top.axvline(x=n_neg, color='white', linewidth=2)

    ax_top.set_xlim(0, total_samples)
    ax_top.set_ylim(-0.5, 0.5)
    ax_top.axis('off')
    
    ax_top.text(n_neg/2, 0, f"{NEGATIVE_LABEL}\n(n={n_neg})", 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    ax_top.text(n_neg + n_pos/2, 0, f"{POSITIVE_LABEL}\n(n={n_pos})", 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)


   
    cmap_style_c = mcolors.LinearSegmentedColormap.from_list("WhiteRed", ["#FFFFFF", "#CB2027"])

    sns.heatmap(X_plot, 
                cmap=cmap_style_c, 
                cbar=True,
                cbar_ax=cbar_ax, 
                cbar_kws={'label': 'Normalized Feature Intensity'},
                xticklabels=False, 
                yticklabels=False, 
                ax=ax_main,
                linewidths=0, 
                rasterized=False
               )

    ax_main.set_xlim(0, total_samples)

    
    for x in range(1, total_samples):
        if x == n_neg:
            ax_main.axvline(x=x, color='black', linewidth=2)
        else:
            ax_main.axvline(x=x, color='white', linewidth=3) 

    ax_main.set_xlabel(f"Individual Samples (N={total_samples})", fontsize=13, fontweight='bold')
    
    
    for spine in ax_main.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    
    save_path = output_dir / "plots" / "mdd_rbd_Feature_Heatmap_With_Prevalence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()