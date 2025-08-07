import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'serif', 
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8, 
    'savefig.dpi': 600,    
    'savefig.format': 'tiff' 
})

hyperopt_search_dict = {
    "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear', 'XGBoostTSF'},
    "pred_len": {6, 12, 18, 24}
}


model_display_names = {
    'TransformerTSFv2': 'Transformer',
    'XGBoostTSF': 'XGBoost',
}

model_colors = {
    'Transformer': '#1f77b4',  
    'XGBoost': '#ff7f0e',    
    'LSTM': '#2ca02c',        
    'MLP': '#d62728',        
    'DLinear': '#9467bd'    
}

project = 'population-level'
sampling_rate_hz = 26
rmse_cutoff = 1.5
rmse_reference = 0.6  
work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)


for pred_len in hyperopt_search_dict['pred_len']:
    fig, ax = plt.subplots(figsize=(10, 5)) 
    
    prediction_horizon_ms = int(pred_len * 1000 / sampling_rate_hz)
    model_share_values = {}

    for model in hyperopt_search_dict['model']:
        settings = f'{model}_pl{pred_len}'
        result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
        
        try:
            rmse_values = np.load(result_path)
            rmse_values = np.clip(rmse_values, None, rmse_cutoff)
            
            counts, bin_edges = np.histogram(rmse_values, bins=200, density=True)
            cdf = np.cumsum(counts) * np.diff(bin_edges)
            
            display_name = model_display_names.get(model, model)
            color = model_colors.get(display_name, '#7f7f7f')
            
            line, = ax.plot(bin_edges[1:], cdf, 
                          label=display_name, 
                          color=color,
                          linewidth=1.8)
            
            x_vals = bin_edges[1:]
            y_vals = cdf
            idx = np.searchsorted(x_vals, rmse_reference) - 1
            
            if idx >= 0 and idx+1 < len(x_vals):
                x0, x1 = x_vals[idx], x_vals[idx+1]
                y0, y1 = y_vals[idx], y_vals[idx+1]
                share_value = y0 + (y1 - y0) * (rmse_reference - x0) / (x1 - x0)
                model_share_values[display_name] = share_value

                ax.scatter(rmse_reference, share_value, 
                         color=color,
                         s=80, 
                         marker='o',
                         edgecolor='white',
                         linewidth=0.8,
                         zorder=5)
                
        except FileNotFoundError:
            print(f"Warning: File not found - {result_path}")
            continue
    
    ax.axvline(x=rmse_reference, 
              color='#333333', 
              linestyle=':', 
              linewidth=1.2,
              zorder=1)
    
    sorted_models = sorted(model_share_values.items(), key=lambda x: x[1], reverse=True)
    
    max_name_len = max(len(name) for name, _ in sorted_models)

    text_lines = []
    for name, value in sorted_models:
        text_lines.append(f"{name.ljust(max_name_len)}: {value:.2f}")
    
    annotation_text = '\n'.join(text_lines)
    
    ax.text(rmse_reference + 0.1, 0.25, annotation_text,
           bbox=dict(facecolor='white', 
                    alpha=0.95,
                    edgecolor='0.8',
                    boxstyle='round,pad=0.5'),
           fontfamily='serif',
           fontsize=9,
           linespacing=1.5,
           zorder=10)
    
    
    # plt.text(rmse_reference + 0.08, 0.3, text_str, 
    #          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'),
    #          fontsize=10, fontfamily='monospace', zorder=10)
    
    ax.set_xlabel('relative RMSE', labelpad=5)
    ax.set_ylabel('share of signals', labelpad=5)
    ax.set_title(f'Latency: {prediction_horizon_ms}ms ({pred_len} samples)', pad=12)
    ax.set_xlim(0, rmse_cutoff)
    ax.set_ylim(0, 1.05)
    

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True, 
             framealpha=0.9, 
             loc='lower right',

             borderpad=0.5)
    output_path = os.path.join(output_dir, f'rmse_cdf_pl{pred_len}.tiff')
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    print(f"Saved publication-ready figure: {output_path}")
    plt.close()



"""
    *"This cumulative distribution plot shows the relative RMSE performance across different models at a prediction horizon of [X]ms. The vertical dashed line marks RMSE=0.6, where we see [Transformer] achieves the highest share (XX%) of predictions below this threshold, followed by [XGBoost] at XX%. The curve steepness indicates how quickly each model accumulates accurate predictions - steeper curves represent more consistent performance."*

Key points to adapt:

Fill in [X] with actual latency (e.g., 462ms for pl=12)
Replace XX% with actual values from your plot
Highlight the top 2-3 models from your results
Optional additions:

"All RMSE values are clipped at 1.5 for better visualization."
*"Marker positions on the reference line show exact performance at RMSE=0.6."*
Would you like me to adjust the technical level or focus on any specific aspect?

"""