import numpy as np
import matplotlib.pyplot as plt
import os

hyperopt_search_dict = {
    "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear', 'XGBoostTSF'},
    "pred_len": {6, 12, 18, 24}
}
project = 'population-level'
work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
sampling_rate_hz = 26
rmse_cutoff = 1.5
rmse_reference = 0.6  


model_display_names = {
    'TransformerTSFv2': 'Transformer',
    'XGBoostTSF': 'XGBoost',
}

output_dir = '/mnt/nas-wang/nas-ssd/Scripts/RMP/'
os.makedirs(output_dir, exist_ok=True)


for pred_len in sorted(hyperopt_search_dict['pred_len']):
    plt.figure(figsize=(12, 6))
    
    prediction_horizon_ms = int(pred_len * 1000 / sampling_rate_hz)

    model_share_values = {}

    for model in sorted(hyperopt_search_dict['model']):
        settings = f'{model}_pl{pred_len}'
        result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
        
        try:
            rmse_values = np.load(result_path)
            rmse_values = np.clip(rmse_values, None, rmse_cutoff)
            
            counts, bin_edges = np.histogram(rmse_values, bins=500, density=True)
            cdf = np.cumsum(counts) * np.diff(bin_edges)
            
            display_name = model_display_names.get(model, model)
            
            line, = plt.plot(bin_edges[1:], cdf, label=display_name, linewidth=2)
            
            x_vals = bin_edges[1:]
            y_vals = cdf
            idx = np.searchsorted(x_vals, rmse_reference) - 1
            
            if idx >= 0 and idx+1 < len(x_vals):
                x0, x1 = x_vals[idx], x_vals[idx+1]
                y0, y1 = y_vals[idx], y_vals[idx+1]
                share_value = y0 + (y1 - y0) * (rmse_reference - x0) / (x1 - x0)
                model_share_values[display_name] = share_value
                
                plt.scatter(rmse_reference, share_value, color=line.get_color(),
                          s=100, zorder=5, marker='o', edgecolor='white', linewidth=1)
            
        except FileNotFoundError:
            print(f"Warning: File not found - {result_path}")
            continue
    
    plt.axvline(x=rmse_reference, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    
    sorted_models = sorted(model_share_values.items(), key=lambda x: x[1], reverse=True)
    
    max_name_len = max(len(display_name) for display_name, _ in sorted_models)
    
    text_lines = []
    for display_name, share in sorted_models:
        name_padded = display_name.ljust(max_name_len)
        value_str = f"{share:.2f}".rjust(5)  
        text_lines.append(f"{name_padded} {value_str}")
    
    text_str = '\n'.join(text_lines)
    
    plt.text(rmse_reference + 0.08, 0.3, text_str, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'),
             fontsize=10, fontfamily='monospace', zorder=10)
    

    plt.xlabel('relative RMSE ', fontsize=12)
    plt.ylabel('share of signals', fontsize=12)
    plt.title(f'Latency: {prediction_horizon_ms}ms ({pred_len} samples)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')
    plt.xlim(0, rmse_cutoff)
    
    output_path = os.path.join(output_dir, f'rmse_cdf_pl{pred_len}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
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