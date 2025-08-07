import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-white')
sns.set_style({
    'axes.edgecolor': 'black',
    'grid.color': '0.9',
    'font.family': 'serif', 
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'savefig.dpi': 600,    
    'savefig.format': 'tiff' 
})

project_abbr = {
    'population-level': 'PL',
    'FT-4DCT': 'PS',
    'FT-4DCT_fine_tuning': 'PS-FT',
    'data-specific': 'DS',
    'data-specific_fine_tuning': 'DS-FT'
}

# 分组
grouped_projects = [
    ['population-level'], 
    ['FT-4DCT', 'FT-4DCT_fine_tuning'], 
    ['data-specific', 'data-specific_fine_tuning']
]
group_labels = ['PL', 'PS', 'DS']

# 同组颜色一致，子类深浅不同
group_colors = [
    ['#1f77b4'],                  # PL
    ['#ff7f0e', '#ffbb78'],     # PS
    ['#2ca02c', '#98df8a']      # DS
]

TARGET_MODEL = "XGBoostTSF"
sampling_rate_hz = 26
pred_lens = [6, 12, 18, 24]

work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)

for pred_len in pred_lens:
    fig, ax = plt.subplots(figsize=(4.2, 2.5))

    # 读取数据
    data = []
    for group in grouped_projects:
        for project in group:
            settings = f'{TARGET_MODEL}_pl{pred_len}'
            result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
            rmse = np.load(result_path)
            data.append(pd.DataFrame({'Project': project, 'RMSE': rmse}))
    df = pd.concat(data)

    # 布局
    positions = []
    tick_positions = []
    current_pos = 0
    width = 0.25
    gap = 0.4
    labels_for_legend = []
    colors_for_legend = []

    for g_idx, group in enumerate(grouped_projects):
        start_pos = current_pos
        for i, project in enumerate(group):
            positions.append(current_pos)
            current_pos += width
        tick_positions.append((start_pos + current_pos - width) / 2)
        current_pos += gap

    # 绘制箱线图
    box_data = [df[df['Project'] == p]['RMSE'] for group in grouped_projects for p in group]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        medianprops=dict(color='black', linewidth=1.2)
    )

    # 设置颜色 & 图例
    color_idx = 0
    for g_idx, group in enumerate(grouped_projects):
        for j, project in enumerate(group):
            box = bp['boxes'][color_idx]
            box.set_facecolor(group_colors[g_idx][j])
            box.set_alpha(0.8)
            labels_for_legend.append(project_abbr[project])
            colors_for_legend.append(group_colors[g_idx][j])
            color_idx += 1

    # 离群点
    for i, group in enumerate(grouped_projects):
        for j, project in enumerate(group):
            idx = sum(len(g) for g in grouped_projects[:i]) + j
            q1 = df[df['Project'] == project]['RMSE'].quantile(0.25)
            q3 = df[df['Project'] == project]['RMSE'].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df['Project'] == project) &
                          ((df['RMSE'] < q1 - 1.5 * iqr) | (df['RMSE'] > q3 + 1.5 * iqr))]
            outlier_y = np.where(outliers['RMSE'] > 2.5, 2.5, outliers['RMSE'])
            ax.plot([positions[idx]] * len(outliers), outlier_y, 'r+', markersize=4, markeredgewidth=0.8)

    # =============================
    # 新增：局部横虚线（仅覆盖每组的箱体范围）
    # =============================
    offset =  0.02
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        for g_idx, group in enumerate(grouped_projects):
            if np.isclose(y, 1.0):   # 跳过 y=1.0，避免绘制虚线覆盖基准线
                continue
            # 该组的最左和最右位置
            start_idx = sum(len(g) for g in grouped_projects[:g_idx])
            end_idx = start_idx + len(group) - 1
            start_x = positions[start_idx] - width * 0.5
            end_x = positions[end_idx] + width * 0.5
            ax.hlines(y, start_x, end_x, colors='gray', linestyles=':', linewidth=0.4, alpha=0.5)

    ax.set_ylim(0, 2.5)

    # 设定刻度
    ax.set_yticks(np.arange(0, 2.6, 0.5))
    ax.yaxis.set_tick_params(direction='in', length=3, width=1)

    # 基准线、标题、标签
    prediction_horizon_ms = int(pred_len * 1000 / sampling_rate_hz)

    
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels)


    ax.set_ylabel('Relative RMSE')
    ax.set_title(f'{TARGET_MODEL} (Latency: {prediction_horizon_ms}ms)', fontsize=10)

    # 图例
    # handles = [plt.Line2D([0], [0], color=c, lw=7) for c in colors_for_legend]
    #ax.legend(handles, labels_for_legend, loc='upper right', fontsize=6, frameon=False)

    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors_for_legend, labels_for_legend)]
    # 调用legend时传入patches列表
    ax.legend(handles=patches, loc='upper right',bbox_to_anchor=(1.2, 1), fontsize=6, frameon=False)


    # 保存
    output_path = os.path.join(output_dir, f'{TARGET_MODEL}_pl{pred_len}_grouped_2.tiff')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()



"""
This box plot shows the distribution of relative RMSE for our LSTM model across different project groups. 
Each box represents one group’s error distribution: 
the central line indicates the median,
the box edges show the interquartile range (IQR) which contains the middle 50% of the data, 
and the whiskers extend to the most extreme data points excluding outliers. 
Red plus signs mark outliers beyond 1.5 times the IQR.

We grouped the data into three main categories: Population-Level (PL), Patient-Specific (PS), and Data-Specific (DS). 
Within PS and DS, we further distinguish between the base models and fine-tuned  (denoted as PS-FT and DS-FT).

Across all prediction lengths, DS-FT performs best with the lowest and most stable errors. 
PL comes next, showing moderate performance. 
PS without fine-tuning and DS without fine-tuning perform worst, with higher errors and variability. 
This highlights the clear benefit of fine-tuning, especially for data-specific models.
"""