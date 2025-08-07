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

# 颜色：同组一致，FT版用浅色
group_colors = [
    ['#1f77b4'],                 # PL
    ['#ff7f0e', '#ffbb78'],      # PS
    ['#2ca02c', '#98df8a']       # DS
]

TARGET_MODEL = "MLP"
sampling_rate_hz = 26
pred_lens = [6, 12, 18, 24]

work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)

for pred_len in pred_lens:
    fig, ax = plt.subplots(figsize=(4.2, 2.5))

    # 数据加载
    data = []
    for group in grouped_projects:
        for project in group:
            settings = f'{TARGET_MODEL}_pl{pred_len}'
            result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
            rmse = np.load(result_path)
            data.append(pd.DataFrame({'Project': project, 'RMSE': rmse}))
    df = pd.concat(data)

    # 布局计算
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

    # 颜色和图例
    color_idx = 0
    for g_idx, group in enumerate(grouped_projects):
        for j, project in enumerate(group):
            box = bp['boxes'][color_idx]
            box.set_facecolor(group_colors[g_idx][j])
            box.set_alpha(0.8)
            labels_for_legend.append(project_abbr[project])
            colors_for_legend.append(group_colors[g_idx][j])
            color_idx += 1

    # 离群点处理 + 超出箭头
    ylim_top = 2.7
    for i, group in enumerate(grouped_projects):
        for j, project in enumerate(group):
            idx = sum(len(g) for g in grouped_projects[:i]) + j
            vals = df[df['Project'] == project]['RMSE']

            # IQR离群点（在ylim内）
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            outliers = vals[(vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)]
            in_range_outliers = outliers[outliers <= ylim_top]
            ax.plot([positions[idx]] * len(in_range_outliers), in_range_outliers, 'r+', markersize=4, markeredgewidth=0.8)

            # 超出 ylim 的离群点数量
            above_limit = (vals > ylim_top).sum()
            if above_limit > 0:
                ax.annotate(f'({above_limit})',
                            xy=(positions[idx], ylim_top - 0.02),
                            ha='center', va='top',
                            fontsize=4, color='black',
                            #arrowprops=dict(arrowstyle='-|>', color='black', lw=0.4)
                            )

    # =========================
    # 横向虚线（仅在每组宽度范围内）
    # =========================
    y_ticks = np.arange(0, ylim_top + 0.1, 0.5)
    for y in y_ticks:
        if np.isclose(y, 1.0):  # 跳过基准线
            continue
        for g_idx, group in enumerate(grouped_projects):
            start_idx = sum(len(g) for g in grouped_projects[:g_idx])
            end_idx = start_idx + len(group) - 1
            start_x = positions[start_idx] - width * 0.5
            end_x = positions[end_idx] + width * 0.5
            ax.hlines(y, start_x, end_x, colors='gray', linestyles=':', linewidth=0.6, alpha=0.7)

    # 基准线 & 坐标轴设置
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(y_ticks)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Relative RMSE')

    prediction_horizon_ms = int(pred_len * 1000 / sampling_rate_hz)
    ax.set_title(f'{TARGET_MODEL} (Latency: {prediction_horizon_ms}ms)', fontsize=10)

    # 图例（右侧偏移）
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors_for_legend, labels_for_legend)]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=6, frameon=False)

    # 保存
    output_path = os.path.join(output_dir, f'{TARGET_MODEL}_pl{pred_len}_grouped_arrow.tiff')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()
