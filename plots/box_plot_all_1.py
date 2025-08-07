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
inner_labels = ['PL', 'PS', 'DS']  # 每个 pred_len 的子组标签

# 颜色：同组一致，FT版用浅色
group_colors = [
    ['#1f77b4'],                  # PL
    ['#ff7f0e', '#ffbb78'],       # PS
    ['#2ca02c', '#98df8a']        # DS
]

TARGET_MODEL = "LSTM"
sampling_rate_hz = 26
pred_lens = [6, 12, 18, 24]

work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)

# ========== 合并所有数据 ==========
data = []
for pred_len in pred_lens:
    for group in grouped_projects:
        for project in group:
            settings = f'{TARGET_MODEL}_pl{pred_len}'
            result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
            rmse = np.load(result_path)
            data.append(pd.DataFrame({
                'Project': project,
                'RMSE': rmse,
                'PredLen': pred_len
            }))
df = pd.concat(data)

# ========== 绘图 ==========
fig, ax = plt.subplots(figsize=(7.5, 3.0))  # 横向放宽

# 布局参数
width = 0.2
gap_inner = 0.05  # 同 pred_len 下各组的间距
gap_outer = 0.5   # 不同 pred_len 组之间的间距
positions = []
tick_positions = []
xtick_labels = []
current_pos = 0

# 计算每个箱子的位置
for pred_len in pred_lens:
    start_pos = current_pos
    for group in grouped_projects:
        for _ in group:
            positions.append(current_pos)
            current_pos += width + gap_inner
    tick_positions.append((start_pos + current_pos - gap_inner - width) / 2)
    xtick_labels.append(f'{pred_len}')
    current_pos += gap_outer

# 绘制箱线图
box_data = []
for pred_len in pred_lens:
    for group in grouped_projects:
        for project in group:
            vals = df[(df['Project'] == project) & (df['PredLen'] == pred_len)]['RMSE']
            box_data.append(vals)

bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=width * 0.9,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(linewidth=0.8),
    whiskerprops=dict(linewidth=0.8),
    medianprops=dict(color='black', linewidth=1.2)
)

# 颜色
color_idx = 0
colors_for_legend = []
labels_for_legend = []
for _ in pred_lens:
    for g_idx, group in enumerate(grouped_projects):
        for j, project in enumerate(group):
            bp['boxes'][color_idx].set_facecolor(group_colors[g_idx][j])
            bp['boxes'][color_idx].set_alpha(0.8)
            color_idx += 1
# Legend（只保留一份）
for g_idx, group in enumerate(grouped_projects):
    for j, project in enumerate(group):
        labels_for_legend.append(project_abbr[project])
        colors_for_legend.append(group_colors[g_idx][j])

# 离群点标注
ylim_top = 2.7
pos_idx = 0
for pred_len in pred_lens:
    for group in grouped_projects:
        for project in group:
            vals = df[(df['Project'] == project) & (df['PredLen'] == pred_len)]['RMSE']
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            outliers = vals[(vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)]
            in_range = outliers[outliers <= ylim_top]
            ax.plot([positions[pos_idx]] * len(in_range), in_range, 'r+', markersize=4, markeredgewidth=0.8)

            above_limit = (vals > ylim_top).sum()
            if above_limit > 0:
                ax.annotate(f'({above_limit})',
                            xy=(positions[pos_idx], ylim_top - 0.05),
                            ha='center', va='top',
                            fontsize=5, color='black')
            pos_idx += 1

# 横虚线
y_ticks = np.arange(0, ylim_top + 0.1, 0.5)
for y in y_ticks:
    if np.isclose(y, 1.0):
        continue
    for i in range(len(pred_lens)):
        block_start = i * (sum(len(g) for g in grouped_projects) + 2)  # 近似分块
    ax.hlines(y, positions[0] - width, positions[-1] + width, colors='gray', linestyles=':', linewidth=0.4, alpha=0.6)

# 坐标轴
ax.set_ylim(0, ylim_top)
ax.set_yticks(y_ticks)
ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
ax.set_xticks(tick_positions)

ax.set_xticklabels([f'{int(int(pl) * 1000 / sampling_rate_hz)}ms' for pl in xtick_labels])

ax.set_ylabel('relative RMSE')
ax.set_xlabel('Prediction horizon')

# 图例
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_for_legend, labels_for_legend)]
ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.12, 1), fontsize=6, frameon=False)

# 保存
output_path = os.path.join(output_dir, f'{TARGET_MODEL}_all_predlen_boxplot.tiff')
plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
plt.close()
