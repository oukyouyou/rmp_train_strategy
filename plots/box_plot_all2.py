import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

plt.style.use('seaborn-v0_8-white')

# 基色
base_colors = {
    'PL': '#1f77b4',  # 蓝
    'PS': '#ff7f0e',  # 橙
    'DS': '#2ca02c'   # 绿
}

projects = ['population-level', 'FT-4DCT', 'data-specific']
group_labels = ['PL', 'PS', 'DS']
pred_lens = [6, 12, 18, 24]
sampling_rate_hz = 26

work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)

# 渐变色生成函数
def generate_shades(base_color, n=4):
    r, g, b = to_rgb(base_color)
    shades = []
    for i in range(n):
        factor = 0.3 + 0.7 * (i / (n - 1))  # 深→浅
        shades.append((r + (1 - r) * (1 - factor),
                       g + (1 - g) * (1 - factor),
                       b + (1 - b) * (1 - factor)))
    return shades

# 载入所有数据到 DataFrame
all_data = []
for project, label in zip(projects, group_labels):
    for pred_len in pred_lens:
        settings = f'LSTM_pl{pred_len}'
        result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
        rmse = np.load(result_path)
        all_data.append(pd.DataFrame({
            'Project': label,
            'PredLen': pred_len,
            'RMSE': rmse
        }))
df = pd.concat(all_data)

# 绘图
fig, ax = plt.subplots(figsize=(6, 3))

# 布局：每组（PL/PS/DS）内部画 4 个箱体
width = 0.15
gap_between_groups = 0.5
positions = []
tick_positions = []
current_pos = 0
colors_for_legend = {}
patches_for_legend = []

for g_idx, group in enumerate(group_labels):
    shades = generate_shades(base_colors[group], len(pred_lens))
    start_pos = current_pos
    for i, pred_len in enumerate(pred_lens):
        positions.append(current_pos)
        # 数据
        vals = df[(df['Project'] == group) & (df['PredLen'] == pred_len)]['RMSE']
        bp = ax.boxplot(vals, positions=[current_pos], widths=width * 0.8,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        medianprops=dict(color='black', linewidth=1.2))
        bp['boxes'][0].set_facecolor(shades[i])
        bp['boxes'][0].set_alpha(0.8)
        current_pos += width
    tick_positions.append((start_pos + current_pos - width) / 2)
    current_pos += gap_between_groups

# Y轴、基准线
ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
ax.set_yticks(np.arange(0, 2.6, 0.5))
ax.set_ylim(0, 2.5)
ax.set_xticks(tick_positions)
ax.set_xticklabels(group_labels)
ax.set_ylabel('Relative RMSE')

# 图例：第一行显示 PredLen（颜色深浅），第二行显示组颜色
from matplotlib.lines import Line2D
shade_example = generate_shades('#000000', len(pred_lens))  # 黑色渐变示例
legend_predlen = [Line2D([0], [0], color=shade, lw=7) for shade in shade_example]
ax.legend(legend_predlen, [f'{pl} steps' for pl in pred_lens],
          title='Prediction Length', loc='upper right', fontsize=6, frameon=False)

plt.savefig(os.path.join(output_dir, 'combined_boxplot.tiff'), dpi=600, bbox_inches='tight')
plt.close()
