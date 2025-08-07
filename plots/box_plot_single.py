import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

model_display_names = {
    'TransformerTSFv2': 'Transformer',
    'XGBoostTSF': 'XGBoost',
}
project_abbr = {
    'population-level': 'PL',
    'FT-4DCT': 'PS',
    'data-specific': 'DS',
    'FT-4DCT_fine_tuning': 'PS-FT',
    'data-specific_fine_tuning': 'DS-FT'
}
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

# 配置参数
model_display_names = {
    'TransformerTSFv2': 'Transformer',
    'XGBoostTSF': 'XGBoost',
}
project_abbr = {
    'population-level': 'PL',
    'FT-4DCT': 'PS',
    'data-specific': 'DS',
    'FT-4DCT_fine_tuning': 'PS-FT',
    'data-specific_fine_tuning': 'DS-FT'
}

TARGET_MODEL = "LSTM"
sampling_rate_hz = 26
projects = ['population-level', 'FT-4DCT', 'FT-4DCT_fine_tuning', 'data-specific', 'data-specific_fine_tuning']
pred_lens = [6, 12, 18, 24]
group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每组一个颜色

work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)

for pred_len in pred_lens:
    fig, ax = plt.subplots(figsize=(5, 3))  # 稍微加宽以适应分组
    
    # 加载数据
    data = []
    for project in projects:
        settings = f'{TARGET_MODEL}_pl{pred_len}'
        result_path = os.path.join(work_space, project, settings, 'test_results', 're_rmse_list.npy')
        try:
            rmse = np.load(result_path)
            # 创建分组信息
            if project == 'population-level':
                group = 'PL'
                subgroup = 'PL'
            elif 'FT-4DCT' in project:
                group = 'PS'
                subgroup = 'PS-FT' if '_fine_tuning' in project else 'PS'
            else:
                group = 'DS'
                subgroup = 'DS-FT' if '_fine_tuning' in project else 'DS'
                
            data.append(pd.DataFrame({
                'Project': project,
                'RMSE': rmse,
                'Group': group,
                'Subgroup': subgroup
            }))
        except Exception as e:
            print(f"Error loading {result_path}: {e}")
            continue
    
    if not data:
        continue
        
    df = pd.concat(data)
    
    # 分组箱线图 - 使用hue参数分组
    sns.boxplot(
        x='Group',
        y='RMSE',
        hue='Subgroup',
        data=df,
        palette={'PL': group_colors[0], 
                'PS': group_colors[1],
                'PS-FT': sns.desaturate(group_colors[1], 0.6),  # 同组颜色变体
                'DS': group_colors[2],
                'DS-FT': sns.desaturate(group_colors[2], 0.6)},
        width=0.6,
        linewidth=0.8,
        flierprops=dict(marker='+', markersize=4, markeredgewidth=0.8),
        ax=ax
    )
    
    # 手动添加压缩的离群点（保持原有逻辑）
    for i, group in enumerate(['PL', 'PS', 'DS']):
        for j, subgroup in enumerate(['', '-FT']):
            if group == 'PL' and j > 0:  # PL没有FT变体
                continue
            project_name = group if not subgroup else f"{group.split('-')[0]}_fine_tuning"
            if project_name not in project_abbr.values():
                continue
                
            q1 = df[df['Subgroup']==group+subgroup]['RMSE'].quantile(0.25)
            q3 = df[df['Subgroup']==group+subgroup]['RMSE'].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df['Subgroup']==group+subgroup) & 
                        ((df['RMSE'] < q1-1.5*iqr) | (df['RMSE'] > q3+1.5*iqr))]
            
            x_pos = i + (0.2 if subgroup else -0.2)  # 调整位置
            y_values = np.minimum(outliers['RMSE'], 2.5)
            ax.plot([x_pos]*len(outliers), y_values, 'r+', markersize=4, markeredgewidth=0.8)
    
    # 参考线和标签
    prediction_horizon_ms = int(pred_len * 1000 / sampling_rate_hz)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_ylim(0, 2.5)
    ax.set_ylabel('Relative RMSE')
    ax.set_title(f'{TARGET_MODEL} (Latency: {prediction_horizon_ms}ms)', fontsize=10)
    
    # 调整图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], ['PL', 'PS (Base)', 'PS (FT)'] + ['DS (Base)', 'DS (FT)'],
              title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整x轴标签
    ax.set_xticklabels(['Population\nLevel', 'Patient\nSpecific', 'Data\nSpecific'])
    
    # 保存
    output_path = os.path.join(output_dir, f'{TARGET_MODEL}_grouped_pl{pred_len}.tiff')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()
TARGET_MODEL = "LSTM"
sampling_rate_hz = 26
projects = ['population-level','FT-4DCT', 'FT-4DCT_fine_tuning','data-specific','data-specific_fine_tuning']
pred_lens = [ 6,12,18,24]
box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728',  '#9467bd'  ]  

work_space = '/mnt/nas-wang/nas-ssd/Results/RMP/'
output_dir = '/mnt/nas-wang/nas-ssd/Results/RMP/figures'
os.makedirs(output_dir, exist_ok=True)

for pred_len in pred_lens:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  
    

    data = []
    for project in projects:
        settings = f'{TARGET_MODEL}_pl{pred_len}'
        
        result_path = os.path.join(work_space, project , settings, 'test_results', 're_rmse_list.npy')
        #project = project.replace('_fine_tuning', '')

        # print('asdasdsad',result_path)
        rmse = np.load(result_path)

        data.append(pd.DataFrame({
            'Project': project,
            'RMSE': rmse
        }))
    df = pd.concat(data)
    
    # 绘制箱线图
    bp = ax.boxplot(
        [df[df['Project'] == p]['RMSE'] for p in projects],
        positions=range(len(projects)),  # x轴位置
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        medianprops=dict(color='black', linewidth=1.2)
    )

    # 设置颜色
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(box_colors[i])
        box.set_alpha(0.7)

    # 手动绘制离群点
    for i, project in enumerate(projects):
        q1 = df[df['Project'] == project]['RMSE'].quantile(0.25)
        q3 = df[df['Project'] == project]['RMSE'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df['Project'] == project) &
                    ((df['RMSE'] < q1 - 1.5 * iqr) | (df['RMSE'] > q3 + 1.5 * iqr))]
        outlier_y = np.where(outliers['RMSE'] > 2.5, 2.5, outliers['RMSE'])
        ax.plot([i] * len(outliers), outlier_y, 'r+',
                markersize=4, markeredgewidth=0.8)

    # =============================
    # 新增：局部横虚线（仅覆盖每组箱体范围）
    # =============================
    y_ticks = ax.get_yticks()  # y轴刻度
    box_width = 0.6            # 和上面 boxplot 的 widths 一致
    for y in y_ticks:
        for i in range(len(projects)):
            start_x = i - box_width / 2
            end_x = i + box_width / 2
            ax.hlines(y, start_x, end_x, colors='gray', linestyles=':', linewidth=0.4, alpha=0.5)

    # 参考线、标签等
    prediction_horizon_ms = int(pred_len * 1000 / sampling_rate_hz)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xticks(range(len(projects)))
    ax.set_xticklabels([project_abbr[p] for p in projects], rotation=15)
    ax.set_ylabel('Relative RMSE')
    ax.set_title(f'{TARGET_MODEL} (Latency: {prediction_horizon_ms}ms)', fontsize=10)



    output_path = os.path.join(output_dir,f'{TARGET_MODEL}_pl{pred_len}.tiff')

    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()