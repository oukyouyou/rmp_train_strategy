import matplotlib.pyplot as plt
from collections import defaultdict 
import matplotlib.patches as patches
import seaborn as sns
import math
from pathlib import Path
from resp_db.client import RpmDatabaseClient
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

def plot_patient_modality_distribution(client: RpmDatabaseClient):
    modality_dist, count_no_cbct_linac = client.get_modality_distribution()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    modality_order = ["4DCT", "CBCT", "LINAC"]

    for i, modality in enumerate(modality_order):
        counts = modality_dist[modality]

        if modality == "LINAC":
            # 重建 bins: 1~10, >10，跳过为0的柱子
            new_counts = {}
            for k, v in counts.items():
                try:
                    n = int(k)
                    if n == 0:
                        new_counts["0"] = v  # 保留0值
                    elif 1 <= n <= 10:
                        new_counts[str(n)] = v
                    else:
                        new_counts[">10"] = new_counts.get(">10", 0) + v
                except ValueError:
                    if ">" in k:
                        new_counts[">10"] = new_counts.get(">10", 0) + v

            # 排序 key:数字升序，最后 ">10"
            keys = sorted([k for k in new_counts if k != ">10"], key=lambda x: int(x))
            if ">10" in new_counts:
                keys.append(">10")
            values = [new_counts[k] for k in keys]
        else:
            # 其他 modality:也跳过 value 为 0 的，并按数字排序
            keys = [k for k in counts if counts[k] > 0]
            try:
                keys = sorted(keys, key=lambda x: int(x))
            except ValueError:
                pass  # 如果是非数字就不排序
            values = [counts[k] for k in keys]

        sns.barplot(x=keys, y=values, ax=axes[i], order=keys)  # 🔒 明确顺序
        axes[i].set_title(f"{modality} distribution per patient")
        axes[i].set_xlabel("Number of fractions")
        axes[i].set_ylabel("Number of patients")

        # 正确位置标注每个柱子上的值
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)

    fig.suptitle(f"Modality Distribution (patients with no CBCT/LINAC: {count_no_cbct_linac}, patient_ID with no 4DCT: 25)", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("/mnt/nas-wang/nas-ssd/Scripts/RMP/results/plot_patient_modality_distribution.png", dpi=300)


class AdvancedModalityPlotter:
    def __init__(self, client):
        self.client = client
        self.cmap = plt.cm.viridis
        self.cell_params = {
            'cols': 50,
            'box_w': 0.25,
            'ct_h': 0.1,
            'col_gap': 0.1,
            'row_gap': 0.2,
            'small_size': 0.2,
        }
        # 获取并处理数据
        self.patient_data = self.client.get_patient_modality_stats()

    def plot(self):
        # 分离有/无 LINAC 的患者
        with_linac = [p for p in self.patient_data if p.get("linac", 0) > 0]  # 使用 .get 防止没有 "linac" 键的情况
        without_linac = [p for p in self.patient_data if p.get("linac", 0) == 0]
        print(f"Patients without LINAC: {len(without_linac)}")
        print(f"Patients with LINAC: {len(with_linac)}")
        
        # 排序逻辑
        with_linac.sort(key=lambda x: (x["ct"] + x["cbct"] + x["linac"], -x["ct"]))
        without_linac.sort(key=lambda x: x["id"])

        # 从配置中获取参数
        params = self.cell_params
        cols = params['cols']
        box_w = params['box_w']
        ct_h = params['ct_h']
        col_gap = params['col_gap']
        row_gap = params['row_gap']
        small_size = params['small_size']
        row_height_small = small_size + row_gap  # 动态计算小方块行高

        # 计算 with_linac 区域布局
        rows_lin = math.ceil(len(with_linac) / cols)
        row_maxs, row_heights = [], []
        for r in range(rows_lin):
            chunk = with_linac[r*cols : (r+1)*cols]
            max_total = max((p["ct"] + max(p["cbct"] , p["linac"]) for p in chunk)) if chunk else 0
            row_maxs.append(max_total)
            row_heights.append((max_total + 1) * ct_h + row_gap)

        # 计算 without_linac 区域高度
        n_without = len(without_linac)
        rows_small = math.ceil(n_without / cols)
        ext_height = rows_small * row_height_small 

        # 创建画布
        fig_h = sum(row_heights) + ext_height
        fig_w = cols * (box_w + col_gap) 
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # 颜色映射函数
        def get_color(idx, modality):
            cmap = {
                "ct": plt.cm.Blues,
                "cbct": plt.cm.YlOrBr,
                "linac": plt.cm.Greens
            }
            return "#cccccc" if idx <= 0 else cmap[modality](0.3 + 0.7 * idx/10)

        # 定义 with_linac 方块绘制函数
        def draw_linac_block(p, x, y_top, row_max):
            dashed_h = (row_max + 1) * ct_h
            y_bottom = y_top - dashed_h
            ax.add_patch(patches.Rectangle(
                (x, y_bottom), box_w, dashed_h,
                linewidth=0.5, edgecolor='black', linestyle='--', facecolor='none'
            ))
            ax.text(x + box_w/2, y_top, p["id"][3:],  # 显示去除了PID的数字
                   ha='center', va='bottom', fontsize=6)
            
            # 绘制CT部分
            for i in range(p["ct"]):
                cy = y_top - ct_h*(i+1)
                color = get_color(i+1, "ct")
                ax.add_patch(patches.Rectangle(
                    (x, cy), box_w, ct_h,
                    facecolor=color, edgecolor='black', linewidth=0.5
                ))
                ax.text(x + box_w/2, cy + ct_h/2, str(i+1),
                       ha='center', va='center', fontsize=5)
            
            # 绘制CBCT（左半）和LINAC（右半）
            offset = p["ct"]
            for i in range(p["cbct"]):
                cy = y_top - ct_h*(offset + i + 1)
                color = get_color(i+1, "cbct")
                ax.add_patch(patches.Rectangle(
                    (x, cy), box_w/2, ct_h,
                    facecolor=color, edgecolor='black', linewidth=0.5
                ))
                ax.text(x + box_w/4, cy + ct_h/2, str(i+1),
                       ha='center', va='center', fontsize=5)
                
            for i in range(p["linac"]):
                ly = y_top - ct_h*(offset + i + 1)
                color = get_color(i+1, "linac")
                ax.add_patch(patches.Rectangle(
                    (x + box_w/2, ly), box_w/2, ct_h,
                    facecolor=color, edgecolor='black', linewidth=0.5
                ))
                ax.text(x + 3*box_w/4, ly + ct_h/2, str(i+1),
                       ha='center', va='center', fontsize=5)

        # 绘制 with_linac 主区域
        cum_y = fig_h  # 从画布顶部开始绘制
        ax.text(0.5, cum_y + 2* box_w,    # 固定X坐标在左侧区域
                f"With LINAC signals \n",
                ha='left', va='top', fontsize=9,fontweight='bold')
        for r in range(rows_lin):
            y_top = cum_y
            
            for c, p in enumerate(with_linac[r*cols : (r+1)*cols]):
                x = c * (box_w + col_gap)

                ax.text(-1.3, y_top + ct_h,    # 固定X坐标在左侧区域
                f"Patient ID: \n"
                f"4DCT num: \n"
                f"CBCT/LINAC num: \n",
                ha='left', va='top', fontsize=7)
                
                draw_linac_block(p, x, y_top, row_maxs[r])
            cum_y -= row_heights[r]  # 移动到下一行

        # 绘制 without_linac 区域（左对齐）
        offset_y = 0.15
        if n_without > 0:
            ax.text(0.5, ext_height + small_size - offset_y + 0.03,    # 固定X坐标在左侧区域
                f"Without LINAC signals \n",
                ha='left', va='top', fontsize=9,fontweight='bold')
            ax.text(0, ext_height - small_size - offset_y, f"Patient ID: \n",ha='left', va='center', fontsize=7,fontweight='bold')
            for idx, p in enumerate(without_linac):
                row = idx // cols
                col = idx % cols
                y_pos = ext_height - row * row_height_small - offset_y
                x_pos =  col * (small_size + col_gap)
                
                ax.add_patch(patches.Rectangle(
                    (x_pos, y_pos - small_size), small_size, small_size,
                    facecolor='lightgrey', edgecolor='black', linewidth=0.5
                ))
                ax.text(x_pos + small_size/2, y_pos - small_size/2,
                       p["id"][3:],  # 显示去除了PID的数字
                       ha='center', va='center', fontsize=5)

        # 添加图例和调整布局
        legend_elements = [
            patches.Patch(facecolor=get_color(7, 'ct'), edgecolor='black', label='4DCT'),
            patches.Patch(facecolor=get_color(7, 'cbct'), edgecolor='black', label='CBCT'),
            patches.Patch(facecolor=get_color(7, 'linac'), edgecolor='black', label='LINAC'),
            patches.Patch(facecolor='#cccccc', edgecolor='black', label='No LINAC'),
        ]
        plt.subplots_adjust(top=0.9)  # 保留顶部空间
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=4,
            frameon=False,
            fontsize=8
        )

        # 设置坐标轴并显示
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.axis('off')
        plt.show()
        plt.savefig("/mnt/nas-wang/nas-ssd/Scripts/RMP/results/Whole_dataset.png", dpi=300)



class CustomedModalityPlotter:
    def __init__(self, client):
        self.client = client
        self.cmap = plt.cm.viridis
        self.cell_params = {
            'cols': 50,
            'box_w': 0.25,
            'ct_h': 0.1,
            'col_gap': 0.1,
            'row_gap': 0.2,
            'small_size': 0.2,
        }
        # 获取并处理数据
        self.patient_data = self.client.get_patient_modality_stats()

    def plot(self):
        # 指定要展示的 with LINAC 患者 ID（保留前导 0 的形式）
        #target_with_ids = {"114", "269", "328", "395", "324", "024", "335", "120", "341", "103", "168", "363", "193", "092", "062"}
        target_with_ids = ["114", "269", "328", "395","120", "341", "062", "024", "335",  "103", "168", "363", "193", "092"]

        # 分离有/无 LINAC 的患者，并筛选
        #with_linac = [p for p in self.patient_data if p.get("linac", 0) > 0 and p["id"][3:] in target_with_ids]
        with_linac = [p for pid in target_with_ids for p in self.patient_data if p.get("linac", 0) > 0 and p["id"][3:] == pid]

        without_linac = [p for p in self.patient_data if p.get("linac", 0) == 0][:35]  # 截取前36个

        # 确保 with_linac 的顺序与 target_with_ids 一致
        #with_linac.sort(key=lambda x: list(target_with_ids).index(x["id"][3:]) if x["id"][3:] in target_with_ids else 999)
        #with_linac.sort(key=lambda x: target_with_ids.index(x["id"][3:]) if x["id"][3:] in target_with_ids else 999)

        params = self.cell_params
        cols = 7  # 每行固定6个
        box_w = params['box_w']
        ct_h = params['ct_h']
        col_gap = params['col_gap']
        row_gap = params['row_gap']
        small_size = params['small_size']
        row_height_small = small_size 

        # 计算布局
        rows_lin = math.ceil(len(with_linac) / cols)
        row_maxs, row_heights = [], []
        for r in range(rows_lin):
            chunk = with_linac[r*cols : (r+1)*cols]
            max_total = max((p["ct"] + max(p["cbct"] , p["linac"]) for p in chunk)) if chunk else 0
            row_maxs.append(max_total)
            row_heights.append((max_total + 1) * ct_h + row_gap)

        rows_small = 6
        ext_height = rows_small * row_height_small

        fig_h = sum(row_heights) + ext_height
        fig_w = cols * (box_w + col_gap) + 1
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        def get_color(idx, modality):
            cmap = {
                "ct": plt.cm.Blues,
                "cbct": plt.cm.YlOrBr,
                "linac": plt.cm.Greens
            }
            return "#cccccc" if idx <= 0 else cmap[modality](0.3 + 0.7 * idx/10)

        def draw_linac_block(p, x, y_top, row_max):
            dashed_h = (row_max + 1) * ct_h
            y_bottom = y_top - dashed_h
            ax.add_patch(patches.Rectangle(
                (x, y_bottom), box_w, dashed_h,
                linewidth=0.5, edgecolor='black', linestyle='--', facecolor='none'
            ))
            ax.text(x + box_w/2, y_top, p["id"][3:], ha='center', va='bottom', fontsize=6)
            for i in range(p["ct"]):
                cy = y_top - ct_h*(i+1)
                color = get_color(i+1, "ct")
                ax.add_patch(patches.Rectangle((x, cy), box_w, ct_h, facecolor=color, edgecolor='black', linewidth=0.5))
                ax.text(x + box_w/2, cy + ct_h/2, str(i+1), ha='center', va='center', fontsize=5)
            offset = p["ct"]
            for i in range(p["cbct"]):
                cy = y_top - ct_h*(offset + i + 1)
                color = get_color(i+1, "cbct")
                ax.add_patch(patches.Rectangle((x, cy), box_w/2, ct_h, facecolor=color, edgecolor='black', linewidth=0.5))
                ax.text(x + box_w/4, cy + ct_h/2, str(i+1), ha='center', va='center', fontsize=5)
            for i in range(p["linac"]):
                ly = y_top - ct_h*(offset + i + 1)
                color = get_color(i+1, "linac")
                ax.add_patch(patches.Rectangle((x + box_w/2, ly), box_w/2, ct_h, facecolor=color, edgecolor='black', linewidth=0.5))
                ax.text(x + 3*box_w/4, ly + ct_h/2, str(i+1), ha='center', va='center', fontsize=5)

        cum_y = fig_h
        ax.text(-1.3, cum_y , f"With LINAC\n" ,ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(-1.1, cum_y - 0.3,  f"51/322\n",ha='left', va='top', fontsize=7)
        ax.text(-1.25, cum_y - 0.55, f" with CBCT\n",ha='left', va='top', fontsize=7)
        for r in range(rows_lin):
            y_top = cum_y
            for c, p in enumerate(with_linac[r*cols : (r+1)*cols]):
                x = c * (box_w + col_gap)
                draw_linac_block(p, x, y_top, row_maxs[r])
            cum_y -= row_heights[r]

        offset_y = 0.15
        if len(without_linac) > 0:
            ax.text(-1.3, ext_height + small_size - offset_y + 0.03, "Without LINAC\n", ha='left', va='top', fontsize=9, fontweight='bold')
            ax.text(-1.1, ext_height + small_size - offset_y + 0.03 - 0.3,  f"1 /94\n",ha='left', va='top', fontsize=7)
            ax.text(-1.25, ext_height + small_size - offset_y + 0.03 - 0.55, f" with CBCT\n",ha='left', va='top', fontsize=7)
            for idx, p in enumerate(without_linac):
                row = idx // cols
                col = idx % cols
                y_pos = ext_height - row * row_height_small - offset_y
                x_pos = col * (small_size + col_gap)
                ax.add_patch(patches.Rectangle((x_pos, y_pos - small_size), small_size, small_size, facecolor='lightgrey', edgecolor='black', linewidth=0.5))
                ax.text(x_pos + small_size/2, y_pos - small_size/2, p["id"][3:], ha='center', va='center', fontsize=5)

        ax.text(fig_w - 0.9, fig_h - 0.3,
        "B. Patient-specific:\n",
        ha='left', va='top', fontsize=9, fontweight='bold')

        ax.text(fig_w - 0.9, fig_h - 0.5,
        "Train data for PS-4DCT (B1):",ha='left', va='top', fontsize=9, fontweight='bold')
        
        ax.add_patch(patches.Rectangle(
            (fig_w- 0.3, fig_h - 0.9), 0.25, 0.15,
            facecolor="yellow",
            edgecolor='black',
            linewidth=0.5
        ))
        
        ax.text(fig_w - 0.9, fig_h - 1,"Train data for PS-4DCT-Fxn (B2):",ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(fig_w - 0.9, fig_h - 1.2,"   - PS-4DCT-Fx1:",ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(fig_w - 0.9, fig_h - 1.4,"   - PS-4DCT-Fx2:",ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(fig_w - 0.9, fig_h - 1.6,"   - PS-4DCT-Fx3:",ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(fig_w - 0.9, fig_h - 1.8,"   - PS-4DCT-Fx4:",ha='left', va='top', fontsize=9, fontweight='bold')

        ax.text(fig_w - 0.9, fig_h - 4,
        "Test data for B1 and B2:",
        ha='left', va='top', fontsize=9, fontweight='bold')
        legend_elements = [
            patches.Patch(facecolor=get_color(7, 'ct'), edgecolor='black', label='4DCT'),
            patches.Patch(facecolor=get_color(7, 'cbct'), edgecolor='black', label='CBCT'),
            patches.Patch(facecolor=get_color(7, 'linac'), edgecolor='black', label='LINAC'),
            patches.Patch(facecolor='#cccccc', edgecolor='black', label='No LINAC'),
        ]
        plt.subplots_adjust(top=0.9)
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=4,
            frameon=False,
            fontsize=8
        )
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.axis('off')
        plt.show()
        plt.savefig("/mnt/nas-wang/nas-ssd/Scripts/RMP/results/Customed_dataset.png", dpi=300)


    

class CustomedModalityPlotter:
        def __init__(self, client):
            self.client = client
            self.cmap = plt.cm.viridis
            self.cell_params = {
                'cols': 50,
                'box_w': 0.25,
                'ct_h': 0.1,
                'col_gap': 0.1,
                'row_gap': 0.2,
                'small_size': 0.2,
            }
            # 获取并处理数据
            self.patient_data = self.client.get_patient_modality_stats()
            self.interactive_squares = []  # 存储所有添加的方块对象
            self.square_size = 0.16        
            self.square_size_y = 0.12   # 默认方块大小
            self.current_color = 'red'      # 默认颜色
            self.fig = None  # 预先声明
            self.ax = None
            self.drag_start = None  # 记录拖动起始坐标
            self.current_rect = None  # 记录正在绘制的矩形
            
        def plot(self):
            # 指定要展示的 with LINAC 患者 ID（保留前导 0 的形式）
            target_with_ids = ["114", "269", "328", "395","120", "341", "062", "024", "335",  "103", "168", "363", "193", "092"]

            with_linac = [p for pid in target_with_ids for p in self.patient_data if p.get("linac", 0) > 0 and p["id"][3:] == pid]
            without_linac = [p for p in self.patient_data if p.get("linac", 0) == 0][:35]  # 截取前36个

            params = self.cell_params
            cols = 7  # 每行固定6个
            box_w = params['box_w']
            ct_h = params['ct_h']
            col_gap = params['col_gap']
            row_gap = params['row_gap']
            small_size = params['small_size']
            row_height_small = small_size 

            # 计算布局
            rows_lin = math.ceil(len(with_linac) / cols)
            row_maxs, row_heights = [], []
            for r in range(rows_lin):
                chunk = with_linac[r*cols : (r+1)*cols]
                max_total = max((p["ct"] + max(p["cbct"] , p["linac"]) for p in chunk)) if chunk else 0
                row_maxs.append(max_total)
                row_heights.append((max_total + 1) * ct_h + row_gap)

            rows_small = 6
            ext_height = rows_small * row_height_small

            fig_h = sum(row_heights) + ext_height
            fig_w = cols * (box_w + col_gap) + 1
            
            # 创建图形并保存到实例变量
            self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))

            def get_color(idx, modality):
                cmap = {
                    "ct": plt.cm.Blues,
                    "cbct": plt.cm.YlOrBr,
                    "linac": plt.cm.Greens
                }
                return "#cccccc" if idx <= 0 else cmap[modality](0.3 + 0.7 * idx/10)

            def draw_linac_block(p, x, y_top, row_max):
                dashed_h = (row_max + 1) * ct_h
                y_bottom = y_top - dashed_h
                self.ax.add_patch(patches.Rectangle(
                    (x, y_bottom), box_w, dashed_h,
                    linewidth=0.5, edgecolor='black', linestyle='--', facecolor='none'
                ))
                self.ax.text(x + box_w/2, y_top, p["id"][3:], ha='center', va='bottom', fontsize=6)
                for i in range(p["ct"]):
                    cy = y_top - ct_h*(i+1)
                    color = get_color(i+1, "ct")
                    self.ax.add_patch(patches.Rectangle((x, cy), box_w, ct_h, facecolor=color, edgecolor='black', linewidth=0.5))
                    self.ax.text(x + box_w/2, cy + ct_h/2, str(i+1), ha='center', va='center', fontsize=5)
                offset = p["ct"]
                for i in range(p["cbct"]):
                    cy = y_top - ct_h*(offset + i + 1)
                    color = get_color(i+1, "cbct")
                    self.ax.add_patch(patches.Rectangle((x, cy), box_w/2, ct_h, facecolor=color, edgecolor='black', linewidth=0.5))
                    self.ax.text(x + box_w/4, cy + ct_h/2, str(i+1), ha='center', va='center', fontsize=5)
                for i in range(p["linac"]):
                    ly = y_top - ct_h*(offset + i + 1)
                    color = get_color(i+1, "linac")
                    self.ax.add_patch(patches.Rectangle((x + box_w/2, ly), box_w/2, ct_h, facecolor=color, edgecolor='black', linewidth=0.5))
                    self.ax.text(x + 3*box_w/4, ly + ct_h/2, str(i+1), ha='center', va='center', fontsize=5)

            cum_y = fig_h
            self.ax.text(-1.3, cum_y , f"With LINAC\n" ,ha='left', va='top', fontsize=9, fontweight='bold')
            self.ax.text(-1.1, cum_y - 0.3,  f"51/322\n",ha='left', va='top', fontsize=7)
            self.ax.text(-1.25, cum_y - 0.55, f" with CBCT\n",ha='left', va='top', fontsize=7)
            for r in range(rows_lin):
                y_top = cum_y
                for c, p in enumerate(with_linac[r*cols : (r+1)*cols]):
                    x = c * (box_w + col_gap)
                    draw_linac_block(p, x, y_top, row_maxs[r])
                cum_y -= row_heights[r]

            offset_y = 0.15
            if len(without_linac) > 0:
                self.ax.text(-1.3, ext_height + small_size - offset_y + 0.03, "Without LINAC\n", ha='left', va='top', fontsize=9, fontweight='bold')
                self.ax.text(-1.1, ext_height + small_size - offset_y + 0.03 - 0.3,  f"1 /94\n",ha='left', va='top', fontsize=7)
                self.ax.text(-1.25, ext_height + small_size - offset_y + 0.03 - 0.55, f" with CBCT\n",ha='left', va='top', fontsize=7)
                for idx, p in enumerate(without_linac):
                    row = idx // cols
                    col = idx % cols
                    y_pos = ext_height - row * row_height_small - offset_y
                    x_pos = col * (small_size + col_gap)
                    self.ax.add_patch(patches.Rectangle((x_pos, y_pos - small_size), small_size, small_size, facecolor='lightgrey', edgecolor='black', linewidth=0.5))
                    self.ax.text(x_pos + small_size/2, y_pos - small_size/2, p["id"][3:], ha='center', va='center', fontsize=5)

            self.ax.text(fig_w - 0.9, fig_h - 0.3,
            "C. Data-specific:\n",
            ha='left', va='top', fontsize=9, fontweight='bold')

            self.ax.text(fig_w - 0.9, fig_h - 0.5,
            "Train data for TS model (C1):",ha='left', va='top', fontsize=9, fontweight='bold')
            
            #self.ax.add_patch(patches.Rectangle((fig_w- 0.3, fig_h - 0.9), 0.25, 0.15,facecolor="yellow",edgecolor='black',
            #    linewidth=0.5))
            
            #self.ax.text(fig_w - 0.9, fig_h - 1,"Train data for Pop-Ft-4DCT (A2):",ha='left', va='top', fontsize=9, fontweight='bold')
            #self.ax.text(fig_w - 0.9, fig_h - 1.2,"   - PS-4DCT-Fx1:",ha='left', va='top', fontsize=9, fontweight='bold')
            #self.ax.text(fig_w - 0.9, fig_h - 1.4,"   - PS-4DCT-Fx2:",ha='left', va='top', fontsize=9, fontweight='bold')
            #self.ax.text(fig_w - 0.9, fig_h - 1.6,"Train data for Pop-Eft (A3)",ha='left', va='top', fontsize=9, fontweight='bold')
            #self.ax.text(fig_w - 0.9, fig_h - 1.8,"   - PS-4DCT-Fx4:",ha='left', va='top', fontsize=9, fontweight='bold')

            self.ax.text(fig_w - 0.9, fig_h - 4,
            "Test data for C1:",
            ha='left', va='top', fontsize=9, fontweight='bold')
            legend_elements = [
                patches.Patch(facecolor=get_color(7, 'ct'), edgecolor='black', label='4DCT'),
                patches.Patch(facecolor=get_color(7, 'cbct'), edgecolor='black', label='CBCT'),
                patches.Patch(facecolor=get_color(7, 'linac'), edgecolor='black', label='LINAC'),
                patches.Patch(facecolor='#cccccc', edgecolor='black', label='No LINAC'),
            ]
            plt.subplots_adjust(top=0.9)
            self.fig.legend(
                handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.97),
                ncol=4,
                frameon=False,
                fontsize=8
            )
            
            self.ax.set_xlim(0, fig_w)
            self.ax.set_ylim(0, fig_h)
            self.ax.axis('off')
            
            # 绑定交互事件
            self.fig.canvas.mpl_connect('button_press_event', self.on_press)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            # 保存并显示图形
            plt.savefig("/mnt/nas-wang/nas-ssd/Scripts/RMP/results/Customed_dataset.png", dpi=300)
            plt.show()
            self.fig.canvas.draw()  # 确保在最后调用
            return self.fig

        def on_press(self, event):
            """鼠标按下时记录起点"""
            if event.inaxes != self.ax or event.button != 1:  # 仅左键
                return
            self.drag_start = (event.xdata, event.ydata)

        def on_motion(self, event):
            """鼠标移动时实时更新预览框"""
            if not self.drag_start or event.inaxes != self.ax:
                return
            
            # 清除旧预览框
            if self.current_rect:
                self.current_rect.remove()
            
            # 计算矩形参数
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            width, height = abs(x1 - x0), abs(y1 - y0)
            x, y = min(x0, x1), min(y0, y1)
            
            # 创建新预览框
            self.current_rect = patches.Rectangle(
                (x, y), width, height,
                facecolor=self.current_color,
                edgecolor='black',
                alpha=0.3,  # 半透明预览
                linestyle='--'
            )
            self.ax.add_patch(self.current_rect)
            self.fig.canvas.draw_idle()

        def on_release(self, event):
            """鼠标释放时固定矩形"""
            if not self.drag_start or event.button != 1:
                return
            
            # 移除预览框
            if self.current_rect:
                self.current_rect.remove()
                self.current_rect = None
            
            # 创建最终矩形
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            width, height = abs(x1 - x0), abs(y1 - y0)
            x, y = min(x0, x1), min(y0, y1)
            
            rect = patches.Rectangle(
                (x, y), width, height,
                facecolor=self.current_color,
                edgecolor='black',
                alpha=0.7
            )
            self.ax.add_patch(rect)
            self.interactive_squares.append(rect)
            self.fig.canvas.draw()
            
            self.drag_start = None  # 重置状态

        def on_key(self, event):
            """键盘按键颜色控制"""
            # color_map = {
            #     '1': '#1E90FF',  # 蓝色
            #     '2': '#D8BFD8',  # 浅紫1
            #     '3': '#DDA0DD',  # 浅紫2（稍浓）
            #     '4': '#EE82EE',  # 浅紫3（更浓）
            #     '5': '#9370DB',  # 紫色
            #     'r': 'red'       # 保持红色
            # }
            color_map = {
                '1': '#1E90FF',  # 蓝色
                '2': 'yellow',  # 浅紫1
                '3': 'green',  # 浅紫2（稍浓）
                '4': '#EE82EE',  # 浅紫3（更浓）
                '5': '#9370DB',  # 紫色
                'r': 'red'       # 保持红色
            }
            if event.key in color_map:
                self.set_color(color_map[event.key])
                print(f"颜色已切换: {event.key} -> {color_map[event.key]}")
            else:
                print(f"未定义的按键: {event.key}")
            # 新增：Ctrl+Z 撤销最后一个矩形
            if event.key == 'ctrl+z' and self.interactive_squares:
                last_rect = self.interactive_squares.pop()  # 移除最后一个矩形
                last_rect.remove()                         # 从图形中删除
                self.fig.canvas.draw()                     # 刷新画布
                print("撤销了最后一个矩形")
        def set_color(self, color):
            """设置当前颜色并给出视觉反馈"""
            self.current_color = color
            
if __name__ == "__main__":
    db_root = Path("/mnt/nas-wang/nas-ssd/Scripts/respiratory-signal-database/open_access_rpm_signals_master.db")
    client = RpmDatabaseClient(db_filepath=db_root)
    #plot_patient_modality_distribution(client)

    #plotter = AdvancedModalityPlotter(client)
    #plotter.plot()
    plotter = CustomedModalityPlotter(client)
    plotter.plot()
