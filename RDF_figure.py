
from Gaussian_Decay_Centrality import *
from baseline import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import networkx as nx
import numpy as np
from collections import defaultdict
from matplotlib.patches import Rectangle
# 需要先确保所有中心性算法函数已在当前环境定义（包括GDC、LGC等）

def dense_ranking(scores):
    """处理并列排名生成密集排名"""
    sorted_scores = sorted(scores, reverse=True)
    rank_dict = {}
    current_rank = 1
    for i, score in enumerate(sorted_scores):
        if i > 0 and score != sorted_scores[i-1]:
            current_rank = i + 1
        rank_dict[score] = current_rank
    return [rank_dict[score] for score in scores]

def calculate_rdf(ranks, max_rank=None):
    """计算排名分布函数"""
    max_observed = max(ranks)
    max_rank = max_rank if max_rank else max_observed
    counts = defaultdict(int)
    for r in ranks:
        counts[r] += 1
    x = np.arange(1, max_rank+1)
    y = np.zeros(max_rank)
    total = len(ranks)
    for r, cnt in counts.items():
        if r <= max_rank:
            y[r-1] = cnt / total
    return x, y

def plot_rdf_comparison(graph_path):
    # 加载网络
    G = nx.read_graphml(graph_path)
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    # 定义所有算法和显示名称
    algorithms = [
        (calculate_degree_centrality, 'DC'),
        (calculate_closeness_centrality, 'CC'),
        (calculate_betweenness_centrality, 'BC'),
        (calculate_pagerank_centrality, 'PR'),
        (calculate_Laplacian_Gravity_centrality, 'LGC'),
        (calculate_improved_laplacian_centrality,'ILC'),
        (calculate_radiation_centrality, 'RC'),
        (calculate_wdks_centrality,'WDKS'),
        (calculate_Gaussian_Decay_Centrality, 'GDC'),
    ]

    # 配置全局样式
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.titlesize': 16,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'mathtext.fontset': 'stix',
    })

    # 创建画布
    fig, ax = plt.subplots(figsize=(18, 8))

    # 颜色配置
    non_red_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b',
                      '#bcbd22', '#e377c2', '#17becf', '#7f7f7f']

    color_map = {'GDC': '#d62728'}
    for i, (_, name) in enumerate([a for a in algorithms if a[1] != 'GDC']):
        color_map[name] = non_red_colors[i % len(non_red_colors)]

    # 数据收集容器
    all_data = []
    max_rank = len(G.nodes())
    zoom_range = max(5, int(max_rank * 0.1))  # 最小显示5个排名

    # 主图绘制
    for algo, name in algorithms:
        try:
            # 处理PageRank的特殊情况
            if name == 'PR':
                G_temp = G.to_directed()
                sorted_nodes = algo(G_temp)
            else:
                sorted_nodes = algo(G)

            scores = [score for _, score in sorted_nodes]
            ranks = dense_ranking(scores)
            x, y = calculate_rdf(ranks, max_rank)

            # 存储数据
            all_data.append((x, y, name))

            # 绘制主图
            ax.plot(x, y,
                    label=name,
                    color=color_map[name],
                    linewidth=2 if name == 'GDC' else 1.5,
                    linestyle='-',
                    alpha=0.9 if name == 'GDC' else 0.7)
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            continue

    # 主图设置
    ax.set_xlabel('Rank Value', fontsize=25)
    ax.set_ylabel('Probability', fontsize=25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # 创建嵌入放大图
    axins = ax.inset_axes([0.48, 0.48, 0.5, 0.5])  # 调整位置参数

    # 绘制放大图
    for x, y, name in all_data:
        mask = x <= zoom_range
        axins.plot(x[mask], y[mask],
                   color=color_map[name],
                   linewidth=2.2 if name == 'GDC' else 1.8,
                   linestyle='-',
                   alpha=0.95 if name == 'GDC' else 0.8)

    # 自动设置放大图Y轴范围
    y_max = max([max(y[x <= zoom_range]) for x, y, _ in all_data]) * 1.1
    axins.set_ylim(0, y_max)
    axins.set_xlim(1, zoom_range)
    axins.grid(True, alpha=0.3)
    axins.set_facecolor('#f8f8f8')

    # 添加连接标记
    mark_inset(ax, axins, loc1=3, loc2=1,
               fc="none", ec='#606060', lw=1.2,
               linestyle='dotted')

    # 标记放大区域
    ax.add_patch(Rectangle((1, 0), zoom_range-1, ax.get_ylim()[1],
                           edgecolor='#606060',
                           facecolor='none',
                           linestyle='dotted',
                           linewidth=1.5))

    # 保存和显示
    plt.tight_layout()

    plt.savefig(f'./data/figures/RDF_{graph_path.split("/")[-1].split(".")[0]}.png', dpi=300)
    plt.show()

# 使用示例（替换为实际路径）
plot_rdf_comparison('./data/real_Graphs/10_WV.graphml')