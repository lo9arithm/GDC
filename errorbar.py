import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import random

from SIR import sir_simulation
from baseline import *
from Gaussian_Decay_Centrality import *

# 设置参数
G = nx.read_graphml('./data/real_Graphs/05_Email.graphml')
G_name = '05_Email'
N, M = len(G.nodes()), len(G.edges())
k = sum([G.degree(i) for i in G.nodes()]) / N
k2 = sum([G.degree(i) ** 2 for i in G.nodes()]) / N
beta_c = k / (k2 - k)
beta = 1.0 * beta_c
num_simulations_1000 = 1000  # 1000次迭代的设置
num_simulations_1 = 1        # 单次迭代的设置
num_experiments = 100
node_list = sorted(G.nodes(), key=lambda x: int(x))

# 所有中心性方法
centrality_functions = {
    "DC": calculate_degree_centrality,
    "CC": calculate_closeness_centrality,
    "BC": calculate_betweenness_centrality,
    "PR": calculate_pagerank_centrality,
    "LGC": calculate_Laplacian_Gravity_centrality,
    "ILC": calculate_improved_laplacian_centrality,
    "RC": calculate_radiation_centrality,
    "WDKS": calculate_wdks_centrality,
    "GDC": calculate_Gaussian_Decay_Centrality,
}

# 计算中心性排序（一次即可）
centrality_rankings = {}
for name, func in centrality_functions.items():
    scores = func(G)
    scores_dict = dict(scores)
    ordered = [scores_dict[node] for node in node_list]
    centrality_rankings[name] = ordered

# 存储τ值 - 为两种SIR设置分别创建存储
tau_results_1000 = {name: [] for name in centrality_functions}  # 1000次迭代平均
tau_results_1 = {name: [] for name in centrality_functions}     # 单次迭代

# 重复 SIR 实验
for i in range(num_experiments):
    print(f"Running SIR experiment {i + 1}/{num_experiments}")

    # 运行1000次迭代平均的SIR模拟
    influence_1000 = sir_simulation(G, beta, num_simulations_1000)
    influence_array_1000 = np.array([influence_1000[node] for node in node_list])

    # 运行单次迭代的SIR模拟
    influence_1 = sir_simulation(G, beta, num_simulations_1)
    influence_array_1 = np.array([influence_1[node] for node in node_list])

    for name in centrality_functions:
        # 计算1000次迭代的τ
        tau_1000, _ = kendalltau(centrality_rankings[name], influence_array_1000)
        tau_results_1000[name].append(tau_1000)

        # 计算单次迭代的τ
        tau_1, _ = kendalltau(centrality_rankings[name], influence_array_1)
        tau_results_1[name].append(tau_1)

# Bootstrap置信区间函数
def bootstrap_confidence_interval(data, num_bootstrap=1000, confidence=0.95):
    means = []
    n = len(data)
    for _ in range(num_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower, upper

# 计算两种设置的置信区间
results_1000 = {}
for method, taus in tau_results_1000.items():
    mean_tau, lower_ci, upper_ci = bootstrap_confidence_interval(taus)
    results_1000[method] = {
        'mean': mean_tau,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'all_taus': taus
    }

results_1 = {}
for method, taus in tau_results_1.items():
    mean_tau, lower_ci, upper_ci = bootstrap_confidence_interval(taus)
    results_1[method] = {
        'mean': mean_tau,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'all_taus': taus
    }

# 打印结果
print("\nResults for 1000 iterations:")
for method, res in results_1000.items():
    print(f"{method}: Mean τ = {res['mean']:.4f}, 95% CI = [{res['lower_ci']:.4f}, {res['upper_ci']:.4f}]")

print("\nResults for 1 iteration:")
for method, res in results_1.items():
    print(f"{method}: Mean τ = {res['mean']:.4f}, 95% CI = [{res['lower_ci']:.4f}, {res['upper_ci']:.4f}]")

# 自定义颜色 - 为两种SIR设置定义不同颜色
method_colors_1000 = {
    'DC': '#1f77b4',   # 蓝色系
    'CC': '#2ca02c',   # 绿色系
    'BC': '#9467bd',   # 紫色系
    'PR': '#8c564b',   # 棕色系
    'LGC': '#bcbd22',  # 橄榄绿
    'ILC': '#e377c2',  # 粉色系
    'RC': '#17becf',   # 青色系
    'WDKS': '#7f7f7f', # 灰色系
    'GDC': '#d62728'   # 红色系
}

# 单次迭代使用更浅的颜色
method_colors_1 = {
    'DC': '#aec7e8',   # 浅蓝
    'CC': '#98df8a',   # 浅绿
    'BC': '#c5b0d5',   # 浅紫
    'PR': '#c49c94',   # 浅棕
    'LGC': '#dbdb8d',  # 浅橄榄
    'ILC': '#f7b6d2',  # 浅粉
    'RC': '#9edae5',   # 浅青
    'WDKS': '#c7c7c7', # 浅灰
    'GDC': '#ff9896'   # 浅红
}

# 绘图设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    #'font.size': 20,
    'axes.titlesize': 16,  # 主标题字号
    'axes.labelsize': 30,  # 坐标轴标签字号(X/Y轴名称)
    'xtick.labelsize': 35,  # X轴刻度数字字号
    'ytick.labelsize': 35,  # Y轴刻度数字字号
    'legend.fontsize': 20,  # 图例字号
    'mathtext.fontset': 'stix',
})
plt.figure(figsize=(16, 12))

# 准备绘图数据
methods = list(results_1000.keys())
x_pos = np.arange(len(methods))



# 为两种设置分别绘制散点图和置信区间
for i, method in enumerate(methods):
    # SIR1000 结果 - 在方法中心位置
    taus_1000 = results_1000[method]['all_taus']
    mean_1000 = results_1000[method]['mean']
    upper_1000 = results_1000[method]['upper_ci']
    lower_1000 = results_1000[method]['lower_ci']

    # 绘制1000次迭代的散点
    plt.scatter([x_pos[i]] * len(taus_1000), taus_1000,
                alpha=1.0, color=method_colors_1000[method], s=60,
                label='1000 iterations' if i == 0 else "")

    # 绘制1000次迭代的置信区间
    plt.errorbar(x_pos[i], mean_1000,
                 yerr=[[mean_1000 - lower_1000], [upper_1000 - mean_1000]],
                 fmt='o', capsize=10, capthick=2, elinewidth=2,
                 color=method_colors_1000[method], markersize=8,
                 )

    # 在置信区间上下限位置添加黑色短横线
    plt.hlines(upper_1000, x_pos[i] - 0.1, x_pos[i] + 0.1,
               colors='black', linewidth=1.5)
    plt.hlines(lower_1000, x_pos[i] - 0.1, x_pos[i] + 0.1,
               colors='black', linewidth=1.5)

    # 标注1000次迭代的置信区间上下限
    plt.text(x_pos[i] - 0.12, upper_1000 + 0.005,
             f'{upper_1000:.3f}',
             ha='center', va='bottom', fontsize=30,
             color='black')
    plt.text(x_pos[i] - 0.12, lower_1000 - 0.005,
             f'{lower_1000:.3f}',
             ha='center', va='top', fontsize=30,
             color='black')

    # SIR1 结果 - 也在方法中心位置
    taus_1 = results_1[method]['all_taus']
    mean_1 = results_1[method]['mean']
    upper_1 = results_1[method]['upper_ci']
    lower_1 = results_1[method]['lower_ci']

    # 绘制单次迭代的散点
    plt.scatter([x_pos[i]] * len(taus_1), taus_1,
                alpha=0.4, color=method_colors_1[method], s=60,
                marker='s',  # 使用不同标记
                label='1 iteration' if i == 0 else "")

    # 绘制单次迭代的置信区间
    plt.errorbar(x_pos[i], mean_1,
                 yerr=[[mean_1 - lower_1], [upper_1 - mean_1]],
                 fmt='s', capsize=10, capthick=2, elinewidth=2,
                 color=method_colors_1[method], markersize=8,
                 )

    # 在置信区间上下限位置添加黑色短横线
    plt.hlines(upper_1, x_pos[i] - 0.1, x_pos[i] + 0.1,
               colors='black', linewidth=1.5)
    plt.hlines(lower_1, x_pos[i] - 0.1, x_pos[i] + 0.1,
               colors='black', linewidth=1.5)

    # 标注单次迭代的置信区间上下限
    plt.text(x_pos[i] + 0.12, upper_1 + 0.005,
             f' {upper_1:.3f}',
             ha='center', va='bottom', fontsize=30,
             color='black')
    plt.text(x_pos[i] + 0.12, lower_1 - 0.005,
             f'{lower_1:.3f}',
             ha='center', va='top', fontsize=30,
             color='black')

# 添加参考线和标签
# plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.ylabel("Kendall's correlation coefficient τ")
plt.xticks(x_pos, methods, rotation=0, ha='center')
plt.xlim(-0.5, len(methods)-0.5)
plt.ylim(0.0, 1.0)
# 图例颜色
colors_1000 = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#bcbd22',
               '#e377c2', '#17becf', '#7f7f7f', '#d62728']
colors_1 = ['#aec7e8', '#98df8a', '#c5b0d5', '#c49c94', '#dbdb8d',
            '#f7b6d2', '#9edae5', '#c7c7c7', '#ff9896']

# 图例起始位置 (右上角偏移)
x0 = 0.65  # 左下角X（更靠左一点）
y0 = 1.1  # 左下角Y（更靠上）
plt.subplots_adjust(top=0.85)
dx = 0.02  # marker间距
dy = 0.04   # 上下行间距
marker_size = 100
text_fontsize = 20


# --- 先绘制图例方框（放在底层） ---
box_width = dx * 9 + 0.15  # 宽度 = 9个marker + 文字 + padding
box_height = dy * 2   # 高度 = 2行 + padding

legend_box = patches.FancyBboxPatch(
    (x0 - 0.005, y0 - dy - 0.015),  # 更靠左 & 更低
    box_width, box_height,
    boxstyle="round,pad=0.02",
    transform=plt.gca().transAxes,
    linewidth=1.2,
    edgecolor='gray',
    facecolor='white',
    alpha=0.75,
    zorder=5,
    clip_on=False
)
plt.gca().add_patch(legend_box)

# --- 绘制 1000 iterations 圆圈图例 ---
for i, color in enumerate(colors_1000):
    plt.scatter(x0 + i * dx, y0, s=marker_size, color=color,
                edgecolors='face', transform=plt.gca().transAxes, zorder=101,clip_on=False)

plt.text(x0 + 9 * dx + 0.02, y0, '1000 iterations',
         transform=plt.gca().transAxes, fontsize=text_fontsize,
         va='center', ha='left', zorder=10)

# --- 绘制 1 iteration 方形图例 ---
for i, color in enumerate(colors_1):
    plt.scatter(x0 + i * dx, y0 - dy, s=marker_size, color=color, marker='s',
                alpha=0.4, edgecolors='face', transform=plt.gca().transAxes, zorder=102,clip_on=False)

plt.text(x0 + 9 * dx + 0.02, y0 - dy, '1 iteration',
         transform=plt.gca().transAxes, fontsize=text_fontsize,
         va='center', ha='left', zorder=10)

plt.savefig(f"./data/results/{G_name}_comparison_kendall_tau_with_CI.png", dpi=300, bbox_inches='tight',pad_inches=0.5)
plt.show()