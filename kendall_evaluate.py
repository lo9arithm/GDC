from scipy.stats import kendalltau

from baseline import *
from Gaussian_Decay_Centrality import *


def calculate_kendall_all_centralities(G, sir_influence_filepath):
    """
    计算图 G 中所有中心性算法与 SIR 模型排序的肯德尔系数。

    参数:
    G (networkx.Graph): 输入图
    sir_influence_filepath (str): SIR 模型影响力数组的文件路径

    返回:
    dict: 包含每个中心性算法与 SIR 模型的肯德尔系数
    """
    # 加载 SIR 模型的影响力数组
    influence_array = np.load(sir_influence_filepath)

    # 计算所有中心性算法的排序
    centrality_functions = {
        "Degree Centrality": calculate_degree_centrality,
        "Closeness Centrality": calculate_closeness_centrality,
        "Betweenness Centrality": calculate_betweenness_centrality,
        "PageRank Centrality": calculate_pagerank_centrality,
        "Laplacian Gravity Centrality":calculate_Laplacian_Gravity_centrality,
        "improved_laplacian_centrality":calculate_improved_laplacian_centrality,
        "Radiation Centrality": calculate_radiation_centrality,
        "wdks Centrality": calculate_wdks_centrality,
        "Gaussian_Decay Centrality": calculate_Gaussian_Decay_Centrality,
    }

    # 存储每个中心性算法与 SIR 模型的肯德尔系数
    kendall_results = {}

    for name, func in centrality_functions.items():
        # 获取中心性算法的排序结果
        sorted_centrality = func(G)
        centrality_dict = dict(sorted_centrality)
        sorted_nodes = sorted(G.nodes(), key=lambda x: int(x))
        centrality_scores = [centrality_dict[node] for node in sorted_nodes]

        # 计算肯德尔系数
        tau, _ = kendalltau(influence_array, centrality_scores)
        kendall_results[name] = tau

    return kendall_results
