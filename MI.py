import networkx as nx
import numpy as np
from collections import defaultdict
from Gaussian_Decay_Centrality import *
from baseline import *


def dense_ranking(scores):
    """将得分转换为密集排名（并列排名相同，后续不跳号）"""
    sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    rank = 1
    ranks = {}
    prev_score = None

    for i, (node, score) in enumerate(sorted_scores):
        if score != prev_score:
            rank = i + 1
            prev_score = score
        ranks[node] = rank
    return ranks

def calculate_MI(ranking_dict):
    """计算单调性指数MI"""
    rank_counts = defaultdict(int)
    for rank in ranking_dict.values():
        rank_counts[rank] += 1

    N = len(ranking_dict)
    sum_term = sum(cnt*(cnt-1) for cnt in rank_counts.values())
    if N == 0 or N == 1:
        return 0.0
    MI = (1 - sum_term / (N*(N-1))) ** 2
    return MI

def run_all_centralities(G):
    """运行所有中心性算法并返回结果"""
    algorithms = {
        "Degree": calculate_degree_centrality(G),
        "Closeness": calculate_closeness_centrality(G),
        "Betweenness": calculate_betweenness_centrality(G),
        "PageRank": calculate_pagerank_centrality(G),
        "LGC": calculate_Laplacian_Gravity_centrality(G),
        "ILC": calculate_improved_laplacian_centrality(G),
        "RC": calculate_radiation_centrality(G),
        "WDKS": calculate_wdks_centrality(G),
        "GDC": calculate_Gaussian_Decay_Centrality(G),
    }

    results = {}
    for name, ranked_nodes in algorithms.items():
        try:
            # 提取得分并生成密集排名
            scores = {node: score for node, score in ranked_nodes}
            ranks = dense_ranking(scores)
            results[name] = ranks
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            results[name] = None
    return results

def main(network_path):
    # 加载网络
    G = nx.read_graphml(network_path)
    # G.remove_edges_from(nx.selfloop_edges(G)) #如果图有环
    # 转换为无向图（如果算法需要）
    G = G.to_undirected()

    # 运行所有算法
    all_ranks = run_all_centralities(G)

    # 计算各算法MI
    mi_results = {}
    for algo, ranks in all_ranks.items():
        if ranks is not None:
            mi = calculate_MI(ranks)
            mi_results[algo] = round(mi, 5)

    # 打印结果
    print("Monotonicity Index (MI) Results:")
    for algo, mi in mi_results.items():
        print(f"{algo:<15} MI = {mi:.6f}")

if __name__ == "__main__":
    network_path = "./data/real_Graphs/10_WV.graphml"  # 修改为实际路径
    main(network_path)