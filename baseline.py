import numpy as np
import networkx as nx


# 度中心性
def calculate_degree_centrality(G):
    # 计算度中心性
    degree_centrality = nx.degree_centrality(G)

    # 按度中心性降序排列
    sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    return sorted_degree


#接近中心性
def calculate_closeness_centrality(G):
    # 计算接近中心性
    closeness = nx.closeness_centrality(G)

    # 按接近中心性降序排列
    sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

    return sorted_closeness


# 介数中心性
def calculate_betweenness_centrality(G):
    # 计算介数中心性
    betweenness = nx.betweenness_centrality(G)

    # 按介数中心性降序排列
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

    return sorted_betweenness


# PageRank中心性
def calculate_pagerank_centrality(G, d=0.85):
    # 计算 PageRank 中心性
    pagerank = nx.pagerank(G, alpha=d)

    # 按 PageRank 值降序排列
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    return sorted_pagerank


def calculate_Laplacian_Gravity_centrality(G):
    """
    Calculate the Laplacian Gravity Centrality (LGC) for nodes in a graph.

    Parameters:
    G (networkx.Graph): The graph for which to calculate LGC.

    Returns:
    list of tuples: A list where each tuple contains a node and its LGC score, sorted in descending order of the score.
    """
    # Precompute Laplacian Centrality (LC) for all nodes
    lc = {}
    for node in G.nodes:
        k_i = G.degree(node)
        neighbors = list(G.neighbors(node))
        sum_k_j = sum(G.degree(j) for j in neighbors)
        LC_i = k_i ** 2 + k_i + 2 * sum_k_j
        lc[node] = LC_i

    # Calculate average path length
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        components = nx.connected_components(G)
        total_pairs = len(G.nodes) * (len(G.nodes) - 1) / 2
        avg_path_length = 0
        for component in components:
            subgraph = G.subgraph(component)
            n = len(subgraph.nodes)
            if n < 2:
                continue
            comp_avg = nx.average_shortest_path_length(subgraph)
            avg_path_length += comp_avg * (n * (n - 1) / 2) / total_pairs

    if avg_path_length / 2 < 1:
        truncation_radius = 1
    else:
        truncation_radius = avg_path_length / 2

    # Compute LGC for each node
    lgc_scores = {}
    for node in G.nodes:
        # Find nodes within truncation_radius
        distances = dict(nx.single_source_shortest_path_length(G, node))
        within_radius = {j: d for j, d in distances.items() if j != node and d <= truncation_radius}

        # Sum LC_i * LC_j / d_ij^2
        LC_i = lc[node]
        sum_lgc = 0
        for j, d_ij in within_radius.items():
            LC_j = lc[j]
            sum_lgc += (LC_i * LC_j) / (d_ij ** 2)

        lgc_scores[node] = sum_lgc

    # Sort the LGC scores in descending order
    sorted_lgc = sorted(lgc_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_lgc


# 辐射中心性
def calculate_radiation_centrality(G, alpha=0.2, beta=-0.5, gamma=1):
    # alpha (float): 衰减系数，默认为 0.2
    # beta (float): 散射系数，默认为 -0.5
    # gamma (float): 局部影响力系数，默认为 1

    # 计算每个节点的辐射中心性
    RC = {}
    for node in G.nodes:
        # 计算辐射源强度 (SI)
        k_v = G.degree(node)
        SI = k_v ** gamma

        # 计算消光影响 (EI)
        EI = 0
        for other_node in G.nodes:
            if other_node != node:
                try:
                    # 计算最短路径
                    shortest_path = nx.shortest_path(G, source=node, target=other_node)
                    d_vu = len(shortest_path) - 1  # 最短路径长度

                    # 计算路径上的散射效应
                    scattering_effect = 1
                    for m in range(1, d_vu):  # 从第二个节点开始
                        k_m = G.degree(shortest_path[m])
                        scattering_effect *= k_m ** beta

                    # 计算消光影响
                    EI += np.exp(-alpha * d_vu) * scattering_effect
                except nx.NetworkXNoPath:
                    # 如果节点之间没有路径，则跳过
                    continue

        # 计算辐射中心性 (RC)
        RC[node] = SI * EI

    # 按辐射中心性降序排列
    sorted_RC = sorted(RC.items(), key=lambda x: x[1], reverse=True)

    return sorted_RC

def calculate_wdks_centrality(G):
    """
    整合所有功能的WDKS中心性计算函数
    输入：networkx.Graph对象
    输出：按WDKS值降序排列的节点元组
    """
    # 处理空图情况
    if not G.nodes():
        return tuple()

    # 深拷贝图结构避免修改原始图
    working_graph = G.copy()

    # 阶段1：计算k-shell值 --------------------------------------------------------
    kshell = {n: 0 for n in working_graph.nodes()}
    current_shell = 1

    while working_graph.number_of_nodes() > 0:
        removed = True
        while removed:
            removed = False
            # 获取当前所有节点的度
            current_degrees = dict(working_graph.degree())
            # 找到所有度 <= current_shell的节点
            to_remove = [n for n, d in current_degrees.items() if d <= current_shell]

            if to_remove:
                removed = True
                for n in to_remove:
                    kshell[n] = current_shell  # 记录k-shell值
                    working_graph.remove_node(n)  # 从工作图中移除

        # 当没有更多节点可移除时，提升current_shell
        if working_graph.number_of_nodes() > 0:
            current_shell = min(dict(working_graph.degree()).values())
        else:
            break

    # 阶段2：计算度中心性 --------------------------------------------------------
    degrees = dict(G.degree())

    # 阶段3：计算边权重和平均值 --------------------------------------------------
    edges = list(G.edges())
    if not edges:  # 处理无边图的情况
        return tuple(sorted(G.nodes()))

    total_weight = 0.0
    for u, v in edges:
        # 计算每条边的权重 (k_i * ks_i) * (k_j * ks_j)
        w_uv = (degrees[u] * kshell[u]) * (degrees[v] * kshell[v])
        total_weight += w_uv

    avg_weight = total_weight / len(edges)  # 平均边权重

    # 阶段4：计算WDKS值 ---------------------------------------------------------
    wdks_scores = {}

    for node in G.nodes():
        # 内在中心性
        intrinsic = degrees[node] * kshell[node]

        # 外在中心性计算
        extrinsic = 0.0
        for neighbor in G.neighbors(node):
            # 邻居节点的k*ks值
            neighbor_value = degrees[neighbor] * kshell[neighbor]
            # 边权重贡献
            edge_contribution = (degrees[node] * kshell[node]) * neighbor_value
            # 标准化后的贡献值
            normalized_contribution = (edge_contribution / avg_weight) * neighbor_value
            extrinsic += normalized_contribution

        # 总分 = 内在 + 外在
        wdks_scores[node] = intrinsic + extrinsic

    # 阶段5：排序处理 -----------------------------------------------------------
    sorted_nodes = sorted(wdks_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes



def calculate_improved_laplacian_centrality(G):
    """计算图中每个节点的改进拉普拉斯中心性(ILC)"""
    # 第一阶段：计算拉普拉斯中心性(LC)
    degree_centrality = dict(G.degree())
    laplacian_cent = {}

    for node in G.nodes():
        # 获取节点度数和邻居列表
        k_i = degree_centrality[node]
        neighbors = list(G.neighbors(node))

        # 计算邻居度数之和
        sum_kj = sum(degree_centrality[nbr] for nbr in neighbors)

        # 计算LC值（公式13）
        laplacian_cent[node] = k_i**2 + k_i + 2 * sum_kj

    # 第二阶段：计算改进的拉普拉斯中心性(ILC)
    ilc_scores = {}

    for node in G.nodes():
        # 获取节点LC值和邻居列表
        lc_i = laplacian_cent[node]
        neighbors = list(G.neighbors(node))

        # 计算邻居LC值之和
        sum_lc_j = sum(laplacian_cent[nbr] for nbr in neighbors)

        # 计算ILC值（公式14）
        ilc_scores[node] = lc_i**2 + lc_i + 2 * sum_lc_j

    # 按ILC值降序排序
    return sorted(ilc_scores.items(), key=lambda x: x[1], reverse=True)