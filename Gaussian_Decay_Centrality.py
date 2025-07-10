import math
import networkx as nx

def calculate_Gaussian_Decay_Centrality(G):
    # 计算特征向量中心性
    eigenvector_centrality = nx.eigenvector_centrality(G, weight=None)

    # 计算边权，设置为端点的度值的几何平均数
    for u, v in G.edges():
        k_u = G.degree(u)
        k_v = G.degree(v)
        w = math.sqrt(k_u * k_v) # 几何平均数
        G[u][v]['weight'] = w

    nodes = list(G.nodes())

    centrality = {}
    #threshold = 25  # 设置适当的阈值
    for i in nodes:
        I = eigenvector_centrality[i]  # 使用特征向量中心性
        total_attenuation = 0.0
        shortest_paths = nx.single_source_dijkstra_path(G, i, weight='weight')
        for j in nodes:
            if i != j and j in shortest_paths:
                path = shortest_paths[j]
                d_ij = len(path) - 1
                edge_weights = [G[path[m]][path[m+1]]['weight'] for m in range(len(path)-1)]
                # 获取最大值和最小值
                wij = sum(edge_weights)
                try:
                    attenuation = wij / (d_ij *math.exp(d_ij**2))
                except OverflowError:
                    attenuation = 0
                total_attenuation += attenuation
        centrality[i] = I * total_attenuation  # 中心性 = 特征向量中心性 * 衰减项

    # 按影响力降序排列
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    return sorted_centrality

