import networkx as nx
import numpy as np
from pathlib import Path

def compute_graph_features(file_path):
    """读取.graphml文件并计算拓扑特征（自动处理非连通图）"""
    try:
        G = nx.read_graphml(file_path)
        if not G.is_directed():
            G = G.to_undirected()

        # 基础特征计算
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = np.mean([d for _, d in G.degree()]) if num_nodes > 0 else 0
        avg_clustering = nx.average_clustering(G) if num_nodes > 0 else 0
        N, M = len(G.nodes()), len(G.edges())
        k = sum([G.degree(i) for i in G.nodes()]) / N
        k2 = sum([G.degree(i) ** 2 for i in G.nodes()]) / N
        beta_c = k / (k2 - k)
        # 平均最短路径计算逻辑
        avg_path_length = np.nan
        if num_nodes > 0:
            current_G = G
            is_connected = nx.is_connected(current_G)

            # 处理非连通图
            if not is_connected:
                # 获取最大连通分量
                components = list(nx.connected_components(current_G))
                largest_component = max(components, key=len)
                current_G = current_G.subgraph(largest_component).copy()
                print(f"Warning: {file_path.stem} is disconnected, using largest component (|V|={current_G.number_of_nodes()})")

            # 大图警告（基于实际计算的节点数）
            current_num_nodes = current_G.number_of_nodes()
            if current_num_nodes > 500:
                print(f"Progress: Calculating <d> for {file_path.stem} (|V|={current_num_nodes}), please wait...")

            # 执行计算
            if current_num_nodes > 1:
                avg_path_length = nx.average_shortest_path_length(current_G)
            else:
                avg_path_length = np.nan  # 单节点图无法计算路径长度

        return {
            "Dataset": file_path.stem,
            "|V|": num_nodes,
            "|E|": num_edges,
            "〈k〉": round(avg_degree, 2),
            "C": round(avg_clustering, 3),
            "<d>": round(avg_path_length, 3) if not np.isnan(avg_path_length) else "NaN",
            "Bth": beta_c,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# 其余代码保持不变...
# 设置路径（替换为你的实际路径）
data_dir = Path("./data/ultra-large-scale_networks")
graph_files = list(data_dir.glob("*.graphml"))

# 批量处理
results = []
for file in graph_files:
    if features := compute_graph_features(file):
        results.append(features)

# 打印结果（优化对齐格式）
print("\nTopological Features Summary:")
print("{:<15} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}".format("Dataset", "|V|", "|E|", "〈k〉","<d>", "C", "Bth"))
for res in results:
    print("{:<15} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}".format(
        res["Dataset"], res["|V|"], res["|E|"], res["〈k〉"], res["<d>"], res["C"], res["Bth"]
    ))


