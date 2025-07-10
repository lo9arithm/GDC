import EoN
import networkx as nx
import numpy as np

def sis_simulation(G, beta, num_simulations):
    node_influences = {}  # 用于存储每个节点的影响力

    for node in G.nodes():
        total_infected_count = 0  # 用于存储模拟中感染的节点数量

        for _ in range(num_simulations):
            # 每次迭代选择不同的初始感染节点
            seed_node = node
            t, S, I = EoN.fast_SIS(G, beta, gamma=1, initial_infecteds=seed_node, tmax=50)
            total_infected_count += len(I)  # 累加每次模拟中感染的节点数量

        # 计算平均影响力并标准化
        total_nodes = len(G.nodes())
        node_influence = total_infected_count / (num_simulations * total_nodes)
        node_influences[node] = node_influence

    return node_influences


def generate_labels(graph, beta, num_simulations, top_percent=0.05):
    influence = sis_simulation(graph, beta, num_simulations)
    sorted_nodes = sorted(influence, key=influence.get, reverse=True)
    cutoff = int(len(graph.nodes) * top_percent)
    labels = {node: 1 if i < cutoff else 0 for i, node in enumerate(sorted_nodes)}
    return labels


G = nx.read_graphml('./data/real_Graphs/Politicalblogs.graphml')
G_name = 'Politicalblogs'
beta = 0.1
N, M = len(G.nodes()), len(G.edges())
# 传播阈值
k = sum([G.degree(i) for i in G.nodes()]) / N
k2 = sum([G.degree(i) ** 2 for i in G.nodes()]) / N
beta_c = k / (k2 - k)
beta = 1.0 * beta_c  # 计算beta值
print("beta:", beta)
num_simulations = 1000  # 模拟次数
influence = sis_simulation(G, beta, num_simulations)  # 获取每个节点的影响力
labels = generate_labels(G, beta, num_simulations, top_percent=0.05)  # 生成标签

print("Influence:", influence)
print("Labels:", labels)

# 创建节点列表
node_list = list(G.nodes())

# 按照节点的影响力排序
influence_array = np.array([influence[node] for node in node_list])
print("influence_array:", influence_array)
# 保存为.npy文件
np.save("./data/SIS_labels/" + G_name + ".npy", influence_array)

print("Influence array saved to ", G_name, ".npy")

# 根据影响力排序节点
sorted_nodes = sorted(influence.items(), key=lambda x: x[1], reverse=True)

# 打印排序后的节点和影响力分数
print("\nSorted Nodes by Influence:")
for node, score in sorted_nodes:
    print(f"Node {node}: Influence {score}")
