import EoN
import networkx as nx
import numpy as np


def sir_simulation(G, beta, num_simulations):
    node_influences = {}  # 用于存储每个节点的影响力

    for node in G.nodes():
        total_R_count = 0  # 用于存储总的R数量

        for _ in range(num_simulations):
            # 每次迭代选择不同的初始感染节点
            seed_node = node
            t, S, I, R = EoN.fast_SIR(G, beta, gamma=1, initial_infecteds=seed_node, tmax=50)
            total_R_count += len(R)  # 累加每次模拟的R数量

        # 计算平均影响力并标准化
        total_nodes = len(G.nodes())
        node_influence = total_R_count / (num_simulations * total_nodes)
        #node_influence = total_R_count / (num_simulations)
        node_influences[node] = node_influence

    return node_influences


def generate_labels(graph, beta, num_simulations, top_percent=0.05):
    # influence = {node: sir_simulation(graph, beta, initial_infected=node) for node in graph.nodes}
    influence = sir_simulation(graph, beta, num_simulations)
    sorted_nodes = sorted(influence, key=influence.get, reverse=True)
    cutoff = int(len(graph.nodes) * top_percent)
    labels = {node: 1 if i < cutoff else 0 for i, node in enumerate(sorted_nodes)}
    return labels


G = nx.read_graphml('./data/real_Graphs/01_Jazz.graphml')
G_name = '01_Jazz'
beta = 0.1
N, M = len(G.nodes()), len(G.edges())
# 传播阈值
k = sum([G.degree(i) for i in G.nodes()]) / N
k2 = sum([G.degree(i) ** 2 for i in G.nodes()]) / N
beta_c = k / (k2 - k)
beta = 1.0*beta_c
print("beta:", beta)
num_simulations = 1000
influence = sir_simulation(G, beta, num_simulations)
labels = generate_labels(G, beta, num_simulations, top_percent=0.05)

print("Influence:", influence)
print("Labels:", labels)

# Create a list of nodes in the desired order
node_list = list(G.nodes())

# Create an array of influence values in the same order
influence_array = np.array([influence[node] for node in node_list])
print("influence_array:", influence_array)
# Save the array to a .npy file
np.save("./data/results/" + G_name + ".npy", influence_array)

print("Influence array saved to ", G_name, ".npy")

# Sort nodes by influence in descending order
sorted_nodes = sorted(influence.items(), key=lambda x: x[1], reverse=True)

# Print the sorted nodes and their influence scores
print("\nSorted Nodes by Influence:")
for node, score in sorted_nodes:
    print(f"Node {node}: Influence {score}")
