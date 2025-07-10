import networkx as nx

from kendall_evaluate import calculate_kendall_all_centralities

# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/adjnoun.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email 08_PB 01_Jazz 13_Power
G_name = "adjnoun"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")
print("————————————————————————————————————————————————————————————")
# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/01_Jazz.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email 08_PB 01_Jazz 13_Power
G_name = "01_Jazz"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")
print("————————————————————————————————————————————————————————————")

# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/04_EEC.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email 08_PB 01_Jazz 13_Power
G_name = "04_EEC"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")
print("————————————————————————————————————————————————————————————")
# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/05_Email.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email
G_name = "05_Email"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")
print("————————————————————————————————————————————————————————————")

# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/Politicalblogs.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email
G_name = "Politicalblogs"
G.remove_edges_from(nx.selfloop_edges(G))
print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")

print("————————————————————————————————————————————————————————————")

# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/soc-hamsterster.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email
G_name = "soc-hamsterster"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")

print("————————————————————————————————————————————————————————————")

# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/06_PG.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email
G_name = "06_PG"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")

print("————————————————————————————————————————————————————————————")

# 在数据集生成的网络上计算肯德尔系数
G = nx.read_graphml('./data/real_Graphs/10_WV.graphml')
# 待使用数据集：12_USAir 04_EEC 05_Email 08_PB
G_name = "10_WV"

print("G_name：", G_name)
# 假设影响力数组保存在文件中
SIR_influence_filepath = './data/real_labels_B_Bth_1.0/' + G_name + '.npy'

# 计算所有中心性算法与 SIR 模型的肯德尔系数
results = calculate_kendall_all_centralities(G, SIR_influence_filepath)

# 输出结果
for name, tau in results.items():
    print(f"{name}: 肯德尔系数 = {tau}")
print("————————————————————————————————————————————————————————————")


