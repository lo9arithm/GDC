import networkx as nx


def generate_er_network(num_nodes, num_edges):
    """
    生成随机网络（ER网络）
    :param num_nodes: 节点数
    :param num_edges: 边数
    :return: 生成的ER网络
    """
    G = nx.gnm_random_graph(num_nodes, num_edges)
    return G


def generate_ws_network(num_nodes, k, p):
    """
    生成小世界网络（WS网络）
    :param num_nodes: 节点数
    :param k: 每个节点连接的最近邻居数
    :param p: 重新连接边的概率
    :return: 生成的WS网络
    """
    G = nx.watts_strogatz_graph(num_nodes, k, p)
    return G


def generate_ba_network(num_nodes, m):
    """
    生成无标度网络（BA网络）
    :param num_nodes: 节点数
    :param m: 每次新增节点时连接的边数
    :return: 生成的BA网络
    """
    G = nx.barabasi_albert_graph(num_nodes, m)
    return G
