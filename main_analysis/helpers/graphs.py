import networkx as nx
#%%  
"""  
    Common Graph Related Algos
"""

def intersection_of_graphs(graphs, edge_filter=None):
    intersect_G = nx.DiGraph()

    if edge_filter is None:
        edge_filter = len(graphs)  # very important conditions

    assert edge_filter <= len(graphs)
    assert edge_filter > 1

    # print(f">> Ingore the Edge if its not appearing in atleast {edge_filter}  graphs")

    edges = [edge for g in graphs for edge in g.edges]
    edges_dict = {edge: 0 for edge in edges}

    for edge in edges:
        edges_dict[edge] += 1

    # rmoving edeges which are not appearing in atleast "edge_filter" times.
    edges_dict = {k: v for k, v in edges_dict.items() if v >= edge_filter}

    intersect_G.add_edges_from(
        [(k[0], k[1], {"weight": 0}) for k in edges_dict.keys()])

    for g in graphs:
        for e in intersect_G.edges:
            if g.has_edge(*e) == True:
                intersect_G.edges[e]["weight"] += g.edges[e]["weight"]

    for n in intersect_G.nodes:
        intersect_G.nodes[n]["count"] = 0  # g.nodes[n]["count"]
        intersect_G.nodes[n]["size"] = 0  # g.nodes[n]["size"]

    for g in graphs:
        for n in intersect_G.nodes:
            if g.has_node(n) == True:
                intersect_G.nodes[n]["count"] += g.nodes[n]["count"]
                intersect_G.nodes[n]["size"] += g.nodes[n]["size"]

    return intersect_G

# def filterEdges(g, edge_drop_factors):
#     # g = nx.DiGraph(g)
#     remove_edges = []
#     for source, target, weight in g.edges.data("weight"):
#         alarm_count = g.nodes[target]["count"]
#         action_count = g.nodes[source]["count"]

#         action_alarms_ratio = alarm_count/action_count
#         weight_alarm_ratio =  weight/alarm_count

#         # if edge_drop_factor * weight < g.nodes[target]["count"]:
#         if  action_alarms_ratio <  0.4:
#             remove_edges.append((source, target))
#         elif action_alarms_ratio <0.3 and weight_alarm_ratio < 0.8:
#             remove_edges.append((source, target))

#     g.remove_edges_from(remove_edges)
#     g.remove_nodes_from(list(nx.isolates(g)))
#     return g

def getInDegree(g, n):
    return g.in_degree(n, weight="weight")

def getOutDegree(g, n):
    return g.out_degree(n, weight="weight")

