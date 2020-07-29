
from datetime import timedelta
from functools import partial

from helpers.graphs import filterEdges, getInDegree, intersection_of_graphs
import networkx as nx
from multiprocessing import Pool


def addOrUpdateNode(g, node):  # using partial thats why node as last arg
    if g.has_node(node) == True:
        g.nodes[node]["size"] += 0.01
        g.nodes[node]["count"] += 1
    else:
        g.add_node(node, size=10, count=0)

    return False

def constructSingleAlarmsRelationGraph(next_alarm_start_gapf, next_alarm_end_gapf, df_alarms):
    g = nx.DiGraph()

    # Adding Nodes
    _ = list(filter(partial(addOrUpdateNode, g), df_alarms["SourceName"]))

    start_records = [v for v in sorted(df_alarms.to_dict(
        orient="records"), key=lambda arg: arg["StartTime"], reverse=False)]

    # Adding Edges
    for i in range(len(start_records)):
        prevd = start_records[i]
        for j in range(i+1, len(start_records)):
            nextd = start_records[j]
            # if next alarm is not triggered within gapf duration then break
            if timedelta.total_seconds(nextd["StartTime"]-prevd["StartTime"]) > next_alarm_start_gapf:
                break
            elif nextd["SourceName"] != prevd["SourceName"] and nextd["StartTime"] >= prevd["StartTime"]:
                if nextd["EndTime"] <= prevd["EndTime"] or timedelta.total_seconds(nextd["EndTime"]-prevd["EndTime"]) < next_alarm_end_gapf:
                    if g.has_edge(prevd["SourceName"], nextd["SourceName"]) == False:
                        g.add_edge(prevd["SourceName"],
                                   nextd["SourceName"], weight=1)
                    else:
                        g.edges[prevd["SourceName"],
                                nextd["SourceName"]]["weight"] += 1

    print(">> # of nodes = {}, # of edges = {}".format(
        g.number_of_nodes(), g.number_of_edges()))
    return g


def constructMultipleAlarmsRelationGraphs(df_alarms, num_sub_graphs, start_time_gapf, end_time_gapf):
    graphs = []
    index_ranges = []
    batch_size = df_alarms.shape[0]//num_sub_graphs
    # Range Indexes
    for start in range(0, df_alarms.shape[0], batch_size):
        if start+batch_size <= df_alarms.shape[0]:
            index_ranges.append((start, start+batch_size))
    index_ranges[-1] = (index_ranges[-1][0], index_ranges[-1]
                        [1] + 1 + (df_alarms.shape[0] % num_sub_graphs))
    

    temp_dfs = []

    for t in index_ranges:
        print("        ----------------------------------")
        df1 = df_alarms.iloc[t[0]:t[1], :]
        # df1.reset_index(drop=True, inplace=True)
        min_date = df1["StartTime"].min()
        max_date = df1["StartTime"].max()
        print(">> Index Range = {}, Min & Max dates = {}".format(t, (min_date.date(), max_date.date())))
        # g = constructSingleAlarmsRelationGraph(df1, start_time_gapf, end_time_gapf)
        # graphs.append(g)
        temp_dfs.append(df1)
    
    with Pool(6) as p:
      graphs = p.map(partial(constructSingleAlarmsRelationGraph,start_time_gapf, end_time_gapf),temp_dfs)

    print("        ----------------------------------")

    return graphs


def getFinalAlarmRelationsGraph(df_alarms, num_sub_graphs, min_intersectionf, start_gap_next_alarmf, end_gap_next_alarmf):
    print(">> Starting to find relations between alarms....")
    print(">> Number of sub graphs to be constructed :", num_sub_graphs)
    print(f">> Next Alarm has to start within {start_gap_next_alarmf}")
    print(
        f">> Next Alarm has to end either before prev alarm or within {end_gap_next_alarmf} seconds after the ending of prev alarm")

    df_filtered_alarms = df_alarms.sort_values('StartTime')
    graphs = constructMultipleAlarmsRelationGraphs(
        df_filtered_alarms, num_sub_graphs, start_gap_next_alarmf, end_gap_next_alarmf)
    assert len(graphs) == num_sub_graphs

    print(f">> Taking intersection of {len(graphs)} sub-graphs")

    main_g = intersection_of_graphs(graphs, min_intersectionf)
    return main_g


# %% plotting

def nodeAnalysis(g, edge_drop_factor):
    def dropNeighbourLink(weight, count, drop_factor):
        if drop_factor*weight < count:
            return False
        else:
            return True

    nodes_dict = {}
    for source in list(g.nodes):
        neighs = [(g.edges[source, neighbour]['weight'], neighbour,
                   g.nodes[neighbour]["count"]) for neighbour in g.neighbors(source)]

        neighs = [t for t in sorted(neighs, key=lambda arg: arg[0], reverse=True) if dropNeighbourLink(
            t[0], t[2], edge_drop_factor)]

        nodes_dict[source] = {"count": g.nodes[source]["count"],
                              "ind": getInDegree(g, source), "neighbours": neighs}

    nodes_dict = {k: v for k, v in sorted(
        nodes_dict.items(), key=lambda arg: arg[1]["count"], reverse=True)}

    p = [f">>Source={k},c={v['count']},ind={v['ind']},Targets (weight,node,count) :{v['neighbours']}\n\n" for k, v in nodes_dict.items(
    ) if len(v['neighbours']) > 0]

    print(f">>Total number of relations found = {len(p)}\n")

    print(*p)


def analyzeAlarmdata(df, num_sub_graphs, min_intersection_f, next_start_gap,  next_end_gap, edge_drop_factor):
    G = getFinalAlarmRelationsGraph(df, num_sub_graphs, min_intersection_f,
                                    start_gap_next_alarmf=next_start_gap, end_gap_next_alarmf=next_end_gap)

    print(
        f">> Number of edges = {G.number_of_edges()}, # of nodes = {G.number_of_nodes()}")
    print(">> Done")

    nodeAnalysis(G, edge_drop_factor=edge_drop_factor)  # just for printing
    G = filterEdges(G, edge_drop_factor=edge_drop_factor)
    components = list(nx.weakly_connected_components(G))

    components = sorted(sorted(component) for component in components)
    node2common = {k: '=>'.join(l) for l in components for k in l}
    return G, components, node2common




    




