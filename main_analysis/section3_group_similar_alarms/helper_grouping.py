
from datetime import timedelta
from functools import partial

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.orca.config.executable = "/home/waris/anaconda3/envs/tupras/bin/orca"


def preProcessAlarmData(df, months=None, sources_filter=[], monmentarly_filter=20, staling_filter=(60*60)*24, ingore_communication_alarms=False, min_alarms_per_source=None):
    if min_alarms_per_source is None:
        min_alarms_per_source = len(list(df["SourceName"].unique()))
    print(
        f">>Preprocessing... \n   Months to include={months}\n  Ignore Sources={sources_filter}\n  Ingnore Momentarlily Alarms Filter={monmentarly_filter}seconds \n   Ignoreing Staling Alarms Filter={staling_filter/3600.0} hours, \n Ignore Communication Alarms = {ingore_communication_alarms} \n Remove sources whose count is less than {min_alarms_per_source}")
    if months is None:
        months = df["Month"].unique()

    df_new = df[(df["TimeDelta"] > monmentarly_filter) & (df["TimeDelta"] < staling_filter) & (
        df["Month"].isin(months)) & (~df["SourceName"].isin(sources_filter))]

    if ingore_communication_alarms:
        df_new = df_new[~df_new["Condition"].isin(["IOP", "IOP-"])]

    source2count = dict(df_new["SourceName"].value_counts())
    select_sources = [k for k, v in source2count.items() if v >=
                      min_alarms_per_source]
    df_new = df_new[df_new["SourceName"].isin(select_sources)]

    return df_new


def plotSourceAndCondtionHistogram(df):
    fig = px.histogram(df, x="SourceName", color='Condition')
    fig.show()


def intersection_of_graphs(graphs, edge_filter=None):
    intersect_G = nx.DiGraph()

    if edge_filter is None:
        edge_filter = len(graphs)  # very important conditions

    assert edge_filter <= len(graphs)
    assert edge_filter > 1

    print(
        f">> Ingore the Edge if its not appearing in atleast {edge_filter}  graphs")

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
        intersect_G.nodes[n]["count"] = 0
        intersect_G.nodes[n]["size"] = 0

    for g in graphs:
        for n in intersect_G.nodes:
            if g.has_node(n) == True:
                intersect_G.nodes[n]["count"] += g.nodes[n]["count"]
                intersect_G.nodes[n]["size"] += g.nodes[n]["size"]

    return intersect_G


def addOrUpdateNode(g, node):  # using partial thats why node as last arg
    if g.has_node(node) == True:
        g.nodes[node]["size"] += 0.01
        g.nodes[node]["count"] += 1
    else:
        g.add_node(node, size=10, count=0)

    return False


def constructSingleAlarmsRelationGraph(df_alarms, next_alarm_start_gapf, next_alarm_end_gapf):
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

    for t in index_ranges:
        print("        ----------------------------------")
        df1 = df_alarms.iloc[t[0]:t[1], :]
        df1.reset_index(drop=True, inplace=True)
        min_date = df1["StartTime"].min()
        max_date = df1["StartTime"].max()
        print(">> Index Range = {}, Min & Max dates = {}".format(
            t, (min_date.date(), max_date.date())))
        g = constructSingleAlarmsRelationGraph(
            df1, start_time_gapf, end_time_gapf)
        graphs.append(g)

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
def plotAlarmsRelationsHeatMap(g):

    int2alarm = dict(enumerate(g.nodes))
    alarm2int = {v: k for k, v in int2alarm.items()}
    data = [[None for i in range(len(alarm2int))]
            for j in range(len(alarm2int))]

    if len(data) == 0:
        print(" --------------> Heatmap:no data exist in heatmap")
        return None

    print(">> Dimension", len(data[0]), len(data))

    for s, d, weight in g.edges.data("weight"):
        # Reversing source and destinatin inorder to make x-axis source
        data[alarm2int[d]][alarm2int[s]] = weight

    fig = go.Figure(
        data=go.Heatmap(
            z=data, x=[v for k, v in int2alarm.items()], y=[v for k, v in int2alarm.items()],
            hoverongaps=False, hovertemplate=None, colorscale='Viridis'
        )
    )  # Greys

    fig.update_layout(
        width=1000,
        height=1000,
        xaxis_nticks=len(alarm2int),
        yaxis_nticks=len(alarm2int),
        yaxis=dict(title='Child', titlefont_size=16, tickfont_size=14),
        xaxis=dict(title='Parent', titlefont_size=16, tickfont_size=14)
    )

    fig.show()
    return data


def getInDegree(g, n):
    return g.in_degree(n, weight="weight")


def getOutDegree(g, n):
    return g.out_degree(n, weight="weight")


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


def filterEdges(g, edge_drop_factor):
    g = nx.DiGraph(g)
    remove_edges = []
    for source, target, weight in g.edges.data("weight"):
        if edge_drop_factor * weight < g.nodes[target]["count"]:
            remove_edges.append((source, target))

    g.remove_edges_from(remove_edges)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


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
    node2common = {k: '-'.join(l) for l in components for k in l}
    return G, components, node2common


def getTimeDelta(start_time, end_time):
    delta = timedelta.total_seconds(end_time-start_time)
    # print(delta)
    assert delta >= 0
    return delta


def concatenateSourceNameAndCondition(sname, condition):
    # print(sname)
    return sname+"-"+condition
