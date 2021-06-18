from helpers.graphs import filterEdges, intersection_of_graphs
import networkx as nx
from functools import partial
from multiprocessing import Pool


def printGraph(g):
    main_g = g
    operator_nodes = [action for action in g.nodes if action.find("Operator") != -1]
    for op in operator_nodes:
        action_count = main_g.nodes[op]['count']
        num_neg = len(list(main_g.neighbors(op)))
        print(f"{op} Count:{action_count}|| Correspodning Alarms = {num_neg} ", end = " " )
        # s = ""
        for n in main_g.neighbors(op):
            alarm_count = main_g.nodes[n]['count']
            edge_weight = main_g.edges[(op,n)]['weight']
            action_alarms_ratio = alarm_count/action_count


            print(f"{n}(count:{alarm_count})(weight:{edge_weight}), (ratio: {action_alarms_ratio})", end = ", ")
    
    print("\n -----------------------------------------------")


def __addOrUpdateEdge(g, e):
    if g.has_edge(*e) == True:
        g.edges[e]["weight"] += 1
    else:
        g.add_edge(*e, weight=1)

    return False


def __addOrUpdateNode(g, operatorf, node):  # using partial thats why node as last arg
    if operatorf == True:
        node = "Operator->" + node
    if g.has_node(node) == True:
        g.nodes[node]["size"] += 0.01
        g.nodes[node]["count"] += 1
    else:
        g.add_node(node, size=10, count=0)

    return False


# ignoring filter -> and timedelta.total_seconds(alarm["EndTime"]-action["EventTime"])<filters["gap"]
def __checkAction_OnAlarm(action, alarm):
    if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= alarm["EndTime"]:
        return True
    else:
        return False

    # if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= min(30*60, alarm["EndTime"]- alarm["StartTime"]) :


def __constructSingleAlarmsActionsGraph(args):  # case 3
    df_alarms  = args[0]
    df_actions = args[1] 
    g = nx.DiGraph()  # Directed graph
    print(">> Test",df_alarms["Year-Month"].unique(), df_actions["Year-Month"].unique())
    assert len(df_alarms["Year-Month"].unique()) == len(
        df_actions["Year-Month"].unique())
    for al_year_month, op_year_month in zip(df_alarms["Year-Month"].unique(), df_actions["Year-Month"].unique()):
        assert al_year_month == op_year_month
     
    print(">> # of Alarms:{} and Operator Actions:{}".format(
        df_alarms.shape[0], df_actions.shape[0]))

    # ---------- Adding Nodes----------------
    _ = list(filter(partial(__addOrUpdateNode, g, False),
                    df_alarms["SourceName"]))
    _ = list(filter(partial(__addOrUpdateNode, g, True),
                    df_actions["SourceName"]))

    # converting rows to dicts
    # [alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
    alarms = df_alarms.to_dict(orient="records")
    # [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    actions = df_actions.to_dict(orient="records")
    # -------- Adding Edges -----------------
    edges = [("Operator->"+action["SourceName"], alarm["SourceName"])
             for action in actions for alarm in alarms if __checkAction_OnAlarm(action, alarm)]

    _ = list(filter(partial(__addOrUpdateEdge, g), edges))

    print(">> # of nodes = {}, # of edges = {}".format(
        g.number_of_nodes(), g.number_of_edges()))

    # printGraph(g)

    return g


def __constructMultipleAlarmsActionsGraphs(df_alarms, df_actions, chunks):
    graphs = []
    index_ranges = []
    batch_size = df_alarms.shape[0]//chunks
    

    # Range Indexes
    for start in range(0, df_alarms.shape[0], batch_size):
        if start+batch_size <= df_alarms.shape[0]:
            index_ranges.append((start, start+batch_size))
    index_ranges[-1] = (index_ranges[-1][0], index_ranges[-1]
                        [1] + 1 + (df_alarms.shape[0] % chunks))
#     index_ranges.reverse()
#     print(">> Chunks = {}, Index Ranges = {}".format(chunks,index_ranges))
    temp_dfs = []

    for t in index_ranges:
        print("        ----------------------------------")
        df1 = df_alarms.iloc[t[0]:t[1], :]
        # df1.reset_index(drop=True, inplace=True)

        min_date = df1["StartTime"].min()
        max_date = df1["StartTime"].max()
        # print(
        #     f">> Index Range = {t}, Min Date ={min_date.date()} & Max date = {max_date.date()}")
        # print(">> Filtering the Operator actions based on min-max dates ...")

        df2 = df_actions.loc[(df_actions["EventTime"] >= min_date)
                         & (df_actions["EventTime"] <= max_date)]
        # df2.reset_index(drop=True, inplace=True)

        temp_dfs.append((df1,df2))

        g = __constructSingleAlarmsActionsGraph((df1, df2))
        graphs.append(g)
    
    
    # graphs = []

    # for temp_df in temp_dfs:
    #     graphs.append(__constructMultipleAlarmsActionsGraphs())
    
    # with Pool(5) as p:
    #     graphs = p.map(__constructSingleAlarmsActionsGraph,temp_dfs)



    print("        ----------------------------------")
    return graphs


def __getFinalOperatorAlarmRelationGraph(df_alarms, df_actions, num_graphs, min_graphs_intersection_filter):
    print(">> Finding relation between operator action and alarm")
    print(">> Number of sub graphs to be constructed :", num_graphs)

    # df_alarms[(df_alarms["TimeDelta"]>durationf)& (df_alarms["TimeDelta"]<stalingf) & (df_alarms["Month"].isin(monthsf)) & (~df_alarms["SourceName"].isin(snamesf))]
    df_filtered_alarms = df_alarms
    # df_actions[df_actions["Month"].isin(monthsf)]
    df_filtered_actions = df_actions
    df_filtered_alarms = df_filtered_alarms.sort_values('StartTime')
    df_filtered_actions = df_filtered_actions.sort_values('EventTime')

    graphs = __constructMultipleAlarmsActionsGraphs(
        df_filtered_alarms, df_filtered_actions, num_graphs)
    assert len(graphs) == num_graphs
    print(">> Taking intersection of {} sub-graphs".format(len(graphs)))

    main_g = intersection_of_graphs(graphs, min_graphs_intersection_filter)

    return main_g

def __getTrueAndNuisanceSources(df, g):
    source_names_requried_action = [
        source for source in g.nodes if source.find("Operator") == -1]
    df_temp = df[~df["SourceName"].isin(source_names_requried_action)]
    sources_not_require_action = df_temp["SourceName"].unique()
    return source_names_requried_action, sources_not_require_action

def getTrueAndNuisanceSourceNames(df_alarms,df_operator, num_sub_graphs, min_intersection_f,edge_filter):
     
    sources_need_action, sources_dont_need_action = __getTrueAndNuisanceSources(df_alarms, g=filterEdges(__getFinalOperatorAlarmRelationGraph(df_alarms=df_alarms, df_actions=df_operator,
                                                num_graphs=num_sub_graphs, min_graphs_intersection_filter=min_intersection_f), edge_drop_factor=edge_filter))
    return sources_need_action, sources_dont_need_action
