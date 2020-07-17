# %%
# from main.section-1-storage-analyis.storage-analyis import path
from datetime import timedelta

import networkx as nx
import pandas as pd

from helper_operator_actions import (getFinalOperatorAlarmRelationGraph,
               plotOperatorAlarmRelationHeatMap,
               preProcessAlarmData)

# %%
filter_short_alarms = [20, 120]  # seconds
staling_alarms_filter = (60*60) * 12
PATH = "/home/waris/Github/tupras-analysis/data/"
path_alarms = PATH + "processed/alarms/"
path_op_actions = PATH + "processed/operator-actions/"

# %%

alarms_fname = "formatted-all-month-alarms.csv"
operator_fname = "operator-all-month-actions.csv"
df_main_alarms = pd.read_csv(path_alarms + alarms_fname,
                             low_memory=False, parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - \
    df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(
    lambda arg: timedelta.total_seconds(arg))
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(
    lambda arg: arg.month)

df_main_actions = pd.read_csv(
    path_op_actions + operator_fname, low_memory=False, parse_dates=["EventTime"])
df_main_actions["Month"] = df_main_actions["EventTime"].apply(
    lambda arg: arg.month)


# %%
df_main_actions.info(), df_main_alarms.info()


# %%
months_f = df_main_alarms["Month"].unique()
num_sub_graphs = 8
min_intersection_f = 6
snames_f = ["47TI1713"]
# put month filter on operator data and then pass
df_alarms_new = preProcessAlarmData(df_main_alarms, months=months_f,
                                    sources_filter=snames_f, monmentarly_filter=20, staling_filter=60*60*12)
df_actions_new = df_main_actions[df_main_actions["Month"].isin(months_f)]
main_g = getFinalOperatorAlarmRelationGraph(df_alarms=df_alarms_new, df_actions=df_actions_new,
                                            num_graphs=num_sub_graphs, min_graphs_intersection_filter=min_intersection_f)
print(">> Done")


# %%
# print(">> # of Edges Main graph1 {} and in graph2 {}".format( main_g.number_of_edges() , main_g.number_of_edges()))
# operator_nodes = [action for action in main_g.nodes if action.find("Operator")!=-1]

# print(f">> Total number of operator nodes in the graph={len(operator_nodes)}")
# print (f">> Total number of Alarm Tags in the graph = {len(main_g.nodes)- len(operator_nodes)}")

# node2num_neigbs = {n: len(list(main_g.neighbors(n))) for n in operator_nodes}
# t3 = [(source,main_g.nodes[source]["count"],num_neighbs) for source,num_neighbs in node2num_neigbs.items()]
# print(*[">>{},{},# of neighbours {}\n".format(*tup) for tup in sorted(t3,key=lambda arg: arg[1], reverse=True)])
# Verify the results based count with orignal csv da


# tg = nx.DiGraph(main_g)
# data = plotOperatorAlarmRelationHeatMap(
#     g=tg.to_directed())  # returning for debugging


# %%

def filterEdges(g,edge_drop_factor):
    g = nx.DiGraph(g)
    remove_edges = []
    for source,target,weight in g.edges.data("weight"):
        if edge_drop_factor * weight < g.nodes[target]["count"]:
            remove_edges.append((source,target))
    
    g.remove_edges_from(remove_edges)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g

g = filterEdges(main_g,1.2)
_=plotOperatorAlarmRelationHeatMap(g)


# %%
