#%%
import pandas as pd
import networkx as nx
from datetime import timedelta


from helper_operator_actions import getFinalOperatorAlarmRelationGraph
from helper_operator_actions import preProcessAlarmData
from helper_operator_actions import plotOperatorAlarmRelationHeatMap

#%%
filter_short_alarms = [20, 120]  #seconds
staling_alarms_filter = (60*60) * 12
path = "../data/new/"


#%%

alarms_fname = "formatted-all-month-alarms.csv" 
operator_fname = "operator-all-month-actions.csv"
df_main_alarms = pd.read_csv(path + alarms_fname, low_memory=False ,parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(lambda arg: timedelta.total_seconds(arg)) 
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(lambda arg: arg.month)
df_main_actions = pd.read_csv(path + operator_fname, low_memory=False ,parse_dates=["EventTime"])
df_main_actions["Month"] = df_main_actions["EventTime"].apply(lambda arg: arg.month)


#%%
df_main_actions.info(), df_main_alarms.info()


# %%
months_f = df_main_alarms["Month"].unique()
num_sub_graphs = 16
min_intersection_f = 10
snames_f = ["47TI1713"]
# put month filter on operator data and then pass
df_alarms_new = preProcessAlarmData(df_main_alarms,months=months_f,sources_filter=snames_f,monmentarly_filter=20,staling_filter=60*60*12)
df_actions_new = df_main_actions[df_main_actions["Month"].isin(months_f)]
main_g = getFinalOperatorAlarmRelationGraph(df_alarms=df_alarms_new,df_actions=df_actions_new,num_graphs=num_sub_graphs,min_graphs_intersection_filter=min_intersection_f)
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



tg = nx.DiGraph(main_g)
data = plotOperatorAlarmRelationHeatMap(g=tg.to_directed(),filter_weight =200) # returning for debugging
 


# %%
