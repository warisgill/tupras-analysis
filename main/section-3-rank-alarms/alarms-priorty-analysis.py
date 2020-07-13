#%%
import pandas as pd
import networkx as nx
from pyvis.network import Network
from datetime import timedelta


from helper_priority import preProcessAlarmData
from helper_priority import getFinalAlarmRelationsGraph

#%%
filter_short_alarms = [20, 120]  #seconds
staling_alarms_filter = (60*60) * 12
PATH = "/home/waris/Github/tupras-analysis/data/"
path_alarms = PATH + "processed/alarms/"
path_op_actions = PATH + "processed/operator-actions/"

#%%

alarms_fname = "formatted-all-month-alarms.csv" 
operator_fname = "operator-all-month-actions.csv"
df_main_alarms = pd.read_csv(path_alarms + alarms_fname, low_memory=False ,parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(lambda arg: timedelta.total_seconds(arg)) 
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(lambda arg: arg.month)


#%%

months_f = df_main_alarms["Month"].unique()

snames_f = ["47TI1713", "47TI931A"]

# put month filter on operator data and then pass
df_alarms_new = preProcessAlarmData(df_main_alarms,months=months_f,sources_filter=snames_f,monmentarly_filter=20,staling_filter=60*60*12)

num_sub_graphs = 16
min_intersection_f = 10
main_g = getFinalAlarmRelationsGraph(df_alarms_new,num_sub_graphs,min_intersection_f, start_gap_next_alarmf= 60 *2 ,end_gap_next_alarmf= 60*60*4)
print(f">> # of Edges Main graph {main_g.number_of_edges()}")
print(">> Done")


#%%


# Google PageRank Algo
result = nx.algorithms.link_analysis.pagerank_alg.pagerank(main_g,weight="weight",max_iter=100000)
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Page Rank (Highest to Lowest) :",list(result.items())[:50])

print("              --------------------------------------------------")

# Eigenvector Centrality Algo
result = nx.eigenvector_centrality(main_g, weight="weight")
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Eigenvector Centrality (Highest to Lowest) :",list(result.items())[:50])

#%%
# Hits Algo
h, a = nx.hits(main_g)
result = h
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Hub => Outgoing Edges: Based on out_degree (max to min):",list(result.items())[:50])

print("              --------------------------------------------------")

result = a
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Auth => Incoming Edges: Based on in_degree (max to min):",list(result.items())[:50])


#%%
# Degree Analysis
nodes_dict = {}
for s in list(main_g.nodes):
    nodes_dict[s] = {"count":main_g.nodes[s]["count"], "outd" : main_g.out_degree(s,"weight"), "ind":main_g.in_degree(s,"weight"), "totald": main_g.degree(s,"weight")}

nodes_dict = {k:v for k, v in sorted(nodes_dict.items(), key=lambda arg: arg[1]["ind"], reverse=True) if v["count"]>4 and v["totald"]>1}
print(nodes_dict)

#%%
""" 
    For Visualization
"""

for edge in main_g.edges:
    main_g.edges[edge]["value"] = main_g.edges[edge]["weight"]
    main_g.edges[edge]["title"] = f"{edge},{main_g.edges[edge]['weight']}"

for node in main_g.nodes:
    main_g.nodes[node]["title"] = "{}, {}".format(node,main_g.nodes[node]['count']) 


nt = Network("400x", "100%", notebook=True)
nt.from_nx(main_g)
nt.options.edges.Color  = "Gray"
nt.toggle_hide_edges_on_drag(True)
# nt.toggle_physics(False)
nt.repulsion()
nt.toggle_stabilization(True)
nt.show_buttons(filter_=['physics','edges'])
nt.show("nt.html")


# %%
