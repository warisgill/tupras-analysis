#%%
# from section_2_operator_actions_analysis.helper_operator_actions import constructSingleAlarmsActionsGraph
from networkx.algorithms import components
import pandas as pd
import networkx as nx
from pyvis.network import Network
from datetime import timedelta

from helper_priority import preProcessAlarmData
from helper_priority import getFinalAlarmRelationsGraph
from helper_priority import plotAlarmsRelationsHeatMap
from helper_priority import constructSingleAlarmsRelationGraph
#%%
filter_short_alarms = [20, 120]  #seconds
staling_alarms_filter = (60*60) * 12
PATH = "/home/waris/Github/tupras-analysis/data/"
path_alarms = PATH + "processed/alarms/"

#%%
def getInDegree(g,n):
        return g.in_degree(n,weight="weight")

def getOutDegree(g,n):
    return g.out_degree(n,weight="weight")

def nodeAnalysis(g, edge_drop_factor):
    def dropNeighbourLink(weight,count,drop_factor):
        if drop_factor*weight < count:
            return False
        else:
            return True
    
    nodes_dict = {}
    for source in list(g.nodes):
        neighs = [(g.edges[source,neighbour]['weight'],neighbour,g.nodes[neighbour]["count"])  for neighbour in  g.neighbors(source)]
        
        neighs = [t for t in sorted(neighs, key=lambda arg: arg[0],reverse=True) if dropNeighbourLink(t[0],t[2],edge_drop_factor)]
        
        nodes_dict[source] = {"count":g.nodes[source]["count"], "ind":getInDegree(g,source), "neighbours" : neighs}


    nodes_dict = {k:v for k,v in sorted(nodes_dict.items(), key=lambda arg:arg[1]["count"],reverse=True)}

    p  = [f">>Source={k},c={v['count']},ind={v['ind']},Targets (weight,node,count) :{v['neighbours']}\n\n" for k,v in nodes_dict.items() if len(v['neighbours'])>0]

    print(f">>Total number of relations found = {len(p)}\n")

    print(*p)

def filterEdges(g,edge_drop_factor):
    g = nx.DiGraph(g)
    remove_edges = []
    for source,target,weight in g.edges.data("weight"):
        if edge_drop_factor * weight < g.nodes[target]["count"]:
            remove_edges.append((source,target))
    
    g.remove_edges_from(remove_edges)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g

def analyzeAlarmdata(df,num_sub_graphs,min_intersection_f, next_start_gap,  next_end_gap, edge_drop_factor):
    G = getFinalAlarmRelationsGraph(df,num_sub_graphs,min_intersection_f, start_gap_next_alarmf= next_start_gap,end_gap_next_alarmf=next_end_gap)

    print(f">> Number of edges = {G.number_of_edges()}, # of nodes = {G.number_of_nodes()}")
    print(">> Done")

    nodeAnalysis(G,edge_drop_factor=edge_drop_factor) # just for printing
    G =filterEdges(G,edge_drop_factor=edge_drop_factor)
    components = list(nx.weakly_connected_components(G))

    components  =sorted(sorted(component) for component in components)
    node2common = {k: '-'.join(l)  for l in components for k in l}
    return G,components, node2common



#%%

alarms_fname = "formatted-all-month-alarms.csv" 
operator_fname = "operator-all-month-actions.csv"
df_main_alarms = pd.read_csv(path_alarms + alarms_fname, low_memory=False ,parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(lambda arg: timedelta.total_seconds(arg)) 
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(lambda arg: arg.month)

"""Chaning name 2 alias """
alias2name =  {f"A{k}":v for k,v in enumerate(df_main_alarms["SourceName"].unique())}
name2alias = {v:k for k,v in alias2name.items()}
df_main_alarms["SourceName"] = df_main_alarms["SourceName"].apply(lambda sname: name2alias[sname])


# source2newName =maskSourceNames(df_main_alarms)
#%%
"""Level 0"""
print(">> ===== Level Zero ==================")
snames_f = ["47TI1713", "47TI931A"]
snames_f = [name2alias[s] for s in snames_f]
months_f = df_main_alarms["Month"].unique()
fleeting_alarms_filter = 20
staling_alarms_filter = 60*60*12

df_alarms_new = preProcessAlarmData(df_main_alarms,months=months_f,sources_filter=snames_f,monmentarly_filter=fleeting_alarms_filter,staling_filter=staling_alarms_filter)
# print(df_alarms_new["SourceName"].unique())

groups = ["".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("-") != -1]
assert len(groups) == 0
print(f">> Length = {len(groups)}, Groups={groups}")

graph,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=1.25)
df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias)
_ = plotAlarmsRelationsHeatMap(graph)

graph,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=1.25)
df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias)
_ = plotAlarmsRelationsHeatMap(graph)

graph,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=1.25)
df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias)
_ = plotAlarmsRelationsHeatMap(graph)

graph,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=1.25)
# df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias)
_ = plotAlarmsRelationsHeatMap(graph)


#%%
"""Level 1"""
# g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=1.5)
iteration = 0
edge_drop_factor = 1.0
while edge_drop_factor<2:
    g = None
    while True:
        iteration += 1
        print(f">> ==========Level=1 Iteration of Merging Components = {iteration} =============")
        g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=edge_drop_factor)
        # if g is not None:
        
        df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias) 
        print(components)
        if len(components) == 0:
            break
    
    
    
    edge_drop_factor += 0.1
    
    groups = ["".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("-") != -1]
    print(f">>Edge drop factor= {edge_drop_factor} and final Groups:  Length = {len(groups)}, Groups={groups}")


#%%
sourcecount = dict(df_alarms_new["SourceName"].value_counts())
print(sourcecount)
# source2count = {k:v for k,v in }

#%%
""" 
    For Visualization
"""

""" Vis Netowrk """


# for edge in graph.edges:
#     # graph.edges[edge]["value"] = graph.edges[edge]["weight"]
#     graph.edges[edge]["title"] = f"{edge},{graph.edges[edge]['weight']}"

# for node in graph.nodes:
#     graph.nodes[node]["title"] = f"{node}, {graph.nodes[node]['count']} \n In_Degree = {graph.in_degree(node,weight='weight')}, Out_Degree= {graph.out_degree(node,weight='weight')}" 

# nt = Network("400x", "100%", notebook=True)
# nt.from_nx(graph)
# # nt.set_options("""var options = {
# #     nodes: {
# #       scaling: {
# #         customScalingFunction: function(min, max, total, value) {
# #           return value / total;
# #         },
# #         min: 5,
# #         max: 150
# #       }
# #     }
# #   }""")
# nt.toggle_hide_edges_on_drag(True)
# nt.hrepulsion(spring_length=300,central_gravity=0,spring_strength=0)
# # nt.force_atlas_2based(spring_strength=0,spring_length=300)

# nt.show_buttons(filter_=['physics','edges'])
# # nt.inherit_edge_colors(True)
# # nt.toggle_physics(False)
# nt.show("nt.html")


#%%
# """Level 2"""
# g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=1.5)
# iteration = 0
# while len(components) !=0:
#     iteration += 1
#     print(f">> ========== Level 2: Iteration of Merging Components = {iteration} =============")
#     g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=2)
#     df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias) 
#     print(components)

# groups = ["".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("-") != -1]
# print(f">> Length = {len(groups)}, Groups={groups}")

# #%%
# """Level 3 """

# g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=3)
# iteration = 0
# while len(components) !=0:
#     iteration += 1
#     print(f">> ==========Level3: Iteration of Merging Components = {iteration} =============")
#     g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=2)
#     df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias) 
#     print(components)

# groups = ["".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("-") != -1]
# print(f">> Length = {len(groups)}, Groups={groups}")

# #%%
# """Level 4 """

# g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=4)
# iteration = 0
# while len(components) !=0:
#     iteration += 1
#     print(f">> ==========Level4: Iteration of Merging Components = {iteration} =============")
#     g,components,node2common_name = analyzeAlarmdata(df_alarms_new,num_sub_graphs=4,min_intersection_f=3, next_start_gap =60*2,  next_end_gap = 60*60*4, edge_drop_factor=2)
#     df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias) 
#     print(components)

# groups = ["".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("-") != -1]
# print(f">> Length = {len(groups)}, Groups={groups}")

#%%



# # Google PageRank Algo
# result = nx.algorithms.link_analysis.pagerank_alg.pagerank(main_g,weight="weight",max_iter=100000)
# result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
# print(">> Page Rank (Highest to Lowest) :",list(result.items())[:50])

# print("              --------------------------------------------------")

# # Eigenvector Centrality Algo
# result = nx.eigenvector_centrality(main_g, weight="weight")
# result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
# print(">> Eigenvector Centrality (Highest to Lowest) :",list(result.items())[:50])

#%%
# Hits Algo
# h, a = nx.hits(main_g)
# result = h
# result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
# print(">> Hub => Outgoing Edges: Based on out_degree (max to min):",list(result.items())[:50])

# print("              --------------------------------------------------")

# result = a
# result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
# print(">> Auth => Incoming Edges: Based on in_degree (max to min):",list(result.items())[:50])


#%%
# # Degree Analysis
# nodes_dict = {}
# for s in list(main_g.nodes):
#     nodes_dict[s] = {"count":main_g.nodes[s]["count"], "outd" : main_g.out_degree(s,"weight"), "ind":main_g.in_degree(s,"weight"), "totald": main_g.degree(s,"weight")}

# nodes_dict = {k:v for k, v in sorted(nodes_dict.items(), key=lambda arg: arg[1]["ind"], reverse=True) if v["count"]>4 and v["totald"]>1}
# print(nodes_dict)


# %%

# _ = plotAlarmsRelationsHeatMap(main_g,0)

#%%



# %%

# sim = nx.simrank_similarity(G=main_g.to_undirected(), source="S64",target="S26",importance_factor=1)
# print(sim)



# sim.keys()

# %%
 


 


#%%


# for n1 in main_g.nodes:
#     for n2 in main_g.nodes:
#         print(list(nx.common_neighbors(main_g.to_undirected(),n1,n2)))



# %%
