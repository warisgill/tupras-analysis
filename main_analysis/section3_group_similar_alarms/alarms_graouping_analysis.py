# %%

import time
from networkx.algorithms import components
from networkx.algorithms.centrality import group
import pandas as pd
import networkx as nx
from pyvis.network import Network
from datetime import timedelta
import plotly.express as px

from section3_group_similar_alarms.helper_grouping import analyzeAlarmdata, getTimeDelta, concatenateSourceNameAndCondition, plotSourceAndCondtionHistogram, preProcessAlarmData
from section2_operator_actions.helper_operator_actions import getTrueAndNuisanceSourceNames


# %%
PATH = "/home/waris/Github/tupras-analysis/data/"
path_alarms = PATH + "processed/alarms/"
path_op_actions = PATH + "processed/operator-actions/"

start = time.time()
alarms_fname = "formatted-all-month-alarms.csv"
operator_fname = "operator-all-month-actions.csv"

df_main_alarms = pd.read_csv(path_alarms + alarms_fname,
                             low_memory=False, parse_dates=["StartTime", "EndTime"])

df_main_alarms["TimeDelta"] = df_main_alarms[["StartTime","EndTime"]].apply(lambda arg: getTimeDelta(*arg),axis=1)
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(lambda arg: arg.month)
df_main_alarms["SourceName"] = df_main_alarms[["SourceName", "Condition"]].apply(lambda arg: concatenateSourceNameAndCondition(*arg),axis =1)

df_main_actions = pd.read_csv(
    path_op_actions + operator_fname, low_memory=False, parse_dates=["EventTime"])
df_main_actions["Month"] = df_main_actions["EventTime"].apply(
    lambda arg: arg.month)

print("End time ", time.time()-start)
df_main_alarms


#%%

"""
    Chaning name 2 alias
"""

alias2name = {f"A{k}": v for k, v in enumerate(df_main_alarms["SourceName"].unique())}
name2alias = {v: k for k, v in alias2name.items()}
df_main_alarms["SourceName"] = df_main_alarms["SourceName"].apply(lambda sname: name2alias[sname])


# %%
"""
    Preparing thed the Data
    1. Ignore Comm alarms.
    2. Ingore Fleeting Alarms
    3. Remove sources which are triggered less than 20 times.
    4. Remove Staling Alarms
    5.  
"""
snames_f = []
months_f = df_main_alarms["Month"].unique()
fleeting_alarms_filter = 20 # secods
staling_alarms_filter = 60*60*12 # seconds

df_alarms_new1 = preProcessAlarmData(df_main_alarms, months=months_f, sources_filter=snames_f,
                                     monmentarly_filter=fleeting_alarms_filter, staling_filter=staling_alarms_filter, ingore_communication_alarms=False, min_alarms_per_source=20)

df_alarms_new2 = preProcessAlarmData(df_main_alarms, months=months_f, sources_filter=snames_f,
                                     monmentarly_filter=fleeting_alarms_filter, staling_filter=staling_alarms_filter, ingore_communication_alarms=True, min_alarms_per_source=20)



plotSourceAndCondtionHistogram(df_alarms_new1) # with comm alarms
plotSourceAndCondtionHistogram(df_alarms_new2) # without comm alarms
df_alarms_new = df_alarms_new2  # analyzing alarms without communicationa alarms
source2count = dict(df_alarms_new["SourceName"].value_counts())

#%% 
"""
    Only grouping nuisance alarms
"""
true_sources, nuisance_sources= getTrueAndNuisanceSourceNames(df_alarms=df_alarms_new,df_operator=df_main_actions)

print(">> Before Filtering of True Alarms", df_alarms_new.shape[0])

df_alarms_new = df_alarms_new[df_alarms_new["SourceName"].isin(nuisance_sources)] # only analyze the nusiance alrms

print(">> After Filtering of True Alarms", df_alarms_new.shape[0])

# %%
"""Grouping of Source Names"""
max_edge_drop_factor = 1.3 # factor x weight < count ignore such edge 
iteration = 0
edge_drop_factor = 1.0 # start from one to one relation and go to max edge drop factor
while edge_drop_factor < max_edge_drop_factor:
    g = None
    while True:
        iteration += 1
        print(
            f">> ==========Level=1 Iteration of Merging Components = {iteration} =============")
        g, components, node2common_name = analyzeAlarmdata(
            df_alarms_new, num_sub_graphs=4, min_intersection_f=3, next_start_gap=60*2,  next_end_gap=60*20, edge_drop_factor=edge_drop_factor)
        # if g is not None:

        df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(
            lambda alias: node2common_name[alias] if alias in node2common_name.keys() else alias)
        print(components)
        if len(components) == 0:
            break
    edge_drop_factor += 0.1

    groups = [
        "".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("-") != -1]
    print(
        f">>Edge drop factor= {edge_drop_factor} and final Groups:  Length = {len(groups)}, Groups={groups}")


plotSourceAndCondtionHistogram(df_alarms_new) # to visualize which sourcenames are grouped

#%%
""" 
    How many alarms will be reduced if we do grouping? 
"""
main_sources =[name for name in df_alarms_new["SourceName"].unique() if name.find("-") ==-1]

groups =  [name for name in df_alarms_new["SourceName"].unique() if name.find("-") !=-1]
print(">> Groups",groups)
groups_count = [[(sname,source2count[sname]) for sname in group.split("-")] for group in groups]  
print(">> Groups heads",groups_count)
groups_heads = [max(l, key=lambda arg: arg[1]) for l in groups_count]
print(">> Final Group heads",groups_heads)

heads_snames = [t[0] for t in groups_heads]
print(">> heads snames",heads_snames)

total_alarms = sum([v for k,v in source2count.items()])
useless_snames = [(t[0],t[1]) for group_count in groups_count for t in group_count if t[0] not in heads_snames]
print(">> Useless",useless_snames)

num_useless_alarms = sum([t[1] for t in useless_snames])

x_axis = ["Raw Nuisance Alarms", "Nuisance Alarms with Grouping"]
y_axis = [total_alarms, total_alarms-num_useless_alarms]

fig =px.bar(x=x_axis, y=y_axis)
fig.update_layout(yaxis=dict(title="# of alarms",
    titlefont_size=16,
    tickfont_size=14,
), xaxis=dict(
    title='',
    titlefont_size=16,
    tickfont_size=14,
))
fig.show()

    

# %%
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



# %%
