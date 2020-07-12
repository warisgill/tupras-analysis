# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Next Paper 

# %%
from alarms2 import *
import pandas as pd
from pyvis.network import Network

# %% [markdown]
# # Section 0: Setting Default Variables & Loading Alarm and Operator Data

# %%
filter_short_alarms = [20, 120]  #seconds
staling_alarms_filter = (60*60) * 12
path = "./data/new/"


# %%
alarms_fname = "formatted-all-month-alarms.csv" 
operator_fname = "operator-all-month-actions.csv"

df_main_alarms = pd.read_csv(path + alarms_fname, low_memory=False ,parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(lambda arg: timedelta.total_seconds(arg)) 
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(lambda arg: arg.month)

df_main_actions = pd.read_csv(path + operator_fname, low_memory=False ,parse_dates=["EventTime"])
df_main_actions["Month"] = df_main_actions["EventTime"].apply(lambda arg: arg.month)


# %%
df_main_actions.info(), df_main_alarms.info()

# %% [markdown]
# # Section 1: Removing Chattering from Alarms Data
# 
# ### use case=> Bandwidht and Storage Reduction
# 
# Retention period.
# 
# Suppose that on average each alarm takes roughly 1KB of sotorage space. 
# 
# 

# %%
# removeShortDurationAlarms(df_main_alarms)
# storageAnalysis(df_main_alarms)

# %% [markdown]
# # Section 2: Relating Operator Action with Alarm Data
# 
# How is this use case important?

# %%
months_f = df_main_alarms["Month"].unique()
chunks = 16
min_intersection_f = 14
gapf = (60 * 60)* 4 
weightf = 2 # to remove edges in the final graph
snames_f = ["47TI1713"]

main_g,tempg = getFinalOperatorAlarmRelationGraph(df_main_alarms,df_main_actions,chunks,min_intersection_f,months_f,snames_f,filter_short_alarms[1],gapf,weightf)
print(">> Done")


# %%
# assert mg1.number_of_edges() == mg2.number_of_edges()
print(">> # of Edges Main graph1 {} and in graph2 {}".format( main_g.number_of_edges() , main_g.number_of_edges()))

operator_nodes = [action for action in main_g.nodes if action.find("Operator")!=-1]

print(f">> Total number of operator nodes in the graph={len(operator_nodes)}")
print (f">> Total number of Alarm Tags in the graph = {len(main_g.nodes)- len(operator_nodes)}")

node2num_neigbs = {n: len(list(main_g.neighbors(n))) for n in operator_nodes}
t3 = [(source,main_g.nodes[source]["count"],num_neighbs) for source,num_neighbs in node2num_neigbs.items()]
# print(*[">>{},{},# of neighbours {}\n".format(*tup) for tup in sorted(t3,key=lambda arg: arg[1], reverse=True)])

# Verify the results based count with orignal csv data


# %%
# print(list(mg1.nodes))
import numpy as np
from scipy.special import softmax

def operatorAlarmRelationHisto(g, filter_weght):

      
    # remove_edges = []
    # for op, al, weight in g.edges.data("weight"):
    #     if weight <=filter_weght:
    #         remove_edges.append((op,al))
    
    # print(">> Edges Being Removed: ", remove_edges)
    # g.remove_edges_from(remove_edges)
    # g.remove_nodes_from(list(nx.isolates(g)))

    int2operator = dict(enumerate([action for action in g.nodes if action.find("Operator")!=-1]))
    int2alarm = dict(enumerate([alarm for alarm in g.nodes if alarm.find("Operator")==-1]))

    alarm2int = {v:k for k,v in int2alarm.items()}
    operator2int = {v:k for k,v in int2operator.items()}

    
    data2 = np.zeros((len(operator2int),len(alarm2int)))
    data = [[None for i in range(len(alarm2int))] for j in range(len(operator2int))]
    print(">> Dimension",len(data[0]), len(data))
#     print(data)

    print(f"After Removol: Alarm Tags = {len(alarm2int)} \n Operator Tags ={len(operator2int)} ")
    
    for op, al, weight in g.edges.data("weight"):
        print(op,al,weight)
        data[operator2int[op]][alarm2int[al]] = weight
        data2[operator2int[op],alarm2int[al]] = weight


    print(data2[1,:])
    
    data2 = softmax(data2, axis=1)
    print(data2[1,:])

    # for i in range(data2.shape[0]):
    #     for j in range(data2.shape[1]):
            # if data2[i,j] > 0:
            # data[i][j] =data2[i,j]

    # data = softmax(data, axis=1)
    # print(data)
    
#     print(data)
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(z=data,y = [int2operator[v] for v in int2operator.keys()],x = [int2alarm[v] for v in int2alarm.keys()],hoverongaps = False,colorscale='Greys'))
    
    fig.update_layout(width=1000,height=1000,xaxis_nticks =150,yaxis_nticks =150)
    fig.update_xaxes(side="top")
    fig.show()

tg = nx.Graph(main_g)

operatorAlarmRelationHisto(tg.to_directed(),200)
    


# %%
print(">> Building Visualization .....")
# print(g.number_of_nodes())
nt = Network("400x", "100%", notebook=True)
nt.from_nx(mg1)
nt.options.edges.Color  = "Gray"
nt.toggle_hide_edges_on_drag(True)
# nt.toggle_physics(False)
nt.repulsion()
nt.toggle_stabilization(True)
nt.show_buttons(filter_=['physics','edges'])
# nt.toggle_ph
nt.show("nt.html")

# %% [markdown]
# # Section 3: Assigning Dynamic Proiorties based on inverted Pagerank

# %%
months_f = df_main_alarms["Month"].unique()
chunks = 20
min_intersection_f = 4
gapf = 60*2 
snames_f = ["47TI1713"]

mg1,mg2 = getFinalAlarmRelationsGraph(df_main_alarms,chunks,min_intersection_f,months_f,snames_f,filter_short_alarms[1],gapf)

print(">> # of Edges Main graph1 {} and in graph2 {}".format( mg1.number_of_edges() , mg2.number_of_edges()))
print(">> Done")


# %%
# Google PageRank Algo
result = nx.algorithms.link_analysis.pagerank_alg.pagerank(mg1,weight="weight",max_iter=100000)
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Page Rank (Highest to Lowest) :",list(result.items())[:50])

print("              --------------------------------------------------")

# Eigenvector Centrality Algo
result = nx.eigenvector_centrality(mg1, weight="weight")
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Eigenvector Centrality (Highest to Lowest) :",list(result.items())[:50])


# %%
# Hits Algo
h, a = nx.hits(mg1)
result = h
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Hub => Outgoing Edges: Based on out_degree (max to min):",list(result.items())[:50])

print("              --------------------------------------------------")

result = a
result = {k:float(format(v, '.4f')) for k,v in sorted(result.items(), key=lambda arg: arg[1], reverse=True)}
print(">> Auth => Incoming Edges: Based on in_degree (max to min):",list(result.items())[:50])


# %%
# Degree Analysis
nodes_dict = {}
for s in list(mg1.nodes):
    nodes_dict[s] = {"count":mg1.nodes[s]["count"], "outd" : mg1.out_degree(s,"weight"), "ind":mg1.in_degree(s,"weight"), "totald": mg1.degree(s,"weight")}

nodes_dict = {k:v for k, v in sorted(nodes_dict.items(), key=lambda arg: arg[1]["ind"], reverse=True) if v["count"]>4 and v["totald"]>1}
print(nodes_dict)


# %%
# for edge in mg1.edges:
#     mg1.edges[edge]["value"] = mg1.edges[edge]["weight"]
#     mg1.edges[edge]["title"] = "{},{}".format(edge,mg1.edges[edge]["weight"])

# for node in mg1.nodes:
#     mg1.nodes[node]["title"] = "{}, {}".format(node,mg1.nodes[node]['count']) 


# nt = Network("400x", "100%", notebook=True)
# nt.from_nx(mg1)
# nt.options.edges.Color  = "Gray"
# nt.toggle_hide_edges_on_drag(True)
# # nt.toggle_physics(False)
# nt.repulsion()
# nt.toggle_stabilization(True)
# nt.show_buttons(filter_=['physics','edges'])
# # nt.toggle_ph
# nt.show("nt.html")
# # visualize_graph(mg1)

# %% [markdown]
# # Extras
# 
# 
# ### Per Sensor Based Analysis

# %%



