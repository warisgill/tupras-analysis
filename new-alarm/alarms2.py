import networkx as nx
from pyvis.network import Network

from alarms import *

def constructDependcyGraph(df_alarms,df_actions,alarms_nodes, action_nodes, filters): # case 3
    assert len(df_alarms["Month"].unique()) == len(df_actions["Month"].unique())
    print(">> # of Alarms:{} and Operator Actions:{}".format(df_alarms.shape[0], df_actions.shape[0]))
    
    g = nx.DiGraph() # Directed graph
    
    #---------- Adding Nodes----------------
    for s in alarms_nodes:
        g.add_node(s,size=10,count=0,color="Black") 
    
    for s in action_nodes:
        g.add_node("Operator->"+s,size=10,count=0,color="Orange")
    
    # ------------- Updating Counts & size of Nodes ------------------
    for s in df_alarms["SourceName"]:
        if g.has_node(s) == True:
            g.nodes[s]["size"] += 0.01
            g.nodes[s]["count"] += 1
            
    for s in df_actions["SourceName"]:
        if g.has_node("Operator->"+s) == True:
            g.nodes["Operator->"+s]["size"] += 0.01
            g.nodes["Operator->"+s]["count"] += 1 

    # sorting
    alarms_by_etime = [alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
    actions_by_time = [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
   
    #------------------ ADDING EDGES --------------------------
    for i in range(len(actions_by_time)):
         action = actions_by_time[i]
         for j in range(len(alarms_by_etime)):            
            alarm = alarms_by_etime[j]
            if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= alarm["EndTime"] and timedelta.total_seconds(alarm["EndTime"]-action["EventTime"])<filters["gap"]:
                if g.has_edge("Operator->"+action["SourceName"],alarm["SourceName"])==False:  
                    g.add_edge("Operator->"+action["SourceName"],alarm["SourceName"],weight=1)
                else:
                    g.edges["Operator->"+action["SourceName"],alarm["SourceName"]]["weight"] +=1
                    
    print(">> # of nodes = {}, # of edges = {}".format(g.number_of_nodes(),g.number_of_edges()))
    return g

def constructMultipleAlarmsActionsGraphs(df_alarms,df_actions,chunks,filters):
    graphs = []
    index_ranges = []
    batch_size = df_alarms.shape[0]//chunks
    alarm_nodes = df_alarms["SourceName"].unique()
    action_nodes = df_actions["SourceName"].unique()
    
    ## Range Indexes
    for start in range(0, df_alarms.shape[0], batch_size):    
        if start+batch_size <= df_alarms.shape[0]: 
            index_ranges.append((start,start+batch_size))
    index_ranges[-1] = (index_ranges[-1][0],index_ranges[-1][1]+ 1 +(df_alarms.shape[0]%chunks))
#     index_ranges.reverse()
#     print(">> Chunks = {}, Index Ranges = {}".format(chunks,index_ranges))
    
    for t in index_ranges:
        print("        ----------------------------------")        
        df1 = df_alarms.iloc[t[0]:t[1],:]
        df1.reset_index(drop=True, inplace=True)
        min_date = df1["StartTime"].min()
        max_date = df1["StartTime"].max()
        print(">> Index Range = {}, Min & Max dates = {}".format(t, (min_date.date(),max_date.date())))
        print(">> Filtering the Operator actions based on min-max dates ...")
        df2 = df_actions[(df_actions["EventTime"]>=min_date) & (df_actions["EventTime"]<=max_date)]
        df2.reset_index(drop=True, inplace=True)
        g = constructDependcyGraph(df1,df2,alarm_nodes,action_nodes,filters)
        graphs.append(g)
        
    print("        ----------------------------------") 
    return graphs

def intersection_of_graphs(graphs):
    assert len(graphs)>=1
    g = nx.intersection_all(graphs)
    remove_nodes = []
    for node in g.nodes:
        if g.degree[node] == 0:
            remove_nodes.append(node)
    g.remove_nodes_from(remove_nodes)
    
    return g




def updateFinalAlarmOpertorDependencyGraph(g, df_alarms, df_actions, filters):
    g = g.copy()
    for n in list(g.nodes):
        g.nodes[n]["size"] = 10
        g.nodes[n]["count"] = 0
    
    for e in list(g.edges):
        g.edges[e]["weight"] = 0
    
    # ------------- Updating Counts & size of Nodes ------------------
    for s in df_alarms["SourceName"]:
        if g.has_node(s) == True:
            g.nodes[s]["size"] += 0.01
            g.nodes[s]["count"] += 1
            
    for s in df_actions["SourceName"]:
        if g.has_node("Operator->"+s) == True:
            g.nodes["Operator->"+s]["size"] += 0.01
            g.nodes["Operator->"+s]["count"] += 1 
        
    # sorting
    alarms_by_etime = [alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
    actions_by_time = [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    
    #------------------ ADDING EDGES --------------------------
    for i in range(len(actions_by_time)):
         action = actions_by_time[i]
         if g.has_node("Operator->"+action["SourceName"]) == False:
            continue
         for j in range(len(alarms_by_etime)):            
            alarm = alarms_by_etime[j]
            if g.has_edge("Operator->"+action["SourceName"],alarm["SourceName"])==False:
                continue
            if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= alarm["EndTime"] and timedelta.total_seconds(alarm["EndTime"]-action["EventTime"])<filters["gap"]:
                if g.has_edge("Operator->"+action["SourceName"],alarm["SourceName"])==True:  
                    g.edges["Operator->"+action["SourceName"],alarm["SourceName"]]["weight"] +=1
    
    # Updating Titles
    for s in list(g.nodes):
        g.nodes[s]["title"] = "{}:{}".format(s,g.nodes[s]["count"])

    for e in list(g.edges):
        g.edges[e]["title"] = "{}:{}".format(e,g.edges[e]["weight"])
    
    remove_edges = []
    for edge in list(g.edges):
        g.edges[edge]["color"] = "Gray"
        if g.edges[edge]["weight"] <= filters["weight"]:
            remove_edges.append(edge)
    g.remove_edges_from(remove_edges)
    
    return g



