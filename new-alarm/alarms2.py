import networkx as nx
from pyvis.network import Network
from functools import partial
from alarms import *

# ================= Section 1: ================
def removeShortDurationAlarms(df, duration_filters=[20,120]):
    df_f1 = df[df["TimeDelta"]>duration_filters[0]] # keep the alarms whose duration is larger than the filter
    df_f2 = df[df["TimeDelta"]>duration_filters[1]]
    
    
    d = df["Month"].value_counts()
    mcounts = [t for t in sorted(d.items(), key= lambda arg: arg[0])]
    trace1 = go.Bar(name='Without filter',x=[t[0] for t in mcounts], y= [t[1] for t in mcounts ],  text=[t[1] for t in mcounts],  textposition='auto')

    
    d = df_f1["Month"].value_counts()
    mcounts = [t for t in sorted(d.items(), key= lambda arg: arg[0])]
    trace2 = go.Bar(name='{}s filter'.format(duration_filters[0]),x=[t[0] for t in mcounts], y= [t[1] for t in mcounts ],  text=[t[1] for t in mcounts],  textposition='auto')
    
    d = df_f2["Month"].value_counts()
    mcounts = [t for t in sorted(d.items(), key= lambda arg: arg[0])]
    trace3 = go.Bar(name="{}s filter".format(duration_filters[1]),x=[t[0] for t in mcounts], y= [t[1] for t in mcounts ],  text=[t[1] for t in mcounts],  textposition='auto')
        
    
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.update_layout(barmode='group', yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ))
    fig.show()
    

def storageAnalysis(df,duration_filters=[20,120]):
    df_f1 = df[df["TimeDelta"]>duration_filters[0]] # keep the alarms whose duration is larger than the filter
    df_f2 = df[df["TimeDelta"]>duration_filters[1]]
    
    x_axis = ["Raw Alarms", "After {}s filter".format(duration_filters[0]),"After {}s filter".format(duration_filters[1])]
    y_axis = [df.shape[0], df_f1.shape[0],df_f2.shape[0]]
    
    trace = go.Bar(x=x_axis, y=y_axis, text=y_axis,  textposition='auto')
        
    
    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_layout(yaxis=dict(
        title='Storage Utilization (KB)',
        titlefont_size=16,
        tickfont_size=14,
    ))
    fig.show()
               
            
# ================================= Section 2

def checkAction_OnAlarm(action,alarm,filters):
    if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= alarm["EndTime"] and timedelta.total_seconds(alarm["EndTime"]-action["EventTime"])<filters["gap"]:
        return True
    else:
        return False

def addOrUpdateEdge(g,e):
    if g.has_edge(*e)==True:  
        g.edges[e]["weight"] +=1
    else:
        g.add_edge(*e,weight=1)

    return False 

def addOrUpdateNode(g,color,operatorf,node):
    if operatorf:
        node = "Operator->"+ node
    if g.has_node(node) == True:
        g.nodes[node]["size"] += 0.01
        g.nodes[node]["count"] += 1
    else:
        g.add_node(node,size=10,count=0,color=color)

    return False 

def constructSingleAlarmsActionsGraph(df_alarms,df_actions,alarms_nodes, action_nodes, filters): # case 3
    g = nx.DiGraph() # Directed graph
    
    assert len(df_alarms["Month"].unique()) == len(df_actions["Month"].unique())
    print(">> # of Alarms:{} and Operator Actions:{}".format(df_alarms.shape[0], df_actions.shape[0]))
    
    #---------- Adding Nodes----------------
    _ = list(filter(partial(addOrUpdateNode,g,"Black",False), df_alarms["SourceName"]))         
    _ = list(filter(partial(addOrUpdateNode,g,"Orange",True),df_actions["SourceName"]))

    # converting rows to dicts
    alarms_by_etime =df_alarms.to_dict(orient="records") #[alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
    actions_by_time =df_actions.to_dict(orient="records") # [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    # -------- Adding Edges -----------------
    edges = [("Operator->"+action["SourceName"],alarm["SourceName"]) for action in actions_by_time for alarm in alarms_by_etime if checkAction_OnAlarm(action,alarm,filters)]
    _ = list(filter(partial(addOrUpdateEdge,g),edges)) 
    
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
        g = constructSingleAlarmsActionsGraph(df1,df2,alarm_nodes,action_nodes,filters)
        graphs.append(g)
        
    print("        ----------------------------------") 
    return graphs

def intersection_of_graphs(graphs,edge_filter=None):
    common_g = nx.DiGraph()
    
    if edge_filter == None:
        edge_filter = len(graphs)

    edges = [edge for g in graphs for edge in g.edges]
    edges_dict={edge:0 for edge in edges}
    
    for edge in edges:
        edges_dict[edge] += 1

    edges_dict = {k:v for k,v in edges_dict.items() if v>=edge_filter}

    common_g.add_edges_from([(k[0],k[1],{"weight":0,"color":"Gray"}) for k in  edges_dict.keys()])
    for g in graphs:
        for e in common_g.edges:
            if g.has_edge(*e) == True:
                common_g.edges[e]["weight"] += g.edges[e]["weight"]

    for n in common_g.nodes:
        common_g.nodes[n]["count"] = 0 #g.nodes[n]["count"]
        common_g.nodes[n]["size"] = 0 #g.nodes[n]["size"]

    for g in graphs:
        for n in common_g.nodes:
            if g.has_node(n) == True:
                common_g.nodes[n]["count"] += g.nodes[n]["count"]
                common_g.nodes[n]["size"] += g.nodes[n]["size"]
        
    return common_g

def getFinalOperatorAlarmRelationGraph(df_alarms,df_actions,chunks,min_intersectionf,monthsf,snamesf,durationf,gapf,weightf):   
    print(">> Finding relation between operator action and alarm")
    print(">> Number of sub graphs to be constructed :", chunks)
    print(">> Filter: Only Include these months: ", monthsf)
    print(">> Filter: Gap Filter for operator EventTime and alarm EndTime = {}s.".format(gapf))
    print(">> Filter: Edges to be removed whose weight <= {}".format(weightf))
    print(">> Filter: Ingore SourceNames ", snamesf)
    
    
    df_filtered_alarms = df_alarms[(df_alarms["TimeDelta"]>durationf) & (df_alarms["Month"].isin(monthsf)) & (~df_alarms["SourceName"].isin(snamesf))]
    df_filtered_actions = df_actions[df_actions["Month"].isin(monthsf)]
    df_filtered_alarms = df_filtered_alarms.sort_values('StartTime')
    df_filtered_actions =df_filtered_actions.sort_values('EventTime')

    graphs = constructMultipleAlarmsActionsGraphs(df_filtered_alarms, df_filtered_actions,chunks,{"gap":gapf})
    assert len(graphs)==chunks
    print(">> Taking intersection of {} sub-graphs".format(len(graphs)))
     
    main_g = intersection_of_graphs(graphs,min_intersectionf)
    temp_g = intersection_of_graphs(graphs,chunks)
    
    return main_g, temp_g
  
# =========== Section 3: Finding Most important allarms

def hasRelationPrevNextAlarms(prevd,nextd,gapf):
    if timedelta.total_seconds(nextd["StartTime"]-prevd["StartTime"]) > gapf: # if next alarm is not triggered within tfilter1 duration then break 
        return False

    """
        if an alarm is activating after the activation of a previous alarm and decativate when 
       the prev is deactivating or before the prev
    """
    if nextd["SourceName"] != prevd["SourceName"] and nextd["StartTime"] >= prevd["StartTime"]: 
        if nextd["EndTime"] <= prevd["EndTime"] or timedelta.total_seconds(nextd["EndTime"]-prevd["EndTime"]) < gapf:
            return True
    else:
        return False
    
def constructSingleAlarmsRelationGraph(df_alarms,gapf):
    g = nx.DiGraph()
    # Adding Nodes
    _ =  list(filter(partial(addOrUpdateNode(g,"Black",False)) ,df_alarms["SourceName"]))

    start_records = [v for v in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["StartTime"], reverse=False)]
    edges = [(start_records[i],start_records[j]) for i in range(len(start_records)) for j in range(i+1,len(start_records)) if hasRelationPrevNextAlarms(start_records[i],start_records[j],gapf)]

    _ = filter(partial(addOrUpdateEdge,g),edges)

    print(">> # of nodes = {}, # of edges = {}".format(g.number_of_nodes(),g.number_of_edges()))
    return g

def constructMultipleAlarmsRelationGraphs(df_alarms,chunks,gapf):
    graphs = []
    index_ranges = []
    batch_size = df_alarms.shape[0]//chunks    
    ## Range Indexes
    for start in range(0, df_alarms.shape[0], batch_size):    
        if start+batch_size <= df_alarms.shape[0]: 
            index_ranges.append((start,start+batch_size))
    index_ranges[-1] = (index_ranges[-1][0],index_ranges[-1][1]+ 1 +(df_alarms.shape[0]%chunks))
    
    for t in index_ranges:
        print("        ----------------------------------")        
        df1 = df_alarms.iloc[t[0]:t[1],:]
        df1.reset_index(drop=True, inplace=True)
        min_date = df1["StartTime"].min()
        max_date = df1["StartTime"].max()
        print(">> Index Range = {}, Min & Max dates = {}".format(t, (min_date.date(),max_date.date())))
        g = constructSingleAlarmsRelationGraph(df1,gapf) 
        graphs.append(g)
        
    print("        ----------------------------------") 

    return graphs

def getFinalAlarmRelationsGraph(df_alarms,chunks,min_intersectionf,monthsf,snamesf,durationf,gapf):   
    print(">> Starting to find relations between alarms....")
    print(">> Number of sub graphs to be constructed :", chunks)
    print(">> Filter: Only Include these months: ", monthsf)
    print(">> Filter: Gap Filter for prev to next alarm = {}s.".format(gapf))
    print(">> Filter: Ingore SourceNames ", snamesf)
    
    df_filtered_alarms = df_alarms[(df_alarms["TimeDelta"]>durationf) & (df_alarms["Month"].isin(monthsf)) & (~df_alarms["SourceName"].isin(snamesf))]
    df_filtered_alarms = df_filtered_alarms.sort_values('StartTime')
    

    graphs = constructMultipleAlarmsRelationGraphs(df_filtered_alarms,chunks,gapf)
    assert len(graphs)==chunks
    print(">> Taking intersection of {} sub-graphs".format(len(graphs)))
     
    main_g = intersection_of_graphs(graphs,min_intersectionf)
    temp_g = intersection_of_graphs(graphs,chunks)
    
    return main_g, temp_g


# =========== Section 3: Finding Most important allarms
    
def constructSingleAlarmsRelationGraph(df_alarms,gapf):
    g = nx.DiGraph()
    # Adding Nodes
    _ =  list(filter(partial(addOrUpdateNode,g,"Black",False), df_alarms["SourceName"]))

    start_records = [v for v in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["StartTime"], reverse=False)]
    edges = []
    # edges = next((start_records[i],start_records[j]) for i in range(len(start_records)) for j in range(i+1,len(start_records)) if hasRelationPrevNextAlarms(start_records[i],start_records[j],gapf))

    for i in range(len(start_records)):
        prevd = start_records[i]
        for j in range(i+1,len(start_records)):        
            nextd = start_records[j]
            if timedelta.total_seconds(nextd["StartTime"]-prevd["StartTime"]) > gapf: # if next alarm is not triggered within tfilter1 duration then break 
                break
            if nextd["SourceName"] != prevd["SourceName"] and nextd["StartTime"] >= prevd["StartTime"]: 
               if nextd["EndTime"] <= prevd["EndTime"] or timedelta.total_seconds(nextd["EndTime"]-prevd["EndTime"]) < gapf:
                    if g.has_edge(prevd["SourceName"],nextd["SourceName"]) == False:
                        g.add_edge(prevd["SourceName"],nextd["SourceName"],color="Red",weight=1)
                    else:
                        g.edges[prevd["SourceName"],nextd["SourceName"]]["weight"] +=1
               

    print(">> # of nodes = {}, # of edges = {}".format(g.number_of_nodes(),g.number_of_edges()))
    return g

def constructMultipleAlarmsRelationGraphs(df_alarms,chunks,gapf):
    graphs = []
    index_ranges = []
    batch_size = df_alarms.shape[0]//chunks    
    ## Range Indexes
    for start in range(0, df_alarms.shape[0], batch_size):    
        if start+batch_size <= df_alarms.shape[0]: 
            index_ranges.append((start,start+batch_size))
    index_ranges[-1] = (index_ranges[-1][0],index_ranges[-1][1]+ 1 +(df_alarms.shape[0]%chunks))
    
    for t in index_ranges:
        print("        ----------------------------------")        
        df1 = df_alarms.iloc[t[0]:t[1],:]
        df1.reset_index(drop=True, inplace=True)
        min_date = df1["StartTime"].min()
        max_date = df1["StartTime"].max()
        print(">> Index Range = {}, Min & Max dates = {}".format(t, (min_date.date(),max_date.date())))
        g = constructSingleAlarmsRelationGraph(df1,gapf) 
        graphs.append(g)
        
    print("        ----------------------------------") 

    return graphs

def getFinalAlarmRelationsGraph(df_alarms,chunks,min_intersectionf,monthsf,snamesf,durationf,gapf):   
    print(">> Starting to find relations between alarms....")
    print(">> Number of sub graphs to be constructed :", chunks)
    print(">> Filter: Only Include these months: ", monthsf)
    print(">> Filter: Gap Filter for prev to next alarm = {}s.".format(gapf))
    print(">> Filter: Ingore SourceNames ", snamesf)
    
    df_filtered_alarms = df_alarms[(df_alarms["TimeDelta"]>durationf) & (df_alarms["Month"].isin(monthsf)) & (~df_alarms["SourceName"].isin(snamesf))]
    df_filtered_alarms = df_filtered_alarms.sort_values('StartTime')
    

    graphs = constructMultipleAlarmsRelationGraphs(df_filtered_alarms,chunks,gapf)
    assert len(graphs)==chunks
    print(">> Taking intersection of {} sub-graphs".format(len(graphs)))
     
    main_g = intersection_of_graphs(graphs,min_intersectionf)
    temp_g = intersection_of_graphs(graphs,chunks)
    
    return main_g, temp_g


# def case4FindImportantSensorNodes(df_alarms, common_sources, tfilter1 = 60):
#     print(">> Alarms:{} ".format(df_alarms.shape[0]))
#     G = nx.DiGraph() # Directed Graph

#     # # Adding Nodes
#     for s in df_alarms["SourceName"]:
#         if G.has_node(s) == True:
#             G.nodes[s]["size"] += 0.1
#             G.nodes[s]["count"] += 1
#         else:
#             G.add_node(s,size=5,count =1)
    
#     for s in list(G.nodes):
#         G.nodes[s]["title"] = "{}:{}".format(s,G.nodes[s]["count"])

#     start_records = [v for v in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["StartTime"], reverse=False)]
#     for i in range(len(start_records)):
#         prevd = start_records[i]
#         for j in range(i+1,len(start_records)):        
#             nextd = start_records[j]
#             if timedelta.total_seconds(nextd["StartTime"]-prevd["StartTime"]) > tfilter1: # if next alarm is not triggered within tfilter1 duration then break 
#                 break 
#             if nextd["SourceName"] != prevd["SourceName"] and nextd["StartTime"] >= prevd["StartTime"]: 
#                if nextd["EndTime"] <= prevd["EndTime"] or timedelta.total_seconds(nextd["EndTime"]-prevd["EndTime"]) < tfilter1:
#                     if G.has_edge(prevd["SourceName"],nextd["SourceName"]) == False:
#                         G.add_edge(prevd["SourceName"],nextd["SourceName"],color="Red",weight=1)
#                     else:
#                         G.edges[prevd["SourceName"],nextd["SourceName"]]["weight"] +=1


#     # remove_edges = []

#     for edge in list(G.edges):
#         G.edges[edge]["title"] = "{}".format(G.edges[edge]["weight"])
#         # G.edges[edge]["weight"] = 1
#         # if G.edges[edge]["weight"] <=2:
#         #     remove_edges.append(edge)

#     # G.remove_edges_from(remove_edges)
    
#     G.remove_nodes_from(list(nx.isolates(G)))
#     print(">> {}".format(nx.classes.function.info(G)))
#     return G 


# def _intersection_of_graphs(graphs):
#     assert len(graphs)>=1
#     g = nx.intersection_all(graphs)
#     remove_nodes = []
#     for node in g.nodes:
#         if g.degree[node] == 0:
#             remove_nodes.append(node)
#     g.remove_nodes_from(remove_nodes)
    
#     return g


# def updateFinalAlarmOpertorDependencyGraph(g, df_alarms, df_actions, filters):
#     g = g.copy()
#     for n in list(g.nodes):
#         g.nodes[n]["size"] = 10
#         g.nodes[n]["count"] = 0
    
#     for e in list(g.edges):
#         g.edges[e]["weight"] = 0
    
#     # ------------- Updating Counts & size of Nodes ------------------
#     for s in df_alarms["SourceName"]:
#         if g.has_node(s) == True:
#             g.nodes[s]["size"] += 0.01
#             g.nodes[s]["count"] += 1
            
#     for s in df_actions["SourceName"]:
#         if g.has_node("Operator->"+s) == True:
#             g.nodes["Operator->"+s]["size"] += 0.01
#             g.nodes["Operator->"+s]["count"] += 1 
    
#     nodes_t = [node.replace("Operator->","") for node in list(g.nodes)]

#     df_actions = df_actions[df_actions["SourceName"].isin(nodes_t)]
#     df_alarms = df_alarms[df_alarms["SourceName"].isin(nodes_t)] 

        
#     # sorting
#     alarms_by_etime = [alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
#     actions_by_time = [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    
#     #------------------ ADDING EDGES --------------------------
#     for i in range(len(actions_by_time)):
#          action = actions_by_time[i]
#          if g.has_node("Operator->"+action["SourceName"]) == False:
#             continue
#          for j in range(len(alarms_by_etime)):            
#             alarm = alarms_by_etime[j]
#             if g.has_edge("Operator->"+action["SourceName"],alarm["SourceName"])==False:
#                 continue
#             if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= alarm["EndTime"] and timedelta.total_seconds(alarm["EndTime"]-action["EventTime"])<filters["gap"]:
#                 if g.has_edge("Operator->"+action["SourceName"],alarm["SourceName"])==True:  
#                     g.edges["Operator->"+action["SourceName"],alarm["SourceName"]]["weight"] +=1
    
#     # Updating Titles
#     for s in list(g.nodes):
#         g.nodes[s]["title"] = "{}:{}".format(s,g.nodes[s]["count"])

#     for e in list(g.edges):
#         g.edges[e]["title"] = "{}:{}".format(e,g.edges[e]["weight"])
    
#     remove_edges = []
#     for edge in list(g.edges):
#         g.edges[edge]["color"] = "Gray"
#         if g.edges[edge]["weight"] <= filters["weight"]:
#             remove_edges.append(edge)
#     g.remove_edges_from(remove_edges)
    
#     return g


# def _constructSingleAlarmsActionsGraph(df_alarms,df_actions,alarms_nodes, action_nodes, filters): # case 3
#     assert len(df_alarms["Month"].unique()) == len(df_actions["Month"].unique())
#     print(">> # of Alarms:{} and Operator Actions:{}".format(df_alarms.shape[0], df_actions.shape[0]))
    
#     g = nx.DiGraph() # Directed graph
    
#     #---------- Adding Nodes----------------
#     # for s in alarms_nodes:
#     #     g.add_node(s,size=10,count=0,color="Black")

#     _ = list(filter(partial(addOrUpdateNode,g,"Black",False), df_alarms["SourceName"]))     
    
#     _ = list(filter(partial(addOrUpdateNode,g,"Orange",True),df_actions["SourceName"]))

#     # for s in action_nodes:
#     #     g.add_node("Operator->"+s,size=10,count=0,color="Orange")
    
#     # # ------------- Updating Counts & size of Nodes ------------------
#     # for s in df_alarms["SourceName"]:
#     #     if g.has_node(s) == True:
#     #         g.nodes[s]["size"] += 0.01
#     #         g.nodes[s]["count"] += 1


            
#     # for s in df_actions["SourceName"]:
#     #     if g.has_node("Operator->"+s) == True:
#     #         g.nodes["Operator->"+s]["size"] += 0.01
#     #         g.nodes["Operator->"+s]["count"] += 1 

#     # sorting
#     alarms_by_etime = [alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
#     actions_by_time = [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
   
#     #------------------ ADDING EDGES --------------------------
#     # for i in range(len(actions_by_time)):
#     #      action = actions_by_time[i]
#     #      for j in range(len(alarms_by_etime)):            
#     #         alarm = alarms_by_etime[j]
#     #         if checkAction_OnAlarm(action,alarm,filters):
#     #             if g.has_edge("Operator->"+action["SourceName"],alarm["SourceName"])==False:  
#     #                 g.add_edge("Operator->"+action["SourceName"],alarm["SourceName"],weight=1)
#     #             else:
#     #                 g.edges["Operator->"+action["SourceName"],alarm["SourceName"]]["weight"] +=1

#     # version 2
#     edges = [("Operator->"+action["SourceName"],alarm["SourceName"]) for action in actions_by_time for alarm in alarms_by_etime if checkAction_OnAlarm(action,alarm,filters)]
#     result = list(filter(partial(addOrUpdateEdge,g),edges)) 
#     print(result) # should be empty list

#     print(">> # of nodes = {}, # of edges = {}".format(g.number_of_nodes(),g.number_of_edges()))
#     return g



