# import networkx as nx
# from functools import partial
# from alarms import *




               
            


  



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



