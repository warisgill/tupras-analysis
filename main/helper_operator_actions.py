import networkx as nx
from functools import partial
import plotly.graph_objects as go

def preProcessAlarmData(df,months=None, sources_filter = [], monmentarly_filter=20, staling_filter=(60*60)*24):
    print(f">>Preprocessing... \n   Months to include={months}\n  Ignore Sources={sources_filter}\n  Ingnore Momentarlily Alarms Filter={monmentarly_filter}seconds \n   Ignoreing Staling Alarms Filter={staling_filter/3600.0} hours")
    if months is None:
        months = df["Month"].unique()
    
    df_new = df[(df["TimeDelta"]>monmentarly_filter) & (df["TimeDelta"]<staling_filter) & (df["Month"].isin(months)) & (~df["SourceName"].isin(sources_filter))]

    # assert df_new[(df_new["TimeDelta"]>monmentarly_filter)] 
    return df_new


# ================================= Section 2

# ignoring filter -> and timedelta.total_seconds(alarm["EndTime"]-action["EventTime"])<filters["gap"]
def checkAction_OnAlarm(action,alarm):
    if action["EventTime"] > alarm["StartTime"] and action["EventTime"] <= alarm["EndTime"] :
        return True
    else:
        return False

def addOrUpdateEdge(g,e):
    if g.has_edge(*e)==True:  
        g.edges[e]["weight"] +=1
    else:
        g.add_edge(*e,weight=1)

    return False 

def addOrUpdateNode(g,operatorf,node): # using partial thats why node as last arg
    if operatorf==True:
        node = "Operator->"+ node
    if g.has_node(node) == True:
        g.nodes[node]["size"] += 0.01
        g.nodes[node]["count"] += 1
    else:
        g.add_node(node,size=10,count=0)

    return False 

def constructSingleAlarmsActionsGraph(df_alarms,df_actions): # case 3
    g = nx.DiGraph() # Directed graph
    
    assert len(df_alarms["Month"].unique()) == len(df_actions["Month"].unique())
    print(">> # of Alarms:{} and Operator Actions:{}".format(df_alarms.shape[0], df_actions.shape[0]))
    
    #---------- Adding Nodes----------------
    _ = list(filter(partial(addOrUpdateNode,g,False), df_alarms["SourceName"]))         
    _ = list(filter(partial(addOrUpdateNode,g,True),df_actions["SourceName"]))

    # converting rows to dicts
    alarms = df_alarms.to_dict(orient="records") #[alarm for alarm in sorted(df_alarms.to_dict(orient="records"), key=lambda arg: arg["EndTime"], reverse=False)]
    actions = df_actions.to_dict(orient="records") # [action for action in sorted(df_actions.to_dict(orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    # -------- Adding Edges -----------------
    edges = [("Operator->"+action["SourceName"],alarm["SourceName"]) for action in actions for alarm in alarms if checkAction_OnAlarm(action,alarm)]

    _ = list(filter(partial(addOrUpdateEdge,g),edges)) 
    
    print(">> # of nodes = {}, # of edges = {}".format(g.number_of_nodes(),g.number_of_edges()))
    return g

def constructMultipleAlarmsActionsGraphs(df_alarms,df_actions,chunks):
    graphs = []
    index_ranges = []
    batch_size = df_alarms.shape[0]//chunks
    # alarm_nodes = df_alarms["SourceName"].unique()
    # action_nodes = df_actions["SourceName"].unique()
    
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
        # df1.reset_index(drop=True, inplace=True)
        
        min_date = df1["StartTime"].min()
        max_date = df1["EndTime"].max()
        print(f">> Index Range = {t}, Min Date ={min_date.date()} & Max date = {max_date.date()}")
        print(">> Filtering the Operator actions based on min-max dates ...")
        
        df2 = df_actions[(df_actions["EventTime"]>=min_date) & (df_actions["EventTime"]<=max_date)]
        # df2.reset_index(drop=True, inplace=True)

        g = constructSingleAlarmsActionsGraph(df1,df2)
        graphs.append(g)
        
    print("        ----------------------------------") 
    return graphs

def intersection_of_graphs(graphs,edge_filter=None):
    intersect_G = nx.DiGraph()
    
    if edge_filter is None:
        edge_filter = len(graphs) # very important conditions

    assert edge_filter <= len(graphs)
    assert edge_filter > 1 
    
    print(f">> Ingore the Edge if its not appearing in atleast {edge_filter}  graphs")

    edges = [edge for g in graphs for edge in g.edges]
    edges_dict={edge:0 for edge in edges}
    
    for edge in edges:
        edges_dict[edge] += 1

    edges_dict = {k:v for k,v in edges_dict.items() if v>=edge_filter} # rmoving edeges which are not appearing in atleast "edge_filter" times.

    intersect_G.add_edges_from([(k[0],k[1],{"weight":0}) for k in  edges_dict.keys()])
    
    for g in graphs:
        for e in intersect_G.edges:
            if g.has_edge(*e) == True:
                intersect_G.edges[e]["weight"] += g.edges[e]["weight"]

    for n in intersect_G.nodes:
        intersect_G.nodes[n]["count"] = 0 #g.nodes[n]["count"]
        intersect_G.nodes[n]["size"] = 0 #g.nodes[n]["size"]

    for g in graphs:
        for n in intersect_G.nodes:
            if g.has_node(n) == True:
                intersect_G.nodes[n]["count"] += g.nodes[n]["count"]
                intersect_G.nodes[n]["size"] += g.nodes[n]["size"]
        
    return intersect_G

def getFinalOperatorAlarmRelationGraph(df_alarms,df_actions,num_graphs, min_graphs_intersection_filter):   
    print(">> Finding relation between operator action and alarm")
    print(">> Number of sub graphs to be constructed :", num_graphs)
     
    df_filtered_alarms = df_alarms #df_alarms[(df_alarms["TimeDelta"]>durationf)& (df_alarms["TimeDelta"]<stalingf) & (df_alarms["Month"].isin(monthsf)) & (~df_alarms["SourceName"].isin(snamesf))]
    df_filtered_actions = df_actions #df_actions[df_actions["Month"].isin(monthsf)]
    df_filtered_alarms = df_filtered_alarms.sort_values('StartTime')
    df_filtered_actions =df_filtered_actions.sort_values('EventTime')

    graphs = constructMultipleAlarmsActionsGraphs(df_filtered_alarms, df_filtered_actions,num_graphs)
    assert len(graphs)==num_graphs
    print(">> Taking intersection of {} sub-graphs".format(len(graphs)))
     
    main_g = intersection_of_graphs(graphs,min_graphs_intersection_filter)
     
    return main_g

def plotOperatorAlarmRelationHeatMap(g, filter_weight):
      
    remove_edges = []
    for op, al, weight in g.edges.data("weight"):
        if weight <=filter_weight:
            remove_edges.append((op,al))
    
    # print(">> Edges Being Removed: ", remove_edges)
    g.remove_edges_from(remove_edges)
    print(f">> Nodes rmoved = {list(nx.isolates(g))} based on edge weighter filter = {filter_weight}")
    g.remove_nodes_from(list(nx.isolates(g)))

    int2operator = dict(enumerate([action for action in g.nodes if action.find("Operator")!=-1]))
    int2alarm = dict(enumerate([alarm for alarm in g.nodes if alarm.find("Operator")==-1]))

    alarm2int = {v:k for k,v in int2alarm.items()}
    operator2int = {v:k for k,v in int2operator.items()}

    # data2 = np.zeros((len(operator2int),len(alarm2int)))
    data = [[None for i in range(len(alarm2int))] for j in range(len(operator2int))]
    print(">> Dimension",len(data[0]), len(data))
#     print(data)

    print(f"After Removol: Alarm Tags = {len(alarm2int)} \n Operator Tags ={len(operator2int)} ")
    
    for op, al, weight in g.edges.data("weight"):
        # print(op,al,weight)
        data[operator2int[op]][alarm2int[al]] = weight
        # data2[operator2int[op],alarm2int[al]] = weight
    


    fig = go.Figure(data=go.Heatmap(z=data,y = [int2operator[v] for v in int2operator.keys()],x = [int2alarm[v] for v in int2alarm.keys()],hoverongaps = False,colorscale='Viridis')) # Greys
    
    fig.update_layout(width=1000,height=1000,xaxis_nticks =150,yaxis_nticks =150)
    fig.update_xaxes(side="top")
    fig.show()
    return data