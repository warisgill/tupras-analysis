# %%
from helpers.group_alarms import analyzeAlarmdata
from helpers.ploting import plotBargraph, plotSourceAndCondtionHistogram
from helpers.alarms import filterAlarmData, loadAlarmsData, loadOperatorData
from helpers import operator_actions as operator
import time


#%%

"""  Lodading the Data and Preprocessing """
PATH = "/home/waris/Github/tupras-analysis/data/"
path_alarms = PATH + "processed/alarms/"
path_op_actions = PATH + "processed/operator-actions/"

start = time.time()
alarms_fname = "formatted-all-month-alarms.csv"
operator_fname = "operator-all-month-actions.csv"

df_main_alarms =loadAlarmsData(path=path_alarms,filename=alarms_fname,preprocess=True)
df_main_actions = loadOperatorData(path=path_op_actions,filename=operator_fname,preprocess=True)

""" Common Sources in all months. Try it but can be skipped. """
# df_main_alarms = getDFWithCommonSourcesInAllMonths(df_main_alarms)

""" Chaning name 2 alias for alarm data but skipping it """
# source2Alias, alias2source = convertSourceNamesToAlias(df_main_alarms)

print("Total Time to load the data ", time.time()-start)
df_main_alarms

# %%
""" 
    Filter the Alarm Data
    1. Ignore the communication Alarms
    2. Ignore the momentary alarms => 20 seconds
    3. Remove Staling Alarms => 12 hours    
    4. Remove sources which are triggered less 20 in whole dataset
    5. Include all the months
    6. DO SKIP ANY SOURCENAME IF IGNORING COMMUNICATION ALARMS
"""
ignore_comm_alarms = True
momentary_alarms_f = 20  # seconds
staling_alarm_f = (60*60) * 12 # hours
min_alarms_per_source_f = 20 # any source which is not triggered atleast 20 times in whole dataset will be removed
months_f = df_main_alarms["Month"].unique()
snames_f = [] # ONLY USE IF NOT IGNORING COMM ALRMS

df_alarms_new = filterAlarmData(df_main_alarms, months=months_f, sources_filter=snames_f,
                                     monmentarly_filter=momentary_alarms_f, staling_filter=staling_alarm_f, ingore_communication_alarms=ignore_comm_alarms, min_alarms_per_source=min_alarms_per_source_f)


plotSourceAndCondtionHistogram(df_alarms_new) # with comm alarms


#%% 
""" 
    Fisrt Find the True  and Nuisance sources with following conditions:
    For Operator Graphs
    1. Construct 8 graphs
    2. Min intersections is 8/2 + 1
    3. Remove edges which are not significantaly contributing to nodes count
        It will vary in op actions and in grouping. In op actions
        it will be higher but in groupign it will be lower.
"""

true_sources, nuisance_sources= operator.getTrueAndNuisanceSourceNames(df_alarms=df_alarms_new,df_operator=df_main_actions,num_sub_graphs=4,min_intersection_f=4//2+1,edge_filter=4)

print(f">> True Sources ({len(true_sources)}) = {true_sources} \n\n>> Nuisance Sources ({len(nuisance_sources)}) = {nuisance_sources}")


# %%
print(">> Before Raw Alarms", df_alarms_new.shape[0])
df_alarms_new = df_alarms_new[df_alarms_new["SourceName"].isin(nuisance_sources)] # only analyze the nusiance alrms
print(">> After: Only Nusiance alarms", df_alarms_new.shape[0])
source2count = dict(df_alarms_new["SourceName"].value_counts()) # do not move it from here. 

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
        g, components, node2common_name = analyzeAlarmdata(df_alarms_new, num_sub_graphs=4, min_intersection_f=3, next_start_gap=60*2,  next_end_gap=60*20, edge_drop_factor=edge_drop_factor)

        df_alarms_new["SourceName"] = df_alarms_new["SourceName"].apply(lambda name: node2common_name[name] if name in node2common_name.keys() else name)
        # print(components)
        if len(components) == 0:
            break
    edge_drop_factor += 0.1

    groups = ["".join(s) for s in df_alarms_new["SourceName"].unique() if s.find("=>") != -1]
    print(f">>Edge drop factor= {edge_drop_factor} and final Groups:  Length = {len(groups)}, Groups={groups}")


plotSourceAndCondtionHistogram(df_alarms_new) # to visualize which sourcenames are grouped

#%%
""" How many Nuisance alarms will be reduced if we do grouping? """

main_sources =[name for name in df_alarms_new["SourceName"].unique() if name.find("=>") ==-1]

groups =  [name for name in df_alarms_new["SourceName"].unique() if name.find("=>") !=-1]
print(">> Groups",groups)

groups_count = [[(sname,source2count[sname]) for sname in group.split("=>")] for group in groups]  
print(">> Groups heads",groups_count)
groups_heads = [max(l, key=lambda arg: arg[1]) for l in groups_count]
print(">> Final Group heads",groups_heads)

heads_snames = [t[0] for t in groups_heads]
print(">> heads snames",heads_snames)

total_alarms = sum([v for k,v in source2count.items()])
useless_snames = [(t[0],t[1]) for group_count in groups_count for t in group_count if t[0] not in heads_snames]
print(">> Useless",useless_snames)

num_useless_alarms = sum([t[1] for t in useless_snames])

x_axis = ["Overall Nuisance Alarms", "Nuisance Alarms with Grouping"]
y_axis = [total_alarms, total_alarms-num_useless_alarms]


plotBargraph(x_axis = x_axis, y_axis=y_axis, xtitle="",ytitle="# of alarms")



# %%
