# %%
from helpers.group_alarms import analyzeAlarmdata
from helpers.ploting import plotBargraph, plotSourceAndCondtionHistogram
from helpers.alarms import filterAlarmData, loadAlarmsData, loadOperatorData
from helpers import operator_actions as operator
import time


#%%

"""  Lodading the Data and Preprocessing """
PATH = "/home/waris/Github/tupras-analysis/data/"
alarm_file_path = PATH + "processed/alarms/final/final-all-months-alarms.csv"
op_action_file_path = PATH + "processed/operator-actions/final/final-all-month-actions.csv"

start = time.time()

df_main_alarms =loadAlarmsData(file_path=alarm_file_path)
df_main_actions = loadOperatorData(file_path=op_action_file_path)

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
months_f = df_main_alarms["Year-Month"].unique()
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

true_sources, nuisance_sources= operator.getTrueAndNuisanceSourceNames(df_alarms=df_alarms_new,df_operator=df_main_actions,num_sub_graphs=len(months_f),min_intersection_f=(len(months_f)//2)+1,edge_filter=3)

print(f">> True Sources ({len(true_sources)}) = {true_sources} \n\n>> Nuisance Sources ({len(nuisance_sources)}) = {nuisance_sources}")


# %%
print(">> Before Raw Alarms", df_alarms_new.shape[0])
df_alarms_nuisance_temp = df_alarms_new[df_alarms_new["SourceName"].isin(nuisance_sources)] # only analyze the nusiance alrms
print(">> After: Only Nusiance alarms", df_alarms_nuisance_temp.shape[0])
source2count = dict(df_alarms_nuisance_temp["SourceName"].value_counts()) # do not move it from here. 

# %%
def groupNuisanceSourceNamesInDF(df, max_edge_drop_factor=1.3, num_sub_graphs=4, min_intersection_f=3, next_start_gap=60*2,  next_end_gap=60*20):
    """Grouping of Source Names"""
    # max_edge_drop_factor = 1.3 # factor x weight < count ignore such edge 
    iteration = 0
    edge_drop_factor = 1.0 # start from one to one relation and go to max edge drop factor
    while edge_drop_factor < max_edge_drop_factor:
        g = None
        while True:
            iteration += 1
            print(
                f">> ==========Level=1 Iteration of Merging Components = {iteration} =============")
            g, components, node2common_name = analyzeAlarmdata(df, num_sub_graphs=num_sub_graphs, min_intersection_f=min_intersection_f, next_start_gap=next_start_gap,  next_end_gap=next_end_gap, edge_drop_factor=edge_drop_factor)

            df["SourceName"] = df["SourceName"].apply(lambda name: node2common_name[name] if name in node2common_name.keys() else name)
            # print(components)
            if len(components) == 0:
                break
        edge_drop_factor += 0.1

        groups = ["".join(s) for s in df["SourceName"].unique() if s.find("=>") != -1]
        print(f">>Edge drop factor= {edge_drop_factor} and final Groups:  Length = {len(groups)}, Groups={groups}")




groupNuisanceSourceNamesInDF(df_alarms_nuisance_temp) # grouped sources as 1 
plotSourceAndCondtionHistogram(df_alarms_nuisance_temp) # to visualize which sourcenames are grouped
#%%
print(f"\n\n>> Number of Grouped Nuisance Alarms: {len(df_alarms_nuisance_temp['SourceName'].unique())}, Source = {df_alarms_nuisance_temp['SourceName'].unique()}" )

#%%
""" How many Nuisance alarms will be reduced if we do grouping? """

main_sources =[name for name in df_alarms_nuisance_temp["SourceName"].unique() if name.find("=>") ==-1]

groups =  [name for name in df_alarms_nuisance_temp["SourceName"].unique() if name.find("=>") !=-1]
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
import plotly.graph_objects as go

from helpers import alarms

print(useless_snames)
print([sname for sname,_ in useless_snames ])


df_alarms_nuisance = df_alarms_new[df_alarms_new["SourceName"].isin(nuisance_sources)] # only analyze the nusiance alrms

# df_all_nuisance = df_alarms_nuisance.loc[df_alarms_nuisance["SourceName"].isin(useless_snames+heads_snames)]
df_use_less = df_alarms_nuisance[df_alarms_nuisance["SourceName"].isin([sname for sname,_ in useless_snames ])]

month2_all_nuisance = dict(df_alarms_nuisance["Year-Month"].value_counts())
month2_groupe_nuisance = dict(df_use_less["Year-Month"].value_counts()) 

print(df_use_less["Year-Month"].value_counts())

print(month2_all_nuisance,month2_groupe_nuisance)


x_axis = alarms.sortMonthYearTuple(month2_all_nuisance.keys()) 

y1 = [month2_all_nuisance[month_year] for month_year in x_axis]
trace1 = go.Bar(x=x_axis,y=y1, name="Total Nuisance Alarms",text=y1, textposition='outside' )
y2 = [month2_groupe_nuisance[month_year] for month_year in x_axis]
trace2 = go.Bar(x= x_axis, y= y2,name="# of useless alarms",text=y2, textposition='outside')

fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)


fig.update_layout(
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Number of Alarms',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Month",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    height=600, 
    width=1100,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside',textfont_size=80)
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()



# %%

# %%

"""
    # For Percentage
"""

print(useless_snames)
print([sname for sname,_ in useless_snames ])


df_alarms_nuisance = df_alarms_new[df_alarms_new["SourceName"].isin(nuisance_sources)] # only analyze the nusiance alrms

# df_all_nuisance = df_alarms_nuisance.loc[df_alarms_nuisance["SourceName"].isin(useless_snames+heads_snames)]
df_use_less = df_alarms_nuisance[df_alarms_nuisance["SourceName"].isin([sname for sname,_ in useless_snames ])]

month2_all_nuisance = dict(df_alarms_nuisance["Year-Month"].value_counts())
month2_groupe_nuisance = dict(df_use_less["Year-Month"].value_counts()) 

print(df_use_less["Year-Month"].value_counts())

print(month2_all_nuisance,month2_groupe_nuisance)


x_axis = alarms.sortMonthYearTuple(month2_all_nuisance.keys()) 

# y1 = [month2_all_nuisance[month_year] for month_year in x_axis]
# trace1 = go.Bar(x=x_axis,y=y1, name="Total Nuisance Alarms",text=y1, textposition='outside' )
y2 = [(month2_groupe_nuisance[month_year]/month2_all_nuisance[month_year])*100 for month_year in x_axis]
trace2 = go.Bar(x= x_axis, y= y2,name="useless alarms",text=y2, textposition='outside')

fig = go.Figure()
# fig.add_trace(trace1)
fig.add_trace(trace2)


fig.update_layout(
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Reduction of Nuisance Alarms',
        titlefont_size=16,
        tickfont_size=14,
        # range=[0,100],
    ),
    xaxis=dict(
        title = "Month",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    height=600, 
    width=1100,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.update_traces(texttemplate='%{text:.3s}%', textposition='outside',textfont_size=80)
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# %%
