
# %%
from typing import Final
from helpers.ploting import plotBargraph, plotSubBarGraphs
from helpers.graphs import filterEdges
from helpers.operator_actions import __getTrueAndNuisanceSources, __getFinalOperatorAlarmRelationGraph, getTrueAndNuisanceSourceNames
from helpers.alarms import  filterAlarmData, getDFWithCommonSourcesInAllMonths, loadAlarmsData, loadOperatorData
from helpers import alarms
import time

# %%
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
df_main_alarms.info()

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
ignore_comm_alarms :Final = True
momentary_alarms_f:Final = 20  # seconds
staling_alarm_f:Final = (60*60) * 12 # hours
min_alarms_per_source_f:Final = 20 # any source which is not triggered atleast 20 times in whole dataset will be removed
months_f:Final = df_main_alarms["Year-Month"].unique()
print(f">> Unique Year-Month {months_f}")
snames_f:Final = [] # ONLY USE IF NOT IGNORING COMM ALRMS

df_alarms_new = filterAlarmData(df_main_alarms, months=months_f, sources_filter=snames_f,
                                     monmentarly_filter=momentary_alarms_f, staling_filter=staling_alarm_f, ingore_communication_alarms=ignore_comm_alarms, min_alarms_per_source=min_alarms_per_source_f)


df_actions_new = df_main_actions[df_main_actions["Year-Month"].isin(months_f)]
df_alarms_new

#%%
""" 
    For Opertartor Graphs
    1. Construct sub-graphs equal to number of months
    2. Min intersections is sub-graphs/2 + 1
    3. Remove edges which are not significantaly contributing to nodes count
        It will vary in op actions and in grouping. In op actions
        it will be higher but in groupign it will be lower.
"""
num_sub_graphs:Final = len(months_f)
min_intersection_f:Final = num_sub_graphs//2 + 1
edge_filter:Final = 3 # BE CAREFUL OVER HERE


temp_g = __getFinalOperatorAlarmRelationGraph(df_alarms=df_alarms_new, df_actions=df_actions_new,
                                            num_graphs=num_sub_graphs, min_graphs_intersection_filter=min_intersection_f)

main_g = filterEdges(temp_g, edge_drop_factor=edge_filter)
print(">> Done")

# %%
operator_nodes = [action for action in main_g.nodes if action.find("Operator") != -1]

print(f">> # of Edges Main graph1 {main_g.number_of_edges()}")
print(f">> Total number of operator Tags in the graph={len(operator_nodes)}")
print(f">> Total number of Alarm Tags in the graph = {len(main_g.nodes)- len(operator_nodes)}")

# %%
"""
    Plotting actions and without action alarms
"""

true_sources, nuisance_sources = __getTrueAndNuisanceSources(df_alarms_new, main_g)

total_number_of_alarms = df_alarms_new.shape[0]
number_of_alarms_not_have_action = df_alarms_new[df_alarms_new["SourceName"].isin(
    nuisance_sources)].shape[0]

x_axis1 = ["Raw Alarms", "True Alarms", "Nuisance Alarms"]
y_axis1 = [total_number_of_alarms, total_number_of_alarms -
          number_of_alarms_not_have_action, number_of_alarms_not_have_action]

x_axis2 = ["True Alarms", "Nuisance Alarms"]
y_axis2 = [((total_number_of_alarms-number_of_alarms_not_have_action)/total_number_of_alarms)
          * 100, (number_of_alarms_not_have_action/total_number_of_alarms)*100]


plotSubBarGraphs(x_axis1,y_axis1,x_axis2,y_axis2)

# %%

# for debugging purposes
assert df_alarms_new.shape[0] == (df_alarms_new[df_alarms_new["SourceName"].isin(
    true_sources)].shape[0] + df_alarms_new[df_alarms_new["SourceName"].isin(nuisance_sources)].shape[0])

print(f"\n>>Total number of operator Actions = {df_actions_new.shape[0]}")
print(
    f"\n\n>>True Sources: Total = {len(true_sources)},  Sources = {true_sources}")
print(
    f"\n\n>> Nuisance Sources: Total = {len(nuisance_sources)}, {nuisance_sources}")


# %%

# for cross check 
true_s, nuisance_s = getTrueAndNuisanceSourceNames(df_alarms_new,df_actions_new,num_sub_graphs,min_intersection_f,edge_filter=edge_filter)
print(f"Cross Checking \n \n True Alarms ({len(true_s)}) = {true_s} \n\nNuisance Sources ({len(nuisance_s)}) = {nuisance_s}")

# %%
import plotly.graph_objects as go

df_true_sources = df_alarms_new.loc[df_alarms_new["SourceName"].isin(true_s)]
df_nuisance_sources = df_alarms_new.loc[df_alarms_new["SourceName"].isin(nuisance_s)]

month2_A_true = dict(df_true_sources["Year-Month"].value_counts())
month2_A_nuisance = dict(df_nuisance_sources["Year-Month"].value_counts()) 






x_axis = alarms.sortMonthYearTuple(month2_A_true.keys()) 
y_true = [month2_A_true[month_year] for month_year in x_axis]
trace1 = go.Bar(x=x_axis,y=y_true, name="True Alarms",text=y_true, textposition='outside' )
y_nuisance = [month2_A_nuisance[month_year] for month_year in x_axis]
trace2 = go.Bar(x= x_axis, y= y_nuisance,name="Nuisance Alarms",text=y_nuisance, textposition='outside')

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
