import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from pyvis.network import Network

pio.orca.config.executable = "/home/waris/anaconda3/envs/tupras/bin/orca"

def plotOperatorAlarmRelationHeatMap(g):

    int2operator = dict(
        enumerate([action for action in g.nodes if action.find("Operator") != -1]))
    int2alarm = dict(
        enumerate([alarm for alarm in g.nodes if alarm.find("Operator") == -1]))

    alarm2int = {v: k for k, v in int2alarm.items()}
    operator2int = {v: k for k, v in int2operator.items()}

    # data2 = np.zeros((len(operator2int),len(alarm2int)))
    data = [[None for i in range(len(alarm2int))]
            for j in range(len(operator2int))]
    print(">> Dimension", len(data[0]), len(data))
#     print(data)

    print(
        f"After Removol: Alarm Tags = {len(alarm2int)} \n Operator Tags ={len(operator2int)} ")

    for op, al, weight in g.edges.data("weight"):
        data[operator2int[op]][alarm2int[al]] = weight
        

    fig = go.Figure(data=go.Heatmap(z=data, y=[int2operator[v] for v in int2operator.keys()], x=[
                    int2alarm[v] for v in int2alarm.keys()], hoverongaps=False, colorscale='Viridis'))  # Greys

    fig.update_layout(width=1000, height=1000,
                      xaxis_nticks=len(alarm2int), yaxis_nticks=len(operator2int))
    fig.update_xaxes(side="top")
    fig.show()
    return data

def plotAlarmsRelationsHeatMap(g):

    int2alarm = dict(enumerate(g.nodes))
    alarm2int = {v: k for k, v in int2alarm.items()}
    data = [[None for i in range(len(alarm2int))]
            for j in range(len(alarm2int))]

    if len(data) == 0:
        print(" --------------> Heatmap:no data exist in heatmap")
        return None

    print(">> Dimension", len(data[0]), len(data))

    for s, d, weight in g.edges.data("weight"):
        # Reversing source and destinatin inorder to make x-axis source
        data[alarm2int[d]][alarm2int[s]] = weight

    fig = go.Figure(
        data=go.Heatmap(
            z=data, x=[v for k, v in int2alarm.items()], y=[v for k, v in int2alarm.items()],
            hoverongaps=False, hovertemplate=None, colorscale='Viridis'
        )
    )  # Greys

    fig.update_layout(
        width=1000,
        height=1000,
        xaxis_nticks=len(alarm2int),
        yaxis_nticks=len(alarm2int),
        yaxis=dict(title='Child', titlefont_size=16, tickfont_size=14),
        xaxis=dict(title='Parent', titlefont_size=16, tickfont_size=14)
    )

    fig.show()
    return data

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





#%% 

"""Currently using these graphs """ 
def plotSourceAndCondtionHistogram(df):
    fig = px.histogram(df, x="SourceName", color='Condition')
    fig.show()

def plotBargraph(x_axis, y_axis, xtitle="", ytitle=""):
    fig = px.bar(x=x_axis, y=y_axis)
    fig.update_layout(yaxis=dict(title=ytitle,
        titlefont_size=16,
        tickfont_size=14,
    ), xaxis=dict(
        title=xtitle,
        titlefont_size=16,
        tickfont_size=14,
    ))
    fig.show()