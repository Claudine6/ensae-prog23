from graph1 import Graph, graph_from_file, find, union, kruskal, estimated_time,estimation_2


data_path = "input/"
file_name = "network.01.in"



import time 
print(estimated_time("input/routes.1.in","input/network.1.in"))
print(estimated_time("input/routes.1.in","input/network.1.in"))
g_1= graph_from_file("input/network.1.in")
print(g_1.dfs)
print(kruskal(g_1))
print(kruskal(g_1).get_power_and_path(1,9))
