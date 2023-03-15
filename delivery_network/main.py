from graph1 import Graph, graph_from_file, find, union, kruskal, estimated_time


data_path = "input/"
file_name = "network.01.in"

g = graph_from_file("input/network.02.in")
print(g)
print(g.dfs())
print(kruskal(g))

import time 
print(estimated_time("input/routes.1.in","input/network.1.in"))

