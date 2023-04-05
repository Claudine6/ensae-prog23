from CMpro import Graph, graph_from_file, find, union, kruskal, route_x_out, preprocessing_test, etape_2, force_brute


data_path = "input/"
file_name = "network.01.in"

"""g= graph_from_file("input/network.1.in")
g_1=kruskal(g)
print(kruskal(g))
print(g_1.dfs())
print(g_1.get_power_and_path(10,10))"""
route_x_out("input/network.1.in","input/routes.1.in")

"""print(preprocessing_test("delivery_network/trucks.in"))
print(etape_2("input/network.1.in","input/routes.1.in","input/trucks1.in"))
print(len(etape_2("input/network.1.in","input/routes.1.in","input/trucks1.in")))"""
print(force_brute("input/network.1.in","input/routes.1.in","input/trucks1.in",500000))
