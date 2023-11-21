import Graph

vertices = [1, 2, 3, 4, 5, 6, 7, 8]
adjacents = {1 : {(2, 3), (3, 2)}, 2 : {(3, 5)}, 3: {(4, 7)}, 4 : {(5, 1), (6, 4), (7, 2)}, 5 : {(7, 2)}, 6 : {(5, 2), (8, 5)}, 7 : {(1, 2)}, 8 : {(3, 4), (2, 3)}}
graph = Graph.Graph(vertices, adjacents)
print("Graphe: ")
print("Sommets:", graph.V)
print("Arcs:", graph.E)
print("")

s = Graph.Graph.glouton_fas(graph)
print(s)
print("")

v = [1, 2, 3, 4]
a = {1 : {(2,-4)}, 2 : {(3, -8)}, 3 : {(1, 7), (4, 2)}}
negative_graph = Graph.Graph(v, a) 
print("Graphe cycle negatif: ")
print("Sommets:", negative_graph.V)
print("Arcs:", negative_graph.E)
print("")

bg, arbre, iter = Graph.Graph.bellmanFord(negative_graph, 1)
print("Distances des plus court chemin de 1: ", bg)
print("Plus court chemin en partant de 1: ", arbre)
print("Le nombre d'iterations: ", iter)
print("")


graph_rand = Graph.Graph.random_graph_unary_weight(5, 0.3)
print("Graphe random: ")
print("Sommets:", graph_rand.V)
print("Arcs:", graph_rand.E)
print("")

dico = Graph.Graph.out_degrees(graph_rand)
deg = Graph.Graph.node_with_high_out_degree(graph_rand)
print("Sommet avec degree sortant > |V|/2: ", deg)

graph_rand_w1 = Graph.Graph.weighed_graph(graph_rand, 3)
graph_rand_w2 = Graph.Graph.weighed_graph(graph_rand, 5)
graph_rand_w3 = Graph.Graph.weighed_graph(graph_rand, 10)

print("Graphe random w1: ")
print("Sommets:", graph_rand_w1.V)
print("Arcs:", graph_rand_w1.E)
print("")

print("Graphe random w2: ")
print("Sommets:", graph_rand_w2.V)
print("Arcs:", graph_rand_w2.E)
print("")

print("Graphe random w3: ")
print("Sommets:", graph_rand_w3.V)
print("Arcs:", graph_rand_w3.E)
print("")


