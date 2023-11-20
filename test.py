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


bg, arbre, iter = Graph.Graph.bellmanFord(graph, 1)
print("Distances des plus court chemin de 1: ", bg)
print("Plus court chemin en partant de 1", arbre)
print("Le nombre d'iterations: ", iter)

print("")

bg, arbre, iter = Graph.Graph.bellmanFordMVP(graph, 1)
print("Distances des plus court chemin de 1: ", bg)
print("Plus court chemin en partant de 1", arbre)
print("Le nombre d'iterations: ", iter)

