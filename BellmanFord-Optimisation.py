import Graph
import copy
import sys

sys.setrecursionlimit(5000)

def exec(question):
    print("---START---\n")

    def q1():
        vertices = [1, 2, 3, 4, 5, 6, 7, 8]
        adjacents = {1 : {(2, 3), (3, 2)}, 2 : {(3, 5)}, 3: {(4, 7)}, 4 : {(5, 1), (6, 4), (7, 2)}, 5 : {(7, 2)}, 6 : {(5, 2), (8, 5)}, 7 : {(1, 2)}, 8 : {(3, 4), (2, 3)}}
        graph = Graph.Graph(vertices, adjacents)
        print("Graphe: ")
        print("Sommets:", graph.V)
        print("Arcs:", graph.E)
        print("")
        bg, arbre, iter = Graph.Graph.bellmanFord(graph, 1)
        print("Bellman-Ford: ", bg, arbre, iter)

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

    def q2():
        vertices = [1, 2, 3, 4, 5, 6, 7, 8]
        adjacents = {1 : {(2, 3), (3, 2)}, 2 : {(3, 5)}, 3: {(4, 7)}, 4 : {(5, 1), (6, 4), (7, 2)}, 5 : {(7, 2)}, 6 : {(5, 2), (8, 5)}, 7 : {(1, 2)}, 8 : {(3, 4), (2, 3)}}
        graph = Graph.Graph(vertices, adjacents)
        s = Graph.Graph.glouton_fas(graph)
        print(s)
        print("")

    def q3():
        bool = True
        while bool:
            bool = False
            graph_rand = Graph.Graph.random_graph_unary_weight(5, 0.3)
    
            deg = Graph.Graph.can_reach_half(graph_rand)
        
            if deg == None:
                bool = True
                continue
        
            print("Graphe random: ")
            print("Sommets:", graph_rand.V)
            print("Arcs:", graph_rand.E)
            print("")

            print("Sommet qui peut atteindre au moins |V|/2: ", deg, "\n")

            graph_rand_w1 = Graph.Graph.weighed_graph(graph_rand, 3)
            graph_rand_w2 = Graph.Graph.weighed_graph(graph_rand, 6)
            graph_rand_w3 = Graph.Graph.weighed_graph(graph_rand, 9)
            H = Graph.Graph.weighed_graph(graph_rand, 12)

            graphs = [graph_rand_w1, graph_rand_w2, graph_rand_w3, H]

            i = 0

            for g in graphs:
                i+=1
                bg, arbre, iter = Graph.Graph.bellmanFord(g, deg)
                if bg == 0 and arbre == 0 and iter == 0:            #Cycle negatif
                    bool = True
                    break
            
                if i == 4:
                    print(f"Graphe random H (tests): ")
                else:
                    print(f"Graphe random w{i}: ")
                    
                print("Sommets:", g.V)
                print("Arcs:", g.E)
                print("")
            
        return graphs, deg
    
    def q4():
        graphs, deg = q3()
        graphs_cpy = copy.deepcopy(graphs)
        graphs_cpy.pop(3)
        i = 0
        list_path = []
        for g in graphs:
            i+=1
            bg, arbre, iter = Graph.Graph.bellmanFord(g, deg)
            if i == 4:
                print(f"Bellman-Ford sur graph H (tests): ")
            else:
                print(f"Bellman-Ford sur graph w{i}: ")
            print(f"Algorithme en partant de {deg}: {bg}\nArbre des plus courts chemins en partant de {deg}: {arbre}\nNombre d'iterations: {iter}\n")
            list_path.append(arbre)

        T = Graph.Graph.union_path(list_path)
        
        print("Union de leurs arborescences des plus courts chemins T = ", T)
        
        return graphs, deg, T
    
    def q5():
        graphs, deg, T = q4()

        graph_T = Graph.Graph.from_tree_to_graph(T)

        print("\nGraph T:")
        print("Sommets:", graph_T.V)
        print("Arcs:", graph_T.E, "\n")

        glouton_T = Graph.Graph.glouton_fas(graph_T)

        print("GloutonFas avec comme entree T: ", glouton_T, "\n")
        return graphs, deg, T, glouton_T
    
    def q6():
        graphs, deg, T, glouton_T = q5()

        print("Rappel Graphe H:")
        print("Sommets:", graphs[3].V)
        print("Arcs:", graphs[3].E, "\n")

        bg, arbre, iter = Graph.Graph.bellmanFord_gloutonFas(graphs[3], deg, glouton_T)
        print(f"Bellman-Ford sur graph H avec ordre <tot :\nAlgorithme en partant de {deg}: {bg}\nArbre des plus courts chemins en partant de {deg}: {arbre}\nNombre d'iterations: {iter}\n")
        
        return graphs, deg

    def q7():
        graphs, deg = q6()

        ordre_aleatoire = Graph.Graph.random_order(graphs[3])
        print("Ordre aleatoire: ", ordre_aleatoire, "\n")

        bg, arbre, iter = Graph.Graph.bellmanFord_gloutonFas(graphs[3], deg, ordre_aleatoire)
        print(f"Bellman-Ford sur graph H avec ordre aleatoire :\nAlgorithme en partant de {deg}: {bg}\nArbre des plus courts chemins en partant de {deg}: {arbre}\nNombre d'iterations: {iter}\n")

    def q9():
        Graph.Graph.analyze_vertex_iteration_nb(3, 10, 0.3, 3, 10)

    def q10():
        """
        # Analyse avec les memes graphs

        graphs = Graph.Graph.analyze_vertex_iteration_nb(1, 12, 0.3, 1, 15)
        Graph.Graph.analyze_vertex_iteration_nb_with_graphs(graphs, 1, 12, 1, 15)
        Graph.Graph.analyze_vertex_iteration_nb_with_graphs(graphs, 1, 12, 4, 15)
        Graph.Graph.analyze_vertex_iteration_nb_with_graphs(graphs, 1, 12, 6, 15)
        """

        # Analyse avec des graphs aleatoires
        
        Graph.Graph.analyze_vertex_iteration_nb(3, 12, 0.3, 1, 10)
        Graph.Graph.analyze_vertex_iteration_nb(1, 12, 0.3, 3, 10)
        Graph.Graph.analyze_vertex_iteration_nb(1, 12, 0.3, 5, 10)
        Graph.Graph.analyze_vertex_iteration_nb(1, 12, 0.3, 7, 10)
    
    def q11():
        #g = Graph.Graph.create_graph_by_level(4, 2500, 10)
        #Graph.Graph.pretraitement_methode(g, 3, 10)
        Graph.Graph.pretraitement_methode_graph(3, 10, 70, 4, 15)

    if question == "1":
        q1()
    elif question == "2":
        q2()
    elif question == "3":
        q3()
    elif question == "4":
        q4()
    elif question == "5":
        q5()
    elif question == "6":
        q6()
    elif question == "7":
        q7()
    elif question == "8":
        print("Le nombre d'iteration de l'algorithme Bellman-Ford est plus bas lorsque l'ordre donné par l'algorithme GloutonFas est utilise compare a un ordre aleatoire.")
    elif question == "9":
        q9()
    elif question == "10":
        q10()
    elif question == "11":
        q11()

if __name__ == '__main__':
    if len(sys.argv) != 2 or int(sys.argv[1]) > 11 or int(sys.argv[1]) < 1:
        raise ValueError("Il faut indiquer le numero de la question en argument de la ligne de commande (de 1 à 11)")

    exec(sys.argv[1])       # Indiquer le numero de la question en argument de la ligne de commande
