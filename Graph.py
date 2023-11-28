import copy
import numpy as np
import random
import matplotlib.pyplot as plt

#Bellman s'applique que pour un graphe oriente sans cycle negatif!!!

class Graph:
    V = set # Ensemble de sommets (la liste de sommets) 
    E = dict # Dictionnaire de arcs avec les poids (la liste d'arretes)


    def __init__(self, V, E=None):
        """
        Permet de creer le graphe avec les valeurs V et E pasees en parametre
        """
        
        self.V = V

        if(E == None):
            self.E = {i : set() for i in V} # Creer le dictionnaire qui relie le sommet i a vide
        else:
            self.E = E


    def insert_edge(self, v1, v2, weight): 
        """
        Permet d'ajouter une arrete
        """

        self.E[v1].add((v2, weight)) # On ajoute v1 a v2 avec le poids


    def remove_vertex(self, v):     
        """
        Retourne un nouveau graphe G_cpy sans le sommet v
        """

        G_cpy = copy.deepcopy(self) # On cree une copie independante de notre Graphe pour pouvoir renvoyer un nouveau Graphe G' 

        if v not in G_cpy.V : # S'il n'est pas dans l'ensemble de sommets, on quitte de la fonction
            return
        
        # Supprimer les arcs sortants du sommet v
        G_cpy = set()

        # Supprimer les arcs entrants vers le sommet v
        for vertex in G_cpy.V:
            if v in G_cpy.E[vertex]:
                edges_to_remove = []
                for u, weight in G_cpy.E[vertex]:
                    if v == u:
                        edges_to_remove.append((u, weight))

                for edge in edges_to_remove:
                    G_cpy.E[vertex].remove(edge)

        del G_cpy.E[v]  # On supprime le sommet dans le dictionnaire d'arcs
        G_cpy.V.remove(v)  # On supprime le sommet dans l'ensemble des sommets
            
        return G_cpy


    
    def remove_many_vertex(self, set_delete) :
        """
        Retourne un nouveau graphe G_cpy sans les sommets de l'ensemble set_delete
        """

        G_cpy = copy.deepcopy(self)   # On cree une copie independante de notre Graphe pour pouvoir renvoyer un nouveau Graphe G' 

        for i in set_delete: # On reutilise l'algorithme de la fonction remove_vertex en faisant une boucle sur les valeurs de set_delete

            if i not in G_cpy.V:         
                continue                 
            
            # Supprimer les arcs sortants du sommet i 
            G_cpy.E[i] = set()

            # Supprimer les arcs entrants vers le sommet i
            for vertex in G_cpy.V:
                if i in G_cpy.E[vertex]:
                    edges_to_remove = []
                    for u, weight in G_cpy.E[vertex]:
                        if i == u:
                            edges_to_remove.append((u, weight))

                    for edge in edges_to_remove:
                        G_cpy.E[vertex].remove(edge)

            del G_cpy.E[i]
            G_cpy.V.remove(i)

        return G_cpy
    

    def sources(self):
        """
        Retourne l'ensemble des sources (sommet sans predecesseur) dans le graphe
        """

        sources = set()

        for v in self.V:
            is_source = True

            # Parcourir tous les sommets du graphe pour vérifier s'ils ont des prédécesseurs
            for vertex in self.V:
                predecessors = set()
                
                # Récupèrer tous les prédécesseurs du sommet actuel
                for u, _ in self.E[vertex]:
                    predecessors.add(u)

                # Vérifier si le sommet v a un prédécesseur
                if v in predecessors:
                    is_source = False
                    break

            # Si le sommet v n'a pas de prédécesseur, il est ajouté à l'ensemble des sources
            if is_source:
                sources.add(v)

        return sources
    

    def sinks(self):
        """
        Retourne l'ensemble des puits (sommet sans successeur) dans le graphe
        """

        sinks = set()

        for v in self.V:
            is_sink = True

            # Parcourt tous les sommets du graphe pour vérifier s'ils ont des successeurs
            for u in self.V:
                successors = set()
                
                # Récupère tous les successeurs du sommet actuel
                for vertex, _ in self.E[u]:
                    successors.add(vertex)

                # Vérifie si le sommet v a un successeur
                if v in successors:
                    is_sink = False
                    break

            # Si le sommet v n'a pas de successeur, il est ajouté à l'ensemble des puits
            if is_sink:
                sinks.add(v)

        return sinks
        

    def max_delta_u(self):
        """
        Choisit le sommet u qui maximise δ(u) = d+(u) - d-(u)
        d+(u) = degree u avec ses sommets sortants
        d-(u) = degree u avec ses sommets entrants
        """

        max_delta = float('-inf')
        max_delta_vertex = None

        # Parcourir tous les sommets du graphe
        for vertex in self.V:
            # Calculer le degré entrant (d-) et sortant (d+) du sommet actuel
            in_degree = sum(1 for u in self.V if vertex in self.E[u])
            out_degree = len(self.E[vertex])
            
            # Calculer la différence entre le degré sortant et le degré entrant
            delta_u = out_degree - in_degree

            # Met à jour le sommet ayant le plus grand delta jusqu'à présent
            if delta_u > max_delta:
                max_delta = delta_u
                max_delta_vertex = vertex

        return max_delta_vertex


    def glouton_fas(self):
        """
        Algorithme GloutonFas pour obtenir une permutation des sommets de G
        """

        s1 = []  # s1 est la partie triée du graphe
        s2 = []  # s2 est la partie non triée du graphe
        G_cpy = copy.deepcopy(self)

        # Tant que le graphe a des sommets
        while G_cpy.V:
            # Ajoute les sources dans la partie triée (s1)
            while G_cpy.sources():
                u_source = G_cpy.sources().pop()
                s1.append(u_source)
                G_cpy.V.remove(u_source)
            
            # Ajoute les puits dans la partie non triée (s2)
            while G_cpy.sinks():
                u_sink = G_cpy.sinks().pop()
                s2.append(u_sink)
                G_cpy.V.remove(u_sink)

            # Ajoute le sommet ayant le plus grand delta dans la partie triée (s1)
            u_max_delta = G_cpy.max_delta_u()
            if u_max_delta is not None:
                s1.append(u_max_delta)
                G_cpy.V.remove(u_max_delta)

        # print("s1: ", s1)
        # print("s2: ", s2)

        return s1 + s2
                        

    def random_graph_unary_weight(n, p):
        """
        Cree un graphe oriente de n sommets, où chaque arête (i, j) est présente avec la probabilité p et avec un poids unaire
        """

        if n < 0:
            raise ValueError("Le parametre n doit etre superieur a 0")

        if p < 0 or p > 1:
            raise ValueError("Le parametre p doit etre entre 0 et 1")

        vertices = set(range(n))
        graph = Graph(vertices)

        for i in range(n):
            for j in range(n):
                if random.random() < p and i != j:
                    graph.insert_edge(i, j, 1) # Chaque arete a un poids = 1

        return graph
    

    def weighed_graph(self, w):
        """
        Crée un nouveau graphe avec des poids entre [-w,w] pour chaque arête
        """

        new_E = {vertex: set() for vertex in self.V}

        for vertex in self.V:
            for edge in self.E[vertex]:
                # Modification du poids de manière aléatoire, excluant le poids 0
                new_weight = random.randint(-w, w)
                while new_weight == 0:
                    new_weight = random.randint(-w, w)
                
                new_edge = (edge[0], new_weight)  # Nouvelle paire (sommet, poids)
                new_E[vertex].add(new_edge)

        # Créer un nouveau graphe avec le nouveau dictionnaire d'arêtes
        new_graph = Graph(copy.deepcopy(self.V), new_E)
        
        return new_graph
    

    def out_degrees(self):
        """
        Calcule le degré sortant de chaque commet dans le graphe
        Retourne un dictionnaire avec les sommets en tant que clés et leur degré sortant en tant que valeurs
        """
        
        degrees = {vertex: 0 for vertex in self.V}

        for vertex in self.V:
            for _, _ in self.E[vertex]:
                degrees[vertex] += 1

        return degrees
    

    def node_with_high_out_degree(self):    # A MODIFIER ET REGRDER DANS LE RAPPORT
        """
        Renvoie un sommet avec un degré sortant supérieur à |V|/2
        Si aucun sommet ne satisfait cette condition, renvoie None
        """

        out_degrees = self.out_degrees()
        threshold = len(self.V) / 2

        for vertex, out_degree in out_degrees.items():
            if out_degree > threshold:
                return vertex

        return None
    
    
    def can_reach_half(self):
        """
        Renvoie un sommet avec un degré sortant supérieur à |V|/2
        Si aucun sommet ne satisfait cette condition, renvoie None
        """

        threshold = len(self.V) // 2 + 1

        def recursive_count(self, vertex, visited, count):
            visited.add(vertex)

            for neighbor, _ in self.E[vertex]:
                if neighbor not in visited:
                    count = recursive_count(self, neighbor, visited, count + 1)

            return count

        for source in self.V:
            visited = set()
            reachable_count = recursive_count(self, source, visited, 0)

            if reachable_count >= threshold:
                return source

        return None


    def random_order(self):
        """
        Renvoie un ordre aleatoire de sommets du graphe
        """

        order = []

        for vertex in self.V:
            order.append(vertex)

        random.shuffle(order)

        return order
    

    def union_path(list_path):
        """
        Renvoie l'union des arborescences des plus courts chemins
        """

        T = {i : [] for i in list_path[0]}
        
        for dico in list_path:
            for vertex, path in dico.items():
                if path not in T[vertex]:
                    T[vertex].append(path)

        return T
    
    
    def from_tree_to_graph(T):
        """
        Renvoie un graphe pondere unaire a partir de l'arborescence T
        """

        graph = Graph(set(T.keys()))  # Création d'un graphe avec les sommets de T

        already_in = []

        for _, paths in T.items():
            for path in paths:
                if len(path) > 1:
                    for i in range(len(path) - 1):
                        v1, v2 = path[i], path[i + 1]
                        if v1 != None and (v1,v2) not in already_in:
                            # Ajout d'une arête entre les sommets successifs dans le chemin avec un poids de 1
                            graph.insert_edge(v1, v2, 1)
                            already_in.append((v1,v2))

        return graph
    

#nb d'iterations qu'il prend, pour chaque sommet, de determiner le plus court chemin => stocker ca dans une liste
#a faire en fonction de l'algo glouton et l'ordre et a comparer avec belman classique comme premier test
#a faire une fonction qui va tracer un graphe en fonction du nombre d'iterations pour trouver le plus court chemin (en fonction des sauts(cb de sauts il doit faire))
#ordre aleatoire test bellman
#ordre glouton test bellman    a essayer plusieurs fois pour voir si le nb d'iterations differe

    def bellmanFord(self, start):
        """
        Retourne les distances et les chemins des plus court chemins de start vers les sommets du graph
        """

        distances = {}
        predecesseurs = {}
        
        for v in self.V:
            distances[v] = np.inf # On fixe dv = infini
            predecesseurs[v] = None
        distances[start] = 0

        # iterations = 0

        for iteration in range(len(self.V) - 1): # On itere sur tous les sommets
            converged = True
            for y in self.E: # On verifie chaque arc entrant
                for v, w in self.E[y]:
                    if distances[y] + w < distances[v]:
                        distances[v] = distances[y] + w
                        predecesseurs[v] = y
                        converged = False
            if converged:
                break

        # On verifie qu'il n'y a pas de Cycle negatif
        for y in self.E:
            for v, w in self.E[y]:
                if distances[v] > distances[y] + w:
                    # print("Cycle negatif")
                    return 0, 0, 0

                
        paths = {}
        for p in predecesseurs: # On regarde le chemin pour chaque sommet du graph
            path = []
            a = p
            while p != start and p != None: # Tant qu'on arrive pas au sommet de depart
                path.append(p) # On ajoute le sommet au chemin
                p = predecesseurs[p] # Puis on passe a son predecesseur
            path.append(p)
            paths[a] = path[::-1] # On retourne la liste

        return distances, paths, iteration+1
    

    def bellmanFord_gloutonFas(self, start, ordre):
        """
        Retourne les distances et les chemins des plus court chemins de start vers les sommets du graph
        """

        distances = {}
        predecesseurs = {}
        
        for v in self.V:
            distances[v] = np.inf # On fixe dv = infini
            predecesseurs[v] = None
        distances[start] = 0

        for iteration in range(len(self.V) - 1): # On itere sur tous les sommets
            converged = True
            for y in ordre:
                if y in self.E: # On verifie chaque arcs entrants
                    for v, w in self.E[y]:
                        if distances[y] + w < distances[v]:
                            distances[v] = distances[y] + w
                            predecesseurs[v] = y
                            converged = False
            if converged:
                break

        # On verifie qu'il n'y a pas de Cycle negatif
        for y in self.E:
            for v, w in self.E[y]:
                if distances[v] > distances[y] + w:
                    print("Cycle negatif")
                    return 0, 0, 0

                
        paths = {}
        for p in predecesseurs: # On regarde le chemin pour chaque sommet du graph
            path = []
            a = p
            while p != start and p != None: # Tant qu'on arrive pas au sommet de depart
                path.append(p) # On ajoute le sommet au chemin
                p = predecesseurs[p] # Puis on passe a son predecesseur
            path.append(p)
            paths[a] = path[::-1] # On retourne la liste

        return distances, paths, iteration+1
    

    def analyze_vertex_iteration_nb(num_graphs_per_size, Nmax, p, nb_g, weight_interval):
        """
        Fonction qui trace un graphe avec en abscisse le nombre d'iterations
        et en ordonne le nombre des sommets pour un nombre fixe des graphes de test
        """
        bool = True

        if (Nmax <= 3):
            raise ValueError("Le parametre Nmax doit etre superieur a 3 pour avoir des tests pertinents")

        list_iterations_vertex_mean_glouton = []
        list_iterations_vertex_mean_alea = []
        iteration_number_glouton = np.zeros(Nmax - 3)
        iteration_number_alea = np.zeros(Nmax - 3)

        for n in range(4, Nmax + 1):
            
            for _ in range(num_graphs_per_size):
                bool = True
                while bool:
                    bool = False
                    graph = Graph.random_graph_unary_weight(n, p)
                    
                    list_graph_w = []
                    list_weights = []
                    for _ in range (nb_g): # Creation des graphes ponderees
                        
                        while True:
                            weight = random.randint(1, weight_interval)
                            if weight not in (list_weights):
                                list_graph_w.append(Graph.weighed_graph(graph, weight))
                                list_weights.append(weight)
                                break

                    while True:
                        weight = random.randint(1, weight_interval)
                        if weight not in (list_weights):
                            graph_test_H = Graph.weighed_graph(graph, weight)
                            list_graph_w.append(graph_test_H)
                            break

                    deg = Graph.can_reach_half(graph)

                    if deg == None:
                        bool = True #recommencer
                        continue
                        
                    list_path = []

                    list_graph_G = copy.deepcopy(list_graph_w)
                    list_graph_G.pop(len(list_graph_G) - 1)

                    for current_graph in list_graph_G:
                        bg, arbre, iter = Graph.bellmanFord(current_graph, deg)
                        if bg == 0 and arbre == 0 and iter == 0: # cycle negatif
                            bool = True #recommencer
                            break
                        list_path.append(arbre)

                    if bool:
                        continue


                    T = Graph.union_path(list_path)

                    graph_T = Graph.from_tree_to_graph(T)
                    
                    glouton_T = Graph.glouton_fas(graph_T)

                    _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, glouton_T)

                    ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])

                    _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, ordre_aleatoire)


                    iteration_number_glouton[n-4] = iteration_number_glouton[n-4] + iter_glouton
                    iteration_number_alea[n-4] = iteration_number_alea[n-4] + iter_alea

                    print("iter gl : ",iter_glouton)
                    print("ite number gl :",iteration_number_glouton)
                    print("iter alea : ",iter_alea)
                    print("iteration_number_alea : ", iteration_number_alea)

            
            list_iterations_vertex_mean_glouton = iteration_number_glouton / num_graphs_per_size
            list_iterations_vertex_mean_alea = iteration_number_alea / num_graphs_per_size
            print("ite number gl apres boucle :",iteration_number_glouton)

        # Tracé du temps d'exécution en fonction de la taille du graphe (n)
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_alea, marker='o', label='Aléatoire')
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_glouton, marker='o', label='Glouton')
        plt.xlabel("Nombre de sommets du graphe (n)")
        plt.ylabel("Nombre moyen d'itérations")
        plt.title("Nombre moyen d'itérations de Bellman-Ford en fonction du nombre de sommets")
        plt.legend()
        plt.show()

 
    def analyze_vertex_iteration_nb_with_graphs(graphs, num_graphs_per_size, Nmax, p, nb_g, weight_interval):
        """
        Fonction qui trace un graphe avec en abscisse le nombre d'iterations
        et en ordonne le nombre des sommets pour un nombre fixe des graphes de test
        """
        bool = True

        if (Nmax <= 3):
            raise ValueError("Le parametre Nmax doit etre superieur a 3 pour avoir des tests pertinents")

        list_iterations_vertex_mean_glouton = []
        list_iterations_vertex_mean_alea = []
        iteration_number_glouton = np.zeros(Nmax - 3)
        iteration_number_alea = np.zeros(Nmax - 3)
        
        for n in range(4, Nmax + 1):
            
            for _ in range(num_graphs_per_size):
                bool = True
                while bool:
                    bool = False
                    graph = graphs[n-4]
                    
                    list_graph_w = []
                    list_weights = []
                    for _ in range (nb_g): # Creation des graphes ponderees
                        
                        while True:
                            weight = random.randint(1, weight_interval)
                            if weight not in (list_weights):
                                print("oioioi")
                                graph_cpy = copy.deepcopy(graph)
                                list_graph_w.append(Graph.weighed_graph(graph_cpy, weight))
                                list_weights.append(weight)
                                break

                    while True:
                        weight = random.randint(1, weight_interval)
                        if weight not in (list_weights):
                            graph_cpy = copy.deepcopy(graph)
                            graph_test_H = Graph.weighed_graph(graph_cpy, weight)
                            list_graph_w.append(graph_test_H)
                            break

                    deg = Graph.can_reach_half(graph)

                    if deg == None:
                        bool = True #recommencer
                        continue
                        
                    list_path = []

                    list_graph_G = copy.deepcopy(list_graph_w)
                    list_graph_G.pop(len(list_graph_G) - 1)

                    for current_graph in list_graph_G:
                        bg, arbre, iter = Graph.bellmanFord(current_graph, deg)
                        if bg == 0 and arbre == 0 and iter == 0: # cycle negatif
                            bool = True #recommencer
                            break
                        list_path.append(arbre)

                    if bool:
                        print("lalalala")
                        continue


                    T = Graph.union_path(list_path)

                    graph_T = Graph.from_tree_to_graph(T)
                    
                    glouton_T = Graph.glouton_fas(graph_T)

                    _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, glouton_T)

                    ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])

                    _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, ordre_aleatoire)


                    iteration_number_glouton[n-4] = iteration_number_glouton[n-4] + iter_glouton
                    iteration_number_alea[n-4] = iteration_number_alea[n-4] + iter_alea

                    print("iter gl : ",iter_glouton)
                    print("ite number gl :",iteration_number_glouton)
                    print("iter alea : ",iter_alea)
                    print("iteration_number_alea : ", iteration_number_alea)

            
            list_iterations_vertex_mean_glouton = iteration_number_glouton / num_graphs_per_size
            list_iterations_vertex_mean_alea = iteration_number_alea / num_graphs_per_size
            print("ite number gl apres boucle :",iteration_number_glouton)

        # Tracé du temps d'exécution en fonction de la taille du graphe (n)
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_alea, marker='o', label='Aléatoire')
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_glouton, marker='o', label='Glouton')
        plt.xlabel("Nombre de sommets du graphe (n)")
        plt.ylabel("Nombre moyen d'itérations")
        plt.title("Nombre moyen d'itérations de Bellman-Ford en fonction du nombre de sommets")
        plt.legend()
        plt.show()

        


    def create_graph_by_level(nb_v, nb_level, weight_interval):
        V = []
        E = {}

        num_v = 0

        for i in range(nb_level):
            for _ in range(nb_v):
                V.append(num_v)
                num_v += 1

            if i != 0:
                for j in range(num_v-2*nb_v, num_v-nb_v):          # niveau j
                    for k in range(num_v-nb_v, num_v):      # niveau j+1
                        weight = random.randint(-weight_interval, weight_interval)
                        while weight == 0:
                            weight = random.randint(-weight_interval, weight_interval)
                        if j not in E:
                            E[j] = set()
                        E[j].add((k,weight))
        
        for i in range(num_v-nb_v, num_v):
            E[i] = set()

        graph = Graph(V, E)
        return graph
    
    def pretraitement_methode(self, nb_g, weight_interval):
        """
        Fonction qui trace un graphe avec en abscisse le nombre d'iterations
        et en ordonne le nombre des sommets pour un nombre fixe des graphes de test
        """        
        bool = True

        if (len(self.V) <= 3):
            raise ValueError("Le parametre Nmax doit etre superieur a 3 pour avoir des tests pertinents")
        
        bool = True
        while bool:
            bool = False
            
            list_graph_w = []
            list_weights = []
            for _ in range (nb_g+1): # Creation des graphes ponderees
                
                while True:
                    weight = random.randint(1, weight_interval)
                    if weight not in (list_weights):
                        list_graph_w.append(Graph.weighed_graph(self, weight))
                        list_weights.append(weight)
                        break

            while True:
                weight = random.randint(1, weight_interval)
                if weight not in (list_weights):
                    graph_test_H = Graph.weighed_graph(self, weight)
                    list_graph_w.append(graph_test_H)
                    break

            deg = Graph.can_reach_half(self)

            if deg == None:
                bool = True #recommencer
                continue
                
            list_path = []

            list_graph_G = copy.deepcopy(list_graph_w)
            list_graph_G.pop(len(list_graph_G) - 1)

            for current_graph in list_graph_G:
                bg, arbre, iter = Graph.bellmanFord(current_graph, deg)
                if bg == 0 and arbre == 0 and iter == 0: # cycle negatif
                    bool = True #recommencer
                    break
                list_path.append(arbre)

            if bool:
                continue


            T = Graph.union_path(list_path)

            graph_T = Graph.from_tree_to_graph(T)
            
            glouton_T = Graph.glouton_fas(graph_T)

            _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, glouton_T)

            ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])

            _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, ordre_aleatoire)

            print("Iterations avec ordre donne par GloutonFas: ", iter_glouton, "\nIterations avec ordre aleatoire: ", iter_alea)
