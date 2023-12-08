import copy
import time
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

        for _, paths in T.items():      # On boucle sur les chemins
            for path in paths:
                if len(path) > 1:
                    for i in range(len(path) - 1):      # On boucle sur chaque noeud du chemin
                        v1, v2 = path[i], path[i + 1] 
                        if v1 != None and (v1,v2) not in already_in:    # Si l'arete n'existe pas deja
                            # Ajout d'une arête entre les sommets successifs dans le chemin avec un poids de 1
                            graph.insert_edge(v1, v2, 1)
                            already_in.append((v1,v2))

        return graph


    def bellmanFord(self, start):
        """
        Retourne les distances et les chemins des plus court chemins de start vers les sommets du graph
        """

        distances = {}
        predecesseurs = {}
        
        for v in self.V:
            distances[v] = np.inf       # On fixe dv = infini
            predecesseurs[v] = None
        distances[start] = 0

        for iteration in range(len(self.V) - 1): # On itere sur tous les sommets
            converged = True
            for y in self.E: # On verifie chaque arc entrant
                for v, w in self.E[y]:
                    if distances[y] + w < distances[v]:     # Est ce qu'on a une distance plus courte ?
                        distances[v] = distances[y] + w
                        predecesseurs[v] = y
                        converged = False       # Si les valeurs changent alors on a pas encore converge (a l'inverse si les valeurs ne change pas alors pas besoin d'iterer jusqu'au bout)
            if converged:
                break

        # On verifie qu'il n'y a pas de Cycle negatif
        for y in self.E:
            for v, w in self.E[y]:
                if distances[v] > distances[y] + w:     # Si apres avoir fini de trouver les plus courts chemins il en existe encore des plus courts cela signifie que le graph possède un cycle negatif
                    # print("Cycle negatif")
                    return 0, 0, 0

                
        paths = {}
        for p in predecesseurs:     # On regarde le chemin pour chaque sommet du graph
            path = []
            a = p
            while p != start and p != None:     # Tant qu'on arrive pas au sommet de depart
                path.append(p)                  # On ajoute le sommet au chemin
                p = predecesseurs[p]            # Puis on passe a son predecesseur
            path.append(p)
            paths[a] = path[::-1]               # On retourne la liste

        return distances, paths, iteration+1
    

    def bellmanFord_gloutonFas(self, start, ordre):
        """
        Retourne les distances et les chemins des plus court chemins de start vers les sommets du graph avec un ordre donné
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
                if y in self.E:         # On verifie chaque arcs entrants
                    for v, w in self.E[y]:
                        if distances[y] + w < distances[v]:     # Est ce qu'on a une distance plus courte ?
                            distances[v] = distances[y] + w
                            predecesseurs[v] = y
                            converged = False                   # Si les valeurs changent alors on a pas encore converge (a l'inverse si les valeurs ne change pas alors pas besoin d'iterer jusqu'au bout)
            if converged:
                break

        # On verifie qu'il n'y a pas de Cycle negatif
        for y in self.E:
            for v, w in self.E[y]:
                if distances[v] > distances[y] + w:     # Si apres avoir fini de trouver les plus courts chemins il en existe encore des plus courts cela signifie que le graph possède un cycle negatif
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
        Fonction qui trace un graphe avec en abscisse le nombre d'iterations et en ordonne le nombre des sommets pour un nombre donne des graphes de test
        """

        bool = True
        graphs = [[] for _ in range(Nmax-3)]        # Nmax - 3 car on commence avec n in range(4, Nmax+1) pour avoir des graohs avec des sommets qui commencent a 4 (pas tres interessant en dessous)

        if (Nmax <= 3):
            raise ValueError("Le parametre Nmax doit etre superieur a 3 pour avoir des tests pertinents")

        list_iterations_vertex_mean_glouton = []
        list_iterations_vertex_mean_alea = []
        iteration_number_glouton = np.zeros(Nmax - 3)
        iteration_number_alea = np.zeros(Nmax - 3)

        for n in range(4, Nmax + 1):        # On itere sur les nombres de noeuds des graphes en partant de 4
            
            for _ in range(num_graphs_per_size):    # On itere sur le nombre de graphs de taille n que l'on aura pour faire une moyenne des iterations sur ces differents graphes
                bool = True
                while bool:         # Si bool passe a False ca signifie que l'on a trouver un noeud qui atteint |V|/2 et un bon graph de base avec des bons graphs d'entrainement G un bon graph de test H (qu'ils n'ont pas de cycle negatif)
                    bool = False
                    graph = Graph.random_graph_unary_weight(n, p)
                    
                    list_graph_w = []   # On remet les listes de G a zero
                    list_weights = []   # On remet les listes de poids a zero
                    for _ in range (nb_g): # Creation des nb_g graphes ponderees G
                        
                        while True:     # Tant qu'on a pas break ca signifie qu'on a pris un poids weight deja utilise dans un autre graph G (Il est preferable de s'entrainer avec des poids differents sur les graphs G)
                            weight = random.randint(1, weight_interval)     # On choisi un poids aleatoire entre [1,weight_interval[
                            if weight not in (list_weights):                # Est ce que ce poids a deja ete utilise pour un autre G ?
                                list_graph_w.append(Graph.weighed_graph(graph, weight))
                                list_weights.append(weight)
                                break

                    # Puis plus qu'a faire le graph de test H
                    while True:
                        weight = random.randint(1, weight_interval)
                        if weight not in (list_weights):
                            graph_test_H = Graph.weighed_graph(graph, weight)
                            list_graph_w.append(graph_test_H)
                            break

                    deg = Graph.can_reach_half(graph)       # On essaie de trouver un noeud qui peut atteindre |V| / 2

                    if deg == None:     # Si deg est egal a None alors cela signifie qu'il n'y a pas de noeud qui peut atteindre |V| / 2
                        bool = True     # Donc nous devons recommencer 
                        continue        # On retourne a la boucle while avec bool = True
                        
                    list_path = []      # On remet la list des chemins a zero

                    list_graph_G = copy.deepcopy(list_graph_w)
                    list_graph_G.pop(len(list_graph_G) - 1)         # Ici on copie la liste des graphs G et H puis on retire H pour creer leur arbre

                    for current_graph in list_graph_G:
                        bg, arbre, iter = Graph.bellmanFord(current_graph, deg)
                        if bg == 0 and arbre == 0 and iter == 0: # Cycle negatif
                            bool = True # Recommencer
                            break
                        list_path.append(arbre)     # On ajoute les arbres des graphs G

                    bg, arbre, iter = Graph.bellmanFord(list_graph_w[len(list_graph_w) - 1], deg)       # Il faut aussi verifier que H n'a pas de circuit negatif
                    if bg == 0 and arbre == 0 and iter == 0:
                        bool = True

                    if bool:    
                        continue

                    # Si on a passer ces etapes il ne nous reste plus qu'a utiliser nos algorithmes pour comparer les differences d'iterations avec un ordre donne ou un ordre aleatoire

                    T = Graph.union_path(list_path)     # On fait l'union des plus courts chemins / arbres des G 

                    graph_T = Graph.from_tree_to_graph(T)       # On doit transformer cette union en un graph pour le donner a GloutonFas
                    
                    glouton_T = Graph.glouton_fas(graph_T)      # On trouve un ordre grace a GloutonFas

                    _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, glouton_T)      # On fait Bellman-Ford avec l'ordre donne par GloutonFas

                    ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])       # On prend un ordre aleatoire

                    _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, ordre_aleatoire)       # On fait Bellman-Ford avec l'ordre donne aleatoire


                    iteration_number_glouton[n-4] = iteration_number_glouton[n-4] + iter_glouton         # Ici a chaque fois n-4 par rapport au nombre de noeuds de bases : 4
                    iteration_number_alea[n-4] = iteration_number_alea[n-4] + iter_alea

                    # print("iter gl : ",iter_glouton)
                    # print("ite number gl :",iteration_number_glouton)
                    # print("iter alea : ",iter_alea)
                    # print("iteration_number_alea : ", iteration_number_alea)

                    graphs[n-4].append(graph)       # On garde en memoire les graphs aleatoires pour potentiellement les donnes a l'algorithme jumeau analyze_vertex_iteration_nb_with_graphs() pour qu'il existe ce meme algo mais avec des graphs predefinis

            
            list_iterations_vertex_mean_glouton = iteration_number_glouton / num_graphs_per_size
            list_iterations_vertex_mean_alea = iteration_number_alea / num_graphs_per_size
            # print("ite number gl apres boucle :",iteration_number_glouton)

        # Tracé du temps d'exécution en fonction de la taille du graphe (n)
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_alea, marker='o', label='Aléatoire')
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_glouton, marker='o', label='Glouton')
        plt.xlabel("Nombre de sommets du graphe (n)")
        plt.ylabel("Nombre moyen d'itérations")
        plt.title("Nombre moyen d'itérations de Bellman-Ford en fonction du nombre de sommets")
        plt.legend()
        plt.show()

        return graphs
 
    def analyze_vertex_iteration_nb_with_graphs(graphs, num_graphs_per_size, Nmax, p, nb_g, weight_interval):
        """
        Fonction qui trace un graphe avec en abscisse le nombre d'iterations et en ordonne le nombre des sommets pour un nombre donne des graphes de test avec des graphs donnes
        """
        bool = True

        if (Nmax <= 3):
            raise ValueError("Le parametre Nmax doit etre superieur a 3 pour avoir des tests pertinents")

        list_iterations_vertex_mean_glouton = []
        list_iterations_vertex_mean_alea = []
        iteration_number_glouton = np.zeros(Nmax - 3)
        iteration_number_alea = np.zeros(Nmax - 3)

        for n in range(4, Nmax + 1):
            for i in range(num_graphs_per_size):
                bool = True
                while bool:
                    bool = False

                    graph = graphs[n-4][i]
                    
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

                    bg, arbre, iter = Graph.bellmanFord(list_graph_w[len(list_graph_w) - 1], deg)       # Il faut aussi verifier que H n'a pas de circuit negatif
                    if bg == 0 and arbre == 0 and iter == 0:
                        bool = True

                    if bool:
                        continue

                    # print("graph ",n-4, i ,"\n Arcs = ",graph.E)

                    T = Graph.union_path(list_path)

                    graph_T = Graph.from_tree_to_graph(T)
                    
                    glouton_T = Graph.glouton_fas(graph_T)

                    _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, glouton_T)

                    ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])

                    _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, ordre_aleatoire)


                    iteration_number_glouton[n-4] = iteration_number_glouton[n-4] + iter_glouton
                    iteration_number_alea[n-4] = iteration_number_alea[n-4] + iter_alea

            
            list_iterations_vertex_mean_glouton = iteration_number_glouton / num_graphs_per_size
            list_iterations_vertex_mean_alea = iteration_number_alea / num_graphs_per_size

        # Tracé du temps d'exécution en fonction de la taille du graphe (n)
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_alea, marker='o', label='Aléatoire')
        plt.plot(range(4, Nmax + 1), list_iterations_vertex_mean_glouton, marker='o', label='Glouton')
        plt.xlabel("Nombre de sommets du graphe (n)")
        plt.ylabel("Nombre moyen d'itérations")
        plt.title("Nombre moyen d'itérations de Bellman-Ford en fonction du nombre de sommets")
        plt.legend()
        plt.show()        

        return


    def create_graph_by_level(nb_v, nb_level, weight_interval):
        """
        Fonction qui creer un graph par niveau avec nb_v le nombre de noeuds par niveau. Chaque noeuds du niveau j sont liees a chaque noeuds du niveau j+1
        """
        V = []
        E = {}

        num_v = 0

        for i in range(nb_level):       # On itere sur les niveaux
            for _ in range(nb_v):       # On creer nb_v noeuds
                V.append(num_v)
                num_v += 1

            if i != 0:
                for j in range(num_v-2*nb_v, num_v-nb_v):          # Niveau j
                    for k in range(num_v-nb_v, num_v):             # Niveau j+1
                        weight = random.randint(-weight_interval, weight_interval)
                        while weight == 0:          
                            weight = random.randint(-weight_interval, weight_interval)
                        if j not in E:
                            E[j] = set()        # On mets aretes vide a chaque noeud
                        E[j].add((k,weight))
        
        for i in range(num_v-nb_v, num_v):      # Les nb_v du dernier niveau n'a pas d'aretes il ne faut pas oublier de les mettres a vide
            E[i] = set()

        graph = Graph(V, E)

        return graph
    
    def pretraitement_methode(self, nb_g, weight_interval):
        """
        Fonction qui calcule le nombre d'iterations d'un graphe par niveau
        en generant nb_g graphes d'entrainement avec de poids differents
        """
        bool = True

        if (len(self.V) <= 3):
            raise ValueError("Le parametre Nmax doit etre superieur a 3 pour avoir des tests pertinents")
        
        bool = True
        while bool:         # Si bool passe a False ca signifie que l'on a trouve un noeud qui atteint |V|/2 et un bon graph de base avec des bons graphs d'entrainement G un bon graph de test H (qu'ils n'ont pas de cycle negatif)
            bool = False
            
            list_graph_w = []       # On remet les listes de G a zero
            list_weights = []       # On remet les listes des poids a zero
            for _ in range (nb_g):      # Creation des graphes ponderees
                                
                while True:         # Tant qu'on a pas break ca signifie qu'on a pris un poids weight deja utilise dans un autre graph G (Il est preferable de s'entrainer avec des poids differents sur les graphs G)
                    weight = random.randint(1, weight_interval)         # On choisi un poids aleatoire entre [1,weight_interval[
                    if weight not in (list_weights):                    # Est ce que ce poids a deja ete utilise pour un autre G ?
                        list_graph_w.append(Graph.weighed_graph(self, weight))
                        list_weights.append(weight)
                        break

            # Puis plus qu'a faire le graph de test H
            while True:
                weight = random.randint(1, weight_interval)
                if weight not in (list_weights):
                    graph_test_H = Graph.weighed_graph(self, weight)
                    list_graph_w.append(graph_test_H)
                    break

            deg = Graph.can_reach_half(self)        # On essaie de trouver un noeud qui peut atteindre |V| / 2

            if deg == None:     # Si deg est egal a None alors cela signifie qu'il n'y a pas de noeud qui peut atteindre |V| / 2
                bool = True     # Donc nous devons recommencer 
                continue        # On retourne a la boucle while avec bool = True
            
            print("deg : ", deg)

            print("Nous avons trouve un sommet qui peut acceder au moins |V|/2 autres sommets")
                        
            list_path = []      # On remet la list des chemins a zero

            list_graph_G = copy.deepcopy(list_graph_w)
            list_graph_G.pop(len(list_graph_G) - 1)         # Ici on copie la liste des graphs G et H puis on retire H pour creer leur arbre

            
            # Pas besoin de verifier s'il y a de cycle negatif dans ce cas
            for current_graph in list_graph_G:
                bg, arbre, iter = Graph.bellmanFord(current_graph, deg)
                list_path.append(arbre)

            if bool:
                continue

            # Si on a passer ces etapes il ne nous reste plus qu'a utiliser nos algorithmes pour comparer les differences d'iterations avec un ordre donne ou un ordre aleatoire

            print("Nous compilons le resultat")
            T = Graph.union_path(list_path)     # On fait l'union des plus courts chemins / arbres des G 

            print("Nous avons pu faire l'union d'arborescences des plus courts chemins")
            graph_T = Graph.from_tree_to_graph(T)       # On doit transformer cette union en un graph pour le donner a GloutonFas
            
            print("Nous avons fait la conversion de l'union en graphe")
            glouton_T = Graph.glouton_fas(graph_T)      # On trouve un ordre grace a GloutonFas
            print(glouton_T)
            _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, glouton_T)      # On fait Bellman-Ford avec l'ordre donne par GloutonFas

            ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])       # On prend un ordre aleatoire

            _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], deg, ordre_aleatoire)       # On fait Bellman-Ford avec l'ordre donne aleatoire

            print("Iterations avec ordre donne par GloutonFas: ", iter_glouton, "\nIterations avec ordre aleatoire: ", iter_alea)

            return
    
    def pretraitement_methode_graph(nb_g, weight_interval, levels, vertex_on_level, num_graphs_per_size):
        """
        Fonction qui trace un graphe avec en abscisse le nombre d'iterations
        et en ordonne le nombre de niveaux pour un nombre fixe des graphes_by_level de test
        """        

        bool = True
        iteration_number_glouton = np.zeros(levels//5)
        #iteration_number_alea = np.zeros(levels//5)
        list_iterations_vertex_mean_glouton = []
        #list_iterations_vertex_mean_alea = []
        cpt = 1

        for i in range(5, levels + 1, 5):
            
            for j in range(num_graphs_per_size):
                bool = True
                g = Graph.create_graph_by_level(vertex_on_level, i, weight_interval)

                while bool:
                    bool = False
                    
                    list_graph_w = []
                    list_weights = []
                    for _ in range (nb_g): # Creation des graphes ponderees
                        
                        while True:
                            weight = random.randint(1, weight_interval)
                            if weight not in (list_weights):
                                list_graph_w.append(Graph.weighed_graph(g, weight))
                                list_weights.append(weight)
                                break

                    while True:
                        weight = random.randint(1, weight_interval)
                        if weight not in (list_weights):
                            graph_test_H = Graph.weighed_graph(g, weight)
                            list_graph_w.append(graph_test_H)
                            break

                    list_path = []

                    list_graph_G = copy.deepcopy(list_graph_w)
                    list_graph_G.pop(len(list_graph_G) - 1)

                    for current_graph in list_graph_G:
                        bg, arbre, iter = Graph.bellmanFord(current_graph, 0) # on commence a partir du sommet 0, deg > v/2
                        list_path.append(arbre)

                    #pas besoin de tester un cycle negatif, car il n'y a pas dans ce cas

                    print("je compile le resultat")
                    T = Graph.union_path(list_path)

                    print("nous avons pu faire l'union d'arborescences des plus courts chemins")
                    #print(T)
                    graph_T = Graph.from_tree_to_graph(T)
                    
                    print("nous avons fait la conversion de l'union en graphe")
                    glouton_T = Graph.glouton_fas(graph_T)

                    print("nous avons execute gloutonfas")

                    _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], 0, glouton_T)

                    #ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])

                    #_, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], 0, ordre_aleatoire)

                    #print("Iterations avec ordre donne par GloutonFas: ", iter_glouton, "\nIterations avec ordre aleatoire: ", iter_alea)
                    print("Iterations avec ordre donne par GloutonFas: ", iter_glouton)
                    iteration_number_glouton[i-(5*cpt-(cpt-1))] = iteration_number_glouton[i-(5*cpt-(cpt-1))] + iter_glouton 
                    print(iteration_number_glouton)
                    #iteration_number_alea[i-(5*cpt-(cpt-1))] = iteration_number_alea[i-(5*cpt-(cpt-1))] + iter_alea
            
            list_iterations_vertex_mean_glouton = iteration_number_glouton / num_graphs_per_size
            #list_iterations_vertex_mean_alea = iteration_number_alea / num_graphs_per_size
            cpt += 1

        #plt.plot(range(5, levels + 1, 5), list_iterations_vertex_mean_alea, marker='o', label='Aléatoire')
        plt.plot(range(5, levels + 1, 5), list_iterations_vertex_mean_glouton, marker='o', label='Glouton')
        plt.xlabel("Nombre de niveaux")
        plt.ylabel("Nombre moyen d'itérations")
        plt.title("Nombre moyen d'itérations de Bellman-Ford en fonction du nombre de niveaux")
        plt.legend()
        plt.show()

    def pretraitement_methode_graph(nb_g, weight_interval, levels, vertex_on_level, num_graphs_per_size):
        """
        Fonction qui trace un graphe par niveau avec en abscisse le nombre d'iterations
        et en ordonne le nombre de niveaux pour un nombre fixe des graphes_by_level de test
        """        

        bool = True
        iteration_number_glouton = np.zeros(levels//5)
        iteration_number_alea = np.zeros(levels//5)
        list_iterations_vertex_mean_glouton = []
        list_iterations_vertex_mean_alea = []

        execution_time_glouton = np.zeros(levels//5)
        execution_time_alea = np.zeros(levels//5)
        list_time_vertex_mean_glouton = []
        list_time_vertex_mean_alea = []

        execution_time_treatment = np.zeros(levels//5)
        list_time_treatment = []
        cpt = 1

        for i in range(5, levels + 1, 5):
            
            for j in range(num_graphs_per_size):
                bool = True
                g = Graph.create_graph_by_level(vertex_on_level, i, weight_interval)

                while bool:
                    bool = False
                    
                    list_graph_w = []
                    list_weights = []
                    for _ in range (nb_g): # Creation des graphes ponderees
                        
                        while True:
                            weight = random.randint(1, weight_interval)
                            if weight not in (list_weights):
                                list_graph_w.append(Graph.weighed_graph(g, weight))
                                list_weights.append(weight)
                                break

                    while True:
                        weight = random.randint(1, weight_interval)
                        if weight not in (list_weights):
                            graph_test_H = Graph.weighed_graph(g, weight)
                            list_graph_w.append(graph_test_H)
                            break

                    list_path = []

                    list_graph_G = copy.deepcopy(list_graph_w)
                    list_graph_G.pop(len(list_graph_G) - 1)

                    for current_graph in list_graph_G:
                        bg, arbre, iter = Graph.bellmanFord(current_graph, 0) # on commence a partir du sommet 0, deg > v/2
                        list_path.append(arbre)

                    #pas besoin de tester un cycle negatif, car il n'y a pas dans ce cas

                    print("je compile le resultat")
                    start_treatment = time.time()
                    T = Graph.union_path(list_path)

                    print("nous avons pu faire l'union d'arborescences des plus courts chemins")
                    #print(T)
                    graph_T = Graph.from_tree_to_graph(T)
                    
                    print("nous avons fait la conversion de l'union en graphe")
                    glouton_T = Graph.glouton_fas(graph_T)
                    end_treatment = time.time()

                    print("nous avons execute gloutonfas")

                    start_gl = time.time()
                    _, _, iter_glouton = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], 0, glouton_T)
                    end_gl = time.time()

                    
                    ordre_aleatoire = Graph.random_order(list_graph_w[len(list_graph_w) - 1])
                    start_alea = time.time()
                    _, _, iter_alea = Graph.bellmanFord_gloutonFas(list_graph_w[len(list_graph_w) - 1], 0, ordre_aleatoire)
                    end_alea = time.time()

                    print("Iterations avec ordre donne par GloutonFas: ", iter_glouton, "\nIterations avec ordre aleatoire: ", iter_alea)
                    iteration_number_glouton[i-(5*cpt-(cpt-1))] = iteration_number_glouton[i-(5*cpt-(cpt-1))] + iter_glouton 
                    iteration_number_alea[i-(5*cpt-(cpt-1))] = iteration_number_alea[i-(5*cpt-(cpt-1))] + iter_alea

                    execution_time_glouton[i-(5*cpt-(cpt-1))] = execution_time_glouton[i-(5*cpt-(cpt-1))] + (end_gl - start_gl) 
                    execution_time_alea[i-(5*cpt-(cpt-1))] = execution_time_alea[i-(5*cpt-(cpt-1))] + (end_alea - start_alea)
                    
                    execution_time_treatment[i-(5*cpt-(cpt-1))] = execution_time_treatment[i-(5*cpt-(cpt-1))] + (end_treatment - start_treatment)
            
            list_iterations_vertex_mean_glouton = iteration_number_glouton / num_graphs_per_size
            list_iterations_vertex_mean_alea = iteration_number_alea / num_graphs_per_size

            list_time_vertex_mean_glouton = execution_time_glouton / num_graphs_per_size
            list_time_vertex_mean_alea = execution_time_alea / num_graphs_per_size

            list_time_treatment = execution_time_treatment / num_graphs_per_size
            cpt += 1

        x_values = np.array(range(5, levels + 1, 5))
        y_values = list_iterations_vertex_mean_glouton
        x_values2 = np.array(range(5, levels + 1, 5))
        y_values2 = list_iterations_vertex_mean_alea
        # Fit a linear regression line (y = mx + b)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        slope2, intercept2 = np.polyfit(x_values2, y_values2, 1)
        print('Pente glouton: ', slope)
        print('Pente alea: ', slope2)

        plt.plot(x_values2, y_values2, marker='o', label='Aléatoire')
        plt.plot(x_values, y_values, marker='o', label='Glouton')
        plt.plot(x_values, slope * x_values + intercept, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}', linestyle='--')
        plt.plot(x_values2, slope2 * x_values2 + intercept2, label=f'Fit: y = {slope2:.2f}x + {intercept2:.2f}', linestyle='--')
        plt.xlabel("Nombre de niveaux")
        plt.ylabel("Nombre moyen d'itérations")
        plt.title("Nombre moyen d'itérations de Bellman-Ford en fonction du nombre de niveaux")
        plt.legend()
        plt.show()

        x_values3 = np.array(range(5, levels + 1, 5))
        y_values3 = list_time_vertex_mean_alea
        x_values4 = np.array(range(5, levels + 1, 5))
        y_values4 = list_time_vertex_mean_glouton
        x_values5 = np.array(range(5, levels + 1, 5))
        y_values5 = list_time_treatment

        # Fit a linear regression line (y = mx + b)
        slope3, intercept3 = np.polyfit(x_values3, y_values3, 1)
        slope4, intercept4 = np.polyfit(x_values4, y_values4, 1)

        plt.plot(x_values3, y_values3, marker='o', label='Aléatoire')
        plt.plot(x_values4, y_values4, marker='o', label='Glouton')
        plt.plot(x_values3, slope3 * x_values3 + intercept3, label=f'Fit: y = {slope3:.8f}x + {intercept3:.8f}', linestyle='--')
        plt.plot(x_values4, slope4 * x_values4 + intercept4, label=f'Fit: y = {slope4:.8f}x + {intercept4:.8f}', linestyle='--')
        plt.xlabel("Nombre de niveaux")
        plt.ylabel("Temps moyen d'exécution")
        plt.title("Temps moyen d'exécution de Bellman-Ford en fonction du nombre de niveaux")
        plt.legend()
        plt.show()

        plt.plot(x_values5, y_values5, marker='o', label='Pretraitement')
        plt.xlabel("Nombre de niveaux")
        plt.ylabel("Temps moyen d'exécution du pretraitement")
        plt.legend()
        plt.show()

        #log log pour voir si polynomial
        log_sizes = np.log(x_values5)
        log_times = np.log(y_values5)
        slope5, intercept5 = np.polyfit(log_sizes, log_times, 1)
        plt.plot(x_values5, y_values5, marker='o', label='Pretraitement')
        plt.xscale("log")
        plt.yscale("log")
        print(f"Pente de la régression linéaire (log log): {slope5:.2f}")

        plt.xlabel("Nombre de niveaux")
        plt.ylabel("Temps moyen d'exécution du pretraitement")
        plt.legend()
        plt.show()
