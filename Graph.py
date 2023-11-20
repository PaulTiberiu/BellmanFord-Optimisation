import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
from scipy.stats import linregress

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

        print("s1: ", s1)
        print("s2: ", s2)

        return s1 + s2
                        

    def random_graph(self, n, p):
        """
        Cree un graphe oriente de n sommets sans cycles, où chaque arête (i, j) est présente avec la probabilité p
        """

        if n < 0:
            raise ValueError("Le parametre n doit etre superieur a 0")

        if p < 0 or p > 1:
            raise ValueError("Le parametre p doit etre entre 0 et 1")

        vertices = set(range(n))
        graph = Graph(vertices)

        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p and not graph.has_cycle(j, i):
                    weight = random.randint(-10, 10)  # Ajoute un poids entre -10 et 10
                    graph.insert_edge(i, j, weight)

        return graph


#nb d'iterations qu'il prend, pour chaque sommet, de determiner le plus court chemin => stocker ca dans une liste
#a faire en fonction de l'algo glouton et l'ordre et a comparer avec belman classique comme premier test
#a faire une fonction qui va tracer un graphe en fonction du nombre d'iterations pour trouver le plus court chemin (en fonction des sauts(cb de sauts il doit faire))
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

        iterations = 0

        for _ in range(len(self.V) - 1):                    # On itere sur tous les sommets
            for y in self.E:                                # On verifie chaque arcs entrants
                for v, w in self.E[y]:
                    if distances[y] + w < distances[v]:
                        distances[v] = distances[y] + w
                        predecesseurs[v] = y
            iterations += 1

        # On verifie qu'il n'y a pas de Cycle negatif
        for y in self.E:
            for v, w in self.E[y]:
                if distances[v] > distances[y] + w:
                    print("Cycle negatif")
                    return

                
        paths = {}
        for p in predecesseurs:         # On regarde le chemin pour chaque sommet du graph
            path = []
            a = p
            while p != start:           # Tant qu'on arrive pas au sommet de depart:
                path.append(p)          # On ajoute le sommet au chemin
                p = predecesseurs[p]    # Puis on passe a son predecesseur
            path.append(p)
            paths[a] = path[::-1]       # On retourne la liste

        return distances, paths, iterations
    

    def bellmanFordMVP(self, start):
        """
        Retourne les distances et les chemins des plus court chemins de start vers les sommets du graph
        """

        distances = {}
        predecesseurs = {}
        
        for v in self.V:
            distances[v] = np.inf       # On fixe dv = infini
            predecesseurs[v] = None
        distances[start] = 0

        iterations = 0

        ordre = self.glouton_fas()

        for _ in range(len(self.V) - 1):                    # On itere sur tous les sommets
            for y in ordre:
                if y in self.E:                                # On verifie chaque arcs entrants
                    for v, w in self.E[y]:
                        if distances[y] + w < distances[v]:
                            distances[v] = distances[y] + w
                            predecesseurs[v] = y
            iterations += 1

        # On verifie qu'il n'y a pas de Cycle negatif
        for y in self.E:
            for v, w in self.E[y]:
                if distances[v] > distances[y] + w:
                    print("Cycle negatif")
                    return

                
        paths = {}
        for p in predecesseurs:         # On regarde le chemin pour chaque sommet du graph
            path = []
            a = p
            while p != start:           # Tant qu'on arrive pas au sommet de depart:
                path.append(p)          # On ajoute le sommet au chemin
                p = predecesseurs[p]    # Puis on passe a son predecesseur
            path.append(p)
            paths[a] = path[::-1]       # On retourne la liste

        return distances, paths, iterations