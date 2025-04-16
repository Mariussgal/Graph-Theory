import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def edge_list(file_path):
    edges = []
    nodes = set()
    
    with open(file_path, 'r') as file:      
        first_line = file.readline().strip()

        if first_line.lower().replace(' ', '').replace(',', '') in ['id1id2', 'sourcetarget', 'fromto', 'frompersonidtopersonid', 'id1id2type']:
            print(f"header ignored: {first_line}")
        else:    
            try:
                if ',' in first_line:
                    source, target = map(int, first_line.split(','))
                else:
                    source, target = map(int, first_line.split())
                
                edges.append((source, target))
                nodes.add(source)
                nodes.add(target)
            except ValueError as e:
                print(f"header ignored: {first_line}  - {e}")

        for line in file:
            try:
                if ',' in line:
                    source, target = map(int, line.strip().split(','))
                else:
                    source, target = map(int, line.strip().split())
                
                edges.append((source, target))
                nodes.add(source)
                nodes.add(target)
            except ValueError as e:
                print(f"line ignored: {line.strip()} - {e}")
    return edges, sorted(list(nodes))

def create_adjacency_matrix(edges, nodes):
    n = len(nodes)
    node_to_index = {nodes[i]: i for i in range(n)}
    adj_matrix = np.zeros((n, n), int)
    
    for source, target in edges:
        adj_matrix[node_to_index[source], node_to_index[target]] = 1
    
    return adj_matrix, node_to_index

def save_adjacency_matrix(adj_matrix, file_path, nodes):  
    with open(file_path, 'w') as file:
        file.write('\t' + '\t'.join(map(str, nodes)) + '\n')

        for i, node in enumerate(nodes):
            file.write(str(node) + '\t' + '\t'.join(map(str, adj_matrix[i])) + '\n')

def find_leaders(adj_matrix, nodes, k=2):
    in_edges = np.sum(adj_matrix, axis=0)
    leaders_indices = np.argsort(-in_edges)[:k]
    leaders = [nodes[i] for i in leaders_indices]
    
    return leaders

def find_followers(adj_matrix, nodes, k=2):
    out_edges = np.sum(adj_matrix, axis=1)
    followers_indices = np.argsort(-out_edges)[:k]
    followers = [nodes[i] for i in followers_indices]
    
    return followers

def bfs_shortest_path(adj_matrix, node_to_index, start, end, nodes):
    start_idx = node_to_index[start]
    end_idx = node_to_index[end]
    n = len(nodes)
    
    color = {i: 'white' for i in range(n)}
    parent = {i: None for i in range(n)}
    color[start_idx] = 'grey'
    history = [start_idx]  
    
    if start_idx == end_idx:
        return [start]
    
    while history:
        current_idx = history[0]
        
        for neighbor_idx in range(n):
            if adj_matrix[current_idx, neighbor_idx] == 1 and color[neighbor_idx] == 'white':
                color[neighbor_idx] = 'grey'
                parent[neighbor_idx] = current_idx
                history.append(neighbor_idx)
                
                if neighbor_idx == end_idx:
                    path = [end_idx]
                    while path[0] != start_idx:
                        path.insert(0, parent[path[0]])  
                    return [nodes[idx] for idx in path]    
        history.pop(0)
        color[current_idx] = 'black'
    return []

def draw_graph(edges, nodes, leaders, title="Graph - Leaders in red", output_file = "graph.png"):
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    non_leaders = [node for node in nodes if node not in leaders]
    nx.draw_networkx_nodes(G, pos, nodelist=non_leaders, node_color='lightblue', node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=leaders, node_color='red', node_size=700, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

output_directory = "C:\\Users\\mariu\\PBL3\\graphs"

def analyze_toy_dataset():

    file_path = os.path.join("data", "exemple.txt")
    output_matrix = os.path.join("data", "toy_adjacency_matrix.txt")
    
    try:
        print("\n=== Analysis of toy dataset ===")
        
        edges, nodes = edge_list(file_path)
        print(f"Number of nodes: {len(nodes)}")
        print(f"number of edges: {len(edges)}")
        
        adj_matrix, node_to_index = create_adjacency_matrix(edges, nodes)
        print("Adjacency matrix created:")
        
        save_adjacency_matrix(adj_matrix, output_matrix, nodes)
        print(f"Adjacency matrix saved in '{output_matrix}'")
        
        leaders = find_leaders(adj_matrix, nodes, k=2)
        print(f"The 2 most important leaders are: {leaders}")

        best_followers = find_followers(adj_matrix, nodes, k=2)
        print(f"The 2 most important followers are: {best_followers}")
        
        path = bfs_shortest_path(adj_matrix, node_to_index, leaders[0], leaders[1], nodes)
        print(f"Shortest path between leaders {leaders[0]} and {leaders[1]} is: {path}")
        
        draw_graph(edges, nodes, leaders, "Toy dataset graph - Leaders in red", output_file=os.path.join(output_directory, "toy_graph.png"))
        
        print("\n=== Toy dataset Analysis done ===\n")
        return True
        
    except FileNotFoundError:
        print(f"File '{file_path}'not found.")
        return False
    except Exception as e:
        print(f"An error occured: {e}")
        return False

def analyze_karate_club_dataset():

    file_path = os.path.join("data", "club.txt")
    output_matrix = os.path.join("data", "karate_adjacency_matrix.txt")
    
    try:
        print("\n=== Analysis of Karate Club dataset ===")

        edges, nodes = edge_list(file_path)
        
        bidirectional_edges = edges.copy()
        for source, target in edges:
            if (target, source) not in bidirectional_edges:
                bidirectional_edges.append((target, source))
        
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of edges (bidirectionnelles): {len(bidirectional_edges)}")
        
        adj_matrix, node_to_index = create_adjacency_matrix(bidirectional_edges, nodes)
        print("Adjacency matrix created:")
        
        save_adjacency_matrix(adj_matrix, output_matrix, nodes)
        print(f"Adjacency matrix saved in: '{output_matrix}'")
        
        leaders = find_leaders(adj_matrix, nodes, k=2)
        print(f"2 most important leaders are: {leaders}")

        best_followers = find_followers(adj_matrix, nodes, k=2)
        print(f"2 nodes with the most conection: {best_followers}")
        
        if len(leaders) >= 2:
            path = bfs_shortest_path(adj_matrix, node_to_index, leaders[0], leaders[1], nodes)
            print(f"shortest path between leaders {leaders[0]} and {leaders[1]} is: {path}")
        
        draw_graph(bidirectional_edges, nodes, leaders, "Karate Club graph - Leaders in red", output_file=os.path.join(output_directory, "karate_graph.png"))
        
        print("\n=== Analysis of Karate Club dataset done ===\n")
        return True
        
    except FileNotFoundError:
        print(f"file '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"an error occured: {e}")
        return False
    

def analyze_student_dataset():
    file_path = os.path.join("data", "students.csv")
    output_matrix = os.path.join("data", "students_adjacency_matrix.txt")
    
    try:
        print("\n=== Analysis of Student Cooperation dataset ===")

        edges = []
        nodes = set()  #set() means that it deeletes all the multiple iterations of a number to keep only one of them
        edge_types = defaultdict(list)

        with open(file_path, 'r') as file: 
            first_line = file.readline().strip()
            if first_line.lower().replace(' ','').replace(',','') in ['id1id2type', 'sourcetargettype']:
                print (f"Header found and ignored: {first_line}")
            for line in file:
                try:
                    parts = line.strip().split(',')
                    if len(parts) ==3:
                        source, target, type = parts
                        source, target = int(source), int(target) 

                        edges.append((source, target))
                        nodes.add(source)
                        nodes.add(target)
                        edge_types[(source, target)].append(edge_types)
                    else:
                        print(f"Line ignored (wrong format):{line.strip()}")
                except ValueError as e: 
                    print(f"Line ignored: {line.strip()} - {e}")
        
        nodes = sorted(list(nodes))
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of edges: {len(edges)}")

        print("=== Analysis of Student Cooperation dataset done ===\n")
        return True
    except FileNotFoundError:
        print(f"file '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"an error occured: {e}")
        return False




def analyze_anybeatAnonymized_dataset():
    
    file_path = os.path.join("data", "anybeatAnonymized.csv")
    output_matrix = os.path.join("data", "anybeatAnonymized_adjacency_matrix.txt")

 
    

