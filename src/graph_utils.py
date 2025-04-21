import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import csv 
from scipy import sparse

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
    
    input_file = ["students.csv"]
    
    output_matrix = os.path.join("data","students_adjacency_matrix.txt")
    
    print("\n=== Analysis of Students dataset ===\n")
    
    for filename in input_file:
        if os.path.exists(filename):
            input_file = filename
            break
        
        data_path = os.path.join("data", filename)
        if os.path.exists(data_path):
            input_file = data_path
            break
    
    if input_file is None:
        print("Error: Could not find student data file")
        return
    
    try:
        edges = []
       
        with open(input_file, 'r') as file:
            csv_reader = csv.reader(file)
            try:
                next(csv_reader)
            except StopIteration:
                print("Warning: CSV file appears to be empty")
                return
                
            for row in csv_reader:
                if len(row) >= 2:
                    id1 = int(row[0])
                    id2 = int(row[1])
                    link_type = row[2] 
                    edges.append((id1, id2, link_type))

        nodes = set()
        for id1, id2, _ in edges:
            nodes.add(id1)
            nodes.add(id2)
        nodes = sorted(list(nodes))
        
        print(f"Number of nodes: {len(nodes)}")            
        print(f"Number of edges: {len(edges)}")
        
        if not edges:
            print("Error: No edges found in the input file")
            return
        
        link_types = {}
        for _, _, link_type in edges:
            link_types[link_type] = link_types.get(link_type, 0) + 1
        
        print("Number of types:")
        for link_type, count in link_types.items():
            print(f"- {link_type}: {count} edges")   
        
        n = len(nodes)
        node_to_index = {node: i for i, node in enumerate(nodes)}

        adj_matrix = np.zeros((n, n), dtype=int)

        for id1, id2, _ in edges:
            i = node_to_index[id1]
            j = node_to_index[id2]
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
   
        print(f"\nAdjacency matrix created:")
        try:
            results_dir = os.path.dirname(output_matrix)
            if results_dir and not os.path.exists(results_dir):
                print(f"Creating directory: {results_dir}")
                os.makedirs(results_dir, exist_ok=True)
            
            with open(output_matrix, 'w') as file:
                file.write(',' + ','.join(map(str, nodes)) + '\n')
                
                for i, node in enumerate(nodes):
                    row_values = ','.join(map(str, adj_matrix[i]))
                    file.write(f"{node},{row_values}\n")
            
            print(f"Adjacency matrix saved in {output_matrix}")

        except Exception as e:
                print(f"Error details: {e}")

        leaders = find_leaders(adj_matrix, nodes, k=5)
        print(f"5 most important leaders are: {leaders}")

        best_followers = find_followers(adj_matrix, nodes, k=5)
        print(f"5 nodes with the most conection: {best_followers}")

        if len(best_followers) >= 2:
            path = bfs_shortest_path(adj_matrix, node_to_index, best_followers[0], best_followers[1], nodes)
            print(f"Shortest path between best followers {best_followers[0]} and {best_followers[1]} is: {path}")
   
        graph_edges = [(id1, id2) for id1, id2, _ in edges]
        
        output_file = os.path.join(output_directory, "students_graph.png")
        
        draw_graph(graph_edges, nodes, leaders,"Student cooperation graph - Leaders in red", output_file=os.path.join(output_directory, "Students_graph.png"))
        print(f"Graph visualization saved in {output_file}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")

    return True

def analyze_anybeatAnonymized_dataset():
    import numpy as np
    from collections import defaultdict, deque
    import random
    
    input_file = ["anybeatAnonymized.csv"]
    output_matrix = os.path.join("data", "anybeat_adjacency_matrix.txt")
    
    print("\n=== Analysis of Anybeat dataset ===\n")
    
    for filename in input_file:
        if os.path.exists(filename):
            input_file = filename
            break
        
        data_path = os.path.join("data", filename)
        if os.path.exists(data_path):
            input_file = data_path
            break
    
    if input_file is None:
        print("Error: Could not find Anybeat data file")
        return
    
    try:
        edges = []
        node_connections = defaultdict(set)  
        
        with open(input_file, 'r') as file:
            csv_reader = csv.reader(file)
            try:
                header = next(csv_reader)  
                print(f"Header ignored: {header}")
            except StopIteration:
                print("Warning: CSV file appears to be empty")
                return
  
            for row in csv_reader:
                        id1 = int(row[0])
                        id2 = int(row[1])
                        edges.append((id1, id2))
                        node_connections[id1].add(id2)
                        node_connections[id2].add(id1)
                        
        nodes = sorted(node_connections.keys())
        
        print(f"Number of nodes: {len(nodes)}")            
        print(f"Number of edges: {len(edges)}")
        
        if not edges:
            print("Error: No edges found in the input file")
            return
        
        node_degrees = [(node, len(connections)) for node, connections in node_connections.items()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        sample_size = 500
        sample_nodes = [node for node, _ in node_degrees[:sample_size]]
        
        node_to_index = {node: i for i, node in enumerate(sample_nodes)}
        
        print(f"Creating adjacency matrix for {sample_size} most connected nodes")
        adj_matrix = np.zeros((sample_size, sample_size), dtype=int)
        edge_count = 0
           
        for i, node1 in enumerate(sample_nodes):
            for node2 in node_connections[node1]:
                j = node_to_index.get(node2)
                if j is not None:  
                    adj_matrix[i, j] = 1
                    edge_count += 1
        print(f"Filled adjacency matrix with {edge_count} connections")
        print(f"Adjacency matrix created:")
      
        try:
            with open(output_matrix, 'w') as file:
              
                file.write("," + ",".join(str(node) for node in sample_nodes) + "\n")
                
                for i, node in enumerate(sample_nodes):
                    file.write(f"{node}," + ",".join(map(str, adj_matrix[i, :])) + "\n")
            print(f"Adjacency matrix saved in '{output_matrix}'")
            
        except Exception as e:
            print(f"Error saving matrix: {e}")
        
       
        matrix_degrees = []
        for i, node in enumerate(sample_nodes):
            degree = np.sum(adj_matrix[i, :])
            matrix_degrees.append((node, degree))
        
        matrix_degrees.sort(key=lambda x: x[1], reverse=True)
        
        matrix_leaders = [node for node, _ in matrix_degrees[:5]]
        matrix_leader_degrees = [degree for _, degree in matrix_degrees[:5]]
        
        print("The 5 most important leaders are:")
        for i, (leader, degree) in enumerate(zip(matrix_leaders, matrix_leader_degrees)):
            print(f"{i+1}. Node {leader} with {int(degree)} connections in matrix")
        
        leaders = matrix_leaders  
        
        print("\nIdentifying followers of each leader from adjacency matrix...")
        for leader in leaders:
            i = node_to_index[leader]
            follower_indices = np.where(adj_matrix[i, :] == 1)[0]
            followers = [sample_nodes[j] for j in follower_indices]

            display_followers = followers[:5]
            print(f"Leader {leader} has {len(followers)} followers in matrix, including: {display_followers}")
        
    
        if len(leaders) >= 2:
            print("\nFinding most important path between top leaders:")
            
            def bfs_shortest_path(adj_matrix, start_idx, end_idx, n):
                queue = deque([(start_idx, [start_idx])])
                visited = set([start_idx])
                
                while queue:
                    (node, path) = queue.popleft()
                    neighbors = np.where(adj_matrix[node, :] == 1)[0]
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            if neighbor == end_idx:
                                return path + [neighbor]
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                return None
        
            leader1_idx = node_to_index[leaders[0]]
            leader2_idx = node_to_index[leaders[1]]
            
            path_indices = bfs_shortest_path(adj_matrix, leader1_idx, leader2_idx, sample_size)
            
            if path_indices:
                important_path = [sample_nodes[idx] for idx in path_indices]
                print(f"Most important path between leaders {leaders[0]} and {leaders[1]} is: {important_path}")
            else:
                print(f"No path exists between leaders {leaders[0]} and {leaders[1]}")
                important_path = []
        else:
            important_path = []
            print("Not enough leaders to find a path")

        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            print("\nCreating visualization of leaders and important path:")
           
            G = nx.Graph()
            G.add_nodes_from(sample_nodes)
            
            for i in range(sample_size):
                for j in range(sample_size):
                    if adj_matrix[i, j] == 1:
                        G.add_edge(sample_nodes[i], sample_nodes[j])
            
         
            plt.figure(figsize=(15, 15))
            nodes_to_draw = set(important_path + leaders)
            other_nodes = [node for node in sample_nodes if node not in nodes_to_draw]
            if len(other_nodes) > 50:  
                other_nodes = random.sample(other_nodes, 50)
            nodes_to_draw.update(other_nodes)
            
            subgraph = G.subgraph(nodes_to_draw)
            pos = nx.spring_layout(subgraph, seed=42)
            regular_nodes = [node for node in subgraph.nodes() if node not in leaders and node not in important_path]
            
            nx.draw_networkx_nodes(subgraph, pos, nodelist=regular_nodes, node_color='lightblue', node_size=300, alpha=0.7)
            
            path_nodes = [node for node in important_path if node not in leaders]
            if path_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=path_nodes, node_color='green', node_size=500, alpha=0.8)
            
            nx.draw_networkx_nodes(subgraph, pos, nodelist=leaders, node_color='red', node_size=700, alpha=0.9)

            path_edges = []
            if len(important_path) > 1:
                path_edges = [(important_path[i], important_path[i+1]) for i in range(len(important_path)-1)]

            other_edges = [(u, v) for u, v in subgraph.edges() if (u, v) not in path_edges and (v, u) not in path_edges]
            
            nx.draw_networkx_edges(subgraph, pos, edgelist=other_edges, width=1, alpha=0.5, edge_color='gray')

            if path_edges:
                nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, width=3, alpha=1.0, edge_color='green')

            labels = {node: str(node) for node in subgraph.nodes()}            
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=10, font_weight='bold')
            
            plt.title("Anybeat graph - Leaders (red), Important Path (green)")
            plt.axis('off')
            plt.tight_layout()
            
            output_file = os.path.join(output_directory, "anybeat_graph.png")
            plt.savefig(output_file, dpi=300)
            print(f"graph saved in: {output_file}")
        
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    return True