# 🔍 Social Network Analysis Tool

## 📚 Overview
As part of an academic assignment, I developed a project focused on analyzing social networks using graph theory and Python.

## 🚀 Features
- Analyze various social network datasets
- Generate adjacency matrices
- Identify network leaders and followers
- Visualize social network graphs
- Implement breadth-first search (BFS) for path finding

## 📂 Datasets
The project includes four datasets of increasing complexity:
1. **Toy Dataset**: A small example network
2. **Karate Club Dataset**: Social network of a karate club
3. **Student Cooperation Dataset**: Network of student interactions
4. **Anybeat Dataset**: Large online community network

## 🧰 Requirements
- Python 3.x
- NumPy
- NetworkX
- Matplotlib

## 🛠️ Installation
1. Clone the repository
```bash
git clone https://github.com/Mariussgal/Graph-Theory.git
cd Graph-Theory
```

2. Install required dependencies
```bash
pip install numpy networkx matplotlib
```

## ▶️ Usage
Run the main script to interact with the analysis tool:
```bash
python main.py
```

### 📋 Menu Options
1. Load and analyze Toy dataset
2. Load and analyze Karate Club dataset
3. Load and analyze Student cooperation dataset
4. Load and analyze Anybeat dataset
0. Quit

## 🗂️ Project Structure
```
project-root/
│
├── data/
│   ├── exemple.txt
│   ├── club.txt
│   ├── students.csv
│   └── ...
│
├── src/
│   └── graph_utils.py
│
├── graphs/
│   └── (generated graph visualizations)
│
├── main.py
└── README.md
```

## 🔧 Key Functions
- `edge_list()`: Parse network edge data from files
- `create_adjacency_matrix()`: Generate adjacency matrix representation
- `find_leaders()`: Identify most connected nodes
- `find_followers()`: Find nodes with most outgoing connections
- `bfs_shortest_path()`: Find shortest path between nodes
- `draw_graph()`: Visualize network with highlighted leaders

## 🎯 Learning Objectives
- Master basic graph concepts
- Implement graph analysis algorithms
- Visualize complex network structures


## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repo and submit a pull request.
