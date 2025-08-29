import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


edges = np.loadtxt("djr-all/BH_fromtobus.csv", delimiter=",", dtype=int)
genidx = np.loadtxt("djr-all/BH_genidx.csv", delimiter=",", dtype=int)

# Create directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Layout (spring_layout spreads nodes to reduce overlap)
# pos = nx.spring_layout(G, seed=42)  # deterministic layout
pos = nx.planar_layout(G)  # planar layout

# Node styling
node_colors = []
node_sizes = []
for node in G.nodes():
    if node in genidx:
        node_colors.append("red")
        node_sizes.append(50)
    else:
        node_colors.append("lightblue")
        node_sizes.append(25)

# Edge styling
edge_colors = "gray"

# Draw graph
plt.figure(figsize=(6, 5))
nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       edgecolors="black")
nx.draw_networkx_edges(G, pos,
                       edge_color=edge_colors,
                       arrows=False,
                       arrowsize=15,
                       width=2)
# nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

plt.axis("off")
plt.tight_layout()
plt.show()