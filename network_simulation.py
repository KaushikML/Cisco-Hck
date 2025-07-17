import random, string, networkx as nx, matplotlib.pyplot as plt

# --------------------------- parameters ---------------------------------
NUM_NODES = 10        # how many nodes in the network
EDGE_PROB  = 0.3      # edge‑creation probability (Erdős‑Rényi p)
SEED       = 42       # set to None for a new random network every run
# ------------------------------------------------------------------------

rnd = random.Random(SEED)

# 1) build a random undirected graph topology
G = nx.erdos_renyi_graph(NUM_NODES, EDGE_PROB, seed=SEED)
# guarantee the graph is connected (simple retry loop)
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(NUM_NODES, EDGE_PROB, seed=rnd.randint(0, 10**6))

# 2) relabel 0,1,2,… with A,B,C,… for nicer visuals
labels = dict(enumerate(string.ascii_uppercase))
G = nx.relabel_nodes(G, labels)

# 3) assign random node types and attributes
for n in G.nodes():
    ntype = rnd.choice(['quantum', 'classical'])
    G.nodes[n]['type'] = ntype
    G.nodes[n]['can_store_entanglement'] = (ntype == 'quantum')

# 4) draw
color_map = [
    'skyblue' if G.nodes[n]['type'] == 'quantum' else 'lightgreen'
    for n in G.nodes
]
pos = nx.spring_layout(G, seed=SEED)
nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=800)
plt.title('Randomised Hybrid Quantum‑Classical Network')
plt.tight_layout()
plt.savefig('hybrid_network.png')
plt.show()
