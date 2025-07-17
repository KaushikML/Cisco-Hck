import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math

# --- Constants and Simulation Parameters ---
L0 = 2.5  # Characteristic decoherence length
TRANSLATION_COST = 0.1 # Cost penalty for switching between quantum/classical domains

# --- Core Network and Protocol Functions from your script ---

def create_hybrid_network(num_nodes=12, quantum_ratio=0.4, seed=42):
    """Creates a hybrid network graph."""
    G = nx.erdos_renyi_graph(num_nodes, 0.4, seed=seed)
    # Ensure there are at least two classical nodes for routing
    num_quantum = int(num_nodes * quantum_ratio)
    if num_nodes - num_quantum < 2:
        num_quantum = max(0, num_nodes - 2)

    quantum_nodes = random.sample(list(G.nodes), num_quantum)
    for node in G.nodes:
        G.nodes[node]['type'] = 'quantum' if node in quantum_nodes else 'classical'
    return G

def assign_link_behaviors(G, pos):
    """Annotates all edges with realistic parameters."""
    for u, v in G.edges():
        distance = float(np.hypot(*(np.array(pos[u]) - np.array(pos[v]))))
        utype, vtype = G.nodes[u]["type"], G.nodes[v]["type"]

        if utype == 'quantum' and vtype == 'quantum':
            G.edges[u, v].update({
                "type": "quantum", "distance": distance,
                "decoherence_prob": 1 - np.exp(-distance / L0),
                "entanglement_success": max(0.05, np.exp(-0.3 * distance)),
            })
        elif utype == 'classical' and vtype == 'classical':
            G.edges[u, v].update({
                "type": "classical", "distance": distance,
                "latency_ms": random.uniform(1, 10),
                "packet_loss": random.uniform(0.01, 0.05),
            })
        else: # Hybrid links
             G.edges[u, v].update({
                "type": "hybrid", "distance": distance,
                "latency_ms": random.uniform(2, 15),
                "fidelity": random.uniform(0.7, 0.99), # C->Q
                "packet_loss": random.uniform(0.02, 0.08), # Q->C
            })
    return G

def get_edge_success_prob(attrs):
    """Calculates the success probability for a single edge."""
    etype = attrs["type"]
    if etype == "quantum":
        return (1 - attrs["decoherence_prob"]) * attrs["entanglement_success"]
    elif etype == "classical":
        return 1 - attrs["packet_loss"]
    elif etype == "hybrid":
        # Simplified for demonstration; a real model would be more complex
        return attrs.get("fidelity", 0.95) * (1 - attrs.get("packet_loss", 0.05))
    return 0.0

def has_qkd_segment(G, path):
    """Checks if a path contains at least one quantum-quantum link."""
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.nodes[u]["type"] == "quantum" and G.nodes[v]["type"] == "quantum":
            return True
    return False

def classify_path(G, path):
    """Categorizes a path based on its links."""
    if has_qkd_segment(G, path):
        return "Quantum-QKD"
    
    has_hybrid = any("hybrid" in G.edges[path[i], path[i+1]].get("type", "") for i in range(len(path) - 1))
    if has_hybrid:
        return "Hybrid"
    return "Classical"

def compute_path_metrics(G, path):
    """Computes all relevant metrics for a given path."""
    total_success = 1.0
    total_distance = 0.0
    translations = 0
    prev_domain = G.nodes[path[0]]["type"]

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        attrs = G.edges[u, v]
        total_success *= get_edge_success_prob(attrs)
        total_distance += attrs["distance"]
        
        curr_domain = G.nodes[v]["type"]
        if curr_domain != prev_domain:
            translations += 1
        prev_domain = curr_domain
        
    return len(path) - 1, total_distance, total_success, translations

def path_cost(hops, distance, reliability, translations, weights):
    """Calculates the final cost score for a path."""
    return (weights['w1'] * hops) + (weights['w2'] * distance) - (weights['w3'] * reliability) + (translations * TRANSLATION_COST)

def choose_best_path(G, paths, weights):
    """Selects the path with the lowest cost from a list of candidates."""
    best_path, best_score = None, float("inf")
    for path in paths:
        hops, dist, success, translations = compute_path_metrics(G, path)
        cost = path_cost(hops, dist, success, translations, weights)
        if cost < best_score:
            best_score, best_path = cost, path
    return best_path

def aqfr_route(G, src, dst, weights, cutoff=8):
    """Main AQFR routing algorithm."""
    try:
        all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=cutoff))
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return None, "No Path"

    if not all_paths:
        return None, "No Path"

    qkd_paths, hybrid_paths, classical_paths = [], [], []
    for p in all_paths:
        ptype = classify_path(G, p)
        if ptype == "Quantum-QKD":
            qkd_paths.append(p)
        elif ptype == "Hybrid":
            hybrid_paths.append(p)
        else:
            classical_paths.append(p)

    if qkd_paths:
        return choose_best_path(G, qkd_paths, weights), "Quantum-QKD"
    elif hybrid_paths:
        return choose_best_path(G, hybrid_paths, weights), "Hybrid"
    elif classical_paths:
        return choose_best_path(G, classical_paths, weights), "Classical"
    return None, "No Path"

def visualize_network(G, pos, best_path=None, path_type=None):
    """Draws the network graph and highlights the selected path."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    edge_color_map = {"quantum": "cyan", "classical": "gray", "hybrid": "purple"}
    edge_colors = [edge_color_map.get(G.edges[e].get("type", "classical"), "gray") for e in G.edges]


    nx.draw(G, pos, with_labels=True,
            node_color=["skyblue" if d["type"] == "quantum" else "orange" for _, d in G.nodes(data=True)],
            edge_color=edge_colors, width=1.5, node_size=2500, font_size=10, font_weight='bold', ax=ax)

    if best_path:
        path_edges = list(zip(best_path, best_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=4, ax=ax)
        ax.set_title(f"AQFR Selected Path ({path_type})", fontsize=16)
    else:
        ax.set_title("Hybrid Network Topology", fontsize=16)
        
    return fig

# --- Function to generate and store the network in session state ---
def generate_and_store_network(num_nodes, quantum_ratio, seed):
    """Generates a new network and stores it and its layout in the session state."""
    G = create_hybrid_network(num_nodes, quantum_ratio, seed)
    pos = nx.spring_layout(G, seed=seed)
    G = assign_link_behaviors(G, pos)
    st.session_state.G = G
    st.session_state.pos = pos
    st.session_state.best_path = None
    st.session_state.category = None


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🛰️ AQFR: Adaptive Quantum-First Routing Simulator")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🌐 Network Generation")
    seed = st.number_input("Random Seed", value=42)
    num_nodes = st.slider("Total Nodes", 5, 30, 12)
    quantum_ratio = st.slider("Quantum Node Ratio", 0.1, 1.0, 0.4)

    if st.button("Generate New Network"):
        generate_and_store_network(num_nodes, quantum_ratio, seed)
        
    # Initialize state on first run
    if 'G' not in st.session_state:
        generate_and_store_network(num_nodes, quantum_ratio, seed)

    G = st.session_state.G
    pos = st.session_state.pos
    
    # Filter for classical nodes for source/destination selection
    classical_node_list = sorted([n for n, d in G.nodes(data=True) if d['type'] == 'classical'])

    st.header("⚙️ Routing Controls")
    
    if len(classical_node_list) < 2:
        st.warning("Not enough classical nodes for routing. Please generate a new network with a lower quantum ratio.")
        source_node, dest_node = None, None
    else:
        source_node = st.selectbox("Select Source Node (Classical Only)", classical_node_list, index=0)
        dest_node = st.selectbox("Select Destination Node (Classical Only)", classical_node_list, index=len(classical_node_list)-1)


    st.subheader("Protocol Weights")
    w1 = st.slider("Hop Count Weight", 0.0, 5.0, 1.0)
    w2 = st.slider("Distance Weight", 0.0, 1.0, 0.01, format="%.3f")
    w3 = st.slider("Reliability Weight", 0.0, 10.0, 2.0)
    w4 = st.slider("Decoherence Probability", 0.0, 1.0, 0.1, format="%.2f")
    w5 = st.slider("Entanglement success probability", 0.0, 1.0, 0.5, format="%.2f")
    
    weights = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5}

# --- Main Display Area ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📍 Routing Results")
    
    best_path = None
    category = None

    if source_node is not None and dest_node is not None and source_node != dest_node:
        best_path, category = aqfr_route(G, source_node, dest_node, weights)

        if best_path:
            st.success(f"Found best path in category: **{category}**")
            hops, dist, success, translations = compute_path_metrics(G, best_path)
            
            st.write(f"**Path:** `{' -> '.join(map(str, best_path))}`")
            st.metric("Hop Count", hops)
            st.metric("Total Distance", f"{dist:.2f}")
            st.metric("End-to-End Success Probability", f"{success:.2%}")
            st.metric("Domain Switches (Q <-> C)", translations)

            with st.expander("View Hop-by-Hop Details"):
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    attrs = G.edges[u,v]
                    edge_success = get_edge_success_prob(attrs)
                    st.text(f"{u} -> {v} [{attrs['type']}]: Success Prob = {edge_success:.3f}")
        else:
            st.error("No path found between the selected nodes.")
    elif source_node == dest_node:
        st.warning("Source and destination must be different.")
    else:
        st.error("Cannot perform routing without at least two classical nodes.")


with col2:
    st.subheader("🗺️ Network Visualization")
    fig = visualize_network(G, pos, best_path, category)
    st.pyplot(fig)
