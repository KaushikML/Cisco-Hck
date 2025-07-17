import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import time

# --- Constants and Simulation Parameters ---
LOSS_PER_KM_DEFAULT = 0.05
SWAP_FAILURE_PROB_DEFAULT = 0.3
CLASSICAL_PACKET_LOSS_DEFAULT = 0.01
LATENCY_SECONDS_DEFAULT = 0.02

# --- Core Network and Protocol Functions ---

def generate_synthetic_network(num_nodes=15, num_quantum_endpoints=4, num_repeaters=2):
    """Generates a random network and assigns node/link types."""
    st.session_state.graph_seed = random.randint(0, 10000) # Use a random seed each time for variety
    
    # 1. Create a random geometric graph
    G = nx.random_geometric_graph(num_nodes, radius=0.4, seed=st.session_state.graph_seed)

    # Make the graph connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i+1]))
            G.add_edge(node1, node2)

    # 2. Assign node types
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for i, node in enumerate(nodes):
        node_name = f"Node-{node}" # Give nodes clearer names
        if i < num_quantum_endpoints:
            G.nodes[node]['type'] = 'quantum_endpoint'
        elif i < num_quantum_endpoints + num_repeaters:
            G.nodes[node]['type'] = 'quantum_repeater'
        else:
            G.nodes[node]['type'] = 'classical'
        G.nodes[node]['label'] = f"{node_name} ({G.nodes[node]['type'][0].upper()})"


    # 3. Assign edge types and distances
    for u, v in G.edges():
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        # Scale distance to a more realistic km range
        distance = math.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2) * 500
        G.edges[u, v]['distance'] = int(distance)

        node_u_type = G.nodes[u]['type']
        node_v_type = G.nodes[v]['type']
        if 'quantum' in node_u_type and 'quantum' in node_v_type:
            G.edges[u, v]['type'] = 'quantum'
        else:
            G.edges[u, v]['type'] = 'classical'
            
    # Relabel nodes for display purposes
    mapping = {node: G.nodes[node]['label'] for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G


def categorize_path(graph, path):
    """Categorizes a path as Quantum-Only, Hybrid, or Classical-Only."""
    path_types = set()
    for i in range(len(path) - 1):
        edge_type = graph.edges[path[i], path[i+1]]['type']
        path_types.add(edge_type)

    if "quantum" in path_types and "classical" in path_types:
        return "Hybrid"
    elif "quantum" in path_types:
        return "Quantum-Only"
    else:
        return "Classical-Only"

def calculate_path_cost(graph, path, reliability, weights):
    """Calculates a cost score for a path based on multiple factors."""
    hop_count = len(path) - 1
    total_distance = sum(graph.edges[path[i], path[i+1]]['distance'] for i in range(hop_count))
    
    cost = (hop_count * weights['w_hops']) + \
           (total_distance * weights['w_dist']) - \
           (reliability * weights['w_rel'])
    return cost

def get_path_reliability(graph, path, loss_per_km, swap_failure_prob):
    """Calculates the end-to-end success probability of a quantum path."""
    reliability = 1.0
    for i in range(len(path) - 1):
        edge = graph.edges[path[i], path[i+1]]
        if edge['type'] == 'quantum':
            distance = edge['distance']
            link_success_prob = 1.0 - min(loss_per_km * distance, 1.0)
            reliability *= link_success_prob
            if 'repeater' in graph.nodes[path[i+1]]['type']:
                reliability *= (1.0 - swap_failure_prob)
    return reliability

def find_best_path(graph, source, dest, weights, loss_per_km, swap_failure_prob):
    """Implements the AQFR protocol to find the best path."""
    try:
        all_paths = list(nx.all_simple_paths(graph, source=source, target=dest, cutoff=6))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, "No path found between selected nodes."

    if not all_paths:
        return None, "No path found (or path exceeds cutoff limit)."

    categorized_paths = { "Quantum-Only": [], "Hybrid": [], "Classical-Only": [] }
    for path in all_paths:
        category = categorize_path(graph, path)
        reliability = get_path_reliability(graph, path, loss_per_km, swap_failure_prob) if "quantum" in category else 1.0
        cost = calculate_path_cost(graph, path, reliability, weights)
        categorized_paths[category].append({"path": path, "cost": cost, "reliability": reliability})

    for category in ["Quantum-Only", "Hybrid", "Classical-Only"]:
        if categorized_paths[category]:
            best = sorted(categorized_paths[category], key=lambda x: x['cost'])[0]
            best['category'] = category
            return best, f"Selected best path from '{category}' category."
            
    return None, "No suitable path found."

def simulate_transmission(graph, path_info, classical_loss_prob):
    """Simulates sending a 'dataset' of 10 packets over the selected path."""
    success_count = 0
    num_packets = 10
    path = path_info['path']
    category = path_info['category']

    for i in range(num_packets):
        is_successful = True
        if "Quantum" in category:
            if random.random() > path_info['reliability']:
                is_successful = False
        if is_successful:
            for j in range(len(path) - 1):
                if graph.edges[path[j], path[j+1]]['type'] == 'classical':
                    if random.random() < classical_loss_prob:
                        is_successful = False
                        break
        if is_successful:
            success_count += 1
    return success_count


# --- Visualization Functions ---

def draw_network(graph, selected_path=None):
    """Draws the network graph, highlighting the selected path."""
    pos = nx.spring_layout(graph, seed=st.session_state.get('graph_seed', 42), k=0.8)
    
    node_colors = { "quantum_endpoint": "skyblue", "quantum_repeater": "lightgreen", "classical": "lightgray" }
    colors = [node_colors[graph.nodes[n]['type']] for n in graph.nodes]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw(graph, pos, with_labels=True, labels=nx.get_node_attributes(graph, 'label'), node_color=colors, node_size=2500,
            font_size=9, font_weight='bold', ax=ax)
    
    if selected_path:
        path_edges = list(zip(selected_path, selected_path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=selected_path, node_color='gold', node_size=2500, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='gold', width=4, ax=ax)
        ax.set_title(f"Network Topology - Path: {' -> '.join(selected_path)}", fontsize=16)
    else:
        ax.set_title("Network Topology", fontsize=16)
        
    return fig

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🛰️ Interactive Hybrid Quantum-Classical Network Simulator")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🌐 Network Generation")
    num_nodes = st.slider("Total Nodes", 5, 30, 15)
    num_q_endpoints = st.slider("Quantum Endpoints", 2, 10, 4)
    num_q_repeaters = st.slider("Quantum Repeaters", 0, 10, 2)

    if st.button("Generate New Network"):
        st.session_state.G = generate_synthetic_network(num_nodes, num_q_endpoints, num_q_repeaters)
        st.session_state.best_path_info = None # Reset path on new network
    
    # Initialize graph on first run
    if 'G' not in st.session_state:
        st.session_state.G = generate_synthetic_network(num_nodes, num_q_endpoints, num_q_repeaters)

    G = st.session_state.G
    node_list = sorted(list(G.nodes()))

    st.header("⚙️ Simulation Controls")
    source_node = st.selectbox("Select Source Node", node_list, index=0)
    dest_node = st.selectbox("Select Destination Node", node_list, index=1 if len(node_list)>1 else 0)

    st.subheader("Protocol Weights")
    w_hops = st.slider("Hop Count Weight (w1)", 0.1, 5.0, 1.0)
    w_dist = st.slider("Distance Weight (w2)", 0.0, 1.0, 0.1, format="%.2f")
    w_rel = st.slider("Reliability Weight (w3)", 0.0, 10.0, 5.0)
    weights = {'w_hops': w_hops, 'w_dist': w_dist, 'w_rel': w_rel}

    st.subheader("Physics Parameters")
    loss_per_km = st.slider("Qubit Loss / km", 0.0, 0.2, LOSS_PER_KM_DEFAULT, format="%.3f")
    swap_failure_prob = st.slider("Repeater Swap Failure %", 0, 100, int(SWAP_FAILURE_PROB_DEFAULT*100)) / 100.0

# --- Main Display Area ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📍 Network & Routing")
    
    # Find the best path
    if source_node == dest_node:
        st.warning("Source and destination must be different.")
        st.session_state.best_path_info = None
    else:
        st.session_state.best_path_info, message = find_best_path(G, source_node, dest_node, weights, loss_per_km, swap_failure_prob)

    if st.session_state.best_path_info:
        best_path_info = st.session_state.best_path_info
        st.success(message)
        
        path = best_path_info['path']
        category = best_path_info['category']
        cost = best_path_info['cost']
        reliability = best_path_info['reliability']

        st.metric("Path Category", category)
        st.write(f"**Path:** `{' -> '.join(path)}`")
        st.write(f"**Calculated Cost:** `{cost:.2f}`")
        if "Quantum" in category:
            st.write(f"**End-to-End Reliability:** `{reliability:.2%}`")
        
        st.markdown("---")
        st.subheader("📡 Transmission Test")
        st.write("Test the performance of the selected route.")

        if st.button("Simulate 100 Packet Transmissions"):
            successes = simulate_transmission(G, best_path_info, CLASSICAL_PACKET_LOSS_DEFAULT)
            st.metric("Transmission Success Rate", f"{successes}/100", f"{successes}%")
            st.progress(successes / 100)
    else:
         st.error("No path found between selected nodes.")

with col2:
    st.subheader("🗺️ Network Visualization")
    path_to_draw = st.session_state.best_path_info['path'] if st.session_state.get('best_path_info') else None
    fig = draw_network(G, selected_path=path_to_draw)
    st.pyplot(fig)
