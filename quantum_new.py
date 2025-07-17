# save as app.py   ───  streamlit run app.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np, math
plt.style.use("default")           # cleaner default look for Colab/Streamlit

# ========== CONSTANTS & DEFAULTS ==========
L0 = 2.5
TRANSLATION_COST = 0.1
DEFAULT_WEIGHTS = dict(w1=1.0, w2=0.01, w3=2.0, w4=0.05)

# ───────────────────────────────────────────
#  PART 1  – NETWORK GENERATION + FIXERS
# ───────────────────────────────────────────
def create_hybrid_network(num_nodes=12, quantum_ratio=0.4, seed=42):
    random.seed(seed); np.random.seed(seed)
    G = nx.erdos_renyi_graph(num_nodes, 0.4, seed=seed)
    q_nodes = random.sample(list(G.nodes), int(num_nodes*quantum_ratio))
    for n in G.nodes:
        G.nodes[n]["type"] = "quantum" if n in q_nodes else "classical"
    return G

def connect_quantum_subgraph(G):
    q_nodes = [n for n,d in G.nodes(data=True) if d["type"]=="quantum"]
    comps = list(nx.connected_components(G.subgraph(q_nodes)))
    for i in range(len(comps)-1):
        u, v = next(iter(comps[i])), next(iter(comps[i+1]))
        G.add_edge(u, v)
    return G

def ensure_classical_two_quantum(G):
    q_nodes = [n for n,d in G.nodes(data=True) if d["type"]=="quantum"]
    for c in [n for n,d in G.nodes(data=True) if d["type"]=="classical"]:
        q_neigh = [nbr for nbr in G.neighbors(c) if G.nodes[nbr]["type"]=="quantum"]
        need = 2 - len(q_neigh)
        if need > 0:
            cand = list(set(q_nodes)-set(q_neigh))
            for q in random.sample(cand, k=min(need, len(cand))):
                G.add_edge(c, q)
    return G

# ───────────────────────────────────────────
#  PART 2  – EDGE ANNOTATION
# ───────────────────────────────────────────
def assign_link_behaviors(G, pos):
    for u,v in G.edges():
        d = float(np.hypot(*(np.array(pos[u])-np.array(pos[v]))))
        ut, vt = G.nodes[u]["type"], G.nodes[v]["type"]

        if ut==vt=="quantum":
            G.edges[u,v].update(dict(type="quantum", distance=d,
                decoherence_prob=1-np.exp(-d/L0),
                entanglement_success=max(0.05, np.exp(-0.3*d))))
        elif ut==vt=="classical":
            G.edges[u,v].update(dict(type="classical", distance=d,
                latency_ms=random.uniform(1,10),
                packet_loss=random.uniform(0.01,0.05),
                amplification_allowed=True))
        elif ut=="quantum" and vt=="classical":
            G.edges[u,v].update(dict(type="hybrid_QC", distance=d,
                latency_ms=random.uniform(2,15),
                packet_loss=random.uniform(0.02,0.08)))
        else:  # C→Q
            G.edges[u,v].update(dict(type="hybrid_CQ", distance=d,
                latency_ms=random.uniform(2,15),
                fidelity=random.uniform(0.7,0.99),
                loss_prob=random.uniform(0.02,0.08)))
    return G

# success-prob
def edge_success(a):
    t=a["type"]
    if t=="quantum":   return (1-a["decoherence_prob"])*a["entanglement_success"]
    if t=="classical": return 1-a["packet_loss"]
    if t=="hybrid_QC": return (1-a["packet_loss"])*0.95
    if t=="hybrid_CQ": return a["fidelity"]*(1-a["loss_prob"])
    return 0.0
def edge_latency(a):   return a.get("latency_ms", 0.0)

# ───────────────────────────────────────────
#  PART 3  – AQFR ROUTING
# ───────────────────────────────────────────
def has_quantum_link(G, path):
    return any(G[u][v]["type"]=="quantum" for u,v in zip(path,path[1:]))

def has_two_consecutive_q_between_c(G, path):
    c_idx=[i for i,n in enumerate(path) if G.nodes[n]["type"]=="classical"]
    for i in range(len(c_idx)-1):
        seg = path[c_idx[i]+1:c_idx[i+1]]
        best=streak=0
        for n in seg:
            if G.nodes[n]["type"]=="quantum":
                streak+=1; best=max(best,streak)
            else: streak=0
        if best<2: return False
    return True

def classify_path(G,path):
    hybrid=any("hybrid" in G[u][v]["type"] for u,v in zip(path,path[1:]))
    classical=any(G[u][v]["type"]=="classical" for u,v in zip(path,path[1:]))
    if has_quantum_link(G,path): return "Quantum-QKD"
    if hybrid:                   return "Hybrid"
    if classical:                return "Classical"
    return "Classical"

def compute_metrics(G,path):
    succ=1; dist=lat=0; hops=len(path)-1; trans=0
    prev_dom=G.nodes[path[0]]["type"]
    for u,v in zip(path,path[1:]):
        a=G[u][v]
        succ*=edge_success(a); dist+=a["distance"]; lat+=edge_latency(a)
        if G.nodes[v]["type"]!=prev_dom: trans+=1
        prev_dom=G.nodes[v]["type"]
    return hops,dist,lat,succ,trans

def cost(h,d,succ,trans,lat,w):
    return w['w1']*h + w['w2']*d - w['w3']*succ + TRANSLATION_COST*trans + w['w4']*lat

def best_path(G, paths, w):
    return min(paths, key=lambda p: cost(*compute_metrics(G,p), w))

def aqfr_route(G, src, dst, w, cutoff=8):
    try: all_p=list(nx.all_simple_paths(G,src,dst,cutoff))
    except (nx.NodeNotFound,nx.NetworkXNoPath): return None,"No Path"
    if not all_p: return None,"No Path"

    q_paths  =[p for p in all_p if classify_path(G,p)=="Quantum-QKD" and has_two_consecutive_q_between_c(G,p)]
    if q_paths:  return best_path(G,q_paths,w),"Quantum-QKD"

    h_paths  =[p for p in all_p if classify_path(G,p)=="Hybrid" and has_two_consecutive_q_between_c(G,p)]
    if h_paths: return best_path(G,h_paths,w),"Hybrid"

    return None,"No Path"

# ───────────────────────────────────────────
#  STREAMLIT UI
# ───────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🛰️ AQFR - Adaptive Quantum-First Routing Simulator")

# Sidebar – network parameters
with st.sidebar:
    st.header("🌐 Network parameters")
    seed = st.number_input("Random seed", 0, 9999, 42)
    num_nodes = st.slider("Total nodes", 5, 30, 12)
    q_ratio   = st.slider("Quantum node ratio", 0.1, 1.0, 0.4)

    if st.button("Generate network"):
        G = create_hybrid_network(num_nodes,q_ratio,seed)
        G = connect_quantum_subgraph(G)
        G = ensure_classical_two_quantum(G)
        pos = nx.spring_layout(G, seed=seed)
        G = assign_link_behaviors(G,pos)
        st.session_state.update(dict(G=G,pos=pos,best=None,cat=None))

    if "G" not in st.session_state:
        st.session_state.G,st.session_state.pos = None,None

    st.header("⚙️ Routing weights")
    weights={}
    for k,default in DEFAULT_WEIGHTS.items():
        lbl = dict(w1="Hop weight",w2="Distance weight",
                   w3="Reliability weight",w4="Latency weight")[k]
        step=0.01 if k=="w2" else 0.1
        weights[k]=st.slider(lbl,0.0,5.0 if k!="w4" else 1.0,default,step=step)

# ===== main columns =====
col1,col2 = st.columns([1,1.5])

# Route controls + metrics
with col1:
    st.subheader("📍 Routing")
    G=st.session_state.G
    if G is None:
        st.info("Click **Generate network** first.")
    else:
        classical_nodes=[n for n,d in G.nodes(data=True) if d["type"]=="classical"]
        if len(classical_nodes)<2:
            st.error("Need at least two classical nodes to route.")
        else:
            src=st.selectbox("Source (classical)",classical_nodes,0)
            dst=st.selectbox("Destination (classical)",classical_nodes,1)
            if src==dst:
                st.warning("Pick distinct nodes.")
            else:
                path,cat=aqfr_route(G,src,dst,weights)
                st.session_state.best,st.session_state.cat=path,cat
                if path:
                    hops,dist,lat,succ,trans=compute_metrics(G,path)
                    st.success(f"Path found ({cat})")
                    st.write("→".join(map(str,path)))
                    st.metric("Hop count",hops)
                    st.metric("Distance",f"{dist:.2f}")
                    st.metric("E2E success",f"{succ:.2%}")
                    st.metric("Domain switches",trans)
                    with st.expander("Hop-by-hop"):
                        for u,v in zip(path,path[1:]):
                            st.write(f"{u} → {v} [{G[u][v]['type']}]  "
                                     f"succ={edge_success(G[u][v]):.3f}")
                else:
                    st.error("No valid path under constraints.")

# Visual column
with col2:
    st.subheader("🗺️ Network")
    if st.session_state.G is not None:
        fig = plt.figure(figsize=(10,7))
        G,pos = st.session_state.G, st.session_state.pos
        edge_cmap={"quantum":"cyan","classical":"gray",
                   "hybrid_QC":"purple","hybrid_CQ":"magenta"}
        ec=[edge_cmap[G[e[0]][e[1]]["type"]] for e in G.edges]
        nc=["skyblue" if d["type"]=="quantum" else "orange"
             for _,d in G.nodes(data=True)]
        nx.draw(G,pos,node_color=nc,edge_color=ec,
                node_size=1400,font_size=10,with_labels=True,width=2)
        # highlight path
        if st.session_state.best:
            pe=list(zip(st.session_state.best,st.session_state.best[1:]))
            nx.draw_networkx_edges(G,pos,edgelist=pe,edge_color="red",width=4)
            plt.title(f"AQFR path ({st.session_state.cat})")
        else:
            plt.title("Hybrid network topology")
        st.pyplot(fig)
