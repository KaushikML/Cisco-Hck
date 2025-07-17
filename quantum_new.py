# ─── app.py  ────────────────────────────────────────────────────────────────
# Streamlit: Adaptive Quantum-First Routing demo
# (intermediate nodes must be quantum-only)
# ---------------------------------------------------------------------------
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np

# ───────── Global parameters ───────────────────────────────────────────────
L0 = 2.5                       # decoherence length
TRANSLATION_COST = 0.1
DEFAULT_W = dict(w1=1.0, w2=0.01, w3=2.0, w4=0.05)

FIDELITY_THRESHOLD   = 0.75
CLASSICAL_THRESHOLD  = 0.90
ROUTE_LOSS_THRESHOLD = 0.30
MAX_HOPS             = 8

# ───────── Part 1: Build topology & enforce architecture rules ────────────
def create_hybrid_network(n=12, q_ratio=0.4, seed=42):
    random.seed(seed); np.random.seed(seed)
    G = nx.erdos_renyi_graph(n, 0.4, seed=seed)
    q_nodes = random.sample(list(G.nodes), int(n*q_ratio))
    for v in G:
        G.nodes[v]["type"] = "quantum" if v in q_nodes else "classical"
    return G

def connect_quantum_component(G):
    q = [n for n,d in G.nodes(data=True) if d["type"]=="quantum"]
    comps = list(nx.connected_components(G.subgraph(q)))
    for i in range(len(comps)-1):
        G.add_edge(next(iter(comps[i])), next(iter(comps[i+1])))
    return G

def ensure_classical_two_q(G):
    q_nodes = [n for n,d in G.nodes(data=True) if d["type"]=="quantum"]
    for c in [n for n,d in G.nodes(data=True) if d["type"]=="classical"]:
        q_neigh = [nbr for nbr in G.neighbors(c)
                   if G.nodes[nbr]["type"]=="quantum"]
        needed = 2 - len(q_neigh)
        if needed>0:
            for q in random.sample(list(set(q_nodes)-set(q_neigh)), needed):
                G.add_edge(c,q)
    return G

# ───────── Part 2: Edge annotation ────────────────────────────────────────
def annotate_edges(G, pos):
    for u,v in G.edges():
        d = np.linalg.norm(np.array(pos[u])-np.array(pos[v]))
        ut,vt = G.nodes[u]["type"], G.nodes[v]["type"]
        if ut==vt=="quantum":
            G.edges[u,v].update(type="quantum", distance=d,
                decoherence_prob=1-np.exp(-d/L0),
                entanglement_success=max(0.05,np.exp(-0.3*d)))
        elif ut==vt=="classical":
            G.edges[u,v].update(type="classical", distance=d,
                latency_ms=random.uniform(1,10),
                packet_loss=random.uniform(0.01,0.05))
        elif ut=="quantum" and vt=="classical":
            G.edges[u,v].update(type="hybrid_QC", distance=d,
                latency_ms=random.uniform(2,15),
                packet_loss=random.uniform(0.02,0.08))
        else:
            G.edges[u,v].update(type="hybrid_CQ", distance=d,
                latency_ms=random.uniform(2,15),
                fidelity=random.uniform(0.7,0.99),
                loss_prob=random.uniform(0.02,0.08))
    return G

# edge helpers
def edge_success(a):
    t=a["type"]
    if t=="quantum":   return (1-a["decoherence_prob"])*a["entanglement_success"]
    if t=="classical": return 1-a["packet_loss"]
    if t=="hybrid_QC": return (1-a["packet_loss"])*0.95
    if t=="hybrid_CQ": return a["fidelity"]*(1-a["loss_prob"])
    return 0.0
def edge_latency(a):   return a.get("latency_ms",0.0)

# ───────── Part 3: Routing with quantum-only intermediates ────────────────
def intermediates_all_quantum(G,path):
    return all(G.nodes[n]["type"]=="quantum" for n in path[1:-1])

def two_consecutive_q_between_c(G,path):
    c_idx=[i for i,n in enumerate(path) if G.nodes[n]["type"]=="classical"]
    for i in range(len(c_idx)-1):
        seg=path[c_idx[i]+1:c_idx[i+1]]
        streak=best=0
        for n in seg:
            if G.nodes[n]["type"]=="quantum":
                streak+=1; best=max(best,streak)
            else: streak=0
        if best<2: return False
    return True

def classify_path(G,path):
    return "quantum_qkd" if any(G.nodes[u]["type"]=="quantum" and
                                G.nodes[v]["type"]=="quantum"
                                for u,v in zip(path,path[1:])) else "classical"

def compute_metrics(G,path):
    succ=1; dist=lat=0; trans=0; prev=G.nodes[path[0]]["type"]
    for u,v in zip(path,path[1:]):
        a=G[u][v]
        succ*=edge_success(a); dist+=a["distance"]; lat+=edge_latency(a)
        if G.nodes[v]["type"]!=prev: trans+=1
        prev=G.nodes[v]["type"]
    hops=len(path)-1; loss=1-succ
    return hops,dist,lat,succ,trans,loss

def cost(h,d,succ,trans,lat,w):
    return w['w1']*h + w['w2']*d - w['w3']*succ + TRANSLATION_COST*trans + w['w4']*lat

def best_path(G,paths,w):
    return min(paths, key=lambda p: cost(*compute_metrics(G,p)[:5],w))

def reliability_ok(G,path):
    hops,dist,lat,succ,trans,loss=compute_metrics(G,path)
    R_q=R_c=1
    for u,v in zip(path,path[1:]):
        a=G[u][v]
        if a["type"]=="quantum": R_q*=edge_success(a)
        if a["type"]=="classical": R_c*=edge_success(a)
    return (R_q>=FIDELITY_THRESHOLD and
            loss<=ROUTE_LOSS_THRESHOLD and
            hops<=MAX_HOPS and
            R_c>=CLASSICAL_THRESHOLD)

def aqfr_route(G,src,dst,w,cutoff=8):
    try: paths=list(nx.all_simple_paths(G,src,dst,cutoff))
    except (nx.NodeNotFound,nx.NetworkXNoPath): return None,"No Path"
    # quantum-only intermediates
    paths=[p for p in paths if intermediates_all_quantum(G,p)]
    if not paths: return None,"No path w/ quantum intermediates"
    # QKD tier
    q_tier=[p for p in paths if classify_path(G,p)=="quantum_qkd"
            and two_consecutive_q_between_c(G,p)]
    if q_tier:
        return best_path(G,q_tier,w),"Quantum-QKD"
    # Classical fallback tier
    c_tier=[p for p in paths if classify_path(G,p)=="classical"
            and two_consecutive_q_between_c(G,p)
            and reliability_ok(G,p)]
    if c_tier:
        return best_path(G,c_tier,w),"Classical"
    return None,"No valid path"

# ───────── NEW: Shortest classical-only path ────────────────
def shortest_classical_path(G, src, dst):
    # create a subgraph with only classical edges
    classical_edges = [(u,v) for u,v,d in G.edges(data=True)
                       if d["type"]=="classical"]
    G_classical = nx.Graph()
    G_classical.add_edges_from(classical_edges)
    if nx.has_path(G_classical, src, dst):
        return nx.shortest_path(G_classical, src, dst)
    else:
        return None

# ───────── Streamlit UI ────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🛰️ AQFR – Quantum-Only-Intermediates Routing Demo")

with st.sidebar:
    st.header("Network")
    seed = st.number_input("Seed",0,9999,42)
    n_nodes = st.slider("Nodes",5,30,12)
    q_ratio = st.slider("Quantum ratio",0.1,1.0,0.4)
    if st.button("Generate / Reset"):
        G0 = create_hybrid_network(n_nodes,q_ratio,seed)
        G0 = connect_quantum_component(G0)
        G0 = ensure_classical_two_q(G0)
        pos = nx.spring_layout(G0, seed=seed)
        G0 = annotate_edges(G0,pos)
        st.session_state.update(dict(G=G0,pos=pos,best=None,cat=None,shortest_classical=None))
    if "G" not in st.session_state:
        st.info("Press **Generate / Reset** to start.")

    st.header("Weights")
    W={}
    for k,lbl in zip("w1 w2 w3 w4".split(),
            ["Hop","Distance","Reliability","Latency"]):
        step=0.01 if k=="w2" else 0.1
        W[k]=st.slider(lbl,0.0,5.0,DEFAULT_W[k],step=step)

# ───────── main area ───────────────────────────────────────────────────────
if "G" in st.session_state:
    G=st.session_state.G; pos=st.session_state.pos
    classical=[n for n,d in G.nodes(data=True) if d["type"]=="classical"]
    col1,col2=st.columns([1,1.5])

    with col1:
        st.subheader("Route")
        if len(classical)<2:
            st.error("Need ≥2 classical nodes.")
        else:
            src=st.selectbox("Source",classical,0)
            dst=st.selectbox("Destination",classical,1)
            if src==dst:
                st.warning("Pick distinct nodes.")
            else:
                # AQFR route
                path,cat=aqfr_route(G,src,dst,W)
                st.session_state.best,st.session_state.cat=path,cat

                # Baseline shortest classical-only path
                shortest_classic = shortest_classical_path(G, src, dst)
                st.session_state.shortest_classical = shortest_classic

                # Display AQFR path details
                if path:
                    hops,dist,lat,succ,trans,loss=compute_metrics(G,path)
                    st.success(f"{cat} path found")
                    st.write("**AQFR Path:** " + " → ".join(map(str,path)))
                    st.metric("Hops",hops)
                    st.metric("Distance",f"{dist:.2f}")
                    st.metric("Success",f"{succ:.2%}")
                    st.metric("Switches",trans)
                    with st.expander("Hop-by-hop details"):
                        for u,v in zip(path,path[1:]):
                            st.write(f"{u}→{v} [{G[u][v]['type']}] "
                                     f"P={edge_success(G[u][v]):.3f}")
                else:
                    st.error(cat)

                # Display baseline shortest classical path
                if shortest_classic:
                    st.info("**Shortest Classical Path:** " +
                            " → ".join(map(str,shortest_classic)))
                    # Compute its metrics
                    hops_c,dist_c,lat_c,succ_c,trans_c,loss_c=compute_metrics(G,shortest_classic)
                    st.write(f"Baseline Classical Metrics: "
                             f"Hops={hops_c}, Dist={dist_c:.2f}, "
                             f"Succ={succ_c:.2%}, Loss={loss_c:.3f}")
                else:
                    st.warning("No purely classical shortest path available.")

    with col2:
        st.subheader("Topology")
        fig=plt.figure(figsize=(9,7))
        ecmap={"quantum":"cyan","classical":"grey",
               "hybrid_QC":"purple","hybrid_CQ":"magenta"}
        # draw base network
        nx.draw(G,pos,
                node_color=["skyblue" if d["type"]=="quantum"
                            else "orange" for _,d in G.nodes(data=True)],
                edge_color=[ecmap[G[e[0]][e[1]]["type"]] for e in G.edges],
                node_size=1200,font_size=9,width=1.8)

        # Draw AQFR path in red
        if st.session_state.best:
            pe=list(zip(st.session_state.best,st.session_state.best[1:]))
            nx.draw_networkx_edges(G,pos,edgelist=pe,edge_color="red",width=4)
            nx.draw_networkx_labels(G,pos)

        # Draw baseline shortest classical path in green dashed
        if st.session_state.shortest_classical:
            pe_classic = list(zip(st.session_state.shortest_classical,
                                  st.session_state.shortest_classical[1:]))
            nx.draw_networkx_edges(G,pos,edgelist=pe_classic,
                                   edge_color="green",width=3,style="dashed")

        if st.session_state.best:
            plt.title(f"AQFR path ({st.session_state.cat}) + Shortest Classical (green dashed)")
        else:
            plt.title("Hybrid network")
        st.pyplot(fig)
