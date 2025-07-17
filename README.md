Design and Simulate a Scalable Quantum-Classical Hybrid Network for the Future 
Internet
Background
Quantum networking is poised to revolu4onize communica4on, but faces major 
challenges: qubit fragility, signal loss, scalability, integra4on with classical networks, and 
lack of standardiza4on245. Most real-world quantum networks will be hybrid—
combining quantum and classical elements. Understanding these challenges is crucial, 
even for those with only basic knowledge of qubits and superposi4on5.
Problem Statement
Design, simulate, and analyze a scalable hybrid network architecture that integrates 
quantum and classical nodes. You will address prac4cal networking challenges - like 
signal loss, decoherence, interoperability, and scalability - using simula4on and crea4ve 
protocol design.
No QKD or advanced quantum protocol implementaEon is required. Only an 
introducEon to qubits, entanglement, and superposiEon is assumed.
Problem Parts
Part 1: Network Topology and Node SimulaEon
a) Design a network with at least 10 nodes: some classical, some quantum-capable 
(suppor4ng qubits and entanglement).
b) Simulate the network using Python and a library like networkx (for topology and 
rou4ng), adding simple quantum aOributes to nodes (e.g., “can store 
entanglement”).
c) Visualize the topology, labeling quantum and classical nodes.
Deliverable:
• Python code for network crea4on and visualiza4on
• Diagram of the hybrid network
Part 2: SimulaEng Quantum Networking Challenges
For quantum links, simulate:
a) Decoherence: Each quantum link has a probability of “qubit loss” over distance.
b) No-Cloning: Quantum data cannot be copied or amplified; simulate failed 
transmission if amplifica4on is aOempted.
Cisco Confiden+al
c) Entanglement DistribuEon: Simulate entanglement swapping between quantum 
nodes, with a probability of failure.
d) For classical links, simulate standard packet loss and latency.
Deliverable:
• Simula4on code for link behavior
• Report/plots showing how quantum and classical links behave differently as 
network size/distance increases
Part 3: Protocol Design for Hybrid RouEng
a) Design a rou4ng protocol that:
b) Uses quantum links when available, but falls back to classical links if quantum 
transmission fails.
c) Priori4zes shortest path but considers quantum link reliability.
d) Handles interoperability between quantum and classical nodes (e.g., protocol 
transla4on).
Deliverable:
• Protocol descrip4on (algorithm or flowchart)
• Implementa4on in Python (pseudo-code or code)
o Example: send a message from Node A to Node H, showing path selec4on 
and fallback
Part 4: Scalability and StandardizaEon Analysis
Simulate the network as you increase the number of nodes and links.
• Analyze:
o How does the probability of successful end-to-end quantum 
communica4on change?
o What are the boOlenecks for scaling up?
o What interoperability/standardiza4on issues arise (e.g., protocol 
mismatches, lack of shared standards)?
Deliverable:
a) Plots/graphs showing scalability trends
b) WriOen analysis of technical and standardiza4on challenges
Part 5: CreaEve Extension (Bonus)
a) Propose and simulate one innova4ve solu4on to a major boOleneck (e.g., using 
quantum repeaters, error correc4on, or a new hybrid protocol).
b) Analyze its impact on network performance.
Deliverable:
• Descrip4on and simula4on code for your solu4on
Cisco Confiden+al
Cisco Confiden+al
• Compara4ve results (before/acer)
Part 6: Design a simple key PKI without public keys
Imagine its 2030 and quantum computers have become ubiquitous. With the advent of quantum 
compu;ng, public key cryptography has broken down. 
A) Design a system for key exchange using symmetric keys so that communica;on between people 
can be secured without others being able to eavesdrop on it. Assume that there are 25 people in 
a group and communica;on between any 2 cannot be seen in clear by the other 23.
B) Propose mul;ple op;ons and compare the tradeoffs between them. 
C) Implement one of the op;ons to solve the problem stated in A.
