# AQFR – Quantum-Only-Intermediates Routing Demo

This is a Streamlit application that demonstrates an "Adaptive Quantum-First Routing" (AQFR) algorithm for a hybrid quantum-classical network. The simulation allows you to:

*   Generate a hybrid network with a specified number of nodes and a ratio of quantum to classical nodes.
*   Visualize the network topology, with different colors for quantum and classical nodes and links.
*   Find the best path between two classical nodes, with the constraint that all intermediate nodes must be quantum-only.
*   Compare the AQFR path with the shortest classical-only path.
*   Adjust the weights for different cost metrics (hops, distance, reliability, latency) to see how they affect the path selection.

## How to run the application

1.  Install the required libraries:
    ```
    pip install streamlit networkx matplotlib numpy
    ```
2.  Run the Streamlit application:
    ```
    streamlit run app.py
    ```

## Features and Parameters

### Network Generation

*   **Seed**: A random seed for reproducibility.
*   **Nodes**: The total number of nodes in the network.
*   **Quantum ratio**: The proportion of quantum nodes in the network.

### Routing

The application implements an AQFR algorithm that prioritizes paths with quantum-only intermediate nodes. It first looks for a "Quantum-QKD" path, and if that's not available, it falls back to a classical path. The cost function for path selection is a weighted sum of hops, distance, reliability, and latency.

### Visualization

The network is visualized using Matplotlib, with:

*   Quantum nodes in sky blue and classical nodes in orange.
*   Quantum links in cyan, classical links in grey, and hybrid links in purple/magenta.
*   The AQFR path is highlighted in red.
*   The shortest classical-only path is highlighted with a dashed green line.
