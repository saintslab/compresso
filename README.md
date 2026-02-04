# Compresso: Minimal Finite Covering for Robust Dataset Condensation

Compresso is an analytical tool designed to facilitate the exploration of Minimal Finite Covering (MFC) as a rigorous framework for dataset condensation. Unlike standard dataset distillation methods that prioritize empirical performance at the cost of adversarial stability, this tool implements a robustness-aware approach by identifying a discrete skeleton $\mathcal{C}$ that forms an $\epsilon$-covering of the high-dimensional manifold $\mathcal{X}$. The theoretical foundation of Compresso ensures that models trained on the condensed set optimize a provable lower bound of the generalized adversarial loss.

## Installation and Environment Configuration

To deploy Compresso on a local system, a Python 3.8+ environment is required alongside several scientific computing libraries. Users should first clone the repository and navigate into the project directory to initialize the environment. The primary dependencies include PyTorch and Torchvision for model architecture and data ingestion, Streamlit for the interactive interface, Scikit-learn for manifold projections, and Plotly for geometric visualization. These can be installed by executing a single command through the package manager using pip install streamlit torch torchvision scikit-learn plotly pandas numpy pillow. It is recommended to perform this installation within a dedicated virtual environment to prevent dependency conflicts with existing system-level libraries.

## Operational Workflow and Data Ingestion

The application is executed by running the command streamlit ```streamlit run main.py``` in a terminal. Upon initialization, the user is presented with a sidebar for data ingestion where they can select from standard manifolds such as MNIST, FashionMNIST, and CIFAR-10. These datasets are automatically downloaded and cached locally to ensure efficient subsequent access. For researchers working with proprietary data, the tool supports a local folder mode where subdirectories within a specified path are treated as independent class labels. The ingestion pipeline flattens images into their high-dimensional feature representations for metric consistency.

## MFC Computation and Manifold Visualization

The core logic of Compresso resides in its greedy set-cover approximation algorithm which operates directly in the native high-dimensional space of the selected manifold. Users can adjust the covering radius $\epsilon$ to observe the phase transition between dense population and sparse skeletal representation. To visualize this high-dimensional covering, the tool offers both linear PCA and non-linear t-SNE projections. The visualization layer overlays the original manifold distribution with the selected MFC anchors and approximates the $\epsilon$-balls in the projected 2D plane. A live gallery of the core samples is provided to allow for semantic inspection of the anchors, ensuring that the selected subset captures the characteristic topology of each class.

## Robust Optimization and Terminal Monitoring

Compresso includes an integrated PyTorch training session that enables users to execute robust optimization sequences on the condensed set. By adjusting the learning rate and epoch count, researchers can monitor the convergence of a robust classifier trained exclusively on the MFC anchors. Throughout the entire process, the tool provides real-time feedback via timestamped logs in the terminal console. These logs detail the progress of the manifold covering, the percentage of data accounted for in each iteration, and the final compression efficiency. This transparency allows for a granular understanding of the computational cost and the mathematical guarantees associated with the resulting condensed dataset.

## Source Work

This repository implements the minimal finite covering problem and its application, and is the official source code for paper "[Is Adversarial Training with Compressed Datasets Effective?](https://arxiv.org/abs/2402.05675)" (Chen & Selvan. 2025).
