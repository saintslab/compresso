import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import datetime
import os
import base64
from PIL import Image
import io

# --- Utility Functions ---
def log_terminal(message, level="INFO"):
    """Prints a timestamped message to the terminal console."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def vector_to_base64(vector, shape=(28, 28, 1)):
    """Converts a high-dim vector back to a base64 encoded PNG thumbnail."""
    if len(shape) == 3 and shape[2] == 3:
        # Handling RGB for CIFAR
        img_array = (vector.reshape(shape[0], shape[1], 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'RGB')
    else:
        # Handling Grayscale
        img_array = (vector.reshape(shape[0], shape[1]) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Pytorch Model Architecture ---
class RobustClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        super(RobustClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- MFC Algorithm Core ---
def compute_mfc(vectors, epsilon):
    """
    Implements the Greedy Set Cover approximation for Minimal Finite Covering.
    Computed in the original high-dimensional space with terminal logging.
    """
    log_terminal(f"Starting MFC computation (N={vectors.shape[0]}, epsilon={epsilon})")
    start_time = time.time()
    
    n_points = vectors.shape[0]
    covered = np.zeros(n_points, dtype=bool)
    covering_indices = []
    
    v_torch = torch.from_numpy(vectors).float()
    
    iteration = 0
    while not np.all(covered):
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0: break
        
        anchor_idx = uncovered_idx[0]
        covering_indices.append(anchor_idx)
        
        anchor_vec = v_torch[anchor_idx].unsqueeze(0)
        dists = torch.norm(v_torch - anchor_vec, dim=1)
        
        new_covered_mask = dists.numpy() <= epsilon
        covered[new_covered_mask] = True
        
        iteration += 1
        if iteration % 50 == 0:
            percent = (np.sum(covered) / n_points) * 100
            log_terminal(f"Iteration {iteration}: {percent:.1f}% of manifold covered")
        
    duration = time.time() - start_time
    log_terminal(f"MFC Finished. Found {len(covering_indices)} anchors in {duration:.2f}s")
    return covering_indices

# --- Multi-Source Data Loading ---
@st.cache_data
def load_dataset(dataset_name, n_samples=1000, local_path=None):
    """
    Generic loader for standard datasets and local folders.
    Local data assumes subfolders represent distinct classes (folder names as labels).
    """
    log_terminal(f"Dataset Loading Triggered: {dataset_name}")
    if not os.path.exists('/home/raghav/dsl/datasets'):
        os.makedirs('/home/raghav/dsl/datasets')

    data_list = []
    label_list = []
    img_shape = (28, 28, 1)
    class_names = []

    if dataset_name == "MNIST":
        ds = torchvision.datasets.MNIST(root='/home/raghav/dsl/datasets', train=True, download=True, transform=transforms.ToTensor())
        img_shape = (28, 28, 1)
        class_names = [str(i) for i in range(10)]
    elif dataset_name == "FashionMNIST":
        ds = torchvision.datasets.FashionMNIST(root='/home/raghav/dsl/datasets', train=True, download=True, transform=transforms.ToTensor())
        img_shape = (28, 28, 1)
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    elif dataset_name == "CIFAR-10":
        ds = torchvision.datasets.CIFAR10(root='/home/raghav/dsl/datasets', train=True, download=True, transform=transforms.ToTensor())
        img_shape = (32, 32, 3)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "Local Folder" and local_path:
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        # Filter for directories to treat folder names as class labels
        subfolders = sorted([d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))])
        
        if not subfolders:
            raise ValueError("Local directory must contain subfolders where each folder name is a class label.")
        
        class_names = subfolders
        all_paths = []
        for i, folder in enumerate(class_names):
            folder_path = os.path.join(local_path, folder)
            for f in os.listdir(folder_path):
                if f.lower().endswith(valid_ext):
                    all_paths.append((os.path.join(folder_path, f), i))
        
        if not all_paths:
            raise ValueError("No valid images found in the specified class subfolders.")
        
        log_terminal(f"Found {len(class_names)} classes: {class_names}")
        n_samples = min(n_samples, len(all_paths))
        selected_indices = np.random.choice(len(all_paths), n_samples, replace=False)
        
        base64_list = []
        for idx in selected_indices:
            path, label = all_paths[idx]
            # Standardizing to 28x28 grayscale for robust classifier compatibility
            img = Image.open(path).convert('L').resize((28, 28))
            vec = np.array(img).flatten() / 255.0
            data_list.append(vec)
            label_list.append(label)
            base64_list.append(vector_to_base64(vec, (28, 28, 1)))
        
        return np.array(data_list), np.array(label_list), (28, 28, 1), base64_list, class_names

    # Standard Dataset Processing
    indices = np.random.choice(len(ds), n_samples, replace=False)
    base64_list = []
    for idx in indices:
        img, label = ds[idx]
        vec = img.view(-1).numpy()
        data_list.append(vec)
        label_list.append(label)
        base64_list.append(vector_to_base64(vec, img_shape))

    log_terminal(f"Data ingestion complete for {dataset_name}")
    return np.array(data_list), np.array(label_list), img_shape, base64_list, class_names

# --- Dimensionality Reduction ---
@st.cache_data
def get_projection(vectors, method="PCA"):
    log_terminal(f"Computing {method} projection...")
    if method == "PCA":
        model = PCA(n_components=2)
        proj = model.fit_transform(vectors)
        variance = model.explained_variance_
    else:
        model = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
        proj = model.fit_transform(vectors)
        variance = [100, 100] 
    return proj, variance

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="MFC Robustness Explorer", layout="wide")
    
    st.title("🛡️ Minimal Finite Covering (MFC) Analysis")
    st.markdown("""
    This analysis tool facilitates the exploration of **Minimal Finite Covering** for dataset condensation.
    By identifying a discrete skeleton $\mathcal{C}$ that forms an $\epsilon$-covering of the high-dimensional manifold $\mathcal{X}$,
    we derive provable bounds for the adversarial risk associated with robust optimization.
    """)

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Data Ingestion")
        dataset_choice = st.selectbox("Source Manifold", ["FashionMNIST", "MNIST", "CIFAR-10", "Local Folder"])
        
        local_path = None
        if dataset_choice == "Local Folder":
            local_path = st.text_input("Local Directory Path", placeholder="/path/to/dataset")
            if not local_path:
                st.warning("Please provide a valid directory path to proceed.")
        
        n_samples = st.slider("Population (N)", 100, 5000, 1000)
        
        st.divider()
        st.header("MFC Parameters")
        epsilon = st.slider("Covering Radius (ε)", 0.5, 20.0, 6.0, help="Metric distance in feature space.")
        viz_method = st.radio("Manifold Projection", ["PCA", "t-SNE"])
        
        st.divider()
        st.header("Robust Optimization")
        epochs = st.number_input("Epochs", 1, 100, 20)
        lr = st.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
        run_training = st.button("🚀 Execute Training", use_container_width=True)

    # --- Load Chosen Dataset ---
    try:
        if dataset_choice == "Local Folder" and not local_path:
            st.stop()
        
        vectors, labels, img_shape, base64_imgs, class_names = load_dataset(dataset_choice, n_samples, local_path)
    except Exception as e:
        st.error(f"Data loading failure: {e}")
        st.stop()

    # --- Execute MFC ---
    with st.spinner(f"Computing MFC for {dataset_choice}..."):
        mfc_indices = compute_mfc(vectors, epsilon)
        mfc_vectors = vectors[mfc_indices]
        mfc_labels = labels[mfc_indices]

    # --- Visualization ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Manifold Geometry ({viz_method})")
        projections, variance = get_projection(vectors, viz_method)
        
        # Scale epsilon projection for visual circle shapes
        if viz_method == "PCA":
            std_dev = np.sqrt(variance)
            scale_factor = np.mean(std_dev) / np.sqrt(vectors.shape[1])
            projected_radius = epsilon * scale_factor * 2.5 
        else:
            projected_radius = (epsilon / 15.0) * (np.max(projections) - np.min(projections)) * 0.05

        fig = go.Figure()
        hover_tpl = "<b>Class: %{customdata}</b><extra></extra>"

        # Background distribution
        fig.add_trace(go.Scattergl(
            x=projections[:, 0], y=projections[:, 1],
            mode='markers',
            marker=dict(size=4, color='#64748b', opacity=0.3),
            name='Manifold Population',
            customdata=[class_names[l] if l < len(class_names) else f"Label {l}" for l in labels],
            hovertemplate=hover_tpl
        ))

        # Anchor covering balls
        balls_to_draw = mfc_indices[:300]
        shapes = []
        for idx in balls_to_draw:
            x_c, y_c = projections[idx]
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=x_c - projected_radius, y0=y_c - projected_radius,
                x1=x_c + projected_radius, y1=y_c + projected_radius,
                line_color="rgba(37, 99, 235, 0.15)",
                fillcolor="rgba(37, 99, 235, 0.04)",
                layer="below"
            ))

        # MFC Core anchors
        fig.add_trace(go.Scattergl(
            x=projections[mfc_indices, 0], y=projections[mfc_indices, 1],
            mode='markers',
            marker=dict(size=8, color='#2563eb', line=dict(width=1.5, color='white')),
            name='MFC Core (𝒞)',
            customdata=[class_names[l] if l < len(class_names) else f"Label {l}" for l in mfc_labels],
            hovertemplate=hover_tpl
        ))

        fig.update_layout(
            shapes=shapes, height=650, template="simple_white",
            xaxis=dict(showgrid=False, zeroline=False, title=f"{viz_method} 1"),
            yaxis=dict(showgrid=False, zeroline=False, title=f"{viz_method} 2"),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Condensation Stats")
        ratio = (len(mfc_indices) / n_samples) * 100
        st.metric("Covering Set Size (|𝒞|)", len(mfc_indices))
        st.metric("Compression Efficiency", f"{100-ratio:.1f}% Reduction")
        
        st.divider()
        st.subheader("Core Sample Gallery")
        num_display = min(12, len(mfc_indices))
        gallery_cols = st.columns(3)
        for i in range(num_display):
            with gallery_cols[i % 3]:
                # Dynamic reshape based on data source
                img_data = mfc_vectors[i].reshape(img_shape[0], img_shape[1], img_shape[2]) if img_shape[2] == 3 else mfc_vectors[i].reshape(img_shape[0], img_shape[1])
                label_idx = mfc_labels[i]
                cap = class_names[label_idx] if label_idx < len(class_names) else f"Anchor {i}"
                st.image(img_data, caption=cap, use_container_width=True)
        
        if len(mfc_indices) > 0:
            # Bar chart mapped to folder names
            mfc_counts = pd.Series([class_names[l] for l in mfc_labels]).value_counts()
            st.bar_chart(mfc_counts)

    # --- Robust Optimization Loop ---
    if run_training:
        st.divider()
        st.subheader("Robust Training Session")
        log_terminal(f"Initiating Pytorch session: training on {len(mfc_indices)} anchors across classes: {class_names}")
        
        X_train = torch.tensor(mfc_vectors, dtype=torch.float32)
        y_train = torch.tensor(mfc_labels, dtype=torch.long)
        
        # Dynamic input_dim and class count
        model = RobustClassifier(input_dim=vectors.shape[1], num_classes=len(class_names))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        progress = st.progress(0)
        loss_chart = []
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            loss_chart.append(loss.item())
            progress.progress((epoch + 1) / epochs)
            if (epoch + 1) % 5 == 0:
                log_terminal(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
            
        st.success("Optimization Sequence Terminated Successfully")
        st.line_chart(loss_chart)

if __name__ == "__main__":
    main()
