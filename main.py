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

def log_terminal(message, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def vector_to_base64(vector, shape=(28, 28, 1)):
    if len(shape) == 3 and shape[2] == 3:
        img_array = (vector.reshape(shape[0], shape[1], 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'RGB')
    else:
        img_array = (vector.reshape(shape[0], shape[1]) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

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

def compute_mfc(vectors, epsilon):
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
            log_terminal(f"Iteration {iteration}: {percent:.1f}% covered")
    duration = time.time() - start_time
    log_terminal(f"MFC Finished. Found {len(covering_indices)} anchors in {duration:.2f}s")
    return covering_indices

@st.cache_data
def load_dataset(dataset_name, n_samples=1000, local_path=None):
    log_terminal(f"Dataset Loading Triggered: {dataset_name}")
    data_list = []
    label_list = []
    class_names = []
    img_shape = (28, 28, 1)

    if dataset_name == "2D Toy Manifold":
        centers = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]])
        class_names = ["Cluster A", "Cluster B", "Cluster C", "Cluster D"]
        for i in range(n_samples):
            label = i % 4
            point = centers[label] + np.random.normal(0, 0.6, size=2)
            data_list.append(point)
            label_list.append(label)
        return np.array(data_list), np.array(label_list), (1, 1, 1), ["" for _ in range(n_samples)], class_names

    if not os.path.exists('./data'):
        os.makedirs('./data')

    if dataset_name == "MNIST":
        ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        class_names = [str(i) for i in range(10)]
    elif dataset_name == "FashionMNIST":
        ds = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    elif dataset_name == "CIFAR-10":
        ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        img_shape = (32, 32, 3)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "Local Folder" and local_path:
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        subfolders = sorted([d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))])
        if not subfolders: raise ValueError("Directory must contain subfolders.")
        class_names = subfolders
        all_paths = []
        for i, folder in enumerate(class_names):
            folder_path = os.path.join(local_path, folder)
            for f in os.listdir(folder_path):
                if f.lower().endswith(valid_ext): all_paths.append((os.path.join(folder_path, f), i))
        n_samples = min(n_samples, len(all_paths))
        selected_indices = np.random.choice(len(all_paths), n_samples, replace=False)
        base64_list = []
        for idx in selected_indices:
            path, label = all_paths[idx]
            img = Image.open(path).convert('L').resize((28, 28))
            vec = np.array(img).flatten() / 255.0
            data_list.append(vec)
            label_list.append(label)
            base64_list.append(vector_to_base64(vec, (28, 28, 1)))
        return np.array(data_list), np.array(label_list), (28, 28, 1), base64_list, class_names

    indices = np.random.choice(len(ds), n_samples, replace=False)
    base64_list = []
    for idx in indices:
        img, label = ds[idx]
        vec = img.view(-1).numpy()
        data_list.append(vec)
        label_list.append(label)
        base64_list.append(vector_to_base64(vec, img_shape))
    return np.array(data_list), np.array(label_list), img_shape, base64_list, class_names

@st.cache_data
def get_projection(vectors, method="PCA"):
    if vectors.shape[1] == 2:
        return vectors, [1.0, 1.0]
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

def main():
    st.set_page_config(page_title="Compresso Explorer", layout="wide")
    st.title("🛡️ Compresso: MFC Analysis Tool")
    st.markdown("This tool implements the Minimal Finite Covering (MFC) algorithm to generate a discrete skeleton of a dataset for robust optimization.")

    with st.sidebar:
        st.header("Data Source")
        dataset_choice = st.selectbox("Manifold", ["2D Toy Manifold", "FashionMNIST", "MNIST", "CIFAR-10", "Local Folder"])
        local_path = None
        if dataset_choice == "Local Folder":
            local_path = st.text_input("Local Directory Path")
        n_samples = st.slider("Population (N)", 100, 5000, 800)
        st.divider()
        st.header("Parameters")
        default_eps = 0.8 if dataset_choice == "2D Toy Manifold" else 6.0
        epsilon = st.slider("Radius (ε)", 0.1, 20.0, default_eps)
        viz_method = st.radio("Visualization", ["PCA", "t-SNE"])
        st.divider()
        st.header("Training")
        epochs = st.number_input("Epochs", 1, 100, 20)
        lr = st.number_input("Learning Rate", 0.001, 0.1, 0.01)
        run_training = st.button("🚀 Execute Training", use_container_width=True)

    vectors, labels, img_shape, _, class_names = load_dataset(dataset_choice, n_samples, local_path)
    with st.spinner("Computing MFC..."):
        mfc_indices = compute_mfc(vectors, epsilon)
        mfc_vectors = vectors[mfc_indices]
        mfc_labels = labels[mfc_indices]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Manifold Covering Visualization")
        projections, variance = get_projection(vectors, viz_method)
        if dataset_choice == "2D Toy Manifold":
            projected_radius = epsilon
        elif viz_method == "PCA":
            std_dev = np.sqrt(variance)
            scale_factor = np.mean(std_dev) / np.sqrt(vectors.shape[1])
            projected_radius = epsilon * scale_factor * 2.5 
        else:
            projected_radius = (epsilon / 15.0) * (np.max(projections) - np.min(projections)) * 0.05

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=projections[:, 0], y=projections[:, 1], mode='markers',
            marker=dict(size=4, color='#64748b', opacity=0.3), name='Full Manifold (𝒳)',
            customdata=[class_names[l] if l < len(class_names) else f"Label {l}" for l in labels],
            hovertemplate="<b>Class: %{customdata}</b><extra></extra>"))

        shapes = []
        for idx in mfc_indices[:300]:
            x_c, y_c = projections[idx]
            shapes.append(dict(type="circle", xref="x", yref="y", x0=x_c - projected_radius, y0=y_c - projected_radius,
                x1=x_c + projected_radius, y1=y_c + projected_radius, line_color="rgba(37, 99, 235, 0.15)",
                fillcolor="rgba(37, 99, 235, 0.04)", layer="below"))

        fig.add_trace(go.Scattergl(x=projections[mfc_indices, 0], y=projections[mfc_indices, 1], mode='markers',
            marker=dict(size=8, color='#2563eb', line=dict(width=1.5, color='white')), name='MFC Core (𝒞)',
            customdata=[class_names[l] if l < len(class_names) else f"Label {l}" for l in mfc_labels],
            hovertemplate="<b>Class: %{customdata}</b><extra></extra>"))

        fig.update_layout(shapes=shapes, height=650, template="simple_white",
            xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False), margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Metrics")
        ratio = (len(mfc_indices) / n_samples) * 100
        st.metric("Set Size (|𝒞|)", len(mfc_indices))
        st.metric("Reduction Efficiency", f"{100-ratio:.1f}%")
        if dataset_choice != "2D Toy Manifold":
            st.divider()
            st.subheader("Anchor Gallery")
            gallery_cols = st.columns(3)
            for i in range(min(9, len(mfc_indices))):
                with gallery_cols[i % 3]:
                    img_data = mfc_vectors[i].reshape(img_shape[0], img_shape[1], img_shape[2]) if img_shape[2] == 3 else mfc_vectors[i].reshape(img_shape[0], img_shape[1])
                    st.image(img_data, caption=class_names[mfc_labels[i]], use_container_width=True)
        st.bar_chart(pd.Series([class_names[l] for l in mfc_labels]).value_counts())

    if run_training:
        st.divider()
        st.subheader("Pytorch Training")
        X_train = torch.tensor(mfc_vectors, dtype=torch.float32)
        y_train = torch.tensor(mfc_labels, dtype=torch.long)
        model = RobustClassifier(input_dim=vectors.shape[1], num_classes=len(class_names))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        progress = st.progress(0)
        loss_chart = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()
            loss_chart.append(loss.item())
            progress.progress((epoch + 1) / epochs)
        st.line_chart(loss_chart)

if __name__ == "__main__": main()
