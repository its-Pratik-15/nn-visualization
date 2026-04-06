import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Page Config
st.set_page_config(page_title="NN Visualizer", layout="wide", page_icon="🧠")

# Custom CSS for Premium Design
st.markdown("""
<style>
    .concept-box {
        background: rgba(43, 48, 62, 0.6);
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 25px;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .concept-title {
        color: #FF4B4B;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .main-title {
        text-align: center;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>How Do Neural Networks Learn?</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>An interactive journey visualizing neurons, architecture, and backpropagation.</div>", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🕹️ Network Parameters")

dataset_type = st.sidebar.selectbox("1️⃣ Dataset Selection", ["Moons (Non-linear)", "Circles (Enclosed)", "Linearly Separable"])
add_noise = st.sidebar.toggle("Add Noise to Data", True)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.15) if add_noise else 0.0

st.sidebar.markdown("---")
n_layers = st.sidebar.slider("2️⃣ Hidden Layers (Depth)", 1, 5, 2)
n_neurons = st.sidebar.slider("3️⃣ Neurons per Layer (Width)", 1, 30, 8)

st.sidebar.markdown("---")
activation = st.sidebar.selectbox("4️⃣ Activation Function", ["ReLU", "Sigmoid", "Tanh"])
loss_function = st.sidebar.selectbox("5️⃣ Loss Function", ["Cross-Entropy (Log Loss)", "Mean Squared Error (MSE)"])

st.sidebar.markdown("---")
epochs = st.sidebar.slider("6️⃣ Training Epochs", 10, 1000, 250, step=10)
lr = st.sidebar.slider("7️⃣ Learning Rate", 0.001, 1.0, 0.03, step=0.005)

# Generate Dataset
@st.cache_data
def get_data(d_type, noise):
    if "Moons" in d_type:
        return make_moons(n_samples=300, noise=noise, random_state=42)
    elif "Circles" in d_type:
        return make_circles(n_samples=300, noise=noise, factor=0.5, random_state=42)
    else:
        rng = np.random.RandomState(42)
        X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=42, n_clusters_per_class=1)
        if noise > 0: X += rng.randn(*X.shape) * noise
        return X, y

X, y = get_data(dataset_type, noise_level)

# ----------------- HELPER FUNCTIONS -----------------
def act_fn(z, name):
    if name == "ReLU": return np.maximum(0, z)
    elif name == "Sigmoid": return 1 / (1 + np.exp(-z))
    else: return np.tanh(z)

act_mapping_sklearn = {"ReLU": "relu", "Sigmoid": "logistic", "Tanh": "tanh"}


# ----------------- TABS -----------------
tabs = st.tabs([
    "1️⃣ Neuron Computation", 
    "2️⃣ Activation Functions", 
    "3️⃣ Network Architecture", 
    "4️⃣ Forward Propagation", 
    "5️⃣ Loss Function", 
    "6️⃣ Backpropagation", 
    "7️⃣ Train & Observe Complexity"
])

# TAB 1: Neuron Computation
with tabs[0]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: The Perceptron</div>
            A neuron acts as a tiny gatekeeper. It takes inputs, scales them by <b>Weights</b>, adds a <b>Bias</b>, and passes the result through an <b>Activation Function</b>.<br>
            <i>What to observe: Notice how the final output changes non-linearly depending on the activation function chosen!</i>
        </div>
    """, unsafe_allow_html=True)
    
    col1, mid, col2 = st.columns([1, 0.1, 1])
    with col1:
        st.subheader("Compute Weighted Sum (z)")
        x1 = st.number_input("Input 1 (x₁)", value=2.0)
        w1 = st.slider("Weight 1 (w₁)", -5.0, 5.0, 1.5)
        x2 = st.number_input("Input 2 (x₂)", value=-1.0)
        w2 = st.slider("Weight 2 (w₂)", -5.0, 5.0, -0.5)
        b = st.slider("Bias (b)", -5.0, 5.0, 0.5)
        
        z = (w1 * x1) + (w2 * x2) + b
        st.latex(r"z = w_1x_1 + w_2x_2 + b")
        st.latex(f"z = ({w1:.2f} \\times {x1:.2f}) + ({w2:.2f} \\times {x2:.2f}) + {b:.2f} = {z:.2f}")

    with col2:
        st.subheader("Apply Activation (a)")
        st.markdown(f"Using **{activation}** (Selected in Sidebar):")
        st.latex(r"a = f_{activation}(z)")
        
        a = act_fn(z, activation)
        
        fig, ax = plt.subplots(figsize=(5,3))
        x_range = np.linspace(-10, 10, 100)
        y_range = act_fn(x_range, activation)
        ax.plot(x_range, y_range, color='#FF4B4B', lw=2)
        ax.scatter([z], [a], color='white', edgecolor='red', s=100, zorder=5, label=f'Current (z={z:.2f}, a={a:.2f})')
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        ax.grid(color='#2A2A2A')
        ax.legend(loc='upper left', facecolor='#0E1117', edgecolor='none', labelcolor='white')
        st.pyplot(fig)
        
        st.success(f"Neuron Output: **{a:.4f}**")

# TAB 2: Activation Functions
with tabs[1]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: Introducing Non-Linearity</div>
            Without activation functions, a Neural Network with millions of layers would mathematically collapse into a single linear equation. <br>
            <i>What to observe: Look at gradients (slopes). Sigmoid flattens out at the edges (vanishing gradient problem), while ReLU maintains a strong signal for all positive numbers.</i>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"Currently active for the network: **{activation}**")
    
    x_val = np.linspace(-8, 8, 300)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0E1117')
    
    funcs = [
        ("ReLU", np.maximum(0, x_val), '#4CAF50', 'max(0, z)'),
        ("Sigmoid", 1/(1+np.exp(-x_val)), '#2196F3', '1 / (1 + e^-z)'),
        ("Tanh", np.tanh(x_val), '#FF9800', '(e^z - e^-z)/(e^z + e^-z)')
    ]
    
    for i, (fname, y_val, col, form) in enumerate(funcs):
        axes[i].set_facecolor('#0E1117')
        axes[i].plot(x_val, y_val, color=col, lw=3)
        axes[i].set_title(fname, color='white', size=16)
        axes[i].tick_params(colors='white')
        axes[i].grid(color='#2A2A2A')
        axes[i].text(-7, max(y_val)*0.8, form, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        axes[i].spines['bottom'].set_color('white')
        axes[i].spines['left'].set_color('white')
        
    st.pyplot(fig)

# TAB 3: Network Architecture
with tabs[2]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: Layering to Build Complexity</div>
            Networks stack multiple layers. The first layers might learn simple lines, the next layers combine them into shapes, and the final layers understand complex concepts.<br>
            <i>What to observe: Adjust the Depth and Width in the Sidebar and watch the architecture grow.</i>
        </div>
    """, unsafe_allow_html=True)
    
    st.write(f"### Current Topology: 2 Inputs → {n_layers} Hidden Layers (x{n_neurons} neurons) → 1 Output")
    
    visual_neurons = min(n_neurons, 10) 
    hidden_layers_str = ", ".join([str(visual_neurons)] * n_layers)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background-color: #0E1117; overflow: hidden; display: flex; justify-content: center; align-items: center; color: white; font-family: sans-serif; }}
            canvas {{ display: block; background-color: #0E1117; border-radius: 8px; border: 1px solid #2A2A2A; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        </style>
    </head>
    <body>
        <canvas id="nnCanvas"></canvas>
        <script>
            const canvas = document.getElementById('nnCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth - 30;
            canvas.height = 450;

            const colors = {{
                input: '#4CAF50',
                hidden: '#2196F3',
                output: '#FF4B4B',
                bg: '#0E1117',
                edge: 'rgba(255, 255, 255, 0.15)',
                particle: '#FFD700'
            }};

            const layerSizes = [2, {hidden_layers_str}, 1];
            const nodes = [];
            const edges = [];
            const particles = [];

            const width = canvas.width;
            const height = canvas.height;
            const xOffset = width / (layerSizes.length + 1);

            for (let i = 0; i < layerSizes.length; i++) {{
                const numNodes = layerSizes[i];
                const yOffset = height / (numNodes + 1);
                for (let j = 0; j < numNodes; j++) {{
                    let type = 'hidden';
                    if (i === 0) type = 'input';
                    if (i === layerSizes.length - 1) type = 'output';

                    nodes.push({{
                        x: xOffset * (i + 1),
                        y: yOffset * (j + 1),
                        layer: i,
                        type: type,
                        radius: i === 0 || i === layerSizes.length - 1 ? 18 : 14
                    }});
                }}
            }}

            for (let i = 0; i < layerSizes.length - 1; i++) {{
                const currentLayerNodes = nodes.filter(n => n.layer === i);
                const nextLayerNodes = nodes.filter(n => n.layer === i + 1);
                currentLayerNodes.forEach(n1 => {{
                    nextLayerNodes.forEach(n2 => {{
                        edges.push({{ from: n1, to: n2 }});
                    }});
                }});
            }}

            function spawnParticle() {{
                const startNodes = nodes.filter(n => n.layer === 0);
                const startNode = startNodes[Math.floor(Math.random() * startNodes.length)];
                
                let currentNode = startNode;
                let path = [currentNode];
                while (currentNode.layer < layerSizes.length - 1) {{
                    const connectedEdges = edges.filter(e => e.from === currentNode);
                    const randomEdge = connectedEdges[Math.floor(Math.random() * connectedEdges.length)];
                    currentNode = randomEdge.to;
                    path.push(currentNode);
                }}

                particles.push({{
                    path: path,
                    targetIndex: 1,
                    x: path[0].x,
                    y: path[0].y,
                    speed: 2.5 + Math.random() * 2,
                    progress: 0
                }});
            }}

            function draw() {{
                ctx.fillStyle = colors.bg;
                ctx.fillRect(0, 0, width, height);

                // Draw Edges
                ctx.lineWidth = 1.5;
                edges.forEach(e => {{
                    ctx.beginPath();
                    ctx.moveTo(e.from.x, e.from.y);
                    ctx.lineTo(e.to.x, e.to.y);
                    ctx.strokeStyle = colors.edge;
                    ctx.stroke();
                }});

                // Spawn Particles Flowing
                if (Math.random() < 0.15) spawnParticle();

                // Draw Particles
                for (let i = particles.length - 1; i >= 0; i--) {{
                    const p = particles[i];
                    const fromNode = p.path[p.targetIndex - 1];
                    const toNode = p.path[p.targetIndex];
                    
                    const dx = toNode.x - fromNode.x;
                    const dy = toNode.y - fromNode.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    p.progress += p.speed / dist;
                    
                    if (p.progress >= 1) {{
                        p.targetIndex++;
                        p.progress = 0;
                        if (p.targetIndex >= p.path.length) {{
                            particles.splice(i, 1);
                            continue;
                        }}
                    }}
                    
                    const currentFrom = p.path[p.targetIndex - 1];
                    const currentTo = p.path[p.targetIndex];
                    const px = currentFrom.x + (currentTo.x - currentFrom.x) * p.progress;
                    const py = currentFrom.y + (currentTo.y - currentFrom.y) * p.progress;

                    ctx.beginPath();
                    ctx.arc(px, py, 4, 0, Math.PI * 2);
                    ctx.fillStyle = colors.particle;
                    ctx.shadowBlur = 15;
                    ctx.shadowColor = colors.particle;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }}

                // Draw Nodes
                nodes.forEach(n => {{
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
                    ctx.fillStyle = colors[n.type];
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#FFFFFF';
                    ctx.stroke();
                }});

                requestAnimationFrame(draw);
            }}
            
            draw();
        </script>
    </body>
    </html>
    """
    
    import streamlit.components.v1 as components
    components.html(html_code, height=480)
    
    if n_neurons > 10:
        st.warning(f"Visualization capped at 10 neurons per layer for smooth 60FPS animation, but the math uses {n_neurons}.")

# TAB 4: Forward Propagation
with tabs[3]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: Forward Pass</div>
            Data physically flows forward through the network by performing continuous Matrix Multiplications: <b>X · W + b</b>.<br>
            <i>What to observe: See how matrix dimensions adapt at each layer, squeezing data into new geometric dimensions.</i>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("Tracking a Batch of 32 Samples flowing through the network:")
    st.code("Input X Tensor Shape: (32, 2)", language='python')
    
    for l in range(n_layers):
        in_dim = 2 if l == 0 else n_neurons
        st.markdown(f"**Layer {l+1}**")
        st.code(f"Weight Matrix W{l+1}: ({in_dim}, {n_neurons})\nBias Vector b{l+1}: (1, {n_neurons})\nOperation: Z = X(32, {in_dim}) @ W({in_dim}, {n_neurons}) + b -> Out: (32, {n_neurons})\nApplying {activation}() -> A{l+1} Shape: (32, {n_neurons})", language="text")

    st.markdown("**Output Layer**")
    st.code(f"Weight Matrix W_out: ({n_neurons}, 1)\nOperation: Z = A{n_layers}(32, {n_neurons}) @ W_out({n_neurons}, 1) + b -> Prediction Y_hat: (32, 1)", language="text")

# TAB 5: Loss Function
with tabs[4]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: Measuring the Error (Loss)</div>
            The Loss function grades the network. High loss = Bad predictions. Low loss = Good predictions. The goal of training is purely to minimize this singular number.<br>
            <i>What to observe: See how Cross-Entropy penalizes confident, yet incorrect predictions far harder than MSE!</i>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"Selected Metric: **{loss_function}**")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0E1117')
    
    # MSE Plot
    axes[0].set_facecolor('#0E1117')
    y_preds = np.linspace(-1, 2, 100)
    axes[0].plot(y_preds, (1.0 - y_preds)**2, color='#FF4B4B', lw=3)
    axes[0].set_title("Mean Squared Error (True Y=1)", color='white')
    axes[0].set_xlabel("Predicted Value", color='white')
    axes[0].tick_params(colors='white')
    axes[0].grid(color='#2A2A2A')
    
    # Log Loss Plot
    axes[1].set_facecolor('#0E1117')
    y_preds_prob = np.linspace(0.01, 0.99, 100)
    axes[1].plot(y_preds_prob, -np.log(y_preds_prob), color='#2196F3', lw=3, label="True Y=1")
    axes[1].plot(y_preds_prob, -np.log(1 - y_preds_prob), color='#4CAF50', lw=3, label="True Y=0", linestyle="--")
    axes[1].set_title("Cross-Entropy Log Loss", color='white')
    axes[1].set_xlabel("Predicted Probability", color='white')
    axes[1].legend(facecolor='#0E1117', edgecolor='white', labelcolor='white')
    axes[1].tick_params(colors='white')
    axes[1].grid(color='#2A2A2A')
    
    st.pyplot(fig)

# TAB 6: Backpropagation
with tabs[5]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: Backward Pass & Gradient Descent</div>
            Using Calculus (the Chain Rule), Backprop calculates how much every single weight contributed to the Loss. We then nudge the weights slightly in the opposite direction of the slope.<br>
            <i>What to observe: The learning rate acts as the "step size" going down the hill. Too big = you jump out of the valley. Too small = takes forever to converge.</i>
        </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"W_{new} = W_{old} - \text{Learning\_Rate} \times \frac{\partial \text{Loss}}{\partial W_{old}}")
    st.write(f"Current Learning Rate: $\\alpha = {lr}$")
    
    # 3D surface simulation visualization
    st.markdown("**(Visualization of the Error Landscape)**")
    w1_vals = np.linspace(-3, 3, 50)
    w2_vals = np.linspace(-3, 3, 50)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Loss_Z = W1**2 + W2**2 + 5*np.sin(W1)*np.cos(W2)
    
    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    v = ax.contourf(W1, W2, Loss_Z, levels=30, cmap='magma')
    ax.annotate("Start", xy=(2.5, 2.5), xytext=(2.0, 2.0),
            arrowprops=dict(facecolor='white', shrink=0.05), color="white", fontsize=12)
    ax.annotate("Minimum Loss Valley", xy=(0, 0), xytext=(-2.5, -2.5),
            arrowprops=dict(facecolor='green', shrink=0.05), color="lightgreen", fontsize=12)
    ax.tick_params(colors='white')
    
    st.pyplot(fig)

# TAB 7: Training & Overfitting
with tabs[6]:
    st.markdown("""
        <div class="concept-box">
            <div class="concept-title">Concept: Model Complexity In Action</div>
            Does your network have enough capacity to separate the data? Or does it have *too much* capacity and is memorizing the noise (Overfitting)?<br>
            <i>Instructions: Click 'Train Model' below to see the decision boundaries generated by your customized network! Play with Layers and Neurons to see the difference.</i>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 TRAIN NETWORK & PLOT RESULTS", type="primary"):
        with st.spinner("Executing Forward Pass, Loss Computation, and Backpropagation..."):
            
            clf = MLPClassifier(
                hidden_layer_sizes=(n_neurons,) * n_layers,
                activation=act_mapping_sklearn[activation],
                max_iter=epochs,
                alpha=0.01,
                learning_rate_init=lr,
                random_state=42
            )
            
            clf.fit(X, y)
            acc = clf.score(X, y)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            # Plot decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                
            Z = Z.reshape(xx.shape)
            cm = plt.cm.RdBu
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
            
            # Plot points
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='white', s=40)
            
            ax.set_title(f"Model Decision Boundary (Testing Accuracy: {acc*100:.1f}%)", color='white', fontsize=16)
            ax.tick_params(colors='white')
            
            st.pyplot(fig)
            
            if acc < 0.7:
                st.error("📉 The model is Underfitting! Try adding more layers/neurons or increasing epochs.")
            elif acc > 0.95 and noise_level > 0.2:
                st.warning("⚠️ The model might be Overfitting! It's learning the noise heavily. Try adding noise or reducing layers/neurons.")
            else:
                st.success("🎯 Good Fit! The model learned the underlying patterns beautifully.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Built for the AI/ML Projects Class by AI Assistant.</p>", unsafe_allow_html=True)
