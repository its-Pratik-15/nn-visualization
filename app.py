import streamlit as st
import numpy as np
import pandas as pd
from utils import generate_dataset
from visualizations import plot_activation_functions, plot_decision_boundary, plot_loss_curve, draw_3b1b_network_html
from nn_core import CustomNeuralNetwork
import nn_core
import streamlit.components.v1 as components

st.set_page_config(page_title="Neural Networks Explorer", layout="wide", page_icon="🧠")

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
.metric-box { background: rgba(43, 48, 62, 0.6); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #2A2A2A; margin: 10px 0; }
.metric-title { font-size: 0.9rem; color: #A0AEC0; }
.metric-value { font-size: 1.5rem; font-weight: bold; color: #4CAF50; }
.explanation-box { background: rgba(33, 150, 243, 0.1); padding: 15px; border-left: 4px solid #2196F3; border-radius: 4px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Neural Networks Explorer")
st.markdown("### Learn How Neural Networks Compute, Predict, and Learn")

# SIDEBAR
st.sidebar.header("🕹️ Global Controls")
dataset_name = st.sidebar.selectbox("1. Dataset Selection", ["Nonlinear (Moons)", "Noisy Circles", "Linear (Separable)"])
add_noise = st.sidebar.checkbox("Add Extra Noise", value=False)
noise_level = st.sidebar.slider("Noise Intensity", 0.0, 0.5, 0.15) if add_noise else 0.05

st.sidebar.markdown("---")
n_layers = st.sidebar.slider("2. Hidden Layers", 1, 5, 2)
n_neurons = st.sidebar.slider("3. Neurons per Hidden Layer", 1, 20, 8)
activation = st.sidebar.selectbox("4. Activation Function", ["ReLU", "Sigmoid", "Tanh"])

st.sidebar.markdown("---")
lr = st.sidebar.slider("5. Learning Rate", 0.001, 1.0, 0.05, step=0.005)
epochs = st.sidebar.slider("6. Epochs", 10, 1000, 200, step=10)
loss_fn = st.sidebar.selectbox("7. Loss Function", ["Cross-Entropy", "Mean Squared Error"])

st.sidebar.markdown("---")
# Session State to store model
if 'model' not in st.session_state: st.session_state.model = None
if 'losses' not in st.session_state: st.session_state.losses = []
if 'epoch_count' not in st.session_state: st.session_state.epoch_count = 0

layer_dims = [2] + [n_neurons]*n_layers + [1]

if st.sidebar.button("♻️ Initialize / Reset Model", use_container_width=True):
    st.session_state.model = CustomNeuralNetwork(layer_dims, activation=activation)
    st.session_state.losses = []
    st.session_state.epoch_count = 0
    st.sidebar.success("Model Initialized!")

X_tr, X_te, y_tr, y_te = generate_dataset(dataset_name, noise=noise_level)
X_train_nn, y_train_nn = X_tr, y_tr
X_test_nn, y_test_nn = X_te, y_te

if st.session_state.model is None or st.session_state.model.activation != activation or st.session_state.model.L != len(layer_dims)-1 or st.session_state.model.params['W1'].shape[0] != n_neurons:
    # Auto re-init if architecture changes
    st.session_state.model = CustomNeuralNetwork(layer_dims, activation=activation)
    st.session_state.losses = []
    st.session_state.epoch_count = 0

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "1️⃣ Neuron Computation", "2️⃣ Activation", "3️⃣ Architecture", 
    "4️⃣ Forward Prop", "5️⃣ Loss", "6️⃣ Backprop", "7️⃣ Model Complexity"
])

with t1:
    st.markdown('<div class="explanation-box"><b>Intuition:</b> A single neuron is the basic building block. It takes inputs, scales them by <b>weights</b>, adds a <b>bias</b>, and passes the sum through an <b>activation function</b> to fire an output.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Inputs & Weights")
        x1 = st.slider("Input x1", -5.0, 5.0, 1.0)
        x2 = st.slider("Input x2", -5.0, 5.0, 2.0)
        w1 = st.slider("Weight w1", -2.0, 2.0, 0.5)
        w2 = st.slider("Weight w2", -2.0, 2.0, -0.6)
        b = st.slider("Bias b", -2.0, 2.0, 0.1)
    with c2:
        st.subheader("Weighted Sum")
        z = w1*x1 + w2*x2 + b
        st.latex(r"z = w_1x_1 + w_2x_2 + b")
        st.latex(f"z = ({w1:.1f})({x1:.1f}) + ({w2:.1f})({x2:.1f}) + ({b:.1f})")
        st.markdown(f'<div class="metric-box"><div class="metric-title">Z (Raw Output)</div><div class="metric-value">{z:.4f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.subheader("Activation Output")
        if activation == 'ReLU': a = nn_core.relu(z)
        elif activation == 'Sigmoid': a = nn_core.sigmoid(z)
        else: a = nn_core.tanh(z)
        st.latex(f"a = \\text{{{activation}}}(z)")
        st.markdown(f'<div class="metric-box"><div class="metric-title">Activated Output (a)</div><div class="metric-value">{a:.4f}</div></div>', unsafe_allow_html=True)
    st.success("**Key Takeaway:** Weights vote on the importance of inputs. Bias shifts the threshold. The Activation function makes it non-linear!")

with t2:
    st.markdown('<div class="explanation-box"><b>Intuition:</b> Without activation functions, neural networks would collapse into a simple linear equation regardless of how deep they are. They introduce "curves" into the math.</div>', unsafe_allow_html=True)
    st.pyplot(plot_activation_functions())
    st.success("**Key Takeaway:** ReLU zeros out negative values (fast, powerful). Sigmoid squashes numbers between 0 and 1 (great for probabilities). Tanh squashes between -1 and 1.")

with t3:
    st.markdown(f'<div class="explanation-box"><b>Intuition:</b> Connecting neurons in layers creates a deep network capable of learning complex abstractions. Data goes from Input -> Hidden Layers -> Output.</div>', unsafe_allow_html=True)
    st.write(f"### Flow: 2 Inputs → {n_layers} Hidden Layers (x{n_neurons} neurons) → 1 Output")
    
    weights = [st.session_state.model.params['W' + str(l)] for l in range(1, len(layer_dims))]
    components.html(draw_3b1b_network_html(layer_dims, weights=weights), height=420)
    
    total_params = sum([layer_dims[i]*layer_dims[i-1] + layer_dims[i] for i in range(1, len(layer_dims))])
    st.markdown(f'<div style="text-align:center;"><b>Total Trainable Parameters (Weights + Biases): {total_params}</b></div>', unsafe_allow_html=True)

with t4:
    st.markdown('<div class="explanation-box"><b>Intuition:</b> In Forward Propagation, an input sample moves sequentially through the matrix multiplications and activations of each layer to produce a final prediction.</div>', unsafe_allow_html=True)
    sample_idx = st.slider("Select Sample to trace", 0, X_train_nn.shape[1]-1, 0)
    x_sample = X_train_nn[:, sample_idx:sample_idx+1]
    y_sample_true = y_train_nn[0, sample_idx]
    
    st.write(f"**Tracing Input Data:** `x1` = {x_sample[0,0]:.2f}, `x2` = {x_sample[1,0]:.2f} | **True Label:** {y_sample_true}")
    pred = st.session_state.model.forward_pass(x_sample)
    
    weights = [st.session_state.model.params['W' + str(l)] for l in range(1, len(layer_dims))]
    acts = [st.session_state.model.cache['A0']]
    for l in range(1, len(layer_dims)):
        acts.append(st.session_state.model.cache[f'A{l}'])
            
    components.html(draw_3b1b_network_html(layer_dims, weights=weights, activations=acts), height=420)
            
    st.info(f"**Final Output Layer Prediction Probability:** {pred[0,0]:.4f}")

with t5:
    st.markdown('<div class="explanation-box"><b>Intuition:</b> The Loss function scores how badly the network is predicting. High Loss = Wrong direction. Low Loss = Predicting correctly.</div>', unsafe_allow_html=True)
    
    colA, colB = st.columns([1, 2])
    with colA:
         if st.button("🚂 Train 10 Epochs"):
             for _ in range(10):
                 preds = st.session_state.model.forward_pass(X_train_nn)
                 loss = st.session_state.model.compute_loss(preds, y_train_nn, loss_fn)
                 grads = st.session_state.model.backward_pass(y_train_nn, preds, loss_fn)
                 st.session_state.model.update_parameters(grads, lr)
                 st.session_state.losses.append(loss)
                 st.session_state.epoch_count += 1
             st.success(f"Trained up to {st.session_state.epoch_count} epochs!")
             
         st.metric("Current Epoch", st.session_state.epoch_count)
         if len(st.session_state.losses) > 0:
             st.metric("Current Loss", f"{st.session_state.losses[-1]:.4f}")
             
    with colB:
        if len(st.session_state.losses) > 0:
            st.pyplot(plot_loss_curve(st.session_state.losses))
        else:
            st.write("No training data yet. Click the button to the left to train!")

with t6:
    st.markdown('<div class="explanation-box"><b>Intuition:</b> Backpropagation looks at the loss error and uses calculus (derivatives) to figure out how much each weight contributed to the error. It computes a "Gradient" (slope), and Gradient Descent steps downhill to update parameters.</div>', unsafe_allow_html=True)
    if st.button("🔄 Compute Gradients (One Forward-Backward Step)"):
        preds = st.session_state.model.forward_pass(X_train_nn)
        loss = st.session_state.model.compute_loss(preds, y_train_nn, loss_fn)
        grads = st.session_state.model.backward_pass(y_train_nn, preds, loss_fn)
        
        st.write(f"**Loss Before Step:** {loss:.4f}")
        st.write(f"Using **{loss_fn}** derivatives backward.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Gradients Computed:")
            st.write("*(Showing Max absolute gradient per parameter matrix)*")
            for k, v in grads.items():
                if 'W' in k:
                     st.write(f"Gradient of **{k}**: {np.max(np.abs(v)):.4f}")
        with c2:
            st.write("### Updating Parameters:")
            st.latex(r"W_{new} = W_{old} - \alpha \cdot \text{Gradient}")
            st.write(f"Learning Rate $\\alpha$ = {lr}")
            
        st.session_state.model.update_parameters(grads, lr)
        new_preds = st.session_state.model.forward_pass(X_train_nn)
        new_loss = st.session_state.model.compute_loss(new_preds, y_train_nn, loss_fn)
        st.success(f"Parameters updated! Loss is now: {new_loss:.4f}")

with t7:
    st.markdown('<div class="explanation-box"><b>Intuition:</b> Larger models (deep/wide) can learn complex decision boundaries. However, if they are too large, they memorize the noise (Overfitting). If they are too small, they cannot learn the pattern (Underfitting).</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Run Full Training Cycle", type="primary"):
        my_bar = st.progress(0)
        for e in range(epochs):
            preds = st.session_state.model.forward_pass(X_train_nn)
            loss = st.session_state.model.compute_loss(preds, y_train_nn, loss_fn)
            grads = st.session_state.model.backward_pass(y_train_nn, preds, loss_fn)
            st.session_state.model.update_parameters(grads, lr)
            st.session_state.losses.append(loss)
            st.session_state.epoch_count += 1
            if e % max(1, epochs//20) == 0:
                my_bar.progress(e / epochs)
        my_bar.progress(1.0)
        
    if st.session_state.epoch_count > 0:
        st.write(f"Trained strictly using handcrafted NumPy arrays for **{st.session_state.epoch_count} epochs.**")
        
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.pyplot(plot_decision_boundary(st.session_state.model, X_train_nn, y_train_nn[0], f"Decision Boundary ({dataset_name})"))
        
        with col_res2:
            train_preds = st.session_state.model.predict(X_train_nn)
            test_preds = st.session_state.model.predict(X_test_nn)
            train_acc = np.mean(train_preds == y_train_nn)
            test_acc = np.mean(test_preds == y_test_nn)
            
            st.markdown(f'<div class="metric-box"><div class="metric-title">Train Accuracy</div><div class="metric-value">{train_acc*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><div class="metric-title">Test Accuracy</div><div class="metric-value">{test_acc*100:.1f}%</div></div>', unsafe_allow_html=True)
            
            if train_acc < 0.65:
                st.warning("Model is **Underfitting**. Increase layers, neurons, or training epochs.")
            elif train_acc > 0.95 and (train_acc - test_acc) > 0.05:
                st.warning("Model might be **Overfitting**. The gap between train and test accuracy is large.")
            elif train_acc >= 0.65:
                st.success("Model generalized well!")
