import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go

def plot_activation_functions():
    x = np.linspace(-6, 6, 200)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    
    # Theme settings
    fig.patch.set_facecolor('#121212')
    for a in ax:
        a.set_facecolor('#121212')
        a.tick_params(colors='#E0E0E0')
        a.spines['bottom'].set_color('#E0E0E0')
        a.spines['left'].set_color('#E0E0E0')
        a.grid(color='#333333')
    
    ax[0].plot(x, np.maximum(0, x), color='#4CAF50', lw=2)
    ax[0].set_title("ReLU", color='#E0E0E0')
    
    ax[1].plot(x, 1/(1+np.exp(-x)), color='#2196F3', lw=2)
    ax[1].set_title("Sigmoid", color='#E0E0E0')
    
    ax[2].plot(x, np.tanh(x), color='#FF9800', lw=2)
    ax[2].set_title("Tanh", color='#E0E0E0')
    
    return fig

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # X shape: (2, m), y shape: (m,)
    h = 0.05
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.tick_params(colors='#E0E0E0')
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF4B4B', '#2196F3'])
    
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.6)
    ax.scatter(X[0, :], X[1, :], c=y, cmap=cm_bright, edgecolors='#121212', s=40)
    ax.set_title(title, color='#E0E0E0')
    return fig

def plot_loss_curve(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.tick_params(colors='#E0E0E0')
    ax.grid(color='#333333')
    
    ax.plot(losses, label="Training Error", color="#FF4B4B", lw=3)
    ax.set_title("Loss Reduction Over Time", color='#E0E0E0')
    ax.set_xlabel("Epoch", color='#E0E0E0')
    ax.set_ylabel("Loss Measurement", color='#E0E0E0')
    ax.legend(facecolor='#121212', labelcolor='#E0E0E0')
    
    return fig

import json

def draw_3b1b_network_html(layer_sizes, weights=None, activations=None):
    weights_json = 'null'
    if weights is not None:
        vis_weights = []
        for w in weights:
            r_max = min(w.shape[0], 12)
            c_max = min(w.shape[1], 12)
            vis_weights.append(w[:r_max, :c_max].tolist())
        weights_json = json.dumps(vis_weights)
        
    activations_json = 'null'
    if activations is not None:
        vis_act = []
        for a in activations:
            vis_act.append(a.flatten()[:12].tolist())
        activations_json = json.dumps(vis_act)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ background-color: #121212; margin: 0; overflow: hidden; display: flex; justify-content: center; align-items: center; }}
            canvas {{ background-color: #121212; width: 100%; height: 400px; }}
        </style>
    </head>
    <body onresize="resizeCanvas()">
        <canvas id="networkCanvas"></canvas>
        <script>
            const canvas = document.getElementById('networkCanvas');
            const ctx = canvas.getContext('2d');
            
            function resizeCanvas() {{
                canvas.width = window.innerWidth - 10;
                canvas.height = 400;
            }}
            resizeCanvas();

            const layerSizes = {layer_sizes};
            const weights = {weights_json}; 
            const activations = {activations_json}; 

            const maxNodes = 12;
            const visualSizes = layerSizes.map(s => Math.min(s, maxNodes));
            
            let time = 0;
            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                let width = canvas.width;
                let height = canvas.height;
                let paddingX = width * 0.1;
                let paddingY = 40;
                
                let plotW = width - paddingX * 2;
                let plotH = height - paddingY * 2;
                
                let xOffset = plotW / Math.max(1, visualSizes.length - 1);
                
                let nodes = [];
                for(let l=0; l<visualSizes.length; l++) {{
                    let layerNodes = visualSizes[l];
                    let yOffset = layerNodes > 1 ? plotH / (layerNodes - 1) : 0;
                    let startY = layerNodes > 1 ? paddingY : height / 2;
                    
                    for(let n=0; n<layerNodes; n++) {{
                        let act = activations ? activations[l][n] : 0;
                        nodes.push({{
                            layer: l,
                            index: n,
                            x: paddingX + l * xOffset,
                            y: startY + n * yOffset,
                            activation: act
                        }});
                    }}
                }}
                
                time += 0.03;
                let currentWave = time % (visualSizes.length + 2) - 1;
                
                // Draw Edges
                if(weights && weights.length > 0) {{
                    for(let l=0; l<visualSizes.length-1; l++) {{
                        let currentNodes = nodes.filter(nd => nd.layer === l);
                        let nextNodes = nodes.filter(nd => nd.layer === l+1);
                        let w_matrix = weights[l]; 
                        
                        for(let i=0; i<nextNodes.length; i++) {{
                            for(let j=0; j<currentNodes.length; j++) {{
                                let w_val = w_matrix[i][j];
                                let absW = Math.abs(w_val);
                                let isPos = w_val > 0;
                                
                                // 3B1B Colors
                                let r = isPos ? 41 : 231;
                                let g = isPos ? 128 : 76;
                                let b = isPos ? 185 : 60;
                                
                                let opacity = Math.min(absW * 0.4, 0.8);
                                
                                if(activations) {{
                                    let edgeLoc = l + 0.5;
                                    let highlight = Math.max(0, 1 - Math.abs(currentWave - edgeLoc)*1.5);
                                    opacity = Math.min(1.0, opacity + highlight * 0.5);
                                }}
                                
                                ctx.beginPath();
                                ctx.moveTo(currentNodes[j].x, currentNodes[j].y);
                                ctx.lineTo(nextNodes[i].x, nextNodes[i].y);
                                ctx.lineWidth = 1 + Math.min(absW, 3.0);
                                ctx.strokeStyle = `rgba(${{r}}, ${{g}}, ${{b}}, ${{opacity}})`;
                                ctx.stroke();
                            }}
                        }}
                    }}
                }}
                
                // Draw Nodes
                nodes.forEach(n => {{
                    let brightness = 0.1;
                    if(activations) {{
                        let pulse = Math.max(0, 1 - Math.abs(currentWave - n.layer)*1.5);
                        brightness = Math.abs(n.activation) * (0.3 + 0.7 * pulse);
                    }}
                    
                    let b_val = Math.floor(Math.min(1.0, brightness) * 255);
                    
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, 14, 0, Math.PI * 2);
                    ctx.fillStyle = `rgb(${{b_val}}, ${{b_val}}, ${{b_val}})`;
                    ctx.fill();
                    
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#E0E0E0';
                    ctx.stroke();
                }});
                
                requestAnimationFrame(draw);
            }}
            draw();
        </script>
    </body>
    </html>
    """
    return html
