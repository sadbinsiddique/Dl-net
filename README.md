<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X-GCN: Facial Micro-Expression Recognition</title>
    
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true, theme: 'dark'});</script>

    <style>
        /* =========================================================
           LIQUID GLASS (GLASSMORPHISM) CSS
           ========================================================= */
        
        :root {
            /* Glass Properties */
            --glass-bg: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.15);
            --glass-highlight: rgba(255, 255, 255, 0.25);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --glass-blur: blur(16px);
            
            /* Text Properties */
            --text-primary: #ffffff;
            --text-secondary: #e2e8f0;
            --accent-color: #00f2fe;
        }

        body {
            margin: 0;
            padding: 40px 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            color: var(--text-primary);
            
            /* Animated Gradient Background for Liquid Effect */
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1f1c2c);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            min-height: 100vh;
            display: flex;
            justify-content: center;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* The Main Glass Container */
        .glass-panel {
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid var(--glass-border);
            box-shadow: var(--glass-shadow);
            border-radius: 24px;
            padding: 50px;
            max-width: 900px;
            width: 100%;
            line-height: 1.6;
        }

        /* Typography & Alignment */
        .text-center { text-align: center; }
        
        h1, h2, h3 {
            color: var(--text-primary);
            margin-top: 40px;
            margin-bottom: 20px;
        }

        h1 {
            font-size: 2.5em;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 15px;
        }
        
        h2 { font-size: 1.8em; color: var(--accent-color); }
        h3 { font-size: 1.4em; }

        .subtitle {
            font-size: 1.1em;
            font-style: italic;
            color: var(--text-secondary);
            margin-bottom: 30px;
        }

        /* Shields / Badges */
        .badges img {
            margin: 5px;
            border-radius: 4px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .badges img:hover { transform: translateY(-3px); }

        /* Tables (Glass Style) */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid var(--glass-border);
        }

        th {
            background: rgba(255, 255, 255, 0.1);
            font-weight: 600;
            color: var(--accent-color);
        }

        tr:last-child td { border-bottom: none; }
        tr:hover { background: rgba(255, 255, 255, 0.05); }

        /* Blockquotes */
        blockquote {
            margin: 30px 0;
            padding: 20px;
            background: rgba(0, 242, 254, 0.1);
            border-left: 4px solid var(--accent-color);
            border-radius: 0 12px 12px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        blockquote p { margin: 0; }
        blockquote strong { color: var(--accent-color); }

        /* Code Blocks */
        pre {
            background: rgba(0, 0, 0, 0.4);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--glass-border);
            overflow-x: auto;
            color: #a3b8c2;
            font-family: 'Courier New', Courier, monospace;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
        }

        code {
            font-family: 'Courier New', Courier, monospace;
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 4px;
        }

        hr {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--glass-highlight), transparent);
            margin: 40px 0;
        }

        ul { list-style: none; padding-left: 0; }
        ul li::before {
            content: "✦";
            color: var(--accent-color);
            margin-right: 10px;
        }
        ul li { margin-bottom: 8px; }

        /* Highlight tags */
        strong { color: #fff; text-shadow: 0 0 5px rgba(255,255,255,0.3); }
        
        .proposed-highlight {
            background: rgba(0, 242, 254, 0.2);
            padding: 30px;
            border-radius: 16px;
            border: 1px solid var(--accent-color);
            margin-top: 40px;
        }
    </style>
</head>
<body>

<div class="glass-panel">
    
    <div class="text-center">
        <h1>🚀 X-GCN</h1>
        <div class="subtitle">A Micro-Motion Guided Lightweight Graph Neural Network for Efficient Facial Micro-Expression Recognition</div>
        
        <div class="badges">
            <img src="https://img.shields.io/badge/Status-Research-blue?style=for-the-badge" alt="Status">
            <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge" alt="Language">
            <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge" alt="Framework">
            <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?style=for-the-badge" alt="Platform">
            <img src="https://img.shields.io/badge/License-Academic-green?style=for-the-badge" alt="License">
        </div>
    </div>

    <hr>

    <h2>📖 Project Overview</h2>
    <p><strong>X-GCN</strong> is a lightweight <strong>Graph Neural Network (GNN)</strong> designed for <strong>Facial Micro-Expression Recognition (MER)</strong>. The proposed framework utilizes <strong>micro-motion-guided graph representation learning</strong> to efficiently capture subtle facial movements while maintaining low computational complexity.</p>

    <hr>

    <h2>🧠 Algorithms</h2>
    
    <h3>Convolutional Neural Networks (CNN)</h3>
    <table>
        <thead>
            <tr><th>Model</th><th class="text-center">Release</th></tr>
        </thead>
        <tbody>
            <tr><td>AlexNet</td><td class="text-center"><strong>2012</strong></td></tr>
            <tr><td>ResNet</td><td class="text-center"><strong>2015</strong></td></tr>
            <tr><td>MobileNet V7 <em>(Custom)</em></td><td class="text-center"><strong>2026</strong></td></tr>
            <tr><td>EfficientNet</td><td class="text-center"><strong>2019</strong></td></tr>
            <tr><td>Vision Transformer (ViT)</td><td class="text-center"><strong>2020</strong></td></tr>
            <tr><td>ConvNeXt</td><td class="text-center"><strong>2022</strong></td></tr>
        </tbody>
    </table>

    <h3>Graph Neural Networks (GNN)</h3>
    <table>
        <thead>
            <tr><th>Model</th><th class="text-center">Release</th></tr>
        </thead>
        <tbody>
            <tr><td>Graph Convolutional Network (GCN)</td><td class="text-center"><strong>2016</strong></td></tr>
            <tr><td>Graph Attention Network (GAT)</td><td class="text-center"><strong>2017</strong></td></tr>
            <tr><td>Graph Isomorphism Network (GIN)</td><td class="text-center"><strong>2018</strong></td></tr>
            <tr><td>GraphEx</td><td class="text-center"><strong>2022</strong></td></tr>
            <tr><td>Spatial & Spectral GNN (SSGNN)</td><td class="text-center"><strong>2022</strong></td></tr>
            <tr><td>Graph Attention-based MER</td><td class="text-center"><strong>2024</strong></td></tr>
            <tr><td>Stochastic GCN (SGCN)</td><td class="text-center"><strong>2024</strong></td></tr>
            <tr><td>OFVIG-Net</td><td class="text-center"><strong>2024</strong></td></tr>
            <tr><td>SpoT-GCN</td><td class="text-center"><strong>2024</strong></td></tr>
            <tr><td>FM-GCN</td><td class="text-center"><strong>2026</strong></td></tr>
            <tr style="background: rgba(0, 242, 254, 0.15);"><td><strong>X-GCN (Proposed)</strong></td><td class="text-center"><strong>2027</strong></td></tr>
        </tbody>
    </table>

    <hr>

    <h2>📂 Datasets</h2>
    <table>
        <thead>
            <tr><th>Dataset</th><th>Purpose</th></tr>
        </thead>
        <tbody>
            <tr><td>CASME II</td><td>Micro-Expression Recognition</td></tr>
            <tr><td>CK+</td><td>Facial Expression Recognition</td></tr>
            <tr><td>SAMM</td><td>Spontaneous Micro-Expression Recognition</td></tr>
        </tbody>
    </table>

    <blockquote>
        <p><strong>Note:</strong> All datasets were collected using <strong>Kaggle Hub</strong> and manually verified before conducting experiments to ensure <strong>data authenticity</strong>, <strong>consistency</strong>, and <strong>integrity</strong>.</p>
    </blockquote>

    <hr>

    <h2>💻 Development Environment</h2>
    
    <h3>IDE</h3>
    <ul><li>Visual Studio Code</li></ul>
    
    <h3>Programming Language</h3>
    <ul><li>Python 3.12+</li></ul>
    
    <h3>Deep Learning Framework</h3>
    <ul>
        <li>PyTorch</li>
        <li>TorchVision</li>
    </ul>
    
    <h3>Environment</h3>
    <ul>
        <li>Conda</li>
        <li>Google Colab</li>
    </ul>
    
    <h3>Database</h3>
    <ul><li>MySQL</li></ul>
    
    <h3>Storage</h3>
    <ul><li>Google Drive</li></ul>

    <hr>

    <h2>🧩 Visual Studio Code Extensions</h2>
    <table>
        <thead>
            <tr><th>Extension</th></tr>
        </thead>
        <tbody>
            <tr><td>Python Extension Pack</td></tr>
            <tr><td>Code Runner</td></tr>
            <tr><td>Jupyter Cell Tags</td></tr>
            <tr><td>Jupyter Keymap</td></tr>
            <tr><td>Jupyter Notebook Renderers</td></tr>
            <tr><td>Jupyter Slide Show</td></tr>
            <tr><td>Python Snippets</td></tr>
            <tr><td>Prettier</td></tr>
            <tr><td>SQLTools</td></tr>
            <tr><td>FastAPI Snippets</td></tr>
            <tr><td>Bruno</td></tr>
            <tr><td>Colab</td></tr>
            <tr><td>Google Colab Keymap</td></tr>
        </tbody>
    </table>

    <blockquote>
        <p><strong>Important:</strong> Install all required software packages, libraries, and dependencies before conducting the experiments.</p>
    </blockquote>

    <hr>

    <h2>⚙️ Experimental Workflow</h2>
<pre>
Dataset Collection
        │
        ▼
 Dataset Verification
        │
        ▼
 Dataset Balancing
 (Middle Sampling)
        │
        ▼
 Data Preprocessing
        │
        ▼
 Store Dataset
 (Google Drive)
        │
        ▼
 Model Training
 (Google Colab)
        │
        ▼
 Hyperparameter Tuning
        │
        ▼
 Save Model Weights
 (MySQL)
        │
        ▼
 Performance Evaluation
        │
        ▼
 Comparison with Existing Models
</pre>

    <hr>

    <h2>📝 Experimental Procedure</h2>
    <table>
        <thead>
            <tr><th>Step</th><th>Description</th></tr>
        </thead>
        <tbody>
            <tr><td>01</td><td>Collect datasets</td></tr>
            <tr><td>02</td><td>Verify dataset authenticity</td></tr>
            <tr><td>03</td><td>Balance the dataset using <strong>Middle Sampling</strong></td></tr>
            <tr><td>04</td><td>Perform preprocessing</td></tr>
            <tr><td>05</td><td>Store processed datasets in <strong>Google Drive</strong></td></tr>
            <tr><td>06</td><td>Train models using <strong>Google Colab</strong></td></tr>
            <tr><td>07</td><td>Tune hyperparameters</td></tr>
            <tr><td>08</td><td>Save trained model weights and metadata</td></tr>
            <tr><td>09</td><td>Evaluate model performance</td></tr>
            <tr><td>10</td><td>Compare with state-of-the-art methods</td></tr>
        </tbody>
    </table>

    <hr>

    <h2>📊 Research Pipeline</h2>
    <div class="mermaid">
    flowchart LR
        A[Dataset]
        B[Preprocessing]
        C[Middle Sampling]
        D[CNN Models]
        E[GNN Models]
        F[X-GCN]
        G[Evaluation]

        A --> B
        B --> C
        C --> D
        C --> E
        D --> G
        E --> G
        G --> F
    </div>

    <hr>

    <h2>📌 Project Structure</h2>
<pre>
X-GCN
│
├── Dataset
│   ├── CASME II
│   ├── CK+
│   └── SAMM
│
├── Notebook
│
├── Models
│   ├── CNN
│   └── GNN
│
├── Preprocessing
│
├── Hyperparameters
│
├── Results
│
├── Figures
│
├── Weights
│
└── README.md
</pre>

    <hr>

    <h2>📢 Important Notes</h2>
    <blockquote>
        <p><strong>Experiment Workflow:</strong> The experimental workflow may be modified whenever necessary to satisfy the research objectives.</p>
    </blockquote>

    <div class="text-center proposed-highlight">
        <h2 style="margin-top: 0;">⭐ Proposed Model</h2>
        <h1 style="border: none; margin: 10px 0;"><strong>X-GCN</strong></h1>
        <p class="subtitle">A Micro-Motion Guided Lightweight Graph Neural Network for Efficient Facial Micro-Expression Recognition</p>
        <p style="font-size: 1.2em;"><strong>Expected Release:</strong> <span style="color: var(--accent-color);">2027</span></p>
    </div>

</div>

</body>
</html>
