# Drought Forecasting over Maharashtra with Climate Indices

## 1. Documentation

### 1.1 Repository Structure

```
drought-forecasting-climate-indices/
├── README.md            # Setup and usage instructions
├── requirements.txt     # Python packages
├── code.py              # Training and evaluation code for both models
├── drought_core.py      # Data loading and forecasting functions used by the web app
├── web.py               # Flask server for the demo web app
├── static/              # Web page files (index.html, app.js, style.css)
├── data/                # Place dataset1 and dataset2 here
└── models/              # Place the trained model files here
```

### 1.2 Installation

```bash
git clone https://github.com/Erick-007/drought-forecasting-climate-indices.git
cd drought-forecasting-climate-indices
pip install -r requirements.txt
```

Requires Python 3.10+, with PyTorch, NumPy, Pandas, Matplotlib and Flask.

### 1.3 Running the Code

Training and evaluation:

```bash
python code.py
```

The script trains and tests both models for the 1-month and 3-month horizons and saves the metrics and plots.

Demo web app:

```bash
python web.py
```

Then open http://localhost:5000 in a browser.

### 1.4 Publication

A paper based on this work has been submitted to ICICC-2027 (Springer LNNS).
