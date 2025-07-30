# 🚀 Quantum Drift: Passenger Fate Prediction

This project predicts whether passengers aboard the Spaceship Titanic were transported to an alternate dimension during a spacetime anomaly. It uses machine learning classification techniques to support GalactiCorp’s rescue operations.

---

## 📁 Folder Structure

```
QuantumDrift-SpaceshipTitanic/
│
├── data/                       # Raw input data from Kaggle
│   ├── train.csv
│   └── test.csv
│
├── output/                     # Cleaned data, engineered features, plots
│   ├── cleaned_train.csv
│   ├── cleaned_test.csv
│   ├── engineered_train.csv
│   ├── engineered_test.csv
│   ├── plot_age_histogram.png
│   ├── plot_totalspend_boxplot.png
│   ├── plot_correlation_matrix.png
│   ├── plot_cryo_count.png
│   ├── plot_vip_count.png
│   └── plot_vip_vs_transported.png
│
├── notebooks/                  # Jupyter notebooks
│   └── QuantumDrift_VisualEDA.ipynb
│
├── scripts/                    # Data cleaning and feature engineering scripts
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── spaceship_titanic_visual_eda.py
│
├── requirements.txt            # Required Python packages
└── README.md                   # This file
```

---

## 🧪 How to Run the Notebook

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
cd notebooks
jupyter notebook QuantumDrift_VisualEDA.ipynb
```

### 3. Outputs
Plots and processed datasets will be saved in the `output/` directory.

---

## 📊 Visualizations
The following plots are generated:
- Age Histogram
- TotalSpend Boxplot
- Correlation Matrix
- CryoSleep Distribution
- VIP Status Distribution
- Transported vs VIP Stacked Bar Chart

---

## 👨‍💻 Team
- Herve Taning Kaffo
- Logan Huba
- Tobias Jones

## 📅 Course
**ACS 5513 – Machine Learning Practice**  
Summer 2025 – Gallogly College of Engineering, The University of Oklahoma