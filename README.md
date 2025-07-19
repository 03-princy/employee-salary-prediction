# 💰 Employee Salary Prediction System

**Predict • Analyze • Optimize Workforce Compensation**

&#x20;  &#x20;

---

## ✨ Features Overview

| Feature              | Description                            | Technology      |
| -------------------- | -------------------------------------- | --------------- |
| 🔍 Data Exploration  | Comprehensive EDA with visual insights | Pandas, Seaborn |
| 🤖 ML Modeling       | Multiple algorithm comparison          | Scikit-learn    |
| 🎯 Interactive UI    | User-friendly prediction interface     | Streamlit       |
| 📊 Real-time Results | Instant salary classification          | Pickle          |
| 📁 Data Management   | CSV upload/download functionality      | Pandas          |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation

```bash
git clone https://github.com/03-princy/employee-salary-prediction.git
cd employee-salary-prediction
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run app.py
```

---

## 🧩 Project Architecture

```mermaid
graph TD
    %% Exploration & Prototyping
    subgraph Exploration["🔍 Exploration Phase"]
        A1["EDA Notebook\nemployee_salary_prediction.ipynb"]
        A2["KNN Notebook\nknn_adult_csv_updated.ipynb"]
    end

    %% Training Workflow
    subgraph Training["⚙️ Training Phase"]
        B1["Raw Dataset\nadult.csv"]
        B2["Training Pipeline\ntrain_model.py"]
        B2 -->|serialize| B3["Preprocessor Artifact\ntarget_encoder.pkl"]
        B2 -->|serialize| B4["Model Artifact\nbest_model.pkl"]
    end

    %% Inference Workflow
    subgraph Inference["🚀 Inference Phase"]
        C1["Inference Service\nFlask API: app.py"]
        C2["Client\ncurl/Postman"]
        B3 -.->|load| C1
        B4 -.->|load| C1
        C1 -->|POST /predict| C2
    end

    %% Connections
    A1 --> B2
    A2 --> B2
    B1 --> B2

    style Exploration fill:#f0f8ff,stroke:#4682b4,stroke-width:2px
    style Training fill:#fff0f5,stroke:#ff69b4,stroke-width:2px
    style Inference fill:#f0fff0,stroke:#3cb371,stroke-width:2px
    linkStyle 0,1,2,3,4,5,6 stroke:#666,stroke-width:2px

```

---

## 🛠️ Tech Stack Deep Dive

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

---

## 📈 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.87  |
| Precision | 0.85  |
| Recall    | 0.82  |
| F1-Score  | 0.83  |

---

## 🖼️ Application Screenshots

---

## 📂 Project Structure

```
employee-salary-prediction/
├── app/                  # Streamlit application
│   ├── app.py            # Main application logic
│   └── utils.py          # Helper functions
├── models/               # Machine learning models
│   ├── train_model.py    # Training script
│   ├── best_model.pkl    # Serialized model
│   └── encoder.pkl       # Feature encoder
├── notebooks/            # Jupyter notebooks
│   └── exploration.ipynb # EDA notebook
├── data/                 # Dataset files
│   └── adult.csv         # Source dataset
└── requirements.txt      # Dependencies
```

---

## 🌟 Future Roadmap

* 🚀 Cloud Deployment (AWS/GCP)
* 📱 Mobile Application Port
* 🔄 Automated Retraining Pipeline
* 📝 PDF Report Generation
* 🔍 Advanced Feature Importance Analysis

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👩‍💻 Author

<div align="center">



**Priyanka Singh**  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/priyanka-singh-aa270123a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/03-princy)

</div>

