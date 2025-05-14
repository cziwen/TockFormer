# 📈 TockFormer

#### 🚧 Note: This project is a work in progress and is not yet production-ready.

---

## 🧠 Overview

##### TockFormer is an integrated pipeline for quantitative financial analysis. Its ultimate goal is to:
- ✅ Ingest real-time tick data  
- ✅ Aggregate and transform raw data into structured CSV files  
- 🛠️ Enable on-demand feature engineering  
- 🤖 Feed processed data into machine learning models  
- 💡 Generate trade signals based on model output

---

## ✅ Current Status

##### The following functionality is currently implemented:
- 📡 Real-time data streaming  
- 🧾 CSV aggregation for downstream analysis  

Users can analyze or process the resulting CSVs as needed.

---

## ⚡ Quick Start

> **Note:** The environment setup is not finalized. Commonly used libraries include:
> - `numpy`  
> - `pandas`  
> - `torch` *(install either the CPU or GPU version depending on your system)*

#### 🔧 Steps to Run:

1. Open `aggrData.py` and configure your **Finnhub API token**.  
   - A free token is available from [https://finnhub.io](https://finnhub.io)
2. Set the desired output paths for **log files** and **CSV storage**.
3. Run the script:

```bash
python aggrData.py
```

## 🖥️ Legacy Scripts

- `main_train.py` and `main_evaluate.py` were initially designed for Linux environments.  
- These scripts are **not required for normal usage** and are currently **not actively maintained**.

---

## 🚀 Coming Soon

##### The following components are under development:
- Feature engineering modules  
- Model training and inference  
- Real-time signal generation  

Stay tuned for updates!