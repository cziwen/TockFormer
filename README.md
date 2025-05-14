# 📈 TockFormer

#### 🚧 Note: This project is a work in progress and is not yet production-ready.

---

## 🧠 Overview

**TockFormer** is an integrated pipeline for quantitative financial analysis. Its ultimate goal is to:

- ✅ Ingest real-time tick data  
- ✅ Aggregate and transform raw data into structured CSV files  
- 🛠️ Enable on-demand feature engineering  
- 🤖 Feed processed data into machine learning models  
- 💡 Generate trade signals based on model output

---

## ✅ Current Status

The following functionality is currently implemented:

- 📡 Real-time data streaming  
- 🧾 CSV aggregation for downstream analysis  

Users can analyze or process the resulting CSVs as needed.

---

## ⚡ Quick Start

> **Note:** The environment setup is still evolving. Commonly used libraries include:
> - `numpy`  
> - `pandas`  
> - `torch` *(install either the CPU or GPU version depending on your system)*

### 🔧 Steps to Run:

1. Open `Utility/AggrData.py` and configure your **Finnhub API token**.  
   - A free token is available at [https://finnhub.io](https://finnhub.io)
2. Set your preferred **log file** and **CSV output paths** inside the script or via environment variables.  
3. Start the pipeline with:

```bash
python launcher.py \
      --script Utility/AggrData.py \
      --start "2025-05-14 16:39:00" \
      --end   "2025-05-15 16:00:00"
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