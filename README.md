# 📈 TockFormer

#### 🚧 Note: This project is a work in progress and is not yet production-ready.

---
## 📌 Log Board (as of 2025-05-16)

<details>
<summary>🔌 Real-time Tick Ingestion (Updated: 2025-05-16)</summary>

- ✅ Implemented workable tick-level ingestion module. *(2025-05-05)*  
- ✅ Built a standalone `launcher.py` script for scheduling pipeline execution over custom time intervals. *(2025-05-14)*  
- ✅ Enhanced IO throughput using **mpi4py** to parallelize and streamline tick data ingestion. *(2025-05-16)*  

</details>

<details>
<summary>🧠 Feature Engineering — FactorFactory Module (Updated: 2025-05-15)</summary>

- ✅ Implemented a **tree-based structure** for automatic factor expansion. *(2025-05-12)*  
- ✅ Added combined IC+IR score metric (**|Spearman IC| + |Pearson IR|**) to guide factor selection. *(2025-05-13)*  
- ✅ Enabled bounded cross-operations (e.g., +, −, ×, ÷, sin, cos) with constraints on output stability. *(2025-05-14)*  
- ✅ Optimized search by pruning correlated factors with a **greedy minimum-correlation selection**. *(2025-05-15)*  
- 🔄 *(Proposed)* Introduce A* search algorithm for intelligent feature exploration based on IC distance. *(Planned)*  

</details>

<details>
<summary>🤖 Model Inference (Not yet updated)</summary>

- ⏳ Placeholder for future logs related to model training, validation, and prediction integration. *(To be started after 2025-05-25)*  
- ❌ No updates yet. This module is currently under development.  

</details>

<details>
<summary>💡 Signal Generation (Not yet updated)</summary>

- ⏳ Placeholder for future logs related to trading signal generation, output formatting, and strategy logic. *(To be started after 2025-06-05)*  
- ❌ No updates yet. This module is currently under development.  

</details>

---

## 🧠 Overview

**TockFormer** is an end-to-end pipeline for real-time quantitative financial signal generation. The system follows this modular workflow:

1. 🔌 **Real-time Tick Ingestion**  
   - Subscribe to live trade data via WebSocket  
   - Aggregate and write to a CSV(or any db you prefer)

2. 🧠 **Feature Engineering (FactorFactory)**  
   - Automatic factor generation and search

3. 🤖 **Model Inference**  
   - Feed curated features into machine learning models for prediction

4. 💡 **Signal Generation**  
   - Output trading signals based on model results (to be finalized)

---
## ✅ Current Status

Project progress is tracked by the four core modules, and part of them are good to use:
###### guides for usage will be provided in the future
<details>
<summary>🔌 Real-time Tick Ingestion — ✅</summary>

- WebSocket subscription and message ingestion implemented  
- Aggregation and parallel I/O via `mpi4py` operational

</details>

<details>
<summary>🧠 Feature Engineering — FactorFactory — ✅</summary>

- Tree-based generation and heuristic search functioning as intended

</details>

<details>
<summary>🤖 Model Inference — ❌</summary>

- Model loading and prediction pipeline not yet implemented

</details>

<details>
<summary>💡 Signal Generation — ❌</summary>
- Signal decoding and trade logic pending

</details>
---

## ⚡ Quick Start

> **Note:** The environment setup is still evolving. Commonly used libraries include:
> - `numpy`  
> - `pandas`  
> - `torch` *(install either the CPU or GPU version depending on your system)*
> - and many other packages...

### 🔧 Steps to Run:
1. Open `Utility/AggrData.py` and configure your **Finnhub API token**.  
   - A free token is available at [https://finnhub.io](https://finnhub.io)
2. Set your preferred **log file** and **CSV output paths** inside the script or via environment variables. 
   - run `benchmark_websocket.ipynb` to see what is your device's IO speed. normally a popular symbol has 3000 ticks/sec, so this file tells you how many symbol you can subscribe at one time.
3. Start the pipeline with:

```bash
python launcher.py \
      --script Utility/AggrData.py \
      --start "2025-05-04 03:00:00" \
      --end   "2025-05-04 21:00:00"
```
> ###### Why use --start and --end ? 
> Finnhub’s historical tick data endpoint only returns data up to the end of a trading day, and the final data for the day is typically only available after 8:00 PM US Eastern Time (during Daylight Saving Time).
> 
> ###### To ensure complete data retrieval, I recommend scheduling the launcher.py script to run at 3:00 AM Eastern Time the following day. This timing guarantees that all tick data for the previous trading day is fully accessible.

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