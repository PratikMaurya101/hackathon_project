# 🌡️ ClimateTwin AI

An intelligent climate control digital twin that combines real-time sensor simulation, machine learning predictions, and local LLM-powered explanations to improve building climate management.

## 🚀 Overview

ClimateTwin AI is a smart building monitoring and decision-support system designed to:

- Simulate environmental sensor data
- Predict future climate conditions using Machine Learning
- Visualize system behavior through an interactive dashboard
- Explain predictions and recommendations in natural language using a local Large Language Model (LLM)

The project demonstrates how AI can enhance transparency and decision-making in climate control systems.

---

## ✨ Features

### 📊 Interactive Dashboard
Built with Streamlit to provide:

- Temperature monitoring
- Humidity monitoring
- Weather information
- Power consumption tracking
- Historical trend visualization

### 🤖 Predictive Analytics
Machine learning models analyze sensor data to:

- Forecast future temperature
- Detect abnormal conditions
- Estimate energy consumption
- Support climate control decisions

### 🧠 AI Explanations
Powered by Qwen 3.5 running locally through Ollama.

The LLM converts model outputs into human-readable insights such as:

> "Temperature is expected to increase over the next 30 minutes. Increasing ventilation is recommended to maintain occupant comfort while minimizing energy usage."

### 🏠 Digital Twin
A virtual representation of an indoor environment that continuously updates using simulated sensor data.

---

## 🏗️ System Architecture

```text
Sensor Simulator
        │
        ▼
Machine Learning Model
        │
        ├────────────► Streamlit Dashboard
        │
        ▼
    Qwen 3.5 (Ollama)
        │
        ▼
Natural Language Insights
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Large Language Model
- Ollama
- Qwen 3.5

### Visualization
- Streamlit Charts

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/PratikMaurya101/hackathon_project.git
cd hackathon_project
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Install Qwen 3.5

Install Ollama:

https://ollama.com

Pull the model:

```bash
ollama pull qwen3.5:9b
```

Verify:

```bash
ollama run qwen3.5:9b
```

---

## ▶️ Run Application

Start the Streamlit dashboard:

```bash
streamlit run lhome-page.py
```

---

## 📁 Project Structure

```text
hackathon_project/
│
├── predictive_maintenance/
│   ├── predict_model_updated.py
│
├── data/
│
├── lhome-page.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

## 🎯 Use Cases

- Smart Buildings
- HVAC Optimization
- Energy Monitoring
- Facility Management
- Predictive Maintenance
- Climate Analytics

---

## 🔮 Future Improvements

- Real sensor integration (IoT devices)
- Energy optimization recommendations
- Multi-room digital twins
- CO₂ concentration monitoring
- Reinforcement learning-based control
- Building Management System (BMS) integration

---

## 👥 Team

Developed during a Hackathon by:

- Pratik Maurya
- Team Members

---

## 📜 License

This project is developed for educational and hackathon purposes.
