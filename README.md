# ID-GNN: Identity-Aware Graph Neural Networks

This repository contains the code, experiments, and results for my Master's thesis: **ID-GNN: Enhancing Structural Awareness in Graph Neural Networks via Identity-Aware Message Passing**.

> **Author:** Sara Mount  
> **Degree:** M.S. in Computer Science, Boise State University  
> **Focus:** Graph Neural Networks, Anomaly Detection, Structural Representation Learning

---

## 🧭 Project Overview

Graph Neural Networks (GNNs) have revolutionized learning on graph-structured data but often fall short in capturing fine-grained structural identity. This project introduces **ID-GNN**, a new GNN variant that improves structural awareness by encoding node identity directly into the message-passing process.

### 🎯 Objectives
- Improve expressive power of GNNs for structural tasks
- Benchmark against standard GNN architectures
- Demonstrate real-world applicability in anomaly detection and classification

---

## 🧪 Key Features

- 🔁 **ID-aware Message Passing**: Encodes ego-node identity at each step
- ⚡ **ID-GNN-Fast**: Lightweight variant using hand-engineered identity features (cycle counts, centrality, etc.)
- 🧪 **Benchmarking**: Extensive evaluation on node classification, link prediction, and graph property prediction tasks
- 📈 **Performance**: Shows up to **40%** relative improvement in structural tasks compared to traditional GNNs

---

## 🧰 Tech Stack

- **Python 3.10**
- **PyTorch**
- **PyTorch Geometric**
- **GraphGym** (Stanford)
- **NetworkX**, **NumPy**, **Scikit-learn**
- Jupyter Notebooks for visualization and exploratory testing

---

## 🧬 Directory Structure

