# GNN-proces-discovery

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the code developed for the scientific article **"A Machine Learning-Based Process Mining Discovery Approach"**.  
The repository provides the tools and scripts needed to reproduce the experiments and results presented in the article.

---

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction
This project presents a novel algorithm for process discovery that leverages graph neural networks to infer sound Petri nets from event logs. The code enables users to replicate the experimental results and adapt the method for further research.

---

## Installation
Clone the repository and install the requirements:
 ```bash
 git clone https://github.com/jaxels20/GNN-proces-discovery.git
 cd GNN-proces-discovery
 python -m venv env
 source env/bin/activate
 pip install -r requirements.txt
 ```
## Usage
1. Generate Synthetic Data
   ```python
      python3 data_generation/data_generation.py 
   ```
3. Train Model
   ```python
      python3 training.py 
   ```
5. Controlled Scenario Evaluation
   ```python
      python3 evaluate_on_controlled_scenarios.py
   ```
7. Real Life Evaluation
   ```python
      python3 evaluate_on_reallife_datasets.py 
   ```

## License
This repository is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
   
