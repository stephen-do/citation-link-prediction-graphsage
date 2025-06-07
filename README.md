# Citation Link Prediction with GraphSAGE

This project implements a **graph-based link prediction model** using **GraphSAGE** on a citation dataset. The goal is to predict whether a citation link exists between two academic papers based on their metadata and embeddings.

## 🚀 Project Overview

- **Dataset**: Processed from `DBLPOnlyCitationOct19.txt` (original DBLP citation dataset).
- **Graph Construction**: Each node represents a paper, and edges represent citation links.
- **Text Embedding**: Uses [GTE](https://huggingface.co/thenlper/gte-small) (general-purpose text embedding) or BERT to encode paper titles and abstracts.
- **Model**: [GraphSAGE](https://arxiv.org/abs/1706.02216) for learning node embeddings.
- **Link Predictor**: MLP-based binary classifier trained on positive and negative edge samples.
- **Evaluation Metrics**: ROC-AUC, Precision, Recall, and Accuracy.

## 📁 Project Structure

```
citation-link-prediction-graphsage/
│
├── dataset/                          # Raw and processed datasets
│   ├── DBLPOnlyCitationOct19.txt
│   ├── citation-cooked.csv
│   └── citation_all.dgl              # Graph with features
│
├── model.py                          # GraphSAGE model
├── predictor.py                      # Dot and MLP edge predictors
├── utils.py                          # Training utilities (loss, metrics)
│
├── data_parsing.py                   # Process raw DBLP into CSV
├── data_cooking.py                   # Build graph, compute text embeddings
├── train.py                          # Train and evaluate model
│
├── requirements.txt
└── README.md
```

## 🧠 How It Works

1. **Preprocessing** (`data_parsing.py`): Parses raw DBLP dataset into structured CSV.
2. **Graph Construction** (`data_cooking.py`):
   - Builds citation graph using DGL.
   - Generates BERT/GTE embeddings from paper titles and abstracts.
   - Saves a graph object with node features.
3. **Training** (`train.py`):
   - Splits edges into train/test, generates negative samples.
   - Trains GraphSAGE + link predictor.
   - Evaluates with standard metrics.

## 🛠 Installation

Make sure you have Python 3.8+ and PyTorch 2.6 installed. Then run:

```bash
pip install -r requirements.txt
```

> **Important**: If you're using GPU with CUDA 12.4, DGL must be installed with:
>
> ```bash
> pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html
> ```

## 🧪 Running the Project

1. **Preprocess dataset:**

```bash
python data_parsing.py
```

2. **Build the citation graph with GTE embeddings:**

```bash
python data_cooking.py
```

3. **Train the GraphSAGE model:**

```bash
python train.py
```

4. **View training logs in TensorBoard:**

```bash
tensorboard --logdir logs/
```

## 📊 Results

During training, the following metrics are reported on the test set:
- Loss
- ROC AUC
- Precision
- Recall
- Accuracy

You can monitor performance in TensorBoard.

## 📌 Requirements

- Python 3.8+
- PyTorch 2+
  - DGL (matching CUDA and Torch version)
- Transformers
- scikit-learn
- pandas, tqdm, tensorboard

All dependencies are listed in `requirements.txt`.

## 📖 Citation

If you use this code or ideas in your research, please consider citing or giving a star ⭐ on GitHub!

## 📬 Contact

Created by [Stephen Do](https://github.com/stephen-do) – contributions and suggestions welcome!
