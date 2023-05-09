## _Disease Prediction via GCN (Graph Neural Networks)_



## Overview

This project uses a Graph Convolutional Network (GCN) to predict the likelihood of a patient having a particular disease. The model is trained on a graph representation of patient data, where nodes represent patients and edges represent similarities between patients. The project is implemented in Python and requires PyTorch, scikit-learn, Selenium, Python version 3.7 or higher, and NetworkX to run.

Project summary ipynb (For Extra Credit) - **report_summary.ipynb**

outputs and evaluation of models - **outputs.ipynb**



## Installation
## Clone the repository
```sh
git clone https://github.com/aviral38/Dl4healtcare.git
```
## Install required packages
```sh
pip install -r requirements.txt
```

## Data
To prepare the data for the disease prediction model, you can find the required dataset inside the data folder. If you want to try the model with a different dataset or change the path of the data, you will need to modify the path of the dataset in the outputs.ipynb file accordingly.

To do this, you should locate the code in the outputs.ipynb file that loads the dataset and modify the path argument to point to the new dataset. Specifically, the code for loading the dataset is located in the following line:

```sh
list_of_nodes, attributes, labels, adjacency_list, train_set, test_set = load_dataset("./data/graph_data/191210/graph-P-191210-00")

```

The dataset for disease prediction consists of several files, each containing different information about the nodes and their relationships in the graph. These files have the following formats:

**filename.nodes.pkl**: A list of nodes in the graph, represented as strings.

**filename.adj.pkl**: An adjacency list of the nodes in the graph, where each node is mapped to a list of its neighboring nodes.

**filename.rare.label.pkl**: A flag indicating whether a node is a rare disease (value=1) or contains a rare disease. This file is a NumPy array of shape (N x 1), where N is the number of nodes in the graph.

**filename.label.pkl**: A NumPy array of shape (N x D), where N is the number of nodes in the graph and D is the number of diseases. This file contains information about which diseases are associated with each node.

**filename.map.pkl**: A mapping between the node names and their corresponding index in the graph.

**filename.train.pkl**: A list of nodes used for training the disease prediction model.

**filename.test.pkl**: A list of nodes used for testing the disease prediction model.

## Usage
Open the **outputs.ipynb** Jupyter notebook file in a Jupyter Notebook or JupyterLab environment.

Once you have installed the dependencies, you can open the outputs.ipynb file in Jupyter Notebook or JupyterLab. The notebook contains several cells that run various baseline models such as SVM, Random Forest, Decision Trees and multi-layer perceptron for disease prediction, as well as a cell that runs a graph convolutional network (GCN)-based disease predictor model.

To generate outputs for the disease prediction task, you will need to run the necessary cells in the notebook.

After running the necessary cells in the outputs.ipynb notebook, you will generate precision scores, f1 scores, and recall values for all of the baseline models (SVM, decision tree, random forest, and MLP) as well as the GCN-based disease predictor model. These scores are generated for different values of k (k=1,2,3,4, and 5)

The precision score measures the proportion of true positive predictions (i.e., correctly predicted disease outcomes) among all positive predictions made by the model. The f1 score is a weighted average of the precision and recall values, which provides a single metric that balances both metrics. The recall value measures the proportion of true positive predictions among all actual positive cases in the test data.

By comparing the precision, f1, and recall scores for each model at different values of k, you can determine which model and which value of k provides the best performance for your specific disease prediction task. This information can be used to optimize your model for accuracy and efficiency, and to make more informed decisions about patient care and treatment.


## Acknowledgments
* The code for the GCN model was adapted from [Z. Sun, H. Yin, H. Chen, T. Chen, L. Cui and F. Yang] of the "Disease Prediction via Graph Neural Networks".
* The data used in this project is Human Phenotype Ontology (HPO) dataset. The Human Phenotype Ontology (HPO) is widely used in clinical genomics research, rare disease diagnosis, drug discovery, and precision medicine. The dataset is available for download from
the HPO website or can be accessed via APIs






