# Neural Network Tutorials

This repository contains a set of Jupyter/Colab notebooks for practicing core neural-network workflows in PyTorch, comparing them with classic machine-learning baselines, and exploring fairness-related datasets. The notebooks move from simple synthetic examples to image classification and tabular fairness experiments.

## Notebook Overview

### `NeuralNetworkBasics.ipynb`

Introductory PyTorch experiments on synthetic data from scikit-learn.

Coverage:

- Built a linear regression model with `nn.Linear` and trained it with mean squared error.
- Built a logistic regression classifier with sigmoid activation and binary cross-entropy loss.
- Implemented a small feed-forward neural network classifier and a deeper neural network.
- Compared neural-network results against `DecisionTreeClassifier` and `RandomForestClassifier`.
- Practiced the basic PyTorch loop: tensor conversion, forward pass, loss calculation, backpropagation, optimizer step, and evaluation.

Saved outputs show strong performance on the synthetic tasks, including low linear-regression MSE and classification accuracies around the high-80% to low-90% range depending on the model.

### `CelebA.ipynb`

Computer-vision tutorial using the CelebA face-attributes dataset from `torchvision`.

Coverage:

- Downloaded and loaded CelebA with `torchvision.datasets.CelebA`.
- Resized images to `128x128`, converted them to tensors, and normalized them.
- Inspected the 40 CelebA attribute labels.
- Built a convolutional neural network with convolution, batch normalization, ReLU, max-pooling, and fully connected layers.
- Trained the model as a multi-label classifier using `BCEWithLogitsLoss`.
- Evaluated accuracy per attribute and average accuracy across all 40 attributes.

The saved run reports an average attribute accuracy of about 89.5% after a short training run.

### `FairnessModel.ipynb`

Tabular classification experiment based on the Adult Income dataset.

Coverage:

- Loaded the Adult dataset in Colab from an uploaded `adult.data` file.
- Assigned readable column names for demographic, work, education, and income fields.
- Cleaned missing rows and one-hot encoded categorical variables.
- Converted the processed tabular data into PyTorch tensors and `DataLoader`s.
- Built a fully connected neural network to predict income class.
- Trained and evaluated the model while tracking training and test losses.
- Saved model and optimizer checkpoints under a `results/` directory during training.

This notebook practices neural networks on demographic tabular data and sets up the groundwork for thinking about protected attributes such as sex and race in prediction tasks.

### `FairJob.ipynb`

Fair ranking/click-prediction experiment using Criteo's FairJob dataset from Hugging Face.

Coverage:

- Loaded `criteo/FairJob` from Hugging Face with pandas.
- Explored columns such as `click`, `protected_attribute`, `senior`, `displayrandom`, `rank`, categorical features, and numeric features.
- Split the dataset into train and test sets.
- Converted the tabular data into PyTorch tensors and `DataLoader`s.
- Built a feed-forward neural network to predict whether a job impression was clicked.
- Trained and evaluated the click model with negative log-likelihood loss.

This notebook focuses on applying PyTorch to a fairness-oriented ranking dataset and experimenting with the mechanics of learning from protected-attribute and ranking features.

## Tools And Libraries

The notebooks primarily use:

- Python
- PyTorch and Torchvision
- scikit-learn
- pandas
- NumPy
- tqdm
- Jupyter Notebook / Google Colab

## How To Run

Open the notebooks in Jupyter or Google Colab and run the cells from top to bottom. Some notebooks require downloading or uploading data:

- `CelebA.ipynb` downloads CelebA through Torchvision, which is a large dataset.
- `FairnessModel.ipynb` expects the Adult Income `adult.data` file to be uploaded in Colab.
- `FairJob.ipynb` reads the FairJob dataset from Hugging Face.

These notebooks are tutorial experiments rather than a packaged Python project, so there is no shared application entry point.
