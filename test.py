from tabpfn_extensions import unsupervised
from tabpfn_extensions.unsupervised import experiments
from sklearn.datasets import load_breast_cancer
import torch
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

# Load and prepare breast cancer dataset
df = load_breast_cancer(return_X_y=False)

print(df)

X, y = df["data"], df["target"]
feature_names = df["feature_names"]

# Initialize TabPFN models
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=TabPFNClassifier(),
    tabpfn_reg=TabPFNRegressor()
)

# Select features for synthetic data generation
# Example features: [mean texture, mean area, mean concavity]
feature_indices = [4, 6, 12]

# Run synthetic data generation experiment
experiment = unsupervised.experiments.GenerateSyntheticDataExperiment(
    task_type="unsupervised"
)

results = experiment.run(
    tabpfn=model_unsupervised,
    X=torch.tensor(X),
    y=torch.tensor(y),
    attribute_names=feature_names,
    temp=1.0,                     # Temperature parameter for sampling
    n_samples=X.shape[0] * 2,     # Generate twice as many samples as original data
    indices=feature_indices,
)

print(experiment.data_synthetic)