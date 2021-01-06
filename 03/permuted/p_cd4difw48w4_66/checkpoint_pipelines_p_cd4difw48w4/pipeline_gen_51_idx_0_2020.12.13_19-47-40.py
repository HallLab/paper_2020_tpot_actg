import numpy as np
import pandas as pd
from custom_transform import AddIntercept
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from tpot.builtins import FeatureSetSelector
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=66)

# Average CV score on the training set was: 0.10854004636202527
exported_pipeline = make_pipeline(
    FeatureSetSelector(sel_subset=4065, subset_list="../02/c7.all.v7.0.symbols.csv"),
    RBFSampler(gamma=0.45),
    AddIntercept(),
    LinearSVR(C=0.5, dual=True, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 66)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
