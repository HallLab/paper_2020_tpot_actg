import numpy as np
import pandas as pd
from custom_transform import AddIntercept
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import FeatureSetSelector
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=90)

# Average CV score on the training set was: 0.013936031683344318
exported_pipeline = make_pipeline(
    FeatureSetSelector(sel_subset=766, subset_list="../02/c7.all.v7.0.symbols.csv"),
    PCA(iterated_power=3, svd_solver="randomized"),
    AddIntercept(),
    KNeighborsRegressor(n_neighbors=16, p=2, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 90)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
