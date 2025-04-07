import argparse
import os
import os.path as op
import pickle

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

WORKING_DIR = "..."  # Set your WORKING_DIR here.
SEED = 0
DB = pd.read_csv(op.join(WORKING_DIR, "data", "predict_db.csv"))


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def compute_metrics(y_true, y_pred):
    return {
        "corr": correlation_score(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }


def detect_categorical(data, unique_ratio_threshold=0.05, max_unique=10):
    """
    Detect if a column is likely categorical using heuristics

    Parameters:
    - data: numpy array or pandas series
    - unique_ratio_threshold: max ratio of unique values to total values
    - max_unique: maximum number of unique values for categorical

    Returns:
    - bool: True if likely categorical
    """
    n_samples = len(data)
    n_unique = len(np.unique(data))

    # Check if number of unique values is small
    if n_unique <= max_unique:
        return True

    # Check ratio of unique values to total samples
    if n_unique / n_samples < unique_ratio_threshold:
        return True

    # Check if values are evenly spaced integers
    if data.dtype.kind in "iu":  # Integer type
        values = np.sort(np.unique(data))
        if len(values) > 1:
            gaps = np.diff(values)
            if np.allclose(gaps, gaps[0]):  # Evenly spaced
                return True

    return False


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self._weights = None
        self._mdl = LinearRegression()
        self._encoder = OneHotEncoder(sparse_output=False, drop="first")

    def _reshape_confound(self, confound):
        if isinstance(confound, pd.DataFrame) or isinstance(confound, pd.Series):
            confound = confound.values
        if confound.ndim == 1:
            confound = confound.reshape(-1, 1)
        return confound

    def _convert_categorical(self, X, fit=True):
        """Convert categorical columns to one-hot encoding"""
        if fit:
            return self._encoder.fit_transform(self._reshape_confound(X))
        return self._encoder.transform(self._reshape_confound(X))

    def fit(self, X, y=None, confound=None):
        """Fit confound regressor."""
        if confound is None:
            raise ValueError("Confound matrix must be provided.")
        confound = self._convert_categorical(confound)
        self._mdl.fit(confound, X)
        return self

    def transform(self, X, confound=None):
        """Remove confound effects."""
        if confound is None:
            raise ValueError("Confound matrix must be provided.")
        confound = self._convert_categorical(confound, fit=False)
        return X - self._mdl.predict(confound)

    def fit_transform(self, X, y=None, confound=None):
        """Fit confound regressor and remove confound effects."""
        self.fit(X, y, confound)
        return self.transform(X, confound)


class SHAPTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that computes SHAP values for a given estimator.

    This transformer is compatible with sklearn's pipeline and can handle
    confounds in both fit and transform methods.

    Parameters
    ----------
    estimator : estimator object
        The estimator for which SHAP values will be computed.
        Note that estimator must be a linear model and must be
        already fitted.
    preprocessor : transformer object, optional (default=None)
        Preprocessor to apply to the data before fitting the estimator.
    """

    def __init__(self, estimator, preprocessor=None):
        self.estimator = estimator
        self.preprocessor = preprocessor

    def _preprocess_data(self, X, confound=None, fit=False):
        """
        Apply preprocessing to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        confound : array-like of shape (n_samples, n_confounds), optional (default=None)
            The confounding variables.
        fit : bool, optional (default=False)
            Whether to fit the preprocessor or just transform.

        Returns
        -------
        X_processed : array-like of shape (n_samples, n_features)
            The preprocessed input samples.
        """
        X_processed = X
        if self.preprocessor is not None:
            if confound is not None and hasattr(self.preprocessor, "transform"):
                X_processed = self.preprocessor.transform(X, confound=confound)
            else:
                X_processed = self.preprocessor.transform(X)

        return X_processed

    def fit(self, X, y=None, confound=None):
        """
        Fit the estimator and create the SHAP explainer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), optional (default=None)
            The target values.
        confound : array-like of shape (n_samples, n_confounds), optional (default=None)
            The confounding variables.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check if estimator is fitted
        if not hasattr(self.estimator, "coef_"):
            raise ValueError("Estimator must be fitted before computing SHAP values.")

        # Apply preprocessing if available
        X_processed = self._preprocess_data(X, confound)

        # Create SHAP explainer
        self.explainer_ = shap.explainers.Linear(self.estimator, X_processed)

        return self

    def transform(self, X, confound=None):
        """
        Compute SHAP values for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        confound : array-like of shape (n_samples, n_confounds), optional (default=None)
            The confounding variables.

        Returns
        -------
        shap_values : array-like of shape (n_samples, n_features)
            The SHAP values for each sample and feature.
        """
        check_is_fitted(self, ["explainer_"])

        # Apply preprocessing if available
        X_processed = self._preprocess_data(X, confound)

        # Compute SHAP values
        shap_values = self.explainer_(X_processed)
        return shap_values

    def fit_transform(self, X, y=None, confound=None):
        """
        Fit the estimator and compute SHAP values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), optional (default=None)
            The target values.
        confound : array-like of shape (n_samples, n_confounds), optional (default=None)
            The confounding variables.

        Returns
        -------
        shap_values : array-like of shape (n_samples, n_features)
            The SHAP values for each sample and feature.
        """
        return self.fit(X, y, confound=confound).transform(X, confound=confound)


def train_evaluate(
    est,
    X,
    y,
    confounds=None,
    n_repeats=10,
    n_outer=10,
    n_inner=5,
    optimize=False,
    n_iter=10,
    cv_outer=None,
    cv_inner=None,
    param_grid=None,
    refit=False,
    shap=False,
    random_state=42,
    verbose=1,
    n_jobs=1,
):
    """
    Perform nested cross-validation for model evaluation.

    Parameters:
    - est: Estimator object.
        The estimator to be evaluated.
    - X: array-like of shape (n_samples, n_features).
        The input samples.
    - y: array-like of shape (n_samples,).
        The target values.
    - confounds: array-like or DataFrame, optional.
        Confounding variables to be controlled for during the model evaluation.
    - n_repeats: int, optional (default=10).
        Number of repeated cross-validation folds.
    - n_outer: int, optional (default=10).
        Number of outer cross-validation folds.
    - n_inner: int, optional (default=5).
        Number of inner cross-validation folds.
    - optimize: bool, optional (default=False).
    - n_iter: int, optional (default=10).
        Number of parameter settings that are sampled.
    - cv_outer: cross-validation generator, optional.
        The outer cross-validation strategy.
    - cv_inner: cross-validation generator, optional.
        The inner cross-validation strategy.
    - param_grid: dict or list of dictionaries.
        The parameter grid to be searched.
    - refit: bool, optional (default=False).
        Refit the best model on the entire dataset.
        If shape is True, refit will be set to True by default.
    - shap: bool, optional (default=False).
        Compute SHAP values.
    - random_state: int, optional (default=42).
        Random state for reproducibility.
    - verbose: int, optional (default=1).
        Verbosity level.
    - n_jobs: int, optional (default=1).
        Number of jobs to run in parallel.

    Returns:
    - cv_scores: list of float.
        List of accuracy scores for each outer fold.
    - est: estimator object.
        Refitted estimator on the entire dataset.
    """
    if confounds is not None:
        set_config(enable_metadata_routing=True)

    # Define outer and inner cross-validation
    if cv_outer is None:
        cv_outer = RepeatedKFold(
            n_splits=n_outer, n_repeats=n_repeats, random_state=random_state
        )
    if optimize:
        if cv_inner is None:
            cv_inner = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
        # Initialize random search
        if param_grid is None:
            raise ValueError("Parameter grid is required for optimization.")
        search = RandomizedSearchCV(
            estimator=est,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_inner,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if shap:
        refit = True

    cv_scores = []
    cv_outer_split = cv_outer.split(X)
    if shap:
        shap_values = {}
        shap_values["cv"] = []
        shap_values["total"] = []
    if verbose == 1:
        cv_outer_split = tqdm(cv_outer_split)

    for train_idx, test_idx in cv_outer_split:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if confounds is not None:
            if optimize:
                est = search.fit(
                    X_train, y_train, confound=confounds[train_idx]
                ).best_estimator_
            else:
                est = est.fit(X_train, y_train, confound=confounds[train_idx])
            y_pred = est.predict(X_test, confound=confounds[test_idx])
            if shap:
                shap_values["cv"].append(
                    SHAPTransformer(est[-1], preprocessor=est[:-1])
                    .fit(X_train, confound=confounds[train_idx])
                    .transform(X_test, confound=confounds[test_idx])
                )
        else:
            if optimize:
                est = search.fit(X_train, y_train).best_estimator_
            else:
                est = est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            if shap:
                shap_values["cv"].append(
                    SHAPTransformer(est[-1], preprocessor=est[:-1])
                    .fit(X_train)
                    .transform(X_test)
                )
        cv_scores.append(compute_metrics(y_test, y_pred))

    if refit:
        # Refit the model on the entire dataset.
        if confounds is not None:
            if optimize:
                est = search.fit(X, y, confound=confounds).best_estimator_
            else:
                est.fit(X, y, confound=confounds)
            if shap:
                shap_values["total"] = (
                    SHAPTransformer(est[-1], preprocessor=est[:-1])
                    .fit(X, confound=confounds)
                    .transform(X, confound=confounds)
                )
                return pd.DataFrame(cv_scores), est, shap_values
            return pd.DataFrame(cv_scores), est
        else:
            if optimize:
                est = search.fit(X, y).best_estimator_
            else:
                est.fit(X, y)
            if shap:
                shap_values["total"] = (
                    SHAPTransformer(est[-1], preprocessor=est[:-1]).fit(X).transform(X)
                )
                return pd.DataFrame(cv_scores), est, shap_values
            return pd.DataFrame(cv_scores), est
    if shap:
        return pd.DataFrame(cv_scores), shap_values
    return pd.DataFrame(cv_scores)


def _permuted_score(est, X, y, confounds=None, seed=None, *args, **kwargs):
    np.random.seed(seed)
    y_perm = np.random.permutation(y)  # Shuffle target variables in each iteration.
    cv_scores = train_evaluate(
        est, X, y_perm, confounds=confounds, verbose=0, *args, **kwargs
    ).mean()
    return cv_scores


def permutation_test(
    est,
    X,
    y,
    confounds=None,
    n_permutations=1000,
    metric="corr",
    is_greater=True,
    n_jobs=1,
    *args,
    **kwargs,
):
    """
    Perform a permutation test to evaluate the significance of a model's performance.

    Parameters
    ----------
    est : estimator object
        The model or estimator to be evaluated.
    X : array-like or DataFrame
        The input features for the model.
    y : array-like or Series
        The target variable.
    confounds : array-like or DataFrame, optional
        Confounding variables to be controlled for during the permutation test.
    n_permutations : int, optional
        The number of permutations to perform (default is 1000).
    metric : str, optional
        The performance metric to evaluate (default is "corr").
    is_greater : bool, optional
        If True, the p-value is calculated as the proportion of permuted scores that are greater than the actual score.
        If False, the p-value is calculated as the proportion of permuted scores that are less than the actual score.
    n_jobs : int, optional
        The number of parallel jobs to run (default is 1).
    *args : tuple
        Additional positional arguments to pass to the train_evaluate function.
    **kwargs : dict
        Additional keyword arguments to pass to the train_evaluate function.

    Returns
    -------
    actual_score : float
        The actual performance score of the model.
    p_value : float
        The p-value indicating the significance of the model's performance.
    actual_df : DataFrame
        The DataFrame containing the actual performance scores.
    perm_df : DataFrame
        The DataFrame containing the performance scores from the permutations.
    trained_est : estimator object
        The trained estimator object.
    shap_values : dict
        The SHAP values for the model.
    """
    perm_df = pd.DataFrame()
    print("Evaluating the actual model performance...")
    actual_df, trained_est, shap_values = train_evaluate(
        est,
        X,
        y,
        confounds=confounds,
        refit=True,
        verbose=0,
        shap=True,
        *args,
        **kwargs,
    )
    actual_score = actual_df[metric].mean()
    pbar = tqdm(range(n_permutations), desc="Permutation test iterations...")
    perm_df = pd.DataFrame(
        Parallel(n_jobs=n_jobs)(
            delayed(_permuted_score)(
                est, X, y, confounds=confounds, seed=i, *args, **kwargs
            )
            for i in pbar
        )
    )
    perm_scores = perm_df[metric]

    # P-value is calculated as the proportion of permuted scores that are greater than the actual score.
    if is_greater:
        p_value = (np.sum(perm_scores >= actual_score) + 1) / (n_permutations + 1)
    else:
        p_value = (np.sum(perm_scores <= actual_score) + 1) / (n_permutations + 1)
    return (actual_score, p_value, actual_df, perm_df, trained_est, shap_values)


PIPELINE_BASE = [
    ("selector", VarianceThreshold()),
    ("scaler", MinMaxScaler()),
]

ESTIMATORS = {
    "ridge": Ridge(),
}

PARAM_GRIDS = {
    "ridge": {"regressor__alpha": np.logspace(-3, 3, 10)},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--est", choices=(ESTIMATORS.keys()), help="Model to be evaluated."
    )
    parser.add_argument(
        "--hyperopt",
        action=argparse.BooleanOptionalAction,
        help="Perform hyperparameter optimization.",
    )
    parser.add_argument(
        "--correct",
        action=argparse.BooleanOptionalAction,
        help="Perform confound correction.",
    )
    parser.add_argument(
        "--holdout", action=argparse.BooleanOptionalAction, help="Use holdout data."
    )
    args = parser.parse_args()

    # Set sklearn config
    if args.correct:
        set_config(enable_metadata_routing=True)

    # Output pickle.
    results_path = op.join(WORKING_DIR, "results")
    os.makedirs(results_path, exist_ok=True)
    out_pkl = op.join(
        results_path,
        "".join(
            [
                f"{args.est}_",
                "no-" if not args.correct else "",
                "correct_",
                "no-" if not args.hyperopt else "",
                "hyperopt_",
                "no-" if not args.holdout else "",
                "holdout.pkl",
            ]
        ),
    )
    print(
        f"Trainig estimator: {args.est} with HyperOpt: {args.hyperopt}, Confound: {args.correct}, Holdout: {args.holdout}"
    )
    print(f"Output file: {out_pkl}")

    # Organize the data for prediction.
    # data = DB[DB["site"] != 3]
    data = DB
    y = data[["increase"]].values.ravel()
    conf = data[["cycle"]].values  # Confounding variable
    groups = data[["cycle"]]  # Grouping variable for stratified sampling
    X = data.iloc[:, 6:].values
    if args.holdout:
        X, X_holdout, y, y_holdout, conf, conf_holdout = train_test_split(
            X,
            y,
            conf,
            test_size=0.1,
            random_state=SEED,
            stratify=groups,
        )

    pipeline = PIPELINE_BASE
    if args.correct:
        pipeline = pipeline + [
            (
                "corrector",
                ConfoundRegressor()
                .set_fit_request(confound=True)
                .set_transform_request(confound=True),
            )
        ]
    pipeline = Pipeline(pipeline + [("regressor", ESTIMATORS[args.est])])

    # Common parameters for permutation test
    perm_params = {
        "n_permutations": 1000,
        "n_jobs": -1,
        "param_grid": PARAM_GRIDS[args.est],
        "optimize": args.hyperopt,
    }

    # Add confounds if needed
    if args.correct:
        perm_params["confounds"] = conf

    # Run permutation test
    actual_score, p_value, actual_df, perm_df, trained_est, shap_values = (
        permutation_test(pipeline, X=X, y=y, **perm_params)
    )
    print(f"Correlation: {actual_score:.3f}, p-value: {p_value:.3f}")

    # Compute holdout scores if needed
    if args.holdout:
        if args.correct:
            pred_hold = trained_est.predict(X_holdout, confound=conf_holdout)
            if shap:
                shap_values["holdout"] = (
                    SHAPTransformer(trained_est[-1], preprocessor=trained_est[:-1])
                    .fit(X, confound=conf)
                    .transform(X_holdout, confound=conf_holdout)
                )
        else:
            pred_hold = trained_est.predict(X_holdout)
            if shap:
                shap_values["holdout"] = (
                    SHAPTransformer(trained_est[-1], preprocessor=trained_est[:-1])
                    .fit(X)
                    .transform(X_holdout)
                )
        score_hold = compute_metrics(y_holdout, pred_hold)
        print(f"Holdout correlation: {score_hold['corr']:.3f}")
        with open(out_pkl, "wb") as f:
            pickle.dump(
                {
                    "actual_score": actual_score,
                    "p_value": p_value,
                    "actual_df": actual_df,
                    "perm_df": perm_df,
                    "holdout_score": score_hold,
                    "shap_values": shap_values,
                },
                f,
            )
    else:
        with open(out_pkl, "wb") as f:
            pickle.dump(
                {
                    "actual_score": actual_score,
                    "p_value": p_value,
                    "actual_df": actual_df,
                    "perm_df": perm_df,
                    "shap_values": shap_values,
                },
                f,
            )
    ##TODO: Add option to select specific datasets.
