import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                           chi2, f_classif)
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    from sklearn.pipeline import Pipeline
    from sklearn.base import clone
    from skfeature.function.similarity_based import reliefF
    import joblib
    from sklearn.preprocessing import LabelEncoder


    import matplotlib.pyplot as plt
    import seaborn as sns
    return (
        KNNImputer,
        LabelEncoder,
        LogisticRegression,
        Pipeline,
        RandomizedSearchCV,
        SelectKBest,
        StandardScaler,
        XGBClassifier,
        chi2,
        clone,
        f_classif,
        joblib,
        mutual_info_classif,
        np,
        pd,
        plt,
        reliefF,
        sns,
        train_test_split,
    )


@app.cell
def _():
    from sklearn.metrics import (classification_report, roc_auc_score,
                                 confusion_matrix, accuracy_score, f1_score, precision_score, 
                                 recall_score)
    return (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, balanced_accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    return (
        RFECV,
        RandomForestClassifier,
        SVC,
        StratifiedKFold,
        balanced_accuracy_score,
        compute_class_weight,
        make_scorer,
    )


@app.cell
def _(mo, pd):
    # Load dataset
    data_path = mo.notebook_location() / "public" / "perovskite_data.csv"
    data = pd.read_csv(data_path)
    data['Lowest distortion'].value_counts()
    return data, data_path


@app.cell
def _(np, pd):
    # Preprocessing
    def preprocess_data(data):
        # Drop non-feature columns
        data = data.drop(
            ["S.No", "Compound", "A", "B", "In literature", "v(B)"],
            axis=1,
            errors="ignore",
        )

        # no_ld_idx = data["Lowest distortion"] == "-"

        # vA_idx = data["v(A)"]
        # no_vA_idx = (vA_idx == "not balanced") | (vA_idx == "element not in BV")
        # # not (no_Lowest_distion) and (no_vA_idx)
        # # only_no_vA_idx = ~(no_ld_idx) & no_vA_idx

        # vB_idx = data["v(B)"]
        # no_vB_idx = (vB_idx == "not balanced") | (vB_idx == "element not in BV")
        # # not (no_Lowest_distion) and (no_vA_idx)
        # # only_no_vB_idx = ~(no_ld_idx) & no_vB_idx

        # rmv_idx = no_ld_idx | no_vA_idx | no_vB_idx
        # data = data[~rmv_idx]

        # data = data.replace(["-"], np.nan)
        data.replace(["element not in BV", "-", "not available"], np.nan, inplace=True)
        # Remove rows with missing values
        data = data.dropna()

        # Convert numerical columns to float
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        data[numeric_cols] = data[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        data["τ"] = data["τ"].apply(pd.to_numeric, errors="coerce")

        # Convert categorical columns to numerical
        catagorical_cols = data.select_dtypes(include=["object"]).columns
        catagorical_cols = catagorical_cols[catagorical_cols != "Lowest distortion"]  # remove tau it

        cat_options = []
        for col in catagorical_cols:
            cat_options.append(data[col].value_counts().index.values)

        # One Hot Encoding
        for col, options in zip(catagorical_cols, cat_options):
            for option in options:
                data[f"{col}_{option}"] = (data[col] == option).values
        data = data.drop(catagorical_cols, axis=1)

        return data
    return (preprocess_data,)


@app.cell
def _(LabelEncoder, data, preprocess_data):
    # Feature-target separation
    prep_data = preprocess_data(data)
    X = prep_data.drop(["Lowest distortion"], axis=1)  # Multiple targets possible
    y_structure = prep_data["Lowest distortion"]  # Binary classification target

    # create a LabelEncoder object
    le = LabelEncoder()
    # fit and transform the categorical data
    y_structure = le.fit_transform(y_structure)
    y_structure
    return X, le, prep_data, y_structure


@app.cell
def _(prep_data):
    prep_data
    return


@app.cell
def _(prep_data):
    prep_data.dtypes
    return


@app.cell
def _(KNNImputer, X, pd, train_test_split, y_structure):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_structure, test_size=0.2, stratify=y_structure, random_state=42
    )

    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5)

    _data_imputed = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(_data_imputed, columns=X_train.columns)

    _data_imputed = imputer.transform(X_test)
    X_test = pd.DataFrame(_data_imputed, columns=X_test.columns)
    return X_test, X_train, imputer, y_test, y_train


@app.cell
def _(X_test, X_train):
    (X_train.isnull().sum() , X_test.isnull().sum())
    return


@app.cell
def _(StandardScaler, X_test, X_train):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, scaler


@app.cell
def _(
    LogisticRegression,
    RFECV,
    RandomForestClassifier,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    np,
    reliefF,
):
    # Feature Selection Strategies
    def apply_feature_selection(method, X, y):
        if method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=10)
        elif method == "chi2":
            selector = SelectKBest(chi2, k=10)
        elif method == "anova":
            selector = SelectKBest(f_classif, k=10)
        elif method == "relieff":
            score = reliefF.reliefF(X.values, y.values, mode="rank")
            top_features = np.argsort(score)[-10:]
            return X.iloc[:, top_features]
        elif method == "rfe":
            estimator = LogisticRegression(class_weight="balanced", max_iter=1000)
            selector = RFECV(estimator, step=1, cv=5)
        elif method == "embedded":
            selector = RandomForestClassifier(class_weight="balanced", n_estimators=100)
        return selector.fit_transform(X, y)
    return (apply_feature_selection,)


@app.cell
def _(RandomForestClassifier, SVC, XGBClassifier, np, y_train):
    # # Model Definitions with Cost-Sensitive Learning
    # models = {
    #     "XGBoost": XGBClassifier(
    #         scale_pos_weight=(y_train == 0).sum()/(y_train == 1).sum(),
    #         eval_metric="logloss",
    #         use_label_encoder=False
    #     ),
    #     "Balanced RF": RandomForestClassifier(
    #         class_weight="balanced",
    #         n_estimators=100
    #     ),
    #     "Weighted SVM": SVC(
    #         class_weight="balanced",
    #         probability=True,
    #         kernel="rbf"
    #     )
    # }

    # Model Definitions with Cost-Sensitive Learning for Multi-Class Classification
    models = {
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            scale_pos_weight=[(y_train == i).sum() / (y_train != i).sum() for i in np.unique(y_train)]
        ),
        "Cost-Sensitive RF": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100
        ),
        "Cost-Sensitive SVM": SVC(
            class_weight="balanced",
            probability=True,
            kernel="rbf",
            decision_function_shape='ovr'
        )
    }
    return (models,)


@app.cell
def _():
    # Hyperparameter Grids
    param_grids = {
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        },
        "Balanced RF": {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10]
        },
        "Weighted SVM": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"]
        }
    }
    return (param_grids,)


@app.cell
def _(
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
):
    # Performance Evaluation Framework
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }

        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        return metrics
    return (evaluate_model,)


@app.cell
def _(
    RFECV,
    RandomForestClassifier,
    StratifiedKFold,
    X_train,
    balanced_accuracy_score,
    compute_class_weight,
    make_scorer,
    np,
    y_train,
):
    # Compute class weights for cost sensitivity
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    # Initialize cost-sensitive classifier
    clf = RandomForestClassifier(
        class_weight=class_weight_dict,  # Cost-sensitive adjustment
        n_estimators=100,
        random_state=42
    )

    # Custom scoring metric for imbalanced multi-class
    scorer = make_scorer(balanced_accuracy_score)

    # RFECV with cost-sensitive learning
    rfecv = RFECV(
        estimator=clf,
        step=1,  # Remove 1 feature per iteration
        cv=StratifiedKFold(5),  # Stratified for imbalanced data
        scoring=scorer,
        min_features_to_select=5,
        n_jobs=-1
    )

    # Fit RFECV
    rfecv.fit(X_train, y_train)
    return class_weight_dict, class_weights, classes, clf, rfecv, scorer


@app.cell
def _(X_train, rfecv):
    # Results
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Selected features: {X_train.columns[rfecv.support_].values}")
    X_features = X_train.columns[rfecv.support_].values.tolist()
    X_features
    return (X_features,)


@app.cell
def _(plt, rfecv):
    # Plot cross-validation performance

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Balanced accuracy (cross-validation)")
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Random Forest""")
    return


@app.cell
def _(
    RandomForestClassifier,
    X_features,
    X_test,
    X_train,
    class_weight_dict,
    y_train,
):
    _model = RandomForestClassifier(
        class_weight=class_weight_dict,  # Cost-sensitive adjustment
        n_estimators=100,
        random_state=42
    )
    _model.fit(X_train[X_features], y_train)
    y_pred = _model.predict(X_test[X_features])
    return (y_pred,)


@app.cell
def _(confusion_matrix, plt, sns, y_pred, y_test):
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="GnBu", fmt="d")
    plt.xlabel("Predicted Stucture")
    plt.ylabel("True Stucture")
    plt.title("Confusion matrix")
    plt.gca()
    return (cm,)


@app.cell
def _(classification_report, mo, pd):
    def classification_report_to_dataframe(y_true, y_pred, class_names=None, digits=2, title="Classification Report", title_level=1):
        """
        Convert scikit-learn classification report to pandas DataFrame

        Parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names in order
        digits (int): Number of decimal places for metrics

        Returns:
        tuple: (metrics_df, summary_df) - Metrics by class and summary statistics
        """
        # Generate raw classification report dictionary
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True,
            digits=digits
        )

        # Extract class-wise metrics and support
        metrics_df = pd.DataFrame(report_dict).transpose().reset_index()
        metrics_df = metrics_df.rename(columns={'index': 'class'})

        # Separate class metrics from summary statistics
        summary_mask = metrics_df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])
        summary_df = metrics_df[summary_mask].copy()
        metrics_df = metrics_df[~summary_mask].copy()

        # Clean up formatting
        for col in ['precision', 'recall', 'f1-score']:
            metrics_df[col] = metrics_df[col].apply(
                lambda x: f"{x:.{digits}f}" if not pd.isna(x) else ""
            )
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.{digits}f}" if not pd.isna(x) else ""
            )

        # Add total support count
        total_support = metrics_df['support'].sum()
        summary_df.loc[summary_df['class'] == 'accuracy', 'support'] = total_support

        report = mo.vstack([
        mo.md(f"{'#'*title_level} {title}"),
        mo.md("## Class-wise Metrics:"),
        mo.md(metrics_df.to_markdown(index=False)),
        mo.md("## Summary Statistics:"),
        mo.md(summary_df.to_markdown(index=False))
    ])
        return report
    return (classification_report_to_dataframe,)


@app.cell
def _(classification_report_to_dataframe, le, y_pred, y_test):
    # Generate report DataFrames
    classification_report_to_dataframe(
        y_test, y_pred,
        class_names=le.classes_,
        digits=3
    )
    return


if __name__ == "__main__":
    app.run()
