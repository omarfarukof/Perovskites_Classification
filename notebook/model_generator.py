import marimo

__generated_with = "0.11.22"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import tabulate
    return np, pd, tabulate


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def _():
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    return RFECV, RandomizedSearchCV, StratifiedKFold, train_test_split


@app.cell
def _():
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    return RandomForestClassifier, SVC, XGBClassifier


@app.cell
def _():
    from sklearn.metrics import make_scorer, balanced_accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    return balanced_accuracy_score, compute_class_weight, make_scorer


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    return LabelEncoder, OneHotEncoder


@app.cell
def _():
    from sklearn.metrics import (classification_report, roc_curve, roc_auc_score,
                                 confusion_matrix, accuracy_score, f1_score, precision_score, 
                                 recall_score)
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import auc
    return (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        f1_score,
        label_binarize,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )


@app.cell
def _():
    import io
    return (io,)


@app.cell
def _(io):
    def to_file(mo_ui_file):
      return io.BytesIO(mo_ui_file.contents())
    return (to_file,)


@app.cell
def _(mo):
    dataset_upload = mo.ui.file(
        kind='area',
        label="# Load Dataset (csv)",
        filetypes=['.csv'],
    )
    return (dataset_upload,)


@app.cell
def _(mo):
    # Run Buttons
    load_data_button = mo.ui.run_button(label="Load Data")
    remove_data_button = mo.ui.run_button(label="Remove Selected Data")
    encode_data_button = mo.ui.run_button(label="Run Encoder")
    model_run_button = mo.ui.run_button(label="Run Model")
    return (
        encode_data_button,
        load_data_button,
        model_run_button,
        remove_data_button,
    )


@app.cell
def _(mo):
    mo.md(r"""# Perovskite Classification""")
    return


@app.cell(hide_code=True)
def _(dataset_upload, load_data_button, mo):
    mo.vstack([
        dataset_upload,
        mo.md(f"**{dataset_upload.name()}**"),
        load_data_button,
    ])
    return


@app.cell(hide_code=True)
def _(dataset_upload, load_data_button, mo):
    # if the button has been clicked, don't run.
    mo.stop((load_data_button.value) and (dataset_upload.name() != None))
    mo.md('No Data is loaded. Please load data.')
    return


@app.cell(hide_code=True)
def _(dataset_upload, load_data_button, mo, pd, to_file):
    # if the button hasn't been clicked, don't run.
    mo.stop((not load_data_button.value) or (dataset_upload.name() == None))
    data = pd.read_csv(to_file(dataset_upload))
    data
    return (data,)


@app.cell
def _(data, mo):
    _default = ["S.No", "Compound", "A", "B", "In literature", "v(B)"]
    feature_drop = mo.ui.table(
        data=data.columns.to_list(),
        label="## Select features to exclude",
        initial_selection= [i for i, x in enumerate(data.columns.to_list()) if x in _default],
    )
    feature_drop
    return (feature_drop,)


@app.cell
def _(data, feature_drop):
    feature_drop_list = feature_drop.value
    final_feature_list = data.columns.difference(feature_drop_list)
    return feature_drop_list, final_feature_list


@app.cell
def _(data, feature_drop, mo):
    remove_data_button_col = mo.ui.table(
        data=list(set(data.columns.to_list()) - set(feature_drop.value)),
        label="## Select features to remove data",
        # initial_selection= [i for i, x in enumerate(_cols) if x in _default],
    )
    remove_data_button_col
    return (remove_data_button_col,)


@app.cell
def _(data, feature_drop, mo, pd, remove_data_button_col):
    def gen_data_remove_table(data: pd.DataFrame, feature_drop: list, remove_data_col: list):

        dr = {}
        for _col in remove_data_col:
            data_remove = mo.ui.table(
                data=data[_col].value_counts().to_dict(),
                label=f"### Select data to remove from [ **{_col}** ]",
            )

            dr[_col] = data_remove
            dr_ui = [ value for _, value in dr.items() ]
        return dr, mo.vstack( dr_ui )

    _data = data.drop(
                feature_drop.value,
                axis=1,
                errors="ignore",
            )
    dr_table, dr_table_ui = gen_data_remove_table(_data, feature_drop, remove_data_button_col.value)
    dr_table_ui
    return dr_table, dr_table_ui, gen_data_remove_table


@app.cell(hide_code=True)
def _(remove_data_button):
    remove_data_button
    return


@app.cell
def get_remove_data(pd):
    def get_remove_data_index(data: pd.DataFrame, dr_table, feature_drop_list):
        dr_dict = {}
        index = data.index
        for key, value in dr_table.items():
            _list = []
            for element in value.value:
                _list.append(element['key'])
            dr_dict[key] = _list
            index = index.difference(data[key].isin(_list).values.nonzero()[0])
        return index
    return (get_remove_data_index,)


@app.cell
def _(
    data,
    dr_table,
    feature_drop_list,
    get_remove_data_index,
    mo,
    remove_data_button,
):
    mo.stop(not remove_data_button.value)
    final_index = get_remove_data_index(data, dr_table, feature_drop_list)
    return (final_index,)


@app.cell
def _(data, final_feature_list, final_index):
    reduced_data = data[final_feature_list].iloc[final_index]
    reduced_data
    return (reduced_data,)


@app.cell
def _(mo):
    mo.md(r"""# Categorical to Numerical Encoding""")
    return


@app.cell
def _(mo, reduced_data):
    # Convert categorical columns to numerical
    catagorical_cols = reduced_data.select_dtypes(include=["object"]).columns
    catagorical_cols
    cat_encoders = {}
    cat_encoders_ui = []

    for _col in catagorical_cols:
        _encoder = mo.ui.dropdown(options=["LabelEncoder", "OneHotEncoder", "Numerical"], label="Choose Encoder", value="Numerical")
        cat_encoders_ui.append(
          mo.hstack([
              mo.md(f"### {_col}"),
              _encoder,
          ]),
        )
        cat_encoders[_col] = _encoder

    mo.vstack(cat_encoders_ui)
    return cat_encoders, cat_encoders_ui, catagorical_cols


@app.cell
def _(encode_data_button):
    encode_data_button
    return


@app.cell
def _(LabelEncoder, OneHotEncoder, pd):
    def get_encoder(encoder_name):
        if encoder_name == "LabelEncoder":
            encoder = LabelEncoder()
        elif encoder_name == "OneHotEncoder":
            encoder = OneHotEncoder()
        else:
            encoder = "to_numeric"
        return encoder

    def to_numeric(data, col):
        return data[col].apply(pd.to_numeric, errors="coerce")

    def get_encoder_from_ui(cat_encoders, catagorical_cols):
        col_encoders = {}
        for _col in catagorical_cols:
            col_encoders[_col] = get_encoder(cat_encoders[_col].value)
        return col_encoders
    return get_encoder, get_encoder_from_ui, to_numeric


@app.cell
def _(LabelEncoder, OneHotEncoder, pd, to_numeric):
    def catagorical_to_numerical(data, col_encoders, catagorical_cols):

        for _col, _ in col_encoders.items():
            if col_encoders[_col] == "to_numeric":
                data[_col] = to_numeric(data, _col)

            elif type(col_encoders[_col]) == type(OneHotEncoder()):
                _encoded_df = col_encoders[_col].fit_transform(data[[_col]]).toarray()
                _one_hot_cols = col_encoders[_col].get_feature_names_out()
                _encoded_df = pd.DataFrame(_encoded_df, columns=_one_hot_cols , dtype=bool)

                data[_encoded_df.columns] = _encoded_df.to_numpy()


            elif type(col_encoders[_col]) == type(LabelEncoder()):
                data[_col] = col_encoders[_col].fit_transform(data[[_col,]].values.ravel())

        return data
    return (catagorical_to_numerical,)


@app.cell
def _(
    cat_encoders,
    catagorical_cols,
    catagorical_to_numerical,
    encode_data_button,
    get_encoder_from_ui,
    mo,
    reduced_data,
):
    mo.stop( not encode_data_button.value )

    col_encoders = get_encoder_from_ui(cat_encoders, catagorical_cols)
    prep_data = catagorical_to_numerical(reduced_data, col_encoders, catagorical_cols)
    prep_data
    return col_encoders, prep_data


@app.cell
def _(mo, prep_data):
    target_ui = mo.ui.dropdown(options=prep_data.columns.to_list() , label="Choose Target")
    target_ui
    return (target_ui,)


@app.cell
def _(mo, prep_data, target_ui):
    mo.stop( target_ui.value == None )

    X = prep_data.drop(target_ui.value, axis=1)
    y = prep_data[target_ui.value]

    mo.vstack([
        mo.md("# X-y Split"),
        mo.md("## [ X Data ]"),
        mo.md("---"),
        X,
        mo.md("---"),
        "",
        "",
        mo.md("## [ y Data ]"),
        mo.md("---"),
        y,
        mo.md("---"),
        "",
        "",
        mo.md('## [ Missing values ]'),
        prep_data.isnull().sum()


    ])
    return X, y


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
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Feature Selection""")
    return


@app.cell
def _(mo):
    feature_selection_button = mo.ui.run_button(label="Run Feature Selection")
    feature_selection_button
    return (feature_selection_button,)


@app.cell
def _(
    RFECV,
    RandomForestClassifier,
    StratifiedKFold,
    X_train,
    balanced_accuracy_score,
    compute_class_weight,
    feature_selection_button,
    make_scorer,
    mo,
    np,
    y_train,
):
    mo.stop(not feature_selection_button.value)



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

    for _ in mo.status.progress_bar(
        range(1),
        title="Loading",
        subtitle="Please wait",
        show_eta=True,
        show_rate=True
    ):
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
    plt.figure(figsize=(8, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Balanced accuracy (cross-validation)")
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Model Training""")
    return


@app.cell
def _(mo):
    _model_names = ['Random Forest Classifier', 'SVM Classifier', 'XGBoost Classifier']
    model_select_ui = mo.ui.dropdown(options=_model_names, value=_model_names[0] , label="Select Model")

    model_select_ui
    return (model_select_ui,)


@app.cell
def _(model_run_button):
    model_run_button
    return


@app.cell
def _(RandomForestClassifier, SVC, XGBClassifier, class_weight_dict):
    def get_model(model_name):
        if model_name == 'Random Forest Classifier':
            _model = RandomForestClassifier(
                class_weight=class_weight_dict,  # Cost-sensitive adjustment
                n_estimators=100,
                random_state=42
            )

        elif model_name == 'XGBoost Classifier':
            _model = XGBClassifier(
                n_estimators=100,
                random_state=42
            )

        elif model_name == 'SVM Classifier':
            _model = SVC(
                class_weight=class_weight_dict,
                random_state=42
            )

        return _model
    return (get_model,)


@app.cell
def _(get_model, mo, model_run_button, model_select_ui):
    mo.stop(not model_run_button.value )
    # model = RandomForestClassifier(
    #     class_weight=class_weight_dict,  # Cost-sensitive adjustment
    #     n_estimators=100,
    #     random_state=42
    # )
    model = get_model(model_select_ui.value)
    model
    return (model,)


@app.cell
def _(X_features, X_test, X_train, mo, model, y_train):
    for _ in mo.status.progress_bar(
        range(1),
        title="Loading",
        subtitle="Please wait",
        show_eta=True,
        show_rate=True
    ):
        model.fit(X_train[X_features], y_train)
    y_pred = model.predict(X_test[X_features])
    return (y_pred,)


@app.cell
def _(confusion_matrix, plt, sns):
    def generate_confusion_matrix(y_true, y_pred, plot=True, title="Confusion Matrix", predicted_lable="Predicted Label", true_lable="Actual Label"):
        # Create a confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if plot:
            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="GnBu", fmt="d")
            plt.xlabel(predicted_lable)
            # plt.xticks(col_encoders[target_ui.value].classes_)
            plt.ylabel(true_lable)
            # plt.yticks(col_encoders[target_ui.value].classes_)
            plt.title(title)

        return cm, plt.gca()
    return (generate_confusion_matrix,)


@app.cell
def _(generate_confusion_matrix, y_pred, y_test):
    _, _plot = generate_confusion_matrix(y_test, y_pred)
    _plot
    return


@app.cell
def _(mo):
    gen_report_button = mo.ui.run_button(label="Generate Report")
    gen_report_button
    return (gen_report_button,)


@app.cell
def _(
    classification_report_to_dataframe,
    col_encoders,
    gen_report_button,
    mo,
    target_ui,
    y_pred,
    y_test,
):
    mo.stop(not gen_report_button.value)
    _labels = col_encoders[target_ui.value].classes_
    # Generate report DataFrames
    classification_report_to_dataframe(
        y_test, y_pred,
        class_names=_labels,
        digits=3
    )
    return


@app.cell
def _(
    X_features,
    X_test,
    auc,
    col_encoders,
    gen_report_button,
    label_binarize,
    mo,
    model,
    np,
    plt,
    roc_curve,
    target_ui,
    y_test,
):
    mo.stop(not gen_report_button.value)

    _class = col_encoders[target_ui.value].classes_

    # Step 1: Get predicted probabilities
    y_probs = model.predict_proba(X_test[X_features])

    # Step 2: Binarize y_test
    n_classes = len(_class)
    y_test_bin = label_binarize(y_test, classes=_class)

    # Step 3: Compute ROC curves for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Step 4: Compute micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Step 5: Compute macro-average (interpolated)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Step 6: Plot
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']

    # Plot individual classes
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Plot micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=4,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=4,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")


    plt.gca()
    return (
        all_fpr,
        color,
        colors,
        fpr,
        i,
        mean_tpr,
        n_classes,
        roc_auc,
        tpr,
        y_probs,
        y_test_bin,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
