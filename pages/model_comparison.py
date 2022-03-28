# Import necessary libraries
import json
import joblib

import pandas as pd
from sklearn.svm import SVC
import streamlit as st
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, f1_score, precision_score
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score, mean_squared_error

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import confusion_matrix
# Custom classes
import os
from pandas.api.types import is_numeric_dtype

# import sys
# sys.path.append("../utils/")

from utils.visualisation import RegressionPlot

def evaclass(model,y_pred,y_test,X_test):
    if y_pred != []:
        st.markdown(
            f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
        st.markdown(
            f"Recall Score: {recall_score(y_test, y_pred, average='macro')}")
        st.markdown(
            f"Precision Score: {precision_score(y_test,y_pred,average='macro')}")
        st.markdown(
            f"F1 Score: {f1_score(y_test,y_pred,average='macro')}")
        if len(np.unique(y_test))==2:
            st.markdown(f"AUC: {roc_auc_score(y_test,y_pred,average='macro',multi_class='ovr')}")
        st.subheader("Confusion Matrix")
        disp = plot_confusion_matrix(
            model, X_test, y_test, display_labels=np.unique(y_test))
        st.pyplot(disp.plot().figure_)
        if len(np.unique(y_test))==2:
            st.subheader("ROC Curve")

            st.pyplot(
                plot_roc_curve(model, X_test, y_test).figure_
            )
            st.subheader("Precision-Recall Curve")

            st.pyplot(
                plot_precision_recall_curve(
                    model, X_test, y_test).figure_
            )
def evaReg(y_pred,y_test):
    st.markdown(f"R^2 score: {r2_score(y_test, y_pred)}")
    st.markdown(
        f"Mean absolute error Score: {mean_absolute_error(y_test,y_pred)}")
    st.markdown(
        f"Mean squared error Score: {mean_squared_error(y_test,y_pred)}")
    st.markdown(
        f"Explained variance score Score: {explained_variance_score(y_test,y_pred)}")
    st.pyplot(
        RegressionPlot.predict_regression_plot(y_test, y_pred))
                
def app():
    """This application helps in running machine learning models without having to write explicit code 
    by the user. It runs some basic models and let's the user select the X and y variables. 
    """

    # Load the data
    if 'data.csv' not in os.listdir(os.getcwd()):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data.csv')

        # Create the model parameters dictionary
        params = {}

        # Use two column technique
        col1, col2 = st.columns(2)

        # Design column 1
        y_var = col1.selectbox(
            "Select the variable to be predicted (y)", options=data.columns)

        # Design column 2
        X_var = col2.multiselect(
            "Select the variables to be used for prediction (X)", options=data.drop(y_var, axis=1).columns)

        # Check if len of x is not zero
        if len(X_var) == 0:
            st.error(
                "You have to put in some X variable and it cannot be left empty.")
        # Check if y not in X
        elif y_var in X_var:
            st.error("Warning! Y variable cannot be present in your X-variable.")
        else:
            # Option to select predition type
            pred_type = st.radio("Select the type of process you want to run.",
                                 options=["Regression", "Classification"],
                                 help="Write about reg and classification")

            # Add to model parameters
            params = {
                'X': X_var,
                'y': y_var,
                'pred_type': pred_type,
            }

            # if st.button("Run Models"):

            st.write(f"**Variable to be predicted:** {y_var}")
            st.write(f"**Variable to be used for prediction:** {X_var}")

            # Divide the data into test and train set
            X = data[X_var]
            y = data[y_var]

            # Perform data imputation
            # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")

            # Perform encoding
            X = pd.get_dummies(X)

            # Check if y needs to be encoded
            if not is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)

                # Print all the classes
                st.write(
                    "The classes and the class allotted to them is the following:-")
                classes = list(le.classes_)
                for i in range(len(classes)):
                    st.write(f"{classes[i]} --> {i}")

            # Perform train test splits
            st.markdown("#### Train Test Splitting")
            size = st.slider("Percentage of value division",
                             min_value=0.1,
                             max_value=0.9,
                             step=0.1,
                             value=0.8,
                             help="This is the value which will be used to divide the data for training and testing. Default = 80%")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=size, random_state=42)
            st.write("Number of training samples:", X_train.shape[0])
            st.write("Number of testing samples:", X_test.shape[0])

            ''' RUNNING THE MACHINE LEARNING MODELS '''

            st.markdown("### Model Comparison")
            st.markdown('### Hyper parametrs setting')
            y_pred = []
            col1, col2, col3 = st.columns(3)

            with col1 :
                if pred_type == 'Classification':
                    st.markdown('## Logistic Regression')
                    penalty = st.selectbox(
                        "Penalty:", ['none', 'l1', 'l2', 'elasticnet'])
                    solver = st.selectbox(
                        "Solver:", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                    max_iter = st.number_input("Number of max iterations :")
                    multiclass_opt = st.selectbox(
                        "Multiclass:", ['auto', 'ovr', 'multinomial'])

                    model1 = LogisticRegression(random_state=0, penalty=penalty, solver=solver,
                                            max_iter=max_iter, multi_class=multiclass_opt).fit(X_train, y_train)
                    y_pred1 = model1.predict(X_test)
                    evaclass(model1,y_pred1,y_test,X_test)
                else:
                    st.markdown('## Linear Regression')
                    cal_inter = st.checkbox('Fit intercept ?')
                    normalize = False
                    if cal_inter:
                        normalize = st.checkbox('Normalize data ?')
                    model1 = LinearRegression(
                        fit_intercept=cal_inter, normalize=normalize).fit(X_train, y_train)
                    y_pred1 = model1.predict(X_test)
                    evaReg(y_pred1,y_test)
            with col2:
                if pred_type == 'Classification':
                    st.markdown('## Decission Tree CLassifier')
                    criterion = st.selectbox("Criterion", ['gini', 'entropy'])
                    splitter = st.selectbox("Splitter", ['best', 'random'])
                    max_depth = st.number_input("Max_depth")

                    # Handling under 0 problem
                    if max_depth < 1:
                        max_depth = 1

                    if max_depth != 0:
                        st.write(f"{max_depth}")
                        model2 = DecisionTreeClassifier(
                            criterion=criterion, splitter=splitter, max_depth=int(max_depth)).fit(X_train, y_train)
                        y_pred2 = model2.predict(X_test)
                        evaclass(model2,y_pred2,y_test,X_test)
                else:
                    st.markdown('## Decision Tree Regressor')
                    criterion = st.selectbox(
                        "Criterion", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
                    splitter = st.selectbox("Splitter", ['best', 'random'])
                    max_depth = st.number_input("Max_depth")
                    # Handling under 0 problem
                    if max_depth < 1:
                        max_depth = 1

                    if max_depth != 0:
                        st.write(f"{max_depth}")
                        model2 = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=int(
                            max_depth)).fit(X_train, y_train)
                        y_pred2 = model2.predict(X_test)
                        evaReg(y_pred2,y_test)
            with col3:
                if pred_type == 'Classification':
                    # SVM Model
                    st.markdown('## Support Vector Machine')
                    kernel = st.selectbox(
                        "Kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
                    gamma = st.selectbox("Gamma", ['auto', 'scale'])
                    probability = st.selectbox("Probability", ['On', 'Off'])
                    if probability == "On":
                        probability = True
                    else:
                        probability = False

                    model3 = SVC(kernel=kernel, gamma=gamma,
                                probability=probability).fit(X_train, y_train)
                    y_pred3 = model3.predict(X_test)
                    evaclass(model3,y_pred3,y_test,X_test)
                else:
                    st.markdown('## Light Gradient Boosting')
                    learning_rate = st.selectbox(
                        'learning rate', [0.0001, 0.001, 0.01, 0.1, 1.0, 'other'])
                    if learning_rate == 'other':
                        learning_rate = st.number_input(
                            'insert the value of the learning rate', value=0.01)
                    num_leaves = st.number_input(
                        'max number of leaves', value=32, min_value=1)
                    n_estimators = st.number_input(
                        'Number of boosted trees to fit', value=100, min_value=1)

                    if learning_rate == 'other':
                        learning_rate = st.number_input(
                            'insert the value of the learning rate')
                    boosting_type = st.selectbox('Boosting type', ['gbdt', 'dart', 'goss'], help="\
                        Gradient Boosting Decision Tree (GDBT).\
                        Dropouts meet Multiple Additive Regression Trees (DART).\
                        Gradient-based One-Side Sampling (GOSS).")
                    model3 = LGBMRegressor(boosting_type=boosting_type, learning_rate=float(learning_rate), num_leaves=num_leaves, n_estimators=n_estimators)\
                        .fit(X_train, y_train)
                    y_pred3 = model3.predict(X_test)
                    evaReg(y_pred3,y_test)