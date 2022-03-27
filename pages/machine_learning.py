# Import necessary libraries
import json
import joblib

import pandas as pd
import streamlit as st

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import recall_score,accuracy_score,roc_auc_score,f1_score,precision_score

from sklearn.svm import SVC

# Custom classes
import os
from pandas.api.types import is_numeric_dtype


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
            "Select the variables to be used for prediction (X)", options=data.drop(y_var,axis=1).columns)

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
            


            st.markdown("### Model Selection")

            model_selection = st.selectbox("Select a model ",[' ','Logistic Regression',
                'Decision Tree',
                'Support Vector Machines'])
            
            y_pred = []

            if model_selection == ' ':
                pass
            else :

                if model_selection == "Logistic Regression":
                    st.markdown("#### Hyper parameters setting")

                    penalty = st.selectbox("Penalty:",['none','l1','l2','elasticnet'])
                    solver = st.selectbox("Solver:",['newton-cg','lbfgs','liblinear','sag','saga'])
                    max_iter = st.number_input("Number of max iterations :")
                    multiclass_opt = st.selectbox("Multiclass:",['auto','ovr','multinomial'])

                    model = LogisticRegression(random_state=0,penalty=penalty,solver=solver,max_iter=max_iter,multi_class=multiclass_opt).fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    recall = recall_score(y_test, y_pred, average='macro')



                elif model_selection == "Decision Tree":
                    st.markdown("#### Hyper parameters setting")

                    criterion = st.selectbox("Criterion",['gini','entropy'])
                    splitter = st.selectbox("Splitter",['best','random'])
                    max_depth = st.number_input("Max_depth")

                    if max_depth != 0:
                        st.write(f"{max_depth}")
                        model = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=int(max_depth)).fit(X_train, y_train)
                        y_pred = model.predict(X_test)


                else : 
                    st.markdown("#### Hyper parameters setting")
                    kernel = st.selectbox("Kernel",['linear','poly','rbf','sigmoid','precomputed'])
                    gamma = st.selectbox("Gamma",['auto','scale'])
                    probability = st.selectbox("Probability",['On','Off'])
                    if probability == "On":
                        probability = True
                    else :
                        probability = False


                    model = SVC(kernel=kernel,gamma=gamma,probability=probability).fit(X_train, y_train)   
                    y_pred = model.predict(X_test)


                st.markdown("################Training ########################")
                #st.multiselect("Choose the evaluation metrics")
                if y_pred != [] :
                    st.markdown(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
                    st.markdown(f"Recall Score: {recall_score(y_test, y_pred, average='macro')}")
                    st.markdown(f"Precision Score: {precision_score(y_test,y_pred,average='macro')}")
                    st.markdown(f"F1 Score: {f1_score(y_test,y_pred,average='macro')}")
                    st.markdown(f"AUC: {roc_auc_score(y_test,y_pred,average='macro')}")

            if pred_type == "Regression":
                st.write("Running Regression Models on Sample")

                # Table to store model and accurcy
                model_r2 = []

                # Linear regression model
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_r2 = lr_model.score(X_test, y_test)
                model_r2.append(['Linear Regression', lr_r2])

                # Decision Tree model
                model_r2.append(['Decision Tree Regression', 0])

                # Make a dataframe of results
                results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(
                    by='R2 Score', ascending=False)
                st.dataframe(results)

            if pred_type == "Classification":
                st.write("Running Classfication Models on Sample")

                # Table to store model and accurcy
                model_acc = []

                # Linear regression model
                lc_model = LogisticRegression()
                lc_model.fit(X_train, y_train)
                lc_acc = lc_model.score(X_test, y_test)
                model_acc.append(['Logistic Regression', lc_acc])

                # Decision Tree classification
                model_acc.append(['Decision Tree Classification', 0])

                # Make a dataframe of results
                results = pd.DataFrame(model_acc, columns=['Models', 'Accuracy']).sort_values(
                    by='Accuracy', ascending=False)
                st.dataframe(results)
