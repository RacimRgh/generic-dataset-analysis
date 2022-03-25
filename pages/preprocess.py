from turtle import onclick
import streamlit as st
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np


def chg_type():
    if newtype == 'numerical':
        df[column_name] = pd.to_numeric(df[column_name], errors='ignore')
    elif newtype == 'object':
        df = df.astype({column_name: str}, errors='ignore')
    elif newtype == 'categorical':
        df = df.astype({column_name: 'category'}, errors='ignore')
    df.to_csv('data/data.csv', index=False)
    st.success("Your changes have been made!")


def onSubmit():
    if dummify:
        df = pd.get_dummies(df, columns=[column_name])
        dummify = False

    if del_na == "Drop column":
        df.dropna(subset=[column_name], inplace=True)
        del_na = False

    elif del_na == "Drop rows":
        df.dropna(subset=[column_name], axis=0, inplace=True)

    elif del_na == "Replace values":
        if replace_vals == "Average":
            new_na_val = df[column_name].mean()
        elif replace_vals == "Interpolation":
            df[column_name] = df[column_name].interpolate(
                method='linear', limit_direction='forward')
        elif replace_vals == "Most frequent":
            new_na_val = df[column_name].mode()

        df[column_name] = df[column_name].fillna(new_na_val)
    df.to_csv('data/data.csv', index=False)
    st.success("Your changes have been made!")


def app():

    if 'data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        global df, df_og, df_last, column_name, newtype, dummify, del_na, replace_vals, new_na_val

        df = pd.read_csv('data/data.csv')
        df_og = df.copy()
        df_last = df.copy()

        st.markdown("#### Change dataframe columns")

        # Use two column technique
        col1, col2 = st.columns(2)

        column_name = col1.selectbox("Select Column", df.columns)

        if is_numeric_dtype(df[column_name]):
            current_type = 'numerical'
        else:
            if len(np.unique(df[column_name])) < 0.2 * len(df[column_name]):
                current_type = 'categorical'
            else:
                current_type = 'object'

        column_options = ['numerical', 'categorical', 'object']
        current_index = column_options.index(current_type)

        newtype = col2.selectbox("Select Column Type",
                                 options=column_options, index=current_index)

        e1, e2 = st.columns(2)
        e1.write("""Select your column column_name and the new type from the data.
                    To submit all the changes, click on *Submit changes* """)
        if e2.button("Change Column Type"):
            chg_type(newtype)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Total number of values", value=len(df[column_name]),
                    delta_color="off")
        col2.metric(label="Number of different values", value=len(np.unique(df[column_name])),
                    delta_color="off")
        dummify = col3.checkbox("Dumify", key=current_index)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Missing values", value=df[column_name].isna().sum(),
                    delta_color="off")
        col2.metric(label="Percentage of missing values", value=df[column_name].isna().sum()/len(df)*100,
                    delta_color="off")

        """
        Check for NA values and show the choices if they exist
            - Drop rows containing at least one NA value
            - Drop the entire column containing NA values
            - Replace the values with: Average/Most frequent, Interpolation (num), Typing
        """
        if df[column_name].isna().sum() > 0:
            del_na = col3.radio(
                "Action:", ("Drop rows", "Drop column", "Replace values"))
            if del_na == 'Replace values':
                if current_type == "numerical":
                    replace_vals = col4.radio(
                        "With:", ("Average", "Value"))
                elif current_type == "numerical":
                    replace_vals = col4.radio(
                        "With:", ("Most frequent", "Interpolation", "A value"))
                if replace_vals == "Value":
                    new_na_val = st.text_input("type value")

        b1, b2, b3, b4, b5 = st.columns(5)
        if b5.button("Submit"):
            onSubmit()

        if b2.button("Revert last change"):
            df = df_last.copy()
            df.to_csv('data/data.csv', index=False)
            st.success("Your changes have been Reverted!")
        if b3.button("Revert all changes"):
            df = df_og.copy()
            df.to_csv('data/data.csv', index=False)
            st.success("Your changes have been reverted!")
        if b4.button("Delete"):
            df.drop(column_name, inplace=True, axis=1)
            df.to_csv('data/data.csv', index=False)
            st.success("Your changes have been made!")

        c = st.columns(6)
        c[0].write('Column')
        c[1].write('Type')
        c[2].write('Column')
        c[3].write('Type')
        c[4].write('Column')
        c[5].write('Type')
        i = 0
        for idx, val in enumerate(df.columns):
            if i == 6:
                i = 0
            c[i].write(val)
            c[i+1].write(df[val].dtype)
            i += 2
