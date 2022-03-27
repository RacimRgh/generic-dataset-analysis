import streamlit as st
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import time


def chg_type():
    global df_last
    df_last = df.copy()

    if newtype == 'numerical':
        df[column_name] = pd.to_numeric(df[column_name], errors='ignore')
    elif newtype == 'object':
        df = df.astype({column_name: str}, errors='ignore')
    elif newtype == 'categorical':
        df = df.astype({column_name: 'category'}, errors='ignore')
    df.to_csv('data.csv', index=False)
    st.success("Your changes have been made!")


def dumm():
    global dummify
    if dummify:
        df = pd.get_dummies(df, columns=[column_name])
        dummify = False


def onSubmit():
    global df_last, df, del_na
    df_last = df.copy()

    if del_na == "Drop column":
        for column in df.columns:
            if df[column].isna().sum():
                df.dropna(subset=[column], inplace=True, axis=1)

    elif del_na == "Drop rows":
        for column in df.columns:
            if df[column].isna().sum():
                df.dropna(subset=[column], inplace=True, axis=0)

    elif del_na == "Replace values":
        for num in numerical:
            if replace_vals == "Average/Most frequent":
                new_na_val = df[column].mean()
                df[num] = df[num].fillna(new_na_val)

            elif replace_vals == "Interpolation":
                df[num] = df[num].interpolate(
                    method='linear', limit_direction='forward')
            elif replace_vals == "Value":
                df[num].fillna(new_num_val)

        for cat in categorical:
            if replace_vals == "Average/Most frequent":
                new_na_val = df[cat].mode()
                df[cat] = df[cat].fillna(new_na_val)
            elif replace_vals == "Value":
                df[cat].fillna(new_cat_val)

    del_na = False
    df.to_csv('data.csv', index=False)


def app():

    if 'data.csv' not in os.listdir(os.getcwd()):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        global df, df_og, df_last, column_name, newtype, dummify, del_na, replace_vals, new_na_val, numerical, categorical, new_num_val, new_cat_val
        with st.spinner("Loading the cached dataset, please wait..."):
            df = pd.read_csv('data.csv')
            df_og = df.copy()
            df_last = df.copy()

        categorical = df.select_dtypes(
            include=['object']).columns.values
        numerical = df.select_dtypes(include=[np.number]).columns.values

        st.title('Preprocessing')
        st.markdown("## Change dataframe columns")
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

        st.markdown("## List of columns with NA values: ")
        col1, col2, col3, col4 = st.columns(4)
        j = 1
        for d in df.columns:
            if df[d].isna().sum():
                if j == 1:
                    col1.write(d)
                else:
                    col2.write(d)
                j *= -1
        """
        Check for NA values and show the choices if they exist
            - Drop rows containing at least one NA value
            - Drop the entire column containing NA values
            - Replace the values with: Average/Most frequent, Interpolation (num), Typing
        """
        if len([d for d in df.columns if df[d].isna().sum()]) > 0:
            del_na = col3.radio(
                "Action:", ("Drop rows", "Drop columns", "Replace values"))
            if del_na == 'Replace values':
                replace_vals = col4.radio(
                    "With:", ("Average/Most frequent", "Interpolation", "Value"))
                if replace_vals == "Value":
                    new_num_val = st.text_input("Replacement for numerical NA")
                    new_cat_val = st.text_input(
                        "Replacement for categorical NA")

        b1, b2, b3, b4, b5 = st.columns(5)
        if b5.button("Submit"):
            with st.spinner("Please wait ..."):
                onSubmit()
                time.sleep(2)
                st.success("Your changes have been made!")

        st.markdown('## Ratio of different values and dummification')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Total number of values", value=len(df[column_name]),
                    delta_color="off")
        col2.metric(label="Number of different values", value=len(np.unique(df[column_name])),
                    delta_color="off")
        dummify = col4.checkbox("Dumify", key=current_index)

        b1, b2, b3, b4, b5 = st.columns(5)
        if b2.button("Revert last change"):
            df = df_last.copy()
            df.to_csv('data.csv', index=False)
            st.success("Your changes have been Reverted!")
        if b3.button("Revert all changes"):
            df = df_og.copy()
            df.to_csv('data.csv', index=False)
            st.success("Your changes have been reverted!")

        if b4.button("Delete"):
            df_last = df.copy()
            df.drop(column_name, inplace=True, axis=1)
            df.to_csv('data.csv', index=False)
            st.success("Your changes have been made!")

        st.markdown('## Dataframe columns and types')
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
