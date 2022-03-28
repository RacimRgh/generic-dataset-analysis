import streamlit as st
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import StandardScaler


def normalize_column(df, df_last):
    df_last = df.copy()
    df = StandardScaler().fit_transform(df)
    return df, df_last


def chg_type(df, df_last, newtype, column_name):
    df_last = df.copy()

    if newtype == 'numerical':
        df[column_name] = pd.to_numeric(df[column_name], errors='ignore')
    elif newtype == 'object':
        df = df.astype({column_name: str}, errors='ignore')
    elif newtype == 'categorical':
        df = df.astype({column_name: 'category'}, errors='ignore')

    df.to_csv('data.csv', index=False)
    st.success("Your changes have been made!")
    return df, df_last


def dumm(df, df_last, column_name):
    df_last = df.copy()
    df = pd.get_dummies(df, columns=[column_name])
    df.to_csv('data.csv', index=False)
    return df, df_last


def onSubmit(df, df_last, del_na, replace_vals, new_num_val, new_cat_val):
    df_last = df.copy()
    categorical = df.select_dtypes(
        include=['object']).columns.values
    numerical = df.select_dtypes(include=[np.number]).columns.values

    if del_na == "Drop columns":
        df.dropna(axis="columns", how="any", inplace=True)

    elif del_na == "Drop rows":
        df.dropna(axis="rows", how="any", inplace=True)

    elif del_na == "Replace values":
        for num in numerical:
            if replace_vals == "Average/Most frequent":
                new_na_val = df[num].mean()
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

    df.to_csv('data.csv', index=False)
    return df, df_last


def app():
    if 'data.csv' not in os.listdir(os.getcwd()):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        with st.spinner("Loading the cached dataset, please wait..."):
            df = pd.read_csv('data.csv')
            df_og = df.copy()
            df_last = df.copy()

        st.title('Preprocessing')

        """
        Change the columns types one by one
        """
        st.markdown("## Change dataframe columns")
        # Use two column technique
        col1, col2, col3 = st.columns(3)

        column_name = col1.selectbox("Select Column", df.columns)

        if is_numeric_dtype(df[column_name]):
            current_type = 'numerical'
        else:
            try:
                numvals = len(np.unique(df[column_name])
                              ) < 0.2 * len(df[column_name])
            except Exception as ex:
                df[column_name] = le().fit_transform(df[column_name])
                numvals = len(np.unique(df[column_name])
                              ) < 0.2 * len(df[column_name])
            if numvals:
                current_type = 'categorical'
            else:
                current_type = 'object'

        column_options = ['numerical', 'categorical', 'object']
        current_index = column_options.index(current_type)
        newtype = col2.selectbox("Select Column Type",
                                 options=column_options, index=current_index)
        newname = col3.text_input('New column name', value=column_name)

        e1, e2, e3 = st.columns(3)
        e1.write("""Select your column column_name and the new type from the data.
                    To submit all the changes, click on *Submit changes* """)

        if e2.button("Change Column Type"):
            with st.spinner("Modifying type..."):
                df, df_last = chg_type(df, df_last, newtype, column_name)

        if e3.button("Change column name"):
            df.rename(columns={column_name: newname}, inplace=True)
        """
            Check for NA values and show the choices if they exist
                - Drop rows containing at least one NA value
                - Drop the entire column containing NA values
                - Replace the values with: Average/Most frequent, Interpolation (num), Typing
            """
        if pd.isna(df).any().any():
            st.markdown("## List of columns with NA values: ")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            j = 0
            for d in df.columns:
                if df[d].isna().sum():
                    if j == 0:
                        col1.info(d)
                    elif j == 1:
                        col2.info(d)
                    else:
                        col3.info(d)
                        j = 0
                    j += 1

            if len([d for d in df.columns if df[d].isna().sum()]) > 0:
                replace_vals, new_cat_val, new_num_val = None, None, None
                del_na = col4.radio(
                    "Action:", ("Drop rows", "Drop columns", "Replace values"))
                if del_na == 'Replace values':
                    replace_vals = col5.radio(
                        "With:", ("Average/Most frequent", "Interpolation", "Value"))
                    if replace_vals == "Value":
                        new_num_val = st.text_input(
                            "Replacement for numerical NA")
                        new_cat_val = st.text_input(
                            "Replacement for categorical NA")

            b1, b2, b3, b4, b5 = st.columns(5)
            if b5.button("Submit"):
                with st.spinner("Please wait ..."):
                    df, df_last = onSubmit(
                        df, df_last, del_na, replace_vals, new_num_val, new_cat_val)
                    time.sleep(1)
                    df = pd.read_csv('data.csv')
                    st.success("Your changes have been made!")

        st.markdown('## Ratio of different values and dummification')

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric(label="Total number of values", value=len(df[column_name]),
                    delta_color="off")
        col2.metric(label="Number of different values", value=len(np.unique(df[column_name])),
                    delta_color="off")
        if col6.button("Dummify current column", key=current_index):
            with st.spinner("Please wait ..."):
                df, df_last = dumm(df, df_last, column_name)
                time.sleep(1)
                st.success("Your changes have been made!")

        if col7.button("Normalise the column"):
            with st.spinner("Please wait ..."):
                df, df_last = dumm(df, df_last, column_name)
                time.sleep(1)
                st.success("Your changes have been made!")

        b1, b2, b3, b4, b5 = st.columns(5)
        if b2.button("Revert last change"):
            df = df_last.copy()
            df.to_csv('data.csv', index=False)
            st.success("Your changes have been Reverted!")

        if b3.button("Revert all changes"):
            df = df_og.copy()
            df.to_csv('data.csv', index=False)
            st.success("Your changes have been reverted!")

        if b4.button("Delete current column"):
            with st.spinner("Processing..."):
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
