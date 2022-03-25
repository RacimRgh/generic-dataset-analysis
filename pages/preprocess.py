from turtle import onclick
import streamlit as st
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np


def app():
    if 'data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = pd.read_csv('data/data.csv')

        st.markdown("#### Change dataframe columns")

        # Use two column technique
        col1, col2 = st.columns(2)

        global name, ctype
        # Design column 1
        name = col1.selectbox("Select Column", df.columns)

        # Design column two
        if is_numeric_dtype(df[name]):
            current_type = 'numerical'
        else:

            if len(np.unique(df[name])) < 0.2 * len(df[name]):
                current_type = 'categorical'
            else:
                current_type = 'object'

        column_options = ['numerical', 'categorical', 'object']
        current_index = column_options.index(current_type)

        ctype = col2.selectbox("Select Column Type",
                               options=column_options, index=current_index)

        e1, e2 = st.columns(2)
        e1.write("""Select your column name and the new type from the data.
                    To submit all the changes, click on *Submit changes* """)
        chg_type = e2.button("Change Column Type")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Total number of values", value=len(df[name]),
                    delta_color="off")
        col2.metric(label="Number of different values", value=len(np.unique(df[name])),
                    delta_color="off")
        dum = col3.checkbox("Dumify", key=current_index)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Missing values", value=df[name].isna().sum(),
                    delta_color="off")
        col2.metric(label="Percentage of missing values", value=df[name].isna().sum()/len(df)*100,
                    delta_color="off")
        if df[name].isna().sum() > 0:
            nas = col3.radio(
                "Action:", ("Drop rows", "Drop column", "Replace values"))
            if nas == 'Replace values':
                if current_type == "numerical":
                    rep = col4.radio(
                        "With:", ("Average", "Value"))
                elif current_type == "numerical":
                    rep = col4.radio(
                        "With:", ("Most frequent", "Interpolation", "A value"))
                if rep == "Value":
                    new_na_val = st.text_input("type value")

        b1, b2, b3, b4, b5 = st.columns(5)
        if b5.button("Submit"):

            if nas == "Drop column":
                df.dropna(subset=[name], inplace=True)
                nas = False

            if nas == "Drop rows":
                df.dropna(subset=[name], axis=0, inplace=True)

            if nas == "Replace values":
                if rep == "Average":
                    new_na_val = df[name].mean()
                elif rep == "Interpolation":
                    df[name] = df[name].interpolate(
                        method='linear', limit_direction='forward')
                elif rep == "Most frequent":
                    new_na_val = df[name].mode()

                df[name] = df[name].fillna(new_na_val)

            if dum:
                df = pd.get_dummies(df, columns=[name])
                dum = False
            if cst:
                pass

        apply = b2.button("Revert last change")
        rev = b3.button("Revert all changes")
        de = b4.button("Delete")

        if chg_type:
            if ctype == 'numerical':
                df[name] = pd.to_numeric(df[name], errors='ignore')
            elif ctype == 'object':
                df = df.astype({name: str}, errors='ignore')
            elif ctype == 'categorical':
                df = df.astype({name: 'category'}, errors='ignore')
        if de:
            df.drop(name, inplace=True, axis=1)
        if rev:
            df = df
        if apply:
            df = df

        st.write("Your changes have been made!")
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

            # c3.checkbox("Dumify", key=idx)
            # c4.checkbox("Delete", key=idx)
            # c5.button('Submit', key=idx)
            # st.write('_____________________________________\n')
            # c5.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
            # typ = c5.radio('Change type to: ', ('Date', 'Object', 'Numeric', 'Auto'), key=idx)
            # c5.selectbox('change type to', options=['Numeric', 'Date', 'Object', 'Auto'], key=idx, index=3)
