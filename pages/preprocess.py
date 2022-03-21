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

        print(name, current_type)
        column_options = ['numerical', 'categorical', 'object']
        current_index = column_options.index(current_type)

        ctype = col2.selectbox("Select Column Type",
                               options=column_options, index=current_index)

        st.write("""Select your column name and the new type from the data.
                    To submit all the changes, click on *Submit changes* """)

        b1, b2, b3, b4, b5 = st.columns(5)
        chg_type = b1.button("Change Column Type")
        dum = b2.button("Dummify")
        de = b3.button("Delete")
        rev = b4.button("Revert all changes")
        apply = b5.button("Apply all changes")
        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        choose = st.radio("Pre treatement", ("Delete NA values",
                          "Delete constant columns", "Dummify"))
        if chg_type:
            if ctype == 'numerical':
                df[name] = pd.to_numeric(df[name], errors='ignore')
            elif ctype == 'object':
                df = df.astype({name: str}, errors='ignore')
            elif ctype == 'categorical':
                df = df.astype({name: 'category'}, errors='ignore')
        if dum:
            df = pd.get_dummies(df, columns=[name])
        if de:
            df.drop(name, inplace=True, axis=1)
        if rev:
            df = df
        if apply:
            df = df

        st.write("Your changes have been made!")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.write('Column')
        c2.write('Type')
        c3.write('Dummiffy')
        c4.write('Delete')
        c5.write('Type modification')
        for idx, val in enumerate(df.columns):
            c1.write(val)
            c2.write(df[val].dtype)

            # c3.checkbox("Dumify", key=idx)
            # c4.checkbox("Delete", key=idx)
            # c5.button('Submit', key=idx)
            # st.write('_____________________________________\n')
            # c5.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
            # typ = c5.radio('Change type to: ', ('Date', 'Object', 'Numeric', 'Auto'), key=idx)
            # c5.selectbox('change type to', options=['Numeric', 'Date', 'Object', 'Auto'], key=idx, index=3)
