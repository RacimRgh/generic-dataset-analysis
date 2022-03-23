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
        e1, e2 = st.columns(2)
        e1.write("""Select your column name and the new type from the data.
                    To submit all the changes, click on *Submit changes* """)
        chg_type = e2.button("Change Column Type")

        a1, a2, a3, a4, a5 = st.columns(5)
        nas = a1.checkbox("Delete NA's", key=current_index)
        dum = a2.checkbox("Dumify", key=current_index)
        cst = a3.checkbox("Delete 3efsa", key=current_index)
        if a5.button("Submit"):
            print(nas, dum, cst)
            if nas:
                df.dropna(subset=[name])
                nas = False
            if dum:
                df = pd.get_dummies(df, columns=[name])
                dum = False
            if cst:
                pass

        b1, b2, b3, b4, b5 = st.columns(5)
        apply = b3.button("Revert last change")
        rev = b4.button("Revert all changes")
        de = b5.button("Delete")
        # st.write(
        #     '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        # choose = st.radio("Pre treatement", ("Delete NA values",
        #                   "Delete constant columns", "Dummify"))
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
