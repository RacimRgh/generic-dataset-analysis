import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def app():
    if 'data.csv' not in os.listdir(os.getcwd()):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = pd.read_csv('data.csv')
        # df_visual = pd.DataFrame(df)
        df_visual = df.copy()

        # categorical,numerical,obj = utils.getColumnTypes(cols)
        categorical = df_visual.select_dtypes(
            include=['object']).columns.values
        numerical = df_visual.select_dtypes(include=[np.number]).columns.values
        obj = df_visual.select_dtypes(include=['object']).columns.values

        unique_Category_val = {cat: np.unique(df[cat]) for cat in categorical}
        cat_groups = {cat: df_visual.groupby(cat) for cat in categorical}

        category = st.selectbox("Select Category ", categorical)

        sizes = (df_visual[category].value_counts() /
                 df_visual[category].count())
        
        labels = sizes.keys()

        maxIndex = np.argmax(np.array(sizes))
        explode = [0]*len(labels)
        explode[int(maxIndex)] = 0.1
        explode = tuple(explode)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=False, startangle=0)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        ax1.set_title(
            'Distribution for categorical Column - ' + (str)(category))
        st.pyplot(fig1)

        categoryobj = st.selectbox(
            "Select " + (str)(category), unique_Category_val[category])

        st.write(cat_groups[category].get_group(categoryobj).describe())
        colName = st.selectbox("Select Column ", numerical)

        st.bar_chart(cat_groups[category].get_group(categoryobj)[colName])

        # Code base to drop redundent columns
