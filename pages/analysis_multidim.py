import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter


def app():
    if 'data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = pd.read_csv('data/data.csv')
        df_visual = df.copy()

        categorical = df_visual.select_dtypes(
            include=['object']).columns.values
        numerical = df_visual.select_dtypes(include=[np.number]).columns.values
        obj = df_visual.select_dtypes(include=['object']).columns.values
        cat_groups = {}
        unique_Category_val = {}

        corr = df.corr(method='pearson')
        fig2, ax2 = plt.subplots()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Colors
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, linewidths=.5,
                    cmap=cmap, center=0, ax=ax2)
        ax2.set_title("Correlation Matrix")
        st.pyplot(fig2)

        df_groupby = df.groupby('diagnosis')
        num = st.multiselect("Select numerical columns ", numerical)
        # for n in num:
        #     res = df.groupby("id")[n].value_counts() / \
        #         df.groupby("id")[n].size()
        #     res = res.unstack()
        #     fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        #     sns.distplot(df.groupby("id").size(),
        #                  ax=ax[0], color="Orange", kde=False, bins=30)

        cats = st.multiselect("Select categorical columns ", categorical)
        print(len(cats), len(num))
        # for c in cats:
        #     df_groupby = df.groupby(c)
        #     st.dataframe(df_groupby)
        #     for u in np.unique(df[c]):
        #         fig = plt.figure(figsize=(20, 20))
        #         ax1 = fig.add_subplot()
        #         ax1.set_title(u)
        #         ax1.pie(c.values(), labels=c.keys(), autopct='%2.f')
        #         ax1.margins()

        #         st.pyplot(fig)
        if len(num) > 0 and len(cats) > 0:
            for c in cats:
                uniques = np.unique(df[c])
                df_grouped = df.groupby(c)
                print(df_grouped.head())
            # fig1, ax1 = plt.subplots()
            # ax1.pie(sizes, explode=explode, labels=labels,
            #         autopct='%1.1f%%', shadow=False, startangle=0)
            # # Equal aspect ratio ensures that pie is drawn as a circle.
            # ax1.axis('equal')
            # ax1.set_title(
            #     'Distribution for categorical Column - ' + (str)(category))
            # cat = st.selectbox("Select categorical variable", categorical)

            # Code base to drop redundent columns
