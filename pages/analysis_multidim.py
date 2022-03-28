import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import plotly.express as px

font = {
    'weight': 'bold',
    'size': 22
}

plt.rc('font', **font)


def app():
    if 'data.csv' not in os.listdir(os.getcwd()):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = pd.read_csv('data.csv')
        df_visual = df.copy()

        categorical = df_visual.select_dtypes(
            include=['object']).columns.values
        numerical = df_visual.select_dtypes(include=[np.number]).columns.values
        obj = df_visual.select_dtypes(include=['object']).columns.values

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

        num = st.multiselect("Select numerical columns ", numerical)

        cats = st.multiselect("Select categorical columns ", categorical)

        if len(num) > 0:
            st.title("Distribution of numerical columns by numerical data")
            for x in num:
                for y in num:
                    if x != y:
                        fig3 = px.box(df, x=x, y=y,
                                      notched=True)
                        st.plotly_chart(fig3)

        if len(cats) > 1:
            st.header("Distribution of categorical columns by categorical data")
            uniques = [np.unique(df[c]) for c in cats]

            for i in range(1, len(cats)):
                st.subheader(f'{cats[0]} by {cats[i]}')
                for u in uniques[0]:
                    data_1 = df[df[cats[0]] == u][cats[i]]
                    val_1 = Counter(data_1)

                    fig = plt.figure(figsize=(20, 20))
                    ax1 = fig.add_subplot()
                    ax1.set_title(f'{cats[0]} : {u}')
                    ax1.pie(val_1.values(), labels=val_1.keys(), autopct='%2.f')
                    ax1.legend()
                    st.pyplot(fig)

        if len(num) > 0 and len(cats) > 0:
            st.title("Distribution of numerical columns by categorical data")
            for c in cats:
                uniques = np.unique(df[c])
                dfs = []
                # We get the dataframes by unique category
                for u in uniques:
                    dfs.append(df[df[c] == u])

                for n in num:
                    fig = plt.figure(figsize=(20, 20))
                    ax = fig.add_subplot()
                    # ax.figure
                    binwidth = (max(df[n]) - min(df[n]))/50
                    ax.hist([dfs[0][n], dfs[1][n]], dfs[2][n], bins=np.arange(min(df[n]), max(
                        df[n]) + binwidth, binwidth), alpha=0.5, stacked=True, label=uniques, color=['r', 'g'])
                    ax.legend(loc='upper right')
                    ax.set_title(n)
                    st.pyplot(fig)
