import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

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
        # for n in num:
        #     res = df.groupby("id")[n].value_counts() / \
        #         df.groupby("id")[n].size()
        #     res = res.unstack()
        #     fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        #     sns.distplot(df.groupby("id").size(),
        #                  ax=ax[0], color="Orange", kde=False, bins=30)

        cats = st.multiselect("Select categorical columns ", categorical)
        # for c in cats:
        #     df_groupby = df.groupby(c)
        #     st.dataframe(df_groupby)
        #     for u in np.unique(df[c]):
        #         fig = plt.figure(figsize=(20, 20))
        #         ax1 = fig.add_subplot()
        #         ax1.set_title(u)q
        #         ax1.pie(c.values(), labels=c.keys(), autopct='%2.f')
        #         ax1.margins()

        #         st.pyplot(fig)
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
                    ax.hist([dfs[0][n], dfs[1][n]], bins=np.arange(min(df[n]), max(
                        df[n]) + binwidth, binwidth), alpha=0.5, stacked=True, label=uniques, color=['r', 'g'])
                    ax.legend(loc='upper right')
                    ax.set_title(n)
                    st.pyplot(fig)

            # fig1, ax1 = plt.subplots()
            # ax1.pie(sizes, explode=explode, labels=labels,
            #         autopct='%1.1f%%', shadow=False, startangle=0)
            # # Equal aspect ratio ensures that pie is drawn as a circle.
            # ax1.axis('equal')
            # ax1.set_title(
            #     'Distribution for categorical Column - ' + (str)(category))
            # cat = st.selectbox("Select categorical variable", categorical)

            # Code base to drop redundent columns
