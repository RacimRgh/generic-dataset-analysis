import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import plotly.express as px
sns.set_theme(style="ticks", color_codes=True)


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
        if len(num) > 1:
            st.title("Boxplots of numerical columns by numerical data")
            for x in num:
                for y in num:
                    if x != y:
                        fig3 = px.box(df, x=x, y=y,
                                      notched=True)
                        st.plotly_chart(fig3)

        if len(cats) > 1:
            st.header("Piecharts of categorical columns by categorical data")
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
            st.title("Histograms of numerical columns by categorical data")
            choice = st.selectbox("Selection type of graph", [
                                  "Catplot", "Boxplot"])
            for c in cats:
                uniques = np.unique(df[c])
                for n in num:
                    if choice == 'Catplot':
                        fig = sns.catplot(x=c, y=n, data=df)
                    elif choice == 'Boxplot':
                        fig = sns.catplot(x=c, y=n,
                                          kind="box", data=df)
                    fig.set_xticklabels(rotation=30)
                    st.pyplot(fig)
