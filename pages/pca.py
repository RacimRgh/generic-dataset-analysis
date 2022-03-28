import numpy as np
import pandas as pd
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


def app():
    if 'data.csv' not in os.listdir(os.getcwd()):
        st.markdown("Please upload data through Upload Data page!")
    else:
        df = pd.read_csv('data.csv')
        df_visual = df.copy()

        st.title("Dimension reduction")

        if pd.isna(df).any().any():
            st.error(
                "You have NAN values in your dataframe, please go to preprocessing before using PCA")
        else:
            categorical = df_visual.select_dtypes(
                include=['object']).columns.values
            numerical = df_visual.select_dtypes(
                include=[np.number]).columns.values

            cats = st.selectbox("Select categorical columns ", categorical)
            st.markdown("#### Variance expliquee ")
            var = st.slider("Percentage explained variance",
                            min_value=0.4,
                            max_value=0.95,
                            step=0.1,
                            value=0.8,
                            help="This is the value which will be used in pca for dimension reduction. Default = 80%")

            df2 = df_visual[numerical]
            df2 = StandardScaler().fit_transform(df2)

            pca = PCA(n_components=var)
            pca.fit(df2)
            df_p = pca.fit_transform(df2)
            if cats:
                st.title(
                    "Visualisation of the dimension reduction in 2D (2 principal components)")
                fig = px.scatter(df_p, x=0, y=1, color=df_visual[cats])
                st.write(fig)
                st.title(
                    "Visualisation of the dimension reduction in 3D (3 principal components)")
                fig2 = px.scatter_3d(
                    df_p, x=0, y=1, z=2,
                    labels={'0': ' PCA1', '1': 'PCA2', '2': 'PCA3'},
                    color=df_visual[cats])
                st.write(fig2)
