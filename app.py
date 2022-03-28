import streamlit as st
from utils.multipage import MultiPage
from pages import load_data, preprocess, machine_learning, analysis_unidimensional, analysis_multidim, pca

st.set_page_config(layout="wide")

# Custom imports
# import your pages here

# Create an instance of the app
app = MultiPage()

# Add all your application here
app.add_page("Uploading Data", load_data.app)
app.add_page("Preprocessing", preprocess.app)
app.add_page("Unidimensional analysis", analysis_unidimensional.app)
app.add_page("Multidimensional analysis", analysis_multidim.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Dimension reduction with PCA", pca.app)

# The main app
app.run()
