import os
import streamlit as st
import numpy as np
from PIL import Image

# Custom imports
from multipage import MultiPage
# import your pages here
from pages import load_data, preprocess, visualize_data, machine_learning

# Create an instance of the app
app = MultiPage()

# Add all your application here
app.add_page("Uploading Data", load_data.app)
app.add_page("Preprocessing", preprocess.app)
app.add_page("Data visualization", visualize_data.app)
app.add_page("Machine Learning", machine_learning.app)

# The main app
app.run()
