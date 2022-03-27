import streamlit as st
import numpy as np
import pandas as pd
import time


def app():
    st.title("Uploading dataset")
    st.audio("soolking-suavemente-clip-officiel.mp3")
    st.subheader("File")
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    global df

    if data_file is not None:

        file_details = {"filename": data_file.name, "filetype": data_file.type,
                        "filesize": data_file.size}

        st.write(file_details)
        df = pd.read_csv(data_file)
        st.dataframe(df)

        df.to_csv('data.csv', index=False)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Rows", value=df.shape[0],
                    delta_color="off")
        col2.metric(label="Columns", value=df.shape[1],
                    delta_color="off")
        col3.metric(label="NA Values", value=df.isna().sum().sum(),
                    delta_color="off")
        col4.metric(label="Constant columns", value=sum([1 for k, v in df.apply(lambda x: len(set(x))).to_dict().items() if v == 1]),
                    delta_color="off")


# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)

# st.text_input("Your name", key="name")

# # You can access the value at any point with:
# st.session_state.name

# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     chart_data


# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
#     })

# option = st.selectbox(
#     'Which number do you like best?',
#      df['first column'])

# 'You selected: ', option


# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )


# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")


# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'

# @st.cache  # 👈 This function will be cached
# def my_slow_function(arg1, arg2):
#     # Do something really slow in here!
#     return 'the_output'
