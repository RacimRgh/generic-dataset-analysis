import streamlit as st
import pandas as pd
import time
import dummy


def app():
    st.title("Uploading dataset")
    st.subheader("File")

    choice = st.radio("Select file type", ["CSV", "XLS", ".data"])

    data_file = st.file_uploader("Upload file")
    global df

    if data_file is not None:

        file_details = {"filename": data_file.name, "filetype": data_file.type,
                        "filesize": data_file.size}

        st.write(file_details)
        if choice == "CSV":
            df = pd.read_csv(data_file)
        elif choice == "XLS":
            df = pd.read_excel(data_file)
            df = df.astype(str)
        print(df.dtypes)
        st.dataframe(df)
        with st.spinner("Loading the dataset, please wait..."):
            df.to_csv('data.csv', index=False)
            time.sleep(1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Rows", value=df.shape[0],
                    delta_color="off")
        col2.metric(label="Columns", value=df.shape[1],
                    delta_color="off")
        col3.metric(label="NA Values", value=df.isna().sum().sum(),
                    delta_color="off")
        col4.metric(label="Constant columns", value=sum([1 for k, v in df.apply(lambda x: len(set(x))).to_dict().items() if v == 1]),
                    delta_color="off")
