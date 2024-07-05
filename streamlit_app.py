import streamlit as st
import pandas as pd
from io import StringIO
from langchain_community.document_loaders import PyPDFLoader
from high_quality import high_quality_summarization
from regular import regular_summary

st.title("Summarization tool with generative AI  🤖📃")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())
        
if uploaded_file: # check if path is not None
    loader = PyPDFLoader(uploaded_file.name)
    with st.container():
        st.write("---")  # Add a horizontal line for spacing
        centered_container = st.container()
        with centered_container:
            with st.container(border=True):
                # Add two buttons inside the container
                st.subheader("Summarize!")
                col1, col2 = st.columns(2)
                with col1:
                    button1 = st.button("Regular", type="primary")
        
                # Add a button to the second column
                with col2:
                    button2 = st.button("High quality", type="primary")

        st.write("---")  # Add another horizontal line for spacing
        if button2:
            with st.spinner("Generating high-quality summary... This will take some minutes, be patient!"):
                sections, json_str, section_list, output_final = high_quality_summarization(loader)
                print(f"Output: {output_final}")
                st.success("Summary generated successfully!")
            st.write(output_final)
        if button1:
            with st.spinner("Generating regular summary..."):
                output_final = regular_summary(loader)
                print(f"Output: {output_final}")
                st.success("Summary generated successfully!")
            st.write(output_final)
    
