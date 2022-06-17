import streamlit as st
from definitions import *

separator = st.sidebar.text_input('Separator', value=',')
uploader = st.sidebar.file_uploader("Upload file")

if uploader is not None:
    graph = read_file(uploader, separator=separator)
    st.image(draw_graphs(graph))
    st.write(graph)

else:
    st.title("Flow diagram visualizer")
    st.write(
        '''
        Start by uploading a CSV file using sidebar controls.
        '''
    )
