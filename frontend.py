import streamlit as st
from definitions import *

uploader = st.sidebar.file_uploader("Etykietka")

if uploader is not None:
    graph = read_file(uploader)
    st.write(graph)
    st.image(draw_graphs(graph))

else:
    st.title("Flow diagram visualizer")
    st.write(
        '''
        Start by uploading a CSV file using sidebar controls.
        '''
    )
