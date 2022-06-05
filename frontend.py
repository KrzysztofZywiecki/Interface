import streamlit as st
from definitions import *

separator = st.sidebar.text_input('Separator', value=',')
uploader = st.sidebar.file_uploader("Etykietka")

if uploader is not None:
    st.sidebar.write(uploader.name)
    graph = read_file(uploader, separator=separator)
    st.write(graph)
    st.image(draw_graphs(graph))
