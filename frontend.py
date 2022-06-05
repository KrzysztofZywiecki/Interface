import streamlit as st
from definitions import *

uploader = st.sidebar.file_uploader("Etykietka")

if uploader is not None:
    st.sidebar.write(uploader.name)
    graph = read_file(uploader)
    st.write(graph)
    st.image(draw_graphs(graph))
