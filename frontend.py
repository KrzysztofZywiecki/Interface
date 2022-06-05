import streamlit as st
import time
import numpy as np
from io import StringIO
import pygraphviz
from definitions import *


image = st.image(np.random.random((128, 128)), width=400)

uploader = st.sidebar.file_uploader("Etykietka")

if uploader is not None:
    st.sidebar.write(uploader.name)
    stringio = StringIO(uploader.getvalue().decode("utf-8"))
    st.write(read_file(stringio))
