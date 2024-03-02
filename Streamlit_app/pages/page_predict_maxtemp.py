# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt

def app(pas):
    st.subheader("Décomposition")
    
    st.image('img/maxtemp_decomposition.png', use_column_width=True)
    
    st.subheader("RNN monovarié")

    st.image('img/maxtemp_mono_loss.png', use_column_width=True)
    st.image('img/maxtemp_mono_pred.png', use_column_width=True)
    st.image('img/maxtemp_mono_futur.png', use_column_width=True)

    st.subheader("RNN multivarié")

    st.image('img/maxtemp_multi_loss.png', use_column_width=True)
    st.image('img/maxtemp_multi_pred.png', use_column_width=True)
