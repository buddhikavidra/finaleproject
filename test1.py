# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:46:57 2020

@author: බුද්ධි
"""

import streamlit as st

name = st.text_input("Enter your name:")
age = st.slider("Your age:", min_value=10, max_value=100)

st.write(f"Hi, ", name, " you are ", age, "years old.")