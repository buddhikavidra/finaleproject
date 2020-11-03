# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:25:23 2020

@author: බුද්ධි
"""

#importing basic libraries
import re
import math 
import numpy as np
import streamlit as st 





#if st.button("Click Me")
def fun():
    xxx = []
    
    x = []
    DATA_DIR = "D:/reserch/dattoprotect"
    keyword = user_input
    import os
    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            print(os.path.join(DATA_DIR+"/", file))
            xx = os.path.join(DATA_DIR+"/", file)
            xxx.append(xx)
    
    
    
    
    #making the dictionary of the document
    def word_dictionary(document):
        dictionary = {}
        with open(document, 'r') as file:
            
            text = file.read().lower()
            # replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
            text = re.sub('[^a-z\ \']+', " ", text)  # For some reason, even though the text is in lower case, the code does't work unless i redo that condition
            Words = list(text.split())  # put text into an empty list using split()
            for i in Words:
                if i in dictionary:
                    dictionary[i] += 1
                else:
                    dictionary[i] = 1
        return dictionary
    
    # making the inverted index of the document
    def make_invertedIndex(document):
        inverted = {}
        with open(document, 'r') as f:
            lines = f.read().splitlines()   # making a list of all documents seperated by a newline character
    
        idx = 1                                         # maintaining the current document index
        for docs in lines:
            doc_words = list(docs.split())               
            # for each word in documents
            for word in doc_words:
                if word in inverted:                    # if the word exists in the inverted index
                    if idx not in inverted[word]:       # if current document is not in the value of this word
                        inverted[word].append(idx)      # add the current document as a value for the current word
                else:
                    inverted[word] = [idx]              # if word is not a key of invertedindex then make a new key
            idx += 1;
        return inverted
    
    
    for iii in xxx:
        with open(iii , 'r') as f:
            lines = f.read().splitlines()
    
    # making the dictionary
        dictionary = word_dictionary(iii)
        print(iii)
        print ("Words in dictionary:  " , len(dictionary))
        
        # making the inverted index
        inverted = make_invertedIndex(iii)
        
        # comparing with queries
    for root, dirs, files in os.walk(DATA_DIR, onerror=None): 
        for filename in files:  
            file_path = os.path.join(root, filename)
            #dictionary = make_dictionary(iii)
            #print ("Words in dictionary:  " , len(dictionary))
            try:
                with open(file_path, "rb") as f:  # open the file for reading
                    
                    for line in f:  
                        try:
                            line = line.decode("utf-8")
                            #queries = f.read().splitlines()
                            #lines = f.read().splitlines()
                            
                        except ValueError:  
                            continue
                        if keyword in line:  
                            print('=========================================')
                            print('keyword  :',keyword,' : contains in')
                            print('contents:')
                            st.write('keyword  :',keyword,' : contains in')
                            st.write('keyword  :',keyword)
                            #st.write
                            print(len(dictionary))
                            print(line)
                            if line not in x:
                                x.append(line)
                                st.write(x)
                            print(file_path)
                            st.write(file_path)
                            continue  
            except (IOError, OSError):  
                pass
    st.write('end')
#st.write(x)
    
st.title('search anything related to find case law')
user_input = st.text_area("label goes here")
if st.button('search'):
    fun()