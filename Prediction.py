#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:58:13 2019
@author: sneha
"""

import pickle
pickle_u_i_bm = pickle.load(open("model.pickle","rb"))
print("Predictions of pickle", pickle_u_i_bm[1])