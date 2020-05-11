#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:32:07 2020

@author: michaelantia
"""

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import time

main_folder = "/Users/michaelantia/Desktop/Quantum-GroundState-Energy/Data/"
file_name   = "roboBohr.csv"
df = pd.read_csv(main_folder+file_name)
df = df.drop(['pubchem_id'], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)
y = df['Eat']
X = df.drop(['Eat'], axis=1)
X_data = X.as_matrix()
y_data = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.38, random_state = 64)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.50, random_state = 64)
