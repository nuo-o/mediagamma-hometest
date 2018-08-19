import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import xgboost as xgb

from abc import abstractmethod
from my_preprocess import *
from sklearn.externals import joblib
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from my_model import *
from my_preprocess import *


class Time_Tracking():
    
    start_time = None
    
    def start_tracking(self):
        
        self.start_time = time.time()
    
    def stop_tracking(self):
        
        print("Time used:", round(((time.time() - self.start_time)/60),2), ' minutes')
