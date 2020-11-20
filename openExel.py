import csv
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np


with open("underexpose_user_feat.csv", "r") as inputFile:
    # 读取csv文件,返回的是迭代类型
    reader = csv.reader(inputFile)
    datarow = []
    feature_name = []
    r = 0
    for row in reader:
        print(row)
        if r > 10:
            break