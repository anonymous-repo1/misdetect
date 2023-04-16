# with open('data/loan/Wine_Quality_Data.csv', 'r') as f:
#     print(len(f.readlines()))
# exit()
import pandas as pd
import numpy as np

# def normalize(dataframe, column):
#     for i in dataframe:
#         dic = {}
#         num = 0
#         if isinstance(dataframe.loc[0, i], str):
#             for j in dataframe.index:
#                 if dataframe.loc[j, i] not in dic:
#                     dic[dataframe.loc[j, i]] = num
#                     num += 1
#                 dataframe.loc[j, i] = dic[dataframe.loc[j, i]]
#             dataframe[i] = pd.to_numeric(dataframe[i])
#     for i in dataframe:
#         if i != column:
#             dataframe[i] = (dataframe[i] - dataframe[i][np.isfinite(dataframe[i])].mean()) / \
#                            dataframe[i][np.isfinite(dataframe[i])].std()

def normalize(dataframe, column):
    for i in dataframe:
        dic = {}
        num = 0
        if isinstance(dataframe.loc[0, i], str):
            for j in dataframe.index:
                if dataframe.loc[j, i] not in dic:
                    dic[dataframe.loc[j, i]] = num
                    num += 1
                dataframe.loc[j, i] = dic[dataframe.loc[j, i]]
            dataframe[i] = pd.to_numeric(dataframe[i])
    for i in dataframe:
        if i != column:
            dataframe[i] = (dataframe[i] - dataframe[i][np.isfinite(dataframe[i])].mean()) / \
                           dataframe[i][np.isfinite(dataframe[i])].std()


p2 = pd.read_csv('data/loan/Wine_Quality_Data.csv')
# p2 = p2.fillna(axis=1, method='ffill')
# p2 = p2.dropna()
# # p2 = p2.drop(columns=['Name'])
# print(p2.isnull().any())
# p = np.array(p2)
# print(type(p))
#
# print(np.isfinite(np.array(p2, dtype=float)))
# p2['Sentiment']=p2['Sentiment'].astype('int')
normalize(p2, '')

p2.to_csv('data/loan/Wine_Quality_Data_normalize.csv', index=None)
