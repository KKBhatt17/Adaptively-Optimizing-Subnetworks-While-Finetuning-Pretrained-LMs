import pandas as pd

# Train data
df = pd.read_csv("data/training.1600000.processed.noemoticon.csv",encoding='latin-1',names=['label','id','time','query','user','tweet'])
sentiment_dataframe = df[['tweet','label']]

# Test data
# df = pd.read_csv("data/testdata.manual.2009.06.14.csv",encoding='latin-1',names=['label','id','time','query','user','tweet'])

df1 = sentiment_dataframe.iloc[0:10].reset_index(drop=True)
df2 = sentiment_dataframe.iloc[10:20].reset_index(drop=True)
df3 = sentiment_dataframe.iloc[10:20].reset_index(drop=True)

df1.to_csv("data/splits/train.csv")
df2.to_csv("data/splits/val.csv")
df3.to_csv("data/splits/test.csv")