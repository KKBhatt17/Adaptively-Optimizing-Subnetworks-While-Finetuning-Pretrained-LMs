## This file has code for loading the dataset,
## and data-preprocessing and creating train-val-test split.


import pandas as pd

# Loading dataset
df = pd.read_csv("../data/sentiment/zipped/training.1600000.processed.noemoticon.csv",encoding='latin-1',names=['label','id','time','query','user','tweet'])
sentiment_dataframe = df[['tweet','label']]

# Function to convert 0/4 labels to negative/positive.
def label_int_to_string(df):
    target = df["label"].tolist()
    for i in range(len(target)):
        if target[i]==0:
            target[i]="negative"
        elif target[i]==4:
            target[i]="positive"
        else:
            target[i]="neutral"
    newdf = df.drop("label", axis='columns')
    newdf['label'] = target
    return newdf

# def label_int_to_int(df):
#     target = df["label"].tolist()
#     for i in range(len(target)):
#         if target[i]==4:
#             target[i]=1
#     newdf = df.drop("label", axis='columns')
#     newdf['label'] = target
#     return newdf

# Test data
# df = pd.read_csv("data/testdata.manual.2009.06.14.csv",encoding='latin-1',names=['label','id','time','query','user','tweet'])

# Shuffle
sentiment_dataframe = sentiment_dataframe.sample(frac=1)

# Train-Val-Test Split
df1 = sentiment_dataframe.iloc[0:5000].reset_index(drop=True)
df2 = sentiment_dataframe.iloc[100000:101000].reset_index(drop=True)
df3 = sentiment_dataframe.iloc[101000:102000].reset_index(drop=True)
df1 = label_int_to_string(df1)
df2 = label_int_to_string(df2)
df3 = label_int_to_string(df3)
# df1 = label_int_to_int(df1)
# df2 = label_int_to_int(df2)
# df3 = label_int_to_int(df3)

# Save
df1.to_csv("../data/sentiment/splits/train.csv",index=False)
df2.to_csv("../data/sentiment/splits/val.csv",index=False)
df3.to_csv("../data/sentiment/splits/test.csv",index=False)