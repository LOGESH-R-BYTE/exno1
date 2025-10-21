# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output

import pandas as pd

data=pd.read_csv(r"C:\Users\MIRDULA\Downloads\Data_set (1).csv")

print(data)

<img width="815" height="826" alt="497386576-644ec88e-42f8-434e-a1e7-b1d6d4e709ca" src="https://github.com/user-attachments/assets/935c1b58-1fbf-4adf-bcb6-f04e49720fcf" />

df.describe()

<img width="824" height="356" alt="image" src="https://github.com/user-attachments/assets/b8e26326-07ff-45de-87cf-7534f0f11c75" />

df=pd.DataFrame(data)

print(df.isnull())

<img width="803" height="680" alt="image" src="https://github.com/user-attachments/assets/dd3eef61-164a-4706-9c57-e78e04722b52" />

df=pd.DataFrame(data)

print(df.isnull().sum())


<img width="375" height="278" alt="image" src="https://github.com/user-attachments/assets/f219e3e0-8e38-4e57-80f8-f9a2f62f9e70" />

df.info()

<img width="502" height="337" alt="image" src="https://github.com/user-attachments/assets/a41ff8ab-bdf9-4845-8a16-856fa6e1f955" />

import pandas as pd

data=pd.read_csv(r"C:\Users\MIRDULA\Downloads\Data_set (1).csv")

df=pd.DataFrame(data)

dfd=df.dropna()

print("AFTER DROPNA")

print(dfd)


<img width="832" height="827" alt="image" src="https://github.com/user-attachments/assets/f84855cd-f3e8-46f1-8793-b0bb3523a31d" />

dfd=df.dropna(axis=1)

print("AFTER DROPNA")

print(dfd)


<img width="658" height="392" alt="image" src="https://github.com/user-attachments/assets/9319dfa5-ba1a-45b3-9ae3-5089bfd89374" />

dfd=df.dropna(axis=1,inplace=True)

print("AFTER DROPNA")

print(dfd)


<img width="625" height="57" alt="image" src="https://github.com/user-attachments/assets/50fc7b69-4821-4c56-9782-034c0ec65ca9" />


df=pd.DataFrame(data)

df1=df.iloc[[1,3,5],[1,3]]

print(df1)


<img width="529" height="108" alt="image" src="https://github.com/user-attachments/assets/5001539d-f26e-4f51-be7f-ce97665cd227" />

dfd=df.dropna(axis=0)

print("AFTER DROPNA")

print(dfd)


<img width="769" height="379" alt="image" src="https://github.com/user-attachments/assets/42006437-2220-41ee-885c-469b0a72f265" />

df=pd.DataFrame(data)

print(df.isnull().any())



<img width="426" height="262" alt="image" src="https://github.com/user-attachments/assets/3057789c-4771-4900-8b2c-f460b59093b4" />

dfd=df.fillna(0)

print("AFTER FILLNA")

print(dfd)


<img width="766" height="734" alt="image" src="https://github.com/user-attachments/assets/128d53b1-4e7e-4e81-bae5-da92aa81331b" />


dfd=df.fillna(method="ffill")

print("AFTER FILLNA")

print(dfd)



<img width="770" height="732" alt="image" src="https://github.com/user-attachments/assets/51be99a9-d23f-43bd-a644-f01876af7a00" />


dfd=df.fillna(method="bfill")

print("AFTER FILLNA")

print(dfd)


<img width="771" height="722" alt="image" src="https://github.com/user-attachments/assets/141a5510-63cb-4c19-b784-b095d1520f33" />


dfd=df.fillna({'show_name':'nandy','aired_on':'wednesday','original_network':'Jio','rating':7.5})

print("AFTER FILLNA")

print(dfd)


<img width="769" height="729" alt="image" src="https://github.com/user-attachments/assets/f33d67d8-bbe5-465e-bd4b-c97d397d9abc" />


import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

data=pd.read_csv("iris.csv")

df=pd.DataFrame(data)

print(df)

x=df["petal_length"]

y=df["sepal_length"]

plt.bar(x,y)

plt.show()


<img width="769" height="760" alt="image" src="https://github.com/user-attachments/assets/cc1528b1-a28d-47bd-bd4e-4ececb57e6b9" />


import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

data=pd.read_csv("iris.csv")

df=pd.DataFrame(data)

print(df)

x=df["petal_length"]

y=df["sepal_length"]

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.plot(x,y)

plt.show()


<img width="771" height="734" alt="image" src="https://github.com/user-attachments/assets/a5a4c9f9-a907-4965-979d-025521cfb30d" />

plt.scatter(x,y)

plt.show()


<img width="782" height="535" alt="image" src="https://github.com/user-attachments/assets/98a47ee9-e57f-485c-a919-59bf99db967a" />


import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

data=pd.read_csv("iris.csv")

df=pd.DataFrame(data)

print(df)

dff=plt.boxplot(x="petal_width",data=df)

print(dff)


<img width="771" height="516" alt="image" src="https://github.com/user-attachments/assets/5c9e7783-b41e-4772-9cc1-d51329c48769" />

import pandas as pd

import numpy as np

from scipy import stats

data=pd.read_csv("iris.csv")

df=pd.DataFrame(data)

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

df_cleaned = df[(z_scores < 3).all(axis=1)]

df_cleaned


<img width="705" height="640" alt="image" src="https://github.com/user-attachments/assets/69eee566-c80b-4c92-b4fd-b4ee679cdc16" />


import pandas as pd

import numpy as np

data_set = pd.read_csv("iris.csv")

df = pd.DataFrame(data_set)

Q1 = df["sepal_width"].quantile(0.25)

Q3 = df["sepal_width"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

print("The Orginal DataSet")

print(df)

outliers = df[(df['sepal_width'] < lower_bound) | (df['sepal_width'] > upper_bound)]

print("The Outliers")

print(outliers)

df_clean = df[(df['sepal_width'] >= lower_bound) & (df['sepal_width'] <= upper_bound)]

print("The Dataset after removing the outliers")

print(df_clean)


<img width="772" height="754" alt="image" src="https://github.com/user-attachments/assets/a2f76634-1fed-4bc7-a5ee-d885101030b7" />

            <<include your coding and its corressponding output screen shots here>>
# Result
        THUS DATA CLEANING IS PERFORMED
        
