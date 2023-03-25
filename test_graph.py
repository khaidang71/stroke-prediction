import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
from sklearn.model_selection import train_test_split


#Thay the gia tri null cua bmi thanh gia tri trung binh
data['bmi'] = data['bmi'].fillna( data['bmi'].mean() ) 
#Xoa cot ID
data = data.drop(columns ='id')
# print(data)
# Thay thuoc tinh other = thuoc tinh tieu bieu (xuat hien nhieu hon)
data['gender'] = data['gender'].replace('Other', list(data.gender.mode().values)[0])


# biểu đồ cột
df_cat = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status', 'stroke']

fig, axs = plt.subplots(4, 2)
axs = axs.flatten()
# # iterate through each column of df_catd and plot
for i, col_name in enumerate(df_cat):
    sns.countplot(x=col_name, data=data, ax=axs[i], hue =data['stroke'], palette = 'deep')
    axs[i].set_xlabel(f"{col_name}", weight = 'bold')
    axs[i].set_ylabel('Count', weight='bold')
    
plt.tight_layout()
plt.show()

# df_num = ['age', 'avg_glucose_level', 'bmi']

# fig, axs = plt.subplots(1, 3, figsize=(16,5))
# axs = axs.flatten()
# # iterate through each column in df_num and plot
# for i, col_name in enumerate(df_num):
#     sns.boxplot(x="stroke", y=col_name, data=data, ax=axs[i],  palette = 'Set1')
#     axs[i].set_xlabel("Stroke", weight = 'bold')
#     axs[i].set_ylabel(f"{col_name}", weight='bold')
# plt.show()



# fig = plt.subplots(nrows=2, ncols=2)

# plt.tight_layout()
# # plt.title("Bar chart of")
# plt.show()