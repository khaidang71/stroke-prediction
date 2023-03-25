# **********************************************

# # Project May Hoc Ung Dung  - Du doan Dot Quy
# # Nguyen Thanh Duy - B1913291 - KHMT
# # Le Huynh Khai Dang - B1913293 - KHMT
# # Phan Van Thanh Ngoan - B1913251 - KHMT

# ***********************************************

#read me: Code dung de so sanh do chinh xac cua thuat toan qua 10 vong lap

import pandas as pd
import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
import matplotlib.ticker as ticker

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
# data.info()

#Thay the gia tri null cua bmi thanh gia tri trung binh
data['bmi'] = data['bmi'].fillna( data['bmi'].mean() ) 
#Xoa cot ID
data = data.drop(columns ='id')
# print(data)
#Thay thuoc tinh other = thuoc tinh tieu bieu (xuat hien nhieu hon)
data['gender'] = data['gender'].replace('Other', list(data.gender.mode().values)[0])


#Chi so BMI toi da la 50
data["bmi"] = pd.to_numeric(data["bmi"])
data["bmi"] = data["bmi"].apply(lambda x: 50 if x>50 else x)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])
df_en = data

# print(df_en.head())

#Xoa di ever_married vi tuong quan du lieu voi age
df_en = df_en.drop(['ever_married'], axis = 1)

#Chuan hoa du lieu: glucose, bmi, age
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
columns = ['avg_glucose_level','bmi','age']
stand_scaled = s.fit_transform(df_en[['avg_glucose_level','bmi','age']])
stand_scaled = pd.DataFrame(stand_scaled,columns=columns)

df_en=df_en.drop(columns=columns,axis=1)
stand_scaled.head()
df = pd.concat([df_en, stand_scaled], axis=1)


x=df.drop(['stroke'], axis=1)
y=df['stroke'] 
# print(df)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

models = dict()
models['KNN'] = KNeighborsClassifier()
models['Naive Bayes'] = GaussianNB()
models['Decision Tree'] = DecisionTreeClassifier()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
list_knn=[]
list_bayes=[]
list_tree=[]


for index in range(10):
    print(f"Number: {index+1}")
    for i in models:
        cnt=0
        total=0
        for a in range (0,10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 100+a)
            # print('-'*20+i+'-'*20)
            models[i].fit(x_train, y_train)
            model = models[i]
            y_pred = model.predict(x_test)
            arg_test = {'y_true':y_test, 'y_pred':y_pred}
            # print(confusion_matrix(**arg_test))
            # print(classification_report(**arg_test))
            total = total + accuracy_score(y_test, y_pred)*100
            cnt = cnt + 1
        avg = round(total / cnt, 2)
        if (i == 'KNN'):
            list_knn.append(avg)
        if (i == 'Naive Bayes'):
            list_bayes.append(avg)
        if (i == 'Decision Tree'):
            list_tree.append(avg)
        print(f"Average accuracy score {i}: ", avg)
        print('-' * 40 )
    print("\n")



# #Ve so do cot so sanh
number = np.array([1,2,3,4,5,6,7,8,9,10])
list_knn = np.array(list_knn)
list_bayes = np.array(list_bayes)
list_tree = np.array(list_tree)
bar_width = 0.27
fig, ax = plt.subplots()

# Vẽ biểu đồ cột
plt.bar(number-bar_width, list_knn, color='#EB7153', width=bar_width, label='KNN')
plt.bar(number, list_bayes, color='#50A625', width=bar_width)
plt.bar(number+bar_width, list_tree, color='#7388C1', width=bar_width)
plt.xticks(number)
# Thêm các giá trị trên mỗi cột
for x, y in zip(number-bar_width, list_knn):
    plt.text(x, y, '%1.2f%%'% y, ha='center', va= 'bottom')
for x, y in zip(number, list_bayes):
    plt.text(x, y, '%1.2f%%'% y, ha='center', va= 'bottom')
for x, y in zip(number+bar_width, list_tree):
    plt.text(x, y, '%1.2f%%'% y, ha='center', va= 'bottom')


# Label x, y axit
plt.legend(["KNN", "Naive Bayes", "Decision Tree"], loc='lower right')
plt.xlabel('Lần Lặp', fontweight ='bold', fontsize = 15)
plt.ylabel('Độ Chính Xác (%)', fontweight ='bold', fontsize = 15)
plt.ylim(0, 100)
# Label title of bar char
plt.title("Biểu Đồ So Sánh Độ Chính Xác Các Giải Thuật\n",fontsize = 20, fontweight ='bold')

# thêm % cho trục y
formatter = ticker.FormatStrFormatter('%1.2f%%')
Axis.set_major_formatter(ax.yaxis, formatter)
for tick in ax.yaxis.get_major_ticks():
	tick.label1.set_color('green')


plt.grid(True)
plt.show()



