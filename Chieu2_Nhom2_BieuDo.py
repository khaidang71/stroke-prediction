# **********************************************

# # Project May Hoc Ung Dung  - Du doan Dot Quy
# # Nguyen Thanh Duy - B1913291 - KHMT
# # Le Huynh Khai Dang - B1913293 - KHMT
# # Phan Van Thanh Ngoan - B1913251 - KHMT

# ***********************************************

#read me: Code dung de ve bieu do tien xu ly du lieu, bieu do trong cua phan lop YES/NO
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

df_cat = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status', 'stroke']

fig, axs = plt.subplots(4, 2, figsize=(14,20))
axs = axs.flatten()

# iterate through each column of df_catd and plot
for i, col_name in enumerate(df_cat):
    sns.countplot(x=col_name, data=data, ax=axs[i], hue =data['stroke'], palette = 'deep')
    axs[i].set_xlabel(f"{col_name}", weight = 'bold')
    axs[i].set_ylabel('Count', weight='bold')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.5)

plt.show()

# Bieu do hinh tron
plt.figure(figsize=(4,4))
data['stroke'].value_counts().plot.pie(autopct='%1.1f%%', colors = ['#66b3ff','#99ff99'])
plt.title("Pie Chart of Stroke Status", fontdict={'fontsize': 14})
plt.tight_layout()
plt.show()

# Chuyen du lieu string thanh number
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


# **************************************************************************
#Lay 1 phan tu ngau nhien
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)



models['KNN'].fit(x_train, y_train)
model = models['KNN']
y_pred = model.predict(x_test)
arg_test = {'y_true':y_test, 'y_pred':y_pred}

# print(confusion_matrix(**arg_test))
# print(classification_report(**arg_test))
matrix = confusion_matrix(**arg_test)
acc_no = matrix[0][0] / (matrix[0][0] + (matrix[0][1])) *100
acc_yes = matrix[1][1] / (matrix[1][0] + (matrix[1][1])) *100
percents = [acc_yes, 100-acc_yes]
stroke = ["Đúng", "Sai"]
colors = ['gray', 'blue']
explode = [0, 0]
plt.pie(percents,
        # colors=colors,
        labels=stroke,
        autopct='%1.2f%%',
        wedgeprops={'edgecolor': 'red', 'linewidth': 1.5},
        explode=explode)
plt.tight_layout()
plt.title('KNN: Biểu đồ tỉ lệ đúng của phân lớp YES')
plt.show()


# print('-'*20+i+'-'*20)
models['Naive Bayes'].fit(x_train, y_train)
model = models['Naive Bayes']
y_pred = model.predict(x_test)
arg_test = {'y_true':y_test, 'y_pred':y_pred}

#gioi_tinh, huyet_ap, tim mach, cong viec, noi o, hut thuoc, luong duong, bmi, tuoi
# print(model.predict([[1,1,1,3,3,1,2,2,2]]))
# print(confusion_matrix(**arg_test))
# print(classification_report(**arg_test))


matrix = confusion_matrix(**arg_test)
acc_no = matrix[0][0] / (matrix[0][0] + (matrix[0][1])) *100
acc_yes = matrix[1][1] / (matrix[1][0] + (matrix[1][1])) *100
# percents = [acc_no, 100-acc_no]
percents = [acc_yes, 100-acc_yes]
stroke = ["Đúng", "Sai"]
colors = ['red', 'green']
explode = [0, 0]
plt.pie(percents,
        # colors=colors,
        labels=stroke,
        autopct='%1.2f%%',
        wedgeprops={'edgecolor': 'red', 'linewidth': 1.5},
        explode=explode)
plt.tight_layout()
plt.title('Naive Bayes: Biểu đồ tỉ lệ đúng của phân lớp YES')
plt.show()



models['Decision Tree'].fit(x_train, y_train)
model = models['Decision Tree']
y_pred = model.predict(x_test)
arg_test = {'y_true':y_test, 'y_pred':y_pred}

# print(confusion_matrix(**arg_test))
# print(classification_report(**arg_test))
matrix = confusion_matrix(**arg_test)
acc_no = matrix[0][0] / (matrix[0][0] + (matrix[0][1])) *100
acc_yes = matrix[1][1] / (matrix[1][0] + (matrix[1][1])) *100
percents = [acc_yes, 100-acc_yes]
stroke = ["Đúng", "Sai"]
colors = ['gray', 'blue']
explode = [0, 0]
plt.pie(percents,
        # colors=colors,
        labels=stroke,
        autopct='%1.2f%%',
        wedgeprops={'edgecolor': 'red', 'linewidth': 1.5},
        explode=explode)
plt.tight_layout()
plt.title('Decision Tree: Biểu đồ tỉ lệ đúng của phân lớp YES')
plt.show()
