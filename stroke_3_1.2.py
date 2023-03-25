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
# df_num = ['age', 'avg_glucose_level', 'bmi']
#
# fig, axs = plt.subplots(1, 3, figsize=(16,5))
# axs = axs.flatten()
#
# # iterate through each column in df_num and plot
# for i, col_name in enumerate(df_num):
#     sns.boxplot(x="stroke", y=col_name, data=data, ax=axs[i],  palette = 'Set1')
#     axs[i].set_xlabel("Stroke", weight = 'bold')
#     axs[i].set_ylabel(f"{col_name}", weight='bold')
# plt.show()

#Chi so BMI toi da la 50
data["bmi"] = pd.to_numeric(data["bmi"])
data["bmi"] = data["bmi"].apply(lambda x: 50 if x>50 else x)

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


# for index in range(10):
#     print(f"Number: {index+1}")
#     for i in models:
#         cnt=0
#         total=0
#         for a in range (0,10):
#             x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#             # print('-'*20+i+'-'*20)
#             models[i].fit(x_train, y_train)
#             model = models[i]
#             y_pred = model.predict(x_test)
#             arg_test = {'y_true':y_test, 'y_pred':y_pred}
#             # print(confusion_matrix(**arg_test))
#             # print(classification_report(**arg_test))
#             total = total + accuracy_score(y_test, y_pred)*100
#             cnt = cnt + 1
#         avg = round(total / cnt, 2)
#         if (i == 'KNN'):
#             list_knn.append(avg)
#         if (i == 'Naive Bayes'):
#             list_bayes.append(avg)
#         if (i == 'Decision Tree'):
#             list_tree.append(avg)
#         print(f"Average accuracy score {i}: ", avg)
#         print('-' * 40 )
#     print("\n")



# # Ve so do cot so sanh
# number = np.array([1,2,3,4,5,6,7,8,9,10])
# list_knn = np.array(list_knn)
# list_bayes = np.array(list_bayes)
# list_tree = np.array(list_tree)
# bar_width = 0.27
# fig, ax = plt.subplots()

# # Vẽ biểu đồ cột
# plt.bar(number-bar_width, list_knn, color='#EB7153', width=bar_width, label='KNN')
# plt.bar(number, list_bayes, color='#50A625', width=bar_width)
# plt.bar(number+bar_width, list_tree, color='#7388C1', width=bar_width)
# plt.xticks(number)
# # Thêm các giá trị trên mỗi cột
# for x, y in zip(number-bar_width, list_knn):
#     plt.text(x, y, '%1.2f%%'% y, ha='center', va= 'bottom')
# for x, y in zip(number, list_bayes):
#     plt.text(x, y, '%1.2f%%'% y, ha='center', va= 'bottom')
# for x, y in zip(number+bar_width, list_tree):
#     plt.text(x, y, '%1.2f%%'% y, ha='center', va= 'bottom')


# # Label x, y axit
# plt.legend(["KNN", "Naive Bayes", "Decision Tree"], loc='lower right')
# plt.xlabel('Lần Lặp', fontweight ='bold', fontsize = 15)
# plt.ylabel('Độ Chính Xác (%)', fontweight ='bold', fontsize = 15)
# plt.ylim(0, 100)
# # Label title of bar char
# plt.title("Biểu Đồ So Sánh Độ Chính Xác Các Giải Thuật\n",fontsize = 20, fontweight ='bold')

# # thêm % cho trục y
# formatter = ticker.FormatStrFormatter('%1.2f%%')
# Axis.set_major_formatter(ax.yaxis, formatter)
# for tick in ax.yaxis.get_major_ticks():
# 	tick.label1.set_color('green')

# plt.grid(True)
# plt.show()

# # **************************************************************************
#Lay 1 phan tu ngau nhien
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
# print('-'*20+i+'-'*20)
models['Naive Bayes'].fit(x_train, y_train)
model = models['Naive Bayes']
models['Decision Tree'].fit(x_train, y_train)
md_Tree = models['Decision Tree']
models['KNN'].fit(x_train, y_train)
md_KNN = models['KNN']
y_pred = md_Tree.predict(x_test)
arg_test = {'y_true':y_test, 'y_pred':y_pred}
df.info()
# gioi_tinh, huyet_ap, tim mach, cong viec, noi o, hut thuoc, luong duong, bmi, tuoi
print(md_Tree.predict([[1,1,1,3,3,1,2,2,2]]))
print(confusion_matrix(**arg_test))
print(classification_report(**arg_test))


matrix = confusion_matrix(**arg_test)
acc_no = matrix[0][0] / (matrix[0][0] + (matrix[0][1])) *100
acc_yes = matrix[1][1] / (matrix[1][0] + (matrix[1][1])) *100
percents_N = [acc_no, 100-acc_no]
percents_Y = [acc_yes, 100-acc_yes]
stroke = ["Đúng", "Sai"]
colors = ['red', 'green']
explode = [0, 0]

# plt.pie(percents_Y,
#         # colors=colors,
#         labels=stroke,
#         autopct='%1.2f%%',
#         wedgeprops={'edgecolor': 'red', 'linewidth': 1.5},
#         textprops={'fontsize': 25},
#         explode=explode)
# plt.tight_layout()
# plt.title('Decision Tree: Biểu đồ tỉ lệ đúng của phân lớp YES', fontsize=30, )
# plt.legend(['Đúng','Sai'], loc='lower right', fontsize=30)


plt.pie(percents_N,
        # colors=colors,
        labels=stroke,
        autopct='%1.2f%%',
        wedgeprops={'edgecolor': 'red', 'linewidth': 1.5},
        textprops={'fontsize': 25},
        explode=explode)
plt.tight_layout()
plt.title('Decision Tree: Biểu đồ tỉ lệ đúng của phân lớp No', fontsize=30, )
# plt.legend(['Đúng','Sai'], loc='lower right', fontsize=30)
plt.show()