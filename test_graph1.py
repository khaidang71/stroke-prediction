from turtle import color
import matplotlib.pyplot as plt
 
# Từ điển chứa danh sách các món ăn và số người thích
Food_counts = {'Fish': 10, 'Meat': 20, 'Egg': 50, 'Milk': 30, 'Crab': 40}
 
# Chuẩn bị dữ liệu:
# Sort Số lượng người thích (giảm dần)
Counts = sorted(Food_counts.values(), reverse=True)
# Sort các món ăn dựa vào số lượng người thích (giảm dần)
Foods = sorted(Food_counts, key=Food_counts.__getitem__, reverse=True)
 
# Chỉ số các món ăn
ind_Foods= range(len(Food_counts))
 
# Vẽ biểu đồ cột
plt.bar(ind_Foods, Counts, align='center')
# plt.xticks(ind_Foods, Foods)
 
# Label x, y axit
plt.xlabel('Foods')
plt.ylabel('Count (like)')
# Label title of bar char
plt.title('FOOD WHICH PEOPLE LIKE')
 
# Thêm các giá trị trên mỗi cột
for x, y in zip(ind_Foods, Counts):
    plt.text(x+0.02, y+0.05,f"{'%d'% y}%"  , ha='center', va= 'bottom', color='green')
 
# Tăng trục y thêm 20 đơn vị
plt.ylim(0, Counts[0] + 20)
 
# Cuối cùng là show kết quả!!!
plt.show()


## Implementation of matplotlib function
# import numpy as np
# from matplotlib.axis import Axis
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
	
# np.random.seed(19680801)

# fig, ax = plt.subplots()
# # ax.plot(100*np.random.rand(20))

# formatter = ticker.FormatStrFormatter('%2.2f')
# Axis.set_major_formatter(ax.yaxis, formatter)

# for tick in ax.yaxis.get_major_ticks():
# 	tick.label1.set_color('green')

# plt.title("Matplotlib.axis.Axis.set_major_formatter()\n\
# Function Example", fontsize = 12, fontweight ='bold')
# line, = ax.plot([1, 2, 3])
# line.set_label('Label via method')
# ax.legend()
# plt.show()
