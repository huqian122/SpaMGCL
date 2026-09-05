import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 18,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False, # 处理负号，即-号
            'font.weight': 'bold'
         }
rcParams.update(config)

# 读取Excel文件
file_path = 'results.xlsx'
df = pd.read_excel(file_path)

# 从Excel文件中提取数据
epochs = df['Epochs']
loss = df['Loss']
acc = df['ACC']
nmi = df['NMI']

# 创建图表
fig, ax1 = plt.subplots()

# 设置主坐标轴（左Y轴）
ax1.set_xlabel('Epoch')
ax1.set_ylabel('ACC and NMI', color='black')
acc_line, = ax1.plot(epochs, acc, color='tab:cyan', label='ACC')
nmi_line, = ax1.plot(epochs, nmi, color='tab:green', label='NMI')
ax1.tick_params(axis='y', labelcolor='black')

# 创建次坐标轴（右Y轴）
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='black')
loss_line, = ax2.plot(epochs, loss, 'r--', label='Loss')
ax2.tick_params(axis='y', labelcolor='black')

# 合并图例
lines = [acc_line, nmi_line, loss_line]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')

# 添加图例
fig.tight_layout()

# 添加虚线网络
ax1.grid(linestyle='dashed')
ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
# # 标题
# plt.title('Loss vs Performance')



# 显示图表
plt.show()
