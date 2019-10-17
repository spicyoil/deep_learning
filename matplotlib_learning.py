import numpy as np


#plot
import matplotlib.pyplot as plt
# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)
# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x,y2,label="cos",linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sin&Cos")
plt.legend()#图示
#plt.show()

import matplotlib.pyplot as plt2  #imshow
from matplotlib.image import imread
img = imread('./deep_learning/example/dataset/lena.png')
plt2.imshow(img)
plt2.show()