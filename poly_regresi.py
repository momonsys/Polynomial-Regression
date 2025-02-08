import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#pembuatan Dataset dengan numpy
x = np.arange(1,16,1)
y= [12,13,15,14,12,15,16,18,17,15,20,22,21,22,23]

line = np.linspace(1,15,100)

#pemanggilan model dari numpy
model = np.poly1d(np.polyfit(x,y,8))

#evaluasi atau mengukur ketepatan model dalam memprediksi dengan R2 score
score = r2_score(y,model(x))
print(score)

#visualisasi grafik dataset dan hasil prediksi model dengan matplotlib
plt.scatter(x,y)
plt.plot(line,model(line))
plt.show()

