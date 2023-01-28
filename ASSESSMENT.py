import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


method = ["k-NN", "Logistic regression", "Random recommendation", "Best recommendation"]

rate_0 = [70, 75, 25, 25]
rate_1 = [72, 65, 50, 0]

plt.bar([0.8, 1.8, 2.8, 3.8], rate_0, tick_label=method, width=0.4, alpha=0.8, label="Multiclass Classification")
plt.bar([1.2, 2.2, 3.2, 4.2], rate_1, width=0.4, alpha=0.8, label="Binary Classification")

plt.xticks(np.arange(1, 5, 1))

x_0 = [0.8, 1.8, 2.8, 3.8]
y_0 = np.array(rate_0)

for a, b in zip(x_0, y_0):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=15)

x_1 = [1.2, 2.2, 3.2, 4.2]
y_1 = np.array(rate_1)

for a, b in zip(x_1, y_1):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=15)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.ylim(0, 100)

plt.legend()
plt.show()


# MULTICLASS_LogisticRegression = [0.25, 0.25, 0.25, 0.3, 0.45, 0.65, 0.57, 0.57, 0.75, 0.57]
# BINARY_LogisticRegression = [0.5, 0.5, 0.5, 0.5, 0.65, 0.6, 0.58, 0.58, 0.58, 0.58]
#
# C = range(-5, 5)
#
# plt.plot(C, MULTICLASS_LogisticRegression, label="MULTICLASS")
# plt.plot(C, BINARY_LogisticRegression, label="BINARY")
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.xticks(C)
# plt.xlabel("lg(C)")
# plt.ylabel("准确率")
# plt.legend()
# plt.show()


# MULTICLASS_KNeighborsClassifier = [0.6, 0.6, 0.7, 0.67, 0.67, 0.67, 0.5, 0.5, 0.5, 0.5]
# BINARY_KNeighborsClassifier = [0.62, 0.62, 0.65, 0.65, 0.65, 0.70, 0.72, 0.70, 0.70, 0.65]
#
# K = range(1, 11)
#
# plt.plot(K, MULTICLASS_KNeighborsClassifier, label="MULTICLASS")
# plt.plot(K, BINARY_KNeighborsClassifier, label="BINARY")
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.xticks(K)
# plt.xlabel("n_neighbors")
# plt.ylabel("准确率")
# plt.legend()
# plt.show()
