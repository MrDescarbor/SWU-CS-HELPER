# 生成周期性特征的可视化数据
import matplotlib.pyplot as plt
import numpy as np
month_values = np.arange(1, 13)
month_sin = np.sin(2 * np.pi * month_values / 12)
month_cos = np.cos(2 * np.pi * month_values / 12)

# 可视化周期特征
plt.figure(figsize=(10, 6))
plt.plot(month_values, month_sin, label="MONTH_SIN", color='orange')
plt.plot(month_values, month_cos, label="MONTH_COS", color='green')
plt.xlabel("Month")
plt.ylabel("Value")
plt.title("Periodic (MONTH_SIN and MONTH_COS)")
plt.legend()
plt.grid()
plt.show()