from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
import math
import os
import pandas as pd
os.environ['PYSPARK_DRIVER_TEMPLATES_DIR'] = '/home/hadoop/桌面/Final/'
os.environ['PYSPARK_EXECUTOR_TEMPLATES_DIR'] = '/home/hadoop/桌面/Final/'
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType



spark = SparkSession.builder.config('spark.local.dir', '/home/hadoop/桌面/Final/').getOrCreate()


data_path2 = "file:///home/hadoop/桌面/Final/chongqing_weather.csv/Chongqing2.csv"
data_path = "file:///home/hadoop/桌面/Final/1975.csv"
chongqing_weather = spark.read.csv(data_path2, header=True)
raw_df = spark.read.csv(data_path, header=False, inferSchema=True)
columns = ["ID", "YEAR/MONTH/DAY", "ELEMENT", "DATA VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "OBS-TIME"]
all_weather = raw_df.toDF(*columns)

# 选择所需字段：日期、降雨量、温度（最高温或平均温）
data = chongqing_weather.selectExpr(
    "CAST(substring(`YEAR/MONTH/DAY`, 1, 4) AS INT) as YEAR",
    "CAST(substring(`YEAR/MONTH/DAY`, 5, 2) AS INT) as MONTH",
    "CAST(substring(`YEAR/MONTH/DAY`, 7, 2) AS INT) as DAY",
    "CAST(`DATA VALUE` AS FLOAT) as DATA_VALUE",
    "ELEMENT"
)

import math
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import concat_ws

'''
data = chongqing_weather.toPandas()
data["DATA VALUE"] = data["DATA VALUE"].astype(int)




# 转换为日期格式
data['DATE'] = pd.to_datetime(data['YEAR/MONTH/DAY'], format='%Y%m%d')




data = data.sort_values(by="DATE")  # 按日期排序

# 按元素分组
tmax = data[data["ELEMENT"] == "TMAX"]
tmin = data[data["ELEMENT"] == "TMIN"]
prcp = data[data["ELEMENT"] == "PRCP"]
tavg = data[data["ELEMENT"] == "TAVG"]

# 绘图
plt.figure(figsize=(12, 6))

# Tmax
plt.plot(tmax["DATE"], tmax["DATA VALUE"], label="TMAX (TMAX)", color="red", marker="o")

# Tmin
plt.plot(tmin["DATE"], tmin["DATA VALUE"], label="TMIN (TMIN)", color="blue", marker="s")

# PRCP
plt.bar(prcp["DATE"], prcp["DATA VALUE"], label="PRCP (PRCP)", color="cyan", alpha=0.5, width=5)

# TAVG
plt.plot(tavg["DATE"], tavg["DATA VALUE"], label="TAVG (TAVG)", color="green", linestyle="--")




plt.title("Visualize of Weather", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)


# 展示图表
plt.show()
'''









# 定义自定义 UDF
def sin_udf(value):
    return math.sin(2 * math.pi * value / 12)

def cos_udf(value):
    return math.cos(2 * math.pi * value / 12)

sin_udf = udf(sin_udf, DoubleType())
cos_udf = udf(cos_udf, DoubleType())

# 添加周期性特征
def add_periodic_features(df):
    df = df.withColumn("MONTH_SIN", sin_udf(df["MONTH"].cast(DoubleType())))
    df = df.withColumn("MONTH_COS", cos_udf(df["MONTH"].cast(DoubleType())))
    return df

# 假设 temp_df 和 rain_df 是包含温度和降雨数据的 DataFrame
temp_df = add_periodic_features(data.filter("ELEMENT IN ('TMAX', 'TMIN', 'TAVG')"))
rain_df = add_periodic_features(data.filter("ELEMENT = 'PRCP'"))

# 构造特征向量
from pyspark.ml.feature import VectorAssembler

assembler_temp = VectorAssembler(inputCols=["MONTH_SIN", "MONTH_COS"], outputCol="features")
assembler_rain = VectorAssembler(inputCols=["MONTH_SIN", "MONTH_COS"], outputCol="features")

temp_data = assembler_temp.transform(temp_df).select("features", "DATA_VALUE")
rain_data = assembler_rain.transform(rain_df).select("features", "DATA_VALUE")

# 拆分训练集和测试集
temp_train, temp_test = temp_data.randomSplit([0.8, 0.2], seed=42)
rain_train, rain_test = rain_data.randomSplit([0.8, 0.2], seed=42)

# 初始化线性回归模型
from pyspark.ml.regression import LinearRegression

lr_temp = LinearRegression(featuresCol="features", labelCol="DATA_VALUE", maxIter=100)
lr_rain = LinearRegression(featuresCol="features", labelCol="DATA_VALUE", maxIter=100)

# 训练温度预测模型
temp_model = lr_temp.fit(temp_train)
rain_model = lr_rain.fit(rain_train)

# 对测试集进行预测
temp_predictions = temp_model.transform(temp_test)
rain_predictions = rain_model.transform(rain_test)

# 模型评估
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="DATA_VALUE", predictionCol="prediction", metricName="rmse")

temp_rmse = evaluator.evaluate(temp_predictions)
rain_rmse = evaluator.evaluate(rain_predictions)

print("Temperature Prediction RMSE: {}".format(temp_rmse))
print("Rainfall Prediction RMSE: {}".format(rain_rmse))


# 转换为 Pandas DataFrame
temp_predictions_pd = temp_predictions.select("DATA_VALUE", "prediction").toPandas()
rain_predictions_pd = rain_predictions.select("DATA_VALUE", "prediction").toPandas()

import matplotlib.pyplot as plt

# 绘制温度预测结果
plt.figure(figsize=(10, 6))
plt.scatter(temp_predictions_pd.index, temp_predictions_pd["DATA_VALUE"], color="blue", label="Actual Values", alpha=0.6)
plt.scatter(temp_predictions_pd.index, temp_predictions_pd["prediction"], color="red", label="Predicted Values", alpha=0.6)

plt.title("Temperature Predictions vs Actual Values")
plt.xlabel("Index")
plt.ylabel("Temperature (DATA_VALUE)")
plt.legend()
plt.grid(True)
plt.show()

# 绘制降雨预测结果
plt.figure(figsize=(10, 6))
plt.scatter(rain_predictions_pd.index, rain_predictions_pd["DATA_VALUE"], color="blue", label="Actual Values", alpha=0.6)
plt.scatter(rain_predictions_pd.index, rain_predictions_pd["prediction"], color="red", label="Predicted Values", alpha=0.6)

plt.title("Rainfall Predictions vs Actual Values")
plt.xlabel("Index")
plt.ylabel("Rainfall (DATA_VALUE)")
plt.legend()
plt.grid(True)
plt.show()

