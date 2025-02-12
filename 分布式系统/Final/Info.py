from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
import math
import os
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

# spark.conf.set("spark.local.dir", "/home/hadoop/桌面/Final/")
# spark.conf.set("spark.executor.memory", "24g")
# spark.conf.set("spark.memory.storageFraction", "1")
# spark.conf.set("spark.driver.memory", "24g")
# spark.conf.set("spark.memory.offHeap.enabled", "true")
# spark.conf.set("spark.memory.offHeap.size", "16g")
# spark.conf.set("spark.sql.shuffle.partitions", "0")
# spark.conf.set("spark.shuffle.compress", "false")




data_path = "file:///home/hadoop/桌面/Final/1975.csv"
raw_df = spark.read.csv(data_path, header=False, inferSchema=True)


columns = ["ID", "YEAR/MONTH/DAY", "ELEMENT", "DATA VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "OBS-TIME"]
df_with_header = raw_df.toDF(*columns)
# df_with_header.show()


# # 去掉缺失值处理
# df_no_missing = df_with_header.na.drop()  # 删除包含缺失值的行
#
# # 或者你也可以选择用特定值填充缺失值
# df_filled = df_with_header.na.fill({
#     'DATA VALUE': 0,  # 为数值字段填充 0
#     'M-FLAG': 'N',    # 为字符字段填充 'N'
#     'Q-FLAG': 'N',
#     'S-FLAG': 'N'
# })
#
# from pyspark.sql.functions import when
#
# # 筛选异常值
# df_invalid = df_with_header.filter((df_with_header["DATA VALUE"] < -200) | (df_with_header["DATA VALUE"] > 300))
#
# # 查看异常值
# print("异常值记录：")
# df_invalid.show()
#
# # 删除包含异常值的行
# df_cleaned = df_with_header.filter((df_with_header["DATA VALUE"] >= -200) & (df_with_header["DATA VALUE"] <= 300))
#
# # 或者处理异常值，例如将异常值替换为固定值
# df_corrected = df_with_header.withColumn(
#     "DATA VALUE",
#     when((df_with_header["DATA VALUE"] < -200) | (df_with_header["DATA VALUE"] > 300), 0).otherwise(df_with_header["DATA VALUE"])
# )


'''
from pyspark.sql.functions import substring, when

# 假设 df_with_header 是当前 DataFrame，包含字段 YEAR/MONTH/DAY 和 OBS-TIME

# 1. 拆分 YEAR/MONTH/DAY 成 YEAR、MONTH 和 DAY
df_split_date = df_with_header.withColumn("YEAR", df_with_header["YEAR/MONTH/DAY"].substr(1, 4).cast("int")) \
                              .withColumn("MONTH", df_with_header["YEAR/MONTH/DAY"].substr(5, 2).cast("int")) \
                              .withColumn("DAY", df_with_header["YEAR/MONTH/DAY"].substr(7, 2).cast("int"))

# 2. 拆分 OBS-TIME 为 HOUR 和 MINUTE
df_split_time = df_split_date.withColumn("HOUR",
                                         when(df_split_date["OBS-TIME"] < 1000, df_split_date["OBS-TIME"].substr(1, 1).cast("int"))
                                         .otherwise(df_split_date["OBS-TIME"].substr(1, 2).cast("int"))
                                        ) \
                             .withColumn("MINUTE",
                                         when(df_split_date["OBS-TIME"] < 1000, df_split_date["OBS-TIME"].substr(2, 2).cast("int"))
                                         .otherwise(df_split_date["OBS-TIME"].substr(3, 2).cast("int"))
                                        )


from pyspark.sql import Row

# 计算总和和总数（用于计算均值）
sum_count = df_split_time.rdd.map(lambda row: (row["DATA VALUE"], 1)) \
                             .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
mean = sum_count[0] / sum_count[1]  # 均值

# 计算标准差
variance = df_split_time.rdd.map(lambda row: (row["DATA VALUE"] - mean) ** 2) \
                            .reduce(lambda x, y: x + y) / sum_count[1]
stddev = variance ** 0.5  # 标准差

# 计算最大值和最小值
min_max = df_split_time.rdd.map(lambda row: (row["DATA VALUE"], row["DATA VALUE"])) \
                           .reduce(lambda x, y: (min(x[0], y[0]), max(x[1], y[1])))
min_value = min_max[0]  # 最小值
max_value = min_max[1]  # 最大值

# 打印计算结果
print("Mean: {}, Standard Deviation: {}, Min: {}, Max: {}".format(mean, stddev, min_value, max_value))

# 添加标准化和归一化列
df_standardized = df_split_time.withColumn(
    "DATA VALUE (Standardized)",
    (df_split_time["DATA VALUE"] - mean) / stddev
)

df_normalized = df_standardized.withColumn(
    "DATA VALUE (Normalized)",
    (df_standardized["DATA VALUE"] - min_value) / (max_value - min_value)
)





df_split_time.show()


df_normalized.show()

'''
# total_rows = df_with_header.count()
# print("数据集总行数: {}".format(total_rows))

# 显示每个重要字段的不重复数据


df_with_header.select("ID").distinct().show()
df_with_header.select("YEAR/MONTH/DAY").distinct().show()
df_with_header.select("ELEMENT").distinct().show()
df_with_header.select("DATA VALUE").distinct().show()
df_with_header.select("M-FLAG").distinct().show()
df_with_header.select("Q-FLAG").distinct().show()
df_with_header.select("S-FLAG").distinct().show()
df_with_header.select("OBS-TIME").distinct().show()


element_count = df_with_header.select("ELEMENT").distinct().count()
data_value_count = df_with_header.select("DATA VALUE").distinct().count()
m_flag_count = df_with_header.select("M-FLAG").distinct().count()
q_flag_count = df_with_header.select("Q-FLAG").distinct().count()
s_flag_count = df_with_header.select("S-FLAG").distinct().count()
obs_time_count = df_with_header.select("OBS-TIME").distinct().count()
ID_count = df_with_header.select("ID").distinct().count()
YMD_count = df_with_header.select("YEAR/MONTH/DAY").distinct().count()

# 统计每个字段的频率
element_pd = df_with_header.groupBy("ELEMENT").count().toPandas()
data_value_pd = df_with_header.groupBy("DATA VALUE").count().toPandas()
m_flag_pd = df_with_header.groupBy("M-FLAG").count().toPandas()
q_flag_pd = df_with_header.groupBy("Q-FLAG").count().toPandas()
s_flag_pd = df_with_header.groupBy("S-FLAG").count().toPandas()
obs_time_pd = df_with_header.groupBy("OBS-TIME").count().toPandas()
ID_pd = df_with_header.groupBy("ID").count().toPandas()
YMD_pd = df_with_header.groupBy("YEAR/MONTH/DAY").count().toPandas()

# 导入必要的可视化工具
import matplotlib.pyplot as plt
import seaborn as sns

# 确保 PySpark 和 Pandas 已导入
from pyspark.sql import SparkSession
import pandas as pd

# 定义一个绘制频率柱状图的函数
def plot_frequency(data, column_name):
    """
    使用柱状图绘制字段频率的分布。
    :param data: Pandas DataFrame，包含字段和对应的频率
    :param column_name: 字段名，作为图表标题
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=data.iloc[:, 0], y=data["count"], palette="viridis")
    plt.title("Frequency Distribution of {}".format(column_name), fontsize=16)
    plt.xlabel(column_name, fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

# 假设 df_with_header 已经是 Spark DataFrame
# 示例：df_with_header = spark.read.csv("your_file.csv", header=True, inferSchema=True)

# 定义需要统计频率的字段列表
fields = ["ELEMENT", "DATA VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "OBS-TIME", "ID", "YEAR/MONTH/DAY"]

# 创建一个字典，用于存储每个字段的频率统计
frequency_tables = {}

# 遍历字段并统计频率
for field in fields:
    # groupBy 并统计 count，然后转换为 Pandas DataFrame
    frequency_tables[field] = df_with_header.groupBy(field).count().toPandas()

# 绘制每个字段的频率分布图
for field, freq_table in frequency_tables.items():
    print("Frequency Table for {}:".format(field))
    print(freq_table)
    print("-" * 50)

    # 限制绘图数量，防止分类过多
    if len(freq_table) > 20:
        freq_table = freq_table.nlargest(20, "count")  # 只绘制前20个
        print("Warning: {} has too many categories. Showing top 20.".format(field))

    # 绘图
    plot_frequency(freq_table, field)


from pyspark.sql import functions as F

# 假设 df_with_header 是我们已经加载并处理好的 DataFrame
'''

from pyspark.sql import functions as F

# 假设 df_with_header 是我们已经加载并处理好的 DataFrame

# 按 'ELEMENT' 和 'ID' 进行分组，计算每组的 'DATA VALUE' 的最大值、最小值和平均值
stats_df = df_with_header.groupBy('ELEMENT', 'ID').agg(
    F.avg('DATA VALUE').alias('AVG_DATA_VALUE'),
    F.max('DATA VALUE').alias('MAX_DATA_VALUE'),
    F.min('DATA VALUE').alias('MIN_DATA_VALUE')
)


# 显示结果
stats_df.show()
'''


'''
# 转换 DataFrame 为 RDD
rdd = df_with_header.rdd

# 筛选出与温度和降雨相关的数据
filtered_rdd = rdd.filter(lambda row: row["ELEMENT"] in ["TMAX", "TMIN", "PRCP"])

# 找到每种 ELEMENT 的最大值及日期
max_rdd = (
    filtered_rdd
    .map(lambda row: (row["ELEMENT"], (row["DATA VALUE"], row["YEAR/MONTH/DAY"])))
    .reduceByKey(lambda a, b: a if a[0] > b[0] else b)  # 按最大值比较
)

# 找到每种 ELEMENT 的最小值及日期
min_rdd = (
    filtered_rdd
    .map(lambda row: (row["ELEMENT"], (row["DATA VALUE"], row["YEAR/MONTH/DAY"])))
    .reduceByKey(lambda a, b: a if a[0] < b[0] else b)  # 按最小值比较
)

# 将结果收集到本地
max_results = max_rdd.collect()
min_results = min_rdd.collect()

# 打印结果
print("最大值及日期：")
for element, (value, date) in max_results:
    print("ELEMENT: {}, MAX_VALUE: {}, DATE: {}".format(element, value, date))

print("最小值及日期：")
for element, (value, date) in min_results:
    print("ELEMENT: {}, MAX_VALUE: {}, DATE: {}".format(element, value, date))
'''

# 读取站点文件，假设文件为固定列宽格式
stations_path = "file:///home/hadoop/桌面/Final/ghcnd-stations.txt"
stations_schema = ["id", "lat", "lng", "altitude", "city", "unknown1", "unknownid"]

# 定义解析函数
def parse_fixed_width(row):
    return (
        row[0:11].strip(),   # ID
        float(row[12:20].strip()),  # LATITUDE
        float(row[21:30].strip()),  # LONGITUDE
        float(row[31:37].strip()) if row[31:37].strip() else None,  # ELEVATION
        row[38:40].strip(),  # STATE
        row[41:71].strip(),  # NAME
        row[72:75].strip(),  # GSN FLAG
        row[76:79].strip()   # HCN/CRN FLAG
    )

# 加载文件并跳过 header
stations_rdd = spark.read.text(stations_path).rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])

# 应用解析函数
parsed_rdd = stations_rdd.map(lambda row: parse_fixed_width(row.value))

# 转换为 DataFrame
stations_df = parsed_rdd.toDF(["ID", "LATITUDE", "LONGITUDE", "ELEVATION", "STATE", "NAME", "GSN_FLAG", "HCN_CRN_FLAG"])

# 定义重庆的经纬度和筛选范围
chongqing_lat = 29.5647398
chongqing_lng = 106.5478767
delta = 1  # 容差范围，单位为度

# 筛选范围内的站点
chongqing_stations = stations_df.filter(
    (stations_df["LATITUDE"] >= chongqing_lat - delta) &
    (stations_df["LATITUDE"] <= chongqing_lat + delta) &
    (stations_df["LONGITUDE"] >= chongqing_lng - delta) &
    (stations_df["LONGITUDE"] <= chongqing_lng + delta)
)

chongqing_stations.show()


# 假设气象数据 DataFrame 为 weather_df，包含 ID 字段
chongqing_weather = chongqing_stations.join(df_with_header, on="ID", how="inner")

chongqing_weather.show(truncate=False)
'''

rainfall_df = chongqing_weather.filter("ELEMENT = 'PRCP'")

# 提取年份和月份，使用 selectExpr 提取子字符串
rainfall_df = rainfall_df.selectExpr(
    "*",
    "substring(`YEAR/MONTH/DAY`, 1, 4) as YEAR",
    "substring(`YEAR/MONTH/DAY`, 5, 2) as MONTH",
    "CAST(`DATA VALUE` AS FLOAT) as DATA_VALUE"
)

# 按年份和月份汇总降雨量
monthly_rainfall = rainfall_df.groupBy("YEAR", "MONTH") \
    .agg({"DATA_VALUE": "sum"}) \
    .withColumnRenamed("sum(DATA_VALUE)", "TOTAL_RAINFALL")

# 找到降雨量最大的月份
max_rainfall = monthly_rainfall.orderBy("TOTAL_RAINFALL", ascending=False).limit(1)

# 打印结果
print("每月降雨量：")
monthly_rainfall.show()

print("降雨量最多的月份：")
max_rainfall.show()
'''

'''
# 过滤温度数据 (假设温度相关的 ELEMENT 为 'TMAX', 'TMIN', 或 'TAVG')
temperature_df = df_with_header.filter("ELEMENT IN ('TMAX', 'TMIN', 'TAVG')")

# 提取年份、月份，并转换 DATA VALUE 为数值类型
temperature_df = temperature_df.selectExpr(
    "*",
    "substring(`YEAR/MONTH/DAY`, 1, 4) as YEAR",
    "substring(`YEAR/MONTH/DAY`, 5, 2) as MONTH",
    "CAST(`DATA VALUE` AS FLOAT) as DATA_VALUE"
)

# 筛选 12 月的数据
december_df = temperature_df.filter("MONTH = '12'")

# 按照 ELEMENT 分类，并统计 12 月份的平均温度、最低温度和最高温度
december_stats = december_df.groupBy("ELEMENT") \
    .agg(
        {"DATA_VALUE": "avg", "DATA_VALUE": "min", "DATA_VALUE": "max"}
    ) \
    .withColumnRenamed("avg(DATA_VALUE)", "AVERAGE_TEMPERATURE") \
    .withColumnRenamed("min(DATA_VALUE)", "MIN_TEMPERATURE") \
    .withColumnRenamed("max(DATA_VALUE)", "MAX_TEMPERATURE")

# 打印结果
print("12 月份的温度统计信息：")
december_stats.show()


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

# # 将模型预测结果保存为CSV文件
# temp_predictions.write.parquet("file:///home/hadoop/桌面/Final/temperature_predictions.parquet")
# rain_predictions.write.parquet("file:///home/hadoop/桌面/Final/rainfall_predictions.parquet")
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 创建月份（1-12）作为横轴
# months = np.arange(1, 13)
#
# # 真实值（取自原始数据）
# true_values = temp_test.select("DATA_VALUE").toPandas()["DATA_VALUE"]
#
# # 预测值（模型输出）
# predicted_values = temp_predictions.select("prediction").toPandas()["prediction"]
#
# # 可视化
# plt.figure(figsize=(10, 6))
# plt.plot(months, predicted_values, color='red', label='预测值', linestyle='--')
# plt.xlabel("月份")
# plt.ylabel("温度")
# plt.title("真实值与预测值对比 - 温度")
# plt.legend()
# plt.grid()
# plt.show()
#
import matplotlib.pyplot as plt
import numpy as np

# 创建月份（1-12）作为横轴
months = np.arange(1, 13)

# 真实值（取自原始数据）
true_values = temp_test.select("DATA_VALUE").toPandas()["DATA_VALUE"]

# 预测值（模型输出）
predicted_values = temp_predictions.select("prediction").toPandas()["prediction"]

# 可视化
# plt.figure(figsize=(10, 6))
# plt.scatter(months, true_values, color='blue', label='真实值', alpha=0.7)
# plt.plot(months, predicted_values, color='red', label='预测值', linestyle='--')
# plt.xlabel("Month")
# plt.ylabel("Temperture")
# plt.title("Difference")
# plt.legend()
# plt.grid()
# plt.show()

# 生成周期性特征的可视化数据
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


# 如果需要将结果保存成文件（例如 CSV），可以使用如下命令：
# stats_df.write.csv('/path/to/output.csv', header=True)


# 如果需要将结果保存成文件（例如 CSV），可以使用如下命令：
# avg_df.write.csv('/path/to/output.csv', header=True)



# output_path = "file:///home/hadoop/桌面/Final/df_normalized.csv"
# df_normalized.write.csv(output_path, header=True)
'''

spark.stop()