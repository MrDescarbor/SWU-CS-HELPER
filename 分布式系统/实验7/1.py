from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# 创建 SparkSession
spark = SparkSession.builder.appName("RandomForestMultiClassification").getOrCreate()

# 加载数据集 (假设数据存储在 CSV 文件中)
data = spark.read.csv("file:///home/hadoop/iris.txt", header=True, inferSchema=True)

# 定义函数将分类标签转换为数字
def change_class(label):
    if label == 'Iris-setosa':
        return 0
    elif label == 'Iris-versicolor':
        return 1
    else:
        return 2

# 使用 UDF 将分类标签转换为数字
change_class_udf = udf(change_class, IntegerType())
data = data.withColumn("TrainingClass", change_class_udf(data["TrainingClass"]))

# 使用 VectorAssembler 将特征列组合成一个名为 'features' 的向量列
assembler = VectorAssembler(inputCols=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'], outputCol="features")
assembled_data = assembler.transform(data)

# 划分训练集和测试集
train_data, test_data = assembled_data.randomSplit([0.7, 0.3])

# 创建 RandomForest 分类器
rf = RandomForestClassifier(labelCol="TrainingClass", featuresCol="features", numTrees=10)

# 训练模型
model = rf.fit(train_data)

# 进行预测
predictions = model.transform(test_data)

# 模型评估
evaluator = MulticlassClassificationEvaluator(labelCol="TrainingClass", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("模型的准确率为: {}".format(accuracy))
