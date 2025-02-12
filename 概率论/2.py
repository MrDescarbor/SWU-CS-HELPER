import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 示例数据
text_data = [
    "This is a positive review for a fantastic product.",
    "Do not buy this, it's a waste of money.",
    "Earn money from home with our easy work-from-home scheme!",
    "Check out the latest news on machine learning and artificial intelligence.",
    "Limited-time offer: Get 50% off on our exclusive products!",
    "You have won a lottery! Claim your prize now.",
    "Great experience with our customer service. Highly recommended.",
    "Meet local singles in your area tonight!",
    "Important update: Your account security needs attention. Click the link to verify.",
    "Upgrade your skills with our online learning platform."
]

labels = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]  # 1 for positive, 0 for negative

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.25, random_state=42)

# 使用CountVectorizer将文本数据转换为词袋模型
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 创建并拟合朴素贝叶斯分类器
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# 预测测试集
y_pred = classifier.predict(X_test_vectorized)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Spam"], yticklabels=["Normal", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
