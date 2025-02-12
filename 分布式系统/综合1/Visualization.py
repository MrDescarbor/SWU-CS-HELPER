import pandas as pd
from pyecharts.charts import Pie, Bar
from pyecharts import options as opts


# 读取CSV数据
file_path = "user_log.csv"
data = pd.read_csv(file_path)

# 检查数据
print(data.info())
print(data.head())




# 用户行为分布
action_counts = data['action'].value_counts()

# 饼图
pie = (
    Pie()
    .add(
        "",
        [list(z) for z in zip(action_counts.index.tolist(), action_counts.tolist())],
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="用户行为分布"))
)
pie.render("用户行为分布.html")


from pyecharts.charts import Map

# 按省份统计交易量
province_counts = data['province'].value_counts()

# 地图可视化
map_chart = (
    Map()
    .add(
        "交易量",
        [list(z) for z in zip(province_counts.index.tolist(), province_counts.tolist())],
        "china",
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="各省交易量分布"),
        visualmap_opts=opts.VisualMapOpts(max_=province_counts.max()),
    )
)
map_chart.render("各省交易量分布.html")


from pyecharts.charts import Line

# 按月份统计交易量
monthly_counts = data.groupby('month').size()

# 折线图
line = (
    Line()
    .add_xaxis(monthly_counts.index.tolist())
    .add_yaxis("交易量", monthly_counts.tolist())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="月度交易量趋势"),
        xaxis_opts=opts.AxisOpts(name="月份"),
        yaxis_opts=opts.AxisOpts(name="交易量"),
    )
)
line.render("月度交易量趋势.html")


from pyecharts.charts import WordCloud

# 品牌统计
brand_counts = data['brand_id'].value_counts().head(20)  # 取前20个品牌

# 柱状图
bar = (
    Bar()
    .add_xaxis(brand_counts.index.astype(str).tolist())
    .add_yaxis("品牌交易量", brand_counts.tolist())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="热门品牌交易量"),
        xaxis_opts=opts.AxisOpts(name="品牌ID"),
        yaxis_opts=opts.AxisOpts(name="交易量"),
    )
)
bar.render("热门品牌交易量.html")

# 词云
wordcloud = (
    WordCloud()
    .add(
        "",
        [list(z) for z in zip(brand_counts.index.astype(str), brand_counts)],
        word_size_range=[20, 100],
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="品牌词云"))
)
wordcloud.render("品牌词云.html")


# 年龄分布
age_counts = data['age_range'].value_counts()

# 条形图
age_bar = (
    Bar()
    .add_xaxis(age_counts.index.astype(str).tolist())
    .add_yaxis("用户数", age_counts.tolist())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="年龄段分布"),
        xaxis_opts=opts.AxisOpts(name="年龄段"),
        yaxis_opts=opts.AxisOpts(name="用户数"),
    )
)
age_bar.render("年龄段分布.html")

# 性别分布
gender_counts = data['gender'].value_counts()

# 饼图
gender_pie = (
    Pie()
    .add(
        "",
        [list(z) for z in zip(["男性", "女性"], gender_counts.tolist())],
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="性别分布"))
)
gender_pie.render("性别分布.html")
