# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023-12-02 9:46
:last_date:
    2023-12-02 9:46
:description:

"""
# 股票代码，例如“600519”代表贵州茅台
import json

import pandas as pd
import requests

stock_code = "600519"

# 构建新浪财经API的URL
url = f"https://finance.sina.com.cn/realstock/company/sz{stock_code}/hisdata/klc_kl.js"

# 发送请求
response = requests.get(url)
data = response.text

# 解析数据
# 新浪财经API返回的数据可能需要一些特定的处理来提取有效信息
# 这里只是一个示例，具体的处理方式取决于API返回的数据格式

# 以下代码仅为示例，可能需要根据实际返回的数据格式进行调整
data = data.split("=")[1]
data = json.loads(data)

# 将数据转换为DataFrame
df = pd.DataFrame(data['data'])
print(df)

# 分析数据以确定涨