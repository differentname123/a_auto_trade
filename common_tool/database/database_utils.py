# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/8 20:40
:last_date:
    2023/10/8 20:40
:description:
    
"""
import pandas as pd
from sqlalchemy import create_engine


def save_to_postgresql(df: pd.DataFrame, db_url: str, table_name: str = 'stock_data', schema_name: str = 'auto_trade'):
    # 修改 DataFrame 的列名以匹配数据库表的字段名
    df.columns = [
        'date', 'opening_price', 'closing_price', 'highest_price', 'lowest_price',
        'trading_volume', 'turnover', 'amplitude', 'price_change_ratio',
        'price_change_amount', 'turnover_rate'
    ]

    # 创建数据库连接
    engine = create_engine(db_url)

    try:
        # 将 DataFrame 写入数据库
        df.to_sql(table_name, engine, if_exists='replace', index=False, schema=schema_name)
        print(f'Data saved to table {schema_name}.{table_name}.')
    except Exception as e:
        print(f'An error occurred: {e}')