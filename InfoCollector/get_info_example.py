# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/6 11:43
:last_date:
    2023/10/6 11:43
:description:
    
"""
import akshare as ak

from common_tool.database.database_utils import save_to_postgresql

if __name__ == '__main__':

    # stock_zh_a_hist_df = ak.stock_hk_hist_min_em(symbol="01753", period="1", start_date="20231005", end_date='20231007', adjust="")
    # result = ak.stock_zh_a_spot_em()
    result = ak.stock_zh_a_hist(symbol="601002", period="daily", start_date="20230905", end_date='20231007', adjust="")
    save_to_postgresql(result,'postgresql://postgres:zxh111111@localhost:5432/auto_trade','stock_data')