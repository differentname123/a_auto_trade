import baostock as bs
import pandas as pd
import os
import time

# 1. 配置股票池（BaoStock 需要区分 sh 和 sz 前缀）
# 规则：60打头是上交所(sh)，00或30打头是深交所(sz)
stocks = {
    "贵州茅台": "sh.600519",
    "宁德时代": "sz.300750",
    "比亚迪": "sz.002594",
    "工业富联": "sh.601138",
    "中际旭创": "sz.300308",
    "迈瑞医疗": "sz.300760",
    "紫金矿业": "sh.601899",
    "招商银行": "sh.600036"
}

output_dir = "k_line"
start_date = "2020-01-01"
end_date = "2026-12-31"

# 2. 创建保存目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")

# 3. 登录 BaoStock 系统 (必须调用)
lg = bs.login()
if lg.error_code != '0':
    print(f"BaoStock 登录失败: {lg.error_msg}")
else:
    print("BaoStock 登录成功！开始拉取数据...\n")

    # 4. 循环拉取数据
    for name, code in stocks.items():
        try:
            print(f"正在拉取 {name} ({code}) 的1小时K线数据...")

            # frequency="60" 代表 60分钟
            # adjustflag="2" 代表 前复权 (1是后复权，3是不复权)
            # fields: 分钟线支持的字段
            rs = bs.query_history_k_data_plus(
                code,
                fields="date,time,code,open,high,low,close,volume,amount,adjustflag",
                start_date=start_date,
                end_date=end_date,
                frequency="60",
                adjustflag="2"
            )

            if rs.error_code == '0':
                # 将返回的数据集装换为 DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())

                df = pd.DataFrame(data_list, columns=rs.fields)

                if not df.empty:
                    # 提取纯数字代码用于文件名
                    pure_code = code.split('.')[1]
                    file_path = os.path.join(output_dir, f"{pure_code}_{name}_1h.csv")
                    df.to_csv(file_path, index=False, encoding='utf_8_sig')
                    print(f"  -> 成功保存至: {file_path}")
                else:
                    print(f"  -> 警告: {name} 数据为空 (可能该时间段无数据)")
            else:
                print(f"  -> 获取 {name} 失败: {rs.error_msg}")

            # 适度休眠，避免对服务器造成压力
            time.sleep(1)

        except Exception as e:
            print(f"拉取 {name} 发生异常，错误信息: {e}")

# 5. 登出 BaoStock 系统
bs.logout()
print("\n所有任务已完成！系统已登出。")