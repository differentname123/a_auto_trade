# a_auto_trade
a股量化交易:自动交易
##
这个模块是真正执行股票交易的执行者，主要包括查询账户信息，及买入买入
主要参考：https://github.com/match5/thsauto
改动：
    1.修复了新版PIL和ddddocr不兼容的问题，主要使用pytesseract进行验证码识别
    2.修复了复制数据时存在'c' 在 'ctrl'前按下导致的问题