# a_auto_trade
a股量化交易:自动交易
##
这个模块是真正股票信息的获取者，时分数据都能够获取到
主要参考：https://github.com/akfamily/akshare

改动：
    1.修复了新版PIL和ddddocr不兼容的问题，主要使用pytesseract进行验证码识别
    2.修复了复制数据时存在'c' 在 'ctrl'前按下导致的问题