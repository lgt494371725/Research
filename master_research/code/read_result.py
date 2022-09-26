import pstats
from pstats import SortKey


p = pstats.Stats("result.cprofile")
# 按照运行时间和函数名进行排序
# p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)
p.strip_dirs().sort_stats("tottime", "name").print_stats(10)
# 按照函数名排序，只打印前 3 行函数的信息, 参数还可为小数, 表示前百分之几的函数信息
# 如果想知道有哪些函数调用了 ccc
p.print_callers(0.5, "ccc")
# 查看 ccc() 函数中调用了哪些函数
p.print_callees("ccc")