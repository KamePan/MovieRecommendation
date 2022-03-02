# import pandas as pd
#
#
# print(pd.DataFrame([1, 2, 3, 4, 5, 6]))
# print("hello world")
# 脚本文件
# import numpy as np
# import main
# a = [1, 3, 4, 5, 2]
# print(a)
# print("hello world")

import requests
import numpy as np
import pandas as pd

def my_test(name, age):
    response = requests.get("http://www.baidu.com")
    print("url："+response.url)
    print("name: "+name)
    print("age: "+age)
    return "success"


my_test('hello', 'world')