import pymysql
import pandas as pd

# 打开数据库连接
try:
    connection = pymysql.connect(host='localhost', user='root', passwd='DAXI9999', port=3306)
    print('连接成功！')
except:
    print('something wrong!')

# 加载CSV文件
csv_file = 'PID_year.csv' # 更改为你的CSV文件路径
df = pd.read_csv(csv_file)

try:
    with connection.cursor() as cursor:
        for index, row in df.iterrows():
            pid = row[0]  # 假设第一列是条件
            year = row[1]  # 假设第二列是要插入的值

            # 构造SQL语句
            # 注意: 这里只是一个示例。根据你的具体需求来修改表名和列名。
            # 如果是插入操作，确保唯一键或主键不会导致冲突
            if pd.isnull(year):
                continue
            # print (pid, year)
            sql = f"UPDATE capat_labelling.Paintings SET year = {year} WHERE PID = '{pid}'"
            
            # # 执行SQL语句
            cursor.execute(sql)
        
        # 提交事务
        connection.commit()
        
finally:
    # 关闭连接
    connection.close()