import pymysql
import json

# 数据库连接参数
db_params = {
    "host": "localhost",
    "user": "root",
    "password": "DAXI9999",
    "port": 3306,
}
dimensions = [
        "category",
        "cunFa",
        "dianFa",
        "sheSe",
        "dianJing"
      ]

def year_to_period(year, start_year=1665, period=5):
    """将年份转换为5年周期的表示形式。"""
    if year < start_year or year > 1705:
        return None
    period_index = (year - start_year) // period
    return f"{start_year + period_index*period}"

# 连接数据库
connection = pymysql.connect(**db_params)

try:
    for dimension in dimensions:
        with connection.cursor() as cursor:
            # 执行 SQL 查询
            query = f"""
            SELECT {dimension}, year, COUNT(*) as num
            FROM capat_labelling.Fragments, capat_labelling.Paintings
            WHERE fragments.PID = Paintings.PID
            and year is not null
            group by {dimension}, year
            """
            cursor.execute(query)
            result = cursor.fetchall()

            # 初始化数据结构
            data_structure = {
                dimension: [],  # 将填充不同的类别
                "data": {}
            }


            for row in result:
                category_str, year, count = row
                categories = category_str.split(';')  # 拆分类别字符串
                period = year_to_period(year)

                if period is not None:
                    for category in categories:
                        if category:
                            # 确保类别在列表中
                            if category not in data_structure[dimension]:
                                data_structure[dimension].append(category)
                                data_structure["data"][category] = {}

                            # 将数据添加到对应的类别和周期
                            if period not in data_structure["data"][category]:
                                data_structure["data"][category][period] = 0
                            data_structure["data"][category][period] += count  # 平均分配到每个类别


            # JSON 格式化输出，确保使用 UTF-8 编码
            with open(f'{dimension}_timeline.json', 'w', encoding='utf-8') as outfile:
                json.dump(data_structure, outfile, ensure_ascii=False, indent=4)
                print (f"File {dimension}_timeline.json has been generated.")

finally:
    connection.close()
