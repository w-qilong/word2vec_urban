import psycopg2
import pandas as pd

conn = psycopg2.connect(database="data", user="postgres",
                        password="123", host="127.0.0.1", port="5432")

# 获取所有待查询POI及其编号
firstCursor = conn.cursor()
firstCursor.execute('select id,third from public.poi_fiveth;')
allThird = firstCursor.fetchall()
print(allThird[:5])
print('All POI conut: %s' % len(allThird))

center_context_list = list()

count = 0
for item in allThird:
    count += 1
    ## 建立游标，用来执行数据库操作
    cursor = conn.cursor()
    ## 执行SQL命令
    cursor.execute("""
        select third from public.poi_fiveth order by public.poi_fiveth.geom <-> 
        (select geom from public.poi_fiveth where id=%s) limit 5;
        """ % item[0])
    ## 获取SELECT返回的元组
    rows = cursor.fetchall()
    for row in rows[1:]:  # 获取的所有结果中，包含了当前所查询的POI，从第二个元素开始是最近的POI
        center_context_list.append([item[1], row[0]])  # 添加[center,context]
    if count % 5000 == 0:
        print(count)

# 输出最终结果
outDf = pd.DataFrame(center_context_list, columns=['center', 'context'])
outDf.to_csv('K_nearest_result.csv', index=False)
print('save finished!')
