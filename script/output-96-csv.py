import numpy as np
import csv

total_data = np.loadtxt('../data/raw_data.csv', delimiter=',', dtype=float)
for i in total_data:
    newRow = []
    for j in range(10000):
        if j % 10 == 0:
            newRow.append(i[j])
    csvFile = open("../data/1000-data.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(newRow)  # 数据写入文件中zz
    csvFile.close()
