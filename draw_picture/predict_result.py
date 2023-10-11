import joblib
import numpy as np
import csv

model = joblib.load('catboost_models/400_material_catboost.pkl')
print(model)
test_data = np.loadtxt("../data/feature_encoder_400+material.csv", delimiter=',')
test_label = np.log10(np.loadtxt("../data/y_data.csv", delimiter=','))

result = model.predict(test_data)
print(result)
print("")
print(test_label)
print("")

# for i in test_label:
#     newRow = [i]
#     csvFile = open("./y_log10.csv", 'a', newline='', encoding='utf-8')
#     writer = csv.writer(csvFile)
#     writer.writerow(newRow)  # 数据写入文件中zz
#     csvFile.close()
#
for i in result:
    newRow = [i]
    csvFile = open("./400_material_catboost.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(newRow)  # 数据写入文件中zz
    csvFile.close()
