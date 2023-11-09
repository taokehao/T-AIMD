import joblib
import numpy as np
import csv

model = joblib.load('../model/catboost_model.pkl')
print(model)
test_data = np.loadtxt("../test_data/jisuan/test_feature_encoder+material.csv", delimiter=',')
test_label = np.log10(np.loadtxt("../test_data/jisuan/test_label.csv", delimiter=','))

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
    csvFile = open("./vae_material_catboost.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(newRow)  # 数据写入文件中zz
