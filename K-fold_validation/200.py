import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
import catboost as cat
import joblib
from sklearn.metrics import mean_absolute_error
import csv

# 导入训练数据
data = np.loadtxt("../data/feature_encoder_200+material.csv", delimiter=',')
label = np.log10(np.loadtxt("../data/y_data.csv", delimiter=','))

# print(traindata.shape, trainlabel.shape)

kf = KFold(n_splits=5, shuffle=False)
mae = []
index = 1

for train, test in kf.split(data):
    train_data = data[train]
    train_label = label[train]
    test_data = data[test]
    test_label = label[test]

    model = cat.CatBoostRegressor(iterations=1500, learning_rate=0.05, max_depth=8, subsample=0.7)
    model.fit(train_data, train_label)

    predict_label = model.predict(test_data)
    # print(type(test_label))
    # print(type(predict_label))
    # mae.append(mean_absolute_error(test_label, predict_label))

    predict_label = predict_label.tolist()
    test_label = test_label.tolist()
    csvFile = open("./cal-pre-data/200+material-"+str(index)+".csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(test_label)  # 数据写入文件中zz
    writer.writerow(predict_label)  # 数据写入文件中zz
    csvFile.close()

    index += 1
# grid = GridSearchCV(cal-pre-data, param_dist, scoring='neg_mean_absolute_error', cv=kf, verbose=0, n_jobs=1)

# print(sum(mae)/len(mae))


# joblib.dump(best_estimator_, './catboost_models/vae_catboost.pkl')
