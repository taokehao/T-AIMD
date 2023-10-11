import joblib
import numpy as np

# model = joblib.load('../model/catboost_model.pkl')
model = joblib.load('../draw_picture/catboost_models/InitialData_catboost.pkl')
print(model)
test_data = np.loadtxt("../test_data/tinghua/test_data.csv", delimiter=',')
test_label = np.log10(np.loadtxt("../test_data/tinghua/test_label.csv", delimiter=','))

result = model.predict(test_data)
print("真实值")
for i in test_label:
    print(i)
print("预测值")
for i in result:
    print(i)
print("")
loss_list = []
for i in range(len(result)):
    loss = abs(result[i] - test_label[i])
    # if 0.77 < loss < 1.9 or 0 < loss < 0.76:
    #     loss_list.append(loss)
        # loss_all += loss
    print(loss)
    loss_list.append(loss)
print("")

print("")
print(len(loss_list))
print('平均loss: ', sum(loss_list) / len(loss_list))
