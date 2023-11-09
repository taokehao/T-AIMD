import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
import catboost as cat
import joblib

# 导入训练数据
traindata = np.loadtxt("../data/feature_encoder_vae+material.csv", delimiter=',')
trainlabel = np.log10(np.loadtxt("../data/y_data.csv", delimiter=','))

print(traindata.shape, trainlabel.shape)

# 分类器使用 xgboost
model = cat.CatBoostRegressor(iterations=1500)

# 设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
    # 'iterations': range(1000, 1500, 100),
    # 'max_depth': range(5, 11, 1),
    # 'max_depth': range(8, 9, 1),
    'learning_rate': np.linspace(0.01, 0.05, 2),
    'subsample': np.linspace(0.7, 0.8, 2),

    # iterations=1500

    # 'iterations': range(1000, 1600, 100),
    # 'max_depth': range(5, 11, 1),
    # 'learning_rate': np.linspace(0.01, 0.1, 5),
    # 'subsample': np.linspace(0.7, 0.9, 2),
}

# RandomizedSearchCV参数说明，clf1设置训练的学习器
# param_dist字典类型，放入参数搜索范围
# scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
# n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
# n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
kf = KFold(n_splits=5, shuffle=False)
grid = GridSearchCV(model, param_dist, scoring='neg_mean_absolute_error', cv=kf, verbose=0, n_jobs=1)

# 在训练集上训练
grid.fit(traindata, np.ravel(trainlabel))
# 返回最优的参数
best_params_ = grid.best_params_
print("best_params_:")
print(best_params_)
# 输出最优训练器的精度
print("best_score_:")
print(grid.best_score_)
best_estimator_ = grid.best_estimator_
print("best_estimator_:")
print(best_estimator_)
joblib.dump(best_estimator_, '../model/catboost_model.pkl')

