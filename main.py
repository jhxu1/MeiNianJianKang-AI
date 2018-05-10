from model_rewrite.date_contract import read_data_and_contact
from model_rewrite.date_clean import clean_feature
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from math import log1p, pow


# 计算误差损失（评估分数）
def calc_logloss(true_df, pred_df):
    loss_sum = 0
    rows = true_df.shape[0]
    for col in true_df.columns:
        # 预测结果必须要>0,否则log函数会报错，导致最终提交结果没有分数
        true_df[col] = true_df[col].apply(lambda x: log1p(x))
        pred_df[col] = pred_df[col].apply(lambda x: log1p(x))
        true_df[col + 'new'] = pred_df[col] - true_df[col]
        true_df[col + 'new'] = true_df[col + 'new'].apply(lambda x: pow(x, 2))
        loss_item = (true_df[col + 'new'].sum()) / rows
        loss_sum += loss_item
        print('%s的loss：%f' % (col, loss_item))
    print('五项指标平均loss分数：', loss_sum / 5)

if __name__ == '__main__':
    train, test = read_data_and_contact()
    train_num = train.shape[0]
    all_set = pd.concat([train, test]).set_index('vid')     # 数据拼接
    all_set = clean_feature(all_set)                                  # 清洗数据

    # for col in all_set.columns:
    #     print(all_set[col].dtype)
    train_set = all_set.iloc[:train_num, :]                 # 重新获取训练数据
    train_set.to_csv('train_set.csv', encoding='gbk')
    print('train_set.shape\t', train_set.shape)
    test_set = all_set.iloc[train_num:, :]
    X_train, X_test, y_train, y_test = train_test_split(
        train_set.iloc[:, 5:],
        train_set.iloc[:, 0:5],
        test_size=0.3)  # 分割训练集
    y_train = y_train.astype('float') * 100
    y_pred_list = []
    for c in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        print(c)
        model = XGBClassifier(
            learning_rate=0.1,
            n_estimators=140,
            max_depth=3,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.6,
            colsample_bytree=0.6,
            scale_pos_weight=1,
            nthread=4,
            seed=27
        )
        model.fit(X_train, y_train[c])
        y_pred = model.predict(X_test)
        y_pred = np.ndarray.round(y_pred, 3) / 100
        y_pred_list.append(y_pred)
        # 最后要输出的结果
        test_pred = model.predict(test_set.iloc[:, 5:])
        test_set[c] = np.ndarray.round(test_pred, 3) / 100
    y_pred_list = np.transpose(y_pred_list)
    y_pred_res = pd.DataFrame(y_pred_list,
                              columns=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'],
                              index=y_test.index
                              )
    calc_logloss(y_test, y_pred_res)

    result = test_set.iloc[:, :5]
    result.to_csv('tmp.csv', header=False, encoding='utf-8')



