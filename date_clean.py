import numpy as np
import re
from sklearn import preprocessing
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA

'''
清洗特征
'''


def clean_feature(df):
    useless_feature = ['0203', '0209', '0702', '0703',
                       '0705', '0706', '0709', '0726',
                       '0730', '0731', '3601', '1308',
                       '1316'
                       ]
    print("*** 清洗无用特征 ***")
    df = _clean_useless_feature(df, useless_feature)      # 清洗无用特征
    print("*** 清洗符号 ***")
    df = clean_label(df)
    print("*** 清洗缺失值过多的特征 ***")
    df = _clean_nan_feature(df=df, thresh=0.96)               # 清洗缺失值过多的特征
    df.to_csv('特征查找.csv', encoding='gbk')
    feature_label = _get_feature_label(df)                  # 获取每个特征的类别

    print("*** 手动清洗关键特征 ***")
    df = _clean_feature_artificial(df, feature_label)

    # 将特征写入文件
    f = open('feature_label.csv', 'w', encoding='gbk')
    for key, value in feature_label.items():
        f.write(key+','+value)
        f.write('\n')
    f.close()

    print("*** 填充缺失值 ***")
    df = _fill_miss_value(df, feature_label)
    df = _one_hot(df, feature_label)

    # 标准化
    # df.to_csv('test.csv', encoding='gbk')
    min_max_scaler = preprocessing.MinMaxScaler()
    df.iloc[:, 5:] = min_max_scaler.fit_transform(df.iloc[:, 5:])
    # PCA降维
    # pca = PCA(n_components=0.9)  # 保证降维后的数据保持90%的信息
    # pca.fit(df.iloc[:, 5:])
    # print('降维前规模', df.shape)
    # feat = pd.DataFrame(pca.transform(df.iloc[:, 5:]))
    # new_df = pd.concat([df.iloc[:, 0:5], feat], axis=1, join_axes=[df.iloc[:, 5:].index])
    # print('降维后规模', new_df.shape)
    return df




def _one_hot(df, feature_label):
    """
    对类型2特征进行one-hot编码
    :param df: 数据
    :param feature_label: 特征类别        
    :return: 处理过的数据
    """
    print('one-hot前特征数量:\t',df.shape[1])
    hot_dic = []
    for col in df.columns[5:]:
        if feature_label[col] == '5':
            hot_dic.append(col)
    df = pd.get_dummies(df, columns=hot_dic)
    print('one-hot后特征数量:\t', df.shape[1])
    return df


def _fill_miss_value(df, feature_label):
    """
    对df特征进行分类并填充缺失值
    :param df: 数据
    :param feature_label: 每个特征的类别
    :return: 填充后数据
    """
    # 删除类型3,4特征
    del_col = []
    for col in df.columns[5:]:
        label = feature_label[col]
        if feature_label[col] == '4' or feature_label[col] == '2':
            del_col.append(col)
        else:
            for n in range(df.shape[0]):
                x = str(df[col].values[n])
                if label == '1':
                    tmp = re.findall(r'\d+\.?\d*', x)
                    if len(tmp) >= 1:
                        df[col].values[n] = float(tmp[0])
                    else:
                        df[col].values[n] = np.nan
                elif label == '5' and (_if_float(str(x)) or str(x).isdigit()) and x is not np.nan:
                    df[col].values[n] = np.nan
    left_col = []
    for col in df.columns:
        if col not in del_col:
            left_col.append(col)
    df = df[left_col]
    # 对类型1和2特征进行填充缺失值
    for col in df.columns:
        if feature_label[col] == '1':
            # print(df.loc[:, col])
            df.loc[:, col] = df.loc[:, col].fillna(df.loc[:, col].mean())
            df.loc[:, col] = df.loc[:, col].astype('float64')
        if feature_label[col] == '5':
            df.loc[:, col] = df.loc[:, col].fillna('miss')
    # 对类型2的特征进行标准化处理
    # for f in df.columns[5:]:
    #     if df[f].dtype == 'object':
    #         lbl = preprocessing.LabelEncoder()
    #         lbl.fit(list(df[f].values))
    #         df.loc[:, f] = lbl.transform(list(df.loc[:, f].values))
    return df


def _if_float(s):
    """
    判断str是否为浮点数
    :param s: 输入字符串
    :return: 是否为字符串
    """
    m = re.match("\d+\.\d+", s)
    if m:
        return True
    else:
        return False


def _get_feature_label(df):
    """
    # 判断每种特征的类别
    # 1:纯数值型，无需处理．
    # 2.枚举型及其变种，如：＂阴性＂，＂未见异常＂等等，简单处理即可转成数值．
    # 3.简单混合型，如：＂>100次/分,窦性心动过速＂，此类型以数值为主．
    # 4.复杂混合型，如：＂甲状腺内见多个低回声结节，最大位于右叶约14mm×8mm，结节周边有血管环绕＂．这种情况可能是纯文字，
    # 可能是文字和数值混合，相对比较复杂，也很有趣，它包含了第二和第三两种情况．
    :param df: 待清洗数据
    :return: 每种特征的类型
    """
    feature_label = {}
    for col in df.columns:
        if col == '809001':
            ss = 'pass'
        dig_num = 0
        str_num = 0
        big_str_num = 0
        for v in df[col]:
            if v is not np.nan and v != 'nan' and v is not None:
                if isinstance(v, float) or str(v).isdigit() or _if_float(str(v)):
                    dig_num += 1
                elif len(str(v)) > 3:
                    big_str_num += 1
                else:
                    str_num += 1
        num_s = dig_num + str_num + big_str_num
        if num_s == 0:
            continue
        if dig_num / num_s > 0.2:
            feature_label[col] = '1'
        elif str_num != 0 and big_str_num * 1.0 / str_num < 0.4:
            feature_label[col] = '2'
        else:
            feature_label[col] = '4'
    count = Counter(feature_label.values())
    print("特征数量：\t", count)
    return feature_label


# def _clean_label(x):
#     """
#     清洗item x的符号
#     :param x: item
#     :return: 清洗过的item
#     """
#     x = str(x)
#     if '+' in x:  # 16.04++
#         i = x.index('+')
#         x = x[0:i]
#     if '>' in x:  # > 11.00
#         i = x.index('>')
#         x = x[i+1:]
#     if '<' in x:  # < 11.00
#         i = x.index('<')
#         x = x[i + 1:]
#     if len(str(x).split(sep='.')) > 2:  # 2.2.8
#         i = x.rindex('.')
#         x = x[0:i]+x[i+1:]
#     if '未做' in x or '未查' in x or '弃查' in x or ',' in x:
#         x = np.nan
#     # if x.isdigit()==False and len(x)>4:
#     #     x=x[0:4]
#     return x

def _clean_label(x):
    x = str(x)
    if len(x.split(sep='.')) > 2:  # 2.2.8
        i = x.rindex('.')
        x = x[0:i]+x[i+1:]
    if '未做' in x or '未查' in x or '弃查' in x:
        return np.nan
    tmp = re.findall(r'\d+\.?\d*', x)
    if len(tmp) != 0:
        return tmp[0]
    else:
        # print(x)
        x = np.nan
        return x


def clean_label(df):
    """
    清洗df的符号
    :param df: 待清洗df
    :return: 清洗过的df
    """
    for c in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        df[c] = df[c].apply(_clean_label)
        df[c] = df[c].astype('float64')
    return df


def _clean_feature_artificial(df, feature_label):
    """
    手动清洗特征
    :param df: 待清洗数据
    :return: 清洗过数据
    """
    # 清除的---------------------------------------------------------------------------------

    # 3730太多，删除
    del df['3730']
    feature_label.pop('3730')

    # 30007
    del df['30007']
    feature_label.pop('30007')

    # 修改的---------------------------------------------------------------------------------

    sample_num = df.shape[0]
    # 0732 ['未见明显异常' nan '沟纹舌' '未见异常' '萎缩性舌炎']
    df.loc[:, '0732'] = df.loc[:, '0732'].replace('未见明显异常', '无')
    df.loc[:, '0732'] = df.loc[:, '0732'].replace('未见异常', '无')
    feature_label['0732'] = '4'

    # 3301 分离成两个特征
    num_3301 = []
    flag_3301 = []
    for i in range(sample_num):
        id = df.index[i]
        x = df.loc[id, '3301']
        tmp = re.findall(r'\d+\.?\d*', str(x))
        if len(tmp) > 0:
            y = float(tmp[0])
            num_3301.append(y)
        else:
            num_3301.append(np.nan)
        tmp2 = re.findall(r'(阴性|阳性)', str(x))
        if len(tmp2) > 0:
            flag_3301.append(tmp2[0])
        else:
            flag_3301.append('miss')
    df['3301_num'] = num_3301
    df['3301_flag'] = flag_3301
    del df['3301']
    feature_label.pop('3301')
    feature_label['3301_num'] = '1'
    feature_label['3301_flag'] = '5'

    return df


def _clean_nan_feature(df, thresh=0.9):
    """
    清洗缺失值过多的特征
    :param df: 待清洗数据
    :param thresh: 缺失阈值上限
    :return: 清洗过的数据
    """
    exclude_feats = []
    print('----------移除数据缺失多的字段-----------')
    print('移除之前总的字段数量', len(df.columns))
    num_rows = df.shape[0]
    for c in df.columns:
        num_missing = df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_percent = num_missing / float(num_rows)
        if missing_percent > thresh:
            exclude_feats.append(c)
    print("移除缺失数据的字段数量: %s" % len(exclude_feats))
    # 保留超过阈值的特征
    feats = []
    for c in df.columns:
        if c not in exclude_feats:
            feats.append(c)
    print('剩余的字段数量', len(feats))
    return df[feats]


def _clean_useless_feature(df, useless_feature):
    """
    清洗无用特征
    :param df: 待清洗数据
    :param useless_feature:无用特征list
    :return: 清洗后数据
    """
    return df.drop(useless_feature, axis=1)

# 单元测试
# data = [[1,'2.4',3],['s',5,6]]
# index = [0,1]
# columns=['a','b','c']
# df = pd.DataFrame(data=data, index=index, columns=columns)
# # useless_item = ['?', '.']
# # useless_feature = ['b']
# print(df['a'].dtype)
# print(df['b'].dtype)
# print(df['c'].dtype)

