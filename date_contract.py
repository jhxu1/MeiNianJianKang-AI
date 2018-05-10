# coding:utf-8
import time
import pandas as pd


# 获取数据库
def read_data_and_contact():
    # 读取数据
    train = pd.read_csv('meinian_round1_train_20180408.csv', sep=',', encoding='gbk')
    # test = pd.read_csv('meinian_round1_test_a_20180409.csv', sep=',', encoding='gbk')
    test = pd.read_csv('meinian_round1_test_b_20180505.csv', sep=',', encoding='gbk')
    data_part1 = pd.read_csv('meinian_round1_data_part1_20180408.txt', sep='$', encoding='utf-8')
    data_part2 = pd.read_csv('meinian_round1_data_part2_20180408.txt', sep='$', encoding='utf-8')

    part1_2 = pd.concat([data_part1, data_part2], axis=0)  # {0/'index', 1/'columns'}, default 0
    part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)

    # part1_2 = part1_2.iloc[:200000, :]

    vid_set = pd.concat([train['vid'], test['vid']], axis=0)
    vid_set = pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
    part1_2 = part1_2[part1_2['vid'].isin(vid_set['vid'])]
    vid_tabid_group = part1_2.groupby(['vid', 'table_id']).size().reset_index()
    print('------------------------------去重和组合-----------------------------')
    vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']  # 制造新的Col
    vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0] > 1]['new_index']

    # print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
    part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']

    dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
    dup_part = dup_part.sort_values(['vid', 'table_id'])  # 重复部分
    unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]  # 非重复部分

    def merge_table(df):
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            # if if_all_digit(list(df['field_results'])):
            #     merge_df = sum(list(df['field_results'])) / df.shape[0]
            # elif if_all_alpha(list(df['field_results'])):
            #     merge_df = " ".join(list(df['field_results']))
            # else:
            merge_df = df['field_results'].values[0]
        else:
            merge_df = df['field_results'].values[0]
        return merge_df

    part1_2_dup = dup_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()  # 重复语句拼接
    part1_2_dup.rename(columns={0: 'field_results'}, inplace=True)
    part1_2_res = pd.concat([part1_2_dup, unique_part[['vid', 'table_id', 'field_results']]])  # 拼接完成part

    table_id_group = part1_2.groupby('table_id').size().sort_values(ascending=False)
    table_id_group.to_csv('part_tabid_size.csv', encoding='utf-8')  # 每个体检项目参与的人数

    # 行列转换
    print('--------------------------重新组织index和columns---------------------------')
    merge_part1_2 = part1_2_res.pivot(index='vid', values='field_results', columns='table_id')
    print('--------------新的part1_2组合完毕----------')
    print(merge_part1_2.shape)
    # merge_part1_2.to_csv('merge_part1_2.csv', encoding='utf-8')
    # # print(merge_part1_2.head())
    # del merge_part1_2

    # time.sleep(10)
    # print('------------------------重新读取数据merge_part1_2--------------------------')
    # merge_part1_2 = pd.read_csv('merge_part1_2.csv', sep=',', encoding='utf-8')
    merge_part1_2 = merge_part1_2.reset_index()
    train_of_part = merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
    test_of_part = merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]
    train = pd.merge(train, train_of_part, on='vid')
    test = pd.merge(test, test_of_part, on='vid')
    return train, test

