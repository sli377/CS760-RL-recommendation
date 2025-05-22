# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Michael Kelly.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the implementation provided by the authors of https://arxiv.org/pdf/2006.05779.pdf
#  and https://arxiv.org/pdf/2111.03474.pdf: Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis
# Joemon M. Jose.
####################################################################################
import os
import pandas as pd
import json
import pickle
import random
from sklearn.model_selection import train_test_split

#############################################
# 工具函数
#############################################

def to_pickled_df(data_directory, **kwargs):
    """
    将DataFrame保存为pickle文件。
    
    Args:
        data_directory (str): 保存pickle文件的目录
        **kwargs: 要保存的DataFrame，键作为文件名
    """
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def create_id_mappings(df):
    """
    为user_id和business_id创建连续的整数映射。
    
    Args:
        df (DataFrame): 包含user_id和business_id的DataFrame
        
    Returns:
        tuple: (user_id映射字典, business_id映射字典, 用户数量, 商品数量)
    """
    # 创建唯一ID的映射
    unique_users = df['user_id'].unique()
    unique_items = df['business_id'].unique()
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # 保存映射到文件
    with open('Yelp/user_id_map.json', 'w') as f:
        json.dump(user_id_map, f)
    with open('Yelp/item_id_map.json', 'w') as f:
        json.dump(item_id_map, f)
        
    print(f"用户数量: {len(user_id_map)}")
    print(f"商品数量: {len(item_id_map)}")
    
    return user_id_map, item_id_map, len(user_id_map), len(item_id_map)

def pad_history(itemlist, length, pad_item):
    """
    将历史列表填充到固定长度。
    
    Args:
        itemlist (list): 商品ID列表
        length (int): 目标长度
        pad_item (int): 用于填充的商品ID
        
    Returns:
        list: 填充后的历史列表
    """
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) == 0:
        return [pad_item] * length
    return itemlist + [pad_item] * (length - len(itemlist))

#############################################
# 数据处理函数
#############################################

def process_sequence(sequence, n, pad_item):
    """
    处理用户交互序列，保持时间顺序。
    
    Args:
        sequence (list): 用户交互序列
        n (int): 序列长度
        pad_item (int): 填充值
        
    Returns:
        tuple: (处理后的序列块, 序列长度列表, 动作列表)
    """
    seq_length = len(sequence)
    
    if seq_length > n:
        # 将序列分割成较小的n长度列表，保持时间顺序
        chunks = []
        chunk_lengths = []
        actions = []
        
        # 从最早的交互开始处理
        for i in range(0, seq_length - n + 1):
            chunk = sequence[i:i + n]
            chunks.append(chunk[:-1])  # 不包括最后一个动作
            chunk_lengths.append(n - 1)
            actions.append(chunk[-1])  # 最后一个动作作为目标
            
    else:
        # 如果序列长度小于n，填充到n长度
        actions = [sequence[-1]]
        sequence = sequence[:-1]  # 移除最后一个动作
        seq_length = len(sequence)
        chunk_lengths = [seq_length]
        padding = [pad_item] * (n - seq_length - 1)
        chunks = [sequence + padding]
    
    return chunks, chunk_lengths, actions

def map_dates_to_sequence(df):
    """
    将日期映射为序列号，同一天同一小时的记录映射为相同的序号
    
    Args:
        df (DataFrame): 包含date字段的DataFrame
        
    Returns:
        DataFrame: 包含映射后日期的DataFrame
    """
    # 确保date字段存在
    if 'date' not in df.columns:
        print("错误: 数据中缺少'date'字段")
        return df
    
    try:
        # 转换为datetime对象
        df['date'] = pd.to_datetime(df['date'])
        
        # 提取日期和小时
        df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d %H')
        
        # 找出所有唯一的日期时间值并排序
        unique_dates = sorted(df['date_formatted'].unique())
        
        # 创建映射: 日期时间 -> 序号(从1开始)
        date_map = {date: idx + 1 for idx, date in enumerate(unique_dates)}
        
        # 保存映射关系
        os.makedirs('Yelp', exist_ok=True)
        with open('Yelp/date_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(date_map, f, ensure_ascii=False, indent=2)
        
        # 应用映射
        df['date'] = df['date_formatted'].map(date_map)
        
        # 删除临时列
        df = df.drop('date_formatted', axis=1)
        
        print(f"日期映射完成，共有{len(date_map)}个唯一时间点")
        
    except Exception as e:
        print(f"日期处理出错: {e}")
        # 如果日期处理失败，使用序列号作为时间戳
        print("使用序列号作为时间戳")
        df['date'] = range(1, len(df) + 1)
    
    return df

def process_json_data(json_file_path):
    """
    处理Yelp JSON数据文件并转换为所需格式。
    
    Args:
        json_file_path (str): JSON文件路径
        
    Returns:
        DataFrame: 处理后的数据
    """
    try:
        # 读取整个JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载JSON文件，包含{len(data)}条记录")
    except json.JSONDecodeError:
        print("JSON格式错误，尝试作为JSONL文件读取...")
        data = []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"解析JSON行出错: {line[:50]}...")
                    continue
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 创建ID映射
    user_id_map, item_id_map, num_users, num_items = create_id_mappings(df)
    
    # 应用ID映射
    df['user_id'] = df['user_id'].map(user_id_map)
    df['business_id'] = df['business_id'].map(item_id_map)
    
    # 处理日期字段
    if 'date' in df.columns:
        df = map_dates_to_sequence(df)
    else:
        print("警告: 没有date字段，创建序列化的日期")
        df['date'] = range(1, len(df) + 1)
    
    # 使用评分作为交互类型
    df['rating'] = df['stars']
    
    # 按user_id和date排序
    if 'date' in df.columns:
        df = df.sort_values(['user_id', 'date'])
    else:
        print("警告: 没有date字段，无法按时间排序")
        df = df.sort_values(['user_id'])
    
    # 确保必要的列存在
    required_columns = ['user_id', 'business_id', 'rating', 'stars', 'date']
    for col in required_columns:
        if col not in df.columns:
            if col == 'date':
                df['date'] = range(len(df))
                print(f"警告: 创建了序列化的{col}列")
            else:
                print(f"错误: 缺少必要的列 {col}")
                return None
    
    return df

#############################################
# 主程序
#############################################

if __name__ == '__main__':
    data_directory = 'Yelp/'
    json_file_path = 'filtered_2019_data.json'
    sequence_length = 20
    
    print('#############################################################')
    print('Processing Yelp dataset with sequence length', sequence_length)
    
    # 处理JSON数据
    print("正在读取并处理Yelp评论数据...")
    sorted_events = process_json_data(json_file_path)
    print(f"读取了 {len(sorted_events)} 条评论记录")
    
    # 保存处理后的事件
    os.makedirs(data_directory, exist_ok=True)
    to_pickled_df(data_directory, sorted_events=sorted_events)
    
    # 获取唯一的item IDs
    item_ids = sorted_events.business_id.unique()
    pad_item = len(item_ids)
    print(f"数据集中共有 {pad_item} 个唯一商品")
    
    # 按时间分割数据
    print("正在按时间顺序分割数据为训练集、验证集和测试集...")
    sorted_events = sorted_events.sort_values('date')
    
    # 使用时间顺序分割数据
    train_idx = int(len(sorted_events) * 0.8)  # 80%用于训练
    valid_idx = int(len(sorted_events) * 0.9)  # 10%用于验证
    train_sessions = sorted_events.iloc[:train_idx]
    valid_sessions = sorted_events.iloc[train_idx:valid_idx]
    test_sessions = sorted_events.iloc[valid_idx:]
    
    print(f"训练集大小: {len(train_sessions)} 条交互记录")
    print(f"验证集大小: {len(valid_sessions)} 条交互记录")
    print(f"测试集大小: {len(test_sessions)} 条交互记录")
    
    # 保存训练集和验证集
    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_valid=valid_sessions)
    # 保存测试集
    to_pickled_df(data_directory, sampled_test=test_sessions)
    
    # 处理训练集序列
    print("正在处理训练集序列...")
    groups = train_sessions.groupby('user_id')
    ids = train_sessions.user_id.unique()
    
    # 计算每个用户的reward平均值
    user_reward_means = train_sessions.groupby('user_id')['rating'].mean()
    
    state, len_state, action, reward, next_state, len_next_state, is_done, quality = [], [], [], [], [], [], [], []
    
    for id in ids:
        group = groups.get_group(id)
        # 确保按时间顺序处理
        group = group.sort_values('date')
        history = []
        user_threshold = user_reward_means[id]
        
        for index, row in group.iterrows():
            s = list(history)
            len_state.append(sequence_length if len(s) >= sequence_length else 1 if len(s) == 0 else len(s))
            s = pad_history(s, sequence_length, pad_item)
            a = row['business_id']
            r = row['rating']  # 直接使用评分作为奖励
            state.append(s)
            action.append(a)
            reward.append(r)
            quality.append('high' if r > user_threshold else 'low')
            history.append(row['business_id'])
            next_s = list(history)
            len_next_state.append(sequence_length if len(next_s) >= sequence_length else 1 if len(next_s) == 0 else len(next_s))
            next_s = pad_history(next_s, sequence_length, pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1] = True

    train_dic = {
        'state': state,
        'len_state': len_state,
        'action': action,
        'reward': reward,
        'next_state': next_state,
        'len_next_states': len_next_state,
        'is_done': is_done,
        'quality': quality
    }
    train_replay_buffer = pd.DataFrame(data=train_dic)
    to_pickled_df(data_directory, replay_buffer=train_replay_buffer)
    
    dic = {'state_size': [sequence_length], 'item_num': [pad_item]}
    data_statis = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, data_statis=data_statis)
    
    print("数据处理完成！所有文件已保存到", data_directory)
    print('#############################################################')