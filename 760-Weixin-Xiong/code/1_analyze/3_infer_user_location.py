import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import warnings

warnings.filterwarnings('ignore')

def load_json_file(file_path):
    """加载JSON文件并返回DataFrame"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def calculate_distance(lat1, lon1, lat2, lon2):
    """计算两点之间的距离（公里）"""
    return great_circle((lat1, lon1), (lat2, lon2)).kilometers

def infer_user_locations(business_df, review_df, tip_df):
    """推断用户位置"""
    print("开始推断用户位置...")
    
    # 创建商家ID到位置的映射
    business_locations = {}
    for _, row in business_df.iterrows():
        business_locations[row['business_id']] = {
            'business_id': row['business_id'],
            'name': row['name'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'city': row['city'],
            'state': row['state'],
            'postal_code': row.get('postal_code', None)
        }
    
    # 提取所有唯一用户ID
    review_users = set(review_df['user_id'].unique())
    tip_users = set(tip_df['user_id'].unique())
    unique_users = review_users.union(tip_users)
    
    print(f"Review数据集中的唯一用户数: {len(review_users)}")
    print(f"Tip数据集中的唯一用户数: {len(tip_users)}")
    print(f"两个数据集合并后的唯一用户数: {len(unique_users)}")
    
    # 合并所有用户交互数据
    user_interactions = defaultdict(list)
    
    # 处理评论数据 (权重最高)
    print("处理评论数据...")
    for _, row in review_df.iterrows():
        business_id = row['business_id']
        user_id = row['user_id']
        if business_id in business_locations:
            # 将评论时间、星级等因素考虑进去
            interaction = {
                'business_id': business_id,
                'type': 'review',
                'timestamp': row['date'],
                'stars': row['stars'],
                'weight': 3.0  # 评论的基础权重
            }
            user_interactions[user_id].append(interaction)
    
    # 处理小贴士数据 (权重较低)
    print("处理小贴士数据...")
    for _, row in tip_df.iterrows():
        business_id = row['business_id']
        user_id = row['user_id']
        if business_id in business_locations:
            interaction = {
                'business_id': business_id,
                'type': 'tip',
                'timestamp': row['date'],
                'compliment_count': row['compliment_count'],
                'weight': 1.0 + min(row['compliment_count'] / 10, 1.0)  # 根据点赞数调整权重
            }
            user_interactions[user_id].append(interaction)
    
    # 确认有交互数据的用户数量
    active_users = set(user_interactions.keys())
    print(f"有交互数据的唯一用户数: {len(active_users)}")
    
    # 推断用户位置
    print(f"开始为{len(active_users)}个唯一用户推断位置...")
    user_locations = {}
    
    for user_id, interactions in user_interactions.items():
        if not interactions:
            continue
        
        # 收集该用户交互过的所有商家位置
        locations = []
        weights = []
        cities = defaultdict(float)
        states = defaultdict(float)
        
        for interaction in interactions:
            business_id = interaction['business_id']
            weight = interaction['weight']
            
            if business_id in business_locations:
                business = business_locations[business_id]
                
                # 添加位置和权重
                locations.append((business['latitude'], business['longitude']))
                weights.append(weight)
                
                # 累计城市和州的权重
                cities[business['city']] += weight
                states[business['state']] += weight
        
        if not locations:
            continue
            
        # 转换为numpy数组以便聚类
        locations_array = np.array(locations)
        weights_array = np.array(weights)
        
        # 尝试使用DBSCAN进行基于密度的聚类
        try:
            # 设置扫描半径为15公里，最小样本为2
            clustering = DBSCAN(eps=0.135, min_samples=2, algorithm='ball_tree', 
                                metric='haversine').fit(np.radians(locations_array))
            labels = clustering.labels_
            
            # 如果所有点都是噪声（没有聚类成功）
            if len(set(labels)) == 1 and -1 in labels:
                # 使用加权中心点
                latitude = np.average(locations_array[:, 0], weights=weights_array)
                longitude = np.average(locations_array[:, 1], weights=weights_array)
                cluster_type = "weighted_center"
            else:
                # 找出最大的聚类
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # 去除噪声点
                
                if not unique_labels:  # 如果没有有效聚类
                    latitude = np.average(locations_array[:, 0], weights=weights_array)
                    longitude = np.average(locations_array[:, 1], weights=weights_array)
                    cluster_type = "weighted_center"
                else:
                    # 计算每个聚类的权重总和
                    cluster_weights = defaultdict(float)
                    for i, label in enumerate(labels):
                        if label != -1:  # 不计算噪声点
                            cluster_weights[label] += weights_array[i]
                    
                    # 找出权重最大的聚类
                    max_cluster = max(cluster_weights, key=cluster_weights.get)
                    cluster_indices = [i for i, label in enumerate(labels) if label == max_cluster]
                    
                    # 计算该聚类的加权中心
                    cluster_locations = locations_array[cluster_indices]
                    cluster_weights = weights_array[cluster_indices]
                    latitude = np.average(cluster_locations[:, 0], weights=cluster_weights)
                    longitude = np.average(cluster_locations[:, 1], weights=cluster_weights)
                    cluster_type = "dbscan_cluster"
        except Exception as e:
            # 如果聚类失败，使用加权中心点
            latitude = np.average(locations_array[:, 0], weights=weights_array)
            longitude = np.average(locations_array[:, 1], weights=weights_array)
            cluster_type = "weighted_center"
        
        # 确定最可能的城市和州
        most_likely_city = max(cities.items(), key=lambda x: x[1])[0] if cities else None
        most_likely_state = max(states.items(), key=lambda x: x[1])[0] if states else None
        
        # 计算位置的可信度分数 (0-100)
        confidence = min(100, len(interactions) * 5)  # 基于交互数量
        
        # 如果用户与特定区域有强关联，增加可信度
        max_city_weight = max(cities.values()) if cities else 0
        total_weight = sum(weights)
        if total_weight > 0 and max_city_weight / total_weight > 0.7:
            confidence = min(100, confidence + 20)  # 如果与某个城市的关联很强
            
        # 根据聚类类型调整可信度
        if cluster_type == "dbscan_cluster":
            confidence = min(100, confidence + 10)  # DBSCAN聚类成功增加可信度
            
        # 计算每个位置的平均距离，距离越大可信度越低
        avg_distance = 0
        if len(locations) > 1:
            distances = []
            center = (latitude, longitude)
            for i, loc in enumerate(locations):
                dist = calculate_distance(center[0], center[1], loc[0], loc[1])
                distances.append(dist)
            avg_distance = np.mean(distances)
            
            # 如果平均距离太大，降低可信度
            if avg_distance > 50:  # 如果平均距离大于50公里
                confidence = max(10, confidence - 20)
        
        # 存储用户位置信息
        user_locations[user_id] = {
            'user_id': user_id,
            'inferred_latitude': float(latitude),
            'inferred_longitude': float(longitude),
            'inferred_city': most_likely_city,
            'inferred_state': most_likely_state,
            'confidence_score': int(confidence),
            'inference_method': cluster_type,
            'interaction_count': len(interactions),
            'avg_distance_km': float(round(avg_distance, 2)) if avg_distance else None,
            'inference_timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
        }
        
    print(f"成功为{len(user_locations)}个用户推断位置")
    return user_locations

def export_user_locations(user_locations, output_file):
    """导出用户位置信息到JSON文件"""
    print(f"导出用户位置信息到{output_file}")
    
    # 将字典转换为列表
    user_locations_list = list(user_locations.values())
    
    # 按可信度分数降序排序
    user_locations_list.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for location in user_locations_list:
            f.write(json.dumps(location) + '\n')
    
    print(f"位置数据已保存到: {output_file}")
    
    # 生成一些统计信息
    confidence_ranges = {
        'very_high': len([loc for loc in user_locations_list if loc['confidence_score'] >= 80]),
        'high': len([loc for loc in user_locations_list if 60 <= loc['confidence_score'] < 80]),
        'medium': len([loc for loc in user_locations_list if 40 <= loc['confidence_score'] < 60]),
        'low': len([loc for loc in user_locations_list if 20 <= loc['confidence_score'] < 40]),
        'very_low': len([loc for loc in user_locations_list if loc['confidence_score'] < 20])
    }
    
    print("\n位置推断可信度分布:")
    print(f"  非常高 (80-100): {confidence_ranges['very_high']} 用户")
    print(f"  高 (60-79): {confidence_ranges['high']} 用户")
    print(f"  中等 (40-59): {confidence_ranges['medium']} 用户")
    print(f"  低 (20-39): {confidence_ranges['low']} 用户")
    print(f"  非常低 (0-19): {confidence_ranges['very_low']} 用户")

def main():
    # 设置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    sampled_dir = os.path.join(parent_dir, "ds_sampled")
    
    business_file = os.path.join(sampled_dir, "yelp_academic_dataset_business.json")
    review_file = os.path.join(sampled_dir, "yelp_academic_dataset_review.json")
    tip_file = os.path.join(sampled_dir, "yelp_academic_dataset_tip.json")
    
    output_file = os.path.join(sampled_dir, "inferred_user_locations.json")
    
    print("加载数据集...")
    # 加载数据
    business_df = load_json_file(business_file)
    print(f"已加载{len(business_df)}条商家记录")
    
    # 加载评论数据（可能很大，我们只需要business_id、user_id、date和stars）
    print("加载评论数据...")
    review_df = pd.DataFrame()
    chunk_size = 100000  # 分块处理以节省内存
    
    with open(review_file, 'r', encoding='utf-8') as f:
        reviews = []
        for i, line in enumerate(f):
            reviews.append(json.loads(line.strip()))
            if (i + 1) % chunk_size == 0:
                chunk_df = pd.DataFrame(reviews)
                review_df = pd.concat([review_df, chunk_df[['business_id', 'user_id', 'date', 'stars']]])
                reviews = []
        
        # 处理剩余的记录
        if reviews:
            chunk_df = pd.DataFrame(reviews)
            review_df = pd.concat([review_df, chunk_df[['business_id', 'user_id', 'date', 'stars']]])
    
    print(f"已加载{len(review_df)}条评论记录")
    
    # 加载小贴士数据
    tip_df = load_json_file(tip_file)
    print(f"已加载{len(tip_df)}条小贴士记录")
    
    # 推断用户位置
    user_locations = infer_user_locations(business_df, review_df, tip_df)
    
    # 导出用户位置
    export_user_locations(user_locations, output_file)
    
    print("位置推断完成!")

if __name__ == "__main__":
    main()