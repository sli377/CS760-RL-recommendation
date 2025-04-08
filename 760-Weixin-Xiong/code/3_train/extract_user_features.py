import os
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import time

# 设置数据目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录(3_train)
parent_dir = os.path.dirname(os.path.dirname(script_dir))  # 项目根目录(760)
data_dir = os.path.join(parent_dir, "ds_sampled")  # 数据目录

def load_datasets(data_dir):
    """加载所有数据集"""
    print(f"从目录 {data_dir} 加载数据集...")
    
    # 加载商家数据
    business_file = os.path.join(data_dir, "yelp_academic_dataset_business.json")
    business_data = []
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            business_data.append(json.loads(line.strip()))
    businesses = pd.DataFrame(business_data)
    print(f"已加载商家数据: {len(businesses)}条记录")
    
    # 加载评论数据
    review_file = os.path.join(data_dir, "yelp_academic_dataset_review.json")
    review_data = []
    with open(review_file, 'r', encoding='utf-8') as f:
        # 限制加载的评论数量以节省内存
        for i, line in enumerate(f):
            if i >= 500000:  # 仅加载部分评论
                break
            review_data.append(json.loads(line.strip()))
    reviews = pd.DataFrame(review_data)
    print(f"已加载评论数据: {len(reviews)}条记录")
    
    # 加载小贴士数据
    tip_file = os.path.join(data_dir, "yelp_academic_dataset_tip.json")
    tip_data = []
    with open(tip_file, 'r', encoding='utf-8') as f:
        for line in f:
            tip_data.append(json.loads(line.strip()))
    tips = pd.DataFrame(tip_data)
    print(f"已加载小贴士数据: {len(tips)}条记录")
    
    # 加载用户数据
    user_file = os.path.join(data_dir, "yelp_academic_dataset_user.json")
    user_data = []
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            user_data.append(json.loads(line.strip()))
    users = pd.DataFrame(user_data)
    print(f"已加载用户数据: {len(users)}条记录")
    
    # 加载用户位置数据
    location_file = os.path.join(data_dir, "inferred_user_locations.json")
    location_data = []
    with open(location_file, 'r', encoding='utf-8') as f:
        for line in f:
            location_data.append(json.loads(line.strip()))
    user_locations = pd.DataFrame(location_data)
    print(f"已加载用户位置数据: {len(user_locations)}条记录")
    
    return businesses, reviews, tips, users, user_locations

class BusinessRecommenderEnv:
    """推荐系统环境 (简化版重建，只为提取用户特征)"""
    
    def __init__(self, users_data, businesses_data, reviews_data, tips_data, user_locations_data):
        """初始化环境"""
        print("初始化环境...")
        self.users = users_data
        self.businesses = businesses_data
        self.reviews = reviews_data
        self.tips = tips_data
        self.user_locations = user_locations_data
        
        # 创建用户-商家交互矩阵
        print("创建用户-商家交互矩阵...")
        self.user_business_ratings = defaultdict(dict)
        for _, review in self.reviews.iterrows():
            self.user_business_ratings[review['user_id']][review['business_id']] = review['stars']
        
        # 创建用户已访问商家集合
        print("创建用户已访问商家集合...")
        self.user_visited_businesses = defaultdict(set)
        for _, review in self.reviews.iterrows():
            self.user_visited_businesses[review['user_id']].add(review['business_id'])
        for _, tip in self.tips.iterrows():
            self.user_visited_businesses[tip['user_id']].add(tip['business_id'])
        
        # 创建简化的用户特征
        print("创建用户特征...")
        start_time = time.time()
        self.user_features = self._create_simplified_user_features()
        end_time = time.time()
        print(f"用户特征创建完成，用时: {end_time - start_time:.2f}秒")
        
        print("环境初始化完成")
    
    def _create_simplified_user_features(self):
        """创建简化的用户特征"""
        features = {}
        total_users = len(self.user_visited_businesses.keys())
        
        for i, user_id in enumerate(self.user_visited_businesses.keys()):
            if i % 10000 == 0:
                print(f"处理用户特征: {i}/{total_users} ({i/total_users*100:.1f}%)")
                
            # 计算用户平均评分
            user_ratings = [self.user_business_ratings[user_id][biz_id] 
                           for biz_id in self.user_business_ratings[user_id]]
            avg_rating = np.mean(user_ratings) if user_ratings else 0
            
            # 获取用户位置信息
            user_location = self.user_locations[
                self.user_locations['user_id'] == user_id]
            
            if not user_location.empty:
                latitude = user_location['inferred_latitude'].values[0]
                longitude = user_location['inferred_longitude'].values[0]
                city = user_location['inferred_city'].values[0]
                state = user_location['inferred_state'].values[0]
                confidence = user_location['confidence_score'].values[0]
            else:
                latitude = longitude = 0
                city = state = ""
                confidence = 0
            
            # 创建特征字典 - 简化版
            features[user_id] = {
                'avg_rating': avg_rating,
                'num_ratings': len(user_ratings),
                'latitude': latitude,
                'longitude': longitude,
                'city': city,
                'state': state,
                'location_confidence': confidence
            }
        
        return features

def save_user_features(user_features, output_file):
    """保存用户特征到文件"""
    print(f"保存用户特征到: {output_file}")
    
    # 将用户特征转换为可序列化的格式
    serializable_features = {}
    for user_id, features in user_features.items():
        # 确保所有值都是可序列化的
        serializable_features[user_id] = {
            'avg_rating': float(features['avg_rating']),
            'num_ratings': int(features['num_ratings']),
            'latitude': float(features['latitude']),
            'longitude': float(features['longitude']),
            'city': str(features['city']),
            'state': str(features['state']),
            'location_confidence': float(features['location_confidence'])
        }
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_features, f)
    
    print(f"成功保存 {len(serializable_features)} 个用户特征")

def main():
    """主函数"""
    # 加载数据集
    businesses, reviews, tips, users, user_locations = load_datasets(data_dir)
    
    # 创建环境并计算用户特征
    env = BusinessRecommenderEnv(users, businesses, reviews, tips, user_locations)
    
    # 保存用户特征
    output_file = os.path.join(data_dir, "user_features.json")
    save_user_features(env.user_features, output_file)
    
    # 还是需要的话，可以保存整个环境
    env_file = os.path.join(data_dir, "recommender_env.pkl")
    with open(env_file, 'wb') as f:
        pickle.dump(env, f)
    print(f"环境对象已保存到: {env_file}")

if __name__ == "__main__":
    main()