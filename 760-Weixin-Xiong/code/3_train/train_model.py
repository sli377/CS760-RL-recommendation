import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class BusinessRecommenderEnv:
    """推荐系统环境 (简化版)"""
    
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
        
        # 创建简化的商家特征
        self.business_features = self._create_simplified_business_features()
        
        # 创建简化的用户特征
        self.user_features = self._create_simplified_user_features()
        
        # 所有可能的商家ID列表
        self.all_business_ids = list(self.businesses['business_id'].unique())
        
        # 当前状态
        self.current_user_id = None
        self.recommended_businesses = []
        self.current_step = 0
        self.max_steps = 10
        
        print("环境初始化完成")
    
    def _create_simplified_business_features(self):
        """创建简化的商家特征"""
        print("创建简化的商家特征...")
        
        features = {}
        for _, business in self.businesses.iterrows():
            business_id = business['business_id']
            
            # 获取平均评分和评论数
            avg_stars = business['stars'] if 'stars' in business else 0
            review_count = business['review_count'] if 'review_count' in business else 0
            
            # 保存位置信息
            latitude = business['latitude'] if 'latitude' in business else 0
            longitude = business['longitude'] if 'longitude' in business else 0
            city = business['city'] if 'city' in business else ""
            state = business['state'] if 'state' in business else ""
            
            # 提取分类（如果存在）- 仅保留，不进行复杂处理
            categories = []
            if 'categories' in business and business['categories']:
                categories = [cat.strip() for cat in str(business['categories']).split(',')]
            
            # 创建特征字典 - 简化版
            features[business_id] = {
                'avg_stars': avg_stars,
                'review_count': review_count,
                'categories': categories,
                'latitude': latitude,
                'longitude': longitude,
                'city': city,
                'state': state
            }
        
        return features
    
    def _create_simplified_user_features(self):
        """创建简化的用户特征"""
        print("创建简化的用户特征...")
        start_time = time.time()

        features = {}
        for i, user_id in enumerate(tqdm(self.user_visited_businesses.keys())):
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
            if i % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"已处理 {i} 个用户，耗时 {elapsed:.2f} 秒")
        
        total_time = time.time() - start_time
        print(f"用户特征创建完成，总耗时: {total_time:.2f} 秒")
        return features
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两点之间的距离（公里）"""
        # 使用Haversine公式
        R = 6371  # 地球半径（公里）
        
        # 转换为弧度
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine公式
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def reset(self):
        """重置环境并返回初始状态"""
        # 随机选择一个用户
        user_ids = list(self.user_features.keys())
        self.current_user_id = random.choice(user_ids)
        
        # 重置推荐列表和步数
        self.recommended_businesses = []
        self.current_step = 0
        
        # 返回初始状态
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态表示"""
        user_feature = self.user_features[self.current_user_id]
        
        # 用户特征
        state = [
            user_feature['avg_rating'],
            user_feature['num_ratings'],
            user_feature['latitude'],
            user_feature['longitude'],
            user_feature['location_confidence'] / 100  # 归一化
        ]
        
        # 添加已推荐商家的特征
        for business_id in self.recommended_businesses:
            if business_id in self.business_features:
                biz_feature = self.business_features[business_id]
                state.extend([
                    biz_feature['avg_stars'] / 5,  # 归一化
                    biz_feature['review_count'] / 1000,  # 归一化
                    biz_feature['latitude'],
                    biz_feature['longitude']
                ])
        
        # 填充以确保状态向量长度固定
        padding_length = 5 + (4 * 10)  # 用户特征 + (商家特征 * 最大步数)
        if len(state) < padding_length:
            state.extend([0] * (padding_length - len(state)))
        
        return np.array(state)
    
    def step(self, action):
        """执行动作（推荐商家）并返回新状态、奖励和是否结束"""
        business_id = action
        
        # 检查是否已经推荐过该商家
        if business_id in self.recommended_businesses:
            reward = -1.0  # 惩罚重复推荐
        else:
            # 计算奖励
            reward = self._calculate_reward(business_id)
            
            # 添加到推荐列表
            self.recommended_businesses.append(business_id)
        
        # 更新步数
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 返回新状态、奖励和是否结束
        return self._get_state(), reward, done
    
    def _calculate_reward(self, business_id):
        """计算推荐商家的奖励"""
        user_id = self.current_user_id
        
        # 基础奖励
        reward = 0.0
        
        # 1. 评分奖励：如果用户给过这个商家评分，使用实际评分
        if user_id in self.user_business_ratings and business_id in self.user_business_ratings[user_id]:
            rating = self.user_business_ratings[user_id][business_id]
            rating_reward = (rating - 3) / 2  # 将3星评分归一化为0，范围为[-1, 1]
            reward += rating_reward * 1.5  # 评分奖励权重
        
        # 2. 类别匹配奖励：基于用户历史偏好
        category_reward = self._calculate_category_similarity(business_id)
        reward += category_reward * 1.0  # 类别奖励权重
        
        # 3. 位置奖励：基于距离和同城
        location_reward = self._calculate_location_reward(business_id)
        reward += location_reward * 2.0  # 位置奖励权重
        
        # 4. 新颖性奖励：推荐用户未访问过的商家
        if business_id not in self.user_visited_businesses[user_id]:
            reward += 0.5  # 新颖性奖励权重
        
        return reward
    
    def _calculate_category_similarity(self, business_id):
        """计算商家类别与用户历史偏好的相似度"""
        user_id = self.current_user_id
        
        # 获取推荐商家的类别
        if business_id not in self.business_features:
            return 0.0
            
        business_categories = set(self.business_features[business_id]['categories'])
        if not business_categories:
            return 0.0
        
        # 获取用户历史访问商家的类别
        user_categories = set()
        for visited_id in self.user_visited_businesses[user_id]:
            if visited_id in self.business_features:
                user_categories.update(self.business_features[visited_id]['categories'])
        
        if not user_categories:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(business_categories.intersection(user_categories))
        union = len(business_categories.union(user_categories))
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity
    
    def _calculate_location_reward(self, business_id):
        """计算位置奖励"""
        user_id = self.current_user_id
        
        # 获取用户位置
        if user_id not in self.user_features:
            return 0.0
            
        user_feature = self.user_features[user_id]
        user_lat = user_feature['latitude']
        user_lon = user_feature['longitude']
        user_city = user_feature['city']
        user_state = user_feature['state']
        location_confidence = user_feature['location_confidence'] / 100  # 归一化
        
        # 获取商家位置
        if business_id not in self.business_features:
            return 0.0
            
        business_feature = self.business_features[business_id]
        business_lat = business_feature['latitude']
        business_lon = business_feature['longitude']
        business_city = business_feature['city']
        business_state = business_feature['state']
        
        # 如果位置信息缺失，返回0
        if (user_lat == 0 and user_lon == 0) or (business_lat == 0 and business_lon == 0):
            return 0.0
        
        # 计算距离奖励（距离越近越好）
        distance = self.calculate_distance(user_lat, user_lon, business_lat, business_lon)
        
        # 根据距离计算奖励（使用高斯函数，10公里为参考距离）
        distance_reward = np.exp(-(distance**2) / (2 * 100**2))  # 标准差为100公里
        
        # 考虑位置推断的可信度
        distance_reward *= location_confidence
        
        # 同城/同州奖励
        same_city = user_city.lower() == business_city.lower() and user_city != ""
        same_state = user_state.lower() == business_state.lower() and user_state != ""
        
        if same_city:
            distance_reward += 0.5  # 同城奖励
        elif same_state:
            distance_reward += 0.2  # 同州奖励
        
        return distance_reward
    
    def get_valid_actions(self, k=20):
        """获取有效动作（可推荐的商家）"""
        # 已经推荐过的商家
        recommended = set(self.recommended_businesses)
        
        # 从未推荐过的商家中随机选择k个
        valid_businesses = [bid for bid in self.all_business_ids if bid not in recommended]
        
        if len(valid_businesses) > k:
            valid_businesses = random.sample(valid_businesses, k)
        
        return valid_businesses

# 定义PyTorch模型
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN代理，用于推荐商家"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.batch_size = 32
        self.model = DQNModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, valid_actions, env):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            # 探索：随机选择一个有效动作
            return random.choice(valid_actions)
        
        # 利用：选择Q值最高的动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            self.model.eval()
            act_values = self.model(state_tensor).cpu().data.numpy()[0]
            self.model.train()
        
        # 仅考虑有效动作
        valid_q_values = [(action, act_values[0]) for action in valid_actions]
        
        # 按Q值排序，选择最高的
        best_action = sorted(valid_q_values, key=lambda x: x[1], reverse=True)[0][0]
        
        return best_action
    
    def replay(self, batch_size):
        """经验回放，训练模型"""
        if len(self.memory) < batch_size:
            return 0  # 返回0表示没有进行训练
        
        # 从经验回放缓冲区中采样
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 转换为PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        # 计算当前Q值和目标Q值
        self.model.train()
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
        
        # 计算目标Q值
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失并更新模型
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()  # 返回损失值

def load_datasets(data_dir):
    """加载所有数据集 (带进度条)"""
    print(f"从目录 {data_dir} 加载数据集...")
    
    # 加载商家数据
    business_file = os.path.join(data_dir, "yelp_academic_dataset_business.json")
    print(f"加载商家数据: {business_file}")
    business_data = []
    
    # 计算文件行数用于进度条
    with open(business_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="加载商家数据"):
            business_data.append(json.loads(line.strip()))
    businesses = pd.DataFrame(business_data)
    print(f"已加载商家数据: {len(businesses)}条记录")
    
    # 加载评论数据 (有限制)
    review_file = os.path.join(data_dir, "yelp_academic_dataset_review.json")
    print(f"加载评论数据: {review_file} (限制500000条)")
    review_data = []
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=500000, desc="加载评论数据")):
            if i >= 500000:  # 仅加载部分评论
                break
            review_data.append(json.loads(line.strip()))
    reviews = pd.DataFrame(review_data)
    print(f"已加载评论数据: {len(reviews)}条记录")
    
    # 加载小贴士数据
    tip_file = os.path.join(data_dir, "yelp_academic_dataset_tip.json")
    print(f"加载小贴士数据: {tip_file}")
    tip_data = []
    
    # 计算文件行数用于进度条
    with open(tip_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    
    with open(tip_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="加载小贴士数据"):
            tip_data.append(json.loads(line.strip()))
    tips = pd.DataFrame(tip_data)
    print(f"已加载小贴士数据: {len(tips)}条记录")
    
    # 加载用户数据
    user_file = os.path.join(data_dir, "yelp_academic_dataset_user.json")
    print(f"加载用户数据: {user_file}")
    user_data = []
    
    # 计算文件行数用于进度条
    with open(user_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="加载用户数据"):
            user_data.append(json.loads(line.strip()))
    users = pd.DataFrame(user_data)
    print(f"已加载用户数据: {len(users)}条记录")
    
    # 加载用户位置数据
    location_file = os.path.join(data_dir, "inferred_user_locations.json")
    print(f"加载用户位置数据: {location_file}")
    location_data = []
    
    # 计算文件行数用于进度条
    with open(location_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    
    with open(location_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="加载用户位置数据"):
            location_data.append(json.loads(line.strip()))
    user_locations = pd.DataFrame(location_data)
    print(f"已加载用户位置数据: {len(user_locations)}条记录")
    
    return businesses, reviews, tips, users, user_locations

def train_dqn_recommender():
    """训练DQN推荐系统 (带实时可视化)"""
    # 设置数据目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(parent_dir, "ds_sampled")
    
    # 加载数据集
    businesses, reviews, tips, users, user_locations = load_datasets(data_dir)
    
    # 创建环境
    env = BusinessRecommenderEnv(users, businesses, reviews, tips, user_locations)
    
    # DQN参数
    state_size = 5 + (4 * 10)  # 用户特征 + (商家特征 * 最大步数)
    action_size = 1  # 推荐一个商家
    
    # 创建DQN代理
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    episodes = 100  # 训练轮数
    max_steps = 10  # 每轮最大步数
    
    # 创建TensorBoard writer
    log_dir = os.path.join(data_dir, "logs", f"dqn_recommender_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志保存到: {log_dir}")
    print(f"启动TensorBoard: tensorboard --logdir={log_dir}")
    
    # 设置实时绘图
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 奖励图表
    reward_line, = ax1.plot([], [], 'b-')
    avg_reward_line, = ax1.plot([], [], 'r-')
    ax1.set_title('DQN推荐系统训练奖励')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('奖励')
    ax1.grid(True)
    ax1.legend(['单轮奖励', '平均奖励(10轮)'])
    
    # 损失和探索率图表
    loss_line, = ax2.plot([], [], 'g-')
    ax2.set_title('损失和探索率')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.grid(True)
    
    # 添加第二个Y轴用于显示探索率
    ax3 = ax2.twinx()
    epsilon_line, = ax3.plot([], [], 'm-')
    ax3.set_ylabel('探索率')
    ax3.legend(['探索率'], loc='lower left')
    ax2.legend(['损失'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # 记录训练进度
    rewards_history = []
    losses_history = []
    epsilon_history = []
    
    # 训练开始
    print(f"开始训练: {episodes}轮, 每轮{max_steps}步")
    start_time = time.time()
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            # 选择动作
            action = agent.act(state, valid_actions, env)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 记忆经验
            agent.remember(state, 0, reward, next_state, done)  # 假设所有商家共享同一个动作ID (0)
            
            # 更新状态和累积奖励
            state = next_state
            total_reward += reward
            
            # 经验回放
            loss = agent.replay(agent.batch_size)
            if loss > 0:
                episode_losses.append(loss)
            
            if done:
                break
        
        # 计算本轮平均损失
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        # 记录本轮奖励、损失和探索率
        rewards_history.append(total_reward)
        losses_history.append(avg_loss)
        epsilon_history.append(agent.epsilon)
        
        # 写入TensorBoard
        writer.add_scalar('Reward/train', total_reward, episode)
        writer.add_scalar('Loss/train', avg_loss, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        
        # 更新图表
        reward_line.set_xdata(range(len(rewards_history)))
        reward_line.set_ydata(rewards_history)
        
        # 计算平均奖励
        if len(rewards_history) >= 10:
            avg_rewards = [np.mean(rewards_history[max(0, i-9):i+1]) for i in range(len(rewards_history))]
            avg_reward_line.set_xdata(range(len(avg_rewards)))
            avg_reward_line.set_ydata(avg_rewards)
            avg_reward = avg_rewards[-1]
        else:
            avg_reward = total_reward
        
        # 更新损失和探索率图表
        loss_line.set_xdata(range(len(losses_history)))
        loss_line.set_ydata(losses_history)
        
        epsilon_line.set_xdata(range(len(epsilon_history)))
        epsilon_line.set_ydata(epsilon_history)
        
        # 自动调整坐标轴范围
        for ax in [ax1, ax2, ax3]:
            ax.relim()
            ax.autoscale_view()
        
        # 更新图表
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # 输出训练进度信息
        elapsed_time = time.time() - start_time
        print(f"轮次: {episode + 1}/{episodes}, 奖励: {total_reward:.2f}, 平均奖励: {avg_reward:.2f}, 损失: {avg_loss:.4f}, 探索率: {agent.epsilon:.2f}, 用时: {elapsed_time:.1f}秒")
    
    # 关闭TensorBoard writer
    writer.close()
    
    print(f"训练完成! 总用时: {time.time() - start_time:.1f}秒")
    
    # 保存最终图表
    plt.ioff()  # 关闭交互模式
    
    # 绘制最终的奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, 'b-', label='单轮奖励')
    
    # 绘制平均奖励曲线
    if len(rewards_history) >= 10:
        avg_rewards = [np.mean(rewards_history[max(0, i-9):i+1]) for i in range(len(rewards_history))]
        plt.plot(avg_rewards, 'r-', label='平均奖励(10轮)')
    
    plt.title('DQN推荐系统训练奖励')
    plt.xlabel('轮次')
    plt.ylabel('累积奖励')
    plt.grid(True)
    plt.legend()
    
    # 保存奖励曲线
    plot_file = os.path.join(data_dir, "dqn_rewards.png")
    plt.savefig(plot_file)
    print(f"奖励曲线已保存到: {plot_file}")
    
    # 保存模型
    model_file = os.path.join(data_dir, "dqn_recommender_model.pt")
    torch.save(agent.model.state_dict(), model_file)
    print(f"模型已保存到: {model_file}")
    
    # 评估模型
    evaluate_recommender(agent, env, data_dir)

def evaluate_recommender(agent, env, data_dir, num_tests=20):
    """评估推荐系统"""
    print(f"\n评估推荐系统 (测试{num_tests}个用户)...")
    
    total_reward = 0
    recommendations = {}
    
    # 创建进度条
    progress_bar = tqdm(total=num_tests, desc="评估进度")
    
    for i in range(num_tests):
        # 重置环境
        state = env.reset()
        user_id = env.current_user_id
        user_recommendations = []
        user_reward = 0
        
        for step in range(env.max_steps):
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            # 选择动作 (使用贪心策略，不探索)
            agent.epsilon = 0  # 禁用探索
            action = agent.act(state, valid_actions, env)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 记录推荐和奖励
            user_recommendations.append({
                'business_id': action,
                'reward': reward
            })
            user_reward += reward
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录该用户的推荐结果
        recommendations[user_id] = {
            'recommendations': user_recommendations,
            'total_reward': user_reward
        }
        total_reward += user_reward
        
        # 更新进度条
        progress_bar.update(1)
    
    progress_bar.close()
    
    # 计算平均奖励
    avg_reward = total_reward / num_tests
    print(f"平均奖励: {avg_reward:.2f}")
    
    # 保存推荐结果
    output_file = os.path.join(data_dir, "dqn_recommendations.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2)
    print(f"推荐结果已保存到: {output_file}")

if __name__ == "__main__":
    train_dqn_recommender()