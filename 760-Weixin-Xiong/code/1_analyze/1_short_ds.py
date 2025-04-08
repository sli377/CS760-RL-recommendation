import json
import os
import random
from collections import defaultdict

def sample_dataset(input_file, output_file, sample_ratio, key_field=None, preserve_keys=None):
    """
    按比例采样JSON数据集，并可选择性地保留指定key的所有记录
    
    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    - sample_ratio: 采样比例 (0-1)
    - key_field: 用于标识记录的字段名
    - preserve_keys: 需要保留的key值列表
    
    返回:
    - 采样后的记录数
    - 保留的key集合
    """
    preserved_keys = set(preserve_keys) if preserve_keys else set()
    sampled_keys = set()
    
    # 读取和采样数据
    sampled_data = []
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            data = json.loads(line.strip())
            
            # 是否保留该记录
            keep = False
            
            # 如果有需要保留的key
            if key_field and key_field in data:
                key_value = data[key_field]
                
                # 如果该key在保留列表中，则保留
                if key_value in preserved_keys:
                    keep = True
                    sampled_keys.add(key_value)
                # 否则按概率采样
                elif random.random() < sample_ratio:
                    keep = True
                    sampled_keys.add(key_value)
            # 没有指定key_field，直接按概率采样
            elif random.random() < sample_ratio:
                keep = True
            
            # 保留该记录
            if keep:
                sampled_data.append(data)
    
    # 写入采样数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in sampled_data:
            f.write(json.dumps(data) + '\n')
    
    print(f"文件 {os.path.basename(input_file)}: 从 {total_count} 条记录中采样 {len(sampled_data)} 条 (比例: {len(sampled_data)/total_count:.2%})")
    return len(sampled_data), sampled_keys

def create_sampled_datasets(input_dir, output_dir, target_total=1000000):
    """
    创建采样数据集，保持各数据集之间的关联关系
    
    参数:
    - input_dir: 输入目录
    - output_dir: 输出目录 
    - target_total: 目标总记录数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据集文件
    datasets = {
        "business": "yelp_academic_dataset_business.json",
        "checkin": "yelp_academic_dataset_checkin.json",
        "review": "yelp_academic_dataset_review.json",
        "tip": "yelp_academic_dataset_tip.json",
        "user": "yelp_academic_dataset_user.json"
    }
    
    # 各数据集的大小和计算采样率
    dataset_sizes = {
        "business": 150346,
        "checkin": 131930,
        "review": 6990280,
        "tip": 908915,
        "user": 1987897
    }
    
    total_records = sum(dataset_sizes.values())
    
    # 根据数据集大小分配目标记录数
    target_sizes = {}
    preserved_business_ids = set()
    preserved_user_ids = set()
    
    # 首先为business分配记录数，作为基础
    business_ratio = 0.05  # 分配5%的目标记录给business
    target_sizes["business"] = int(target_total * business_ratio)
    remaining_target = target_total - target_sizes["business"]
    
    # 根据其他数据集的原始大小分配剩余记录
    remaining_total = total_records - dataset_sizes["business"]
    for dataset, size in dataset_sizes.items():
        if dataset != "business":
            ratio = size / remaining_total
            target_sizes[dataset] = int(remaining_target * ratio)
    
    # 修正可能的舍入误差
    total_allocated = sum(target_sizes.values())
    if total_allocated < target_total:
        # 将差值添加到最大的数据集
        largest_dataset = max(dataset_sizes.items(), key=lambda x: x[1])[0]
        target_sizes[largest_dataset] += (target_total - total_allocated)
    
    print("目标采样大小:")
    for dataset, target in target_sizes.items():
        print(f"  {dataset}: {target} 条记录")
    
    # 步骤1: 采样business数据集
    business_ratio = target_sizes["business"] / dataset_sizes["business"]
    _, preserved_business_ids = sample_dataset(
        os.path.join(input_dir, datasets["business"]),
        os.path.join(output_dir, datasets["business"]),
        business_ratio,
        "business_id"
    )
    
    # 步骤2: 采样user数据集
    user_ratio = target_sizes["user"] / dataset_sizes["user"]
    _, preserved_user_ids = sample_dataset(
        os.path.join(input_dir, datasets["user"]),
        os.path.join(output_dir, datasets["user"]),
        user_ratio,
        "user_id"
    )
    
    # 步骤3: 采样review数据集，保持与采样的business和user的关联
    review_ratio = target_sizes["review"] / dataset_sizes["review"]
    sample_dataset(
        os.path.join(input_dir, datasets["review"]),
        os.path.join(output_dir, datasets["review"]),
        review_ratio,
        "business_id",
        preserved_business_ids
    )
    
    # 步骤4: 采样tip数据集，保持与采样的business的关联
    tip_ratio = target_sizes["tip"] / dataset_sizes["tip"]
    sample_dataset(
        os.path.join(input_dir, datasets["tip"]),
        os.path.join(output_dir, datasets["tip"]),
        tip_ratio,
        "business_id",
        preserved_business_ids
    )
    
    # 步骤5: 采样checkin数据集，保持与采样的business的关联
    checkin_ratio = target_sizes["checkin"] / dataset_sizes["checkin"]
    sample_dataset(
        os.path.join(input_dir, datasets["checkin"]),
        os.path.join(output_dir, datasets["checkin"]),
        checkin_ratio,
        "business_id",
        preserved_business_ids
    )
    
    print("\n采样完成! 所有数据集已保存到:", output_dir)

if __name__ == "__main__":
    # 设置输入和输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    input_dir = os.path.join(parent_dir, "ds")
    output_dir = os.path.join(parent_dir, "ds_sampled")
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    # 创建采样数据集，目标总记录数为100万
    create_sampled_datasets(input_dir, output_dir, target_total=1000000)
    
    # 分析采样后的数据集
    print("\n采样后的数据集分析:")
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        print(f"  {filename}: {count} 条记录")