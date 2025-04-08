import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np

def analyze_business_location_missing_values(file_path):
    """
    分析商家数据集中位置信息的缺失值情况
    
    参数:
    - file_path: 商家数据集文件路径
    
    返回:
    - 缺失值分析结果
    """
    print(f"正在分析商家位置信息: {file_path}")
    
    # 用于存储各项位置属性的情况
    location_attributes = ['address', 'city', 'state', 'postal_code', 'latitude', 'longitude']
    location_data = {attr: [] for attr in location_attributes}
    location_data['business_id'] = []
    
    # 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            location_data['business_id'].append(data['business_id'])
            
            # 检查每个位置属性
            for attr in location_attributes:
                if attr in data:
                    # 检查值是否为None、空字符串或NaN
                    value = data[attr]
                    if value is None or (isinstance(value, str) and value.strip() == '') or \
                       (isinstance(value, float) and (np.isnan(value) or value == 0.0)):
                        location_data[attr].append(None)
                    else:
                        location_data[attr].append(value)
                else:
                    location_data[attr].append(None)
    
    # 转换为DataFrame进行分析
    df = pd.DataFrame(location_data)
    total_records = len(df)
    
    # 计算每个属性的缺失率
    missing_counts = df[location_attributes].isna().sum()
    missing_rates = (missing_counts / total_records) * 100
    
    # 检查经纬度为0的情况（可能表示缺失或错误值）
    zero_lat = ((df['latitude'] == 0) | (df['latitude'].abs() < 0.001)).sum()
    zero_lng = ((df['longitude'] == 0) | (df['longitude'].abs() < 0.001)).sum()
    
    # 综合分析结果
    results = {
        'total_businesses': total_records,
        'missing_counts': missing_counts.to_dict(),
        'missing_rates': missing_rates.to_dict(),
        'zero_coordinates': {
            'latitude_zero_count': zero_lat,
            'latitude_zero_rate': (zero_lat / total_records) * 100,
            'longitude_zero_count': zero_lng,
            'longitude_zero_rate': (zero_lng / total_records) * 100
        }
    }
    
    # 检查同时缺失多个位置属性的情况
    all_location_missing = df[['city', 'state', 'latitude', 'longitude']].isna().all(axis=1).sum()
    results['businesses_with_no_location'] = {
        'count': all_location_missing,
        'rate': (all_location_missing / total_records) * 100
    }
    
    # 检查经纬度和地址信息的一致性
    address_missing_but_coords_present = ((df['address'].isna()) & (~df['latitude'].isna()) & (~df['longitude'].isna())).sum()
    coords_missing_but_address_present = ((~df['address'].isna()) & (df['latitude'].isna() | df['longitude'].isna())).sum()
    
    results['inconsistent_location_data'] = {
        'address_missing_but_coords_present': {
            'count': address_missing_but_coords_present,
            'rate': (address_missing_but_coords_present / total_records) * 100
        },
        'coords_missing_but_address_present': {
            'count': coords_missing_but_address_present,
            'rate': (coords_missing_but_address_present / total_records) * 100
        }
    }
    
    return results

def print_missing_value_analysis(results):
    """打印缺失值分析结果"""
    print("\n===== 商家位置信息缺失值分析 =====")
    print(f"总商家数量: {results['total_businesses']}")
    
    print("\n各位置属性缺失情况:")
    for attr, count in results['missing_counts'].items():
        rate = results['missing_rates'][attr]
        print(f"  - {attr}: {count}个缺失 ({rate:.2f}%)")
    
    print("\n经纬度为零或接近零的情况:")
    zero_lat = results['zero_coordinates']['latitude_zero_count']
    zero_lat_rate = results['zero_coordinates']['latitude_zero_rate']
    zero_lng = results['zero_coordinates']['longitude_zero_count']
    zero_lng_rate = results['zero_coordinates']['longitude_zero_rate']
    print(f"  - 纬度(latitude)为零: {zero_lat}个 ({zero_lat_rate:.2f}%)")
    print(f"  - 经度(longitude)为零: {zero_lng}个 ({zero_lng_rate:.2f}%)")
    
    print("\n完全没有位置信息的商家:")
    no_location_count = results['businesses_with_no_location']['count']
    no_location_rate = results['businesses_with_no_location']['rate']
    print(f"  - {no_location_count}个商家没有任何位置信息 ({no_location_rate:.2f}%)")
    
    print("\n位置数据不一致的情况:")
    addr_missing = results['inconsistent_location_data']['address_missing_but_coords_present']['count']
    addr_missing_rate = results['inconsistent_location_data']['address_missing_but_coords_present']['rate']
    coords_missing = results['inconsistent_location_data']['coords_missing_but_address_present']['count']
    coords_missing_rate = results['inconsistent_location_data']['coords_missing_but_address_present']['rate']
    print(f"  - 缺少地址但有坐标: {addr_missing}个 ({addr_missing_rate:.2f}%)")
    print(f"  - 有地址但缺少坐标: {coords_missing}个 ({coords_missing_rate:.2f}%)")

if __name__ == "__main__":
    # 设置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # 使用采样后的数据集
    business_file = os.path.join(parent_dir, "ds_sampled", "yelp_academic_dataset_business.json")
    
    # 分析商家位置信息的缺失值
    results = analyze_business_location_missing_values(business_file)
    
    # 打印分析结果
    print_missing_value_analysis(results)
    
    # 可选：保存分析结果到CSV文件
    output_file = os.path.join(parent_dir, "ds_sampled", "business_location_analysis.csv")
    missing_df = pd.DataFrame({
        'Attribute': list(results['missing_rates'].keys()),
        'Missing_Count': list(results['missing_counts'].values()),
        'Missing_Rate': list(results['missing_rates'].values())
    })
    missing_df.to_csv(output_file, index=False)
    print(f"\n分析结果已保存到: {output_file}")