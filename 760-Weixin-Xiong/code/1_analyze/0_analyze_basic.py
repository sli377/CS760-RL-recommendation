import json
import os
from collections import defaultdict

def analyze_json_files(directory_path, files=None):
    """分析指定目录下指定JSON文件的行数和属性"""
    results = {}
    
    # 如果没有提供文件列表，则使用目录下所有的json文件
    if files is None:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # 尝试读取整个文件作为一个JSON对象
                try:
                    data = json.load(file)
                    # 如果是列表，计算列表中的项目数量
                    if isinstance(data, list):
                        line_count = len(data)
                        attributes = defaultdict(int)
                        
                        # 分析列表中第一个项目的属性（假设所有项目结构相似）
                        if line_count > 0 and isinstance(data[0], dict):
                            sample_item = data[0]
                            for key in sample_item.keys():
                                attributes[key] = line_count  # 假设所有项目都有相同的键
                        
                        # 存储结果
                        results[file_name] = {
                            'line_count': line_count,
                            'attributes': dict(attributes)
                        }
                        
                        # 打印当前文件的分析结果
                        print(f"\n文件: {file_name}")
                        print(f"数据项数量: {line_count}")
                        print("属性列表 (假设所有项目结构相似):")
                        for attr in sorted(attributes.keys()):
                            print(f"  - {attr}")
                    
                    # 如果是单个对象，分析对象的属性
                    elif isinstance(data, dict):
                        attributes = {key: 1 for key in data.keys()}
                        
                        # 存储结果
                        results[file_name] = {
                            'line_count': 1,
                            'attributes': attributes
                        }
                        
                        # 打印当前文件的分析结果
                        print(f"\n文件: {file_name}")
                        print("数据类型: 单个JSON对象")
                        print("属性列表:")
                        for attr in sorted(attributes.keys()):
                            print(f"  - {attr}")
                    
                except json.JSONDecodeError:
                    # 如果不是单个JSON对象，尝试按行解析JSONL
                    file.seek(0)  # 重置文件指针到开始
                    line_count = 0
                    attributes = defaultdict(int)
                    
                    for line in file:
                        if line.strip():  # 跳过空行
                            line_count += 1
                            try:
                                # 解析JSON行
                                item_data = json.loads(line.strip())
                                # 统计属性出现次数
                                if isinstance(item_data, dict):
                                    for key in item_data.keys():
                                        attributes[key] += 1
                            except json.JSONDecodeError:
                                print(f"警告: 在文件 {file_name} 的第 {line_count} 行发现无效JSON")
                                continue
                    
                    # 存储结果
                    results[file_name] = {
                        'line_count': line_count,
                        'attributes': dict(attributes)
                    }
                    
                    # 打印当前文件的分析结果
                    print(f"\n文件: {file_name}")
                    print(f"总行数: {line_count}")
                    print("属性统计:")
                    for attr, count in sorted(attributes.items()):
                        percentage = (count / line_count) * 100 if line_count > 0 else 0
                        print(f"  - {attr}: 出现 {count} 次 ({percentage:.2f}%)")
                
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    return results

if __name__ == "__main__":
    # 直接在代码中指定目录和文件，不需要命令行参数
    directory = "ds"  # 相对路径，假设脚本在 760/code/1_analyze 目录下
    
    # 你的特定文件列表
    yelp_files = [
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_checkin.json",
        "yelp_academic_dataset_review.json",
        "yelp_academic_dataset_tip.json",
        "yelp_academic_dataset_user.json"
    ]
    
    # 调整目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
    parent_dir = os.path.dirname(os.path.dirname(script_dir))  # 获取父目录的父目录 (760)
    ds_dir = os.path.join(parent_dir, "ds")  # 构建ds目录的完整路径
    
    print(f"正在分析目录: {ds_dir}")
    analyze_json_files(ds_dir, yelp_files)