#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import datetime
import os

# 设置输入和输出文件路径
INPUT_FILE_PATH = "./yelp-dataset/yelp_academic_dataset_review.json"  # 您的输入文件路径
OUTPUT_FILE_PATH = "filtered_2019_data.json"  # 输出文件路径

def filter_2019_data_jsonl(input_file_path, output_file_path):
    """
    从JSONL文件(每行一个JSON对象)中读取数据，并筛选出日期为2019年的数据
    
    参数:
    input_file_path -- 输入JSONL文件路径
    output_file_path -- 输出文件路径
    
    返回:
    只包含2019年数据的列表
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误：输入文件 '{input_file_path}' 不存在")
        return []
    
    # 用于存储筛选后的数据
    filtered_data = []
    # 记录处理的行数和有效的2019年数据数
    processed_lines = 0
    valid_2019_count = 0
    error_count = 0
    
    print(f"开始读取并处理文件: {input_file_path}")
    
    try:
        # 逐行读取和处理JSONL文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed_lines += 1
                
                # 每处理1000行打印一次进度
                if processed_lines % 1000 == 0:
                    print(f"已处理 {processed_lines} 行，发现 {valid_2019_count} 条2019年数据...")
                
                try:
                    # 解析当前行的JSON对象
                    entry = json.loads(line.strip())
                    
                    # 解析日期字符串
                    date_str = entry.get('date', '')
                    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    
                    # 检查年份是否为2019
                    if date_obj.year == 2019:
                        filtered_data.append(entry)
                        valid_2019_count += 1
                        
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"第 {processed_lines} 行JSON解析错误: {e}")
                    continue
                except (ValueError, AttributeError) as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"第 {processed_lines} 行日期解析错误: {e}")
                    continue
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return filtered_data
    
    if error_count > 5:
        print(f"... 共有 {error_count} 行出现错误，仅显示前5个")
    
    # 将筛选结果保存到文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print(f"已将筛选结果保存到文件: {output_file_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")
    
    print(f"处理完成: 共处理 {processed_lines} 行数据，筛选出 {valid_2019_count} 条2019年的数据")
    return filtered_data

# 主程序
if __name__ == "__main__":
    print(f"开始从文件 '{INPUT_FILE_PATH}' 筛选2019年数据...")
    filtered_data = filter_2019_data_jsonl(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
    print("程序执行完毕!")