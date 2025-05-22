import pandas as pd
import numpy as np
import torch
import argparse
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import torch.nn.init as init
import utils
from tqdm import tqdm
import random
import torch.nn.functional as F
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="SASRec CCQL.")

    parser.add_argument('--epoch',
                        type=int,
                        default=20,
                        help='Number of max epochs.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor',
                        type=int,
                        default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help='Learning rate.')
    parser.add_argument('--lr_2',
                        type=float,
                        default=0.001,
                        help='Learning rate for the second optimizer.')
    parser.add_argument('--discount',
                        type=float,
                        default=0.5,
                        help='Discount factor for RL.')
    parser.add_argument('--r_negative',
                        type=float,
                        default=-1.0,
                        help='reward for the negative behavior.')
    parser.add_argument('--neg_sample',
                        type=int,
                        default=10,
                        help='number of negative samples.')
    parser.add_argument('--cql_temp',
                        type=float,
                        default=0.5,
                        help='Temperature for contrastive loss')
    parser.add_argument('--cql_min_q_weight',
                        type=float,
                        default=0.01,
                        help='Minimum Q weight for conservative Q learning')
    parser.add_argument('--q_loss_weight',
                        type=float,
                        default=1.0,
                        help='Weight for Q learning loss in conservative Q learning')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--eval_interval', default=2000, type=int)
    parser.add_argument('--switch_interval', default=15000, type=int)
    parser.add_argument('--console', default=500, type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_id', type=str, default='SASRecCCQL')
    parser.add_argument('--contrastive_loss',
                        type=str,
                        default='InfoNCECosine')
    parser.add_argument('--aug', type=str, default='permutation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    return parser.parse_args()


def save_results(metrics, epoch, is_validation=True):
    """
    保存评估结果到文件
    """
    # 只创建一个结果目录
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    
    # 转换指标为可序列化的格式
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            serializable_metrics[key] = int(value)
        elif isinstance(value, (np.float64, np.float32, np.float16)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list):
            # 处理列表中的NumPy类型
            serializable_metrics[key] = [
                float(x) if isinstance(x, (np.float64, np.float32, np.float16)) else
                int(x) if isinstance(x, (np.int64, np.int32, np.int16, np.int8)) else
                x.tolist() if isinstance(x, np.ndarray) else x
                for x in value
            ]
        else:
            serializable_metrics[key] = value
    
    # 添加额外信息
    serializable_metrics['epoch'] = int(epoch)
    serializable_metrics['is_validation'] = is_validation
    
    # 保存到单个JSON文件
    filename = f'{result_dir}/metrics_{"val" if is_validation else "test"}_epoch_{epoch}.json'
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    return metrics


def plot_metrics(metrics_history, save_dir):
    """
    绘制训练过程中的所有指标变化，每个指标单独一张图
    metrics_history: 包含训练和验证指标的历史记录
    save_dir: 图表保存目录
    """
    # 提取epoch
    epochs = [m['epoch'] for m in metrics_history['val']]
    
    # 为每个指标创建单独的图表
    metrics = {
        'precision': ['precision_5', 'precision_10', 'precision_20'],
        'recall': ['recall_5', 'recall_10', 'recall_20'],
        'hr': ['hr_5', 'hr_10', 'hr_20'],
        'ndcg': ['ndcg_5', 'ndcg_10', 'ndcg_20']
    }
    
    # 创建可视化目录
    vis_dir = os.path.join(save_dir, 'plots')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 为三个不同的版本创建子目录
    for version in ['original', 'removed', 'adjusted']:
        version_dir = os.path.join(vis_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        for metric_type, metric_keys in metrics.items():
            plt.figure(figsize=(10, 6))
            for k in [5, 10, 20]:
                metric_key = f'{version}_{metric_type}_{k}'
                if metric_key in metrics_history['val'][0]['metrics']:
                    plt.plot(epochs, 
                            [m['metrics'][metric_key] for m in metrics_history['val']], 
                            label=f'@{k}', 
                            color=f'blue' if k==5 else 'green' if k==10 else 'red',
                            marker='o',  # 添加数据点标记
                            markersize=3)  # 设置标记大小
            
            # 设置图表标题和标签 (使用英文)
            title_map = {
                'precision': 'Precision',
                'recall': 'Recall',
                'hr': 'Hit Ratio (HR)',
                'ndcg': 'Normalized Discounted Cumulative Gain (NDCG)'
            }
            plt.title(f'{title_map.get(metric_type, metric_type.capitalize())} @{version.capitalize()}')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # 保存单个指标的图表
            save_path = os.path.join(version_dir, f'{metric_type}.png')
            plt.savefig(save_path)
            plt.close()


def evaluate(model, epoch, dataset, is_validation=True):
    model.eval()
    # 根据is_validation参数选择使用验证集还是测试集
    if is_validation:
        eval_sessions = pd.read_pickle(
            os.path.join(data_directory, 'sampled_valid.df'))
        print("\nEvaluating on validation set...")
    else:
        eval_sessions = pd.read_pickle(
            os.path.join(data_directory, 'sampled_test.df'))
        print("\nEvaluating on test set...")
    
    eval_ids = eval_sessions.user_id.unique()
    groups = eval_sessions.groupby('user_id')
    
    batch = 512
    evaluated = 0
    eval_start = time.time()
    
    # 初始化指标结果存储
    metrics = {
        'original': {k: {'precision': 0, 'recall': 0, 'hr': 0, 'ndcg': 0} for k in topk},
        'removed': {k: {'precision': 0, 'recall': 0, 'hr': 0, 'ndcg': 0} for k in topk},
        'adjusted': {k: {'precision': 0, 'recall': 0, 'hr': 0, 'ndcg': 0} for k in topk}
    }
    user_count = 0
    
    pbar = tqdm(total=len(eval_ids), desc='Evaluating')
    
    while evaluated < len(eval_ids):
        states, len_states, actions, ratings = [], [], [], []
        user_indices = []  # 保存用户索引
        
        for i in range(batch):
            if evaluated >= len(eval_ids):
                break
            
            user_id = eval_ids[evaluated]
            group = groups.get_group(user_id)
            history = []
            
            for idx, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = utils.pad_history(state, state_size, item_num)
                states.append(state)
                action = row['business_id']
                rating = row['rating']  # 保存原始评分
                actions.append(action)
                ratings.append(rating)
                history.append(row['business_id'])
                user_indices.append(evaluated)  # 记录用户索引
            
            evaluated += 1
            pbar.update(1)
        
        if not actions:  # 如果没有操作，继续下一批
            continue
            
        states = np.asarray(states)
        len_states = np.asarray(len_states)
        
        # 获取模型预测
        logits = model(states, len_states)
        predictions = model.output2(logits).detach().cpu().numpy()
        
        # 按用户ID分组处理
        user_predictions = {}
        user_histories = {}
        user_test_items = {}
        user_test_ratings = {}
        
        # 将结果按用户分组
        for i, user_idx in enumerate(user_indices):
            user_id = eval_ids[user_idx]
            
            if user_id not in user_predictions:
                user_predictions[user_id] = []
                user_histories[user_id] = []
                user_test_items[user_id] = []
                user_test_ratings[user_id] = []
            
            # 获取历史物品
            history_items = states[i][:len_states[i]]
            history_items = [item for item in history_items if item != item_num]
            
            user_predictions[user_id].append(predictions[i])
            user_histories[user_id].append(history_items)
            user_test_items[user_id].append(actions[i])
            user_test_ratings[user_id].append(ratings[i])
        
        # 为每个用户计算指标
        for user_id in user_predictions:
            # 对用户的所有预测取最大值，作为最终预测分数
            user_pred = np.zeros_like(user_predictions[user_id][0])
            for pred in user_predictions[user_id]:
                user_pred = np.maximum(user_pred, pred)
            
            # 获取用户的测试集物品和对应评分
            test_items = user_test_items[user_id]
            test_ratings = user_test_ratings[user_id]
            
            # 使用最后一个历史记录作为用户的完整历史
            history_items = user_histories[user_id][-1] if user_histories[user_id] else []
            
            # 创建物品-评分映射
            item_ratings = {}
            for item, rating in zip(test_items, test_ratings):
                item_ratings[item] = rating
            
            if len(test_items) == 0:
                continue
            
            # 原始推荐（不过滤历史物品）
            recommended_original = np.argsort(-user_pred)
            
            # 创建移除历史物品的分数副本
            user_pred_no_history = user_pred.copy()
            for item in history_items:
                if item < len(user_pred_no_history):
                    user_pred_no_history[item] = -np.inf
            
            # 移除历史物品后的推荐
            recommended_no_history = np.argsort(-user_pred_no_history)
            
            # 计算三种指标
            for k in topk:
                # 1. 原始版本（不过滤历史物品）
                top_k_original = recommended_original[:k]
                history_in_topk = sum(1 for item in top_k_original if item in history_items)
                history_in_test = sum(1 for item in test_items if item in history_items)
                
                # 2. 移除历史物品
                top_k_no_history = recommended_no_history[:k]
                
                # 3. 调整k值（适应性版本）
                adjusted_k = max(1, k - history_in_topk)
                adjusted_test_size = max(1, len(test_items) - history_in_test)
                top_k_adjusted = recommended_no_history[:adjusted_k]
                
                # 计算三个版本的指标
                versions = {
                    'original': {'top_k': top_k_original, 'k': k, 'test_size': len(test_items)},
                    'removed': {'top_k': top_k_no_history, 'k': k, 'test_size': len(test_items)},
                    'adjusted': {'top_k': top_k_adjusted, 'k': adjusted_k, 'test_size': adjusted_test_size}
                }
                
                for version_name, version_params in versions.items():
                    top_k_items = version_params['top_k']
                    current_k = version_params['k']
                    current_test_size = version_params['test_size']
                    
                    # 计算推荐命中的物品
                    if version_name == 'original':
                        # 原始版本：包括历史物品
                        hit_items = [item for item in top_k_items if item in test_items]
                    else:
                        # 其他版本：排除历史物品
                        hit_items = [item for item in top_k_items 
                                    if item in test_items and item not in history_items]
                    
                    hit_count = len(hit_items)
                    
                    # Precision@K
                    if current_k > 0:
                        metrics[version_name][k]['precision'] += hit_count / current_k
                    
                    # Recall@K
                    if current_test_size > 0:
                        metrics[version_name][k]['recall'] += hit_count / current_test_size
                    
                    # HR@K
                    metrics[version_name][k]['hr'] += 1 if hit_count > 0 else 0
                    
                    # NDCG@K
                    dcg = 0
                    idcg = 0
                    
                    # 计算DCG
                    for i, item in enumerate(top_k_items):
                        if item in test_items and (version_name == 'original' or item not in history_items):
                            rel = item_ratings.get(item, 0)
                            dcg += (2**rel - 1) / np.log2(i + 2)
                    
                    # 计算IDCG
                    relevant_items = []
                    for item in test_items:
                        if version_name == 'original' or item not in history_items:
                            relevant_items.append((item, item_ratings.get(item, 0)))
                    
                    # 按评分从高到低排序
                    relevant_items.sort(key=lambda x: x[1], reverse=True)
                    
                    for i in range(min(len(relevant_items), current_k)):
                        rel = relevant_items[i][1]
                        idcg += (2**rel - 1) / np.log2(i + 2)
                    
                    if idcg > 0:
                        metrics[version_name][k]['ndcg'] += dcg / idcg
            
            user_count += 1
    
    pbar.close()
    eval_total_time = time.time() - eval_start
    
    # 计算平均指标
    result_metrics = {}
    result_metrics['eval_time'] = eval_total_time
    result_metrics['user_count'] = user_count
    
    print('#############################################################')
    print(f'Total users evaluated: {user_count}')
    
    # 打印三种方式的指标并保存
    for version in ['original', 'removed', 'adjusted']:
        print(f'\n{version.upper()} Version Metrics:')
        for k in topk:
            # 计算平均值
            precision_k = metrics[version][k]['precision'] / user_count if user_count > 0 else 0
            recall_k = metrics[version][k]['recall'] / user_count if user_count > 0 else 0
            hr_k = metrics[version][k]['hr'] / user_count if user_count > 0 else 0
            ndcg_k = metrics[version][k]['ndcg'] / user_count if user_count > 0 else 0
            
            # 保存指标 - 使用标准命名格式
            result_metrics[f'{version}_precision_{k}'] = float(precision_k)
            result_metrics[f'{version}_recall_{k}'] = float(recall_k)
            result_metrics[f'{version}_hr_{k}'] = float(hr_k)
            result_metrics[f'{version}_ndcg_{k}'] = float(ndcg_k)
            
            # 打印指标
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'Metrics @ {k}:')
            print(f'Precision@{k}: {precision_k:.4f}')
            print(f'Recall@{k}: {recall_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')
    
    print('#############################################################')
    
    # 保存结果
    save_results(result_metrics, epoch, is_validation)
    
    return result_metrics


class SASRecnetwork(torch.nn.Module):

    def __init__(self, hidden_size, learning_rate, item_num, state_size,
                 batch_size, device):
        super(SASRecnetwork, self).__init__()
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.batch_size = batch_size
        self.is_training = torch.BoolTensor()
        self.dev = device

        self.state_embeddings = torch.nn.Embedding(self.item_num + 1,
                                                   self.hidden_size)
        # Positional Encoding
        self.pos_embeddings = torch.nn.Embedding(self.state_size,
                                                 self.hidden_size)

        # Initialize the weights of the Embedding layers
        init.normal_(self.state_embeddings.weight, mean=0.0, std=0.01)
        init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.01)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # to be Q for self-attention
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)

        # Build Blocks
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = utils.MultiheadAttention(self.hidden_size,
                                                args.dropout_rate,
                                                num_heads=args.num_heads,
                                                device=self.dev,
                                                causality=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = utils.PointWiseFeedForward(self.hidden_size,
                                                 args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # Initialize the weights of the MultiheadAttention and PointWiseFeedForward layers
        for attn_layer, fwd_layer in zip(self.attention_layers,
                                         self.forward_layers):
            # Initialize weights of MultiheadAttention layer
            init.normal_(attn_layer.Q.weight, mean=0.0, std=0.01)
            init.zeros_(attn_layer.Q.bias)
            init.normal_(attn_layer.K.weight, mean=0.0, std=0.01)
            init.zeros_(attn_layer.K.bias)
            init.normal_(attn_layer.V.weight, mean=0.0, std=0.01)
            init.zeros_(attn_layer.V.bias)
            # Initialize weights of PointWiseFeedForward layer
            init.normal_(fwd_layer.conv1.weight, mean=0.0, std=0.01)
            init.zeros_(fwd_layer.conv1.bias)
            init.normal_(fwd_layer.conv2.weight, mean=0.0, std=0.01)
            init.zeros_(fwd_layer.conv2.bias)

        self.output1 = torch.nn.Linear(self.hidden_size, self.item_num)
        self.output2 = torch.nn.Linear(self.hidden_size, self.item_num)

        # Initialize the weights of the Linear layers
        init.normal_(self.output1.weight, mean=0.0, std=0.01)
        init.zeros_(self.output1.bias)
        init.normal_(self.output2.weight, mean=0.0, std=0.01)
        init.zeros_(self.output2.bias)

        self.celoss1 = torch.nn.CrossEntropyLoss()
        self.celoss2 = torch.nn.CrossEntropyLoss()

        self.opt = torch.optim.Adam(self.parameters(),
                                    lr=args.lr,
                                    betas=(0.9, 0.999))

        self.opt2 = torch.optim.Adam(self.parameters(),
                                     lr=args.lr_2,
                                     betas=(0.9, 0.999))

        self.best_ckpt_tracker = utils.Tracker()

    def forward(self, state_seq, len_state):
        seqs = self.state_embeddings(torch.LongTensor(state_seq).to(self.dev))
        seqs *= self.state_embeddings.embedding_dim**0.5
        positions = np.tile(np.array(range(state_seq.shape[1])),
                            [state_seq.shape[0], 1])
        seqs += self.pos_embeddings(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = ~torch.BoolTensor(state_seq == self.item_num).to(
            self.dev)
        # broadcast in last dim
        seqs *= timeline_mask.unsqueeze(-1)

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= timeline_mask.unsqueeze(-1)

        # (U, T, C) -> (U, -1, C)
        seqs = self.last_layernorm(seqs)

        layer_slices = []
        for b_index, len_s in enumerate(len_state - 1):
            last_layer_norm_slice = seqs[b_index, len_s, :]
            layer_slices.append(last_layer_norm_slice)

        model_output = torch.stack(layer_slices)

        return model_output

    def double_qlearning(self, q_vals_state, actions, rewards, discount,
                         q_vals_next_state, q_vals_next_state_selector):
        """ Double-Q operator.
          Args:
            q_vals_state: Tensor holding Q-values for s in a batch of transitions,
                shape `[B x num_actions]`.
            actions: Tensor holding action indices, shape `[B]`.
            rewards: Tensor holding rewards, shape `[B]`.
            discount: Tensor holding pcontinue values, shape `[B]`.
            q_vals_next_state: Tensor of Q-values for s' in a batch of transitions,
                used to estimate the value of the best action, shape `[B x num_actions]`.
            q_vals_next_state_selector: Tensor of Q-values for s' in a batch of
                transitions used to estimate the best action, shape `[B x num_actions]`.
        """

        with torch.no_grad():
            # Build target and select head to update.
            best_action = torch.argmax(q_vals_next_state_selector, 1)
            double_q_bootstrapped = self.batched_index(q_vals_next_state,
                                                       best_action)
            target = rewards + discount * double_q_bootstrapped

        q_pred = self.batched_index(q_vals_state, actions)
        # Temporal difference error and loss.
        # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
        td_error = target - q_pred
        loss = 0.5 * torch.square(td_error)

        return loss, q_pred, best_action

    def batched_index(self, values, indices):
        """Equivalent to `values[:, indices]` or tf.gather`.
        """
        one_hot_indices = torch.nn.functional.one_hot(
            indices, num_classes=self.item_num)
        sum_vals = torch.sum(values * one_hot_indices, dim=-1)
        return sum_vals


class SASRecnetworkConservative():

    def __init__(self, args, device):

        self.qf1 = SASRecnetwork(hidden_size=args.hidden_factor,
                                 learning_rate=args.lr,
                                 item_num=item_num,
                                 state_size=state_size,
                                 batch_size=args.batch_size,
                                 device=device)

        self.qf2 = SASRecnetwork(hidden_size=args.hidden_factor,
                                 learning_rate=args.lr,
                                 item_num=item_num,
                                 state_size=state_size,
                                 batch_size=args.batch_size,
                                 device=device)

        self.qf1 = self.qf1.to(device)
        self.qf2 = self.qf2.to(device)

        self.cql_temp = args.cql_temp
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        self.cql_min_q_weight = args.cql_min_q_weight
        self.q_loss_weight = args.q_loss_weight

        self.batch_size = args.batch_size

        self.reward_negative = args.r_negative
        self.negative_samples = args.neg_sample

    def contrastive_loss(self, mainQN, current_state_logits, states,
                         len_states, total_step, console):

        if args.aug == 'permutation':
            rng = np.random.default_rng()
            states_permuted = rng.permuted(states, axis=1)
            logits_aug_state = mainQN(states_permuted, len_states)
        elif args.aug == 'crop':
            cropped_seq = utils.mask_crop_2d_array(states, pad_item)
            logits_aug_state = mainQN(cropped_seq, len_states)
        elif args.aug == 'mask':
            masked_seq = utils.mask_2d_array(states, pad_item)
            logits_aug_state = mainQN(masked_seq, len_states)
        else:
            raise ValueError('Invalid augmentation.')

        if args.contrastive_loss == "InfoNCECosine":
            contrastive_loss = utils.info_nce_cosim_loss(
                current_state_logits, logits_aug_state, total_step, console)
        elif args.contrastive_loss == "InfoNCE":
            current_state_logits = current_state_logits.unsqueeze(-1)
            logits_aug_state = logits_aug_state.unsqueeze(-1)
            contrastive_loss = utils.contrastive_loss(current_state_logits,
                                                      logits_aug_state)
        elif args.contrastive_loss == "Hierarchical":
            current_state_logits = current_state_logits.unsqueeze(-1)
            logits_aug_state = logits_aug_state.unsqueeze(-1)
            contrastive_loss = utils.hierarchical_contrastive_loss(
                current_state_logits, logits_aug_state)

        return contrastive_loss

    def cql_loss(self, action, target_Qs, q_tm1, q_pred, discounts, target_Qs_selector):
        neg_reward = torch.full((self.batch_size, ),
                                self.reward_negative).to(device)
        negative_actions = []

        for index in range(target_Qs.shape[0]):
            negative_list = []
            for i in range(self.negative_samples):
                neg = np.random.randint(item_num)
                while neg == action[index]:
                    neg = np.random.randint(item_num)
                negative_list.append(neg)
            negative_actions.append(negative_list)
        negative_actions = torch.Tensor(
            np.asarray(negative_actions)).long().to(device)

        neg_act_predictions = []

        for i in range(self.negative_samples):
            negatives = negative_actions[:, i]
            _, q_pred_neg, action_preds_neg = self.qf1.double_qlearning(
                q_tm1, negatives, neg_reward, discounts, target_Qs,
                target_Qs_selector)
            neg_act_predictions.append(action_preds_neg)
        # Feed-in action predictions.
        neg_act_predictions = torch.cat(neg_act_predictions).reshape(
            target_Qs.shape[0], self.negative_samples)
        _, q_pred_neg, cql_q1_current_actions = self.qf1.double_qlearning(
            q_tm1, action_preds_neg, neg_reward, discounts, target_Qs,
            target_Qs_selector)

        # Conservative Q-Learning
        q_pred_neg = torch.unsqueeze(q_pred_neg, 1)
        q_pred = torch.unsqueeze(q_pred, 1)
        cql_q1_current_actions = torch.unsqueeze(cql_q1_current_actions, 1)
        cql_cat_q1 = torch.cat([q_pred_neg, q_pred, cql_q1_current_actions],
                               dim=1)

        cql_std = torch.std(cql_cat_q1, dim=1)

        cql_qf_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp,
                                     dim=1) * self.cql_temp
        """Subtract the log likelihood of data"""
        cql_qf_diff = torch.clamp(
            cql_qf_ood - q_pred,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        cql_min_qf_loss = cql_qf_diff * self.cql_min_q_weight

        return cql_min_qf_loss, cql_std

    def train(self, mainQN, target_QN, batch, total_step):
        # 从batch中提取所需的数据
        state = list(batch['state'].values())
        len_state = list(batch['len_state'].values())
        action = list(batch['action'].values())
        reward = list(batch['reward'].values())

        # 将reward归一化到0-1区间
        min_rating = 1.0  # Yelp评分最小值为1
        max_rating = 5.0  # Yelp评分最大值为5
        normalized_rewards = [(r - min_rating) / (max_rating - min_rating) for r in reward]

        next_state = list(batch['next_state'].values())
        len_next_state = list(batch['len_next_states'].values())
        next_states = np.asarray(next_state)
        len_next_states = np.asarray(len_next_state)

        discount = [args.discount] * len(action)

        states = np.asarray(state)
        len_states = np.asarray(len_state)
        actions = torch.Tensor(np.asarray(action)).long().to(device)
        rewards = torch.Tensor(np.asarray(normalized_rewards)).to(device)
        discounts = torch.Tensor(np.asarray(discount)).to(device)

        target_model_output = target_QN(next_states, len_next_states)
        target_Qs = target_QN.output1(target_model_output)

        main_model_output = mainQN(next_states, len_next_states)
        target_Qs_selector = mainQN.output1(main_model_output)

        clip_value = 1.0

        # Set target_Qs to 0 for states where episode ends
        is_done = list(batch['is_done'].values())
        for index in range(target_Qs.shape[0]):
            if is_done[index]:
                target_Qs[index] = torch.Tensor(np.zeros([item_num])).to(device)

        if total_step < switch_interval:
            main_model_current_state = mainQN(states, len_states)
            q_tm1 = mainQN.output1(main_model_current_state)

            q_loss, q_pred, _ = mainQN.double_qlearning(
                q_tm1, actions, rewards, discounts, target_Qs,
                target_Qs_selector)

            predictions = mainQN.output2(main_model_current_state)

            contrastive_loss = self.contrastive_loss(mainQN,
                                                     main_model_current_state,
                                                     states, len_states,
                                                     total_step, console)

            mainQN.opt.zero_grad()
            loss = mainQN.celoss1(predictions, actions)

            cql_min_qf_loss, cql_std = self.cql_loss(action, target_Qs, q_tm1,
                                                     q_pred, discounts,
                                                     target_Qs_selector)

            final_loss = torch.mean(loss + (self.q_loss_weight*q_loss) + cql_min_qf_loss +
                                    contrastive_loss)
            final_loss.backward()
            for param in mainQN.parameters():
                param.grad.data.clamp_(-clip_value, clip_value)

            mainQN.opt.step()

        else:
            main_model_current_state = mainQN(states, len_states)
            q_tm1 = mainQN.output1(main_model_current_state)

            q_loss, q_pred, _ = mainQN.double_qlearning(
                q_tm1, actions, rewards, discounts, target_Qs,
                target_Qs_selector)

            predictions = mainQN.output2(main_model_current_state)

            with torch.no_grad():
                q_indexed = mainQN.batched_index(q_tm1, actions)

            contrastive_loss = self.contrastive_loss(mainQN,
                                                     main_model_current_state,
                                                     states, len_states,
                                                     total_step, console)

            celoss2 = mainQN.celoss2(predictions, actions)
            loss_multi = torch.multiply(q_indexed, celoss2)

            cql_min_qf_loss, cql_std = self.cql_loss(action, target_Qs, q_tm1,
                                                     q_pred, discounts,
                                                     target_Qs_selector)

            mainQN.opt2.zero_grad()
            final_loss = torch.mean(loss_multi + (self.q_loss_weight*q_loss) + cql_min_qf_loss +
                                    contrastive_loss)
            final_loss.backward()
            for param in mainQN.parameters():
                param.grad.data.clamp_(-clip_value, clip_value)

            mainQN.opt2.step()
        
        return final_loss


def save_test_results_table(test_metrics_dict, save_dir):
    """
    将所有测试集结果保存为表格形式并生成比较图表
    
    Args:
        test_metrics_dict: 包含不同模型测试集结果的字典
        save_dir: 保存结果的目录
    """
    # 创建测试结果目录
    test_summary_dir = os.path.join(save_dir, 'test_summary')
    os.makedirs(test_summary_dir, exist_ok=True)
    
    # 总的DataFrame用于后续可视化
    all_data = {
        'Model': [],
        'Metric': [],
        'Version': [],  # 添加版本信息
        '@5': [],
        '@10': [],
        '@20': []
    }
    
    # 指标名称映射
    metrics_names = {
        'precision': 'Precision',
        'recall': 'Recall',
        'hr': 'HR',
        'ndcg': 'NDCG'
    }
    
    # 为三个版本分别创建子目录和结果表
    for version in ['original', 'removed', 'adjusted']:
        version_dir = os.path.join(test_summary_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # 创建表格数据
        version_table_data = {
            'Model': [],
            'Metric': [],
            '@5': [],
            '@10': [],
            '@20': []
        }
        
        # 整理数据成表格格式，只包含当前版本的模型
        for model_name, metrics in test_metrics_dict.items():
            # 只处理包含当前版本的模型
            if version.lower() in model_name.lower():
                for metric_type in ['precision', 'recall', 'hr', 'ndcg']:
                    version_table_data['Model'].append(model_name)
                    version_table_data['Metric'].append(metrics_names[metric_type])
                    
                    # 同时添加到总的DataFrame
                    all_data['Model'].append(model_name)
                    all_data['Metric'].append(metrics_names[metric_type])
                    all_data['Version'].append(version)
                    
                    for k in [5, 10, 20]:
                        key = f"{version}_{metric_type}_{k}"
                        value = metrics.get(key, 0)
                        version_table_data[f'@{k}'].append(f"{value:.4f}")
                        all_data[f'@{k}'].append(f"{value:.4f}")
        
        # 保存为CSV格式
        import pandas as pd
        df = pd.DataFrame(version_table_data)
        csv_path = os.path.join(version_dir, 'test_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {version} test results table to: {csv_path}")
        
        # 保存为格式化的文本表格
        txt_path = os.path.join(version_dir, 'test_results.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Test Results Summary - {version.capitalize()}\n")
            f.write("===================\n\n")
            
            # 为每个模型创建一个部分
            for model_name in set(version_table_data['Model']):
                f.write(f"{model_name}\n")
                f.write("-" * len(model_name) + "\n")
                
                # 表头
                f.write(f"{'Metric':<10} {'@5':<10} {'@10':<10} {'@20':<10}\n")
                
                # 获取该模型的所有指标
                model_rows = [(i, row) for i, row in enumerate(version_table_data['Model']) if row == model_name]
                for idx, _ in model_rows:
                    metric = version_table_data['Metric'][idx]
                    val_5 = version_table_data['@5'][idx]
                    val_10 = version_table_data['@10'][idx]
                    val_20 = version_table_data['@20'][idx]
                    f.write(f"{metric:<10} {val_5:<10} {val_10:<10} {val_20:<10}\n")
                
                f.write("\n")
        
        # 创建JSON格式
        json_path = os.path.join(version_dir, 'test_results.json')
        json_data = {}
        
        for model_name, metrics in test_metrics_dict.items():
            # 只处理包含当前版本的模型
            if version.lower() in model_name.lower():
                json_data[model_name] = {}
                for metric_type in ['precision', 'recall', 'hr', 'ndcg']:
                    json_data[model_name][metric_type] = {}
                    for k in [5, 10, 20]:
                        key = f"{version}_{metric_type}_{k}"
                        json_data[model_name][metric_type][f'@{k}'] = metrics.get(key, 0)
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Saved {version} JSON test results to: {json_path}")
    
    # 返回全部数据的DataFrame用于可视化
    return pd.DataFrame(all_data)


def visualize_test_results(df, save_dir):
    """
    创建测试结果的可视化
    
    Args:
        df: 包含测试结果的DataFrame
        save_dir: 保存图表的目录
    """
    # 创建测试结果可视化目录
    test_vis_dir = os.path.join(save_dir, 'test_summary')
    os.makedirs(test_vis_dir, exist_ok=True)
    
    # 为三个版本分别创建可视化
    for version in ['original', 'removed', 'adjusted']:
        version_dir = os.path.join(test_vis_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # 获取该版本的数据
        version_df = df[df['Model'].str.contains(version, case=False)]  # 过滤特定版本的数据
        
        if version_df.empty:
            print(f"No data found for version: {version}")
            continue
        
        # 准备绘图
        models = version_df['Model'].unique()
        metrics = version_df['Metric'].unique()
        
        # 为每个指标创建一个柱状图，比较不同模型
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # 过滤出当前指标的数据
            metric_df = version_df[version_df['Metric'] == metric]
            
            # 设置柱状图的位置
            bar_width = 0.2
            r1 = np.arange(len(models))
            r2 = [x + bar_width for x in r1]
            r3 = [x + bar_width for x in r2]
            
            # 绘制三个K值的柱状图
            plt.bar(r1, metric_df['@5'], width=bar_width, label='@5', color='blue')
            plt.bar(r2, metric_df['@10'], width=bar_width, label='@10', color='green')
            plt.bar(r3, metric_df['@20'], width=bar_width, label='@20', color='red')
            
            # 添加标签和图例 (使用英文)
            plt.xlabel('Model')
            plt.xticks([r + bar_width for r in range(len(models))], models)
            plt.ylabel(f'{metric}')
            plt.title(f'{metric} @{version.capitalize()}')
            plt.legend()
            
            # 保存图表
            save_path = os.path.join(version_dir, f'test_{metric.lower()}_comparison.png')
            plt.savefig(save_path)
            plt.close()
        
        # 创建一个汇总图表，展示@20的所有指标
        plt.figure(figsize=(12, 6))
        
        # 准备数据 - 只使用@20的数据
        metric_values = []
        model_names = []
        metric_names = []
        
        for model in models:
            for metric in metrics:
                model_metric_df = version_df[(version_df['Model'] == model) & (version_df['Metric'] == metric)]
                if not model_metric_df.empty:
                    metric_values.append(float(model_metric_df['@20'].values[0]))
                    model_names.append(model)
                    metric_names.append(metric)
        
        # 创建分组柱状图
        bar_width = 0.2
        positions = np.arange(len(metrics))
        
        for i, model in enumerate(models):
            model_data = [float(version_df[(version_df['Model'] == model) & (version_df['Metric'] == metric)]['@20'].values[0]) 
                         for metric in metrics]
            plt.bar(positions + i*bar_width, model_data, width=bar_width, label=model)
        
        plt.xlabel('Metric')
        plt.ylabel('Value @20')
        plt.title(f'Test Results @20 @{version.capitalize()}')
        plt.xticks(positions + bar_width * (len(models) - 1) / 2, metrics)
        plt.legend()
        
        # 保存图表
        summary_path = os.path.join(version_dir, 'test_results_summary_20.png')
        plt.savefig(summary_path)
        plt.close()
        
        print(f"Saved {version} test results visualizations to: {version_dir}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 记录训练开始时间
    training_start_time = datetime.datetime.now()
    print(f'\nTraining started at: {training_start_time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Network parameters
    args = parse_args()

    data_directory = 'Yelp/'
    data_statis = pd.read_pickle(os.path.join(data_directory,
                                              'data_statis.df'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    pad_item = item_num

    topk = [5, 10, 20]

    console = args.console
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_interval = args.eval_interval
    switch_interval = args.switch_interval
    
    # 全局变量定义
    contrastive_objective = args.contrastive_loss
    augmentation = args.aug
    total_step = 0

    model = SASRecnetworkConservative(args, device)
    
    replay_buffer = pd.read_pickle(
        os.path.join(data_directory, 'replay_buffer.df'))

    total_step = 0
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)
    model_parameters = filter(lambda p: p.requires_grad,
                              model.qf1.parameters())
    total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Total number of parameters : ', total_parameters * 2)
    print('Model : SASRec Contrastive CQL')
    print('Dataset : Yelp')
    print('Experiment ID', args.exp_id)
    print('Seed: ', args.seed)
    print('Hyperparams: ')
    print('##############################################')
    print('Batch_size: ', args.batch_size)
    print('Hidden_size: ', args.hidden_factor)
    print('Learning Rate: ', args.lr)
    print('Discount: ', args.discount)
    print('Contrastive Loss: ', args.contrastive_loss)
    print('Augmentation : ', args.aug)
    print('Negative Reward: ', args.r_negative)
    print('Negative Samples: ', args.neg_sample)
    print('CQL Temperature: ', args.cql_temp)
    print('CQL Min Q Weight: ', args.cql_min_q_weight)
    print('Q Loss Weight: ', args.q_loss_weight)
    print('##############################################')

    # 在主循环开始前添加最佳指标记录
    # 用于存储训练过程中的指标
    metrics_history = {
        'val': []     # 验证集指标
    }
    
    # 记录不同指标的最佳模型
    best_models = {
        'precision': {
            'metric': float('-inf'),
            'state': None,
            'epoch': 0
        },
        'recall': {
            'metric': float('-inf'),
            'state': None,
            'epoch': 0
        },
        'hr': {
            'metric': float('-inf'),
            'state': None,
            'epoch': 0
        },
        'ndcg': {
            'metric': float('-inf'),
            'state': None,
            'epoch': 0
        }
    }
    
    # 创建checkpoints目录
    checkpoint_dir = 'checkpoints_normalized'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 如果指定了断点续训
    if args.resume is not None:
        print(f'\nResuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.qf1.load_state_dict(checkpoint['model_state_dict'])
        model.qf2.load_state_dict(checkpoint['model_state_dict'])
        model.qf1.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        model.qf2.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        total_step = checkpoint['step']
        # 尝试兼容旧版本的checkpoint
        if 'metric_type' in checkpoint and checkpoint['metric_type'] == 'precision':
            best_models['precision']['metric'] = checkpoint['val_metric']
            best_models['precision']['state'] = checkpoint['model_state_dict']
            best_models['precision']['epoch'] = total_step
        
        # 加载指标历史记录
        if 'metrics_history' in checkpoint:
            metrics_history = checkpoint['metrics_history']
            print(f'Loaded metrics history up to step {total_step}')
        else:
            print('No metrics history found in checkpoint, starting fresh')
            # 对当前模型进行一次评估
            print('Evaluating current model state...')
            val_metrics = evaluate(model.qf1, total_step, 'Yelp', is_validation=True)
            metrics_history['val'].append({
                'epoch': total_step,
                'metrics': val_metrics
            })
        
        print(f'Resumed from step {total_step}')
        print(f'Best validation metric (precision): {best_models["precision"]["metric"]}')
    else:
        print('Initial Evaluation.')
        # 初始验证集评估
        val_metrics = evaluate(model.qf1, total_step, 'Yelp', is_validation=True)
        metrics_history['val'].append({
            'epoch': total_step,
            'metrics': val_metrics
        })
    
    # 训练循环
    for epoch in range(1, args.epoch + 1):
        print(f"\nEpoch {epoch}/{args.epoch}")
        model.qf1.train()  # 设置为训练模式
        model.qf2.train()  # 设置为训练模式
        
        # 训练一个epoch
        for _ in range(num_batches):
            total_step += 1
            
            # 获取一个批次的数据
            batch = replay_buffer.sample(n=args.batch_size).to_dict()
            
            # 随机选择使用哪个网络作为主网络
            pointer = np.random.randint(0, 2)
            if pointer == 0:
                mainQN = model.qf1
                target_QN = model.qf2
            else:
                mainQN = model.qf2
                target_QN = model.qf1
            
            model.train(mainQN, target_QN, batch, total_step)
            
            # 每100步更新一次目标网络
            if total_step % 100 == 0:
                model.qf2.load_state_dict(model.qf1.state_dict())
        
        # 每个epoch结束后评估
        if epoch % 1 == 0:  # 每个epoch都评估
            val_metrics = evaluate(model.qf1, epoch, 'Yelp', is_validation=True)
            
            # 保存指标
            metrics_history['val'].append({
                'epoch': epoch,
                'metrics': val_metrics
            })
            
            # 检查是否是验证集上的最佳表现
            for version in ['original', 'removed', 'adjusted']:
                # 获取当前版本的指标
                current_precision = val_metrics[f'{version}_precision_20']
                current_recall = val_metrics[f'{version}_recall_20']
                current_hr = val_metrics[f'{version}_hr_20']
                current_ndcg = val_metrics[f'{version}_ndcg_20']
                
                # 为不同版本分别创建模型目录
                version_checkpoint_dir = os.path.join(checkpoint_dir, version)
                os.makedirs(version_checkpoint_dir, exist_ok=True)
                
                # 更新最佳precision模型
                if current_precision > best_models['precision']['metric']:
                    best_models['precision']['metric'] = current_precision
                    best_models['precision']['state'] = model.qf1.state_dict()
                    best_models['precision']['epoch'] = epoch
                    best_models['precision']['version'] = version
                    # 保存最佳precision模型
                    checkpoint_path = os.path.join(version_checkpoint_dir, 
                        f'best_precision_epoch_{epoch}_metric_{current_precision:.4f}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_models['precision']['state'],
                        'val_metric': current_precision,
                        'optimizer_state_dict': model.qf1.opt.state_dict(),
                        'optimizer2_state_dict': model.qf1.opt2.state_dict() if hasattr(model.qf1, 'opt2') else None,
                        'metrics_history': metrics_history,
                        'metric_type': 'precision',
                        'metric_value': current_precision,
                        'metric_version': version,
                        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, checkpoint_path)
                    print(f'\nSaved best {version} precision model checkpoint: {os.path.basename(checkpoint_path)}')
                
                # 更新最佳recall模型
                if current_recall > best_models['recall']['metric']:
                    best_models['recall']['metric'] = current_recall
                    best_models['recall']['state'] = model.qf1.state_dict()
                    best_models['recall']['epoch'] = epoch
                    best_models['recall']['version'] = version
                    # 保存最佳recall模型
                    checkpoint_path = os.path.join(version_checkpoint_dir, 
                        f'best_recall_epoch_{epoch}_metric_{current_recall:.4f}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_models['recall']['state'],
                        'val_metric': current_recall,
                        'optimizer_state_dict': model.qf1.opt.state_dict(),
                        'optimizer2_state_dict': model.qf1.opt2.state_dict() if hasattr(model.qf1, 'opt2') else None,
                        'metrics_history': metrics_history,
                        'metric_type': 'recall',
                        'metric_value': current_recall,
                        'metric_version': version,
                        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, checkpoint_path)
                    print(f'\nSaved best {version} recall model checkpoint: {os.path.basename(checkpoint_path)}')
                
                # 更新最佳HR模型
                if current_hr > best_models['hr']['metric']:
                    best_models['hr']['metric'] = current_hr
                    best_models['hr']['state'] = model.qf1.state_dict()
                    best_models['hr']['epoch'] = epoch
                    best_models['hr']['version'] = version
                    # 保存最佳HR模型
                    checkpoint_path = os.path.join(version_checkpoint_dir, 
                        f'best_hr_epoch_{epoch}_metric_{current_hr:.4f}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_models['hr']['state'],
                        'val_metric': current_hr,
                        'optimizer_state_dict': model.qf1.opt.state_dict(),
                        'optimizer2_state_dict': model.qf1.opt2.state_dict() if hasattr(model.qf1, 'opt2') else None,
                        'metrics_history': metrics_history,
                        'metric_type': 'hr',
                        'metric_value': current_hr,
                        'metric_version': version,
                        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, checkpoint_path)
                    print(f'\nSaved best {version} HR model checkpoint: {os.path.basename(checkpoint_path)}')
                
                # 更新最佳NDCG模型
                if current_ndcg > best_models['ndcg']['metric']:
                    best_models['ndcg']['metric'] = current_ndcg
                    best_models['ndcg']['state'] = model.qf1.state_dict()
                    best_models['ndcg']['epoch'] = epoch
                    best_models['ndcg']['version'] = version
                    # 保存最佳NDCG模型
                    checkpoint_path = os.path.join(version_checkpoint_dir, 
                        f'best_ndcg_epoch_{epoch}_metric_{current_ndcg:.4f}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_models['ndcg']['state'],
                        'val_metric': current_ndcg,
                        'optimizer_state_dict': model.qf1.opt.state_dict(),
                        'optimizer2_state_dict': model.qf1.opt2.state_dict() if hasattr(model.qf1, 'opt2') else None,
                        'metrics_history': metrics_history,
                        'metric_type': 'ndcg',
                        'metric_value': current_ndcg,
                        'metric_version': version,
                        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, checkpoint_path)
                    print(f'\nSaved best {version} NDCG model checkpoint: {os.path.basename(checkpoint_path)}')
            
            # 每5个epoch保存一次指标数据
            if epoch % 5 == 0:
                save_path = os.path.join('result', f'metrics_history_epoch_{epoch}.json')
                with open(save_path, 'w') as f:
                    json.dump(metrics_history, f, indent=4)
                print(f'\nSaved metrics history to: {save_path}')
    
    # 训练结束后保存最终的可视化图表
    final_plot_dir = os.path.join('result', 'plots')
    os.makedirs(final_plot_dir, exist_ok=True)
    plot_metrics(metrics_history, final_plot_dir)
    print(f'\nSaved final metrics plots to: {final_plot_dir}')
    
    # 训练结束后在测试集上评估
    print('\nEvaluating on test set...')
    
    # 为每个版本和指标类型分别评估测试集
    test_results = {}

    for version in ['original', 'removed', 'adjusted']:
        for metric_type in ['precision', 'recall', 'hr', 'ndcg']:
            print(f'\nEvaluating best {version} {metric_type} model on test set:')
            # 加载对应的最佳模型
            model.qf1.load_state_dict(best_models[metric_type]['state'])
            test_metrics = evaluate(model.qf1, args.epoch, 'Yelp', is_validation=False)
            
            # 保存测试结果
            model_name = f'Best_{version.capitalize()}_{metric_type.capitalize()}'
            test_results[model_name] = test_metrics
            
            # 打印简要的测试结果
            print(f'\nTest metrics (best {version} {metric_type} model):')
            for k in [5, 10, 20]:
                print(f'  {version.capitalize()} {metric_type.capitalize()}@{k}: {test_metrics[f"{version}_{metric_type}_{k}"]:.4f}')

    # 整理所有测试集结果并保存表格和可视化
    result_table_dir = os.path.join('result', 'test_summary')
    df = save_test_results_table(test_results, result_table_dir)
    visualize_test_results(df, result_table_dir)
    print(f"\nTest results summary and visualizations saved to: {result_table_dir}")
