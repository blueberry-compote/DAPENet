#!/usr/bin/env python3
"""
配置文件热力图生成工具
用于对比多个配置文件的参数差异
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from mmcv import Config
except ImportError:
    print("Warning: mmcv not found, using basic config parser")
    Config = None


def parse_config_file(config_path):
    """解析配置文件，返回参数字典"""
    if Config is not None:
        try:
            cfg = Config.fromfile(config_path)
            return cfg._cfg_dict
        except Exception as e:
            print(f"Warning: Failed to parse {config_path} with mmcv: {e}")
    
    # 备用方案：使用exec执行配置文件
    config_dict = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        code = f.read()
        exec(compile(code, config_path, 'exec'), {'__builtins__': __builtins__}, config_dict)
    return config_dict


def extract_key_params(config_dict, config_name):
    """从配置字典中提取关键参数"""
    params = {'config_name': config_name}
    
    # 优化器参数
    if 'optimizer' in config_dict:
        opt = config_dict['optimizer']
        params['optimizer_type'] = opt.get('type', 'N/A')
        params['lr'] = opt.get('lr', np.nan)
        params['momentum'] = opt.get('momentum', np.nan)
        params['weight_decay'] = opt.get('weight_decay', np.nan)
    
    # 学习率配置
    if 'lr_config' in config_dict:
        lr_cfg = config_dict['lr_config']
        params['lr_policy'] = lr_cfg.get('policy', 'N/A')
        params['warmup_iters'] = lr_cfg.get('warmup_iters', np.nan)
        params['warmup_ratio'] = lr_cfg.get('warmup_ratio', np.nan)
    
    # 模型结构参数
    if 'model' in config_dict:
        model = config_dict['model']
        
        # Backbone
        if 'backbone' in model:
            backbone = model['backbone']
            params['backbone_type'] = backbone.get('type', 'N/A')
            params['backbone_depth'] = backbone.get('depth', np.nan)
            params['frozen_stages'] = backbone.get('frozen_stages', np.nan)
        
        # Neck
        if 'neck' in model:
            neck = model['neck']
            params['neck_type'] = neck.get('type', 'N/A')
            params['neck_out_channels'] = neck.get('out_channels', np.nan)
            params['neck_num_outs'] = neck.get('num_outs', np.nan)
        
        # Head
        if 'bbox_head' in model:
            head = model['bbox_head']
            params['head_type'] = head.get('type', 'N/A')
            params['num_classes'] = head.get('num_classes', np.nan)
            params['stacked_convs'] = head.get('stacked_convs', np.nan)
            params['feat_channels'] = head.get('feat_channels', np.nan)
            
            # Loss权重
            if 'loss_cls' in head:
                params['loss_cls_weight'] = head['loss_cls'].get('loss_weight', np.nan)
                params['loss_cls_beta'] = head['loss_cls'].get('beta', np.nan)
            if 'loss_bbox' in head:
                params['loss_bbox_weight'] = head['loss_bbox'].get('loss_weight', np.nan)
            if 'loss_dfl' in head:
                params['loss_dfl_weight'] = head['loss_dfl'].get('loss_weight', np.nan)
    
    # 数据配置
    if 'data' in config_dict:
        data = config_dict['data']
        params['samples_per_gpu'] = data.get('samples_per_gpu', np.nan)
        params['workers_per_gpu'] = data.get('workers_per_gpu', np.nan)
        if 'train' in data and isinstance(data['train'], dict):
            if 'times' in data['train']:
                params['repeat_times'] = data['train']['times']
    
    # 数据增强参数（提取第一个augmentation的概率）
    if 'albu_train_transforms' in config_dict:
        albu_transforms = config_dict['albu_train_transforms']
        if isinstance(albu_transforms, list) and len(albu_transforms) > 0:
            first_transform = albu_transforms[0]
            if isinstance(first_transform, dict):
                params['aug_shift_scale_rotate_p'] = first_transform.get('p', np.nan)
    
    # FP16配置
    if 'fp16' in config_dict:
        fp16 = config_dict['fp16']
        params['fp16_loss_scale'] = fp16.get('loss_scale', np.nan) if isinstance(fp16, dict) else np.nan
    
    return params


def collect_configs(config_dir):
    """收集所有配置文件并提取参数"""
    config_files = glob.glob(os.path.join(config_dir, '*.py'))
    config_files = [f for f in config_files if not os.path.basename(f).startswith('_')]
    
    all_params = []
    for config_file in sorted(config_files):
        config_name = os.path.basename(config_file).replace('.py', '')
        try:
            config_dict = parse_config_file(config_file)
            params = extract_key_params(config_dict, config_name)
            all_params.append(params)
            print(f"✓ Parsed: {config_name}")
        except Exception as e:
            print(f"✗ Failed to parse {config_name}: {e}")
    
    return pd.DataFrame(all_params)


def normalize_numeric_columns(df):
    """归一化数值列，保留非数值列"""
    df_normalized = df.copy()
    
    # 分离数值列和非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 归一化数值列（使用min-max归一化）
    for col in numeric_cols:
        if col == 'config_name':
            continue
        col_data = df[col].dropna()
        if len(col_data) > 0:
            min_val = col_data.min()
            max_val = col_data.max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized, numeric_cols, non_numeric_cols


def create_heatmap(df, output_path, figsize=(20, 12), cmap='viridis'):
    """创建热力图"""
    # 设置config_name为索引
    df_indexed = df.set_index('config_name')
    
    # 分离数值列和非数值列
    numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found for heatmap")
        return
    
    # 只使用数值列
    df_numeric = df_indexed[numeric_cols]
    
    # 转置：行为参数，列为配置文件
    df_transposed = df_numeric.T
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 创建热力图
    sns.heatmap(
        df_transposed,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        cbar_kws={'label': 'Normalized Value'},
        linewidths=0.5,
        linecolor='gray',
        xticklabels=True,
        yticklabels=True
    )
    
    plt.title('Configuration Files Parameter Heatmap', fontsize=16, pad=20)
    plt.xlabel('Configuration Files', fontsize=12)
    plt.ylabel('Parameters', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Heatmap saved to: {output_path}")
    plt.close()


def create_clustered_heatmap(df, output_path, figsize=(20, 12), cmap='viridis'):
    """创建聚类热力图"""
    df_indexed = df.set_index('config_name')
    numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found for clustered heatmap")
        return
    
    df_numeric = df_indexed[numeric_cols]
    df_transposed = df_numeric.T
    
    # 创建聚类热力图
    plt.figure(figsize=figsize)
    
    g = sns.clustermap(
        df_transposed,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        cbar_kws={'label': 'Normalized Value'},
        linewidths=0.5,
        linecolor='gray',
        figsize=figsize,
        method='average',
        metric='euclidean'
    )
    
    g.fig.suptitle('Configuration Files Parameter Clustered Heatmap', fontsize=16, y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Clustered heatmap saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成配置文件热力图')
    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs/SSG',
        help='配置文件目录路径（默认: configs/SSG）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='work_dirs/config_heatmap.png',
        help='输出图片路径（默认: work_dirs/config_heatmap.png）'
    )
    parser.add_argument(
        '--clustered',
        action='store_true',
        help='生成聚类热力图'
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        choices=['viridis', 'plasma', 'coolwarm', 'RdYlBu', 'seismic'],
        help='颜色映射方案（默认: viridis）'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default='20,12',
        help='图形大小，格式: width,height（默认: 20,12）'
    )
    parser.add_argument(
        '--save-csv',
        type=str,
        default=None,
        help='保存提取的参数到CSV文件（可选）'
    )
    
    args = parser.parse_args()
    
    # 解析图形大小
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except:
        figsize = (20, 12)
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / args.config_dir
    
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return
    
    print(f"Scanning config directory: {config_dir}")
    print("-" * 60)
    
    # 收集配置参数
    df = collect_configs(str(config_dir))
    
    if len(df) == 0:
        print("Error: No valid config files found")
        return
    
    print(f"\nFound {len(df)} configuration files")
    print("-" * 60)
    
    # 保存CSV（如果指定）
    if args.save_csv:
        csv_path = project_root / args.save_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"✓ Parameters saved to CSV: {csv_path}")
    
    # 归一化数据
    df_normalized, numeric_cols, non_numeric_cols = normalize_numeric_columns(df)
    
    # 创建输出目录
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成热力图
    if args.clustered:
        clustered_output = str(output_path).replace('.png', '_clustered.png')
        create_clustered_heatmap(df_normalized, clustered_output, figsize, args.cmap)
    else:
        create_heatmap(df_normalized, str(output_path), figsize, args.cmap)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  - Config files processed: {len(df)}")
    print(f"  - Parameters extracted: {len(numeric_cols)} numeric, {len(non_numeric_cols)} non-numeric")
    print(f"  - Output: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
