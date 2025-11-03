#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本

用于生成联合AEC-VAD训练数据并保存为SCP格式。
"""

import argparse
from pathlib import Path
from preprocessing import JointAECVADPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description='生成联合AEC-VAD训练数据的SCP文件'
    )
    
    # 数据集路径参数
    parser.add_argument(
        '--dns_root',
        type=str,
        default=None,
        help='DNS-Challenge数据集根目录'
    )
    parser.add_argument(
        '--librispeech_root',
        type=str,
        default=None,
        help='LibriSpeech数据集根目录'
    )
    
    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed',
        help='输出目录'
    )
    
    # 数据生成参数
    parser.add_argument(
        '--num_train',
        type=int,
        default=10000,
        help='训练集样本数量'
    )
    parser.add_argument(
        '--num_val',
        type=int,
        default=1000,
        help='验证集样本数量'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=1000,
        help='测试集样本数量'
    )
    
    # 音频参数
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='采样率（Hz）'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=4.0,
        help='样本时长（秒）'
    )
    parser.add_argument(
        '--ser_min',
        type=float,
        default=-5.0,
        help='信号回声比最小值（dB）'
    )
    parser.add_argument(
        '--ser_max',
        type=float,
        default=15.0,
        help='信号回声比最大值（dB）'
    )
    parser.add_argument(
        '--snr_min',
        type=float,
        default=0.0,
        help='信噪比最小值（dB）'
    )
    parser.add_argument(
        '--snr_max',
        type=float,
        default=30.0,
        help='信噪比最大值（dB）'
    )
    
    args = parser.parse_args()
    
    # 检查是否至少提供了一个数据集路径
    if not args.dns_root and not args.librispeech_root:
        print("警告: 未提供数据集路径。将创建演示用的空SCP文件。")
        print("请使用 --dns_root 和/或 --librispeech_root 指定数据集路径。")
        return
    
    # 创建预处理器
    print("初始化联合AEC-VAD预处理器...")
    preprocessor = JointAECVADPreprocessor(
        dns_root=args.dns_root,
        librispeech_root=args.librispeech_root,
        sample_rate=args.sample_rate,
        ser_range=(args.ser_min, args.ser_max),
        snr_range=(args.snr_min, args.snr_max),
        duration=args.duration,
    )
    
    # 扫描文件
    print("\n扫描数据集文件...")
    preprocessor.scan_files()
    
    # 生成训练集
    if args.num_train > 0:
        print(f"\n生成训练集（{args.num_train}个样本）...")
        preprocessor.preprocess_dataset(
            num_samples=args.num_train,
            output_dir=args.output_dir,
            split='train'
        )
    
    # 生成验证集
    if args.num_val > 0:
        print(f"\n生成验证集（{args.num_val}个样本）...")
        preprocessor.preprocess_dataset(
            num_samples=args.num_val,
            output_dir=args.output_dir,
            split='val'
        )
    
    # 生成测试集
    if args.num_test > 0:
        print(f"\n生成测试集（{args.num_test}个样本）...")
        preprocessor.preprocess_dataset(
            num_samples=args.num_test,
            output_dir=args.output_dir,
            split='test'
        )
    
    print("\n数据预处理完成！")
    print(f"输出目录: {args.output_dir}")
    print("\nSCP文件说明:")
    print("  - microphone.scp: 麦克风信号（用于AEC和VAD）")
    print("  - far_end.scp: 远端参考信号（用于AEC）")
    print("  - near_end.scp: 近端纯净语音（AEC目标）")
    print("  - vad_labels.scp: VAD标签（VAD目标）")


if __name__ == '__main__':
    main()
