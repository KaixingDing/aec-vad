"""SCP文件读写工具

用于处理Kaldi风格的SCP（脚本）文件，这是语音处理中常用的数据索引格式。
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_scp(scp_file: str) -> Dict[str, str]:
    """
    读取SCP文件
    
    Args:
        scp_file: SCP文件路径
    
    Returns:
        字典，键为样本ID，值为文件路径
    
    示例:
        >>> scp_dict = read_scp('wav.scp')
        >>> # scp_dict = {'utt001': '/path/to/utt001.wav', ...}
    """
    scp_dict = {}
    
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id, path = parts
                scp_dict[utt_id] = path
    
    return scp_dict


def write_scp(scp_dict: Dict[str, str], scp_file: str):
    """
    写入SCP文件
    
    Args:
        scp_dict: 字典，键为样本ID，值为文件路径
        scp_file: 输出SCP文件路径
    """
    output_path = Path(scp_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(scp_file, 'w', encoding='utf-8') as f:
        for utt_id in sorted(scp_dict.keys()):
            f.write(f"{utt_id} {scp_dict[utt_id]}\n")


def read_ark_scp(scp_file: str) -> Dict[str, Tuple[str, int]]:
    """
    读取指向ark文件的SCP文件
    
    Args:
        scp_file: SCP文件路径
    
    Returns:
        字典，键为样本ID，值为(ark文件路径, 偏移量)元组
    """
    scp_dict = {}
    
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                utt_id = parts[0]
                ark_path = parts[1]
                
                # 解析ark路径（格式：文件名:偏移量）
                if ':' in ark_path:
                    ark_file, offset = ark_path.split(':', 1)
                    scp_dict[utt_id] = (ark_file, int(offset))
                else:
                    scp_dict[utt_id] = (ark_path, 0)
    
    return scp_dict


def create_scp_list(data_dir: str, 
                    pattern: str = '*.wav',
                    output_scp: str = None) -> Dict[str, str]:
    """
    从目录创建SCP文件列表
    
    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式
        output_scp: 输出SCP文件路径（可选）
    
    Returns:
        SCP字典
    """
    data_path = Path(data_dir)
    scp_dict = {}
    
    for file_path in sorted(data_path.rglob(pattern)):
        # 使用文件名（不含扩展名）作为ID
        utt_id = file_path.stem
        scp_dict[utt_id] = str(file_path.absolute())
    
    if output_scp:
        write_scp(scp_dict, output_scp)
    
    return scp_dict
