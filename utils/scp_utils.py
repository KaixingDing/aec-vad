"""SCP文件读写工具

用于处理联合AEC-VAD任务的SCP文件。
每行格式: utt_id mic_path far_path near_path vad_path
"""

from pathlib import Path
from typing import Dict, Tuple


def read_scp(scp_file: str) -> Dict[str, Tuple[str, str, str, str]]:
    """
    读取联合SCP文件
    
    Args:
        scp_file: SCP文件路径
    
    Returns:
        字典，键为样本ID，值为(麦克风路径, 远端路径, 近端路径, VAD标签路径)元组
    
    示例:
        >>> scp_dict = read_scp('data.scp')
        >>> # scp_dict = {'utt001': ('/path/mic.wav', '/path/far.wav', '/path/near.wav', '/path/vad.npy'), ...}
    """
    scp_dict = {}
    
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) == 5:
                utt_id, mic_path, far_path, near_path, vad_path = parts
                scp_dict[utt_id] = (mic_path, far_path, near_path, vad_path)
    
    return scp_dict


def write_scp(scp_dict: Dict[str, Tuple[str, str, str, str]], scp_file: str):
    """
    写入联合SCP文件
    
    Args:
        scp_dict: 字典，键为样本ID，值为(麦克风路径, 远端路径, 近端路径, VAD标签路径)元组
        scp_file: 输出SCP文件路径
    """
    output_path = Path(scp_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(scp_file, 'w', encoding='utf-8') as f:
        for utt_id in sorted(scp_dict.keys()):
            mic_path, far_path, near_path, vad_path = scp_dict[utt_id]
            f.write(f"{utt_id} {mic_path} {far_path} {near_path} {vad_path}\n")

