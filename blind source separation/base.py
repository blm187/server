from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SpeechSeparationProviderBase(ABC):
    """语音分离提供者基类"""
    
    @abstractmethod
    async def separate(
        self, 
        audio_data: bytes,
        sample_rate: int = 16000,
        num_speakers: int = None
    ) -> List[Dict[str, Any]]:
        """
        分离语音
        
        Args:
            audio_data: 混合音频数据（PCM bytes）
            sample_rate: 采样率，默认16000Hz
            num_speakers: 说话人数量（可选，如果None则自动检测）
        
        Returns:
            分离后的音频列表，每个元素包含：
            {
                "speaker_id": str,  # 说话人ID (如 "SPEAKER_00", "SPEAKER_01")
                "audio": bytes,      # 分离后的音频数据（PCM）
                "start": float,      # 开始时间（秒）
                "end": float         # 结束时间（秒）
            }
        """
        pass

