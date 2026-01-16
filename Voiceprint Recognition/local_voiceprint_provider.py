"""
本地声纹识别提供者

使用MFCC特征进行本地声纹识别，替代外部API
"""
import time
from typing import Optional, Dict, Tuple
from config.logger import setup_logging
from core.utils.voiceprint_library import VoiceprintLibrary
from core.utils.mfcc_extractor import MFCCExtractor

TAG = __name__
logger = setup_logging()


class LocalVoiceprintProvider:
    """本地声纹识别提供者（替代外部API）"""
    
    def __init__(self, config: dict):
        """
        初始化本地声纹识别提供者
        
        Args:
            config: 配置字典，包含：
                - library_path: 声纹库文件路径（可选，默认data/voiceprint_library.pkl）
                - similarity_threshold: 相似度阈值（可选，默认0.7）
                - speakers: 说话人配置列表（格式：speaker_id,名称,描述）
        """
        self.config = config
        self.speakers = config.get("speakers", [])
        self.speaker_map = self._parse_speakers()
        
        # 声纹库配置
        library_path = config.get("library_path", "data/voiceprint_library.pkl")
        similarity_threshold = config.get("similarity_threshold", 0.7)
        
        # 创建MFCC提取器
        mfcc_extractor = MFCCExtractor(
            sample_rate=16000,
            n_mfcc=config.get("n_mfcc", 13),
            n_fft=config.get("n_fft", 2048)
        )
        
        # 创建声纹库管理器
        self.library = VoiceprintLibrary(
            library_path=library_path,
            mfcc_extractor=mfcc_extractor
        )
        
        self.similarity_threshold = similarity_threshold
        self.speaker_ids = list(self.speaker_map.keys())
        
        # 检查是否有有效的说话人配置
        if not self.speaker_ids:
            logger.bind(tag=TAG).warning("未配置有效的说话人，本地声纹识别将被禁用")
            self.enabled = False
        else:
            # 检查声纹库中是否有这些说话人
            library_speakers = self.library.list_speakers()
            library_speaker_ids = [s["speaker_id"] for s in library_speakers]
            
            missing_speakers = [sid for sid in self.speaker_ids if sid not in library_speaker_ids]
            if missing_speakers:
                logger.bind(tag=TAG).warning(
                    f"以下说话人未在声纹库中注册: {missing_speakers}，"
                    f"请先注册这些说话人"
                )
            
            self.enabled = True
            logger.bind(tag=TAG).info(
                f"本地声纹识别已启用: 声纹库={library_path}, "
                f"说话人={len(self.speaker_ids)}个, 阈值={similarity_threshold}"
            )
    
    def _parse_speakers(self) -> Dict[str, Dict[str, str]]:
        """解析说话人配置"""
        speaker_map = {}
        for speaker_str in self.speakers:
            try:
                parts = speaker_str.split(",", 2)
                if len(parts) >= 3:
                    speaker_id, name, description = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    speaker_map[speaker_id] = {
                        "name": name,
                        "description": description
                    }
            except Exception as e:
                logger.bind(tag=TAG).warning(f"解析说话人配置失败: {speaker_str}, 错误: {e}")
        return speaker_map
    
    async def identify_speaker(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """
        识别说话人
        
        Args:
            audio_data: WAV格式的音频数据（16kHz, 单声道, 16bit）
            session_id: 会话ID（未使用，但保持接口兼容）
        
        Returns:
            说话人名称，如果未识别则返回None
        """
        if not self.enabled:
            logger.bind(tag=TAG).debug("本地声纹识别功能已禁用，跳过识别")
            return None
        
        try:
            start_time = time.monotonic()
            
            # 识别说话人
            speaker_id, confidence, speaker_name = self.library.identify_speaker(
                audio_data,
                similarity_threshold=self.similarity_threshold
            )
            
            elapsed_time = time.monotonic() - start_time
            
            if speaker_id:
                # 验证说话人ID是否在配置中
                if speaker_id in self.speaker_map:
                    logger.bind(tag=TAG).info(
                        f"本地声纹识别耗时: {elapsed_time:.3f}s | "
                        f"识别成功: {speaker_name}, 置信度={confidence:.3f} (阈值={self.similarity_threshold})"
                    )
                    return speaker_name
                else:
                    logger.bind(tag=TAG).warning(
                        f"本地声纹识别耗时: {elapsed_time:.3f}s | "
                        f"识别出的说话人ID({speaker_id})不在配置中，置信度={confidence:.3f}"
                    )
                    return None
            else:
                logger.bind(tag=TAG).info(
                    f"本地声纹识别耗时: {elapsed_time:.3f}s | "
                    f"未识别到说话人，最高置信度={confidence:.3f} < 阈值={self.similarity_threshold}"
                )
                return None
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"本地声纹识别失败: {e}")
            import traceback
            logger.bind(tag=TAG).error(traceback.format_exc())
            return None
    
    def identify_speaker_with_confidence(
        self, 
        audio_data: bytes, 
        session_id: str
    ) -> Tuple[Optional[str], float]:
        """
        识别说话人并返回置信度（同步方法，用于在base.py中检查）
        
        Args:
            audio_data: WAV格式的音频数据
            session_id: 会话ID
        
        Returns:
            (speaker_name, confidence)
            - speaker_name: 说话人名称，如果未识别则返回None
            - confidence: 置信度分数（0-1）
        """
        if not self.enabled:
            return None, 0.0
        
        try:
            # 识别说话人
            speaker_id, confidence, speaker_name = self.library.identify_speaker(
                audio_data,
                similarity_threshold=0.0  # 使用0.0阈值，获取实际置信度
            )
            
            if speaker_id and speaker_id in self.speaker_map:
                # 检查置信度是否达到阈值
                if confidence >= self.similarity_threshold:
                    return speaker_name, confidence
                else:
                    logger.bind(tag=TAG).info(
                        f"置信度不足: {confidence:.3f} < 阈值={self.similarity_threshold}"
                    )
                    return None, confidence
            else:
                return None, confidence
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"声纹识别失败: {e}")
            return None, 0.0

