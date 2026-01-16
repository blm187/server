"""
声纹库管理器

用于管理说话人的声纹特征库，包括注册、存储、加载和查询功能
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from config.logger import setup_logging
from core.utils.mfcc_extractor import MFCCExtractor

TAG = __name__
logger = setup_logging()


class VoiceprintLibrary:
    """声纹库管理器"""
    
    def __init__(
        self,
        library_path: str = "data/voiceprint_library.pkl",
        mfcc_extractor: Optional[MFCCExtractor] = None
    ):
        """
        初始化声纹库管理器
        
        Args:
            library_path: 声纹库文件路径
            mfcc_extractor: MFCC提取器，如果为None则创建默认提取器
        """
        self.library_path = library_path
        self.library: Dict[str, Dict] = {}
        
        # 创建MFCC提取器
        if mfcc_extractor is None:
            self.mfcc_extractor = MFCCExtractor()
        else:
            self.mfcc_extractor = mfcc_extractor
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.library_path), exist_ok=True)
        
        # 加载声纹库
        self.load()
        
        logger.bind(tag=TAG).info(
            f"声纹库管理器初始化: 路径={self.library_path}, "
            f"说话人数量={len(self.library)}"
        )
    
    def register_speaker(
        self,
        speaker_id: str,
        speaker_name: str,
        audio_samples: List[bytes],
        description: str = ""
    ) -> bool:
        """
        注册说话人
        
        Args:
            speaker_id: 说话人ID（唯一标识）
            speaker_name: 说话人名称
            audio_samples: 音频样本列表（PCM或WAV格式字节数据）
            description: 说话人描述（可选）
        
        Returns:
            是否注册成功
        """
        try:
            if not audio_samples:
                logger.bind(tag=TAG).error("音频样本列表为空，无法注册说话人")
                return False
            
            logger.bind(tag=TAG).info(
                f"开始注册说话人: ID={speaker_id}, 名称={speaker_name}, "
                f"音频样本数量={len(audio_samples)}"
            )
            
            # 提取每段音频的MFCC特征
            templates = []
            for i, audio_data in enumerate(audio_samples):
                try:
                    # 尝试多种方式提取特征
                    features = None
                    
                    # 方式1：尝试作为WAV格式解析
                    try:
                        features = self.mfcc_extractor.extract_from_wav_bytes(audio_data)
                    except Exception as e1:
                        # 方式2：尝试使用pydub处理（支持MP3、M4A等格式）
                        try:
                            from pydub import AudioSegment
                            import io
                            
                            # 使用pydub加载音频（自动识别格式）
                            audio_segment = AudioSegment.from_file(
                                io.BytesIO(audio_data),
                                format=None  # 自动识别格式
                            )
                            
                            # 转换为标准格式：16kHz, 单声道, 16bit
                            audio_segment = audio_segment.set_channels(1)
                            audio_segment = audio_segment.set_frame_rate(16000)
                            audio_segment = audio_segment.set_sample_width(2)
                            
                            # 获取PCM数据
                            pcm_data = audio_segment.raw_data
                            
                            # 提取MFCC特征
                            features = self.mfcc_extractor.extract_from_pcm_bytes(pcm_data)
                            
                            logger.bind(tag=TAG).debug(
                                f"样本 {i+1}/{len(audio_samples)} 使用pydub处理成功"
                            )
                        except Exception as e2:
                            # 方式3：尝试直接作为PCM数据
                            try:
                                features = self.mfcc_extractor.extract_from_pcm_bytes(audio_data)
                            except Exception as e3:
                                raise Exception(f"所有方式都失败: WAV={e1}, pydub={e2}, PCM={e3}")
                    
                    if features is not None:
                        templates.append(features)
                        logger.bind(tag=TAG).debug(
                            f"样本 {i+1}/{len(audio_samples)} 特征提取成功: "
                            f"形状={features.shape}"
                        )
                    else:
                        raise Exception("特征提取返回None")
                    
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"样本 {i+1}/{len(audio_samples)} 特征提取失败: {e}，跳过该样本"
                    )
                    continue
            
            if not templates:
                logger.bind(tag=TAG).error("所有音频样本特征提取失败，无法注册说话人")
                return False
            
            # 存储说话人信息
            self.library[speaker_id] = {
                "name": speaker_name,
                "description": description,
                "templates": templates,  # 多个MFCC特征模板
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio_samples_count": len(audio_samples),
                "valid_templates_count": len(templates)
            }
            
            # 自动保存
            self.save()
            
            logger.bind(tag=TAG).info(
                f"说话人注册成功: ID={speaker_id}, 名称={speaker_name}, "
                f"有效模板数量={len(templates)}"
            )
            
            return True
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"注册说话人失败: {e}")
            import traceback
            logger.bind(tag=TAG).error(traceback.format_exc())
            return False
    
    def identify_speaker(
        self,
        audio_data: bytes,
        similarity_threshold: float = 0.7
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        识别说话人
        
        Args:
            audio_data: 待识别的音频数据（PCM或WAV格式字节数据）
            similarity_threshold: 相似度阈值（0-1），默认0.7
        
        Returns:
            (speaker_id, confidence, speaker_name)
            - speaker_id: 识别出的说话人ID，如果未识别则返回None
            - confidence: 置信度分数（0-1）
            - speaker_name: 说话人名称
        """
        try:
            if not self.library:
                logger.bind(tag=TAG).warning("声纹库为空，无法识别说话人")
                return None, 0.0, None
            
            # 提取查询音频的MFCC特征
            try:
                query_features = self.mfcc_extractor.extract_from_wav_bytes(audio_data)
            except:
                query_features = self.mfcc_extractor.extract_from_pcm_bytes(audio_data)
            
            logger.bind(tag=TAG).debug(
                f"查询音频特征提取成功: 形状={query_features.shape}"
            )
            
            # 与所有说话人模板匹配
            best_match = None
            best_score = 0.0
            best_speaker_id = None
            best_speaker_name = None
            
            for speaker_id, speaker_data in self.library.items():
                templates = speaker_data["templates"]
                
                # 与每个模板计算相似度，取最高分
                max_similarity = 0.0
                for template in templates:
                    similarity = self._compute_similarity(query_features, template)
                    max_similarity = max(max_similarity, similarity)
                
                # 记录最佳匹配
                if max_similarity > best_score:
                    best_score = max_similarity
                    best_speaker_id = speaker_id
                    best_speaker_name = speaker_data["name"]
            
            # 判断是否超过阈值
            if best_score >= similarity_threshold:
                logger.bind(tag=TAG).info(
                    f"识别成功: 说话人={best_speaker_name}({best_speaker_id}), "
                    f"置信度={best_score:.3f}"
                )
                return best_speaker_id, best_score, best_speaker_name
            else:
                logger.bind(tag=TAG).info(
                    f"识别失败: 最高置信度={best_score:.3f} < 阈值={similarity_threshold}"
                )
                return None, best_score, None
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"识别说话人失败: {e}")
            import traceback
            logger.bind(tag=TAG).error(traceback.format_exc())
            return None, 0.0, None
    
    def _compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        计算两个MFCC特征矩阵的相似度
        
        使用平均MFCC向量 + 余弦相似度算法
        解决不同长度音频的特征矩阵维度不匹配问题
        
        Args:
            features1: 第一个特征矩阵，形状为 (n_frames1, n_mfcc)
            features2: 第二个特征矩阵，形状为 (n_frames2, n_mfcc)
        
        Returns:
            相似度分数（0-1）
        """
        try:
            # 方法1：计算每帧MFCC系数的平均值，得到固定长度的特征向量
            # 这样可以处理不同长度的音频
            vec1 = np.mean(features1, axis=0)  # 形状: (n_mfcc,)
            vec2 = np.mean(features2, axis=0)  # 形状: (n_mfcc,)
            
            # 确保维度一致
            if vec1.shape != vec2.shape:
                logger.bind(tag=TAG).warning(
                    f"特征向量维度不匹配: {vec1.shape} vs {vec2.shape}, "
                    f"尝试截断或填充"
                )
                # 取较小的维度
                min_dim = min(len(vec1), len(vec2))
                vec1 = vec1[:min_dim]
                vec2 = vec2[:min_dim]
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            
            # 将相似度从[-1, 1]映射到[0, 1]
            # MFCC特征通常都是正数，所以相似度应该在[0, 1]范围内
            similarity = max(0.0, min(1.0, (cosine_similarity + 1) / 2))
            
            return similarity
            
        except Exception as e:
            logger.bind(tag=TAG).warning(f"计算相似度失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(traceback.format_exc())
            return 0.0
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        删除说话人
        
        Args:
            speaker_id: 说话人ID
        
        Returns:
            是否删除成功
        """
        try:
            if speaker_id not in self.library:
                logger.bind(tag=TAG).warning(f"说话人不存在: {speaker_id}")
                return False
            
            speaker_name = self.library[speaker_id]["name"]
            del self.library[speaker_id]
            
            # 自动保存
            self.save()
            
            logger.bind(tag=TAG).info(f"说话人已删除: ID={speaker_id}, 名称={speaker_name}")
            return True
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"删除说话人失败: {e}")
            return False
    
    def list_speakers(self) -> List[Dict]:
        """
        列出所有说话人
        
        Returns:
            说话人信息列表
        """
        speakers = []
        for speaker_id, data in self.library.items():
            speakers.append({
                "speaker_id": speaker_id,
                "name": data["name"],
                "description": data.get("description", ""),
                "created_at": data.get("created_at", ""),
                "templates_count": len(data["templates"]),
                "audio_samples_count": data.get("audio_samples_count", 0)
            })
        return speakers
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """
        获取说话人信息
        
        Args:
            speaker_id: 说话人ID
        
        Returns:
            说话人信息字典，如果不存在则返回None
        """
        return self.library.get(speaker_id)
    
    def save(self):
        """保存声纹库到文件"""
        try:
            with open(self.library_path, 'wb') as f:
                pickle.dump(self.library, f)
            
            logger.bind(tag=TAG).info(
                f"声纹库已保存: 路径={self.library_path}, "
                f"说话人数量={len(self.library)}"
            )
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存声纹库失败: {e}")
            raise
    
    def load(self):
        """从文件加载声纹库"""
        try:
            if os.path.exists(self.library_path):
                with open(self.library_path, 'rb') as f:
                    self.library = pickle.load(f)
                
                logger.bind(tag=TAG).info(
                    f"声纹库已加载: 路径={self.library_path}, "
                    f"说话人数量={len(self.library)}"
                )
            else:
                self.library = {}
                logger.bind(tag=TAG).info(
                    f"声纹库文件不存在，创建新的空库: {self.library_path}"
                )
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载声纹库失败: {e}")
            # 如果加载失败，创建空库
            self.library = {}
    
    def clear(self):
        """清空声纹库"""
        self.library = {}
        self.save()
        logger.bind(tag=TAG).info("声纹库已清空")
    
    def get_statistics(self) -> Dict:
        """
        获取声纹库统计信息
        
        Returns:
            统计信息字典
        """
        total_speakers = len(self.library)
        total_templates = sum(len(data["templates"]) for data in self.library.values())
        total_audio_samples = sum(
            data.get("audio_samples_count", 0) 
            for data in self.library.values()
        )
        
        return {
            "total_speakers": total_speakers,
            "total_templates": total_templates,
            "total_audio_samples": total_audio_samples,
            "average_templates_per_speaker": (
                total_templates / total_speakers if total_speakers > 0 else 0
            )
        }

