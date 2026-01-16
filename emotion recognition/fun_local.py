import time
import os
import sys
import io
import re
import psutil
from config.logger import setup_logging
from typing import Optional, Tuple, List, Dict, Any
from core.providers.asr.base import ASRProviderBase
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import shutil
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()

MAX_RETRIES = 2
RETRY_DELAY = 1  # 重试延迟（秒）


# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        
        # 内存检测，要求大于2G
        min_mem_bytes = 2 * 1024 * 1024 * 1024
        total_mem = psutil.virtual_memory().total
        if total_mem < min_mem_bytes:
            logger.bind(tag=TAG).error(f"可用内存不足2G，当前仅有 {total_mem / (1024*1024):.2f} MB，可能无法启动FunASR")
        
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")  # 修正配置键名
        self.delete_audio_file = delete_audio_file

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        with CaptureOutput():
            self.model = AutoModel(
                model=self.model_dir,
                vad_kwargs={"max_single_segment_time": 30000},
                disable_update=True,
                hub="hf",
                # device="cuda:0",  # 启用GPU加速
            )

    def _extract_emotion_from_result(self, result: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
        """
        从 SenseVoice 模型返回结果中提取情感标签
        
        Args:
            result: FunASR generate 方法返回的结果列表
            
        Returns:
            (情感标签, 置信度) 元组，如果未找到则返回 (None, 0.0)
        """
        if not result or len(result) == 0:
            logger.bind(tag=TAG).warning("FunASR返回结果为空")
            return None, 0.0
        
        first_result = result[0]
        
        # SenseVoice 可能的情感标签字段名（根据 FunASR 实现可能不同）
        emotion_fields = [
            "emotion", "emotion_label", "ser", "ser_label", 
            "emotion_tag", "emotion_type", "sentiment",
            "emo", "emotion_class", "emotion_category"
        ]
        
        # 方法1：尝试从独立字段提取情感标签
        for field in emotion_fields:
            if field in first_result:
                emotion_value = first_result[field]
                if emotion_value:
                    emotion = self._map_emotion_label(emotion_value)
                    confidence = first_result.get(f"{field}_confidence", 0.8)
                    if isinstance(confidence, (int, float)):
                        confidence = float(confidence)
                    else:
                        confidence = 0.8
                    logger.bind(tag=TAG).info(
                        f"✓ 从字段 '{field}' 提取情感标签: {emotion_value} -> {emotion}, 置信度={confidence:.3f}"
                    )
                    return emotion, confidence
        
        # 方法2：从 text 字段中解析特殊标记（SenseVoice 的格式：<|zh|><|SAD|><|Speech|><|withitn|>文本）
        if "text" in first_result:
            text = first_result["text"]
            if text:
                # 使用正则表达式提取 <|EMOTION|> 格式的标记
                # 匹配 <|XXX|> 格式的标记
                pattern = r'<\|([A-Z_]+)\|>'
                matches = re.findall(pattern, text)
                
                # SenseVoice 情感标签列表
                emotion_tags = ["SAD", "HAPPY", "ANGRY", "NEUTRAL", "FEAR", "SURPRISE", "DISGUST", "EMO_UNK"]
                
                for match in matches:
                    if match in emotion_tags:
                        emotion_value = match
                        emotion = self._map_emotion_label(emotion_value)
                        logger.bind(tag=TAG).info(
                            f"✓ 从text字段解析情感标签: {emotion_value} -> {emotion} (原始text: {text})"
                        )
                        return emotion, 0.8
                
                # 如果没有找到情感标签，记录调试信息
                logger.bind(tag=TAG).debug(
                    f"text字段中找到的标记: {matches}，但未找到情感标签"
                )
        
        # 如果未找到情感字段，记录调试信息
        logger.bind(tag=TAG).warning(
            f"⚠ 未找到情感标签字段，可用字段: {list(first_result.keys())}"
        )
        return None, 0.0
    
    def _map_emotion_label(self, emotion_label: str) -> str:
        """
        将 SenseVoice 的情感标签映射到中文
        
        Args:
            emotion_label: 原始情感标签（可能是英文或中文，如 "SAD", "HAPPY", "sad", "happy" 等）
            
        Returns:
            中文情感标签
        """
        if not emotion_label:
            return "平静"
        
        emotion_label = str(emotion_label).upper().strip()  # 转为大写，因为SenseVoice使用大写标签
        
        # SenseVoice 情感标签映射（支持大写格式，如 SAD, HAPPY, ANGRY 等）
        emotion_mapping = {
            # SenseVoice 大写标签格式
            "SAD": "悲伤",
            "HAPPY": "高兴",
            "ANGRY": "愤怒",
            "NEUTRAL": "平静",
            "FEAR": "恐惧",
            "SURPRISE": "惊讶",
            "DISGUST": "厌恶",
            "EMO_UNK": "平静",  # 未知情感，映射为平静
            # 小写格式（兼容）
            "neutral": "平静",
            "happy": "高兴",
            "sad": "悲伤",
            "angry": "愤怒",
            "fear": "恐惧",
            "surprise": "惊讶",
            "disgust": "厌恶",
            "emo_unk": "平静",
            # 中文标签（如果已经是中文，直接返回）
            "平静": "平静",
            "高兴": "高兴",
            "悲伤": "悲伤",
            "愤怒": "愤怒",
            "恐惧": "恐惧",
            "惊讶": "惊讶",
            "厌恶": "厌恶",
            "中性": "平静",
        }
        
        # 查找映射
        mapped_emotion = emotion_mapping.get(emotion_label)
        if mapped_emotion:
            return mapped_emotion
        
        # 如果未找到映射，尝试部分匹配（不区分大小写）
        emotion_label_lower = emotion_label.lower()
        for key, value in emotion_mapping.items():
            if key.lower() == emotion_label_lower or emotion_label_lower in key.lower() or key.lower() in emotion_label_lower:
                return value
        
        # 默认返回平静
        logger.bind(tag=TAG).warning(f"未知情感标签: {emotion_label}，映射为'平静'")
        return "平静"

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """
        语音转文本主处理逻辑（包含情感识别）
        
        Returns:
            (文本, 文件路径, 情感标签, 情感置信度) 元组
        """
        file_path = None
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # 合并所有opus数据包
                if audio_format == "pcm":
                    pcm_data = opus_data
                else:
                    pcm_data = self.decode_opus(opus_data)

                combined_pcm_data = b"".join(pcm_data)

                # 检查磁盘空间
                if not self.delete_audio_file:
                    free_space = shutil.disk_usage(self.output_dir).free
                    if free_space < len(combined_pcm_data) * 2:  # 预留2倍空间
                        raise OSError("磁盘空间不足")

                # 判断是否保存为WAV文件
                if self.delete_audio_file:
                    pass
                else:
                    file_path = self.save_audio_to_file(pcm_data, session_id)

                # 语音识别（包含情感识别）
                start_time = time.time()
                result = self.model.generate(
                    input=combined_pcm_data,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    ban_emo_unk=False,  # 重要：禁用emo_unk标签，确保所有句子都被赋予情感标签
                )
                
                # 提取文本
                text = rich_transcription_postprocess(result[0]["text"])
                
                # 提取情感标签（SenseVoice 原生情感识别）
                emotion, emotion_confidence = self._extract_emotion_from_result(result)
                
                if emotion:
                    logger.bind(tag=TAG).info(
                        f"✓ SenseVoice情感识别成功: {emotion} (置信度={emotion_confidence:.3f})"
                    )
                else:
                    logger.bind(tag=TAG).warning(
                        f"⚠ SenseVoice未返回情感标签，将使用MFCC备选方案（如果已启用）"
                    )
                
                logger.bind(tag=TAG).info(
                    f"语音识别耗时: {time.time() - start_time:.3f}s | "
                    f"文本: {text} | "
                    f"情感: {emotion or '未识别'} (置信度={emotion_confidence:.3f})"
                )

                return text, file_path, emotion, emotion_confidence

            except OSError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.bind(tag=TAG).error(
                        f"语音识别失败（已重试{retry_count}次）: {e}", exc_info=True
                    )
                    return "", file_path, None, 0.0
                logger.bind(tag=TAG).warning(
                    f"语音识别失败，正在重试（{retry_count}/{MAX_RETRIES}）: {e}"
                )
                time.sleep(RETRY_DELAY)

            except Exception as e:
                logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
                return "", file_path, None, 0.0

            finally:
                # 文件清理逻辑
                if self.delete_audio_file and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(
                            f"文件删除失败: {file_path} | 错误: {e}"
                        )
