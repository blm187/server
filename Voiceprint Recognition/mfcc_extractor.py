"""
MFCC特征提取模块

用于从音频数据中提取MFCC（Mel-Frequency Cepstral Coefficients）特征
"""
import numpy as np
import wave
import io
from typing import Union, Optional
from python_speech_features import mfcc
from config.logger import setup_logging

# 尝试导入pydub用于重采样
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

TAG = __name__
logger = setup_logging()


class MFCCExtractor:
    """MFCC特征提取器"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 26,
        preemph: float = 0.97,
        ceplifter: int = 22,
        append_energy: bool = True
    ):
        """
        初始化MFCC提取器
        
        Args:
            sample_rate: 采样率，默认16000Hz
            n_mfcc: MFCC系数数量，默认13
            n_fft: FFT窗口大小，默认2048
            hop_length: 帧移（hop length），默认512
            n_mels: 梅尔滤波器数量，默认26
            preemph: 预加重系数，默认0.97
            ceplifter: 倒谱提升系数，默认22
            append_energy: 是否添加能量特征，默认True
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.preemph = preemph
        self.ceplifter = ceplifter
        self.append_energy = append_energy
        
        logger.bind(tag=TAG).debug(
            f"MFCC提取器初始化: 采样率={sample_rate}Hz, "
            f"MFCC系数={n_mfcc}, FFT窗口={n_fft}"
        )
    
    def extract_from_pcm_bytes(
        self, 
        pcm_data: bytes,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        从PCM字节数据提取MFCC特征
        
        Args:
            pcm_data: PCM格式的音频字节数据（16bit, 小端序）
            sample_rate: 采样率，如果为None则使用初始化时的采样率
        
        Returns:
            MFCC特征矩阵，形状为 (n_frames, n_mfcc)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # 将PCM字节转换为numpy数组
            # PCM数据是16bit（2字节）小端序
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # 转换为浮点数并归一化到[-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # 检查音频长度
            if len(audio_float) < self.n_fft:
                logger.bind(tag=TAG).warning(
                    f"音频长度({len(audio_float)}样本)小于FFT窗口大小({self.n_fft})，"
                    f"可能影响特征提取质量"
                )
            
            # 提取MFCC特征
            mfcc_features = mfcc(
                audio_float,
                sample_rate,
                numcep=self.n_mfcc,
                nfilt=self.n_mels,
                nfft=self.n_fft,
                lowfreq=0,
                highfreq=None,  # None表示使用奈奎斯特频率（sample_rate/2）
                preemph=self.preemph,
                ceplifter=self.ceplifter,
                appendEnergy=self.append_energy
            )
            
            logger.bind(tag=TAG).debug(
                f"MFCC特征提取成功: 音频长度={len(audio_float)/sample_rate:.2f}秒, "
                f"特征形状={mfcc_features.shape}"
            )
            
            return mfcc_features
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"MFCC特征提取失败: {e}")
            raise
    
    def extract_from_wav_bytes(
        self, 
        wav_data: bytes
    ) -> np.ndarray:
        """
        从WAV格式字节数据提取MFCC特征
        
        Args:
            wav_data: WAV格式的音频字节数据
        
        Returns:
            MFCC特征矩阵，形状为 (n_frames, n_mfcc)
        """
        try:
            # 使用BytesIO将字节数据转换为文件对象
            wav_file = io.BytesIO(wav_data)
            
            # 读取WAV文件
            with wave.open(wav_file, 'rb') as wf:
                # 获取WAV文件参数
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # 读取PCM数据
                pcm_data = wf.readframes(n_frames)
                
                logger.bind(tag=TAG).debug(
                    f"WAV文件信息: 采样率={framerate}Hz, "
                    f"声道数={n_channels}, 位深={sample_width*8}bit, "
                    f"帧数={n_frames}"
                )
                
                # 如果是立体声，转换为单声道（取平均值）
                if n_channels == 2:
                    # 将字节数据转换为numpy数组
                    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                    # 重塑为(帧数, 声道数)
                    audio_array = audio_array.reshape(-1, 2)
                    # 取平均值转换为单声道
                    audio_array = audio_array.mean(axis=1).astype(np.int16)
                    # 转换回字节
                    pcm_data = audio_array.tobytes()
                
                # 如果采样率不匹配，进行重采样
                if framerate != self.sample_rate:
                    if PYDUB_AVAILABLE:
                        # 使用pydub进行重采样
                        logger.bind(tag=TAG).info(
                            f"检测到采样率不匹配: {framerate}Hz -> {self.sample_rate}Hz，"
                            f"正在进行重采样..."
                        )
                        try:
                            # 创建临时WAV文件对象
                            wav_io = io.BytesIO()
                            wav_io.write(wav_data)
                            wav_io.seek(0)
                            
                            # 使用pydub加载并重采样
                            audio = AudioSegment.from_wav(wav_io)
                            audio = audio.set_frame_rate(self.sample_rate)
                            audio = audio.set_channels(1)  # 确保单声道
                            audio = audio.set_sample_width(2)  # 确保16bit
                            
                            # 获取重采样后的PCM数据
                            pcm_data = audio.raw_data
                            sample_rate = self.sample_rate
                            
                            logger.bind(tag=TAG).info(
                                f"重采样完成: {framerate}Hz -> {self.sample_rate}Hz"
                            )
                        except Exception as e:
                            logger.bind(tag=TAG).warning(
                                f"重采样失败: {e}，将使用原始采样率，可能影响识别效果"
                            )
                            sample_rate = framerate
                    else:
                        logger.bind(tag=TAG).warning(
                            f"WAV采样率({framerate}Hz)与目标采样率({self.sample_rate}Hz)不匹配，"
                            f"且pydub未安装，无法重采样。建议：\n"
                            f"  1. 安装pydub: pip install pydub\n"
                            f"  2. 或使用16kHz的音频文件\n"
                            f"当前将使用原始采样率，可能严重影响识别效果！"
                        )
                        sample_rate = framerate
                else:
                    sample_rate = self.sample_rate
            
            # 从PCM数据提取MFCC特征
            return self.extract_from_pcm_bytes(pcm_data, sample_rate)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"从WAV数据提取MFCC特征失败: {e}")
            raise
    
    def extract_from_wav_file(
        self, 
        wav_file_path: str
    ) -> np.ndarray:
        """
        从WAV文件提取MFCC特征
        
        Args:
            wav_file_path: WAV文件路径
        
        Returns:
            MFCC特征矩阵，形状为 (n_frames, n_mfcc)
        """
        try:
            with open(wav_file_path, 'rb') as f:
                wav_data = f.read()
            
            return self.extract_from_wav_bytes(wav_data)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"从WAV文件提取MFCC特征失败: {e}")
            raise
    
    def extract_from_audio_segment(
        self,
        audio_segment
    ) -> np.ndarray:
        """
        从pydub AudioSegment对象提取MFCC特征
        
        Args:
            audio_segment: pydub.AudioSegment对象
        
        Returns:
            MFCC特征矩阵，形状为 (n_frames, n_mfcc)
        """
        try:
            # 确保音频格式正确
            audio = audio_segment.set_channels(1)  # 单声道
            audio = audio.set_frame_rate(self.sample_rate)  # 16kHz
            audio = audio.set_sample_width(2)  # 16bit
            
            # 获取PCM数据
            pcm_data = audio.raw_data
            
            # 提取MFCC特征
            return self.extract_from_pcm_bytes(pcm_data, self.sample_rate)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"从AudioSegment提取MFCC特征失败: {e}")
            raise


# 便捷函数
def extract_mfcc(
    audio_data: Union[bytes, str],
    sample_rate: int = 16000,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    便捷函数：提取MFCC特征
    
    Args:
        audio_data: 音频数据，可以是：
                   - PCM字节数据（bytes）
                   - WAV文件路径（str）
        sample_rate: 采样率，默认16000Hz
        n_mfcc: MFCC系数数量，默认13
    
    Returns:
        MFCC特征矩阵
    """
    extractor = MFCCExtractor(sample_rate=sample_rate, n_mfcc=n_mfcc)
    
    if isinstance(audio_data, str):
        # 文件路径
        return extractor.extract_from_wav_file(audio_data)
    elif isinstance(audio_data, bytes):
        # 字节数据，尝试作为WAV格式解析
        try:
            return extractor.extract_from_wav_bytes(audio_data)
        except:
            # 如果失败，尝试作为PCM数据
            return extractor.extract_from_pcm_bytes(audio_data, sample_rate)
    else:
        raise ValueError(f"不支持的音频数据类型: {type(audio_data)}")

