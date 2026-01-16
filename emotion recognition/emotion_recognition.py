"""
情感识别模块 - 基于MFCC特征的情感识别

使用MFCC特征提取 + 机器学习/深度学习模型进行情感识别
支持：
1. PyTorch模型 (.pth文件) - 推荐，如innnky/speech-emotion-recognition
2. Pickle模型 (.pkl文件) - sklearn等传统ML模型
3. 简单分类器 - 基于统计特征的规则分类
"""
import numpy as np
import pickle
import os
from typing import Optional, Tuple
from pathlib import Path
from config.logger import setup_logging
from core.utils.mfcc_extractor import MFCCExtractor

# 尝试导入PyTorch（可选）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TAG = __name__
logger = setup_logging()

# PyTorch情感模型类别映射（innnky/speech-emotion-recognition标准）
# 注意：PyTorch模型通常只支持4类情感，如果需要7类情感，请使用.pkl模型或简单分类器
PYTORCH_EMOTION_MAP = ['愤怒', '高兴', '悲伤', '平静']  # 将"中性"改为"平静"

# 情感类别映射（7类：愤怒、厌恶、恐惧、高兴、平静、悲伤、惊讶）
EMOTION_MAP = {
    0: "平静",  # 原"中性"，改为"平静"以符合用户需求
    1: "高兴",
    2: "悲伤",
    3: "愤怒",
    4: "恐惧",
    5: "惊讶",
    6: "厌恶"
}

# 简化的情感类别（常用，4类）
SIMPLE_EMOTION_MAP = {
    0: "平静",  # 原"中性"，改为"平静"
    1: "高兴",
    2: "悲伤",
    3: "愤怒"
}


# PyTorch LSTM情感识别模型（innnky/speech-emotion-recognition标准架构）
class EmotionLSTMModel:
    """PyTorch LSTM情感识别模型"""
    
    def __init__(self, model_path: str):
        """
        初始化PyTorch模型
        
        Args:
            model_path: .pth模型文件路径
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用PyTorch模型。请运行: pip install torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_pytorch_model(model_path)
        self.model.eval()
        logger.bind(tag=TAG).info(f"PyTorch情感模型已加载到设备: {self.device}")
    
    def _load_pytorch_model(self, model_path: str):
        """加载PyTorch模型"""
        try:
            # 尝试加载为完整模型（如果保存的是完整模型）
            try:
                model = torch.load(model_path, map_location=self.device)
                if isinstance(model, torch.nn.Module):
                    return model
            except:
                pass
            
            # 尝试加载为state_dict（如果保存的是权重）
            # 使用标准LSTM架构（40维MFCC输入）
            class EmotionLSTM(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(40, 128, batch_first=True)
                    self.fc = torch.nn.Linear(128, 4)  # 4类情感
                
                def forward(self, x):
                    _, (h, _) = self.lstm(x)
                    out = self.fc(h[-1])
                    return out
            
            model = EmotionLSTM()
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            return model
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载PyTorch模型失败: {e}")
            raise
    
    def predict(self, mfcc_features: np.ndarray) -> Tuple[str, float]:
        """
        使用PyTorch模型预测情感
        
        Args:
            mfcc_features: MFCC特征矩阵，形状为 (n_frames, n_mfcc)
                           n_mfcc应该是40（innnky模型标准）
        
        Returns:
            (情感类别, 置信度)
        """
        if len(mfcc_features) == 0:
            return "平静", 0.5
        
        # 确保MFCC是40维（如果不是，需要调整）
        if mfcc_features.shape[1] != 40:
            logger.bind(tag=TAG).warning(
                f"MFCC特征维度不匹配: 期望40维，实际{mfcc_features.shape[1]}维。"
                f"将使用前40维或补零"
            )
            if mfcc_features.shape[1] < 40:
                # 补零
                padding = np.zeros((mfcc_features.shape[0], 40 - mfcc_features.shape[1]))
                mfcc_features = np.concatenate([mfcc_features, padding], axis=1)
            else:
                # 截取前40维
                mfcc_features = mfcc_features[:, :40]
        
        # 转换为tensor
        input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)  # 增加batch维
        input_tensor = input_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(input_tensor)
            probas = torch.softmax(logits, dim=1)
            emotion_idx = logits.argmax(dim=1).item()
            confidence = probas[0, emotion_idx].item()
        
        # 映射到情感类别
        emotion = PYTORCH_EMOTION_MAP[emotion_idx] if emotion_idx < len(PYTORCH_EMOTION_MAP) else "平静"
        
        return emotion, float(confidence)


class EmotionRecognizer:
    """基于MFCC特征的情感识别器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_simple_emotions: bool = True,
        sample_rate: int = 16000,
        n_mfcc: int = 13
    ):
        """
        初始化情感识别器
        
        Args:
            model_path: 预训练模型路径（.pkl文件），如果为None则使用基于统计特征的简单分类器
            use_simple_emotions: 是否使用简化的情感类别（4类 vs 7类）
            sample_rate: 音频采样率，默认16000Hz
            n_mfcc: MFCC系数数量，默认13
        """
        self.model_path = model_path
        self.use_simple_emotions = use_simple_emotions
        self.emotion_map = SIMPLE_EMOTION_MAP if use_simple_emotions else EMOTION_MAP
        
        # 初始化MFCC提取器（默认使用n_mfcc，如果加载PyTorch模型会自动调整为40）
        self.mfcc_extractor = MFCCExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc
        )
        
        # 加载模型
        self.model = None
        self.feature_extractor = None
        self.is_pytorch_model = False
        self.n_mfcc_for_model = n_mfcc  # 默认使用初始化的n_mfcc
        
        if model_path and os.path.exists(model_path):
            try:
                # 检查文件扩展名
                file_ext = os.path.splitext(model_path)[1].lower()
                
                if file_ext == '.pth':
                    # PyTorch LSTM/CNN模型
                    if not TORCH_AVAILABLE:
                        logger.bind(tag=TAG).warning(
                            "PyTorch未安装，无法加载.pth模型。"
                            "请运行: pip install torch"
                        )
                    else:
                        self.model = EmotionLSTMModel(model_path)
                        self.is_pytorch_model = True
                        self.n_mfcc_for_model = 40  # PyTorch模型使用40维MFCC
                        logger.bind(tag=TAG).info(f"成功加载PyTorch情感识别模型: {model_path}")
                else:
                    # Pickle模型（.pkl或其他）
                    self._load_pickle_model(model_path)
                    logger.bind(tag=TAG).info(f"成功加载Pickle情感识别模型: {model_path}")
            except Exception as e:
                logger.bind(tag=TAG).warning(f"加载情感识别模型失败: {e}，将使用基于统计特征的简单分类器")
                self.model = None
        else:
            logger.bind(tag=TAG).info("未提供模型路径，将使用基于统计特征的简单分类器")
    
    def _load_pickle_model(self, model_path: str):
        """加载Pickle格式的预训练模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_extractor = model_data.get('feature_extractor')
            else:
                self.model = model_data
    
    def _compute_delta(self, features: np.ndarray, N: int = 2) -> np.ndarray:
        """
        计算MFCC特征的一阶差分（delta）
        
        Args:
            features: MFCC特征矩阵，形状为 (n_frames, n_mfcc)
            N: 用于计算差分的窗口大小，默认2
        
        Returns:
            Delta特征矩阵，形状为 (n_frames, n_mfcc)
        """
        if len(features) == 0:
            return np.zeros_like(features)
        
        delta = np.zeros_like(features)
        for t in range(len(features)):
            numerator = sum(i * (features[min(t + i, len(features) - 1)] - 
                                 features[max(t - i, 0)]) 
                           for i in range(1, N + 1))
            denominator = 2 * sum(i * i for i in range(1, N + 1))
            delta[t] = numerator / denominator if denominator > 0 else 0
        
        return delta
    
    def _extract_statistical_features(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        从MFCC特征中提取增强的统计特征（包含delta和delta-delta）
        
        Args:
            mfcc_features: MFCC特征矩阵，形状为 (n_frames, n_mfcc)
        
        Returns:
            增强的统计特征向量
        """
        if len(mfcc_features) == 0:
            # 如果MFCC特征为空，返回零向量
            n_features = mfcc_features.shape[1] if len(mfcc_features.shape) > 1 else 13
            return np.zeros(n_features * 7)  # 7个统计量
        
        # 计算delta（一阶差分）
        delta_features = self._compute_delta(mfcc_features, N=2)
        
        # 计算delta-delta（二阶差分）
        delta_delta_features = self._compute_delta(delta_features, N=2)
        
        # 提取基础统计特征：均值、标准差、最大值、最小值
        mean_features = np.mean(mfcc_features, axis=0)  # (n_mfcc,)
        std_features = np.std(mfcc_features, axis=0)  # (n_mfcc,)
        max_features = np.max(mfcc_features, axis=0)  # (n_mfcc,)
        min_features = np.min(mfcc_features, axis=0)  # (n_mfcc,)
        
        # 提取delta的统计特征
        delta_mean = np.mean(delta_features, axis=0)  # (n_mfcc,)
        delta_std = np.std(delta_features, axis=0)  # (n_mfcc,)
        
        # 提取delta-delta的统计特征
        delta_delta_mean = np.mean(delta_delta_features, axis=0)  # (n_mfcc,)
        
        # 计算额外的统计特征：偏度（skewness）和峰度（kurtosis）
        # 偏度：衡量分布的对称性
        skew_features = np.array([
            self._compute_skewness(mfcc_features[:, i]) 
            for i in range(mfcc_features.shape[1])
        ])
        
        # 峰度：衡量分布的尖锐程度
        kurtosis_features = np.array([
            self._compute_kurtosis(mfcc_features[:, i]) 
            for i in range(mfcc_features.shape[1])
        ])
        
        # 拼接所有统计特征
        n_mfcc = mfcc_features.shape[1]
        statistical_features = np.concatenate([
            mean_features,        # n_mfcc
            std_features,         # n_mfcc
            max_features,         # n_mfcc
            min_features,         # n_mfcc
            delta_mean,           # n_mfcc
            delta_std,            # n_mfcc
            delta_delta_mean,     # n_mfcc
            skew_features,        # n_mfcc
            kurtosis_features,    # n_mfcc
        ])  # 总共 9 * n_mfcc 维
        
        return statistical_features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度（skewness）"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度（kurtosis）"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0  # 减3使其为0均值
    
    def _simple_classifier(self, mfcc_features: np.ndarray) -> Tuple[str, float]:
        """
        基于统计特征的简单情感分类器
        
        使用MFCC特征的统计量（均值、标准差等）进行简单的情感判断
        支持4类或7类情感识别，根据use_simple_emotions参数决定
        
        Args:
            mfcc_features: MFCC特征矩阵
        
        Returns:
            (情感类别, 置信度)
        """
        if len(mfcc_features) == 0:
            default_emotion = "平静" if not self.use_simple_emotions else "平静"
            return default_emotion, 0.5
        
        # 提取统计特征
        stats = self._extract_statistical_features(mfcc_features)
        
        # 简单的规则分类（基于MFCC特征的统计特性）
        # 这些规则是基于经验，实际应用中应该使用训练好的模型
        
        # 计算MFCC特征的动态范围（标准差）
        mfcc_std = np.std(mfcc_features, axis=0)
        avg_std = np.mean(mfcc_std)
        
        # 计算MFCC特征的能量（均值）
        mfcc_mean = np.mean(mfcc_features, axis=0)
        avg_mean = np.mean(np.abs(mfcc_mean))
        
        # 计算MFCC特征的变化率（相邻帧的差异）
        if len(mfcc_features) > 1:
            mfcc_diff = np.diff(mfcc_features, axis=0)
            avg_diff = np.mean(np.abs(mfcc_diff))
        else:
            avg_diff = 0.0
        
        # 计算MFCC特征的峰值（最大值）
        mfcc_max = np.max(mfcc_features)
        mfcc_min = np.min(mfcc_features)
        mfcc_range = mfcc_max - mfcc_min
        
        # 计算delta特征的统计量
        delta_features = self._compute_delta(mfcc_features, N=2)
        delta_mean = np.mean(np.abs(delta_features))
        delta_std = np.std(delta_features)
        
        # 计算delta-delta特征的统计量
        delta_delta_features = self._compute_delta(delta_features, N=2)
        delta_delta_mean = np.mean(np.abs(delta_delta_features))
        
        # 计算MFCC特征的偏度和峰度
        mfcc_skew = np.mean([self._compute_skewness(mfcc_features[:, i]) 
                            for i in range(mfcc_features.shape[1])])
        mfcc_kurtosis = np.mean([self._compute_kurtosis(mfcc_features[:, i]) 
                                for i in range(mfcc_features.shape[1])])
        
        # 计算能量特征（RMS）
        mfcc_rms = np.sqrt(np.mean(mfcc_features ** 2))
        
        # 计算零交叉率（ZCR）的近似值（基于MFCC变化）
        zero_crossings = np.sum(np.diff(np.sign(mfcc_features - np.mean(mfcc_features)), axis=0) != 0)
        zcr = zero_crossings / len(mfcc_features) if len(mfcc_features) > 0 else 0
        
        # 基于增强特征的经验规则进行情感判断
        # 注意：这些规则基于MFCC、delta、delta-delta和统计特征
        
        if self.use_simple_emotions:
            # 4类情感模式：平静、高兴、悲伤、愤怒
            # 愤怒：通常MFCC特征变化很大，能量很高
            if avg_std > 2.5 and avg_mean > 0.6:
                emotion = "愤怒"
                confidence = min(0.65 + (avg_std - 2.5) * 0.1, 0.9)
            # 高兴：通常MFCC特征变化较大，能量较高
            elif avg_std > 2.0 and avg_mean > 0.5:
                emotion = "高兴"
                confidence = min(0.7 + (avg_std - 2.0) * 0.1, 0.95)
            # 悲伤：通常MFCC特征变化较小，能量较低
            elif avg_std < 1.5 and avg_mean < 0.3:
                emotion = "悲伤"
                confidence = min(0.7 + (1.5 - avg_std) * 0.1, 0.9)
            # 平静：其他情况
            else:
                emotion = "平静"
                confidence = 0.6
        else:
            # 7类情感模式：愤怒、厌恶、恐惧、高兴、平静、悲伤、惊讶
            # 愤怒：通常MFCC特征变化很大，能量很高，峰值高
            if avg_std > 2.8 and avg_mean > 0.7 and mfcc_range > 3.0:
                emotion = "愤怒"
                confidence = min(0.65 + (avg_std - 2.8) * 0.1, 0.9)
            # 厌恶：通常MFCC特征变化大，能量中等偏高，但峰值较低
            elif avg_std > 2.3 and avg_mean > 0.55 and avg_mean < 0.7 and mfcc_range < 2.5:
                emotion = "厌恶"
                confidence = min(0.65 + (avg_std - 2.3) * 0.1, 0.85)
            # 恐惧：通常MFCC特征变化很大，能量高，但变化率不稳定
            elif avg_std > 2.6 and avg_mean > 0.65 and avg_diff > 1.5:
                emotion = "恐惧"
                confidence = min(0.65 + (avg_std - 2.6) * 0.1, 0.85)
            # 惊讶：通常MFCC特征变化突然，能量突然升高
            elif avg_std > 2.2 and avg_mean > 0.5 and avg_diff > 1.8:
                emotion = "惊讶"
                confidence = min(0.7 + (avg_std - 2.2) * 0.1, 0.9)
            # 高兴：通常MFCC特征变化较大，能量较高，变化稳定
            elif avg_std > 2.0 and avg_mean > 0.5 and avg_diff < 1.5:
                emotion = "高兴"
                confidence = min(0.7 + (avg_std - 2.0) * 0.1, 0.95)
            # 悲伤：通常MFCC特征变化较小，能量较低
            elif avg_std < 1.5 and avg_mean < 0.3:
                emotion = "悲伤"
                confidence = min(0.7 + (1.5 - avg_std) * 0.1, 0.9)
            # 平静：其他情况（中等变化，中等能量）
            else:
                emotion = "平静"
                confidence = 0.6
        
        return emotion, float(confidence)
    
    def recognize_from_pcm(
        self,
        pcm_data: bytes,
        sample_rate: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        从PCM音频数据识别情感
        
        Args:
            pcm_data: PCM格式的音频字节数据（16bit, 小端序）
            sample_rate: 采样率，如果为None则使用初始化时的采样率
        
        Returns:
            (情感类别, 置信度)
        """
        try:
            # 提取MFCC特征（根据模型类型使用不同的MFCC维度）
            if self.is_pytorch_model:
                # PyTorch模型需要40维MFCC
                mfcc_extractor = MFCCExtractor(
                    sample_rate=sample_rate or self.mfcc_extractor.sample_rate,
                    n_mfcc=40
                )
                mfcc_features = mfcc_extractor.extract_from_pcm_bytes(
                    pcm_data,
                    sample_rate=sample_rate
                )
            else:
                # 其他模型使用默认MFCC维度
                mfcc_features = self.mfcc_extractor.extract_from_pcm_bytes(
                    pcm_data,
                    sample_rate=sample_rate
                )
            
            # 如果使用PyTorch LSTM/CNN模型
            if self.is_pytorch_model and self.model is not None:
                emotion, confidence = self.model.predict(mfcc_features)
                logger.bind(tag=TAG).debug(
                    f"情感识别结果: {emotion}, 置信度={confidence:.3f}, "
                    f"MFCC特征形状={mfcc_features.shape}"
                )
                return emotion, confidence
            
            # 如果使用其他预训练模型
            elif self.model is not None:
                # 提取特征
                if self.feature_extractor:
                    features = self.feature_extractor(mfcc_features)
                else:
                    features = self._extract_statistical_features(mfcc_features)
                
                # 预测情感
                if hasattr(self.model, 'predict_proba'):
                    # 如果有概率预测方法
                    proba = self.model.predict_proba(features.reshape(1, -1))[0]
                    emotion_idx = np.argmax(proba)
                    confidence = float(proba[emotion_idx])
                elif hasattr(self.model, 'predict'):
                    # 只有预测方法
                    emotion_idx = self.model.predict(features.reshape(1, -1))[0]
                    confidence = 0.7  # 默认置信度
                else:
                    # 如果模型不支持标准接口，回退到简单分类器
                    return self._simple_classifier(mfcc_features)
                
                # 映射到情感类别
                emotion = self.emotion_map.get(emotion_idx, "平静")
                
                logger.bind(tag=TAG).debug(
                    f"情感识别结果: {emotion}, 置信度={confidence:.3f}, "
                    f"MFCC特征形状={mfcc_features.shape}"
                )
                
                return emotion, confidence
            else:
                # 使用简单分类器
                return self._simple_classifier(mfcc_features)
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"情感识别失败: {e}", exc_info=True)
            return "平静", 0.5  # 失败时返回平静
    
    def recognize_from_wav_bytes(
        self,
        wav_data: bytes
    ) -> Tuple[str, float]:
        """
        从WAV格式字节数据识别情感
        
        Args:
            wav_data: WAV格式的音频字节数据
        
        Returns:
            (情感类别, 置信度)
        """
        try:
            # 提取MFCC特征
            mfcc_features = self.mfcc_extractor.extract_from_wav_bytes(wav_data)
            
            # 如果使用预训练模型
            if self.model is not None:
                # 提取特征
                if self.feature_extractor:
                    features = self.feature_extractor(mfcc_features)
                else:
                    features = self._extract_statistical_features(mfcc_features)
                
                # 预测情感
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(features.reshape(1, -1))[0]
                    emotion_idx = np.argmax(proba)
                    confidence = float(proba[emotion_idx])
                elif hasattr(self.model, 'predict'):
                    emotion_idx = self.model.predict(features.reshape(1, -1))[0]
                    confidence = 0.7
                else:
                    return self._simple_classifier(mfcc_features)
                
                emotion = self.emotion_map.get(emotion_idx, "平静")
                
                logger.bind(tag=TAG).debug(
                    f"情感识别结果: {emotion}, 置信度={confidence:.3f}"
                )
                
                return emotion, confidence
            else:
                return self._simple_classifier(mfcc_features)
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"情感识别失败: {e}", exc_info=True)
            return "中性", 0.5


# 全局情感识别器实例（延迟初始化）
_emotion_recognizer: Optional[EmotionRecognizer] = None


def get_emotion_recognizer(
    model_path: Optional[str] = None,
    use_simple_emotions: bool = True,
    sample_rate: int = 16000
) -> EmotionRecognizer:
    """
    获取情感识别器实例（单例模式）
    
    Args:
        model_path: 模型路径
        use_simple_emotions: 是否使用简化情感类别
        sample_rate: 采样率
    
    Returns:
        情感识别器实例
    """
    global _emotion_recognizer
    
    if _emotion_recognizer is None:
        _emotion_recognizer = EmotionRecognizer(
            model_path=model_path,
            use_simple_emotions=use_simple_emotions,
            sample_rate=sample_rate
        )
    
    return _emotion_recognizer


def recognize_emotion_from_pcm(
    pcm_data: bytes,
    model_path: Optional[str] = None,
    use_simple_emotions: bool = True,
    sample_rate: int = 16000
) -> Tuple[str, float]:
    """
    便捷函数：从PCM数据识别情感
    
    Args:
        pcm_data: PCM音频数据
        model_path: 模型路径（可选）
        use_simple_emotions: 是否使用简化情感类别
        sample_rate: 采样率
    
    Returns:
        (情感类别, 置信度)
    """
    recognizer = get_emotion_recognizer(
        model_path=model_path,
        use_simple_emotions=use_simple_emotions,
        sample_rate=sample_rate
    )
    return recognizer.recognize_from_pcm(pcm_data, sample_rate)

