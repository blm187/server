import os
import numpy as np
import torch
import tempfile
from typing import List, Dict, Any, Optional
from config.logger import setup_logging
from core.providers.speech_separation.base import SpeechSeparationProviderBase

TAG = __name__
logger = setup_logging()


class SpeechSeparationProvider(SpeechSeparationProviderBase):
    """Conv-TasNet完整盲源分离Pipeline（5阶段）"""
    
    def __init__(self, config: dict):
        """
        初始化Conv-TasNet语音分离提供者
        
        Args:
            config: 配置字典，包含：
                - model_name: Conv-TasNet模型名称或路径
                - max_speakers: 最多说话人数量（默认3）
                - similarity_threshold: 聚类相似度阈值（默认0.4）
                - use_webrtc_vad: 是否使用WebRTC VAD（默认True）
                - use_ecapa: 是否使用ECAPA-TDNN（默认True）
                - cache_dir: 模型缓存目录（可选，默认使用HuggingFace默认缓存）
        """
        self.config = config
        self.model_name = config.get("model_name", "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
        self.max_speakers = config.get("max_speakers", 3)
        self.model_n_src = None  # 模型实际支持的说话人数量（加载模型后设置）
        self.similarity_threshold = config.get("similarity_threshold", 0.4)
        self.use_webrtc_vad = config.get("use_webrtc_vad", True)
        self.use_ecapa = config.get("use_ecapa", True)
        self.cache_dir = config.get("cache_dir", None)  # 模型缓存目录
        self.sample_rate = 16000
        
        # 新增配置项
        self.energy_threshold = config.get("energy_threshold", 0.015)  # 能量阈值（使用配置值）
        self.skip_single_speaker = config.get("skip_single_speaker", False)  # 是否跳过单人场景
        self.use_adaptive_threshold = config.get("use_adaptive_threshold", True)  # 是否使用动态阈值
        self.single_speaker_energy_std_threshold = config.get("single_speaker_energy_std_threshold", 0.15)  # 单人场景能量标准差阈值
        
        # 单人检测方法配置
        self.single_speaker_detection_method = config.get("single_speaker_detection_method", "enhanced")  # "simple" | "enhanced" | "xvector" | "hybrid"
        self.single_speaker_xvector_threshold = config.get("single_speaker_xvector_threshold", 0.7)  # x-vector相似度阈值
        
        # 流式处理配置
        self.streaming_enabled = config.get("streaming_enabled", False)  # 是否启用流式处理
        self.chunk_size_ms = config.get("chunk_size_ms", 400)  # 块大小（毫秒）
        self.context_size_ms = config.get("context_size_ms", 200)  # 上下文大小（毫秒）
        self.overlap_size_ms = config.get("overlap_size_ms", 50)  # 重叠大小（毫秒），用于边界平滑
        
        # 流式处理状态
        self.streaming_buffer = {}  # {speaker_id: list of chunks} 用于流式缓冲
        self.streaming_context = {}  # {speaker_id: context_audio} 用于上下文缓存
        
        # 流式聚类状态（Single-Pass聚类）
        self.streaming_cluster_centers = []  # 簇中心列表 [center1, center2, ...]
        self.streaming_cluster_weights = []  # 簇权重列表（用于EMA更新）
        self.streaming_cluster_count = 0  # 簇数量
        self.streaming_similarity_threshold = config.get("similarity_threshold", 0.6)  # Single-Pass相似度阈值（0.65-0.75推荐）
        self.streaming_max_clusters = config.get("max_speakers", 3)  # 最大簇数量（防止内存爆炸）
        self.streaming_ema_alpha = 0.3  # EMA更新系数（α=0.3，抗顺序抖动）
        
        # 流式VAD状态（滑动窗口+状态机）
        self.streaming_vad_state = "OFF"  # ON / OFF
        self.streaming_vad_on_duration = 0.0  # ON状态持续时间（秒）
        self.streaming_vad_off_duration = 0.0  # OFF状态持续时间（秒）
        self.streaming_vad_min_on_duration = 0.2  # 最少ON持续时间（200ms）
        self.streaming_vad_min_off_duration = 0.2  # 最少OFF持续时间（200ms）
        self.streaming_vad_window_size_ms = 30  # 滑动窗口大小（30ms）
        self.streaming_vad_window_buffer = []  # 滑动窗口缓冲区
        
        # 因果缓存对齐（ring-buffer）
        self.causal_buffer_size_ms = 2000  # 2s因果感受野
        self.causal_buffer_size_samples = int(self.causal_buffer_size_ms * 16000 / 1000)  # 32000 samples
        self.causal_hop_ms = 16  # hop步长（16ms，对应模型stride=16）
        self.causal_hop_samples = int(self.causal_hop_ms * 16000 / 1000)  # 256 samples
        self.causal_ring_buffer = None  # ring-buffer（延迟初始化）
        self.causal_buffer_index = 0
        
        # 实时AGC配置
        self.agc_enabled = config.get("agc_enabled", True)  # 是否启用实时AGC
        self.agc_target_rms = config.get("agc_target_rms", 0.1)  # 目标RMS
        self.agc_max_gain_db = 12.0  # 最大增益（12dB = 3.98倍）
        self.agc_max_gain = 10 ** (self.agc_max_gain_db / 20)  # 转换为线性增益
        
        # SNR估计和动态阈值调整
        self.snr_estimation_enabled = config.get("snr_estimation_enabled", True)  # 是否启用SNR估计
        self.current_snr_db = 20.0  # 当前SNR（默认20dB）
        
        # 模型实例
        self.separation_model = None
        self.vad = None
        self.ecapa_model = None
        
        # 加载状态
        self._separation_loaded = False
        self._vad_loaded = False
        self._ecapa_loaded = False
        
        logger.bind(tag=TAG).info("Conv-TasNet完整Pipeline已初始化（延迟加载模式）")
    
    def _ensure_separation_model_loaded(self):
        """确保Conv-TasNet模型已加载"""
        if self._separation_loaded and self.separation_model is not None:
            return
        
        try:
            import time
            start_time = time.time()
            logger.bind(tag=TAG).info(f"正在加载Conv-TasNet模型: {self.model_name}")
            
            # 检查是否是本地路径
            is_local_path = (
                self.model_name.startswith("models/") or 
                self.model_name.startswith("/") or 
                os.path.exists(self.model_name)
            )
            
            from asteroid.models import ConvTasNet
            
            if is_local_path:
                logger.bind(tag=TAG).info(f"检测到本地模型路径，从本地加载: {self.model_name}")
                # 确保路径存在
                if not os.path.exists(self.model_name):
                    raise FileNotFoundError(f"模型路径不存在: {self.model_name}")
                
                # 查找模型文件
                model_file = os.path.join(self.model_name, "pytorch_model.bin")
                if not os.path.exists(model_file):
                    # 尝试直接使用路径作为模型文件
                    if os.path.isfile(self.model_name):
                        model_file = self.model_name
                    else:
                        raise FileNotFoundError(f"模型文件不存在: {model_file}")
                
                logger.bind(tag=TAG).info(f"从本地文件加载模型: {model_file}")
                
                # 直接加载本地模型文件，避免网络请求
                try:
                    # 加载checkpoint
                    checkpoint = torch.load(model_file, map_location="cpu")
                    
                    # 检查checkpoint格式
                    if not isinstance(checkpoint, dict):
                        raise ValueError(f"模型文件格式错误: 期望字典，得到 {type(checkpoint)}")
                    
                    # 获取state_dict
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    elif "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        raise ValueError("模型文件中未找到state_dict或model_state_dict")
                    
                    # 从checkpoint获取模型参数（如果存在）
                    if "model_args" in checkpoint:
                        model_args = checkpoint["model_args"]
                        logger.bind(tag=TAG).info(f"从checkpoint读取模型参数: {model_args}")
                        
                        # 检查模型支持的说话人数量
                        model_n_src = model_args.get("n_src", 2)
                        if model_n_src < self.max_speakers:
                            logger.bind(tag=TAG).warning(
                                f"⚠️ 模型只支持 {model_n_src} 个说话人，但配置要求最多 {self.max_speakers} 个。"
                                f"对于 {self.max_speakers} 声源场景，分离效果可能不佳。"
                                f"建议：使用支持 {self.max_speakers} 人的模型，或将 max_speakers 改为 {model_n_src}。"
                            )
                        
                        # 创建模型实例
                        self.separation_model = ConvTasNet(**model_args)
                        # 保存模型实际支持的说话人数量
                        self.model_n_src = model_n_src
                    else:
                        # 使用README中的配置参数
                        logger.bind(tag=TAG).info("使用默认模型配置参数")
                        self.separation_model = ConvTasNet(
                            n_src=2,  # 2个说话人
                            n_filters=512,
                            kernel_size=32,
                            stride=16,
                            n_repeats=3,
                            n_blocks=8,
                            bn_chan=128,
                            hid_chan=512,
                            skip_chan=128,
                            mask_act="relu",
                        )
                    
                    # 加载权重
                    self.separation_model.load_state_dict(state_dict, strict=False)
                    logger.bind(tag=TAG).info("模型权重加载成功（离线模式）")
                    
                except Exception as e:
                    logger.bind(tag=TAG).error(f"直接加载本地模型失败: {e}")
                    import traceback
                    logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
                    raise RuntimeError(f"无法加载本地模型: {model_file}，错误: {e}")
            else:
                logger.bind(tag=TAG).info("提示：首次加载需要从HuggingFace下载模型，可能需要1-3分钟，请耐心等待...")
                # 检查并设置HuggingFace镜像（如果未设置）
                hf_endpoint = os.environ.get("HF_ENDPOINT")
                if not hf_endpoint:
                    # 尝试使用国内镜像
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                    logger.bind(tag=TAG).info("已自动设置HuggingFace镜像: https://hf-mirror.com")
                    logger.bind(tag=TAG).info("提示：如果网络仍有问题，请手动设置: export HF_ENDPOINT=https://hf-mirror.com")
                else:
                    logger.bind(tag=TAG).info(f"使用HuggingFace端点: {hf_endpoint}")
                
                # 加载预训练模型（支持本地路径和HuggingFace）
                load_kwargs = {}
                if self.cache_dir:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    load_kwargs["cache_dir"] = self.cache_dir
                    logger.bind(tag=TAG).info(f"使用本地缓存目录: {self.cache_dir}")
                
                # 加载预训练模型
                self.separation_model = ConvTasNet.from_pretrained(self.model_name, **load_kwargs)
            
            if torch.cuda.is_available():
                self.separation_model = self.separation_model.cuda()
                logger.bind(tag=TAG).info("Conv-TasNet使用GPU加速")
            else:
                logger.bind(tag=TAG).info("Conv-TasNet使用CPU运行")
            
            self.separation_model.eval()
            self._separation_loaded = True
            
            load_time = time.time() - start_time
            logger.bind(tag=TAG).info(f"Conv-TasNet模型加载成功（耗时: {load_time:.2f}秒）")
            logger.bind(tag=TAG).info("提示：模型已缓存，后续加载会更快")
            
        except ImportError:
            logger.bind(tag=TAG).error("asteroid未安装，无法使用Conv-TasNet")
            logger.bind(tag=TAG).error("请运行: pip install asteroid")
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载Conv-TasNet模型失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
            self._separation_loaded = False
    
    def _ensure_vad_loaded(self):
        """确保WebRTC VAD已加载"""
        if self._vad_loaded and self.vad is not None:
            return
        
        if not self.use_webrtc_vad:
            self._vad_loaded = True
            return
        
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(3)  # 模式3：最激进
            self._vad_loaded = True
            logger.bind(tag=TAG).info("WebRTC VAD加载成功")
        except ImportError:
            logger.bind(tag=TAG).warning("webrtcvad未安装，将跳过VAD切分")
            logger.bind(tag=TAG).warning("请运行: pip install webrtcvad")
            self.use_webrtc_vad = False
            self._vad_loaded = True
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载WebRTC VAD失败: {e}")
            self.use_webrtc_vad = False
            self._vad_loaded = True
    
    def _ensure_ecapa_loaded(self):
        """确保ECAPA-TDNN模型已加载"""
        if self._ecapa_loaded:
            return
        
        if not self.use_ecapa:
            self._ecapa_loaded = True
            return
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            # 修复版本兼容性问题：新版本huggingface_hub移除了use_auth_token参数
            # 设置环境变量以兼容旧版本speechbrain
            import os
            original_hf_token = os.environ.get("HF_TOKEN", None)
            
            # 尝试设置离线模式或使用镜像
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ["HF_ENDPOINT"] = hf_endpoint
            
            try:
                # 使用SpeechBrain的ECAPA-TDNN预训练模型
                self.ecapa_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
                self._ecapa_loaded = True
                logger.bind(tag=TAG).info("ECAPA-TDNN模型加载成功")
            except TypeError as e:
                # 如果是参数错误，尝试降级huggingface_hub或使用离线模式
                if "use_auth_token" in str(e) or "unexpected keyword argument" in str(e):
                    logger.bind(tag=TAG).warning(f"ECAPA-TDNN版本兼容性问题: {e}")
                    logger.bind(tag=TAG).warning("建议：pip install 'huggingface_hub<0.20' 或升级speechbrain")
                    logger.bind(tag=TAG).warning("将使用简化版x-vector提取")
                    self.use_ecapa = False
                    self._ecapa_loaded = True
                else:
                    raise
            finally:
                # 恢复原始token设置
                if original_hf_token:
                    os.environ["HF_TOKEN"] = original_hf_token
                elif "HF_TOKEN" in os.environ:
                    del os.environ["HF_TOKEN"]
                    
        except ImportError:
            logger.bind(tag=TAG).warning("speechbrain未安装，将使用简化版x-vector提取")
            logger.bind(tag=TAG).warning("请运行: pip install speechbrain")
            self.use_ecapa = False
            self._ecapa_loaded = True
        except Exception as e:
            logger.bind(tag=TAG).warning(f"加载ECAPA-TDNN失败: {e}，将使用简化版")
            self.use_ecapa = False
            self._ecapa_loaded = True
    
    # ===== 音频预处理 =====
    def _preprocess_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """
        音频预处理：去除DC偏移、归一化、能量归一化
        
        Args:
            audio_np: 原始音频数组（float32，范围[-1, 1]）
        
        Returns:
            预处理后的音频数组（范围[-0.95, 0.95]）
        """
        # 预处理：只做必要的clip，保持原始音频特性
        # 1. 去除DC偏移（轻微处理，不影响动态范围）
        dc_offset = np.mean(audio_np)
        if abs(dc_offset) > 0.01:  # 只在DC偏移明显时才去除
            audio_np = audio_np - dc_offset
        
        # 2. 关键：clip到[-0.999, 0.999]防止幅度爆炸和整型饱和
        # 不进行归一化，保持原始音频的动态范围和特性
        # Conv-TasNet对幅度极度敏感，输入峰值>1.0会溢出导致噪声
        # 使用0.999而不是0.95，避免整型转换时的饱和
        audio_np = np.clip(audio_np, -0.999, 0.999)
        
        return audio_np
    
    # ===== 单人场景检测 =====
    def _detect_single_speaker(self, audio_np: np.ndarray) -> bool:
        """
        检测是否为单人场景（支持多种检测方法）
        
        Args:
            audio_np: 预处理后的音频数组
        
        Returns:
            True表示可能是单人场景，False表示可能是多人场景
        """
        method = getattr(self, 'single_speaker_detection_method', 'enhanced')
        
        if method == "simple":
            return self._detect_single_speaker_simple(audio_np)
        elif method == "enhanced":
            return self._detect_single_speaker_enhanced(audio_np)
        elif method == "xvector":
            return self._detect_single_speaker_by_xvector(audio_np)
        elif method == "hybrid":
            # 混合方案：优先使用x-vector，降级到enhanced
            if self.use_ecapa and self._ecapa_loaded:
                return self._detect_single_speaker_by_xvector(audio_np)
            else:
                return self._detect_single_speaker_enhanced(audio_np)
        else:
            # 默认使用enhanced
            return self._detect_single_speaker_enhanced(audio_np)
    
    def _detect_single_speaker_simple(self, audio_np: np.ndarray) -> bool:
        """
        简单方法：仅使用能量标准差
        
        Args:
            audio_np: 预处理后的音频数组
        
        Returns:
            True表示可能是单人场景
        """
        # 计算能量分布的标准差
        energy = np.abs(audio_np)
        energy_std = np.std(energy)
        
        # 阈值：如果能量分布很集中，可能是单人
        threshold = self.single_speaker_energy_std_threshold
        
        is_single = energy_std < threshold
        
        if is_single:
            logger.bind(tag=TAG).info(
                f"[简单检测] 检测到单人场景（能量标准差: {energy_std:.4f} < {threshold}）"
            )
        else:
            logger.bind(tag=TAG).debug(
                f"[简单检测] 检测到多人场景（能量标准差: {energy_std:.4f} >= {threshold}）"
            )
        
        return is_single
    
    def _detect_single_speaker_enhanced(self, audio_np: np.ndarray) -> bool:
        """
        增强方法：多特征融合（能量标准差 + 归一化标准差 + 时间域变化率）
        
        Args:
            audio_np: 预处理后的音频数组
        
        Returns:
            True表示可能是单人场景
        """
        # 特征1：能量标准差
        energy = np.abs(audio_np)
        energy_std = np.std(energy)
        energy_mean = np.mean(energy)
        
        # 特征2：归一化标准差（更鲁棒）
        normalized_std = energy_std / (energy_mean + 1e-6)
        
        # 特征3：时间域变化率
        energy_diff = np.diff(energy)
        change_rate = np.std(energy_diff) / (energy_mean + 1e-6)
        
        # 特征4：能量包络平滑度
        # 计算能量包络的局部方差
        window_size = int(self.sample_rate * 0.1)  # 100ms窗口
        if len(energy) > window_size:
            local_vars = []
            for i in range(0, len(energy) - window_size, window_size // 2):
                window = energy[i:i+window_size]
                local_vars.append(np.var(window))
            smoothness = np.std(local_vars) / (np.mean(local_vars) + 1e-6)
        else:
            smoothness = 0.0
        
        # 综合得分（加权平均）
        # 归一化各特征到[0, 1]范围
        score = (
            min(normalized_std / 0.5, 1.0) * 0.4 +  # 归一化标准差（权重40%）
            min(change_rate / 0.3, 1.0) * 0.3 +     # 变化率（权重30%）
            min(smoothness / 0.2, 1.0) * 0.3        # 平滑度（权重30%）
        )
        
        # 阈值判断
        threshold = 0.5  # 可配置
        is_single = score < threshold
        
        if is_single:
            logger.bind(tag=TAG).info(
                f"[增强检测] 检测到单人场景（得分: {score:.4f} < {threshold}, "
                f"归一化标准差: {normalized_std:.4f}, 变化率: {change_rate:.4f}, 平滑度: {smoothness:.4f}）"
            )
        else:
            logger.bind(tag=TAG).debug(
                f"[增强检测] 检测到多人场景（得分: {score:.4f} >= {threshold}, "
                f"归一化标准差: {normalized_std:.4f}, 变化率: {change_rate:.4f}, 平滑度: {smoothness:.4f}）"
            )
        
        return is_single
    
    def _detect_single_speaker_by_xvector(self, audio_np: np.ndarray) -> bool:
        """
        基于x-vector的单人检测（最准确）
        
        方法：将音频分段，提取每段的x-vector，计算相似度
        
        Args:
            audio_np: 预处理后的音频数组
        
        Returns:
            True表示可能是单人场景
        """
        if not self.use_ecapa or not self._ecapa_loaded:
            logger.bind(tag=TAG).warning("ECAPA未加载，降级到增强方法")
            return self._detect_single_speaker_enhanced(audio_np)
        
        # 将音频分段（每段3秒）
        segment_length = 3.0  # 秒
        segment_samples = int(segment_length * self.sample_rate)
        
        if len(audio_np) < segment_samples:
            # 音频太短，无法分段，使用增强方法
            logger.bind(tag=TAG).debug("音频太短，无法使用x-vector检测，降级到增强方法")
            return self._detect_single_speaker_enhanced(audio_np)
        
        segments = []
        for i in range(0, len(audio_np), segment_samples):
            seg = audio_np[i:i+segment_samples]
            if len(seg) >= segment_samples * 0.5:  # 至少50%长度
                segments.append(seg)
        
        if len(segments) < 2:
            # 分段太少，无法判断
            logger.bind(tag=TAG).debug("分段太少，默认单人场景")
            return True
        
        # 提取每段的x-vector
        xvectors = []
        for seg in segments:
            try:
                xvector = self._extract_xvector(seg)
                xvectors.append(xvector)
            except Exception as e:
                logger.bind(tag=TAG).debug(f"提取x-vector失败: {e}，跳过该段")
                continue
        
        if len(xvectors) < 2:
            logger.bind(tag=TAG).debug("x-vector数量不足，默认单人场景")
            return True
        
        # 计算所有x-vector对的相似度
        xvectors = np.array(xvectors)
        similarity_matrix = np.dot(xvectors, xvectors.T)
        
        # 提取上三角矩阵（避免重复）
        n = len(xvectors)
        upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
        
        if len(upper_tri) == 0:
            return True
        
        # 计算平均相似度
        avg_similarity = np.mean(upper_tri)
        std_similarity = np.std(upper_tri)
        
        # 阈值判断
        threshold = getattr(self, 'single_speaker_xvector_threshold', 0.7)
        is_single = avg_similarity > threshold
        
        if is_single:
            logger.bind(tag=TAG).info(
                f"[x-vector检测] 检测到单人场景（平均相似度: {avg_similarity:.4f} > {threshold}, "
                f"标准差: {std_similarity:.4f}, 分段数: {len(xvectors)}）"
            )
        else:
            logger.bind(tag=TAG).debug(
                f"[x-vector检测] 检测到多人场景（平均相似度: {avg_similarity:.4f} <= {threshold}, "
                f"标准差: {std_similarity:.4f}, 分段数: {len(xvectors)}）"
            )
        
        return is_single
    
    # ===== 边界平滑处理 =====
    def _smooth_boundary(
        self, 
        prev_chunk: np.ndarray, 
        curr_chunk: np.ndarray, 
        overlap_samples: int
    ) -> np.ndarray:
        """
        使用重叠-相加法平滑音频块边界
        
        Args:
            prev_chunk: 前一个音频块
            curr_chunk: 当前音频块
            overlap_samples: 重叠采样点数
        
        Returns:
            平滑后的合并音频
        """
        if overlap_samples <= 0 or len(prev_chunk) < overlap_samples or len(curr_chunk) < overlap_samples:
            # 如果重叠太小或块太小，直接拼接
            return np.concatenate([prev_chunk, curr_chunk])
        
        # 提取重叠区域
        prev_overlap = prev_chunk[-overlap_samples:]
        curr_overlap = curr_chunk[:overlap_samples]
        
        # 创建淡入淡出窗口（汉宁窗）
        # 关键：确保窗函数满足"平方和恒定"性质，适合重叠-相加法（OLA）
        # 对于50%重叠（hop = frame/2），需要归一化使fade_out^2 + fade_in^2 = 1
        full_window = np.hanning(overlap_samples * 2)
        fade_out = full_window[:overlap_samples]
        fade_in = full_window[overlap_samples:]
        
        # 归一化：确保平方和恒定（用于OLA）
        # 对于50%重叠，fade_out^2 + fade_in^2 应该归一化到1
        sum_squared = np.sum(fade_out**2 + fade_in**2)
        if sum_squared > 0:
            # 归一化使平方和等于overlap_samples（保持能量）
            fade_out = fade_out / np.sqrt(sum_squared / overlap_samples)
            fade_in = fade_in / np.sqrt(sum_squared / overlap_samples)
        
        # 平滑重叠区域（使用归一化的汉宁窗，满足平方和恒定）
        smoothed_overlap = prev_overlap * fade_out + curr_overlap * fade_in
        
        # 拼接：前块（不含重叠）+ 平滑重叠 + 当前块（不含重叠）
        result = np.concatenate([
            prev_chunk[:-overlap_samples],
            smoothed_overlap,
            curr_chunk[overlap_samples:]
        ])
        
        return result
    
    # ===== 流式分离处理 =====
    async def _separate_sources_streaming(
        self, 
        audio_chunk: bytes,
        is_first_chunk: bool = False,
        is_last_chunk: bool = False
    ) -> Dict[str, List[np.ndarray]]:
        """
        流式分离：处理单个音频块
        
        Args:
            audio_chunk: 音频块数据（PCM bytes）
            is_first_chunk: 是否是第一个块
            is_last_chunk: 是否是最后一个块
        
        Returns:
            {speaker_id: [audio_chunk]} 分离后的音频块字典
        """
        self._ensure_separation_model_loaded()
        
        if not self._separation_loaded or self.separation_model is None:
            raise RuntimeError("Conv-TasNet模型未加载")
        
        # 将PCM bytes转换为numpy数组
        chunk_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 关键：先clip到[-0.999, 0.999]防止输入幅度问题和整型饱和
        chunk_np = np.clip(chunk_np, -0.999, 0.999)
        
        # 音频预处理（内部也会clip，但这里先确保）
        chunk_np = self._preprocess_audio(chunk_np)
        
        # 如果是第一个块，初始化上下文和聚类状态
        if is_first_chunk:
            self.streaming_context = {}
            self.streaming_buffer = {}
            # 重置流式VAD状态
            self.streaming_vad_state = "OFF"
            self.streaming_vad_on_duration = 0.0
            self.streaming_vad_off_duration = 0.0
            self.streaming_vad_window_buffer = []
            # 重置因果缓存
            self.causal_ring_buffer = None
            self.causal_buffer_index = 0
            # 注意：流式聚类状态在_separate_streaming_mode中初始化，这里不重置
        
        # 1. 实时AGC（自动增益控制）
        chunk_np = self._streaming_agc(chunk_np)
        
        # 2. 流式VAD检测（滑动窗口+状态机）- 暂时禁用，避免过滤掉所有块
        # chunk_duration_ms = len(chunk_np) / self.sample_rate * 1000
        # has_speech = self._streaming_vad(chunk_np, chunk_duration_ms)
        # 
        # if not has_speech:
        #     # 如果没有语音，返回空字典（跳过分离）
        #     logger.bind(tag=TAG).debug(f"VAD检测为无语音，跳过分离")
        #     return {}
        
        # 3. 因果缓存对齐（ring-buffer，2s因果感受野）
        chunk_with_context = self._add_to_causal_buffer(chunk_np)
        
        # 4. 估计SNR并动态调整阈值
        snr_db = self._estimate_snr(chunk_np)
        self._adaptive_threshold_adjustment(snr_db)
        
        # 转换为tensor
        audio_tensor = torch.from_numpy(chunk_with_context).float().unsqueeze(0).unsqueeze(0)
        
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
        
        # 分离
        with torch.no_grad():
            separated = self.separation_model(audio_tensor)  # [1, K, T]
        
        # 如果有上下文，需要去除上下文部分
        if not is_first_chunk and len(self.streaming_context) > 0:
            context_samples = int(self.context_size_ms * self.sample_rate / 1000)
            separated = separated[:, :, context_samples:]  # 去除前文上下文
        
        # 转换为numpy数组列表，并过滤低能量源
        separated_chunks = {}
        
        for k in range(separated.shape[1]):
            source_k = separated[0, k, :].cpu().numpy()
            
            # 关键：输出后立即clip到[-0.999, 0.999]防止幅度爆炸和整型饱和
            # 不进行额外的归一化，保持模型原始输出特性
            source_k = np.clip(source_k, -0.999, 0.999)
            
            # 计算源的能量
            source_energy = np.mean(np.abs(source_k))
            
            # 过滤低能量源
            if source_energy < self.energy_threshold:
                continue
            
            speaker_id = f"SPEAKER_{k:02d}"
            
            if speaker_id not in separated_chunks:
                separated_chunks[speaker_id] = []
            
            separated_chunks[speaker_id].append(source_k)
        
        # 注意：因果缓存已经通过ring-buffer自动维护，不需要手动更新streaming_context
        # 保留streaming_context用于兼容性（如果有其他代码依赖）
        context_samples = int(self.context_size_ms * self.sample_rate / 1000)
        for speaker_id, chunks in separated_chunks.items():
            if len(chunks) > 0:
                last_chunk = chunks[-1]
                if len(last_chunk) >= context_samples:
                    self.streaming_context[speaker_id] = last_chunk[-context_samples:]
        
        # 合并同一说话人的多个块（如果有）
        result = {}
        for speaker_id, chunks in separated_chunks.items():
            if len(chunks) == 1:
                result[speaker_id] = chunks[0]
            else:
                # 多个块需要合并（带边界平滑）
                merged = chunks[0]
                overlap_samples = int(self.overlap_size_ms * self.sample_rate / 1000)
                for i in range(1, len(chunks)):
                    merged = self._smooth_boundary(merged, chunks[i], overlap_samples)
                result[speaker_id] = merged
        
        return result
    
    # ===== 阶段1：Conv-TasNet源分离 =====
    def _separate_sources(self, audio_data: bytes) -> List[np.ndarray]:
        """阶段1：使用Conv-TasNet分离音频源"""
        self._ensure_separation_model_loaded()
        
        if not self._separation_loaded or self.separation_model is None:
            raise RuntimeError("Conv-TasNet模型未加载")
        
        # 将PCM bytes转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # ===== 音频预处理 =====
        audio_np = self._preprocess_audio(audio_np)
        
        # ===== 单人场景检测 =====
        if self.skip_single_speaker:
            if self._detect_single_speaker(audio_np):
                logger.bind(tag=TAG).info("单人场景，跳过分离，直接返回原音频")
                return [audio_np]
        
        # 转换为tensor
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
        
        # 分离
        with torch.no_grad():
            separated = self.separation_model(audio_tensor)  # [1, K, T]
        
        # 转换为numpy数组列表，并过滤低能量源（使用配置的能量阈值）
        sources = []
        
        for k in range(separated.shape[1]):
            source_k = separated[0, k, :].cpu().numpy()
            
            # 关键：输出后立即clip到[-0.999, 0.999]防止幅度爆炸和整型饱和
            # 分离模型输出可能>1.0，直接削顶会导致巨大噪声
            # 使用0.999而不是0.95，避免整型转换时的饱和
            # 不进行额外的归一化，保持模型原始输出特性
            source_k = np.clip(source_k, -0.999, 0.999)
            
            # 计算源的能量
            source_energy = np.mean(np.abs(source_k))
            
            # 如果能量太低，可能是噪声源，跳过（使用配置的阈值）
            if source_energy < self.energy_threshold:
                logger.bind(tag=TAG).debug(
                    f"源 {k} 能量过低({source_energy:.6f} < {self.energy_threshold})，可能是噪声，跳过"
                )
                continue
            
            sources.append(source_k)
        
        # 如果过滤后没有源，保留能量最高的源（单人场景）
        if len(sources) == 0:
            logger.bind(tag=TAG).warning("所有源能量都过低，保留能量最高的源")
            all_sources = [separated[0, k, :].cpu().numpy() for k in range(separated.shape[1])]
            energies = [np.mean(np.abs(s)) for s in all_sources]
            best_idx = np.argmax(energies)
            sources = [all_sources[best_idx]]
            logger.bind(tag=TAG).info(f"保留能量最高的源 {best_idx} (能量: {energies[best_idx]:.6f})")
        
            logger.bind(tag=TAG).info(
                f"阶段1完成：分离出 {len(sources)} 个有效音频源（能量阈值: {self.energy_threshold}）"
            )
            
            # 检查模型限制
            if hasattr(self, 'model_n_src') and self.model_n_src is not None:
                if len(sources) >= self.model_n_src:
                    logger.bind(tag=TAG).warning(
                        f"⚠️ 模型限制：当前模型只支持 {self.model_n_src} 个说话人分离。"
                        f"如果输入音频包含超过 {self.model_n_src} 个声源，"
                        f"多余的声源可能被合并或丢失，导致分离效果不佳。"
                        f"建议：使用支持更多说话人的模型（如 Libri3Mix 或更高）。"
                    )
        return sources
    
    # ===== 阶段2：WebRTC VAD3切分 =====
    def _vad_segmentation(self, source: np.ndarray) -> List[Dict[str, Any]]:
        """阶段2：使用WebRTC VAD切分语音段"""
        if not self.use_webrtc_vad:
            # 如果不使用VAD，返回完整音频作为一个段
            duration = len(source) / self.sample_rate
            return [{
                "start": 0.0,
                "end": duration,
                "audio": source
            }]
        
        self._ensure_vad_loaded()
        
        if self.vad is None:
            # VAD未加载，返回完整音频
            duration = len(source) / self.sample_rate
            return [{
                "start": 0.0,
                "end": duration,
                "audio": source
            }]
        
        segments = []
        frame_duration_ms = 30  # 30ms帧
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        
        is_speech = False
        speech_start = None
        
        # 按帧处理
        for i in range(0, len(source), frame_size):
            frame = source[i:i+frame_size]
            
            # 确保帧大小正确
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
            
            # 转换为int16 bytes
            # 清理NaN和Inf
            if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
                frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=-1.0)
            # 确保在有效范围内
            # 关键：clip到[-0.999, 0.999]并使用32767避免整型饱和
            frame = np.clip(frame, -0.999, 0.999)
            frame_int16 = (frame * 32767.0).astype(np.int16)
            frame_bytes = frame_int16.tobytes()
            
            # VAD检测
            try:
                is_speech_frame = self.vad.is_speech(frame_bytes, self.sample_rate)
            except:
                is_speech_frame = False
            
            # 能量检测（辅助）
            energy = np.mean(np.abs(frame))
            energy_threshold = 0.01
            
            if is_speech_frame and energy > energy_threshold:
                if not is_speech:
                    speech_start = i / self.sample_rate
                    is_speech = True
            else:
                if is_speech:
                    speech_end = i / self.sample_rate
                    # 提取语音段
                    start_idx = int(speech_start * self.sample_rate)
                    end_idx = int(speech_end * self.sample_rate)
                    segment_audio = source[start_idx:end_idx]
                    
                    if len(segment_audio) > 0:
                        segments.append({
                            "start": speech_start,
                            "end": speech_end,
                            "audio": segment_audio
                        })
                    is_speech = False
        
        # 处理最后一段
        if is_speech:
            speech_end = len(source) / self.sample_rate
            start_idx = int(speech_start * self.sample_rate)
            segment_audio = source[start_idx:]
            if len(segment_audio) > 0:
                segments.append({
                    "start": speech_start,
                    "end": speech_end,
                    "audio": segment_audio
                })
        
        logger.bind(tag=TAG).debug(f"阶段2完成：检测到 {len(segments)} 个语音段")
        return segments
    
    # ===== 阶段3：ECAPA-TDNN提取x-vector =====
    def _extract_xvector(self, audio_segment: np.ndarray) -> np.ndarray:
        """阶段3：提取x-vector（512维）"""
        # 检查段长
        duration = len(audio_segment) / self.sample_rate
        
        # 如果段长<0.8s，左右扩展0.5s
        if duration < 0.8:
            padding_samples = int(0.5 * self.sample_rate)
            audio_segment = np.pad(
                audio_segment,
                (padding_samples, padding_samples),
                mode='constant'
            )
        
        if self.use_ecapa:
            self._ensure_ecapa_loaded()
            
            if self.ecapa_model is not None:
                try:
                    # 转换为tensor
                    audio_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0)
                    
                    # 提取x-vector
                    with torch.no_grad():
                        embedding = self.ecapa_model.encode_batch(audio_tensor)
                        embedding = embedding.squeeze(0).cpu().numpy()
                    
                    # L2归一化
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    return embedding
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"ECAPA-TDNN提取失败: {e}，使用简化版")
        
        # 简化版：使用MFCC特征（如果ECAPA不可用）
        try:
            from scipy import signal
            from scipy.fft import dct
            
            # 计算MFCC特征
            # 简化的MFCC提取
            frame_length = int(0.025 * self.sample_rate)  # 25ms
            frame_shift = int(0.010 * self.sample_rate)    # 10ms
            n_fft = 512
            n_mels = 40
            
            # 分帧
            frames = []
            for i in range(0, len(audio_segment) - frame_length, frame_shift):
                frame = audio_segment[i:i+frame_length]
                frames.append(frame)
            
            if len(frames) == 0:
                # 如果帧数太少，返回零向量
                return np.zeros(512)
            
            # 计算功率谱
            power_spectrum = []
            for frame in frames:
                # 加窗
                windowed = frame * np.hamming(len(frame))
                # FFT
                fft = np.fft.rfft(windowed, n_fft)
                power = np.abs(fft) ** 2
                power_spectrum.append(power)
            
            power_spectrum = np.array(power_spectrum)
            
            # 简化的特征提取：取平均并降维到512维
            mean_power = np.mean(power_spectrum, axis=0)
            # 使用PCA或简单降维（这里简化处理）
            if len(mean_power) >= 512:
                embedding = mean_power[:512]
            else:
                embedding = np.pad(mean_power, (0, 512 - len(mean_power)), mode='constant')
            
            # L2归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            logger.bind(tag=TAG).warning(f"简化版特征提取失败: {e}")
            # 返回随机向量（最后的后备方案）
            embedding = np.random.randn(512)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    # ===== 动态阈值调整 =====
    def _compute_adaptive_threshold(self, similarity_matrix: np.ndarray) -> float:
        """
        根据相似度矩阵的分布自适应调整阈值
        
        方法：基于平均值和标准差，动态调整阈值
        
        Args:
            similarity_matrix: 相似度矩阵 [N, N]
        
        Returns:
            调整后的相似度阈值（0-1之间）
        """
        if not self.use_adaptive_threshold:
            # 如果未启用动态阈值，返回配置的固定阈值
            return self.similarity_threshold
        
        # 提取上三角矩阵（避免重复计算）
        n = similarity_matrix.shape[0]
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
        
        if len(upper_triangle) == 0:
            # 如果只有一个向量，返回固定阈值
            return self.similarity_threshold
        
        # 计算统计量
        avg_similarity = np.mean(upper_triangle)
        std_similarity = np.std(upper_triangle)
        median_similarity = np.median(upper_triangle)
        
        # 动态调整策略：
        # 1. 如果平均相似度很高（>0.7），说明可能是同一说话人，降低阈值
        # 2. 如果平均相似度很低（<0.3），说明可能是不同说话人，提高阈值
        # 3. 使用中位数作为基准，结合标准差调整
        
        if avg_similarity > 0.7:
            # 相似度很高，可能是同一说话人，降低阈值（更宽松）
            adaptive_threshold = max(0.3, median_similarity - 0.1)
        elif avg_similarity < 0.3:
            # 相似度很低，可能是不同说话人，提高阈值（更严格）
            adaptive_threshold = min(0.6, median_similarity + 0.1)
        else:
            # 中等相似度，使用中位数结合标准差
            adaptive_threshold = max(
                0.3,
                min(0.6, median_similarity - 0.5 * std_similarity)
            )
        
        # 确保阈值在合理范围内
        adaptive_threshold = max(0.3, min(0.6, adaptive_threshold))
        
        logger.bind(tag=TAG).info(
            f"动态阈值调整：平均相似度={avg_similarity:.3f}, "
            f"标准差={std_similarity:.3f}, "
            f"中位数={median_similarity:.3f}, "
            f"调整后阈值={adaptive_threshold:.3f} "
            f"(原阈值={self.similarity_threshold:.3f})"
        )
        
        return adaptive_threshold
    
    # ===== 阶段4：余弦距离+聚类（考虑源ID） =====
    def _cluster_speakers_with_source_id(
        self, 
        xvectors: np.ndarray, 
        all_segments: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        改进的聚类：考虑源ID，来自不同源的段不应该被聚类到一起
        
        Args:
            xvectors: x-vector矩阵 [N, 512]
            all_segments: 所有语音段，包含source_id信息
        
        Returns:
            聚类标签
        """
        if len(xvectors) == 0:
            return np.array([])
        
        if len(xvectors) == 1:
            return np.array([0])
        
        # 提取源ID
        source_ids = [seg.get("source_id", 0) for seg in all_segments]
        unique_sources = len(set(source_ids))
        
        # 如果只有1个源，使用普通聚类
        if unique_sources == 1:
            return self._cluster_speakers(xvectors)
        
        # 如果有多个源，先按源ID分组，再在组内聚类
        logger.bind(tag=TAG).info(
            f"检测到 {unique_sources} 个分离源，"
            f"将按源ID分组后再聚类"
        )
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # 计算余弦距离矩阵
            similarity_matrix = np.dot(xvectors, xvectors.T)  # [N, N]
            distance_matrix = 1 - similarity_matrix
            
            # 确保对称性
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            
            # 关键改进：来自不同源的段，距离设为很大值（强制不合并）
            for i in range(len(source_ids)):
                for j in range(i + 1, len(source_ids)):
                    if source_ids[i] != source_ids[j]:
                        # 不同源的段，距离设为很大值（10.0），强制不合并
                        distance_matrix[i, j] = 10.0
                        distance_matrix[j, i] = 10.0
            
            # 动态阈值调整
            if self.use_adaptive_threshold:
                adaptive_similarity_threshold = self._compute_adaptive_threshold(similarity_matrix)
                distance_threshold = 1 - adaptive_similarity_threshold
            else:
                distance_threshold = 1 - self.similarity_threshold
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='average',
                metric='precomputed'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # 确保标签从0开始连续
            unique_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])
            
            n_clusters = len(unique_labels)
            logger.bind(tag=TAG).info(
                f"阶段4完成：识别出 {n_clusters} 个说话人 "
                f"(距离阈值: {distance_threshold:.3f}, "
                f"考虑了 {unique_sources} 个分离源)"
            )
            
            return labels
        except Exception as e:
            logger.bind(tag=TAG).error(f"聚类失败: {e}")
            # 降级：直接使用源ID
            return np.array(source_ids)
    
    # ===== 流式VAD（滑动窗口+状态机） =====
    def _streaming_vad(self, audio_chunk: np.ndarray, chunk_duration_ms: float) -> bool:
        """
        流式VAD：使用滑动窗口+状态机，避免突发噪声切成段
        
        Args:
            audio_chunk: 音频块
            chunk_duration_ms: 块持续时间（毫秒）
        
        Returns:
            True表示有语音，False表示无语音
        """
        if not self.use_webrtc_vad:
            # 如果未启用VAD，默认返回True
            return True
        
        self._ensure_vad_loaded()
        if not self._vad_loaded or self.vad is None:
            return True
        
        chunk_duration_s = chunk_duration_ms / 1000.0
        
        # 添加到滑动窗口
        self.streaming_vad_window_buffer.append(audio_chunk)
        window_size_samples = int(self.streaming_vad_window_size_ms * self.sample_rate / 1000)
        if len(self.streaming_vad_window_buffer) * len(audio_chunk) > window_size_samples:
            # 保持窗口大小
            total_samples = sum(len(chunk) for chunk in self.streaming_vad_window_buffer)
            while total_samples > window_size_samples and len(self.streaming_vad_window_buffer) > 1:
                self.streaming_vad_window_buffer.pop(0)
                total_samples = sum(len(chunk) for chunk in self.streaming_vad_window_buffer)
        
        # 对窗口进行VAD检测
        window_audio = np.concatenate(self.streaming_vad_window_buffer) if self.streaming_vad_window_buffer else audio_chunk
        
        # 转换为int16 bytes
        window_audio_clipped = np.clip(window_audio, -0.999, 0.999)
        window_int16 = (window_audio_clipped * 32767.0).astype(np.int16)
        window_bytes = window_int16.tobytes()
        
        # VAD检测
        try:
            is_speech = self.vad.is_speech(window_bytes, self.sample_rate)
        except:
            is_speech = False
        
        # 状态机更新（ON → OFF 最少持续200ms）
        if is_speech:
            if self.streaming_vad_state == "OFF":
                # OFF → ON：需要持续min_on_duration
                self.streaming_vad_on_duration += chunk_duration_s
                if self.streaming_vad_on_duration >= self.streaming_vad_min_on_duration:
                    self.streaming_vad_state = "ON"
                    self.streaming_vad_on_duration = 0.0
                    logger.bind(tag=TAG).debug(f"VAD状态：OFF → ON（持续{chunk_duration_s*1000:.0f}ms）")
            else:
                # 已经是ON，重置计数器
                self.streaming_vad_on_duration = 0.0
        else:
            if self.streaming_vad_state == "ON":
                # ON → OFF：需要持续min_off_duration
                self.streaming_vad_off_duration += chunk_duration_s
                if self.streaming_vad_off_duration >= self.streaming_vad_min_off_duration:
                    self.streaming_vad_state = "OFF"
                    self.streaming_vad_off_duration = 0.0
                    logger.bind(tag=TAG).debug(f"VAD状态：ON → OFF（持续{chunk_duration_s*1000:.0f}ms）")
            else:
                # 已经是OFF，重置计数器
                self.streaming_vad_off_duration = 0.0
        
        return self.streaming_vad_state == "ON"
    
    # ===== 因果缓存对齐（ring-buffer） =====
    def _init_causal_buffer(self):
        """初始化因果缓存ring-buffer"""
        if self.causal_ring_buffer is None:
            self.causal_ring_buffer = np.zeros(self.causal_buffer_size_samples, dtype=np.float32)
            self.causal_buffer_index = 0
            logger.bind(tag=TAG).debug(f"初始化因果缓存ring-buffer（大小: {self.causal_buffer_size_samples} samples = {self.causal_buffer_size_ms}ms）")
    
    def _add_to_causal_buffer(self, frame: np.ndarray) -> np.ndarray:
        """
        添加帧到因果缓存，返回带上下文的输入（2s因果感受野）
        
        Args:
            frame: 新帧
        
        Returns:
            带上下文的输入（2s因果感受野）
        """
        self._init_causal_buffer()
        
        frame_len = len(frame)
        
        # 添加到ring-buffer
        end_idx = (self.causal_buffer_index + frame_len) % self.causal_buffer_size_samples
        
        if end_idx > self.causal_buffer_index:
            # 没有wrap around
            self.causal_ring_buffer[self.causal_buffer_index:end_idx] = frame
        else:
            # wrap around
            part1_len = self.causal_buffer_size_samples - self.causal_buffer_index
            self.causal_ring_buffer[self.causal_buffer_index:] = frame[:part1_len]
            self.causal_ring_buffer[:end_idx] = frame[part1_len:]
        
        # 更新索引（使用hop步长，确保与模型stride对齐）
        self.causal_buffer_index = (self.causal_buffer_index + self.causal_hop_samples) % self.causal_buffer_size_samples
        
        # 返回带上下文的输入（从buffer_index往前取2s）
        start_idx = (self.causal_buffer_index - self.causal_buffer_size_samples) % self.causal_buffer_size_samples
        
        if start_idx < self.causal_buffer_index:
            context = self.causal_ring_buffer[start_idx:self.causal_buffer_index].copy()
        else:
            context = np.concatenate([
                self.causal_ring_buffer[start_idx:],
                self.causal_ring_buffer[:self.causal_buffer_index]
            ])
        
        return context
    
    # ===== 实时AGC（自动增益控制） =====
    def _streaming_agc(self, frame: np.ndarray) -> np.ndarray:
        """
        实时自动增益控制（AGC）
        
        每帧做RMS自动增益控制，上限12dB，防止远场小声道被放大成噪声
        
        Args:
            frame: 音频帧
        
        Returns:
            增益调整后的音频帧
        """
        if not self.agc_enabled:
            return frame
        
        # 计算当前RMS
        rms = np.sqrt(np.mean(frame ** 2))
        
        if rms < 1e-6:
            # 静音，不处理
            return frame
        
        # 计算增益
        gain = self.agc_target_rms / rms
        
        # 限制增益范围（上限12dB，下限-20dB）
        min_gain = 0.1  # -20dB
        gain = np.clip(gain, min_gain, self.agc_max_gain)
        
        # 应用增益
        frame_gained = frame * gain
        
        # 防止溢出
        frame_gained = np.clip(frame_gained, -0.999, 0.999)
        
        return frame_gained
    
    # ===== SNR估计 =====
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        估计信噪比（SNR）
        
        Args:
            audio: 音频数据
        
        Returns:
            SNR in dB
        """
        if not self.snr_estimation_enabled:
            return 20.0  # 默认高SNR
        
        # 简化的SNR估计：使用能量比
        signal_energy = np.mean(audio ** 2)
        
        # 估计噪声能量（使用低能量段）
        low_energy_threshold = np.percentile(np.abs(audio), 10)
        noise_mask = np.abs(audio) < low_energy_threshold
        if np.sum(noise_mask) > 0:
            noise_energy = np.mean(audio[noise_mask] ** 2)
        else:
            noise_energy = signal_energy * 0.01  # 假设1%是噪声
        
        if noise_energy < 1e-10:
            return 30.0  # 默认高SNR
        
        snr_linear = signal_energy / noise_energy
        snr_db = 10 * np.log10(snr_linear)
        
        # 限制范围
        snr_db = np.clip(snr_db, 0.0, 40.0)
        
        return snr_db
    
    # ===== 动态阈值调整（根据SNR） =====
    def _adaptive_threshold_adjustment(self, snr_db: float):
        """
        根据SNR自适应调整相似度阈值
        
        Args:
            snr_db: 信噪比（dB）
        """
        self.current_snr_db = snr_db
        
        if snr_db < 5.0:
            # 低SNR：降低相似度阈值，减少漏拆
            new_threshold = 0.55
            if abs(self.streaming_similarity_threshold - new_threshold) > 0.01:
                logger.bind(tag=TAG).info(f"低SNR({snr_db:.1f}dB)，降低相似度阈值到{new_threshold}")
                self.streaming_similarity_threshold = new_threshold
        elif snr_db < 10.0:
            # 中等SNR：使用中等阈值
            new_threshold = 0.60
            if abs(self.streaming_similarity_threshold - new_threshold) > 0.01:
                self.streaming_similarity_threshold = new_threshold
        else:
            # 高SNR：使用正常阈值
            new_threshold = 0.65
            if abs(self.streaming_similarity_threshold - new_threshold) > 0.01:
                self.streaming_similarity_threshold = new_threshold
    
    # ===== 流式聚类：Single-Pass算法 =====
    def _streaming_cluster_single_pass(self, xvector: np.ndarray) -> int:
        """
        Single-Pass流式聚类算法
        
        每段x-vector到来即与现有簇中心算余弦相似度；
        > 阈值则并入并在线更新中心（EMA），否则开新簇。
        
        Args:
            xvector: 新来的x-vector（512维，已L2归一化）
        
        Returns:
            簇ID（0, 1, 2, ...）
        """
        if len(xvector.shape) > 1:
            xvector = xvector.flatten()
        
        # L2归一化（确保）
        norm = np.linalg.norm(xvector)
        if norm > 0:
            xvector = xvector / norm
        
        # 如果没有现有簇，创建第一个簇
        if len(self.streaming_cluster_centers) == 0:
            self.streaming_cluster_centers.append(xvector.copy())
            self.streaming_cluster_weights.append(1.0)
            self.streaming_cluster_count = 1
            logger.bind(tag=TAG).debug(f"创建第一个簇（簇ID: 0）")
            return 0
        
        # 计算与所有现有簇中心的余弦相似度
        similarities = []
        for center in self.streaming_cluster_centers:
            # 余弦相似度 = 点积（因为都已归一化）
            similarity = np.dot(xvector, center)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        max_similarity = np.max(similarities)
        best_cluster_id = np.argmax(similarities)
        
        # 如果最大相似度 > 阈值，并入该簇
        if max_similarity >= self.streaming_similarity_threshold:
            # EMA更新簇中心（抗顺序抖动）
            # center = α * new_vec + (1-α) * center
            alpha = self.streaming_ema_alpha
            old_center = self.streaming_cluster_centers[best_cluster_id]
            new_center = alpha * xvector + (1 - alpha) * old_center
            
            # 重新归一化
            norm = np.linalg.norm(new_center)
            if norm > 0:
                new_center = new_center / norm
            
            self.streaming_cluster_centers[best_cluster_id] = new_center
            self.streaming_cluster_weights[best_cluster_id] += 1.0
            
            logger.bind(tag=TAG).debug(
                f"并入簇 {best_cluster_id}（相似度: {max_similarity:.4f} >= {self.streaming_similarity_threshold:.4f}）"
            )
            return best_cluster_id
        
        # 否则，创建新簇
        # 如果簇数已达到上限，合并最近两簇
        if len(self.streaming_cluster_centers) >= self.streaming_max_clusters:
            # 找到权重最小的两个簇并合并
            weights = np.array(self.streaming_cluster_weights)
            min_indices = np.argsort(weights)[:2]
            
            # 合并两个簇（取加权平均）
            merged_center = (
                self.streaming_cluster_centers[min_indices[0]] * self.streaming_cluster_weights[min_indices[0]] +
                self.streaming_cluster_centers[min_indices[1]] * self.streaming_cluster_weights[min_indices[1]]
            ) / (self.streaming_cluster_weights[min_indices[0]] + self.streaming_cluster_weights[min_indices[1]])
            
            # 归一化
            norm = np.linalg.norm(merged_center)
            if norm > 0:
                merged_center = merged_center / norm
            
            # 替换第一个簇，删除第二个簇
            self.streaming_cluster_centers[min_indices[0]] = merged_center
            self.streaming_cluster_weights[min_indices[0]] = (
                self.streaming_cluster_weights[min_indices[0]] + self.streaming_cluster_weights[min_indices[1]]
            )
            
            # 删除第二个簇
            self.streaming_cluster_centers.pop(min_indices[1])
            self.streaming_cluster_weights.pop(min_indices[1])
            
            logger.bind(tag=TAG).info(
                f"簇数达到上限({self.streaming_max_clusters})，合并簇 {min_indices[0]} 和 {min_indices[1]}"
            )
        
        # 创建新簇
        new_cluster_id = len(self.streaming_cluster_centers)
        self.streaming_cluster_centers.append(xvector.copy())
        self.streaming_cluster_weights.append(1.0)
        self.streaming_cluster_count = len(self.streaming_cluster_centers)
        
        logger.bind(tag=TAG).debug(
            f"创建新簇 {new_cluster_id}（相似度: {max_similarity:.4f} < {self.streaming_similarity_threshold:.4f}）"
        )
        return new_cluster_id
    
    # ===== 阶段4：余弦距离+聚类 =====
    def _cluster_speakers(self, xvectors: np.ndarray) -> np.ndarray:
        """阶段4：使用余弦距离和AgglomerativeClustering聚类（支持动态阈值）"""
        if len(xvectors) == 0:
            return np.array([])
        
        if len(xvectors) == 1:
            return np.array([0])
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # 计算余弦距离矩阵
            similarity_matrix = np.dot(xvectors, xvectors.T)  # [N, N]
            distance_matrix = 1 - similarity_matrix
            
            # 确保对称性
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            
            # ===== 动态阈值调整 =====
            if self.use_adaptive_threshold:
                adaptive_similarity_threshold = self._compute_adaptive_threshold(similarity_matrix)
                distance_threshold = 1 - adaptive_similarity_threshold
            else:
                # 使用固定阈值
                distance_threshold = 1 - self.similarity_threshold
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='average',
                metric='precomputed'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # 确保标签从0开始连续
            unique_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])
            
            n_clusters = len(unique_labels)
            logger.bind(tag=TAG).info(
                f"阶段4完成：识别出 {n_clusters} 个说话人 "
                f"(距离阈值: {distance_threshold:.3f})"
            )
            
            return labels
        except ImportError:
            logger.bind(tag=TAG).error("sklearn未安装，无法进行聚类")
            logger.bind(tag=TAG).error("请运行: pip install scikit-learn")
            # 返回每个段一个标签（每人一段）
            return np.arange(len(xvectors))
        except Exception as e:
            logger.bind(tag=TAG).error(f"聚类失败: {e}")
            # 返回每个段一个标签
            return np.arange(len(xvectors))
    
    # ===== 阶段5：时间戳写回 =====
    def _write_labels_to_timestamps(
        self, 
        all_segments: List[Dict[str, Any]], 
        labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        阶段5：把聚类ID写回时间戳（使用带边界平滑的版本）
        
        注意：此方法已被 _write_labels_to_timestamps_with_smoothing 替代
        保留此方法以保持向后兼容
        """
        # 调用带边界平滑的版本
        return self._write_labels_to_timestamps_with_smoothing(all_segments, labels)
    
    # ===== 流式分离：处理音频块 =====
    async def separate_chunk(
        self,
        audio_chunk: bytes,
        sample_rate: int = 16000,
        is_first_chunk: bool = False,
        is_last_chunk: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        流式分离：处理单个音频块
        
        Args:
            audio_chunk: 音频块数据（PCM bytes）
            sample_rate: 采样率，默认16000Hz
            is_first_chunk: 是否是第一个块
            is_last_chunk: 是否是最后一个块
        
        Returns:
            {speaker_id: audio_chunk} 分离后的音频块字典
        """
        if not audio_chunk or len(audio_chunk) == 0:
            logger.bind(tag=TAG).warning("音频块为空，返回空字典")
            return {}
        
        self.sample_rate = sample_rate
        
        try:
            # 流式分离
            separated_chunks = await self._separate_sources_streaming(
                audio_chunk, 
                is_first_chunk=is_first_chunk,
                is_last_chunk=is_last_chunk
            )
            
            logger.bind(tag=TAG).debug(
                f"流式分离完成：块大小={len(audio_chunk)}字节, "
                f"分离出{len(separated_chunks)}个说话人"
            )
            
            return separated_chunks
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"流式分离失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
            return {}
    
    # ===== 完整Pipeline =====
    async def separate(
        self, 
        audio_data: bytes,
        sample_rate: int = 16000,
        num_speakers: int = None
    ) -> List[Dict[str, Any]]:
        """
        完整Pipeline：5阶段盲源分离（支持流式和批量模式）
        
        Args:
            audio_data: 混合音频数据（PCM bytes）
            sample_rate: 采样率，默认16000Hz
            num_speakers: 说话人数量（可选，如果None则自动检测）
        
        Returns:
            分离后的音频列表
        """
        if not audio_data or len(audio_data) == 0:
            logger.bind(tag=TAG).warning("音频数据为空，返回空列表")
            return []
        
        self.sample_rate = sample_rate
        
        # ===== 流式模式（暂时禁用，使用批量模式，回到原来的分离方法） =====
        # 强制使用批量模式，回到原来的分离方法
        # if self.streaming_enabled:
        #     return await self._separate_streaming_mode(audio_data, sample_rate)
        
        # ===== 批量模式（原有逻辑） =====
        try:
            # ===== 阶段1：Conv-TasNet源分离 =====
            logger.bind(tag=TAG).info("开始阶段1：Conv-TasNet源分离（批量模式）...")
            sources = self._separate_sources(audio_data)
            
            if len(sources) == 0:
                logger.bind(tag=TAG).warning("未分离出任何音频源")
                return []
            
            # ===== 阶段2：VAD切分 =====
            logger.bind(tag=TAG).info("开始阶段2：WebRTC VAD切分...")
            all_segments = []
            for source_id, source in enumerate(sources):
                segments = self._vad_segmentation(source)
                for seg in segments:
                    seg["source_id"] = source_id
                all_segments.extend(segments)
            
            if len(all_segments) == 0:
                logger.bind(tag=TAG).warning("未检测到任何语音段")
                return []
            
            logger.bind(tag=TAG).info(f"检测到 {len(all_segments)} 个语音段")
            
            # ===== 检查：如果分离出的源数量与语音段数量匹配，直接使用源ID =====
            # 如果每个源都有语音段，且源数量>=2，直接使用源ID作为说话人ID
            source_segment_count = {}
            for seg in all_segments:
                source_id = seg.get("source_id", 0)
                source_segment_count[source_id] = source_segment_count.get(source_id, 0) + 1
            
            # 如果每个源都有语音段，且源数量>=2，直接使用源ID
            if len(sources) >= 2 and len(source_segment_count) == len(sources):
                logger.bind(tag=TAG).info(
                    f"检测到 {len(sources)} 个分离源，每个源都有语音段，"
                    f"直接使用源ID作为说话人ID（跳过聚类）"
                )
                # 直接使用源ID作为说话人ID
                result = []
                for segment in all_segments:
                    source_id = segment.get("source_id", 0)
                    speaker_id = f"SPEAKER_{source_id:02d}"
                    audio_segment = segment["audio"]
                    if isinstance(audio_segment, np.ndarray):
                        # 清理NaN和Inf
                        if np.any(np.isnan(audio_segment)) or np.any(np.isinf(audio_segment)):
                            audio_segment = np.nan_to_num(audio_segment, nan=0.0, posinf=0.999, neginf=-0.999)
                        
                        # 关键：clip到[-0.999, 0.999]并使用32767避免整型饱和
                        audio_segment = np.clip(audio_segment, -0.999, 0.999)
                        
                        audio_int16 = (audio_segment * 32767.0).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                    else:
                        audio_bytes = audio_segment
                    
                    result.append({
                        'speaker_id': speaker_id,
                        'audio': audio_bytes,
                        'start': segment['start'],
                        'end': segment['end']
                    })
                
                # 按时间排序
                result.sort(key=lambda x: x['start'])
                logger.bind(tag=TAG).info(f"直接使用源ID：生成 {len(result)} 个说话人片段")
                return result
            
            # ===== 阶段3：提取x-vector =====
            logger.bind(tag=TAG).info("开始阶段3：提取x-vector...")
            xvectors = []
            for segment in all_segments:
                xvector = self._extract_xvector(segment["audio"])
                xvectors.append(xvector)
            
            xvectors = np.array(xvectors)  # [N, 512]
            
            # ===== 阶段4：聚类（改进：考虑源ID） =====
            logger.bind(tag=TAG).info("开始阶段4：说话人聚类（考虑源ID）...")
            labels = self._cluster_speakers_with_source_id(xvectors, all_segments)
            
            # ===== 阶段5：时间戳写回（带边界平滑） =====
            logger.bind(tag=TAG).info("开始阶段5：时间戳写回（带边界平滑）...")
            result = self._write_labels_to_timestamps_with_smoothing(all_segments, labels)
            
            logger.bind(tag=TAG).info(f"完整Pipeline完成：生成 {len(result)} 个说话人片段")
            
            return result
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"语音分离Pipeline失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
            return []
    
    # ===== 流式模式处理 =====
    async def _separate_streaming_mode(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        流式模式：将完整音频分块处理
        
        Args:
            audio_data: 完整音频数据（PCM bytes）
            sample_rate: 采样率
        
        Returns:
            分离后的音频列表
        """
        logger.bind(tag=TAG).info("使用流式模式处理音频（收集x-vector后批量聚类）")
        
        # 计算块大小
        chunk_size_samples = int(self.chunk_size_ms * sample_rate / 1000)
        chunk_size_bytes = chunk_size_samples * 2  # 16bit = 2字节/采样
        
        # 将音频转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # 收集所有块的x-vector和音频段（用于后续批量聚类）
        all_segments = []  # 收集所有音频段（包含source_id和x-vector）
        
        # 分块处理
        total_samples = len(audio_np)
        offset = 0
        chunk_index = 0
        
        while offset < total_samples:
            # 计算当前块的范围
            chunk_end = min(offset + chunk_size_samples, total_samples)
            chunk_samples = audio_np[offset:chunk_end]
            
            # 转换为bytes
            chunk_bytes = chunk_samples.astype(np.int16).tobytes()
            
            # 判断是否是第一个/最后一个块
            is_first = (chunk_index == 0)
            is_last = (chunk_end >= total_samples)
            
            # 流式分离
            separated_chunks = await self._separate_sources_streaming(
                chunk_bytes,
                is_first_chunk=is_first,
                is_last_chunk=is_last
            )
            
            # 对每个分离出的源提取x-vector并收集
            for source_id, audio_chunk in separated_chunks.items():
                # 提取x-vector（用于聚类）
                try:
                    xvector = self._extract_xvector(audio_chunk)
                    
                    # 收集音频段和x-vector（用于后续批量聚类）
                    all_segments.append({
                        'audio': audio_chunk,
                        'start_time': offset / sample_rate,
                        'end_time': chunk_end / sample_rate,
                        'source_id': source_id,  # 保存源ID
                        'xvector': xvector  # 保存x-vector用于聚类
                    })
                    
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"提取x-vector失败: {e}，跳过该段")
                    # 失败时也收集音频段（使用源ID）
                    all_segments.append({
                        'audio': audio_chunk,
                        'start_time': offset / sample_rate,
                        'end_time': chunk_end / sample_rate,
                        'source_id': source_id
                    })
            
            # 关键：使用hop步长而不是chunk_size步长，确保正确重叠
            # hop = chunk_size - overlap，避免相位错位
            overlap_samples = int(self.overlap_size_ms * sample_rate / 1000)
            hop_samples = chunk_size_samples - overlap_samples
            offset += hop_samples  # 使用hop步长，确保块之间有正确重叠
            chunk_index += 1
        
        if len(all_segments) == 0:
            logger.bind(tag=TAG).warning("未收集到任何音频段")
            return []
        
        # 检查：如果分离出的源数量与语音段数量匹配，直接使用源ID
        unique_sources = set(seg.get('source_id') for seg in all_segments if 'source_id' in seg)
        if len(unique_sources) >= 2:
            logger.bind(tag=TAG).info(
                f"检测到 {len(unique_sources)} 个分离源，直接使用源ID作为说话人ID（跳过聚类）"
            )
            # 直接使用源ID作为说话人ID
            result = []
            for segment in all_segments:
                source_id = segment.get('source_id', 'SOURCE_00')
                speaker_id = f"SPEAKER_{source_id}" if isinstance(source_id, str) else f"SPEAKER_{source_id:02d}"
                
                audio_chunk = segment['audio']
                if isinstance(audio_chunk, np.ndarray):
                    audio_chunk = np.clip(audio_chunk, -0.999, 0.999)
                    audio_int16 = (audio_chunk * 32767.0).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                else:
                    audio_bytes = audio_chunk
                
                result.append({
                    'speaker_id': speaker_id,
                    'audio': audio_bytes,
                    'start': segment['start_time'],
                    'end': segment['end_time']
                })
            
            logger.bind(tag=TAG).info(f"直接使用源ID：生成 {len(result)} 个说话人片段")
            return result
        
        # 批量聚类：收集所有x-vector，使用批量聚类（改回之前的聚类方式）
        logger.bind(tag=TAG).info(f"收集到 {len(all_segments)} 个音频段，开始批量聚类...")
        
        # 提取所有x-vector
        xvectors = []
        valid_segments = []
        for segment in all_segments:
            if 'xvector' in segment:
                xvectors.append(segment['xvector'])
                valid_segments.append(segment)
        
        if len(xvectors) < 2:
            logger.bind(tag=TAG).warning("x-vector数量不足，无法聚类")
            # 使用源ID
            result = []
            for segment in all_segments:
                source_id = segment.get('source_id', 'SOURCE_00')
                speaker_id = f"SPEAKER_{source_id}" if isinstance(source_id, str) else f"SPEAKER_{source_id:02d}"
                
                audio_chunk = segment['audio']
                if isinstance(audio_chunk, np.ndarray):
                    audio_chunk = np.clip(audio_chunk, -0.999, 0.999)
                    audio_int16 = (audio_chunk * 32767.0).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                else:
                    audio_bytes = audio_chunk
                
                result.append({
                    'speaker_id': speaker_id,
                    'audio': audio_bytes,
                    'start': segment['start_time'],
                    'end': segment['end_time']
                })
            return result
        
        xvectors = np.array(xvectors)  # [N, 512]
        
        # 使用批量聚类（改回之前的聚类方式）
        logger.bind(tag=TAG).info("开始批量聚类（AgglomerativeClustering）...")
        
        # 构建相似度矩阵（考虑源ID）
        similarity_matrix = np.dot(xvectors, xvectors.T)  # [N, N]
        distance_matrix = 1 - similarity_matrix
        
        # 确保对称性
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # 关键改进：来自不同源的段，距离设为很大值（强制不合并）
        for i in range(len(valid_segments)):
            for j in range(i + 1, len(valid_segments)):
                source_i = valid_segments[i].get('source_id')
                source_j = valid_segments[j].get('source_id')
                if source_i != source_j:
                    # 不同源的段，距离设为很大值（10.0），强制不合并
                    distance_matrix[i, j] = 10.0
                    distance_matrix[j, i] = 10.0
        
        # 动态阈值调整
        if self.use_adaptive_threshold:
            adaptive_similarity_threshold = self._compute_adaptive_threshold(similarity_matrix)
            distance_threshold = 1 - adaptive_similarity_threshold
        else:
            distance_threshold = 1 - self.similarity_threshold
        
        # 使用AgglomerativeClustering进行聚类
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='average',
                metric='precomputed'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # 确保标签从0开始连续
            unique_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])
            
            n_clusters = len(unique_labels)
            logger.bind(tag=TAG).info(f"批量聚类完成：识别出 {n_clusters} 个说话人簇")
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"批量聚类失败: {e}，使用源ID")
            # 失败时使用源ID
            labels = np.arange(len(valid_segments))
        
        # 合并每个说话人的所有块（带边界平滑）
        result = []
        overlap_samples = int(self.overlap_size_ms * sample_rate / 1000)
        
        # 按说话人ID分组
        speaker_buffers = {}  # {speaker_id: list of segments}
        for i, segment in enumerate(valid_segments):
            speaker_id = f"SPEAKER_{labels[i]:02d}"
            if speaker_id not in speaker_buffers:
                speaker_buffers[speaker_id] = []
            speaker_buffers[speaker_id].append(segment)
        
        for speaker_id, segments in speaker_buffers.items():
            if len(segments) == 0:
                continue
            
            # 按时间排序
            segments.sort(key=lambda x: x['start_time'])
            
            # 合并所有块（使用OLA，确保相位对齐）
            merged_audio = segments[0]['audio']
            start_time = segments[0]['start_time']
            
            for i in range(1, len(segments)):
                # 关键：边界平滑（使用归一化的汉宁窗，满足平方和恒定）
                # 确保相位对齐，避免"颤音"调制
                merged_audio = self._smooth_boundary(
                    merged_audio,
                    segments[i]['audio'],
                    overlap_samples
                )
                
                # 关键：每次合并后clip，防止幅度累积导致溢出
                merged_audio = np.clip(merged_audio, -0.999, 0.999)
            
            end_time = segments[-1]['end_time']
            
            # 关键：最终输出前clip，防止幅度溢出
            # 不进行额外的归一化，保持原始特性
            merged_audio = np.clip(merged_audio, -0.999, 0.999)
            
            # 转换为PCM bytes（清理NaN和Inf）
            if np.any(np.isnan(merged_audio)) or np.any(np.isinf(merged_audio)):
                logger.bind(tag=TAG).warning(f"检测到NaN或Inf值，进行清理")
                merged_audio = np.nan_to_num(merged_audio, nan=0.0, posinf=0.999, neginf=-0.999)
            # 关键：clip到[-0.999, 0.999]并使用32767避免整型饱和
            merged_audio = np.clip(merged_audio, -0.999, 0.999)
            audio_int16 = (merged_audio * 32767.0).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            result.append({
                'speaker_id': speaker_id,
                'audio': audio_bytes,
                'start': start_time,
                'end': end_time
            })
        
        logger.bind(tag=TAG).info(f"流式分离完成：生成 {len(result)} 个说话人音频")
        return result
    
    # ===== 带边界平滑的时间戳写回 =====
    def _write_labels_to_timestamps_with_smoothing(
        self,
        all_segments: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        阶段5：把聚类ID写回时间戳（带边界平滑）
        
        Args:
            all_segments: 所有语音段
            labels: 聚类标签
        
        Returns:
            分离后的音频列表（边界已平滑）
        """
        if len(all_segments) != len(labels):
            logger.bind(tag=TAG).warning(f"段数({len(all_segments)})与标签数({len(labels)})不匹配")
            labels = np.arange(len(all_segments))
        
        # 按说话人分组
        speaker_segments = {}  # {speaker_id: list of segments}
        
        for i, segment in enumerate(all_segments):
            speaker_id = f"SPEAKER_{labels[i]:02d}"
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)
        
        # 对每个说话人的段进行边界平滑
        result = []
        overlap_samples = int(self.overlap_size_ms * self.sample_rate / 1000)
        
        for speaker_id, segments in speaker_segments.items():
            if len(segments) == 0:
                continue
            
            # 按时间排序
            segments.sort(key=lambda x: x['start'])
            
            # 合并同一说话人的所有段（无论是否相邻，都合并成一个完整音频）
            # 这样同一个音源的所有片段会合并在一起，然后给ASR识别
            merged_audio = segments[0]['audio'].copy()
            start_time = segments[0]['start']
            end_time = segments[0]['end']
            
            for i in range(1, len(segments)):
                next_segment = segments[i]
                next_audio = next_segment['audio']
                next_start = next_segment['start']
                next_end = next_segment['end']
                
                # 计算需要填充的静音长度（如果段之间有间隔）
                gap_samples = int((next_start - end_time) * self.sample_rate)
                if gap_samples > 0:
                    # 有间隔，填充静音
                    silence = np.zeros(gap_samples, dtype=merged_audio.dtype)
                    merged_audio = np.concatenate([merged_audio, silence])
                
                # 合并下一段（使用边界平滑）
                merged_audio = self._smooth_boundary(
                    merged_audio,
                    next_audio,
                    overlap_samples
                )
                
                # 关键：每次合并后clip，防止幅度累积导致溢出
                merged_audio = np.clip(merged_audio, -0.999, 0.999)
                
                end_time = next_end
            
            # 转换为最终格式（每个说话人只有一个完整的音频）
            merged_segments = [{
                'audio': merged_audio,
                'start': start_time,
                'end': end_time
            }]
            
            # 转换为最终格式
            for seg in merged_segments:
                audio_segment = seg['audio']
                if isinstance(audio_segment, np.ndarray):
                    # 清理NaN和Inf
                    if np.any(np.isnan(audio_segment)) or np.any(np.isinf(audio_segment)):
                        logger.bind(tag=TAG).warning(f"检测到NaN或Inf值，进行清理")
                        audio_segment = np.nan_to_num(audio_segment, nan=0.0, posinf=0.999, neginf=-0.999)
                    # 关键：clip到[-0.999, 0.999]并使用32767避免整型饱和
                    audio_segment = np.clip(audio_segment, -0.999, 0.999)
                    audio_int16 = (audio_segment * 32767.0).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                else:
                    audio_bytes = audio_segment
                
                result.append({
                    'speaker_id': speaker_id,
                    'audio': audio_bytes,
                    'start': seg['start'],
                    'end': seg['end']
                })
        
        # 按时间排序
        result.sort(key=lambda x: x['start'])
        
        logger.bind(tag=TAG).info(f"阶段5完成：生成 {len(result)} 个标注段（边界已平滑）")
        return result

