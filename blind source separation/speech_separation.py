import os
import sys
from typing import Optional, Dict, Any
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


def create_instance(provider_type: str, config: Dict[str, Any]) -> Optional[Any]:
    """
    创建语音分离提供者实例
    
    Args:
        provider_type: 提供者类型（如 "pyannote"）
        config: 配置字典
    
    Returns:
        语音分离提供者实例，如果创建失败返回None
    """
    try:
        # 根据provider_type动态导入对应的模块
        if provider_type == "pyannote":
            logger.bind(tag=TAG).error("pyannote已移除，请使用conv_tasnet或其他方案")
            return None
        elif provider_type == "mossformer2":
            # 未来可以支持其他实现
            logger.bind(tag=TAG).error(f"暂不支持 {provider_type} 提供者")
            return None
        elif provider_type == "conv_tasnet":
            module_name = "core.providers.speech_separation.conv_tasnet_separation"
        else:
            logger.bind(tag=TAG).error(f"未知的语音分离提供者类型: {provider_type}")
            return None
        
        # 检查模块文件是否存在
        module_path = module_name.replace(".", "/") + ".py"
        if not os.path.exists(module_path):
            logger.bind(tag=TAG).error(f"模块文件不存在: {module_path}")
            return None
        
        # 动态导入模块
        if module_name not in sys.modules:
            __import__(module_name)
        
        module = sys.modules[module_name]
        
        # 获取Provider类并实例化
        if hasattr(module, "SpeechSeparationProvider"):
            provider = module.SpeechSeparationProvider(config)
            logger.bind(tag=TAG).info(f"语音分离提供者创建成功: {provider_type}")
            return provider
        else:
            logger.bind(tag=TAG).error(f"模块 {module_name} 中没有找到 SpeechSeparationProvider 类")
            return None
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"创建语音分离提供者失败 ({provider_type}): {e}")
        import traceback
        logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
        return None


def initialize_speech_separation(config: Dict[str, Any]) -> Optional[Any]:
    """
    初始化语音分离模块
    
    Args:
        config: 配置字典，包含 speech_separation 配置
    
    Returns:
        语音分离提供者实例，如果未启用或初始化失败返回None
    """
    speech_sep_config = config.get("speech_separation")
    
    # 检查是否启用
    if not speech_sep_config or not speech_sep_config.get("enabled", False):
        logger.bind(tag=TAG).info("语音分离功能未启用")
        return None
    
    # 获取提供者类型
    provider_type = speech_sep_config.get("provider", "conv_tasnet")
    
    # 获取提供者配置
    provider_config = speech_sep_config.get(provider_type, {})
    
    if not provider_config:
        logger.bind(tag=TAG).error(f"语音分离提供者 {provider_type} 的配置不存在")
        return None
    
    # 创建提供者实例
    provider = create_instance(provider_type, provider_config)
    
    if provider:
        # 确保配置被正确传递
        if hasattr(provider, 'config'):
            logger.bind(tag=TAG).info(
                f"语音分离模块初始化成功: {provider_type}, "
                f"skip_single_speaker={provider.config.get('skip_single_speaker', '未设置')}"
            )
        else:
            logger.bind(tag=TAG).info(f"语音分离模块初始化成功: {provider_type}")
    else:
        logger.bind(tag=TAG).warning(f"语音分离模块初始化失败: {provider_type}")
    
    return provider

