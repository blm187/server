import os
import io
import wave
import uuid
import json
import time
import queue
import asyncio
import traceback
import threading
import opuslib_next
import concurrent.futures
from abc import ABC, abstractmethod
from config.logger import setup_logging
from typing import Optional, Tuple, List, Dict
from core.handle.receiveAudioHandle import startToChat
from core.handle.reportHandle import enqueue_asr_report
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import handleAudioMessage
from core.utils.emotion_recognition import get_emotion_recognizer

TAG = __name__
logger = setup_logging()


class ASRProviderBase(ABC):
    def __init__(self):
        pass

    # 打开音频通道
    async def open_audio_channels(self, conn):
        conn.asr_priority_thread = threading.Thread(
            target=self.asr_text_priority_thread, args=(conn,), daemon=True
        )
        conn.asr_priority_thread.start()

    # 有序处理ASR音频
    def asr_text_priority_thread(self, conn):
        frame_count = 0
        logger.bind(tag=TAG).info("ASR音频处理线程已启动")
        while not conn.stop_event.is_set():
            try:
                message = conn.asr_audio_queue.get(timeout=1)
                frame_count += 1
                if frame_count == 1 or frame_count % 100 == 0:
                    logger.bind(tag=TAG).info(
                        f"ASR线程处理音频数据（帧#{frame_count}，大小: {len(message)} 字节）"
                    )
                future = asyncio.run_coroutine_threadsafe(
                    handleAudioMessage(conn, message),
                    conn.loop,
                )
                # 使用超时避免无限阻塞，如果处理时间超过5秒则记录警告
                try:
                    future.result(timeout=5.0)
                except TimeoutError:
                    logger.bind(tag=TAG).warning(
                        f"处理音频帧#{frame_count}超时（超过5秒），继续处理下一帧"
                    )
                except Exception as e:
                    logger.bind(tag=TAG).error(
                        f"处理音频帧#{frame_count}时出错: {str(e)}, 类型: {type(e).__name__}"
                    )
            except queue.Empty:
                # 检查是否需要超时处理累积的音频
                self._check_timeout_processing(conn)
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理ASR文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )
                continue
    
    def _check_timeout_processing(self, conn):
        """检查是否需要超时处理累积的音频数据"""
        # 只在设备端直接发送音频数据且未收到控制消息时检查
        is_realtime_mode_without_control = (
            hasattr(conn, 'has_received_control_message') 
            and not conn.has_received_control_message
            and (conn.client_listen_mode == "auto" or conn.client_listen_mode == "realtime")
        )
        
        if not is_realtime_mode_without_control:
            return
        
        # 检查是否有累积的音频数据
        if len(conn.asr_audio) == 0:
            return
        
        # 检查最后接收音频的时间
        if not hasattr(conn, 'last_audio_receive_time'):
            return
        
        # 如果超过2秒没有收到新的音频数据，且累积的音频帧数>=15，则自动处理
        timeout_seconds = 2.0
        min_frames_for_timeout = 15
        current_time = time.time()
        time_since_last_audio = current_time - conn.last_audio_receive_time
        
        if time_since_last_audio >= timeout_seconds and len(conn.asr_audio) >= min_frames_for_timeout:
            logger.bind(tag=TAG).info(
                f"设备端停止发送音频数据超过{timeout_seconds}秒，自动处理累积的音频（{len(conn.asr_audio)} 帧）"
            )
            # 复制音频数据并清空
            asr_audio_task = conn.asr_audio.copy()
            conn.asr_audio.clear()
            # 异步触发识别
            asyncio.run_coroutine_threadsafe(
                self.handle_voice_stop(conn, asr_audio_task),
                conn.loop,
            )

    # 接收音频
    async def receive_audio(self, conn, audio, audio_have_voice):
        # 每帧都进行Opus解码并记录日志（类似图片中的功能）
        if audio and len(audio) > 0:
            try:
                from core.utils.util import config_to_sample_rate
                
                # 初始化帧计数器（每个连接独立计数）
                if not hasattr(conn, '_opus_decode_frame_count'):
                    conn._opus_decode_frame_count = 0
                conn._opus_decode_frame_count += 1
                
                # 解析TOC字节
                toc = audio[0]
                config = (toc >> 3) & 0x1F
                stereo = (toc >> 2) & 0x01
                frame_code = toc & 0x03
                opus_sample_rate = config_to_sample_rate(config)
                channels = 2 if stereo == 1 else 1
                
                # 记录Opus解码信息（每帧都记录，类似图片中的格式）
                logger.bind(tag=TAG).info(
                    f"[Opus解码] 帧 {conn._opus_decode_frame_count - 1}: TOC=0x{toc:02X}, config={config}, "
                    f"采样率={opus_sample_rate}Hz, stereo={stereo}, frame_code={frame_code}, "
                    f"长度={len(audio)}"
                )
            except Exception as e:
                logger.bind(tag=TAG).warning(f"解析Opus TOC字节失败: {e}")
        
        if conn.client_listen_mode == "auto" or conn.client_listen_mode == "realtime":
            have_voice = audio_have_voice
        else:
            # manual模式下，忽略VAD检测，使用client_have_voice状态
            have_voice = conn.client_have_voice
        
        conn.asr_audio.append(audio)
        
        # 如果设备端直接发送音频数据而不发送控制消息，需要特殊处理
        is_realtime_mode_without_control = (
            hasattr(conn, 'has_received_control_message') 
            and not conn.has_received_control_message
            and (conn.client_listen_mode == "auto" or conn.client_listen_mode == "realtime")
        )
        
        if is_realtime_mode_without_control:
            # 设备端实时发送音频数据，累积所有帧以便后续识别
            # 更新最后接收音频的时间戳
            if not hasattr(conn, 'last_audio_receive_time'):
                conn.last_audio_receive_time = time.time()
            conn.last_audio_receive_time = time.time()
            
            # 当累积到一定数量（如50帧，约3秒）时，自动触发识别
            max_frames_before_auto_trigger = 50
            if len(conn.asr_audio) >= max_frames_before_auto_trigger:
                logger.bind(tag=TAG).info(
                    f"设备端实时发送音频数据，累积音频帧达到{max_frames_before_auto_trigger}帧，自动触发识别"
                )
                # 自动触发识别
                asr_audio_task = conn.asr_audio.copy()
                conn.asr_audio.clear()
                # 异步触发识别，不阻塞音频接收流程
                asyncio.create_task(self.handle_voice_stop(conn, asr_audio_task))
                return
            else:
                logger.bind(tag=TAG).debug(
                    f"设备端实时发送音频数据，累积音频帧（当前: {len(conn.asr_audio)} 帧）"
                )
        elif (conn.client_listen_mode == "auto" or conn.client_listen_mode == "realtime") and not have_voice and not conn.client_have_voice:
            # 正常auto/realtime模式，如果没有检测到语音，只保留最后10帧
            conn.asr_audio = conn.asr_audio[-10:]
            return

        # 如果设备端直接发送音频数据而不发送控制消息，不依赖client_voice_stop
        # 而是依赖累积帧数或超时机制来触发识别
        if conn.client_voice_stop and not is_realtime_mode_without_control:
            asr_audio_task = conn.asr_audio.copy()
            conn.asr_audio.clear()
            conn.reset_vad_states()

            logger.bind(tag=TAG).info(f"收到语音停止信号，音频帧数: {len(asr_audio_task)}")
            # 在manual模式下，只要有音频帧就处理；在auto/realtime模式下，需要至少15帧
            min_frames = 1 if conn.client_listen_mode == "manual" else 15
            if len(asr_audio_task) >= min_frames:
                logger.bind(tag=TAG).info(f"开始处理语音停止，音频帧数: {len(asr_audio_task)}")
                await self.handle_voice_stop(conn, asr_audio_task)
            else:
                logger.bind(tag=TAG).warning(f"音频帧数不足，跳过处理: {len(asr_audio_task)} (需要至少{min_frames}帧)")

    # 处理语音停止
    async def handle_voice_stop(self, conn, asr_audio_task: List[bytes]):
        """并行处理ASR和声纹识别（支持语音分离）"""
        try:
            total_start_time = time.monotonic()
            
            # 准备音频数据
            if conn.audio_format == "pcm":
                pcm_data = asr_audio_task
            else:
                pcm_data = self.decode_opus(asr_audio_task)
            
            combined_pcm_data = b"".join(pcm_data)
            
            # 检查是否启用语音分离
            if conn.speech_separation_provider and combined_pcm_data:
                try:
                    # 检测是否是单人场景（简单检测：音频时长和能量）
                    audio_duration = len(combined_pcm_data) / (16000 * 2)  # 16kHz, 16bit = 2字节/采样
                    
                    # 计算音频能量
                    import numpy as np
                    audio_np = np.frombuffer(combined_pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_energy = np.mean(np.abs(audio_np))
                    
                    # 检查配置：是否跳过单人场景
                    sep_config = getattr(conn.speech_separation_provider, 'config', {})
                    skip_single = sep_config.get("skip_single_speaker", True)  # 默认跳过
                    
                    logger.bind(tag=TAG).info(
                        f"语音分离配置检查: skip_single_speaker={skip_single}, "
                        f"音频时长={len(combined_pcm_data)/(16000*2):.2f}s, 能量={np.mean(np.abs(np.frombuffer(combined_pcm_data, dtype=np.int16).astype(np.float32)/32768.0)):.4f}"
                    )
                    
                    if skip_single:
                        # 改进的单人场景判断：使用Conv-TasNet分离结果来判断
                        # 先快速分离，如果分离出>=2个有效源，说明是混合音频
                        try:
                            # 快速分离检测（只分离，不进行后续处理）
                            sources = conn.speech_separation_provider._separate_sources(combined_pcm_data)
                            
                            if len(sources) >= 2:
                                # 分离出2个或更多源，是混合音频，继续完整分离流程
                                logger.bind(tag=TAG).info(
                                    f"检测到混合音频（分离出{len(sources)}个源），启用完整分离流程"
                                )
                                await self._handle_with_speech_separation(conn, asr_audio_task, combined_pcm_data, total_start_time)
                                return
                            elif len(sources) == 1:
                                # 只分离出1个源，可能是单人音频
                                logger.bind(tag=TAG).info(
                                    f"检测到单人音频（分离出1个源），跳过分离，直接使用原始音频"
                                )
                                # 跳过分离，使用原有流程
                            else:
                                # 没有分离出源，使用原有流程
                                logger.bind(tag=TAG).warning("分离未检测到有效源，使用原有流程")
                        except Exception as e:
                            logger.bind(tag=TAG).warning(f"快速分离检测失败: {e}，使用原有流程")
                            import traceback
                            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
                            # 分离检测失败，使用原有流程
                    else:
                        # 配置为不跳过，强制使用分离
                        logger.bind(tag=TAG).info("强制启用语音分离（skip_single_speaker=false）")
                        await self._handle_with_speech_separation(conn, asr_audio_task, combined_pcm_data, total_start_time)
                        return
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"语音分离处理异常，回退到原有流程: {e}")
                    # 继续执行原有流程
            
            # 原有流程（不使用语音分离）
            
            # 预先准备WAV数据
            wav_data = None
            if conn.voiceprint_provider and combined_pcm_data:
                wav_data = self._pcm_to_wav(combined_pcm_data)
            
            # 定义ASR任务
                def run_asr():
                    start_time = time.monotonic()
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                self.speech_to_text(asr_audio_task, conn.session_id, conn.audio_format)
                            )
                            end_time = time.monotonic()
                            logger.bind(tag=TAG).info(f"ASR耗时: {end_time - start_time:.3f}s")
                            return result
                        finally:
                            loop.close()
                    except Exception as e:
                        end_time = time.monotonic()
                        logger.bind(tag=TAG).error(f"ASR失败: {e}")
                        return ("", None)
                
                # 定义声纹识别任务
                def run_voiceprint():
                    if not wav_data:
                        return None, 0.0
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # 如果是本地声纹识别提供者，使用带置信度的方法
                            if hasattr(conn.voiceprint_provider, 'identify_speaker_with_confidence'):
                                # 同步调用，因为identify_speaker_with_confidence是同步方法
                                speaker_name, confidence = conn.voiceprint_provider.identify_speaker_with_confidence(
                                    wav_data, conn.session_id
                                )
                                return speaker_name, confidence
                            else:
                                # 外部API提供者，只返回说话人名称
                                speaker_name = loop.run_until_complete(
                                    conn.voiceprint_provider.identify_speaker(wav_data, conn.session_id)
                                )
                                # 外部API不提供置信度，假设识别成功则置信度为1.0
                                return speaker_name, 1.0 if speaker_name else 0.0
                        finally:
                            loop.close()
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"声纹识别失败: {e}")
                        return None, 0.0
                
                # 使用线程池执行器并行运行
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as thread_executor:
                    asr_future = thread_executor.submit(run_asr)
                    
                    if conn.voiceprint_provider and wav_data:
                        voiceprint_future = thread_executor.submit(run_voiceprint)
                        
                        # 等待两个线程都完成
                        asr_result = asr_future.result(timeout=15)
                        voiceprint_result = voiceprint_future.result(timeout=15)
                        
                        results = {"asr": asr_result, "voiceprint": voiceprint_result}
                    else:
                        asr_result = asr_future.result(timeout=15)
                        results = {"asr": asr_result, "voiceprint": None}
                
                # 处理结果
                asr_result = results.get("asr", ("", None))
                voiceprint_result = results.get("voiceprint", None)
                
                # 解析ASR结果（可能包含情感信息）
                # 支持两种格式：
                # 1. (text, file_path) - 旧格式，不包含情感
                # 2. (text, file_path, emotion, emotion_confidence) - 新格式，包含情感
                if len(asr_result) >= 4:
                    raw_text, _, emotion_from_asr, emotion_conf_from_asr = asr_result
                elif len(asr_result) >= 2:
                    raw_text, _ = asr_result[:2]
                    emotion_from_asr, emotion_conf_from_asr = None, 0.0
                else:
                    raw_text = asr_result[0] if asr_result else ""
                    emotion_from_asr, emotion_conf_from_asr = None, 0.0
                
                # 解析声纹识别结果（可能是元组或None）
                if voiceprint_result and isinstance(voiceprint_result, tuple):
                    speaker_name, confidence = voiceprint_result
                else:
                    speaker_name = voiceprint_result
                    confidence = 1.0 if speaker_name else 0.0
            
            # 情感识别（优先使用ASR提供的情感，否则使用MFCC方案）
            emotion = None
            emotion_confidence = 0.0
            
            # 优先使用ASR模型提供的情感识别（如SenseVoice）
            if emotion_from_asr:
                emotion = emotion_from_asr
                emotion_confidence = emotion_conf_from_asr
                logger.bind(tag=TAG).info(
                    f"✓ 使用SenseVoice情感识别: {emotion}, 置信度={emotion_confidence:.3f}"
                )
            # 如果ASR未提供情感，且启用了情感识别，则使用MFCC方案
            elif hasattr(conn, 'emotion_recognition_enabled') and conn.emotion_recognition_enabled:
                try:
                    emotion_recognizer = get_emotion_recognizer(
                        model_path=getattr(conn, 'emotion_model_path', None),
                        use_simple_emotions=getattr(conn, 'emotion_use_simple', True),
                        sample_rate=16000
                    )
                    emotion, emotion_confidence = emotion_recognizer.recognize_from_pcm(
                        combined_pcm_data,
                        sample_rate=16000
                    )
                    logger.bind(tag=TAG).info(
                        f"⚠ 使用MFCC备选方案情感识别: {emotion}, 置信度={emotion_confidence:.3f}"
                    )
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"MFCC情感识别失败: {e}")
                    emotion = None
            
            # 记录识别结果（格式：识别文本[情感判断：情感类别]）
            if raw_text:
                if emotion:
                    logger.bind(tag=TAG).info(f"[ASR] 识别结果:{raw_text}[情感判断：{emotion}]")
                else:
                    logger.bind(tag=TAG).info(f"[ASR] 识别结果:{raw_text}")
            if speaker_name:
                logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}, 置信度={confidence:.3f}")
            elif conn.voiceprint_provider and hasattr(conn.voiceprint_provider, 'similarity_threshold'):
                # 如果声纹识别失败，也输出置信度信息
                threshold = conn.voiceprint_provider.similarity_threshold
                logger.bind(tag=TAG).info(f"声纹识别置信度={confidence:.3f} < 阈值={threshold:.3f}，跳过LLM处理")
            
            # 性能监控
            total_time = time.monotonic() - total_start_time
            logger.bind(tag=TAG).info(f"总处理耗时: {total_time:.3f}s")
            
            # 检查文本长度
            text_len, _ = remove_punctuation_and_length(raw_text)
            self.stop_ws_connection()
            
            # 检查声纹识别置信度（如果启用了声纹识别）
            should_process = True
            if conn.voiceprint_provider and hasattr(conn.voiceprint_provider, 'similarity_threshold'):
                threshold = conn.voiceprint_provider.similarity_threshold
                if confidence < threshold:
                    logger.bind(tag=TAG).info(
                        f"声纹识别置信度={confidence:.3f} < 阈值={threshold:.3f}，"
                        f"仅输出ASR识别结果，不进行LLM处理"
                    )
                    should_process = False
            
            if text_len > 0 and should_process:
                # 构建包含说话人信息的JSON字符串
                enhanced_text = self._build_enhanced_text(raw_text, speaker_name)
                
                # 使用自定义模块进行上报
                await startToChat(conn, enhanced_text)
                enqueue_asr_report(conn, enhanced_text, asr_audio_task)
            elif text_len > 0:
                # 只输出识别结果，不进行LLM处理
                logger.bind(tag=TAG).info(f"仅输出ASR识别结果: {raw_text}，跳过LLM处理")
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理语音停止失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
    
    async def _handle_with_speech_separation(self, conn, asr_audio_task: List[bytes], combined_pcm_data: bytes, total_start_time: float):
        """使用语音分离处理音频"""
        try:
            logger.bind(tag=TAG).info("开始语音分离处理...")
            
            # 进行语音分离
            separated_audios = await conn.speech_separation_provider.separate(
                combined_pcm_data,
                sample_rate=16000,
                num_speakers=None  # 自动检测
            )
            
            if not separated_audios or len(separated_audios) == 0:
                logger.bind(tag=TAG).warning("语音分离未检测到说话人，使用原有流程处理")
                # 如果分离失败，回退到原有流程（已有代码会继续执行）
                return
            
            logger.bind(tag=TAG).info(f"语音分离完成，检测到 {len(separated_audios)} 个说话人片段")
            
            # 对每个分离出的音频片段进行处理
            for i, separated in enumerate(separated_audios):
                logger.bind(tag=TAG).info(
                    f"处理说话人片段 {i+1}/{len(separated_audios)}: "
                    f"speaker_id={separated['speaker_id']}, "
                    f"时间段={separated['start']:.2f}s-{separated['end']:.2f}s"
                )
                
                # 将分离出的PCM数据转换为列表格式（用于ASR）
                audio_segment = separated["audio"]
                # 模拟分帧（每帧60ms，16kHz，即960字节）
                frame_size = 960
                audio_frames = [audio_segment[j:j+frame_size] for j in range(0, len(audio_segment), frame_size)]
                audio_frames = [f for f in audio_frames if len(f) > 0]
                
                if not audio_frames:
                    logger.bind(tag=TAG).warning(f"说话人片段 {i+1} 音频数据为空，跳过")
                    continue
                
                # 准备WAV数据用于声纹识别
                wav_data = self._pcm_to_wav(audio_segment)
                
                # 定义ASR任务
                def run_asr():
                    start_time = time.monotonic()
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                self.speech_to_text(audio_frames, conn.session_id, "pcm")
                            )
                            end_time = time.monotonic()
                            logger.bind(tag=TAG).info(f"ASR耗时: {end_time - start_time:.3f}s (片段 {i+1})")
                            return result
                        finally:
                            loop.close()
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"ASR失败 (片段 {i+1}): {e}")
                        return ("", None, None, 0.0)
                
                # 定义声纹识别任务
                def run_voiceprint():
                    if not wav_data or not conn.voiceprint_provider:
                        return None, 0.0
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            if hasattr(conn.voiceprint_provider, 'identify_speaker_with_confidence'):
                                speaker_name, confidence = conn.voiceprint_provider.identify_speaker_with_confidence(
                                    wav_data, conn.session_id
                                )
                                return speaker_name, confidence
                            else:
                                speaker_name = loop.run_until_complete(
                                    conn.voiceprint_provider.identify_speaker(wav_data, conn.session_id)
                                )
                                return speaker_name, 1.0 if speaker_name else 0.0
                        finally:
                            loop.close()
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"声纹识别失败 (片段 {i+1}): {e}")
                        return None, 0.0
                
                # 并行执行
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    asr_future = executor.submit(run_asr)
                    voiceprint_future = executor.submit(run_voiceprint) if (wav_data and conn.voiceprint_provider) else None
                    
                    asr_result = asr_future.result(timeout=15)
                    voiceprint_result = voiceprint_future.result(timeout=15) if voiceprint_future else (None, 0.0)
                
                # 解析ASR结果（可能包含情感信息）
                if len(asr_result) >= 4:
                    raw_text, _, emotion_from_asr, emotion_conf_from_asr = asr_result
                elif len(asr_result) >= 2:
                    raw_text, _ = asr_result[:2]
                    emotion_from_asr, emotion_conf_from_asr = None, 0.0
                else:
                    raw_text = asr_result[0] if asr_result else ""
                    emotion_from_asr, emotion_conf_from_asr = None, 0.0
                
                speaker_name, confidence = voiceprint_result if isinstance(voiceprint_result, tuple) else (voiceprint_result, 1.0 if voiceprint_result else 0.0)
                
                # 情感识别（优先使用ASR提供的情感，否则使用MFCC方案）
                emotion = None
                emotion_confidence = 0.0
                
                # 优先使用ASR模型提供的情感识别（如SenseVoice）
                if emotion_from_asr:
                    emotion = emotion_from_asr
                    emotion_confidence = emotion_conf_from_asr
                    logger.bind(tag=TAG).debug(
                        f"使用ASR模型情感识别 (片段 {i+1}): {emotion}, 置信度={emotion_confidence:.3f}"
                    )
                # 如果ASR未提供情感，且启用了情感识别，则使用MFCC方案
                elif hasattr(conn, 'emotion_recognition_enabled') and conn.emotion_recognition_enabled:
                    try:
                        emotion_recognizer = get_emotion_recognizer(
                            model_path=getattr(conn, 'emotion_model_path', None),
                            use_simple_emotions=getattr(conn, 'emotion_use_simple', True),
                            sample_rate=16000
                        )
                        emotion, emotion_confidence = emotion_recognizer.recognize_from_pcm(
                            audio_segment,  # 使用分离出的PCM数据
                            sample_rate=16000
                        )
                        logger.bind(tag=TAG).debug(
                            f"使用MFCC情感识别 (片段 {i+1}): {emotion}, 置信度={emotion_confidence:.3f}"
                        )
                    except Exception as e:
                        logger.bind(tag=TAG).warning(f"MFCC情感识别失败 (片段 {i+1}): {e}")
                        emotion = None
                
                # 记录结果
                if raw_text:
                    if emotion:
                        logger.bind(tag=TAG).info(f"[ASR-片段 {i+1}] 识别结果: {raw_text}[情感判断：{emotion}]")
                    else:
                        logger.bind(tag=TAG).info(f"[ASR-片段 {i+1}] 识别结果: {raw_text}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"[声纹-片段 {i+1}] 说话人: {speaker_name}, 置信度={confidence:.3f}")
                
                # 处理每个片段的结果
                text_len, _ = remove_punctuation_and_length(raw_text)
                
                # 检查声纹识别置信度（如果启用了声纹识别）
                should_process = True
                if conn.voiceprint_provider and hasattr(conn.voiceprint_provider, 'similarity_threshold'):
                    threshold = conn.voiceprint_provider.similarity_threshold
                    if confidence < threshold:
                        logger.bind(tag=TAG).info(
                            f"[片段 {i+1}] 声纹识别置信度={confidence:.3f} < 阈值={threshold:.3f}，"
                            f"仅输出ASR识别结果，不进行LLM处理"
                        )
                        should_process = False
                
                if text_len > 0 and should_process:
                    # 构建包含说话人信息的文本
                    enhanced_text = self._build_enhanced_text(raw_text, speaker_name)
                    
                    # 使用自定义模块进行上报
                    await startToChat(conn, enhanced_text)
                    enqueue_asr_report(conn, enhanced_text, audio_frames)
                elif text_len > 0:
                    # 只输出识别结果，不进行LLM处理
                    logger.bind(tag=TAG).info(f"[片段 {i+1}] 仅输出ASR识别结果: {raw_text}，跳过LLM处理")
            
            # 性能监控
            total_time = time.monotonic() - total_start_time
            logger.bind(tag=TAG).info(f"语音分离处理总耗时: {total_time:.3f}s")
            self.stop_ws_connection()
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"语音分离处理失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")
            # 异常时让原有流程继续处理
    
    def _build_enhanced_text(self, text: str, speaker_name: Optional[str]) -> str:
        """构建包含说话人信息的文本"""
        if speaker_name and speaker_name.strip():
            return json.dumps({
                "speaker": speaker_name,
                "content": text
            }, ensure_ascii=False)
        else:
            return text

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """将PCM数据转换为WAV格式"""
        if len(pcm_data) == 0:
            logger.bind(tag=TAG).warning("PCM数据为空，无法转换WAV")
            return b""
        
        # 确保数据长度是偶数（16位音频）
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        
        # 创建WAV文件头
        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)      # 单声道
                wav_file.setsampwidth(2)      # 16位
                wav_file.setframerate(16000)  # 16kHz采样率
                wav_file.writeframes(pcm_data)
            
            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            
            return wav_data
        except Exception as e:
            logger.bind(tag=TAG).error(f"WAV转换失败: {e}")
            return b""

    def stop_ws_connection(self):
        pass

    def save_audio_to_file(self, pcm_data: List[bytes], session_id: str) -> str:
        """PCM数据保存为WAV文件"""
        module_name = __name__.split(".")[-1]
        file_name = f"asr_{module_name}_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    @abstractmethod
    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass

    @staticmethod
    def decode_opus(opus_data: List[bytes]) -> List[bytes]:
        """将Opus音频数据解码为PCM数据"""
        try:
            from core.utils.util import config_to_sample_rate
            
            pcm_data = []
            decoder = None
            last_sample_rate = 16000
            last_channels = 1
            
            for i, opus_packet in enumerate(opus_data):
                try:
                    if not opus_packet or len(opus_packet) == 0:
                        continue
                    
                    # 解析TOC字节
                    toc = opus_packet[0]
                    config = (toc >> 3) & 0x1F
                    stereo = (toc >> 2) & 0x01
                    frame_code = toc & 0x03
                    opus_sample_rate = config_to_sample_rate(config)
                    channels = 2 if stereo == 1 else 1
                    
                    # 如果采样率或声道数变化，重新创建解码器
                    if decoder is None or opus_sample_rate != last_sample_rate or channels != last_channels:
                        decoder = opuslib_next.Decoder(opus_sample_rate, channels)
                        last_sample_rate = opus_sample_rate
                        last_channels = channels
                    
                    # 根据采样率确定帧大小（60ms）
                    if opus_sample_rate == 8000:
                        buffer_size = 480
                    elif opus_sample_rate == 12000:
                        buffer_size = 720
                    elif opus_sample_rate == 16000:
                        buffer_size = 960
                    elif opus_sample_rate == 24000:
                        buffer_size = 1440
                    elif opus_sample_rate == 48000:
                        buffer_size = 2880
                    else:
                        buffer_size = 960  # 默认
                    
                    # 记录Opus解码信息（详细日志，类似图片中的格式）
                    logger.bind(tag=TAG).info(
                        f"[Opus解码] 帧 {i}: TOC=0x{toc:02X}, config={config}, "
                        f"采样率={opus_sample_rate}Hz, stereo={stereo}, frame_code={frame_code}, "
                        f"长度={len(opus_packet)}"
                    )
                    
                    pcm_frame = decoder.decode(opus_packet, buffer_size)
                    if pcm_frame and len(pcm_frame) > 0:
                        pcm_data.append(pcm_frame)
                        
                except opuslib_next.OpusError as e:
                    logger.bind(tag=TAG).warning(f"Opus解码错误，跳过数据包 {i}: {e}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"音频处理错误，数据包 {i}: {e}")
            
            return pcm_data
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"音频解码过程发生错误: {e}")
            return []
