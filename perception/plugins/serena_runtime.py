"""Serena's three-stage Kokoro runtime with explicit GPU backends.

The host phase DSP follows tools/phase_dsp.py from the training export.
The working JP6 candidate uses TRT encoder/acoustic and isolated ORT CUDA
vocoder; the all-TRT candidate has a recorded numerical/content regression.
"""
from pathlib import Path
import json
import os
import threading

import numpy as np


def excitation_inputs(f0, seed=42):
    f0 = np.asarray(f0)
    if (f0.dtype != np.float32 or f0.ndim != 2 or f0.shape[0] != 1
            or f0.shape[1] < 2 or not np.isfinite(f0).all()):
        raise ValueError("Expected finite float32 F0[1,F], F >= 2")
    rng = np.random.default_rng(seed)
    initial = rng.random((1, 9), dtype=np.float32)
    initial[:, 0] = 0
    rad = np.remainder(f0[:, :, None] * np.arange(1, 10, dtype=np.float32)[None, None, :]
                       / np.float32(24000), np.float32(1))
    phase = np.cumsum(rad, axis=1, dtype=np.float64).astype(np.float32)
    phase = phase * np.float32(2) * np.float32(np.pi) * np.float32(300)
    count = f0.shape[1]
    result = np.empty((1, count * 300, 9), dtype=np.float32)
    for start in range(0, count * 300, 32768):
        stop = min(count * 300, start + 32768)
        at = (np.arange(start, stop, dtype=np.float32) + np.float32(.5)) * np.float32(1 / 300) - np.float32(.5)
        at = np.maximum(at, np.float32(0))
        left = np.floor(at).astype(np.int64)
        right = np.minimum(left + 1, count - 1)
        fraction = (at - left).astype(np.float32)[None, :, None]
        right_product = phase[:, right, :] * fraction
        result[:, start:stop, :] = (phase[:, left, :].astype(np.float64)
            * (np.float32(1) - fraction).astype(np.float64)
            + right_product.astype(np.float64)).astype(np.float32)
    noise = rng.standard_normal(result.shape, dtype=np.float32)
    return result, np.ascontiguousarray(noise)


class OrtVocoderSession:
    """The standalone ORT CUDA provider lives outside sherpa's process."""

    def __init__(self, model_path):
        import uuid
        from plugins import ort_worker
        self.worker = ort_worker.get_worker()
        self.key = 'serena.vocoder.' + uuid.uuid4().hex
        self.session = self.worker.load(self.key, str(model_path),
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            {'intra_op_num_threads': 2, 'provider_options': [
                {'cudnn_conv_algo_search': 'HEURISTIC', 'use_tf32': '0'}, {}]})
        if self.session.get_providers()[0] != 'CUDAExecutionProvider':
            self.close()
            raise RuntimeError('Serena vocoder requires ONNX CUDA provider')

    def run(self, inputs):
        return dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, inputs)))

    def close(self):
        if self.session is not None:
            self.worker.unload(self.key)
            self.session = None


class SerenaRuntime:
    sample_rate = 24000
    max_frames = 191  # Matches compile_jp6_candidate.py's acoustic profile.

    def __init__(self, model_dir, vocoder_backend='tensorrt'):
        # TensorRT's Myelin engine requires the same override at build and run.
        if os.environ.get("NVIDIA_TF32_OVERRIDE") not in (None, "0"):
            raise RuntimeError("Serena FP32 plans require NVIDIA_TF32_OVERRIDE=0")
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        from plugins.vits2_tts_trt.runtime.backends.trt_cuda_session import (
            CudaRuntime, TensorRTCudaSession,
        )
        root = Path(model_dir)
        self.style = np.load(root / "style.npy", allow_pickle=False)
        if (self.style.shape != (1, 256) or self.style.dtype != np.float32
                or not np.isfinite(self.style).all()):
            raise ValueError("Invalid Serena style vector")
        self.cuda = CudaRuntime()
        self.sessions = {}
        self.lock = threading.RLock()
        self.host_source_excitation = False
        if vocoder_backend not in ('tensorrt', 'ort-cuda'):
            raise ValueError('Unsupported Serena vocoder backend')
        try:
            for stage in ("encoder", "acoustic", "vocoder"):
                if stage == 'vocoder' and vocoder_backend == 'ort-cuda':
                    self.sessions[stage] = OrtVocoderSession(root / 'vocoder.onnx')
                    continue
                metadata = json.loads((root / (stage + ".json")).read_text())
                if metadata.get("status") != "compiled_not_runtime_validated":
                    raise ValueError("Missing successful build record: " + stage)
                self.sessions[stage] = TensorRTCudaSession(
                    root / (stage + ".plan"), self.cuda,
                    expected_sha256=metadata["plan_sha256"],
                )
                if stage == 'vocoder':
                    self.host_source_excitation = 'source_excitation' in metadata.get('inputs', {})
        except BaseException:
            self.close()
            raise

    def synthesize_tokens(self, ids, speed=1.0):
        """Yield 24kHz audio; split rather than truncate at the duration limit."""
        ids = list(ids)
        if not ids:
            return
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Speed must be finite and positive")
        # Keep encoder inputs inside its profile, including boundary tokens.
        if len(ids) > 254:
            yield from self.synthesize_tokens(ids[:254], speed)
            yield from self.synthesize_tokens(ids[254:], speed)
            return
        with self.lock:
            tokens = np.array([[0, *ids, 0]], dtype=np.int64)
            enc = self.sessions["encoder"].run({"tokens": tokens, "style": self.style,
                "speed": np.array([speed], dtype=np.float32)})
            durations = enc["duration"].reshape(-1)
            if len(durations) != tokens.shape[1] or np.any(durations < 1):
                raise ValueError("Invalid encoder durations")
            if int(durations.sum()) > self.max_frames:
                if len(ids) == 1:
                    raise ValueError("Single phoneme exceeds the acoustic profile")
                # Re-encode both pieces; no durations or audio are discarded.
                middle = len(ids) // 2
                yield from self.synthesize_tokens(ids[:middle], speed)
                yield from self.synthesize_tokens(ids[middle:], speed)
                return
            indices = np.repeat(np.arange(tokens.shape[1], dtype=np.int32), durations)
            acoustic = self.sessions["acoustic"].run({"d": enc["d"], "t_en": enc["t_en"],
                "indices": indices, "style": self.style})
            phase, noise = excitation_inputs(acoustic["f0"])
            vocoder_inputs = {**acoustic, "style": self.style,
                "source_phase": phase, "excitation_noise": noise}
            if self.host_source_excitation:
                uv = np.repeat((acoustic['f0'] > 10).astype(np.float32), 300, axis=1)[:, :, None]
                noise_amp = uv * np.float32(.003) + (np.float32(1) - uv) * np.float32(.1) / np.float32(3)
                excitation = np.sin(phase) * np.float32(.1) * uv + noise_amp * noise
                vocoder_inputs = {'features': acoustic['features'], 'style': self.style,
                                  'source_excitation': excitation}
            audio = self.sessions["vocoder"].run(vocoder_inputs)["audio"].reshape(-1)
            if audio.size != int(durations.sum()) * 600 or not np.isfinite(audio).all():
                raise ValueError("Invalid Serena waveform")
        yield audio

    def close(self):
        # Contexts/engines must be released before destroying their CUDA stream.
        with self.lock:
            for session in self.sessions.values():
                session.close()
                if hasattr(session, 'context'):
                    session.context = None
                    session.engine = None
                    session.runtime = None
            self.sessions.clear()
            self.cuda.close()
