"""Serena adapter using the same Misaki frontend as the training pipeline."""
import json
from pathlib import Path
import threading
import select
import subprocess
import sys
import os

import numpy as np

from plugins.tts import TTSAdapter
from plugins.serena_runtime import SerenaRuntime


class SerenaFrontendWorker:
    def __init__(self, config_path):
        self.lock = threading.RLock()
        root = Path(config_path).parent
        env = dict(os.environ)
        paths = [str(root / name) for name in ('frontend', 'language_assets') if (root / name).is_dir()]
        if paths:
            env['PYTHONPATH'] = os.pathsep.join(paths + [env.get('PYTHONPATH', '')])
        self.process = subprocess.Popen([sys.executable, '-u',
            str(Path(__file__).with_name('serena_frontend.py')), str(config_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        try:
            self._read(60)
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            self.close()
            raise TimeoutError('Serena frontend worker timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Serena frontend worker exited')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def __call__(self, text, language):
        with self.lock:
            self.process.stdin.write(json.dumps({'text': text, 'language': language}) + '\n')
            self.process.stdin.flush()
            return self._read(30)['result']

    def close(self):
        with self.lock:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()


class SerenaTTSAdapter(TTSAdapter):
    dry_run_text = "您好。"

    def __init__(self, model_dir, speed=1.0, language="auto", vocoder_backend='ort-cuda',
                 first_chunk_tokens=48, chunk_tokens=48):
        self.lock = threading.RLock()
        self.first_chunk_tokens = int(first_chunk_tokens)
        self.chunk_tokens = int(chunk_tokens)
        if not 8 <= self.chunk_tokens <= 48:
            raise ValueError('Serena chunk_tokens must be in 8..48')
        if not 8 <= self.first_chunk_tokens <= 48:
            raise ValueError('Serena first_chunk_tokens must be in 8..48')
        self.frontend = SerenaFrontendWorker(Path(model_dir) / "config.json")
        self.set_speed(speed)
        self.set_language(language)
        try:
            self.runtime = SerenaRuntime(model_dir, vocoder_backend)
        except BaseException:
            self.frontend.close()
            raise

    def set_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Speed must be finite and positive")
        with self.lock:
            self.speed = speed

    def set_language(self, language):
        if language not in ("auto", "zh", "en"):
            raise ValueError("Serena supports auto, zh and en")
        with self.lock:
            self.language = language

    def synthesize_stream(self, text):
        from utils.resample import downsample_24k_to_16k
        if not text.strip():
            return
        with self.lock:
            phonemes, ids = self.frontend(text, self.language)
            first_chunk = True
            # Prefer word/punctuation boundaries when limiting first-audio work.
            while ids:
                stop = min(self.first_chunk_tokens if first_chunk else self.chunk_tokens, len(ids))
                if stop < len(ids):
                    breaks = [i + 1 for i, p in enumerate(phonemes[:stop])
                              if p in ' .,;:!?，。；：！？' and i >= stop // 2]
                    if breaks:
                        stop = breaks[-1]
                for audio in self.runtime.synthesize_tokens(ids[:stop], self.speed):
                    yield downsample_24k_to_16k(audio)
                ids, phonemes = ids[stop:], phonemes[stop:]
                first_chunk = False

    def synthesize(self, text):
        return b''.join(self.synthesize_stream(text))

    def close(self):
        with self.lock:
            self.runtime.close()
            self.frontend.close()
