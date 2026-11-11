"""Training-compatible CPU frontend, isolated from optional Torch imports."""
import json
from pathlib import Path
import re
import unicodedata


class SerenaFrontend:
    def __init__(self, config_path):
        from misaki import en, espeak, zh
        self.en = en.G2P(trf=False, british=False,
            fallback=espeak.EspeakFallback(british=False), unk='❓')
        self.zh = zh.ZHG2P(version=None, unk='❓')
        self.vocab = json.loads(Path(config_path).read_text())["vocab"]

    def __call__(self, text, language="auto"):
        normalized = unicodedata.normalize('NFKC', text).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.replace('·', ' ').replace('–', '—')
        if re.search(r'[À-ÖØ-öø-ÿ]', normalized):
            raise ValueError('Unsupported non-English Latin text')
        if language == 'en' or (language == 'auto' and not re.search(r'[\u3400-\u9fff]', normalized)):
            phonemes = self.en(normalized)[0]
        else:
            spans = re.split(r"([A-Za-z][A-Za-z0-9_'’\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_'’\-]*)*)", normalized)
            result = []
            for span in spans:
                if not span.strip():
                    continue
                result.append(self.en(span)[0] if re.match('[A-Za-z]', span) else self.zh(span)[0])
            phonemes = ' '.join(result)
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        unknown = sorted(set(phonemes) - set(self.vocab))
        if unknown:
            raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        if not phonemes:
            raise ValueError('Empty phoneme sequence')
        return phonemes, [self.vocab[p] for p in phonemes]


def worker_main():
    import contextlib
    import sys
    # This worker only uses spaCy's CPU tagger. Thinc otherwise imports the
    # robot image's entire unused Torch/CUDA stack (~700 MB per frontend).
    # The sentinel is confined to this process; perception/ORT are unaffected.
    sys.modules['torch'] = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            frontend = SerenaFrontend(sys.argv[1])
        print(json.dumps({'ready': True}), flush=True)
    except Exception as error:
        print(json.dumps({'error': str(error)}), flush=True)
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = frontend(request['text'], request['language'])
            reply = {'result': result}
        except Exception as error:
            reply = {'error': f'{type(error).__name__}: {error}'}
        print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    worker_main()
