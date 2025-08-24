import re
import time
import hashlib
import numpy as np
import torch
import lmstudio as lms
from comfy.utils import ProgressBar

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Transformers library not found. Please install it with: pip install transformers")
    AutoTokenizer = None

from .decoder import (
    convert_to_audio,
    get_snac_device,
)

CUSTOM_RE = re.compile(r"<custom_token_(\d+)>")
SAMPLE_RATE = 24000

VOICES = [
    "tara","leah","jess","leo","dan","mia","zac","zoe",  # English
]

TOKENIZER = None
def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None and AutoTokenizer is not None:
        try:
            TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"Warning: Could not load GPT-2 tokenizer for estimation: {e}")
    return TOKENIZER


def _format_prompt(text, voice):
    return f"<|audio|>{voice}: {text}<|eot_id|>"

class OrpheusLMStudioTTS:
    """
    Streams text completions from LM Studio, collects all audio tokens,
    then decodes with SNAC using a two-phase progress bar that accurately
    reflects the workload of the LLM (70%) and the SNAC decoder (30%).
    Includes performance timers for debugging.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello there. <chuckle> This should sound natural.", "multiline": True}),
                "voice": (VOICES, ),
                "model_key": ("STRING", {"default": ""}),
                "max_tokens": ("INT", {"default": 4096, "min": 256, "max": 131072, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01}),
                "timeout_seconds": ("INT", {"default": 300, "min": 5, "max": 3600, "step": 1}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "stop_at_eot": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": { "custom_stop": ("STRING", {"default": ""}) }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "Orpheus/LM Studio"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        m = hashlib.sha256()
        for k, v in kwargs.items():
            m.update(str(v).encode())
        return m.hexdigest()

    def _get_model(self, client, model_key, auto_unload, unload_delay):
        if model_key and model_key.strip():
            return client.llm.model(model_key, ttl=unload_delay) if (auto_unload == "True" and unload_delay > 0) else client.llm.model(model_key)
        return client.llm.model()

    def run(
        self, text, voice, model_key, max_tokens, temperature, top_p,
        repeat_penalty, timeout_seconds, auto_unload, unload_delay,
        stop_at_eot, seed, debug, custom_stop="",
    ):
        if seed == -1:
            seed = int(time.time() * 1000) & 0xffffffffffffffff

        prompt = _format_prompt(text, voice)

        stop_strings = ["<|eot_id|>"] if stop_at_eot else []
        if custom_stop.strip():
            stop_strings.append(custom_stop.strip())

        config = {
            "temperature": float(temperature), "topP": float(top_p),
            "maxTokens": int(max_tokens), "seed": int(seed),
            "repeatPenalty": float(repeat_penalty),
        }
        if stop_strings:
            config["stopStrings"] = stop_strings

        if debug:
            print(f"[Orpheus-LMS] SNAC device: {get_snac_device()}")
            print(f"[Orpheus-LMS] Using model_key='{model_key or '(default)'}'")
            print(f"[Orpheus-LMS] Config: {config}")

        pbar = ProgressBar(100)
        
        start_time = time.time()
        llm_end_time, snac_prep_end_time, snac_decode_end_time = 0, 0, 0
        
        # --- Phase 1: LLM Token Generation (0% -> 70%) ---
        estimated_total_audio_tokens = 100
        tokenizer = get_tokenizer()
        if tokenizer:
            input_token_count = len(tokenizer.encode(text))
            estimated_total_audio_tokens = max(1, input_token_count * 25)
            if debug:
                print(f"[Orpheus-LMS] Input text tokens: {input_token_count}, estimated audio tokens for progress bar: {estimated_total_audio_tokens}")
        
        llm_output_fragments = []
        model = None
        current_audio_tokens = 0

        with lms.Client() as client:
            try:
                model = self._get_model(client, model_key, auto_unload, unload_delay)
                stream = model.complete_stream(prompt, config=config)
                deadline = time.time() + timeout_seconds

                for frag in stream:
                    if time.time() > deadline:
                        try: stream.cancel()
                        except: pass
                        raise TimeoutError(f"LLM prediction exceeded {timeout_seconds}s")
                    
                    content = frag.content or ""
                    llm_output_fragments.append(content)
                    
                    if content.startswith('<custom_token_'):
                        current_audio_tokens += 1
                        progress_ratio = min(1.0, current_audio_tokens / estimated_total_audio_tokens)
                        pbar.update_absolute(int(progress_ratio * 70))

                try:
                    result = stream.result()
                    if debug and getattr(result, "stats", None):
                        s = result.stats
                        print(f"[Orpheus-LMS] LLM Stats: TTFT={s.time_to_first_token_sec:.2f}s, Tokens={s.predicted_tokens_count}, Stop Reason='{s.stop_reason}'")
                except: pass

            finally:
                llm_end_time = time.time()
                pbar.update_absolute(70)
                if model and auto_unload == "True" and unload_delay == 0:
                    try: model.unload()
                    except: pass
        
        llm_output_str = "".join(llm_output_fragments)

        # --- Phase 2: SNAC Audio Decoding (70% -> 100%) ---
        collected_pcm = bytearray()
        
        all_tokens = [int(m.group(1)) for m in CUSTOM_RE.finditer(llm_output_str)]
        
        total_accepted_tokens = 0
        mapped_tokens = []
        for n in all_tokens:
            mapped = n - 10 - ((total_accepted_tokens % 7) * 4096)
            if mapped > 0:
                mapped_tokens.append(mapped)
                total_accepted_tokens += 1

        total_snac_calls = max(0, ((total_accepted_tokens - 28) // 7) + 1) if total_accepted_tokens > 27 else 0
        snac_prep_end_time = time.time()
        
        if debug:
            # ADDED BACK: Log to compare the estimate with the actual result.
            print(f"[Orpheus-LMS] Audio Token Estimate vs. Actual: {estimated_total_audio_tokens} vs. {total_accepted_tokens}")
            print(f"[Orpheus-LMS] Expecting {total_snac_calls} SNAC decoding calls.")

        if total_snac_calls > 0:
            for i in range(total_snac_calls):
                start_idx = i * 7
                end_idx = start_idx + 28
                sub = mapped_tokens[start_idx:end_idx]

                audio_chunk = convert_to_audio(sub, i)
                if audio_chunk:
                    collected_pcm.extend(audio_chunk)

                progress = 70 + int(((i + 1) / total_snac_calls) * 30)
                pbar.update_absolute(progress)
        
        snac_decode_end_time = time.time()
        pbar.update_absolute(100)

        if debug:
            llm_duration = llm_end_time - start_time
            prep_duration = snac_prep_end_time - llm_end_time
            decode_duration = snac_decode_end_time - snac_prep_end_time
            total_duration = snac_decode_end_time - start_time
            secs = len(collected_pcm) / 2 / SAMPLE_RATE
            
            print("--- Performance ---")
            print(f"  LLM Stream Collection: {llm_duration:.2f}s")
            print(f"  SNAC Pre-calculation:  {prep_duration:.4f}s")
            print(f"  SNAC Decoding:         {decode_duration:.2f}s")
            print(f"  Total Generation Time: {total_duration:.2f}s")
            print(f"  Final Audio Duration:  {secs:.2f}s")
            print("-------------------")

        if not collected_pcm:
            silence = np.zeros(int(SAMPLE_RATE / 10), dtype=np.int16).tobytes()
            collected_pcm.extend(silence)

        pcm_i16 = np.frombuffer(bytes(collected_pcm), dtype=np.int16)
        audio_f32 = (pcm_i16.astype(np.float32) / 32767.0).reshape(1, -1)
        waveform = torch.from_numpy(audio_f32)

        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": SAMPLE_RATE}
        return (audio,)

NODE_CLASS_MAPPINGS = {"OrpheusLMStudioTTS": OrpheusLMStudioTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"OrpheusLMStudioTTS": "Orpheus TTS (LM Studio)"}