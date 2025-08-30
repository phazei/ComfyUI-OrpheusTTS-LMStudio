import re
import time
import math
import threading
import queue
import hashlib
from collections import deque

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

# --- Constants ----------------------------------------------------------------

CUSTOM_RE = re.compile(r"<custom_token_(\d+)>")
SAMPLE_RATE = 24000

VOICES = ["tara","leah","jess","leo","dan","mia","zac","zoe"]  # English voices

# Progress smoothing (single bar)
PROGRESS_EMA_ALPHA = 0.85   # smooth the bar
TOTAL_EMA_BETA     = 0.20   # smooth the estimated total time
PROGRESS_TAIL_LOCK = 99.5   # hold until both LLM & SNAC done

# Sliding window sizes (seconds) for live rates
LLM_RATE_WINDOW_S  = 0.5
SNAC_RATE_WINDOW_S = 0.6

# --- Tokenizer cache ----------------------------------------------------------

TOKENIZER = None
def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None and AutoTokenizer is not None:
        try:
            TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"Warning: Could not load GPT-2 tokenizer for estimation: {e}")
    return TOKENIZER

# --- Helpers ------------------------------------------------------------------

def _format_prompt(text, voice):
    return f"<|audio|>{voice}: {text}<|eot_id|>"

def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s)-1, max(0, int(round((p/100.0) * (len(s)-1)))))
    return s[idx]

class SlidingRate:
    """Lightweight tokens/sec (or calls/sec) over a sliding time window."""
    def __init__(self, window_s):
        self.window_s = float(window_s)
        self.events = deque()  # (time, count)
        self.total = 0

    def add(self, n=1, now=None):
        now = time.time() if now is None else now
        self.events.append((now, n))
        self.total += n
        cutoff = now - self.window_s
        while self.events and self.events[0][0] < cutoff:
            t, c = self.events.popleft()
            self.total -= c

    def rate(self, now=None):
        now = time.time() if now is None else now
        if not self.events:
            return 0.0
        earliest_t = self.events[0][0]
        dt = max(1e-6, now - max(earliest_t, now - self.window_s))
        return self.total / dt

# --- Node ---------------------------------------------------------------------

class OrpheusTTSLMStudio:
    """
    Streaming LM Studio with parallel SNAC decoding and a single progress bar
    that reflects the critical path (max of LLM vs SNAC). Detailed logs kept.
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
                "auto_unload": (["True", "False"], {"default": "False"}),
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

        # --- Estimate total audio tokens upfront (hardware-agnostic) ----------
        estimated_total_audio_tokens = 100
        tokenizer = get_tokenizer()
        if tokenizer:
            input_token_count = len(tokenizer.encode(text))
            estimated_total_audio_tokens = max(1, input_token_count * 25)
            if debug:
                print(f"[Orpheus-LMS] Input text tokens: {input_token_count}, "
                      f"estimated audio tokens: {estimated_total_audio_tokens}")
        N_est = estimated_total_audio_tokens

        # Derived estimated SNAC calls from N_est
        def est_snac_calls_from(n):
            return 0 if n <= 27 else ((n - 28) // 7) + 1
        C_est = est_snac_calls_from(N_est)

        # --- Streaming + Parallel SNAC setup ---------------------------------
        llm_frag_count = 0
        chars_total = 0
        current_audio_tokens = 0  # N
        total_accepted_tokens = 0
        mapped_tokens = []        # grows as we accept tokens
        next_snac_index = 0       # how many windows were queued
        C_done = 0                # completed SNAC calls

        # performance breakdown for the loop
        first_fragment_rel = None
        total_wait_time = 0.0
        total_proc_time = 0.0
        wait_samples, proc_samples = [], []
        max_wait = (0.0, -1)
        max_proc = (0.0, -1)

        # live rates
        llm_rate = SlidingRate(LLM_RATE_WINDOW_S)   # tokens/sec (audio tokens)
        snac_rate = SlidingRate(SNAC_RATE_WINDOW_S) # calls/sec
        t0_stream = None
        t0_first_audio_token = None

        # progress state
        progress_prev = 0.0
        last_progress_push_t = 0.0
        T0 = None  # baseline total time (smoothed)

        # SNAC worker (single-threaded; order preserved)
        snac_q = queue.Queue(maxsize=1024)
        pcm_lock = threading.Lock()
        collected_pcm = bytearray()

        def snac_worker():
            nonlocal C_done
            while True:
                item = snac_q.get()
                if item is None:
                    snac_q.task_done()
                    break
                i, sub = item
                t_call_start = time.time()
                audio_chunk = convert_to_audio(sub, i)
                t_call_end = time.time()
                dt = t_call_end - t_call_start

                # append PCM in-order
                if audio_chunk:
                    with pcm_lock:
                        collected_pcm.extend(audio_chunk)

                C_done += 1
                snac_rate.add(1, now=t_call_end)

                # if debug:
                #     print(f"[SNAC] call #{i:03d} len={len(sub)} "
                #           f"time={dt*1000:.2f}ms  calls/sec={snac_rate.rate():.2f}")
                snac_q.task_done()

        worker = threading.Thread(target=snac_worker, daemon=True)
        worker.start()

        # helper: queue any ready SNAC windows as mapped_tokens grows
        def queue_ready_windows():
            """Queue 28-token windows every stride of 7 when available."""
            nonlocal next_snac_index
            queued = 0
            while True:
                start_idx = next_snac_index * 7
                end_idx = start_idx + 28
                if len(mapped_tokens) >= end_idx:
                    sub = mapped_tokens[start_idx:end_idx]
                    snac_q.put((next_snac_index, sub))
                    queued += 1
                    next_snac_index += 1
                else:
                    break
            return queued

        # progress computation: single bar from critical path
        def update_progress(force=False):
            nonlocal progress_prev, last_progress_push_t, T0
            now = time.time()

            r_llm = llm_rate.rate(now)
            r_sn  = snac_rate.rate(now)

            # Remaining LLM time (audio tokens critical path)
            t_llm = 0.0
            rem_tokens = max(0, N_est - current_audio_tokens)
            if r_llm > 0.0:
                t_llm = rem_tokens / r_llm
            else:
                # before any tokens, keep bar at 0
                t_llm = float('inf') if current_audio_tokens == 0 else 0.0

            # Remaining SNAC time
            # compute C_est from current N_est
            c_est = est_snac_calls_from(N_est)
            rem_calls = max(0, c_est - C_done)
            if current_audio_tokens < 28:
                # SNAC not started: don't let it influence early progress
                t_snac = 0.0
            else:
                if r_sn > 0.0:
                    t_snac = rem_calls / r_sn
                else:
                    # first call not finished yet -> treat as unknown; let LLM drive
                    t_snac = 0.0

            # Critical path remaining
            t_rem = max(t_llm, t_snac)

            # Establish / smooth baseline total time
            if r_llm > 0.0:
                llm_total = N_est / r_llm
            else:
                llm_total = float('inf')

            if current_audio_tokens >= 28 and r_sn > 0.0:
                snac_total = c_est / r_sn
            else:
                snac_total = 0.0

            base_total = max(llm_total, snac_total)
            if math.isfinite(base_total) and base_total > 1e-3:
                if T0 is None:
                    T0 = base_total
                else:
                    T0 = (1 - TOTAL_EMA_BETA) * T0 + TOTAL_EMA_BETA * base_total

            # Raw progress
            if T0 is None or not math.isfinite(t_rem):
                p_raw = 0.0
            else:
                p_raw = max(0.0, min(0.999, 1.0 - (t_rem / max(T0, 1e-6))))

            # Smooth, monotonic
            p_smoothed = PROGRESS_EMA_ALPHA * progress_prev + (1.0 - PROGRESS_EMA_ALPHA) * p_raw
            p_final = max(progress_prev, p_smoothed)

            # Tail lock until both done
            if (current_audio_tokens >= N_est) and (C_done >= est_snac_calls_from(N_est)):
                # allow final completion later
                pass
            else:
                p_final = min(p_final, PROGRESS_TAIL_LOCK / 100.0)

            # Throttle pushes: >=1% or >=50ms or forced
            push = force or ( (p_final - progress_prev) >= 0.01 ) or ( (now - last_progress_push_t) >= 0.05 )
            if push:
                p_int = int(min(100, max(0, round(p_final * 100))))
                pbar.update_absolute(p_int)
                progress_prev = p_final
                last_progress_push_t = now

            if debug:
                # occasional progress debug (every ~0.5s)
                if (now - getattr(update_progress, "_last_dbg", 0.0)) >= 0.5:
                    setattr(update_progress, "_last_dbg", now)
                    # print(f"[PROG] p={p_final*100:5.1f}%  "
                    #       f"N={current_audio_tokens}/{N_est}  "
                    #       f"C={C_done}/{est_snac_calls_from(N_est)}  "
                    #       f"r_llm={r_llm:.1f}/s  r_snac={r_sn:.1f}/s  "
                    #       f"t_llm={t_llm:.2f}s  t_snac={t_snac:.2f}s  T0={T0 if T0 else 0:.2f}s")

        # --- LLM streaming loop ------------------------------------------------
        with lms.Client() as client:
            client_enter_time = time.time() - start_time

            t_model_start = time.time()
            model = self._get_model(client, model_key, auto_unload, unload_delay)
            model_resolve_time = time.time() - t_model_start

            t_stream_start = time.time()
            stream = model.complete_stream(prompt, config=config)
            t_stream_ready = time.time()
            stream_create_time = t_stream_ready - t_stream_start

            deadline = time.time() + timeout_seconds
            t0_stream = t_stream_ready
            prev_iter_end = t_stream_ready

            if debug:
                print("--- LLM Stream (with parallel SNAC) ---")
                print(f"  Client enter:   {client_enter_time:.4f}s")
                print(f"  Model resolve:  {model_resolve_time:.4f}s")
                print(f"  Stream create:  {stream_create_time:.4f}s")

            for frag in stream:
                iter_start = time.time()

                # wait time since previous frag processed
                wait_dt = iter_start - prev_iter_end
                total_wait_time += wait_dt
                wait_samples.append(wait_dt)
                if wait_dt > max_wait[0]:
                    max_wait = (wait_dt, llm_frag_count)

                if time.time() > deadline:
                    try:
                        stream.cancel()
                    except:
                        pass
                    raise TimeoutError(f"LLM prediction exceeded {timeout_seconds}s")

                if first_fragment_rel is None:
                    first_fragment_rel = iter_start - t_stream_ready

                proc_start = time.time()

                content = frag.content or ""
                chars_total += len(content)
                llm_frag_count += 1

                # Fast path: count & parse audio tokens only when fragment starts with the marker
                if content and content[0] == '<' and content.startswith('<custom_token_'):
                    # count
                    current_audio_tokens += 1
                    if t0_first_audio_token is None:
                        t0_first_audio_token = proc_start
                    llm_rate.add(1, now=proc_start)

                    # parse numeric id (match only, not search)
                    m = CUSTOM_RE.match(content)
                    if m:
                        n = int(m.group(1))
                        # mapping exactly as in your offline phase
                        mapped = n - 10 - ((total_accepted_tokens % 7) * 4096)
                        if mapped > 0:
                            mapped_tokens.append(mapped)
                            total_accepted_tokens += 1

                            # queue any ready 28-token windows (stride 7)
                            queued = queue_ready_windows()
                            # if debug and queued:
                            #     print(f"[SNAC] queued {queued} window(s); "
                            #           f"mapped_tokens={len(mapped_tokens)}  next_idx={next_snac_index}")

                # update progress (critical path)
                update_progress()

                proc_end = time.time()
                proc_dt = proc_end - proc_start
                total_proc_time += proc_dt
                proc_samples.append(proc_dt)
                if proc_dt > max_proc[0]:
                    max_proc = (proc_dt, llm_frag_count)

                prev_iter_end = proc_end

            loop_end = time.time()

            # result & stats
            try:
                result = stream.result()
                if debug and getattr(result, "stats", None):
                    s = result.stats
                    print(f"[Orpheus-LMS] LLM Stats: TTFT={getattr(s,'time_to_first_token_sec',0.0):.2f}s, "
                          f"Tokens={getattr(s,'predicted_tokens_count',0)}, "
                          f"Stop Reason='{getattr(s,'stop_reason','')}'")
            except Exception:
                result = None

            # If auto_unload immediate
            if model and auto_unload == "True" and unload_delay == 0:
                try:
                    model.unload()
                except:
                    pass

        # --- Flush remaining SNAC windows, then finish worker -----------------
        # Any final windows after the stream ends?
        queued = queue_ready_windows()
        if debug and queued:
            print(f"[SNAC] final queued {queued} window(s) after stream end")

        # Compute expected total windows from actual accepted tokens
        total_snac_calls = 0
        if total_accepted_tokens > 27:
            total_snac_calls = ((total_accepted_tokens - 28) // 7) + 1

        # Wait until all queued work is done; if more windows are still expected (shouldn't be),
        # we queue them (defensive, but no 'fallback' logic paths).
        snac_q.join()

        # Stop worker
        snac_q.put(None)
        snac_q.join()
        worker.join(timeout=2.0)

        # Final progress to 99.5% (if not already), then 100% when both are done.
        update_progress(force=True)
        pbar.update_absolute(int(PROGRESS_TAIL_LOCK))
        # Now both paths are complete by construction
        pbar.update_absolute(100)

        # --- Build audio tensor ------------------------------------------------
        if not collected_pcm:
            # keep exactly the previous behavior: supply short silence if nothing
            silence = np.zeros(int(SAMPLE_RATE / 10), dtype=np.int16).tobytes()
            with pcm_lock:
                collected_pcm.extend(silence)

        pcm_i16 = np.frombuffer(bytes(collected_pcm), dtype=np.int16)
        audio_f32 = (pcm_i16.astype(np.float32) / 32767.0).reshape(1, -1)
        waveform = torch.from_numpy(audio_f32)

        # --- Logs --------------------------------------------------------------
        llm_duration = loop_end - start_time
        snac_duration = 0.0  # approximate via rate or derived from debug timings is noisy; end-to-end below matters most
        total_duration = time.time() - start_time
        secs = len(collected_pcm) / 2 / SAMPLE_RATE

        if debug:
            avg_wait = (total_wait_time / llm_frag_count) if llm_frag_count else 0.0
            avg_proc = (total_proc_time / llm_frag_count) if llm_frag_count else 0.0
            p50_wait = _pct(wait_samples, 50)
            p95_wait = _pct(wait_samples, 95)
            p50_proc = _pct(proc_samples, 50)
            p95_proc = _pct(proc_samples, 95)

            print("--- LLM Stream Granular ---")
            print(f"  TTFF (loop):            {first_fragment_rel or 0.0:.4f}s")
            print(f"  Loop total:             {loop_end - t_stream_ready:.4f}s")
            print(f"    ├─ Wait (blocked):    {total_wait_time:.4f}s   avg={avg_wait:.4f}s  p50={p50_wait:.4f}s  p95={p95_wait:.4f}s  max={max_wait[0]:.4f}s@{max_wait[1]}")
            print(f"    └─ Proc (your code):  {total_proc_time:.4f}s   avg={avg_proc:.4f}s  p50={p50_proc:.4f}s  p95={p95_proc:.4f}s  max={max_proc[0]:.4f}s@{max_proc[1]}")
            print(f"  Fragments:              {llm_frag_count}")
            print(f"  Chars received:         {chars_total}")
            print(f"  Audio-token events:     {current_audio_tokens}")
            print(f"  Accepted tokens:        {total_accepted_tokens}")
            print(f"  SNAC windows total:     {total_snac_calls}")
            print("---------------------------")
            print("--- Performance ---")
            print(f"  LLM Stream (incl. IO):  {llm_duration:.2f}s")
            print(f"  Total Generation Time:  {total_duration:.2f}s")
            print(f"  Final Audio Duration:   {secs:.2f}s")
            print("-------------------")

        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": SAMPLE_RATE}
        return (audio,)

# Comfy registration
NODE_CLASS_MAPPINGS = {"OrpheusTTSLMStudio": OrpheusTTSLMStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"OrpheusTTSLMStudio": "Orpheus TTS (LM Studio)"}
