"""Temporal-support sensitivity extraction (revision). For every run, recompute the PRIMARY-window feature (mean amplitude, 600 ms ending ~150 ms before the response)
from the 1 kHz data supplied with ds002739 under several operations, to test whether any operation uses samples after the window end.
Variants (all use the same window: decimated samples k in [b-188, b-38), i.e. 1 kHz indices [4b-752, 4b-152), b = tresp // 4):
  V1_boxcar        mean of the raw 1 kHz samples in the window (no decimation filter at all)
  V2_causal_decim  causal (lfilter) version of the decimation anti-alias filter (scipy decimate design: Chebyshev I, order 8, 0.05 dB, 100 Hz), window mean of the decimated samples
  V3_trunc_decim   zero-phase decimation filter applied to data TRUNCATED at the window end (nothing after the window end enters the output)
Proxy experiments for the (unspecified) filtering of the dataset: an additional 4th-order Butterworth 0.5-40 Hz band-pass applied
  P0_zerophase_full  zero-phase over the whole recording (uses future samples)   P1_causal  causal over the whole recording   P2_zerophase_trunc  zero-phase on data truncated at the window end
The reference variant V0 (zero-phase decimation of the whole recording) is already stored in epochs/*.npz.
Usage: python v2/extract_temporal.py   (streams each run from S3, deletes it after use; resumable)"""
import os, sys, threading, queue, time, warnings
import numpy as np, scipy.io as sio, scipy.signal as ss
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import dl
OUT = os.path.join(HERE, "features"); TMP = os.path.join(ROOT, "raw_tmp"); os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
FS = 1000; Q = 4
# decimation filter exactly as scipy.signal.decimate(x, 4) builds it (IIR, n=8, Chebyshev I, rp=0.05, Wn=0.8/q)
SOS_DEC = ss.cheby1(8, 0.05, 0.8 / Q, output="sos")
SOS_BP = ss.butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
VARS = ["V1_boxcar", "V2_causal_decim", "V3_trunc_decim", "P0_zerophase_full", "P1_causal", "P2_zerophase_trunc", "P3_zerophase_inside_part", "P4_zerophase_future_part"]
# exact decomposition of the zero-phase proxy band-pass feature: the window mean of the filtered signal is a linear functional  sum_j g_j x_j  of the input samples (g = box kernel convolved with the filter's effective impulse response);
# P3 = part from samples up to the window end (j < N), P4 = part from samples AFTER the window end (j >= N); P3 + P4 = P0 up to edge effects.
KH = 20000; _imp = np.zeros(2 * KH + 1); _imp[KH] = 1.0; _h = ss.sosfiltfilt(SOS_BP, _imp, padlen=0)
NWIN = 600; _g = np.convolve(np.ones(NWIN) / NWIN, _h)                      # index m <-> input sample offset j = m - KH relative to the window start
K1, K2 = 6000, 6000
G_IN = _g[KH - K1: KH + NWIN]                                                # offsets j = -K1 .. NWIN-1
G_FUT = _g[KH + NWIN: KH + NWIN + K2]                                        # offsets j = NWIN .. NWIN+K2-1
PAD_DEC = 2000     # samples of history kept before the window start for truncated decimation
PAD_BP = 8000      # samples of history for the truncated band-pass proxy

def win_dec(b): return b - 188, b - 38     # decimated-sample window [k0, k1)

def process(sub, r):
    out = f"{OUT}/sub-{sub:02d}_run-{r:02d}.npz"
    if os.path.exists(out): return
    ev, dt = f"{TMP}/ev_{sub}_{r}.mat", f"{TMP}/dat_{sub}_{r}.mat"
    e = sio.loadmat(ev); d = sio.loadmat(dt, struct_as_record=False, squeeze_me=True); s = d["EEGdata"]
    Y = np.asarray(s.Y, dtype=np.float64); assert s.fd == 1000
    exch = np.atleast_1d(d["excludedchannels"]).ravel().astype(int); valid = [c for c in range(1, 65) if c not in set(exch.tolist())]; vi = np.array(valid) - 1
    tresp = e["tresp"].ravel().astype(int); n = len(tresp); RT = e["RT"].ravel().astype(float); T = Y.shape[1]
    F = np.full((n, 64, len(VARS)), np.nan, np.float32)
    Y_causal_dec = ss.sosfilt(SOS_DEC, Y, axis=1)                  # causal anti-alias filter on the whole recording
    Y_bp0 = ss.sosfiltfilt(SOS_BP, Y, axis=1)                      # proxy: zero-phase band-pass, whole recording
    Y_bp1 = ss.sosfilt(SOS_BP, Y, axis=1)                          # proxy: causal band-pass
    for i in range(n):
        if RT[i] < 750: continue
        b = tresp[i] // Q; k0, k1 = win_dec(b); a_ms, e_ms = Q * k0, Q * k1      # window in 1 kHz samples [a_ms, e_ms)
        if a_ms - PAD_BP < 0 or e_ms > T: continue
        ks = np.arange(k0, k1) * Q                                  # 1 kHz indices of the decimated samples used by V0
        F[i, vi, 0] = Y[:, a_ms:e_ms].mean(1)
        F[i, vi, 1] = Y_causal_dec[:, ks].mean(1)
        seg = Y[:, e_ms - PAD_DEC:e_ms]                              # truncated at the window end (exclusive)
        z = ss.sosfiltfilt(SOS_DEC, seg, axis=1); F[i, vi, 2] = z[:, ks - (e_ms - PAD_DEC)].mean(1)
        F[i, vi, 3] = Y_bp0[:, a_ms:e_ms].mean(1)
        F[i, vi, 4] = Y_bp1[:, a_ms:e_ms].mean(1)
        seg = Y[:, e_ms - PAD_BP:e_ms]; z = ss.sosfiltfilt(SOS_BP, seg, axis=1); F[i, vi, 5] = z[:, -(e_ms - a_ms):].mean(1)
        if a_ms - K1 >= 0 and e_ms + K2 <= T:
            F[i, vi, 6] = Y[:, a_ms - K1:e_ms] @ G_IN; F[i, vi, 7] = Y[:, e_ms:e_ms + K2] @ G_FUT
    np.savez_compressed(out, F=F, vars=np.array(VARS), valid_ch=np.array(valid), RT=RT, tresp=tresp, sub=sub, run=r)
    os.remove(ev); os.remove(dt)
    print("done", sub, r, n, int(np.isfinite(F[:, vi[0], 0]).sum()), flush=True)

def fetch(sub, r):
    ev, dt = f"{TMP}/ev_{sub}_{r}.mat", f"{TMP}/dat_{sub}_{r}.mat"; base = f"sub-{sub:02d}/EEG/"
    if not os.path.exists(ev): dl.get(base + f"EEG_events_sub-{sub:02d}_run-{r:02d}.mat", ev, workers=1)
    if not os.path.exists(dt): dl.get(base + f"EEG_data_sub-{sub:02d}_run-{r:02d}.mat", dt)

def main():
    todo = [(s, r) for s in range(1, 25) for r in (1, 2) if not os.path.exists(f"{OUT}/sub-{s:02d}_run-{r:02d}.npz")]
    q = queue.Queue(maxsize=1)
    def producer():
        for s, r in todo:
            for t in range(4):
                try: fetch(s, r); break
                except Exception as ex: print("fetch retry", s, r, repr(ex)[:120], flush=True); time.sleep(10)
            q.put((s, r))
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is None: break
        for t in range(3):
            try: process(*item); break
            except Exception as ex: print("process retry", item, repr(ex)[:200], flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__": main()
