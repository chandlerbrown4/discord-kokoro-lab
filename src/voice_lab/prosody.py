"""Per-utterance delivery measurements for comparing two readings of the same words."""
import math
from pathlib import Path
from statistics import mean

RATE = 8000
ERROR_FIELDS = ("duration_log_error", "pitch_center_error_st", "pitch_span_error_st")


def delivery(path):
    import numpy as np
    from scipy.signal import find_peaks
    from .distill_teacher import _read

    x = _read(Path(path), RATE).astype("float64")
    hop, frame = RATE // 100, RATE // 20
    n = len(x) // hop
    empty = {"speech_seconds": None, "pitch_center_st": None, "pitch_span_st": None}
    if n < 5:
        return empty
    x -= np.mean(x)
    energy = np.sqrt(np.mean(x[:n * hop].reshape(-1, hop)**2, axis=1))
    threshold = max(.002, .07 * float(np.quantile(energy, .9)))
    active = np.flatnonzero(energy >= threshold)
    if len(active) < 5:
        return empty
    start, stop = int(active[0]) * hop, (int(active[-1]) + 1) * hop
    result = dict(empty, speech_seconds=(stop-start) / RATE)
    # Estimate periodicity (normalized square difference) only in voiced frames.
    # Taking the first sufficiently clear peak avoids doubling the pitch period.
    low, high = math.ceil(RATE / 600), int(RATE / 55)
    periods = []
    for pos in range(start, stop-frame+1, hop*2):
        if energy[(pos + frame//2)//hop] < threshold:
            continue
        window = x[pos:pos+frame]
        window = window - window.mean()
        squared = np.r_[0., np.cumsum(window**2)]
        corr = np.correlate(window, window, mode="full")[frame-1:frame + high]
        lags = np.arange(len(corr))
        norm = squared[frame-lags] + squared[-1] - squared[lags]
        clarity = 2*corr / np.maximum(norm, 1e-12)
        peaks = find_peaks(clarity)[0]
        peaks = peaks[(peaks >= low) & (peaks <= high)]
        if not len(peaks):
            continue
        best = np.max(clarity[peaks])
        if best < .78:
            continue
        periods.append(peaks[np.flatnonzero(clarity[peaks] >= .92*best)[0]])
    if len(periods) >= max(8, math.ceil(len(active)*.075)):
        # Semitones make the same proportional shift comparable across speakers.
        pitch = 12*np.log2(RATE / np.asarray(periods) / 55)
        bottom, middle, top = np.quantile(pitch, [.1, .5, .9])
        result.update(pitch_center_st=float(middle), pitch_span_st=float(top-bottom))
    return result


def errors(original, generated):
    ref, test = delivery(original), delivery(generated)
    duration = (abs(math.log(test["speech_seconds"] / ref["speech_seconds"]))
                if test["speech_seconds"] and ref["speech_seconds"] else None)
    values = {"duration_log_error": duration}
    for source, target in (("pitch_center_st", "pitch_center_error_st"), ("pitch_span_st", "pitch_span_error_st")):
        left, right = ref[source], test[source]
        values[target] = abs(left-right) if left is not None and right is not None else None
    return values


def summary(rows):
    result = {}
    for field in ERROR_FIELDS:
        values = [r[field] for r in rows if r.get(field) is not None]
        result[field] = mean(values) if values else None
    return result
