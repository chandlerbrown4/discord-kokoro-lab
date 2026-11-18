"""Opt-in macOS text-voice acceptance with two stock voices, a fresh library and no audio devices.

Pass a fixture directory with one subdirectory per voice. Each contains inputs.json
with training_clips and heldout_clips lists of PCM WAV paths (relative to inputs.json or absolute).
Only training clips are uploaded. The harness downloads and verifies the WAVs served to
the player, then runs independent ASR and speaker comparisons using the teacher runtime.
"""
import argparse
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import wave
import zipfile


def fixtures(directory):
    result = {}
    for manifest in sorted(directory.glob("*/inputs.json")):
        data = json.loads(manifest.read_text())
        result[manifest.parent.name] = {key: [str((manifest.parent / p).resolve(strict=True)) for p in data[key]]
                                        for key in ("training_clips", "heldout_clips")}
    if len(result) != 2 or any(not row[key] for row in result.values() for key in row):
        raise ValueError("Supply two stock voice fixture directories, each with training clips and separate holdouts.")
    hashes = []
    for row in result.values():
        for key in ("training_clips", "heldout_clips"):
            hashes.extend(hashlib.sha256(Path(p).read_bytes()).digest() for p in row[key])
    if len(set(hashes)) != len(hashes):
        raise ValueError("Stock fixtures contain duplicated training or held-out audio.")
    return result


def combined(paths, ffmpeg):
    rate = 24000
    buffer = BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        for path in paths:
            pcm = subprocess.check_output([str(ffmpeg), "-v", "error", "-nostdin", "-i", path,
                                           "-t", "121", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"])
            out.writeframes(b"\0\0" * (rate//4))
            out.writeframes(pcm)
            if out.getnframes() > 120*rate:
                raise ValueError("The stock import must fit Studio's 120-second limit.")
    return buffer.getvalue()


class Client:
    def __init__(self, url, proc, *, fixture_capture=False):
        self.fixture_capture = fixture_capture
        self.url, self.proc = url.rstrip("/"), proc
        self.client = build_opener(ProxyHandler({}))

    def request(self, path, data=None, headers=None):
        headers = headers or {}
        if data is not None:
            headers = {"Origin": self.url, "Content-Type": "application/json", **headers}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        request = Request(self.url + path, data=data, headers=headers)
        try:
            with self.client.open(request, timeout=90) as resp:
                if path.startswith("/audio/"):
                    expected = "application/octet-stream" if "/profile/" in path else "audio/wav"
                    if resp.headers.get_content_type() != expected:
                        raise ValueError(f"Studio served the wrong content type for {path}.")
                    return resp.read()
                return json.load(resp)
        except HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc

    def wait(self, predicate, timeout=900):
        until = time.monotonic() + timeout
        last = None
        while time.monotonic() < until:
            if self.proc.poll() is not None:
                raise RuntimeError("Studio exited. Inspect studio.log.")
            state = self.request("/api/state")
            capture_states = ("idle", "starting", "running", "stopped") if self.fixture_capture else ("idle",)
            if state["capture"]["state"] not in capture_states or state["route"]["state"] != "idle":
                raise RuntimeError("Unexpected capture or routing during stock-only test.")
            if state["setup"]["state"] == "error" or state.get("analysis_errors"):
                raise RuntimeError(str(state["setup"]) + " " + str(state.get("analysis_errors")))
            for row in state["text_voices"].values():
                if row.get("state") == "error":
                    raise RuntimeError(row["message"])
            detail = (state["setup"]["message"], [(r["id"], r["clip_count"]) for r in state["voices"]["speakers"]],
                      [r.get("message") for r in state["text_voices"].values()])
            if detail != last:
                print(detail, flush=True)
                last = detail
            if predicate(state):
                return state
            time.sleep(.35)
        raise TimeoutError(str(last))


@contextmanager
def studio(args, env):
    url_file = args.work_dir / "url"
    url_file.unlink(missing_ok=True)
    env = dict(env, VOICE_LAB_BETA_URL_FILE=str(url_file))
    cmd = [str(args.python), "-m", "voice_lab", "beta", "--port", "0", "--device", "cpu",
           "--data-dir", str(args.library), "--model-dir", str(args.model_dir)]
    with (args.work_dir / "studio.log").open("ab") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic()+30
            while not url_file.is_file():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Could not start the loopback studio. Inspect studio.log.")
                time.sleep(.1)
            client = Client(url_file.read_text().strip(), proc, fixture_capture=bool(args.additional))
            client.wait(lambda s: s["setup"]["state"] == "ready")
            yield client
            client.request("/api/quit", {"call": client.request("/api/state")["call"]})
            if proc.wait(timeout=30):
                raise RuntimeError("Studio exited unsuccessfully.")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def save_wav(client, url, path):
    data = client.request(url)
    with wave.open(BytesIO(data)) as f:
        if f.getsampwidth() != 2 or f.getnframes() < f.getframerate()//2:
            raise ValueError(f"Studio player received invalid audio from {url}.")
        duration = f.getnframes()/f.getframerate()
    path.write_bytes(data)
    return duration


def speak(client, call, sid, text, destination):
    start = time.monotonic()
    client.request("/api/text/speak", {"call": call, "speaker": sid, "text": text})
    state = client.wait(lambda s: s["text_voices"][sid].get("rendered") == text)
    duration = save_wav(client, f"/audio/{call}/tts/{sid}", destination)
    return {"text": text, "wav": str(destination), "audio_seconds": duration,
            "inference_seconds": state["text_voices"][sid]["seconds"], "request_seconds": time.monotonic()-start}


def evaluate(args, env):
    # Run ASR and WeSpeaker in the optional worker environment, never in Studio.
    if not args.teacher_python:
        raise ValueError("Pass --teacher-python for independent evaluation.")
    subprocess.run([str(args.teacher_python), str(Path(__file__).resolve()), "--score", str(args.work_dir),
                    str(args.model_dir)], env=env, check=True)


def score(root, models):
    from voice_lab.diarize import EMBEDDING
    from voice_lab.distill_teacher import QualityChecker, _audio_digest, word_error_rate

    summary = json.loads((root/"result.json").read_text())
    for sid, upgrade in summary.get("upgrades", {}).items():
        library = root/"library"
        call = (library/"selected-call.txt").read_text().strip()
        challenger = upgrade["alternate"] if upgrade["decision"] == "kept" else upgrade["selected"]
        build = library/"calls"/call/"text-voices"/sid/challenger
        teacher = json.loads((build/"teacher-decision.json").read_text())
        enrolled = set(teacher.get("fit_sha256", [])) | set(teacher.get("development_sha256", []))
        release = {_audio_digest(path) for path in report_holdouts(build, library, call)}
        if release & enrolled:
            raise ValueError("A release evaluation recording appeared in Qwen fitting or selection.")
    checker = QualityChecker(models/EMBEDDING)
    genuine = {name: [checker.embedding(path) for path in row["heldout_clips"]]
               for name, row in summary["fixtures"].items()}
    failures = []
    for name, row in summary["voices"].items():
        other = next(k for k in genuine if k != name)
        for item in row["generated"]:
            vec = checker.embedding(item["wav"])
            item["target_cosine"] = statistics.mean(float(vec @ e) for e in genuine[name])
            item["other_cosine"] = statistics.mean(float(vec @ e) for e in genuine[other])
            item["heard"] = checker.transcribe(item["wav"])
            item["wer"] = word_error_rate(item["text"], item["heard"])
            if item["target_cosine"] <= item["other_cosine"] or item["wer"] > .2:
                failures.append((name, item["text"], item["target_cosine"], item["other_cosine"], item["wer"]))
    (root/"result.json").write_text(json.dumps(summary, indent=2)+"\n")
    if failures:
        raise ValueError(f"Stock speech failed cross-voice discrimination or intelligibility checks: {failures}")


def recheck(args, env, samples):
    """Restore saved profiles, then exercise a fresh build through the updated release API."""
    previous = json.loads((args.previous_run/"result.json").read_text())
    if previous["fixtures"] != samples:
        raise ValueError("Use the same stock fixtures and genuine holdouts as the previous test.")
    summary = {"fixtures": samples, "voices": {}}
    with studio(args, env) as client:
        restored = client.request("/api/state")
        call = restored["call"]
        for name, old in previous["voices"].items():
            sid = old["speaker"]
            status = restored["text_voices"][sid]
            if status.get("available") != old["evaluation"]["available"]:
                raise ValueError(f"Updated Studio could not restore stock voice {name}.")
            profile = args.library/"calls"/call/"text-voices"/sid/status["available"]
            before = hashlib.sha256((profile/"student"/"profile.pt").read_bytes()).hexdigest()
            downloaded = client.request(f"/audio/{call}/profile/{sid}")
            if hashlib.sha256(downloaded).hexdigest() != before:
                raise ValueError("The exported profile differs from the selected speaker’s saved model.")
            (args.work_dir/f"exported-{sid}.pt").write_bytes(downloaded)
            save_wav(client, f"/audio/{call}/review/{sid}", args.work_dir/f"restored-review-{sid}.wav")
            generated = []
            for i, text in enumerate(("The green mailbox is beside the corner store.",
                                      "Were you there when the curtains opened?")):
                generated.append(speak(client, call, sid, text, args.work_dir/f"updated-{sid}-{i}.wav"))
            if before != hashlib.sha256((profile/"student"/"profile.pt").read_bytes()).hexdigest():
                raise RuntimeError("Rendering altered a committed profile.")
            summary["voices"][name] = {"speaker": sid, "evaluation": status, "generated": generated,
                                        "saved_profile": str(profile), "profile_sha256": before}
        # A fresh call forces the release to create its own relative manifests,
        # collect genuine holdouts, and run every build stage through the Studio API.
        name, fixture = next(iter(samples.items()))
        client.request("/api/new", {"call": call})
        new_call = client.request("/api/state")["call"]
        client.request("/api/voice/file", combined(fixture["training_clips"], args.python.parent/"ffmpeg"),
                       {"X-Voice-Call": new_call, "Content-Type": "audio/wav"})
        state = client.wait(lambda s: any(v["text_eligible"] for v in s["voices"]["speakers"]))
        sid = state["voices"]["speakers"][0]["id"]
        client.request("/api/text/build", {"call": new_call, "speaker": sid, "mode": "preview"})
        state = client.wait(lambda s: s["text_voices"].get(sid, {}).get("state") == "ready", timeout=args.timeout)
        evaluation = state["text_voices"][sid]
        if not evaluation["passes_quality"]:
            raise RuntimeError("The new release failed its genuine held-out quality check.")
        run = args.library/"calls"/new_call/"text-voices"/sid/evaluation["available"]
        manifest = Path(json.loads((run/"manifest-path.json").read_text()))
        held = json.loads((run/"review-lines.json").read_text())
        relative = [manifest] + [Path(row["original"]) for row in held]
        if any(path.is_absolute() or not (run/path).is_file() for path in relative):
            raise RuntimeError("Expected working, library-relative evidence and manifest paths.")
        data = summary["voices"][name]
        data["refit_evaluation"] = json.loads((run/"evaluation.json").read_text())
        data["generated"].append(speak(client, new_call, sid, "We found the silver key underneath a branch.",
                                       args.work_dir/"fresh-preview.wav"))
        # A detached worker must terminate even while its input pipe remains open.
        command = ["/bin/bash", env["VOICE_LAB_TEXT_RUNTIME"], "student", "-m", "voice_lab.distill_jobs",
                   "render", str(run), str(args.model_dir)]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              env=dict(env, VOICE_LAB_JOB_PARENT="-1")) as worker:
            try:
                if worker.wait(timeout=15) != -signal.SIGTERM:
                    raise RuntimeError("An orphaned speech worker did not terminate with the expected signal.")
            finally:
                if worker.poll() is None:
                    worker.terminate()
    summary["orphan_worker_exited"] = True
    (args.work_dir/"result.json").write_text(json.dumps(summary, indent=2)+"\n")
    evaluate(args, env)


def fixture_capture(args, samples, env):
    """Publish only explicit stock WAVs using the native helper's finalized-chunk contract."""
    batches = {name: sorted((args.additional/name).glob("*.wav")) for name in samples}
    if any(not paths for paths in batches.values()):
        raise ValueError("Provide fresh non-empty WAV folders matching both stock voice names.")
    seen = {hashlib.sha256(Path(p).read_bytes()).digest()
            for fixture in samples.values() for paths in fixture.values() for p in paths}
    ordered = []
    for index in range(max(map(len, batches.values()))):
        for paths in batches.values():
            if index < len(paths):
                path = paths[index]
                digest = hashlib.sha256(path.read_bytes()).digest()
                if digest in seen:
                    raise ValueError(f"Training, held-out and additional WAVs must be disjoint: {path}")
                seen.add(digest)
                ordered.append(str(path.resolve()))
    schedule = args.work_dir/"capture-inputs.json"
    schedule.write_text(json.dumps(ordered))
    helper = args.work_dir/"stock-capture"
    helper.write_text("""#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path
directory = Path(sys.argv[sys.argv.index("--chunks") + 1])
directory.mkdir()
print("Recording Discord audio.", flush=True)
for i, source in enumerate(json.loads(Path(os.environ["STOCK_CAPTURE_INPUTS"]).read_text())):
    part = directory / "writing.part"
    shutil.copyfile(source, part)
    part.rename(directory / f"chunk-{i:06}.wav")
sys.stdin.buffer.read()
""")
    helper.chmod(0o700)
    return dict(env, VOICE_LAB_CAPTURE_TOOL=str(helper), STOCK_CAPTURE_INPUTS=str(schedule)), len(ordered)


def speaker(state, sid):
    return next(v for v in state["voices"]["speakers"] if v["id"] == sid)


def profile_hash(client, call, sid):
    return hashlib.sha256(client.request(f"/audio/{call}/profile/{sid}")).hexdigest()


def report_holdouts(directory, library, call):
    # A migrated profile may name its original library absolutely. Always compare
    # the PCM hashes in this isolated run, so the previous test's files cannot mask a bad move.
    rows = json.loads((directory/"evaluation.json").read_text())["lines"]
    result = []
    for row in rows:
        path = Path(row["original"])
        if path.is_absolute():
            index = next(i for i in range(len(path.parts) - 1) if path.parts[i:i+2] == ("calls", call))
            path = library.joinpath(*path.parts[index:])
        else:
            path = (directory/path).resolve()
        if not path.is_relative_to(library):
            raise ValueError("An evaluation line points outside the test library.")
        result.append(path)
    return result


def inspect_decision(args, client, state, sid, old, before, output):
    call, row = state["call"], state["text_voices"][sid]
    decision = row["last_decision"]
    winner, alternate = row["available"], row["alternate"]
    if winner == alternate or (decision == "kept" and winner != old) or (decision == "promoted" and alternate != old):
        raise RuntimeError(f"Unexpected winner after comparison: {row}")
    root = args.library/"calls"/call/"text-voices"/sid
    originals = [report_holdouts(root/run, args.library, call) for run in (winner, alternate)]
    hashes = [[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] for paths in originals]
    if hashes[0] != hashes[1] or len(set(hashes[0])) < 2:
        raise RuntimeError("The incumbent and candidate were evaluated on different or insufficient recordings.")
    after = profile_hash(client, call, sid)
    if (after == before) != (decision == "kept"):
        raise RuntimeError("The exported bytes disagree with the automatic selection.")
    for role in ["review", "alternate-review"]:
        save_wav(client, f"/audio/{call}/{role}/{sid}", args.work_dir/f"voice-{sid}-{role}.wav")
    metrics = {run: json.loads((root/run/"evaluation.json").read_text())
               for run in (winner, alternate)}
    candidate = alternate if decision == "kept" else winner
    candidate_dir = root/candidate
    manifest = candidate_dir/json.loads((candidate_dir/"manifest-path.json").read_text())
    pairs = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(pairs) != 128 or {r["split"] for r in pairs} != {"train", "validation"}:
        raise RuntimeError("The automatic update did not finish the full teacher/student recipe.")
    teacher = json.loads((candidate_dir/"teacher-decision.json").read_text())
    fit = set(teacher.get("fit_sha256", []))
    dev = set(teacher.get("development_sha256", []))
    if fit & dev:
        raise RuntimeError("Teacher fitting and selection shared recordings.")
    if decision == "promoted" and metrics[old]["passes_quality"] and not metrics[winner]["passes_quality"]:
        raise RuntimeError("An update replaced a passing incumbent with a failing voice.")
    output[sid] = {"decision": decision, "reason": row["decision_reason"], "incumbent": old,
                   "selected": winner, "alternate": alternate, "before_sha256": before,
                   "after_sha256": after, "holdout_sha256": hashes[0], "teacher": teacher["selected"],
                   "teacher_reason": teacher.get("reason"), "teacher_fit_clips": len(fit),
                   "teacher_development_clips": len(dev), "corpus_lines": len(pairs),
                   "evaluations": {run: {k: data[k] for k in ("mean_cosine", "mean_word_error_rate", "passes_quality")}
                                   for run, data in metrics.items()}}


def check_background_upgrades(args, env, samples):
    previous = json.loads((args.previous_run/"result.json").read_text())
    if previous["fixtures"] != samples:
        raise ValueError("Use the original stock fixtures corresponding to the saved library.")
    env, chunks = fixture_capture(args, samples, env)
    ids = [previous["voices"][name]["speaker"] for name in samples]
    first, second = ids
    decisions = {}
    with studio(args, env) as client:
        state = client.request("/api/state")
        call = state["call"]
        old = {sid: state["text_voices"][sid]["available"] for sid in ids}
        before = {sid: profile_hash(client, call, sid) for sid in ids}
        original_counts = {sid: speaker(state, sid)["clip_count"] for sid in ids}
        for sid in ids:
            if not speaker(state, sid)["auto_update"] or speaker(state, sid)["new_training_clips"]:
                raise RuntimeError(f"The old profile did not migrate with the correct evidence baseline: {sid}")

        def set_updates(sid, enabled):
            client.request("/api/text/updates", {"call": call, "speaker": sid, "enabled": enabled})
        set_updates(second, False)
        client.request("/api/attach", {"call": call})
        def building(state, sid):
            job = state["text_voices"][sid]
            return job.get("state") == "building" and job.get("automatic")
        state = client.wait(lambda s: s["capture"]["state"] == "running" and building(s, first), args.timeout)
        if state["text_voices"][first]["available"] != old[first] or profile_hash(client, call, first) != before[first]:
            raise RuntimeError("A candidate replaced the active voice before evaluation.")
        foreground = speak(client, call, first, "There is room for one more chair by the window.",
                           args.work_dir/"foreground.wav")
        client.wait(lambda s: building(s, first), args.timeout)
        state = client.wait(lambda s: s["capture"]["state"] == "running" and
                            s["text_voices"][first].get("last_decision") in ("kept", "promoted"), args.timeout)
        inspect_decision(args, client, state, first, old[first], before[first], decisions)
        state = client.wait(lambda s: not s["backlog"] and not s["analysis"])
        second_voice = speaker(state, second)
        if (second_voice["auto_update"] or second_voice["new_training_clips"] < second_voice["update_after_clips"] or
                state["text_voices"][second].get("last_decision") or profile_hash(client, call, second) != before[second]):
            raise RuntimeError("Disabling upgrades lost evidence or changed the disabled profile.")
        collected = {sid: speaker(state, sid)["clip_count"] for sid in ids}
        client.request("/api/detach", {"call": call})
        # Leave the second voice disabled when quitting. Both the decision and
        # the backlog must survive restart in the relocated library.
    with studio(args, env) as client:
        state = client.request("/api/state")
        if (speaker(state, second)["auto_update"] or
                state["text_voices"][first]["available"] != decisions[first]["selected"]):
            raise RuntimeError("A selected voice or the speaker preference changed on reopen.")
        client.request("/api/text/updates", {"call": call, "speaker": second, "enabled": True})
        client.wait(lambda s: building(s, second), args.timeout)
        client.request("/api/new", {"call": call})
        unselected = client.request("/api/state")
        if unselected["call"] == call or (unselected.get("text_job") or {}).get("call") != call:
            raise RuntimeError("Switching to a new call cancelled an automatic upgrade.")
        other_call = unselected["call"]
    with studio(args, env) as client:
        def hidden_job(s):
            job = s.get("text_job") or {}
            return s["call"] == other_call and job.get("automatic") and job.get("call") == call and job["speaker"] == second
        client.wait(hidden_job, args.timeout)
        client.wait(lambda s: s["call"] == other_call and not s.get("text_job"), args.timeout)
        client.request("/api/select", {"call": other_call, "id": call})
        state = client.wait(lambda s: s["text_voices"][second].get("last_decision") in ("kept", "promoted"), args.timeout)
        inspect_decision(args, client, state, second, old[second], before[second], decisions)
        pending_before = {sid: speaker(state, sid)["new_training_clips"] for sid in ids}
        client.request("/api/attach", {"call": call})
        def replayed(s):
            route = s["capture"]
            return (route["state"] == "running" and len(list(Path(route["directory"]).glob("chunk-*.wav"))) == chunks
                    and not s["backlog"] and not s["analysis"])
        state = client.wait(replayed)
        if any(speaker(state, sid)["clip_count"] != collected[sid]
               or speaker(state, sid)["new_training_clips"] != pending_before[sid]
               or state["text_voices"][sid].get("state") == "building" for sid in ids):
            raise RuntimeError("Replaying identical audio scheduled work or inflated the evidence.")
        client.request("/api/detach", {"call": call})
        summary = {"fixtures": samples, "voices": {}, "upgrades": decisions, "chunks": chunks,
                   "original_counts": original_counts, "after_counts": collected, "replay_changed_counts": False,
                   "resumed_while_another_call_selected": True,
                   "foreground_request_seconds": foreground["request_seconds"]}
        for name, sid in zip(samples, ids):
            wav = speak(client, call, sid, "The blue boat was already tied up beside the dock.",
                        args.work_dir/f"selected-{sid}.wav")
            summary["voices"][name] = {"speaker": sid, "generated": [wav]}
    (args.work_dir/"result.json").write_text(json.dumps(summary, indent=2) + "\n")
    evaluate(args, env)
    print(f"Autonomous two-speaker upgrade acceptance passed: {args.work_dir/'result.json'}", flush=True)


def run(args):
    samples = fixtures(args.fixtures)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if any(args.work_dir.iterdir()):
        raise ValueError("Choose an empty scratch directory for the isolated Studio library.")
    args.library = args.work_dir / "library"
    if args.previous_run:
        shutil.copytree(args.previous_run/"library", args.library)
    base = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    if args.archive:
        with zipfile.ZipFile(args.archive) as zip_file:
            for item in zip_file.namelist():
                if Path(item).is_absolute() or ".." in Path(item).parts:
                    raise ValueError("Unsafe bundle member.")
        extract = args.work_dir / "extracted"
        subprocess.run(["/usr/bin/ditto", "-x", "-k", str(args.archive), str(extract)], check=True)
        resources = extract/"Voice Lab Beta.app"/"Contents"/"Resources"
        subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(resources.parents[1])], check=True)
        env.update(PYTHONPATH=str(resources/"core"), VOICE_LAB_TEXT_RUNTIME=str(resources/"text-runtime.sh"))
    else:
        env.setdefault("VOICE_LAB_TEXT_RUNTIME", str(base/"packaging/macos/text-runtime.sh"))
        env["PYTHONPATH"] = os.pathsep.join([str(base/"src"), env.get("PYTHONPATH", "")])
    env["PATH"] = os.pathsep.join([str(args.python.parent), env["PATH"]])
    if args.additional:
        if not args.previous_run:
            raise ValueError("Supply a previous two-speaker acceptance run together with --additional.")
        check_background_upgrades(args, env, samples)
        return
    if args.previous_run:
        recheck(args, env, samples)
        print(f"Saved profiles and new API preview checked: {args.work_dir/'result.json'}")
        return
    summary = {"fixtures": samples, "voices": {}}
    lines = ["I left a warm blanket on the rocking chair.", "Do we have time to stop and take a picture?",
             "Someone left an extra ticket beside the radio."]
    with studio(args, env) as client:
        call = None
        known = set()
        for name, fixture in samples.items():
            client.request("/api/voice/file", combined(fixture["training_clips"], args.python.parent/"ffmpeg"),
                           {"X-Voice-Call": call or "new", "Content-Type": "audio/wav"})
            state = client.wait(lambda s: s["voices"].get("processed_chunks", 0) >= len(known)+1 and not s["backlog"])
            current = [v["id"] for v in state["voices"]["speakers"] if v["id"] not in known and v["text_eligible"]]
            if len(current) != 1:
                raise ValueError(f"Expected one eligible stock voice for {name}; supply 40–90s of clear speech. "
                                 f"Studio returned {state['voices']['speakers']}")
            sid = current[0]
            known.add(sid)
            call = state["call"]
            summary["voices"][name] = {"speaker": sid, "generated": []}
        for name, row in summary["voices"].items():
            sid = row["speaker"]
            client.request("/api/text/build", {"call": call, "speaker": sid, "mode": "preview"})
            state = client.wait(lambda s: s["text_voices"].get(sid, {}).get("state") == "ready", timeout=args.timeout)
            row["evaluation"] = state["text_voices"][sid]
            if not row["evaluation"].get("passes_quality"):
                raise ValueError(f"Studio's comparison with separate original speech failed for {name}.")
            save_wav(client, f"/audio/{call}/review/{sid}", args.work_dir/f"review-{sid}.wav")
            for i, line in enumerate(lines[:2]):
                row["generated"].append(speak(client, call, sid, line, args.work_dir/f"novel-{sid}-{i}.wav"))
    with studio(args, env) as client:
        state = client.request("/api/state")
        if state["call"] != call:
            raise RuntimeError("Reopening Studio did not restore the selected call.")
        for row in summary["voices"].values():
            sid = row["speaker"]
            if state["text_voices"][sid]["available"] != row["evaluation"]["available"]:
                raise RuntimeError("The trained profile changed across relaunch.")
            row["generated"].append(speak(client, call, sid, lines[2], args.work_dir/f"reopened-{sid}.wav"))
    (args.work_dir/"result.json").write_text(json.dumps(summary, indent=2)+"\n")
    evaluate(args, env)
    print(f"Two-voice text acceptance passed. Report: {args.work_dir/'result.json'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--score":
        score(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--fixtures", type=Path, required=True)
        parser.add_argument("--work-dir", type=Path, required=True)
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--python", type=Path, required=True, help="Installed primary Studio runtime")
        parser.add_argument("--teacher-python", type=Path, required=True, help="Installed optional teacher for QA")
        parser.add_argument("--archive", type=Path, help="Verify and run this extracted Mac ZIP instead of source")
        parser.add_argument("--previous-run", type=Path, help="Prior successful smoke directory for an upgrade/retrain check")
        parser.add_argument("--additional", type=Path, help="Test autonomous upgrades using fresh stock WAVs in one folder per fixture voice")
        parser.add_argument("--timeout", type=int, default=3600, help="Maximum seconds per preview build")
        run(parser.parse_args())
