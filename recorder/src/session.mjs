import { randomUUID } from 'node:crypto';
import { mkdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const SAMPLE_RATE = 48_000;
const IDLE_MS = 1_000;
const MAX_WAV_BYTES = 30 * SAMPLE_RATE * 2;
const RETRY_MS = 500;

function publishFile(path, contents) {
  const partial = `${path}.${randomUUID()}.part`;
  try {
    writeFileSync(partial, contents, { mode: 0o600, flag: 'wx' });
    renameSync(partial, path);
  } finally {
    rmSync(partial, { force: true });
  }
}

function monoPcm(stereo) {
  if (!stereo.length || stereo.length % 4 !== 0) {
    throw new Error('Decoder returned invalid stereo PCM');
  }
  const mono = Buffer.allocUnsafe(stereo.length / 2);
  for (let i = 0; i < stereo.length; i += 4) {
    const sample = (stereo.readInt16LE(i) + stereo.readInt16LE(i + 2)) / 2;
    mono.writeInt16LE(Math.trunc(sample), i / 2);
  }
  return mono;
}

function wavHeader(size) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + size, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // Mono
  header.writeUInt32LE(SAMPLE_RATE, 24);
  header.writeUInt32LE(SAMPLE_RATE * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(size, 40);
  return header;
}

class SpeakerTrack {
  stream;
  decoder;
  retry;
  idle;
  stopped = false;
  chunks = [];
  size = 0;
  lastWarning = -Infinity;

  constructor({ save, subscribe, createDecoder, isPresent, onDeparture, onFatal, onWarning }) {
    Object.assign(this, { save, subscribe, createDecoder, isPresent, onDeparture, onFatal, onWarning });
  }

  start() {
    this.stream = this.subscribe();
    this.stream.once('close', () => this.onClose());
    // A quick opt-out and opt-in can occur before the receiver processes close.
    if (this.stream.destroyed) return;
    this.decoder = this.createDecoder();
    this.stream.on('data', (packet) => this.onPacket(packet));
    this.stream.on('error', (error) => this.warn(error));
  }

  onPacket(packet) {
    if (this.stopped) return;
    if (!this.isPresent()) {
      this.onDeparture();
      return;
    }

    let audio;
    try {
      audio = monoPcm(this.decoder.decode(packet));
    } catch (error) {
      this.warn(error);
      this.finishTurn();
      return;
    }

    try {
      // Split at a packet boundary to avoid cutting mid-frame.
      if (this.size + audio.length > MAX_WAV_BYTES) this.flush();
      this.chunks.push(audio);
      this.size += audio.length;
      if (this.idle) this.idle.refresh();
      else this.idle = setTimeout(() => this.finishTurn(), IDLE_MS);
    } catch (error) {
      this.onFatal(error);
    }
  }

  onClose() {
    if (this.stopped) return;
    this.decoder?.delete();
    this.decoder = undefined;
    this.finishTurn();
    if (this.stopped) return;
    // The receiver closes an individual stream on a decryption error. Allow a
    // new subscription once it has removed the old stream from its map.
    this.retry = setTimeout(() => {
      if (!this.isPresent()) {
        this.onDeparture();
        return;
      }
      try {
        this.start();
      } catch (error) {
        this.onFatal(error);
      }
    }, RETRY_MS);
  }

  warn(error) {
    const now = performance.now();
    if (now - this.lastWarning < 10_000) return;
    this.lastWarning = now;
    this.onWarning(error);
  }

  finishTurn() {
    try {
      this.flush();
    } catch (error) {
      this.onFatal(error);
    }
  }

  flush() {
    clearTimeout(this.idle);
    this.idle = undefined;
    if (!this.size) return;

    const wav = Buffer.concat([wavHeader(this.size), ...this.chunks], 44 + this.size);
    this.chunks = [];
    this.size = 0;
    this.save(wav);
  }

  stop() {
    this.stopped = true;
    clearTimeout(this.retry);
    this.stream?.destroy();
    this.decoder?.delete();
    this.decoder = undefined;
    this.flush();
  }
}

export class RecordingSession {
  participants = new Map();
  tracks = new Map();
  startedAt = new Date().toISOString();
  id = `${this.startedAt.replace(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}`;
  finished = false;
  clips = 0;

  constructor({ dataDirectory, voiceChannelId, subscribe, createDecoder, isInChannel, onFatal, onWarning }) {
    Object.assign(this, { voiceChannelId, subscribe, createDecoder, isInChannel, onFatal, onWarning });
    const recordings = join(dataDirectory, 'recordings');
    mkdirSync(recordings, { recursive: true, mode: 0o700 });
    this.directory = join(recordings, this.id);
    mkdirSync(this.directory, { mode: 0o700 });
    mkdirSync(join(this.directory, 'raw'), { mode: 0o700 });
  }

  join(id, displayName) {
    if (!/^\d{17,20}$/.test(id)) throw new Error('Invalid Discord user ID');
    if (this.finished || !this.isInChannel(id)) return false;
    if (this.tracks.has(id)) return true;

    const participant = this.participants.get(id) ?? {
      id, display_name: displayName, consented: true, consented_at: [], stopped_at: [], clips: 0,
    };
    const directory = join(this.directory, 'raw', id);
    const track = new SpeakerTrack({
      save: (wav) => {
        const sequence = participant.clips + 1;
        const name = `${String(sequence).padStart(6, '0')}.wav`;
        mkdirSync(directory, { recursive: true, mode: 0o700 });
        publishFile(join(directory, name), wav);
        participant.clips = sequence;
        this.clips += 1;
      },
      subscribe: () => this.subscribe(id),
      createDecoder: this.createDecoder,
      isPresent: () => !this.finished && this.isInChannel(id),
      onDeparture: () => {
        try {
          this.leave(id);
        } catch (error) {
          this.onFatal(error);
        }
      },
      onFatal: this.onFatal,
      onWarning: (error) => this.onWarning(id, error),
    });
    this.tracks.set(id, track);
    this.participants.set(id, participant);
    participant.consented_at.push(new Date().toISOString());
    try {
      track.start();
    } catch (error) {
      this.leave(id);
      throw error;
    }
    return true;
  }

  leave(id) {
    const track = this.tracks.get(id);
    if (!track) return false;
    this.tracks.delete(id);
    this.participants.get(id).stopped_at.push(new Date().toISOString());
    track.stop();
    return true;
  }

  finish(reason) {
    if (this.finished) return;
    this.finished = true;
    const errors = [];
    for (const id of this.tracks.keys()) {
      try {
        this.leave(id);
      } catch (error) {
        errors.push(error);
      }
    }
    const manifest = {
      schema_version: 1,
      source: 'discord',
      session_id: this.id,
      voice_channel_id: this.voiceChannelId,
      started_at: this.startedAt,
      ended_at: new Date().toISOString(),
      stop_reason: reason,
      participants: [...this.participants.values()],
    };
    publishFile(join(this.directory, 'session.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    if (errors.length) throw new AggregateError(errors, 'Could not save all audio clips');
  }
}
