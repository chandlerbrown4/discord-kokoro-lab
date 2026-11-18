import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import OpusScript from 'opusscript';
import { RecordingSession, SAMPLE_RATE } from '../src/session.mjs';

const ALICE = '100000000000000001';
const BOB = '100000000000000002';

function decoder() {
  return new OpusScript(SAMPLE_RATE, 2, OpusScript.Application.VOIP);
}

function harness(t, createDecoder = decoder) {
  const directory = mkdtempSync(join(tmpdir(), 'discord-recorder-'));
  const streams = new Map();
  const voice = new Set([ALICE]);
  const session = new RecordingSession({
    dataDirectory: directory,
    voiceChannelId: '100000000000000007',
    subscribe: (id) => {
      const stream = new PassThrough({ objectMode: true });
      streams.set(id, stream);
      return stream;
    },
    createDecoder,
    isInChannel: (id) => voice.has(id),
    onFatal: (error) => assert.fail(error),
    onWarning: (_id, error) => assert.fail(error),
  });
  t.after(() => {
    session.finish('test_cleanup');
    rmSync(directory, { recursive: true, force: true });
  });
  return { session, streams, voice };
}

function readWav(path, frames) {
  const wav = readFileSync(path);
  assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
  assert.equal(wav.readUInt32LE(4), wav.length - 8);
  assert.equal(wav.toString('ascii', 8, 16), 'WAVEfmt ');
  assert.equal(wav.readUInt16LE(20), 1);
  assert.equal(wav.readUInt16LE(22), 1);
  assert.equal(wav.readUInt32LE(24), SAMPLE_RATE);
  assert.equal(wav.readUInt16LE(34), 16);
  assert.equal(wav.readUInt32LE(40), frames * 2);
  assert.equal(wav.length, frames * 2 + 44);
  assert.equal(statSync(path).mode & 0o077, 0);
  return wav.subarray(44);
}

test('records actual Opus packets to separate WAVs only while opted in and present', (t) => {
  const { session, streams, voice } = harness(t);
  const raw = join(session.directory, 'raw');
  const manifestPath = join(session.directory, 'session.json');
  const encoder = decoder();
  t.after(() => encoder.delete());
  const pcm = Buffer.alloc(960 * 4);
  for (let i = 0; i < 960; i += 1) {
    const sample = Math.round(12_000 * Math.sin(2 * Math.PI * 440 * i / SAMPLE_RATE));
    pcm.writeInt16LE(sample, i * 4);
    pcm.writeInt16LE(sample, i * 4 + 2);
  }
  const packet = encoder.encode(pcm, 960);

  assert.equal(streams.size, 0);
  assert.equal(session.join(BOB, 'Bob'), false);
  assert.equal(session.join(ALICE, 'Alice'), true);
  const alice = streams.get(ALICE);
  alice.write(packet);
  voice.add(BOB);
  assert.equal(streams.has(BOB), false);
  assert.equal(session.join(BOB, 'Bob'), true);
  const bob = streams.get(BOB);
  bob.write(packet);
  assert.equal(existsSync(manifestPath), false);

  session.leave(ALICE);
  alice.emit('data', packet); // Model a previously queued packet after withdrawal.
  assert.equal(alice.destroyed, true);
  voice.delete(BOB);
  bob.write(packet); // Membership check must handle departure even before a callback.
  assert.equal(bob.destroyed, true);
  session.finish('operator_stop');

  for (const speaker of [ALICE, BOB]) {
    assert.deepEqual(readdirSync(join(raw, speaker)), ['000001.wav']);
    assert(readWav(join(raw, speaker, '000001.wav'), 960).some((byte) => byte !== 0));
  }
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  assert.equal(manifest.source, 'discord');
  assert.equal(manifest.session_id, session.id);
  assert.equal(manifest.stop_reason, 'operator_stop');
  assert.deepEqual(manifest.participants.map(({ id, consented, clips }) => [id, consented, clips]), [
    [ALICE, true, 1], [BOB, true, 1],
  ]);
  assert.equal(statSync(manifestPath).mode & 0o077, 0);
  assert.equal(session.join(ALICE, 'Alice'), false);
});

test('caps clips at 30 seconds and flushes the remainder on stop', (t) => {
  const stereo = Buffer.alloc(960 * 4);
  stereo.writeInt16LE(1_000, 0);
  stereo.writeInt16LE(3_000, 2);
  const { session, streams } = harness(t, () => ({ decode: () => stereo, delete() {} }));
  assert.equal(session.join(ALICE, 'Alice'), true);
  const source = streams.get(ALICE);
  for (let i = 0; i < 1501; i += 1) source.write(Buffer.alloc(1));
  session.finish('sigint');

  const raw = join(session.directory, 'raw', ALICE);
  assert.deepEqual(readdirSync(raw), ['000001.wav', '000002.wav']);
  assert.equal(readWav(join(raw, '000001.wav'), 30 * SAMPLE_RATE).readInt16LE(0), 2_000);
  assert.equal(readWav(join(raw, '000002.wav'), 960).readInt16LE(0), 2_000);
  assert.equal(session.clips, 2);
});
