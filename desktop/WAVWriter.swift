import Darwin
import Foundation

let captureSampleRate: Int32 = 48_000

struct CaptureError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

// Access each writer exclusively on the serial audio callback queue until capture stops.
protocol CaptureWriter: AnyObject {
    func append(_ samples: UnsafeBufferPointer<Float>, at frame: Int64?) throws
    func publish() throws
}

final class WAVWriter: CaptureWriter {
    private static let maxFrames = Int64((UInt32.max - 36) / 2)
    private static let silence = Data(count: 16_384)
    private let destination: URL
    private let temporary: URL
    private let handle: FileHandle
    private var samplesWritten: Int64 = 0
    private var hasSound = false
    private let allowSilence: Bool

    init(destination: URL, allowSilence: Bool = false) throws {
        self.destination = destination
        self.allowSilence = allowSilence
        let directory = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        temporary = directory.appendingPathComponent(".capture-\(UUID().uuidString).partial")
        let descriptor = open(temporary.path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard descriptor >= 0 else {
            throw CaptureError("Could not open a temporary WAV in \(directory.path): \(String(cString: strerror(errno)))")
        }
        handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        do {
            try handle.write(contentsOf: Self.header(dataSize: 0))
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    deinit {
        try? handle.close()
        try? FileManager.default.removeItem(at: temporary)
    }

    func append(_ samples: UnsafeBufferPointer<Float>, at frame: Int64?) throws {
        guard !samples.isEmpty else { return }
        let offset = max(0, frame ?? samplesWritten)
        guard offset <= Self.maxFrames, Int64(samples.count) <= Self.maxFrames - offset else {
            throw CaptureError("Recording reached the 4 GB WAV limit. Start a new file for the next part.")
        }
        // Keep the sample timeline when ScreenCaptureKit skips silent buffers.
        let delta = offset - samplesWritten
        if delta > 1 { try appendSilence(delta) }

        let skipped = delta < -1 ? min(samples.count, Int(-delta)) : 0
        var pcm = [Int16]()
        pcm.reserveCapacity(samples.count - skipped)
        for sample in samples.dropFirst(skipped) {
            let amplitude = sample.isFinite ? min(1, max(-1, sample)) : 0
            let value = Int16((amplitude * Float(Int16.max)).rounded())
            hasSound = hasSound || value != 0
            pcm.append(value.littleEndian)
        }
        guard Int64(pcm.count) <= Self.maxFrames - samplesWritten else {
            throw CaptureError("Recording reached the 4 GB WAV limit. Start a new file for the next part.")
        }
        try pcm.withUnsafeBytes { bytes in
            try handle.write(contentsOf: Data(bytes))
        }
        samplesWritten += Int64(pcm.count)
    }

    func appendSilence(_ frames: Int64) throws {
        guard frames >= 0, frames <= Self.maxFrames - samplesWritten else {
            throw CaptureError("Silence exceeds the WAV size limit.")
        }
        var bytes = frames * 2
        while bytes > 0 {
            let count = Int(min(bytes, Int64(Self.silence.count)))
            try handle.write(contentsOf: Self.silence.prefix(count))
            bytes -= Int64(count)
        }
        samplesWritten += frames
    }

    func publish() throws {
        guard hasSound || (allowSilence && samplesWritten > 0) else {
            throw CaptureError("Discord had no audible output. Check macOS capture permission, then try while Discord is playing audio.")
        }
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Self.header(dataSize: UInt32(samplesWritten * 2)))
        try handle.close()
        // A hard link publishes the completed file without clobbering another recording.
        try FileManager.default.linkItem(at: temporary, to: destination)
    }

    private static func header(dataSize: UInt32) -> Data {
        var data = Data("RIFF".utf8)
        data.appendInteger(UInt32(36) + dataSize)
        data.append(contentsOf: "WAVEfmt ".utf8)
        data.appendInteger(UInt32(16))
        data.appendInteger(UInt16(1)) // PCM
        data.appendInteger(UInt16(1)) // mono
        data.appendInteger(UInt32(captureSampleRate))
        data.appendInteger(UInt32(captureSampleRate * 2))
        data.appendInteger(UInt16(2))
        data.appendInteger(UInt16(16))
        data.append(contentsOf: "data".utf8)
        data.appendInteger(dataSize)
        return data
    }
}

/// Keep the ScreenCaptureKit stream open while publishing immutable, frame-aligned WAVs.
final class ChunkedWAVWriter: CaptureWriter {
    private let directory: URL
    private let framesPerChunk: Int
    private var writer: WAVWriter?
    private var nextChunk = 1
    private var position = 0
    private var absoluteFrame: Int64 = 0

    init(directory: URL, framesPerChunk: Int) throws {
        guard framesPerChunk > 0, Int64(framesPerChunk) <= Int64((UInt32.max - 36) / 2) else {
            throw CaptureError("Choose a positive chunk length within the WAV size limit.")
        }
        self.directory = directory
        self.framesPerChunk = framesPerChunk
        try FileManager.default.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(directory.path, 0o700) == 0 else {
            throw CaptureError("Choose a new capture directory: \(String(cString: strerror(errno)))")
        }
    }

    private func currentWriter() throws -> WAVWriter {
        if let writer { return writer }
        let path = directory.appendingPathComponent(String(format: "chunk-%06d.wav", nextChunk))
        let opened = try WAVWriter(destination: path, allowSilence: true)
        writer = opened
        return opened
    }

    private func advance(_ count: Int) throws {
        position += count
        absoluteFrame += Int64(count)
        if position == framesPerChunk { try publish() }
    }

    func append(_ samples: UnsafeBufferPointer<Float>, at frame: Int64?) throws {
        guard let base = samples.baseAddress, !samples.isEmpty else { return }
        let delta = max(0, frame ?? absoluteFrame) - absoluteFrame
        var gap = delta > 1 ? delta : 0
        while gap > 0 {
            let count = min(gap, Int64(framesPerChunk - position))
            try currentWriter().appendSilence(count)
            try advance(Int(count))
            gap -= count
        }
        var index = delta < -1 ? Int(min(Int64(samples.count), -delta)) : 0
        while index < samples.count {
            let count = min(samples.count - index, framesPerChunk - position)
            try currentWriter().append(UnsafeBufferPointer(start: base + index, count: count), at: nil)
            try advance(count)
            index += count
        }
    }

    func publish() throws {
        guard let writer else { return }
        try writer.publish()
        self.writer = nil
        position = 0
        nextChunk += 1
    }
}

private extension Data {
    mutating func appendInteger<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }
}
