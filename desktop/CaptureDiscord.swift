import CoreGraphics
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

private let usage = """
Usage: capture-discord OUTPUT.wav [--seconds N]
       capture-discord --chunks NEW_DIRECTORY [--chunk-seconds N] [--seconds N]

Record the Discord desktop app's playback as a mono 48 kHz PCM WAV.
Chunk mode publishes complete WAVs while capture continues. It saves the final partial chunk on stop.
Press Ctrl+C to stop or supply a duration in seconds.
"""

private struct Options {
    let destination: URL
    let seconds: Double?
    let chunkSeconds: Int?
    let parentPipe: Bool

    init(_ arguments: [String]) throws {
        var args = arguments[...]
        let chunked = args.first == "--chunks"
        if chunked { args = args.dropFirst() }
        guard let path = args.popFirst(), !path.hasPrefix("-"), !path.isEmpty,
              chunked || path.lowercased().hasSuffix(".wav") else {
            throw CaptureError(usage)
        }
        destination = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        var duration: Double?
        var chunk: Int?
        var pipe = false
        while let flag = args.popFirst() {
            switch flag {
            case "--seconds":
                guard duration == nil, let value = args.popFirst(), let limit = Double(value),
                      limit.isFinite, limit > 0, limit <= 43_200 else {
                    throw CaptureError("--seconds must be greater than 0 and at most 43200.")
                }
                duration = limit
            case "--chunk-seconds":
                guard chunked, chunk == nil, let value = args.popFirst(), let limit = Int(value),
                      (1...300).contains(limit) else {
                    throw CaptureError("--chunk-seconds must be between 1 and 300 in chunk mode.")
                }
                chunk = limit
            case "--parent-pipe":
                pipe = true
            default:
                throw CaptureError(usage)
            }
        }
        seconds = duration
        chunkSeconds = chunked ? (chunk ?? 20) : nil
        parentPipe = pipe
    }
}

private enum CaptureEnd: Sendable {
    case requested
    case failed(Error)
}

private final class CaptureOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    private let writer: any CaptureWriter
    private let events: AsyncStream<CaptureEnd>.Continuation
    private var initialTimestamp = CMTime.invalid
    private var acceptingSamples = true
    private var writeError: Error?

    init(writer: any CaptureWriter, events: AsyncStream<CaptureEnd>.Continuation) {
        self.writer = writer
        self.events = events
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sample: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, acceptingSamples, sample.isValid, sample.numSamples > 0 else { return }
        do {
            guard let format = sample.formatDescription?.audioStreamBasicDescription,
                  format.mSampleRate == Double(captureSampleRate), format.mChannelsPerFrame == 1,
                  format.mFormatID == kAudioFormatLinearPCM, format.mBitsPerChannel == 32,
                  format.mBytesPerFrame == MemoryLayout<Float>.stride,
                  format.mFormatFlags & kAudioFormatFlagIsBigEndian == 0,
                  format.mFormatFlags & kAudioFormatFlagIsFloat != 0 else {
                throw CaptureError("ScreenCaptureKit returned an unexpected audio format.")
            }
            let timestamp = sample.presentationTimeStamp
            if !initialTimestamp.isNumeric { initialTimestamp = timestamp }
            var frame: Int64?
            if initialTimestamp.isNumeric, timestamp.isNumeric {
                frame = CMTimeConvertScale(
                    timestamp - initialTimestamp, timescale: captureSampleRate, method: .roundHalfAwayFromZero
                ).value
            }
            try sample.withAudioBufferList { buffers, _ in
                guard buffers.count == 1, let pcm = buffers.first,
                      pcm.mDataByteSize >= sample.numSamples * MemoryLayout<Float>.stride,
                      let data = pcm.mData else {
                    throw CaptureError("ScreenCaptureKit returned an incomplete audio buffer.")
                }
                try writer.append(UnsafeBufferPointer(
                    start: data.assumingMemoryBound(to: Float.self), count: sample.numSamples
                ), at: frame)
            }
        } catch {
            acceptingSamples = false
            writeError = error
            events.yield(.failed(error))
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: any Error) {
        events.yield(.failed(error))
    }

    func publish() throws {
        acceptingSamples = false
        if let writeError { throw writeError }
        try writer.publish()
    }
}

@MainActor
private func capture(_ options: Options) async throws {
    guard !FileManager.default.fileExists(atPath: options.destination.path) else {
        throw CaptureError("Output already exists: \(options.destination.path)")
    }
    guard CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess() else {
        throw CaptureError("Grant the requesting app Screen & System Audio Recording access in System Settings > Privacy & Security, then relaunch the app and retry.")
    }
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    let discord = content.applications.filter { $0.bundleIdentifier == "com.hnc.Discord" }
    guard !discord.isEmpty else {
        throw CaptureError("Open the Discord desktop app before starting capture.")
    }
    guard let display = content.displays.first(where: { $0.displayID == CGMainDisplayID() }) ?? content.displays.first else {
        throw CaptureError("ScreenCaptureKit could not find an active display.")
    }
    let filter = SCContentFilter(display: display, including: discord, exceptingWindows: [])
    let configuration = SCStreamConfiguration()
    configuration.capturesAudio = true
    configuration.excludesCurrentProcessAudio = true
    configuration.sampleRate = Int(captureSampleRate)
    configuration.channelCount = 1
    configuration.width = 16
    configuration.height = 16
    configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
    configuration.showsCursor = false

    let (events, continuation) = AsyncStream.makeStream(of: CaptureEnd.self, bufferingPolicy: .bufferingOldest(1))
    let writer: any CaptureWriter
    if let seconds = options.chunkSeconds {
        writer = try ChunkedWAVWriter(directory: options.destination, framesPerChunk: seconds * Int(captureSampleRate))
    } else {
        writer = try WAVWriter(destination: options.destination)
    }
    let output = CaptureOutput(writer: writer, events: continuation)
    let stream = SCStream(filter: filter, configuration: configuration, delegate: output)
    let audioQueue = DispatchQueue(label: "voice-lab.discord-audio")
    try stream.addStreamOutput(output, type: .audio, sampleHandlerQueue: audioQueue)
    // Supply the screen stream output for versions that expect it. Its frames are discarded.
    try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: DispatchQueue(label: "voice-lab.discard-frames"))

    let signals = [SIGINT, SIGTERM].map { signalNumber in
        signal(signalNumber, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: signalNumber)
        source.setEventHandler { continuation.yield(.requested) }
        source.resume()
        return source
    }
    defer {
        signals.forEach { $0.cancel() }
        continuation.finish()
    }

    // The beta owns the write end of this pipe. EOF also stops capture if the UI host is killed.
    let parent = options.parentPipe ? DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO) : nil
    parent?.setEventHandler {
        var byte: UInt8 = 0
        if read(STDIN_FILENO, &byte, 1) == 0 {
            continuation.yield(.requested)
            parent?.cancel()
        }
    }
    parent?.resume()
    defer { parent?.cancel() }

    try await stream.startCapture()
    print("Recording Discord audio. Press Ctrl+C to stop.")
    fflush(stdout)
    let timer = options.seconds.map { seconds in
        let source = DispatchSource.makeTimerSource()
        source.schedule(deadline: .now() + seconds)
        source.setEventHandler { continuation.yield(.requested) }
        source.resume()
        return source
    }
    defer { timer?.cancel() }

    var next = events.makeAsyncIterator()
    let end = await next.next()
    var stopError: Error?
    if case .failed(let error) = end { stopError = error }
    do {
        try await stream.stopCapture()
    } catch {
        if stopError == nil { stopError = error }
    }
    try? stream.removeStreamOutput(output, type: .audio)
    try audioQueue.sync { try output.publish() }
    print("Saved \(options.destination.path)")
    if let stopError { throw CaptureError("Capture stopped early: \(stopError.localizedDescription)") }
}

@main
private enum CaptureDiscord {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--help"] || arguments == ["-h"] {
            print(usage)
            return
        }
        umask(0o077)
        do {
            try await capture(Options(arguments))
        } catch {
            FileHandle.standardError.write(Data("capture-discord: \(error.localizedDescription)\n".utf8))
            exit(EXIT_FAILURE)
        }
    }
}
