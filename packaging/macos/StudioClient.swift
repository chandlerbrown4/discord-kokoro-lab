import Foundation

private struct StudioError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

// A local studio request never needs to follow a redirect off its loopback origin.
private final class LocalSession: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
}

final class StudioClient {
    typealias Object = [String: Any]
    typealias Reply = (Result<Object, Error>) -> Void

    private let baseURL: URL
    private let session: URLSession
    private let origin: String

    init(baseURL: URL) {
        self.baseURL = baseURL
        origin = "http://127.0.0.1:" + String(baseURL.port!)
        let config = URLSessionConfiguration.ephemeral
        config.urlCache = nil
        config.urlCredentialStorage = nil
        config.httpCookieStorage = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = 20
        config.connectionProxyDictionary = ["HTTPEnable": 0, "HTTPSEnable": 0, "SOCKSEnable": 0]
        session = URLSession(configuration: config, delegate: LocalSession(), delegateQueue: nil)
    }

    func stop() { session.invalidateAndCancel() }

    private func request(_ path: String) -> URLRequest {
        // Construct paths below the launcher's validated origin, including for audio previews.
        URLRequest(url: baseURL.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))))
    }

    private static func validate(_ response: URLResponse?, _ data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { throw StudioError("The local studio did not respond.") }
        guard (200..<300).contains(http.statusCode) else {
            if let data, let json = try? JSONSerialization.jsonObject(with: data) as? Object,
               let reason = json["error"] as? String { throw StudioError(reason) }
            throw StudioError("The local studio returned HTTP \(http.statusCode).")
        }
    }

    private static func receive(_ completion: @escaping Reply) -> @Sendable (Data?, URLResponse?, Error?) -> Void {
        { data, response, error in
            let result = Result<Object, Error> {
                if let error { throw error }
                try validate(response, data)
                guard let data, let json = try JSONSerialization.jsonObject(with: data) as? Object else {
                    throw StudioError("The local studio returned an unreadable response.")
                }
                return json
            }
            DispatchQueue.main.async { completion(result) }
        }
    }

    func state(_ completion: @escaping Reply) {
        session.dataTask(with: request("api/state"), completionHandler: Self.receive(completion)).resume()
    }

    func devices(_ completion: @escaping Reply) {
        session.dataTask(with: request("api/devices"), completionHandler: Self.receive(completion)).resume()
    }

    func action(_ path: String, body: Object, completion: @escaping Reply) {
        do {
            var req = request(path)
            req.httpMethod = "POST"
            req.setValue(origin, forHTTPHeaderField: "Origin")
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            session.dataTask(with: req, completionHandler: Self.receive(completion)).resume()
        } catch { DispatchQueue.main.async { completion(.failure(error)) } }
    }

    func upload(_ file: URL, mime: String, voiceCall: String? = nil, completion: @escaping Reply) {
        do {
            let info = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard info.isRegularFile == true, let bytes = info.fileSize, (1...16 * 1024 * 1024).contains(bytes) else {
                throw StudioError("Choose a nonempty audio file up to 16 MiB.")
            }
            var req = request(voiceCall == nil ? "api/mic/file" : "api/voice/file")
            if let voiceCall { req.setValue(voiceCall, forHTTPHeaderField: "X-Voice-Call") }
            req.httpMethod = "POST"
            req.setValue(origin, forHTTPHeaderField: "Origin")
            req.setValue(mime, forHTTPHeaderField: "Content-Type")
            req.setValue(String(bytes), forHTTPHeaderField: "Content-Length")
            session.uploadTask(with: req, fromFile: file, completionHandler: Self.receive(completion)).resume()
        } catch { DispatchQueue.main.async { completion(.failure(error)) } }
    }

    func download(_ path: String, profile: Bool = false, completion: @escaping (Result<URL, Error>) -> Void) {
        session.downloadTask(with: request(path)) { file, response, error in
            let result = Result<URL, Error> {
                if let error { throw error }
                guard let file else { throw StudioError("The clip could not be downloaded.") }
                if (response as? HTTPURLResponse)?.statusCode != 200 {
                    let handle = try FileHandle(forReadingFrom: file)
                    defer { try? handle.close() }
                    try Self.validate(response, try handle.read(upToCount: 4096))
                }
                guard response?.mimeType == (profile ? "application/octet-stream" : "audio/wav") else { throw StudioError("The file is unavailable.") }
                let target = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + (profile ? ".pt" : ".wav"))
                try FileManager.default.moveItem(at: file, to: target)
                return target
            }
            DispatchQueue.main.async { completion(result) }
        }.resume()
    }
}
