import AppKit
import Darwin
import Foundation

final class VoiceLabApp: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let files = FileManager.default
    private let support = URL(fileURLWithPath: ProcessInfo.processInfo.environment["VOICE_LAB_APP_SUPPORT"]
        ?? NSHomeDirectory() + "/Library/Application Support/Voice Lab Beta", isDirectory: true)
    private let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 500, height: 205),
                                  styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
    private let title = NSTextField(labelWithString: "Starting Voice Lab")
    private let detail = NSTextField(wrappingLabelWithString: "Preparing the local studio…")
    private let spinner = NSProgressIndicator()
    private let action = NSButton()
    private var worker: Process?
    private var logHandle: FileHandle?
    private var timer: Timer?
    private var lock: Int32 = -1
    private var stopping = false
    private var studio: URL?
    private var studioWindow: StudioWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let appMenuItem = NSMenuItem()
        menu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Hide Voice Lab", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Voice Lab", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenuItem.submenu = appMenu
        let windows = NSMenu(title: "Window")
        windows.addItem(withTitle: "Show Call Studio", action: #selector(openStudio), keyEquivalent: "0").target = self
        windows.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windows.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windows.addItem(.separator())
        windows.addItem(withTitle: "View Setup Log", action: #selector(openLog), keyEquivalent: "").target = self
        let windowItem = NSMenuItem()
        menu.addItem(windowItem)
        windowItem.submenu = windows
        NSApp.mainMenu = menu
        NSApp.windowsMenu = windows
        window.title = "Voice Lab Beta"
        window.delegate = self
        window.isReleasedWhenClosed = false
        title.font = .systemFont(ofSize: 23, weight: .semibold)
        detail.textColor = .secondaryLabelColor
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.startAnimation(nil)
        action.title = "Open Studio"
        action.bezelStyle = .rounded
        action.isEnabled = false
        action.target = self
        action.action = #selector(openStudio)
        let log = NSButton(title: "View Setup Log", target: self, action: #selector(openLog))
        log.bezelStyle = .rounded
        let buttons = NSStackView(views: [action, log])
        let header = NSStackView(views: [spinner, title])
        header.spacing = 10
        let stack = NSStackView(views: [header, detail, buttons])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 15
        stack.translatesAutoresizingMaskIntoConstraints = false
        let content = window.contentView!
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            stack.centerYAnchor.constraint(equalTo: content.centerYAnchor),
            detail.widthAnchor.constraint(lessThanOrEqualToConstant: 450)
        ])
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        do {
            try files.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            lock = Darwin.open(support.appendingPathComponent("launcher.lock").path, O_CREAT | O_RDWR, 0o600)
            guard lock >= 0, flock(lock, LOCK_EX | LOCK_NB) == 0 else {
                title.stringValue = "Voice Lab is already open"
                detail.stringValue = "Return to the running Voice Lab app to open the studio."
                spinner.stopAnimation(nil)
                return
            }
            try launch()
        } catch { showError(error.localizedDescription) }
    }

    private func launch() throws {
        guard let resources = Bundle.main.resourceURL else { throw CocoaError(.fileNoSuchFile) }
        studio = nil
        studioWindow?.stop()
        studioWindow = nil
        title.stringValue = "Starting Voice Lab"
        detail.stringValue = "Preparing the local studio…"
        action.title = "Open Studio"
        action.isEnabled = false
        action.action = #selector(openStudio)
        spinner.startAnimation(nil)
        for name in ["studio-url", "launch-status"] {
            let path = support.appendingPathComponent(name)
            if files.fileExists(atPath: path.path) { try files.removeItem(at: path) }
        }
        let logfile = support.appendingPathComponent("setup.log")
        try Data().write(to: logfile, options: .atomic)
        let handle = try FileHandle(forWritingTo: logfile)
        logHandle = handle
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [resources.appendingPathComponent("bootstrap.sh").path]
        process.standardOutput = handle
        process.standardError = handle
        process.standardInput = FileHandle.nullDevice
        process.terminationHandler = { [weak self] finished in
            DispatchQueue.main.async { self?.finished(status: finished.terminationStatus) }
        }
        try process.run()
        worker = process
        timer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { [weak self] _ in self?.refresh() }
    }

    private func refresh() {
        if studio != nil || stopping { return }
        if let raw = try? String(contentsOf: support.appendingPathComponent("studio-url"), encoding: .utf8),
           let url = URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines)),
           url.scheme == "http", url.host == "127.0.0.1", url.port != nil, url.path == "/",
           url.user == nil, url.password == nil, url.query == nil, url.fragment == nil {
            studio = url
            spinner.stopAnimation(nil)
            timer?.invalidate()
            let startHidden = NSApp.isHidden || window.isMiniaturized
            window.orderOut(nil)
            studioWindow = StudioWindow(baseURL: url)
            if !startHidden { studioWindow?.show() }
        } else if let raw = try? String(contentsOf: support.appendingPathComponent("launch-status"), encoding: .utf8) {
            detail.stringValue = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private func finished(status: Int32) {
        timer?.invalidate()
        worker = nil
        try? logHandle?.close()
        logHandle = nil
        if stopping { NSApp.reply(toApplicationShouldTerminate: true) }
        else if status == 0 { NSApp.terminate(nil) }
        else { showError("Check your connection and available disk space. View the setup log for the error, then retry.") }
    }

    private func showError(_ reason: String) {
        studioWindow?.stop()
        studioWindow = nil
        studio = nil
        spinner.stopAnimation(nil)
        title.stringValue = "Voice Lab could not start"
        detail.stringValue = reason
        action.title = "Retry"
        action.action = #selector(retryLaunch)
        action.isEnabled = lock >= 0
        window.makeKeyAndOrderFront(nil)
    }

    @objc private func openStudio() {
        if let studioWindow { studioWindow.show() }
        else { window.makeKeyAndOrderFront(nil) }
    }
    @objc private func openLog() {
        NSWorkspace.shared.open(support.appendingPathComponent("setup.log"))
    }
    @objc private func retryLaunch() {
        do { try launch() } catch { showError(error.localizedDescription) }
    }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openStudio()
        return true
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        NSApp.terminate(nil)
        return false
    }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        studioWindow?.stop()
        guard let process = worker, process.isRunning else { return .terminateNow }
        if !stopping {
            stopping = true
            title.stringValue = "Stopping Voice Lab"
            detail.stringValue = "Finishing the current recording and closing the local studio…"
            action.isEnabled = false
            process.terminate()
        }
        return .terminateLater
    }
}

@main
struct VoiceLabMain {
    static func main() {
        let app = NSApplication.shared
        let delegate = VoiceLabApp()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        withExtendedLifetime(delegate) { app.run() }
    }
}
