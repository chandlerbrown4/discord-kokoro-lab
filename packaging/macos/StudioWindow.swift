import AppKit
import UniformTypeIdentifiers
import UserNotifications

private typealias Fields = [String: Any]

private struct Speaker: Equatable {
    let id: String
    let name: String
    let seconds: Int
    let clips: Int
    let requiredSeconds: Int
    let requiredClips: Int
    let autoUpdate: Bool
    let newTrainingClips: Int
    let updateAfterClips: Int
    let textEligible: Bool
    let status: String

    init?(_ fields: Fields) {
        guard let id = fields["id"] as? String, let name = fields["name"] as? String else { return nil }
        self.id = id
        self.name = name
        seconds = Int(fields["usable_seconds"] as? Double ?? 0)
        clips = fields["clip_count"] as? Int ?? 0
        requiredSeconds = fields["required_seconds"] as? Int ?? 30
        requiredClips = fields["required_clips"] as? Int ?? 6
        autoUpdate = fields["auto_update"] as? Bool ?? false
        newTrainingClips = fields["new_training_clips"] as? Int ?? 0
        updateAfterClips = fields["update_after_clips"] as? Int ?? 6
        textEligible = fields["text_eligible"] as? Bool ?? false
        status = fields["model_status"] as? String ?? "collecting"
    }

    var ready: Bool { status == "optimized" || status == "reference_only" }
    var stage: String {
        switch status {
        case "optimized": return "Fitted voice ready"
        case "reference_only": return "Reference voice ready"
        case "ready_to_fit": return "Preparing voice"
        case "needs_more_audio": return "More speech needed"
        default: return "Collecting speech"
        }
    }
}

private extension NSTextField {
    func display(_ text: String) { if stringValue != text { stringValue = text } }
}

private func note(_ text: String = "") -> NSTextField {
    let field = NSTextField(wrappingLabelWithString: text)
    field.textColor = .secondaryLabelColor
    field.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
    field.setContentCompressionResistancePriority(.required, for: .vertical)
    return field
}

private func heading(_ text: String) -> NSTextField {
    let field = NSTextField(labelWithString: text)
    field.font = .systemFont(ofSize: 15, weight: .semibold)
    return field
}

private func stack(_ views: [NSView], vertical: Bool = false, gap: CGFloat = 8) -> NSStackView {
    let view = NSStackView(views: views)
    view.orientation = vertical ? .vertical : .horizontal
    view.alignment = vertical ? .leading : .centerY
    view.spacing = gap
    return view
}

private final class StudioDocument: NSView {
    override var isFlipped: Bool { true }
}

private final class SpeakerCell: NSTableCellView {
    private let name = NSTextField(labelWithString: "")
    private let status = note()

    override init(frame: NSRect) {
        super.init(frame: frame)
        name.font = .systemFont(ofSize: 13, weight: .medium)
        let content = stack([name, status], vertical: true, gap: 2)
        content.translatesAutoresizingMaskIntoConstraints = false
        addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 5),
            content.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -3),
            content.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
        textField = name
    }

    required init?(coder: NSCoder) { nil }

    func configure(_ speaker: Speaker) {
        name.display(speaker.name)
        status.display("\(speaker.stage) · \(speaker.seconds) s")
        setAccessibilityLabel("\(speaker.name), \(speaker.stage), \(speaker.seconds) seconds, \(speaker.clips) clips")
    }

    override var backgroundStyle: NSView.BackgroundStyle {
        didSet {
            name.textColor = backgroundStyle == .emphasized ? .alternateSelectedControlTextColor : .labelColor
            status.textColor = backgroundStyle == .emphasized ? .alternateSelectedControlTextColor : .secondaryLabelColor
        }
    }
}

final class StudioWindow: NSObject, NSWindowDelegate, NSTableViewDataSource, NSTableViewDelegate, NSSoundDelegate {
    private let client: StudioClient
    private let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 850, height: 590),
                                  styleMask: [.titled, .closable, .resizable, .miniaturizable],
                                  backing: .buffered, defer: false)
    private let calls = NSPopUpButton()
    private let microphones = NSPopUpButton()
    private let cables = NSPopUpButton()
    private let table = NSTableView()
    private let captureInfo = note("Connecting to the local studio…")
    private let analysis = note("Detected speakers appear here as the call progresses.")
    private let selectedName = heading("Select a speaker")
    private let enrollment = note()
    private let micInfo = note()
    private let outputInfo = note()
    private let routeInfo = note()
    private let modelInfo = note()
    private let textInfo = note()
    private let voiceLine = NSTextField()
    private let buildSize = NSPopUpButton()
    private lazy var importVoice = button("Import recording…", #selector(importFile))
    private lazy var buildVoice = button("Build text voice", #selector(buildTextVoice))
    private lazy var autoUpdates = NSButton(checkboxWithTitle: "Improve automatically with new speech",
                                            target: self, action: #selector(toggleUpdates))
    private let updateInfo = note()
    private lazy var listenAlternate = button("Hear alternate version", #selector(playAlternate))
    private lazy var useLatest = button("Build with latest clips", #selector(rebuildTextVoice))
    private lazy var pauseBuild = button("Pause", #selector(pauseTextVoice))
    private lazy var reviewVoice = button("Compare original and generated", #selector(playReview))
    private lazy var speakLine = button("Generate and play", #selector(generateLine))
    private lazy var saveVoice = button("Export voice…", #selector(exportVoice))
    private lazy var saveLine = button("Save WAV…", #selector(exportLine))
    private lazy var replayLine = button("Replay line", #selector(playLine))
    private var pendingLine: String?

    private let feedback = note()
    private let progress = NSProgressIndicator()

    private lazy var attach = button("Attach to Discord", #selector(toggleCapture))
    private lazy var newCall = button("New call", #selector(createCall))
    private lazy var notify = NSButton(checkboxWithTitle: "Notify when ready", target: self, action: #selector(toggleNotify))
    private lazy var retry = button("Retry", #selector(retrySetup))
    private lazy var listen = button("Hear caller", #selector(playCaller))
    private lazy var compare = button("Compare with my mic", #selector(compareMic))
    private lazy var replay = button("Replay comparison", #selector(replayComparison))
    private lazy var stopAudio = button("Stop playback", #selector(cancelPlayback))
    private lazy var record = button("Record 6 seconds", #selector(recordMic))
    private lazy var chooseFile = button("Choose mic file…", #selector(pickFile))
    private lazy var refreshDevices = button("Refresh devices", #selector(discoverDevices))
    private lazy var installer = button("Open BlackHole installer", #selector(installCable))
    private lazy var check = button("Check devices", #selector(checkRoute))
    private lazy var route = button("Use selected voice", #selector(routeVoice))
    private lazy var normal = button("Use my normal voice", #selector(routeMic))
    private lazy var stopRoute = button("Stop output", #selector(disconnect))

    private var snapshot: Fields?
    private var speakers: [Speaker] = []
    private var callID: String?
    private var callIDs: [String] = []
    private var selectedID: String?
    private var seenReady = Set<String>()
    private var pendingPreview: String?
    private var playback: NSSound?
    private var audioFile: URL?
    private var audioGeneration = 0
    private var pollTimer: Timer?
    private var polling = false
    private var refreshAfterPoll = false
    private var controlRevision = 0
    private var busy = false
    private var updatingSelection = false
    private var stopped = false
    private var closed = true
    private var discovering = false
    private var connectionError = false
    private var notifications = UserDefaults.standard.bool(forKey: "NotifyWhenVoiceReady")
    private var isVisible: Bool { !closed && window.occlusionState.contains(.visible) && !window.isMiniaturized }
    private var selected: Speaker? { speakers.first { $0.id == selectedID } }
    private var routeFields: Fields { snapshot?["route"] as? Fields ?? [:] }
    private var captureFields: Fields { snapshot?["capture"] as? Fields ?? [:] }

    init(baseURL: URL) {
        client = StudioClient(baseURL: baseURL)
        super.init()
        makeWindow()
    }

    private func button(_ title: String, _ selector: Selector) -> NSButton {
        let control = NSButton(title: title, target: self, action: selector)
        control.bezelStyle = .rounded
        return control
    }

    private func makeWindow() {
        window.title = "Voice Lab Beta"
        window.delegate = self
        window.isReleasedWhenClosed = false
        window.contentMinSize = NSSize(width: 820, height: 560)
        window.center()
        window.setFrameAutosaveName("VoiceLabCallStudio")
        calls.target = self
        calls.action = #selector(selectCall)
        calls.setAccessibilityLabel("Saved calls")
        microphones.setAccessibilityLabel("Physical microphone")
        cables.setAccessibilityLabel("Virtual cable output")
        for menu in [microphones, cables] {
            menu.target = self
            menu.action = #selector(deviceChanged)
            menu.widthAnchor.constraint(equalToConstant: 225).isActive = true
            menu.addItem(withTitle: menu === microphones ? "Select a microphone" : "Select a virtual cable")
        }
        calls.widthAnchor.constraint(equalToConstant: 185).isActive = true
        calls.addItem(withTitle: "Current call")
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        let column = NSTableColumn(identifier: .init("speakers"))
        column.resizingMask = .autoresizingMask
        table.addTableColumn(column)
        table.headerView = nil
        table.style = .inset
        table.rowHeight = 44
        table.allowsEmptySelection = false
        table.columnAutoresizingStyle = .uniformColumnAutoresizingStyle
        table.delegate = self
        table.dataSource = self
        table.setAccessibilityLabel("Call speakers")
        scroll.documentView = table

        let side = NSView()
        let sideHeading = heading("Call speakers")
        let sideBottom = stack([analysis, notify], vertical: true, gap: 10)
        for v in [sideHeading, scroll, sideBottom] as [NSView] {
            v.translatesAutoresizingMaskIntoConstraints = false
            side.addSubview(v)
        }
        analysis.widthAnchor.constraint(equalTo: side.widthAnchor).isActive = true
        NSLayoutConstraint.activate([
            side.widthAnchor.constraint(equalToConstant: 211),
            sideHeading.topAnchor.constraint(equalTo: side.topAnchor),
            sideHeading.leadingAnchor.constraint(equalTo: side.leadingAnchor),
            scroll.topAnchor.constraint(equalTo: sideHeading.bottomAnchor, constant: 9),
            scroll.leadingAnchor.constraint(equalTo: side.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: side.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: sideBottom.topAnchor, constant: -8),
            sideBottom.leadingAnchor.constraint(equalTo: side.leadingAnchor),
            sideBottom.bottomAnchor.constraint(equalTo: side.bottomAnchor)
        ])
        notify.state = notifications ? .on : .off
        progress.isIndeterminate = false
        progress.style = .bar
        progress.controlSize = .small
        progress.minValue = 0
        progress.maxValue = 1
        progress.widthAnchor.constraint(equalToConstant: 150).isActive = true
        progress.setAccessibilityLabel("Speaker enrollment progress")
        let audition = stack([selectedName, progress, enrollment, stack([listen, compare]),
                              stack([replay, stopAudio])], vertical: true, gap: 5)
        let mic = stack([heading("Microphone"), stack([microphones, refreshDevices]),
                         stack([record, chooseFile]), micInfo], vertical: true, gap: 5)
        let output = stack([heading("Discord output"), stack([cables, installer]), outputInfo,
                            stack([check, route]), stack([normal, stopRoute]), routeInfo], vertical: true, gap: 5)
        buildSize.addItems(withTitles: ["Quick preview", "Full build"])
        voiceLine.placeholderString = "Enter an English line"
        voiceLine.setAccessibilityLabel("Text to speak")
        voiceLine.target = self
        voiceLine.action = #selector(generateLine)
        let typing = stack([heading("Text voice"), stack([buildSize, buildVoice, pauseBuild]), useLatest, autoUpdates, updateInfo, textInfo,
                            stack([reviewVoice, saveVoice]), listenAlternate, voiceLine, stack([speakLine, replayLine, saveLine])], vertical: true, gap: 5)
        let right = stack([audition, mic, output, typing], vertical: true, gap: 18)
        voiceLine.widthAnchor.constraint(equalTo: right.widthAnchor).isActive = true
        for label in [enrollment, micInfo, outputInfo, routeInfo, textInfo, updateInfo] {
            label.widthAnchor.constraint(equalTo: right.widthAnchor).isActive = true
        }
        let rightScroll = NSScrollView()
        rightScroll.drawsBackground = false
        rightScroll.hasVerticalScroller = true
        rightScroll.autohidesScrollers = true
        let doc = StudioDocument()
        rightScroll.documentView = doc
        doc.translatesAutoresizingMaskIntoConstraints = false
        right.translatesAutoresizingMaskIntoConstraints = false
        doc.addSubview(right)
        NSLayoutConstraint.activate([
            doc.leadingAnchor.constraint(equalTo: rightScroll.contentView.leadingAnchor),
            doc.topAnchor.constraint(equalTo: rightScroll.contentView.topAnchor),
            doc.widthAnchor.constraint(equalTo: rightScroll.contentView.widthAnchor),
            right.leadingAnchor.constraint(equalTo: doc.leadingAnchor),
            right.topAnchor.constraint(equalTo: doc.topAnchor),
            right.trailingAnchor.constraint(equalTo: doc.trailingAnchor, constant: -4),
            right.bottomAnchor.constraint(equalTo: doc.bottomAnchor)
        ])
        let divider = NSBox()
        divider.boxType = .separator
        let bar = stack([attach, calls, newCall, importVoice])
        let top = stack([bar, captureInfo], vertical: true, gap: 6)
        let bottom = stack([stack([modelInfo, retry]), feedback], vertical: true, gap: 4)
        let content = window.contentView!
        for view in [top, side, divider, rightScroll, bottom] as [NSView] {
            view.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(view)
        }
        NSLayoutConstraint.activate([
            top.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            top.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            top.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            side.leadingAnchor.constraint(equalTo: top.leadingAnchor),
            side.topAnchor.constraint(equalTo: top.bottomAnchor, constant: 20),
            side.bottomAnchor.constraint(equalTo: bottom.topAnchor, constant: -14),
            divider.leadingAnchor.constraint(equalTo: side.trailingAnchor, constant: 16),
            divider.widthAnchor.constraint(equalToConstant: 1),
            divider.topAnchor.constraint(equalTo: side.topAnchor),
            divider.bottomAnchor.constraint(equalTo: side.bottomAnchor),
            rightScroll.leadingAnchor.constraint(equalTo: divider.trailingAnchor, constant: 20),
            rightScroll.trailingAnchor.constraint(equalTo: top.trailingAnchor),
            rightScroll.topAnchor.constraint(equalTo: side.topAnchor),
            rightScroll.bottomAnchor.constraint(equalTo: side.bottomAnchor),
            bottom.leadingAnchor.constraint(equalTo: top.leadingAnchor),
            bottom.trailingAnchor.constraint(equalTo: top.trailingAnchor),
            bottom.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -10),
            feedback.widthAnchor.constraint(equalTo: bottom.widthAnchor)
        ])
        attach.keyEquivalent = ""
        draw()
    }

    func show() {
        guard !stopped else { return }
        closed = false
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        draw()
        refresh()
        if !discovering { discoverDevices() }
    }

    func stop() {
        stopped = true
        pollTimer?.invalidate()
        pollTimer = nil
        endPlayback()
        window.orderOut(nil)
        client.stop()
    }

    func windowWillClose(_ notification: Notification) {
        closed = true
        pendingPreview = nil
        endPlayback()
        schedule()
    }

    func windowDidChangeOcclusionState(_ notification: Notification) {
        guard !stopped else { return }
        if isVisible { draw(); refresh() } else { schedule() }
    }

    private func active(_ child: Fields) -> Bool {
        ["starting", "running"].contains(child["state"] as? String ?? "")
    }

    private var textFields: Fields {
        guard let selectedID else { return [:] }
        return (snapshot?["text_voices"] as? [String: Fields])?[selectedID] ?? [:]
    }

    private func schedule() {
        pollTimer?.invalidate()
        pollTimer = nil
        guard !stopped else { return }
        let awaiting = snapshot?["previewing"] as? Bool == true || snapshot?["calibrating"] as? Bool == true ||
            (snapshot?["setup"] as? Fields)?["state"] as? String == "loading" || pendingPreview != nil || pendingLine != nil || snapshot?["text_job"] is Fields
        let processing = active(captureFields) || active(routeFields) ||
            !(snapshot?["analysis"] as? String ?? "").isEmpty || snapshot?["installing_driver"] as? Bool == true
        guard isVisible || notifications else { return }
        let delay: TimeInterval = isVisible ? (awaiting ? 1 : processing ? 2 : 5) : 15
        let timer = Timer(timeInterval: delay, repeats: false) { [weak self] _ in self?.refresh() }
        timer.tolerance = delay * 0.2
        RunLoop.main.add(timer, forMode: .common)
        pollTimer = timer
    }

    private func refresh() {
        guard !stopped else { return }
        if polling { refreshAfterPoll = true; return }
        pollTimer?.invalidate()
        pollTimer = nil
        polling = true
        let revision = controlRevision
        client.state { [weak self] result in
            guard let self, !self.stopped else { return }
            self.polling = false
            if revision != self.controlRevision {
                self.refreshAfterPoll = false
                self.refresh()
                return
            }
            switch result {
            case .success(let state):
                let recovered = self.connectionError
                self.connectionError = false
                if recovered { self.message("") }
                self.update(state)
                if recovered { self.draw() }
            case .failure(let error):
                self.connectionError = true
                self.message(error.localizedDescription, error: true)
                self.draw()
            }
            if self.refreshAfterPoll {
                self.refreshAfterPoll = false
                self.refresh()
            } else { self.schedule() }
        }
    }

    private func update(_ state: Fields) {
        if let previous = snapshot, pendingPreview == nil, pendingLine == nil, (previous as NSDictionary).isEqual(to: state) { return }
        let id = state["call"] as? String
        let changedCall = id != callID || snapshot == nil
        snapshot = state
        if changedCall {
            selectedID = nil
            pendingLine = nil
            pendingPreview = nil
            endPlayback()
            callID = id
        }
        let incoming = ((state["voices"] as? Fields)?["speakers"] as? [Fields] ?? []).compactMap(Speaker.init)
        let ready = Set(incoming.filter(\.ready).map(\.id))
        if notifications && !changedCall {
            for speaker in incoming where ready.contains(speaker.id) && !seenReady.contains(speaker.id) {
                let content = UNMutableNotificationContent()
                content.title = "\(speaker.name) is ready"
                content.body = "Open Voice Lab to hear the local comparison."
                let request = UNNotificationRequest(identifier: "\(id ?? "")-\(speaker.id)", content: content, trigger: nil)
                UNUserNotificationCenter.current().add(request, withCompletionHandler: nil)
            }
        }
        seenReady = ready
        if speakers != incoming || changedCall {
            speakers = incoming
            if !speakers.contains(where: { $0.id == selectedID }) { selectedID = speakers.first?.id }
            updatingSelection = true
            table.reloadData()
            if let index = speakers.firstIndex(where: { $0.id == selectedID }) {
                table.selectRowIndexes(IndexSet(integer: index), byExtendingSelection: false)
            }
            updatingSelection = false
        }
        let ids = state["calls"] as? [String] ?? []
        if ids != callIDs || changedCall {
            callIDs = ids
            calls.removeAllItems()
            if id == nil { calls.addItem(withTitle: "Select a saved call") }
            for value in ids {
                calls.menu?.addItem(withTitle: Self.callTitle(value), action: nil, keyEquivalent: "").representedObject = value
            }
            calls.selectItem(whereID: id)
        }
        if isVisible { draw() }
        if let wanted = pendingLine, textFields["state"] as? String != "rendering" {
            pendingLine = nil
            if isVisible, wanted == selectedID, textFields["rendered"] is String { play("tts") }
        }
        if let wanted = pendingPreview, state["previewing"] as? Bool != true {
            pendingPreview = nil
            if isVisible, wanted == selectedID, state["preview"] as? String == wanted { play("preview") }
            else if let error = state["preview_error"] as? String { message(error, error: true) }
        }
    }

    private static let storedDate: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        return f
    }()

    private static func callTitle(_ id: String) -> String {
        guard let date = storedDate.date(from: String(id.prefix(16))) else { return id }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private func draw() {
        let state = snapshot ?? [:]
        let usable = !busy && snapshot != nil && !connectionError
        let capturing = active(captureFields), routing = active(routeFields)
        let calibrating = state["calibrating"] as? Bool == true
        let rendering = state["previewing"] as? Bool == true
        let calibrated = state["calibrated"] as? Bool == true
        let free = usable && !calibrating && !rendering
        let chosen = selected
        let devicesChosen = microphoneID != nil && cableID != nil
        let setup = state["setup"] as? Fields ?? [:]
        let errors = state["analysis_errors"] as? [String] ?? []
        let activity = state["analysis"] as? String ?? ""
        let sameVoice = routing && routeFields["speaker_id"] as? String == selectedID &&
            routeFields["mic"] as? Int == microphoneID && routeFields["output"] as? Int == cableID

        attach.title = capturing ? "Stop capture" : "Attach to Discord"
        attach.isEnabled = usable
        captureInfo.display(capturing ? "Capturing Discord playback. Stop capture when you leave the call." :
            captureFields["state"] as? String == "error" ? captureFields["message"] as? String ?? "Capture stopped." :
            snapshot == nil ? "Connecting to the local studio…" : "Join a call in Discord, then attach to collect its audio.")
        newCall.isEnabled = free && !capturing && !routing && activity.isEmpty
        calls.isEnabled = newCall.isEnabled && !callIDs.isEmpty
        analysis.display(!activity.isEmpty ? activity : errors.first ?? state["text_update_error"] as? String ?? (speakers.isEmpty ?
            "Detected speakers appear here as the call progresses." : "\(speakers.count) detected · \(seenReady.count) ready"))
        modelInfo.display(setup["state"] as? String == "ready" ?
            "Models ready · \(state["device"] as? String == "mps" ? "Apple Metal" : "CPU")" :
            setup["message"] as? String ?? "Starting audio engine…")
        retry.isHidden = setup["state"] as? String != "error" && errors.isEmpty
        retry.isEnabled = usable
        selectedName.display(chosen?.name ?? "No speaker selected")
        progress.isHidden = chosen == nil
        if let s = chosen {
            let fraction = min(1, Double(s.seconds) / Double(max(1, s.requiredSeconds)),
                               Double(s.clips) / Double(max(1, s.requiredClips)))
            progress.doubleValue = s.ready ? 1 : fraction
            let target = s.status == "collecting" ? " · Goal \(s.requiredSeconds) s / \(s.requiredClips) clips" : ""
            var text = "\(s.stage) · \(s.seconds) s of clean speech · \(s.clips) clips\(target)"
            if s.status == "collecting" && fraction == 1 { text += "\nCollect more consistent, isolated speech." }
            if state["preview"] as? String == s.id, let score = state["preview_score"] as? Fields,
               let value = score["score"] as? Double, let actual = score["genuine"] as? Double,
               let count = score["sample_count"] as? Int {
                text += String(format: "\nConverted mic %.3f · Genuine %.3f across %d held-out clips.", value, actual, count)
            }
            enrollment.display(text)
        } else { enrollment.display("Select a caller to hear their original voice or compare it with your mic.") }
        listen.isEnabled = usable && (chosen?.clips ?? 0) > 0
        compare.isEnabled = free && !routing && calibrated && (chosen?.ready == true || chosen?.status == "needs_more_audio")
        compare.title = rendering ? "Rendering comparison…" : "Compare with my mic"
        replay.isHidden = selectedID == nil || selectedID != state["preview"] as? String
        stopAudio.isHidden = playback == nil
        importVoice.isEnabled = usable
        let textState = textFields["state"] as? String
        let statuses = (state["text_voices"] as? [String: Fields]) ?? [:]
        let building = textState == "building" || textState == "rendering"
        let textJob = state["text_job"] as? Fields
        let textBusy = textJob != nil || statuses.values.contains { ["building", "rendering"].contains($0["state"] as? String ?? "") }
        let automatic = textFields["automatic"] as? Bool == true
        let pendingUpdate = textFields["pending_update"] is Fields
        let resume = !automatic && !pendingUpdate && textFields["task"] as? String == "build" && ["paused", "error"].contains(textState ?? "")
        let hasVoice = textFields["available"] is String
        var detail = textFields["message"] as? String ?? (chosen?.textEligible == true ?
            "Preview builds use 12 checked teaching lines. Full builds use 128." :
            "Collect 30 seconds across six clear excerpts to build an English text voice.")
        if let score = textFields["similarity"] as? Double, let count = textFields["lines"] as? Int {
            detail += String(format: "\nSpeaker similarity %.3f across %d held-out lines.", score, count)
            if let genuine = textFields["genuine"] as? Double, let wer = textFields["wer"] as? Double {
                detail += String(format: " Genuine %.3f · word error %.0f%%.", genuine, 100 * wer)
            }
        }
        textInfo.display(detail)
        buildVoice.isEnabled = usable && !routing && !textBusy && (resume || chosen?.textEligible == true)
        buildVoice.title = resume ? "Resume build" : "Build text voice"
        if resume { buildSize.selectItem(at: textFields["mode"] as? String == "full" ? 1 : 0) }
        buildSize.isEnabled = buildVoice.isEnabled && !resume
        useLatest.isHidden = !resume
        useLatest.isEnabled = usable && !routing && !textBusy && chosen?.textEligible == true
        pauseBuild.isHidden = !building || automatic
        autoUpdates.isHidden = !hasVoice
        autoUpdates.isEnabled = usable && hasVoice
        autoUpdates.state = chosen?.autoUpdate == true ? .on : .off
        updateInfo.isHidden = !hasVoice || chosen?.autoUpdate != true
        if let s = chosen, hasVoice, s.autoUpdate {
            var update: String
            if automatic && building {
                update = "Checking a new version. Your current voice is available."
            } else if pendingUpdate {
                update = "An update is queued. Your current voice is available."
            } else {
                update = "\(min(s.newTrainingClips, s.updateAfterClips)) of \(s.updateAfterClips) new training clips for the next check."
            }
            if let reason = textFields["decision_reason"] as? String, !detail.contains(reason) {
                update += " Last check. " + reason
            }
            updateInfo.display(update)
        }
        listenAlternate.isHidden = !(textFields["alternate"] is String)
        listenAlternate.title = textFields["last_decision"] as? String == "promoted" ? "Hear previous build" : "Hear candidate"
        reviewVoice.isHidden = !hasVoice
        saveVoice.isHidden = !hasVoice
        saveLine.isHidden = !(textFields["rendered"] is String)
        speakLine.isEnabled = usable && !routing && (!textBusy || textJob?["automatic"] as? Bool == true) && hasVoice
        voiceLine.isEnabled = speakLine.isEnabled
        replayLine.isHidden = !(textFields["rendered"] is String)
        microphones.isEnabled = !busy
        cables.isEnabled = !busy
        record.isEnabled = free && !routing && microphoneID != nil
        chooseFile.isEnabled = free && !routing
        micInfo.display(calibrating ? "Recording now. Speak normally for six seconds." :
            state["calibration_error"] as? String ?? (calibrated ? "Mic sample saved." : "Record or choose a short sample of your microphone."))
        installer.isHidden = cables.numberOfItems > 1
        installer.isEnabled = usable && state["installing_driver"] as? Bool != true
        let driver = state["driver_message"] as? String ?? ""
        let guide = cableID == nil ? "Install or select a virtual cable, then refresh devices." :
            "In Discord → Voice & Video, select \(cables.titleOfSelectedItem ?? "this cable") as input. Keep headphones as output."
        outputInfo.display(driver.isEmpty ? guide : driver + "\n" + guide)
        let canRoute = free && devicesChosen && calibrated && chosen?.ready == true
        check.isEnabled = canRoute && !routing
        route.isEnabled = canRoute && !sameVoice
        route.title = sameVoice ? "Voice in use" : routing ? "Switch to \(chosen?.name ?? "selected voice")" : "Use selected voice"
        normal.isHidden = !routing || routeFields["mode"] as? String == "microphone"
        normal.isEnabled = free && devicesChosen
        stopRoute.isHidden = !routing
        stopRoute.isEnabled = usable
        let routeState = routeFields["state"] as? String ?? ""
        let timing = routeFields["timing"] as? String ?? ""
        let timingText = timing.isEmpty ? "" : "\n" + timing
        if routeState == "running" {
            let name = routeFields["mode"] as? String == "microphone" ? "Your normal microphone" :
                speakers.first { $0.id == routeFields["speaker_id"] as? String }?.name ?? "Selected voice"
            routeInfo.display("\(name) is routed to \(routeFields["output_name"] as? String ?? "your virtual cable")." + timingText)
        } else {
            let detail = routeFields["message"] as? String ?? ""
            routeInfo.display((detail.isEmpty ? "Live output is off." : detail) + timingText)
        }
    }

    func numberOfRows(in tableView: NSTableView) -> Int { speakers.count }

    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let key = NSUserInterfaceItemIdentifier("speakerCell")
        let cell = tableView.makeView(withIdentifier: key, owner: nil) as? SpeakerCell ?? SpeakerCell(frame: .zero)
        cell.identifier = key
        cell.configure(speakers[row])
        return cell
    }

    func tableViewSelectionDidChange(_ notification: Notification) {
        guard !updatingSelection, speakers.indices.contains(table.selectedRow) else { return }
        let id = speakers[table.selectedRow].id
        guard id != selectedID else { return }
        selectedID = id
        pendingLine = nil
        pendingPreview = nil
        endPlayback()
        message("")
        draw()
    }

    private var microphoneID: Int? { microphones.selectedItem?.representedObject as? Int }
    private var cableID: Int? { cables.selectedItem?.representedObject as? Int }

    private func message(_ text: String, error: Bool = false) {
        feedback.textColor = error ? .systemRed : .secondaryLabelColor
        feedback.display(text)
        feedback.isHidden = text.isEmpty
    }

    @objc private func discoverDevices() {
        guard !discovering && !stopped else { return }
        discovering = true
        refreshDevices.isEnabled = false
        client.devices { [weak self] result in
            guard let self, !self.stopped else { return }
            self.discovering = false
            self.refreshDevices.isEnabled = true
            switch result {
            case .success(let payload):
                let devices = payload["devices"] as? [Fields] ?? []
                self.fillDevices(self.microphones, devices.filter { $0["virtual"] as? Bool == false && ($0["inputs"] as? Int ?? 0) > 0 })
                self.fillDevices(self.cables, devices.filter { $0["virtual"] as? Bool == true && ($0["outputs"] as? Int ?? 0) > 0 })
                self.draw()
            case .failure(let error): self.message(error.localizedDescription, error: true)
            }
        }
    }

    private func fillDevices(_ popup: NSPopUpButton, _ devices: [Fields]) {
        let previous = popup.selectedItem?.representedObject as? Int
        popup.removeAllItems()
        popup.addItem(withTitle: popup === microphones ? "Select a microphone" : "Select a virtual cable")
        for device in devices {
            guard let id = device["id"] as? Int, let name = device["name"] as? String else { continue }
            popup.menu?.addItem(withTitle: name, action: nil, keyEquivalent: "").representedObject = id
            if id == previous { popup.selectItem(at: popup.numberOfItems - 1) }
        }
    }

    private func act(_ name: String, _ extra: Fields = [:], then: (() -> Void)? = nil) {
        guard !busy, snapshot != nil, !stopped else { return }
        busy = true
        draw()
        var body = extra
        body["call"] = callID as Any? ?? NSNull()
        client.action("/api/" + name, body: body) { [weak self] result in
            guard let self, !self.stopped else { return }
            self.busy = false
            switch result {
            case .success:
                self.controlRevision += 1
                self.message("")
                then?()
            case .failure(let error): self.message(error.localizedDescription, error: true)
            }
            self.draw()
            self.refresh()
        }
    }

    @objc private func deviceChanged() { draw() }
    @objc private func toggleCapture() { act(active(captureFields) ? "detach" : "attach") }
    @objc private func createCall() { act("new") }
    @objc private func selectCall() {
        if let id = calls.selectedItem?.representedObject as? String, id != callID {
            calls.selectItem(whereID: callID)
            act("select", ["id": id])
        }
    }
    @objc private func retrySetup() { act("retry") }
    @objc private func installCable() { act("cable/installer") }
    @objc private func recordMic() { if let id = microphoneID { act("mic/record", ["mic": id]) } }
    @objc private func checkRoute() { changeRoute("check") }
    @objc private func routeVoice() { changeRoute("route") }
    @objc private func routeMic() { changeRoute("route/microphone") }
    @objc private func disconnect() { act("route/stop") }

    private func changeRoute(_ endpoint: String) {
        guard let mic = microphoneID, let cable = cableID else { return }
        var body: Fields = ["mic": mic, "output": cable]
        if endpoint != "route/microphone" {
            guard let id = selectedID else { return }
            body["speaker"] = id
        }
        act(endpoint, body)
    }

    @objc private func toggleNotify() {
        if notify.state == .off {
            notifications = false
            UserDefaults.standard.set(false, forKey: "NotifyWhenVoiceReady")
            schedule()
            return
        }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert]) { [weak self] ok, error in
            DispatchQueue.main.async {
                guard let self, !self.stopped else { return }
                self.notifications = ok
                self.notify.state = ok ? .on : .off
                UserDefaults.standard.set(ok, forKey: "NotifyWhenVoiceReady")
                if !ok { self.message(error?.localizedDescription ?? "Enable Voice Lab notifications in System Settings to receive alerts.") }
                self.schedule()
            }
        }
    }

    @objc private func buildTextVoice() {
        guard let id = selectedID else { return }
        let interrupted = textFields["automatic"] as? Bool != true && !(textFields["pending_update"] is Fields) && textFields["task"] as? String == "build" &&
            ["paused", "error"].contains(textFields["state"] as? String ?? "")
        let mode = interrupted ? "resume" : buildSize.indexOfSelectedItem == 1 ? "full" : "preview"
        act("text/build", ["speaker": id, "mode": mode])
    }
    @objc private func rebuildTextVoice() {
        guard let id = selectedID else { return }
        act("text/build", ["speaker": id, "mode": textFields["mode"] as? String ?? "preview"])
    }
    @objc private func toggleUpdates() {
        guard let id = selectedID else { return }
        act("text/updates", ["speaker": id, "enabled": autoUpdates.state == .on])
    }
    @objc private func playAlternate() { play("alternate-review") }
    @objc private func pauseTextVoice() { act("text/pause") }
    @objc private func playReview() { play("review") }
    @objc private func playLine() { play("tts") }
    @objc private func generateLine() {
        guard let id = selectedID, speakLine.isEnabled else { return }
        act("text/speak", ["speaker": id, "text": voiceLine.stringValue]) { [weak self] in
            if self?.selectedID == id { self?.pendingLine = id }
        }
    }
    @objc private func exportVoice() { saveGenerated(profile: true) }
    @objc private func exportLine() { saveGenerated(profile: false) }

    private func saveGenerated(profile: Bool) {
        guard let call = callID, let sid = selectedID else { return }
        let panel = NSSavePanel()
        let ext = profile ? "pt" : "wav"
        panel.nameFieldStringValue = "Speaker-\(sid).\(ext)"
        panel.beginSheetModal(for: window) { [weak self] answer in
            guard let self, answer == .OK, let destination = panel.url else { return }
            let role = profile ? "profile" : "tts"
            self.client.download("/audio/\(call)/\(role)/\(sid)", profile: profile) { result in
                do {
                    let file = try result.get()
                    defer { try? FileManager.default.removeItem(at: file) }
                    try Data(contentsOf: file).write(to: destination, options: .atomic)
                    self.message("Saved " + destination.lastPathComponent)
                } catch { self.message(error.localizedDescription, error: true) }
            }
        }
    }
    @objc private func playCaller() { play("sample") }
    @objc private func replayComparison() { play("preview") }
    @objc private func compareMic() {
        guard let id = selectedID else { return }
        endPlayback()
        act("preview", ["speaker": id]) { [weak self] in
            guard let self, self.selectedID == id else { return }
            self.pendingPreview = id
        }
    }

    private func play(_ role: String) {
        guard let id = selectedID, let call = callID else { return }
        endPlayback()
        let generation = audioGeneration
        client.download("/audio/\(call)/\(role)/\(id)") { [weak self] result in
            switch result {
            case .failure(let error):
                if let self, !self.stopped, generation == self.audioGeneration { self.message(error.localizedDescription, error: true) }
            case .success(let file):
                guard let self, !self.stopped, generation == self.audioGeneration,
                      self.selectedID == id, self.callID == call else {
                    try? FileManager.default.removeItem(at: file)
                    return
                }
                self.audioFile = file
                guard let sound = NSSound(contentsOf: file, byReference: true) else {
                    self.endPlayback()
                    self.message("The audio sample could not be opened.", error: true)
                    return
                }
                self.playback = sound
                sound.delegate = self
                if sound.play() {
                    let labels = ["sample": "Playing the original caller.", "review": "Original first, then the trained text voice.",
                                  "alternate-review": "Original first, then the alternate text voice.",
                                  "tts": "Playing the generated line.", "preview": "Your microphone first, then the converted voice."]
                    self.message(labels[role] ?? "")
                    self.stopAudio.isHidden = false
                } else {
                    self.endPlayback()
                    self.message("Playback could not start. Check your Mac’s audio output.", error: true)
                }
            }
        }
    }

    @objc private func cancelPlayback() { endPlayback(); message("") }

    private func endPlayback() {
        audioGeneration += 1
        playback?.delegate = nil
        playback?.stop()
        playback = nil
        if let file = audioFile { try? FileManager.default.removeItem(at: file); audioFile = nil }
        stopAudio.isHidden = true
    }

    func sound(_ sound: NSSound, didFinishPlaying flag: Bool) {
        DispatchQueue.main.async { [weak self] in
            guard let self, self.playback === sound else { return }
            self.endPlayback()
            self.message(flag ? "" : "Audio playback was interrupted.", error: !flag)
        }
    }

    @objc private func pickFile() { chooseAudio(recording: false) }
    @objc private func importFile() { chooseAudio(recording: true) }

    private func chooseAudio(recording: Bool) {
        let types = ["wav": "audio/wav", "m4a": "audio/mp4", "mp3": "audio/mpeg", "webm": "audio/webm",
                     "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac", "aif": "audio/aiff", "aiff": "audio/aiff"]
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types.keys.compactMap { UTType(filenameExtension: $0) }
        panel.prompt = recording ? "Import recording" : "Use mic sample"
        let targetCall = recording ? (callID ?? "new") : nil
        panel.beginSheetModal(for: window) { [weak self] response in
            guard let self, !self.stopped, response == .OK, let file = panel.url,
                  let mime = types[file.pathExtension.lowercased()] else { return }
            self.busy = true
            self.draw()
            self.client.upload(file, mime: mime, voiceCall: targetCall) { [weak self] result in
                guard let self, !self.stopped else { return }
                self.busy = false
                switch result {
                case .success:
                    self.controlRevision += 1
                    self.message(recording ? "Recording added. Speakers will appear after analysis." : "Mic sample saved.")
                case .failure(let error): self.message(error.localizedDescription, error: true)
                }
                self.draw()
                self.refresh()
            }
        }
    }
}

private extension NSPopUpButton {
    func selectItem(whereID id: String?) {
        if let index = itemArray.firstIndex(where: { $0.representedObject as? String == id }) {
            selectItem(at: index)
        }
    }
}
