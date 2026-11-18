import AppKit

// Render the waveform at each Finder and Dock resolution so the gaps survive at 16 px.
let folder = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
for points in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = points * scale
        let image = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
                                     bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                     bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: image)
        let cg = NSGraphicsContext.current!.cgContext
        cg.scaleBy(x: CGFloat(pixels) / 128, y: CGFloat(pixels) / 128)
        let tile = NSBezierPath(roundedRect: NSRect(x: 5, y: 5, width: 118, height: 118), xRadius: 28, yRadius: 28)
        NSGradient(starting: NSColor(srgbRed: 0.16, green: 0.23, blue: 0.24, alpha: 1),
                   ending: NSColor(srgbRed: 0.035, green: 0.07, blue: 0.09, alpha: 1))?.draw(in: tile, angle: -60)
        let heights = [13, 32, 57, 43, 73, 47, 30, 17]
        for (i, height) in heights.enumerated() {
            let h = CGFloat(height)
            NSColor(srgbRed: 0.47 + 0.045 * Double(i), green: 0.91 + 0.005 * Double(i),
                    blue: 0.80 - 0.02 * Double(i), alpha: 1).setFill()
            NSBezierPath(roundedRect: NSRect(x: 20 + CGFloat(i) * 12, y: (128-h)/2, width: 7, height: h),
                         xRadius: 3.5, yRadius: 3.5).fill()
        }
        NSGraphicsContext.restoreGraphicsState()
        let suffix = scale == 2 ? "@2x" : ""
        try image.representation(using: .png, properties: [:])!.write(to:
            folder.appendingPathComponent("icon_\(points)x\(points)\(suffix).png"))
    }
}
