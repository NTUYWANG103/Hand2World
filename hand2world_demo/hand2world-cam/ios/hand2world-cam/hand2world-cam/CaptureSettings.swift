import ARKit
import Foundation

/// User-controllable capture settings persisted via `@AppStorage`.
///
/// The ARKit video format is selected by a stable descriptor (W x H @ fps)
/// rather than an array index, so reorderings in Apple's format list can't
/// silently change what the app uses.
struct CaptureSettings {
    /// All rear-cam world-tracking formats supported on this device.
    static let supportedFormats: [ARConfiguration.VideoFormat] = {
        let all = ARWorldTrackingConfiguration.supportedVideoFormats
        // De-duplicate by (w, h, fps) preserving priority order, then sort
        // large→small for a sensible picker default.
        var seen = Set<String>()
        var uniq: [ARConfiguration.VideoFormat] = []
        for f in all {
            let key = CaptureSettings.descriptor(for: f)
            if seen.insert(key).inserted { uniq.append(f) }
        }
        return uniq.sorted {
            let a = ($0.imageResolution.width * $0.imageResolution.height, $0.framesPerSecond)
            let b = ($1.imageResolution.width * $1.imageResolution.height, $1.framesPerSecond)
            return (a.0, a.1) > (b.0, b.1)
        }
    }()

    static func descriptor(for f: ARConfiguration.VideoFormat) -> String {
        "\(Int(f.imageResolution.width))x\(Int(f.imageResolution.height))@\(f.framesPerSecond)"
    }

    static func format(matching descriptor: String) -> ARConfiguration.VideoFormat {
        supportedFormats.first(where: { CaptureSettings.descriptor(for: $0) == descriptor })
            ?? supportedFormats.first!
    }

    /// Default format for first launch — auto-adapts to the device:
    /// picks 1280×720 at the device's highest supported fps; if 720p is
    /// not in the list, falls back to the overall highest-fps format
    /// (tie-breaking on larger resolution).
    static var defaultFormatDescriptor: String {
        let at720 = supportedFormats.filter {
            Int($0.imageResolution.width) == 1280 && Int($0.imageResolution.height) == 720
        }
        if let best720 = at720.max(by: { $0.framesPerSecond < $1.framesPerSecond }) {
            return CaptureSettings.descriptor(for: best720)
        }
        if let anyBest = supportedFormats.max(by: {
            let a = ($0.framesPerSecond, Int($0.imageResolution.width * $0.imageResolution.height))
            let b = ($1.framesPerSecond, Int($1.imageResolution.width * $1.imageResolution.height))
            return a < b
        }) {
            return CaptureSettings.descriptor(for: anyBest)
        }
        return CaptureSettings.descriptor(for: supportedFormats.first!)
    }
}
