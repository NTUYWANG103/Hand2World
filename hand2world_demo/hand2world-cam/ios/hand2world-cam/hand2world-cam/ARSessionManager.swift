import ARKit
import Combine

/// Wraps a rear-camera `ARWorldTrackingConfiguration` session and publishes
/// human-readable status for the UI to observe. Every ARKit frame is handed
/// off through `onFrame` (set by the owner) — no queuing or throttling here.
final class ARSessionManager: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published private(set) var isRunning = false
    @Published private(set) var trackingState: String = "Not started"
    @Published private(set) var resolution: String = "-"
    @Published private(set) var captureFPS: Double = 0

    var onFrame: ((ARFrame) -> Void)?

    /// Dedicated high-QoS queue so the ARSession's per-frame delegate
    /// callbacks aren't queued behind main-thread UI work.
    private let sessionQueue = DispatchQueue(
        label: "hand2world.arsession", qos: .userInteractive
    )
    private var frameCountInWindow = 0
    private var windowStartedAt: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    override init() {
        super.init()
        session.delegateQueue = sessionQueue
        session.delegate = self
    }

    /// Start (or restart) the session using the video format identified by
    /// `formatDescriptor` (e.g. "1920x1440@60"). Falls back to the device's
    /// first supported format if the descriptor doesn't match anything.
    func start(formatDescriptor: String) {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = []
        config.environmentTexturing = .none
        config.isLightEstimationEnabled = false
        config.videoFormat = CaptureSettings.format(matching: formatDescriptor)

        let r = config.videoFormat.imageResolution
        DispatchQueue.main.async {
            self.resolution = "\(Int(r.width))x\(Int(r.height)) @ \(config.videoFormat.framesPerSecond)fps"
        }

        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        DispatchQueue.main.async { self.isRunning = true }
    }

    func stop() {
        session.pause()
        DispatchQueue.main.async {
            self.isRunning = false
            self.trackingState = "Stopped"
            self.captureFPS = 0
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameCountInWindow += 1
        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - windowStartedAt
        if dt >= 0.5 {
            let fps = Double(frameCountInWindow) / dt
            frameCountInWindow = 0
            windowStartedAt = now
            DispatchQueue.main.async { self.captureFPS = fps }
        }
        onFrame?(frame)
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let s: String
        switch camera.trackingState {
        case .normal: s = "Normal"
        case .notAvailable: s = "Not available"
        case .limited(.initializing): s = "Initializing"
        case .limited(.excessiveMotion): s = "Excessive motion"
        case .limited(.insufficientFeatures): s = "Insufficient features"
        case .limited(.relocalizing): s = "Relocalizing"
        case .limited: s = "Limited"
        }
        DispatchQueue.main.async { self.trackingState = s }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.trackingState = "Error: \(error.localizedDescription)"
            self.isRunning = false
        }
    }
}
