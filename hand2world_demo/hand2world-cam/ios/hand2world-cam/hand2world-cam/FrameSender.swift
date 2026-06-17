import ARKit
import Combine
import CoreImage
import Foundation
import UIKit

/// Sends per-ARFrame payloads to the Mac ingest server over WebSocket.
///
/// Wire format (one binary WS message per frame):
///
///     [4 bytes: header_len (uint32, little-endian)]
///     [header_len bytes: UTF-8 JSON header]
///     [remaining bytes: JPEG-encoded RGB image]
///
/// Header JSON:
///
///     {
///       "frame_id":   uint,
///       "timestamp":  double (ARKit clock),
///       "width":      int,
///       "height":     int,
///       "fx": float, "fy": float, "cx": float, "cy": float,
///       "T_cw": [r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz, 0,0,0,1]
///     }
///
/// `T_cw` is row-major camera-to-world in the ARKit world frame (right-handed,
/// +Y up, origin at session start).
final class FrameSender: ObservableObject {
    @Published private(set) var status: String = "Disconnected"
    @Published private(set) var framesSent: Int = 0
    @Published private(set) var framesDropped: Int = 0
    @Published private(set) var lastKbps: Double = 0
    /// True while the user intends to stay connected. Unlike `status`, this
    /// stays true across automatic reconnects so the UI can keep the button
    /// in its "Disconnect" state while the socket is down.
    @Published private(set) var wantsConnection: Bool = false

    /// Max send rate (Hz). Values above the ARKit camera format's fps are
    /// effectively clamped by source availability.
    var targetFPS: Double = 30

    /// JPEG encoder quality in [0.05, 1.0]. 0.6 is a good default for 1080p+.
    var jpegQuality: CGFloat = 0.6

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    /// Concurrent queue — JPEG encoding parallelises across cores so we can
    /// keep up with 60+ fps ARKit formats without serialising on one core.
    private let encodeQueue = DispatchQueue(
        label: "hand2world.encode",
        qos: .userInitiated,
        attributes: .concurrent
    )
    private let inFlightLock = NSLock()

    private var session: URLSession?
    private var wsTask: URLSessionWebSocketTask?
    private var urlString: String?
    private var frameCounter: UInt64 = 0
    private var inFlight = 0
    private var lastSentAt: CFAbsoluteTime = 0
    private var bytesSinceTick: Int = 0
    private var lastTickAt: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    /// Exponential backoff for auto-reconnect. Resets to min on any sign of
    /// life from the socket (successful receive or send).
    private var reconnectDelay: Double = 1.0
    private let reconnectMinDelay: Double = 1.0
    private let reconnectMaxDelay: Double = 5.0

    /// Pipeline depth: how many encode-and-send operations may overlap.
    /// 6 keeps memory bounded while giving the concurrent encode queue
    /// enough parallelism to hit 60+ fps on modern iPhones.
    private let maxInFlight = 6

    // MARK: - Connection

    func connect(urlString: String) {
        DispatchQueue.main.async {
            self.urlString = urlString
            self.wantsConnection = true
            self.reconnectDelay = self.reconnectMinDelay
            self.openSocket()
        }
    }

    func disconnect() {
        DispatchQueue.main.async {
            self.wantsConnection = false
            self.teardownSocket()
            self.status = "Disconnected"
        }
    }

    /// Open a fresh WebSocket on the main queue. Safe to call repeatedly;
    /// any existing socket is torn down first.
    private func openSocket() {
        // main queue only
        guard wantsConnection, let s = urlString, let url = URL(string: s) else {
            if wantsConnection { status = "Invalid URL" }
            wantsConnection = false
            return
        }
        teardownSocket()
        let cfg = URLSessionConfiguration.default
        cfg.waitsForConnectivity = false
        cfg.timeoutIntervalForRequest = 10
        let s2 = URLSession(configuration: cfg)
        let t = s2.webSocketTask(with: url)
        self.session = s2
        self.wsTask = t
        status = "Connecting to \(url.host ?? "?")…"
        t.resume()
        receiveLoop(task: t)
    }

    /// Cancel the current task/session without touching `wantsConnection`.
    private func teardownSocket() {
        // main queue only
        wsTask?.cancel(with: .goingAway, reason: nil)
        wsTask = nil
        session?.invalidateAndCancel()
        session = nil
    }

    /// Called on main when the current socket reports any sign of life.
    /// Resets backoff and clears any "reconnecting" status.
    private func noteAlive(task: URLSessionWebSocketTask) {
        // main queue only
        guard wsTask === task else { return }
        reconnectDelay = reconnectMinDelay
        if let host = task.originalRequest?.url?.host {
            status = "Connected to \(host)"
        }
    }

    /// Called on main when the current socket dies. If the user still wants
    /// a connection, schedule a reconnect with exponential backoff.
    private func noteFailure(task: URLSessionWebSocketTask, error: Error) {
        // main queue only
        guard wsTask === task else { return }   // stale callback — ignore
        teardownSocket()
        guard wantsConnection else {
            status = "Disconnected"
            return
        }
        let delay = reconnectDelay
        reconnectDelay = min(reconnectDelay * 2, reconnectMaxDelay)
        status = String(format: "Reconnecting in %.1fs: %@", delay, error.localizedDescription)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.openSocket()
        }
    }

    /// Keep a `receive` outstanding so URLSession surfaces disconnect errors.
    private func receiveLoop(task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                DispatchQueue.main.async { self.noteAlive(task: task) }
                self.receiveLoop(task: task)
            case .failure(let err):
                DispatchQueue.main.async { self.noteFailure(task: task, error: err) }
            }
        }
    }

    // MARK: - Frame ingestion

    func enqueue(frame: ARFrame) {
        guard wantsConnection, wsTask != nil else { return }

        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastSentAt >= 1.0 / targetFPS else {
            DispatchQueue.main.async { self.framesDropped += 1 }
            return
        }

        inFlightLock.lock()
        if inFlight >= maxInFlight {
            inFlightLock.unlock()
            DispatchQueue.main.async { self.framesDropped += 1 }
            return
        }
        inFlight += 1
        inFlightLock.unlock()
        lastSentAt = now

        let pixelBuffer = frame.capturedImage
        let intrinsics = frame.camera.intrinsics
        let transform = frame.camera.transform
        let resolution = frame.camera.imageResolution
        let timestamp = frame.timestamp
        frameCounter += 1
        let thisID = frameCounter

        encodeQueue.async { [weak self] in
            guard let self = self else { return }
            defer {
                self.inFlightLock.lock()
                self.inFlight -= 1
                self.inFlightLock.unlock()
            }
            guard let payload = self.encode(
                id: thisID,
                pixelBuffer: pixelBuffer,
                K: intrinsics,
                T: transform,
                resolution: resolution,
                timestamp: timestamp
            ) else { return }
            self.sendData(payload)
        }
    }

    // MARK: - Encoding

    private func encode(
        id: UInt64,
        pixelBuffer: CVPixelBuffer,
        K: simd_float3x3,
        T: simd_float4x4,
        resolution: CGSize,
        timestamp: TimeInterval
    ) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return nil }
        guard let jpeg = UIImage(cgImage: cg).jpegData(compressionQuality: jpegQuality) else {
            return nil
        }

        // simd matrices are column-major; K[col][row], T[col][row].
        let fx = K[0][0], fy = K[1][1]
        let cx = K[2][0], cy = K[2][1]
        let T_cw_row_major: [Float] = [
            T[0][0], T[1][0], T[2][0], T[3][0],
            T[0][1], T[1][1], T[2][1], T[3][1],
            T[0][2], T[1][2], T[2][2], T[3][2],
            T[0][3], T[1][3], T[2][3], T[3][3],
        ]

        let header: [String: Any] = [
            "frame_id": id,
            "timestamp": timestamp,
            "width": Int(resolution.width),
            "height": Int(resolution.height),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "T_cw": T_cw_row_major,
        ]
        guard let headerData = try? JSONSerialization.data(withJSONObject: header) else {
            return nil
        }

        var out = Data(capacity: 4 + headerData.count + jpeg.count)
        var len = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(jpeg)
        return out
    }

    private func sendData(_ data: Data) {
        guard let ws = wsTask else { return }
        let size = data.count
        ws.send(.data(data)) { [weak self] err in
            guard let self = self else { return }
            if let err = err {
                DispatchQueue.main.async { self.noteFailure(task: ws, error: err) }
                return
            }
            DispatchQueue.main.async {
                self.noteAlive(task: ws)
                self.framesSent += 1
                self.bytesSinceTick += size
                let now = CFAbsoluteTimeGetCurrent()
                let dt = now - self.lastTickAt
                if dt >= 1.0 {
                    self.lastKbps = Double(self.bytesSinceTick) * 8.0 / 1000.0 / dt
                    self.bytesSinceTick = 0
                    self.lastTickAt = now
                }
            }
        }
    }
}
