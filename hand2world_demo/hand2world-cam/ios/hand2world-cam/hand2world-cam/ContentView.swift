import ARKit
import SwiftUI

struct ContentView: View {
    @StateObject private var ar = ARSessionManager()
    @StateObject private var sender = FrameSender()

    @AppStorage("macWebSocketURL")  private var macURL: String = ""
    @AppStorage("captureFormat")    private var formatDescriptor: String = CaptureSettings.defaultFormatDescriptor
    @AppStorage("sendTargetFPS")    private var sendFPS: Double = 120
    @AppStorage("jpegQuality")      private var jpegQuality: Double = 0.8

    @State private var showSettings = false

    var body: some View {
        ZStack {
            CameraBackdrop(session: ar.session)
                .ignoresSafeArea()

            VStack {
                Spacer()
                controls
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(16)
                    .padding()
            }
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            sender.targetFPS = sendFPS
            sender.jpegQuality = CGFloat(jpegQuality)
            ar.onFrame = { [weak sender] frame in
                sender?.enqueue(frame: frame)
            }
            ar.start(formatDescriptor: formatDescriptor)
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
        }
        .onChange(of: sendFPS) { _, new in sender.targetFPS = new }
        .onChange(of: jpegQuality) { _, new in sender.jpegQuality = CGFloat(new) }
        .onChange(of: formatDescriptor) { _, new in ar.start(formatDescriptor: new) }
        .sheet(isPresented: $showSettings) {
            SettingsSheet(
                formatDescriptor: $formatDescriptor,
                sendFPS: $sendFPS,
                jpegQuality: $jpegQuality
            )
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField("ws://<mac-ip>:8765", text: $macURL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled(true)
                .keyboardType(.URL)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
                .onChange(of: macURL) { _, new in
                    let stripped = new.trimmingCharacters(in: .whitespacesAndNewlines)
                    if stripped != new { macURL = stripped }
                }

            HStack {
                Button(sender.wantsConnection ? "Disconnect" : "Connect") {
                    if sender.wantsConnection {
                        sender.disconnect()
                    } else {
                        sender.connect(urlString: macURL)
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)

                Spacer()

                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gearshape")
                        .font(.title3)
                }
            }

            statLine("AR", ar.isRunning ? ar.trackingState : "Stopped")
            statLine("Res", ar.resolution)
            statLine("Src", String(format: "%.1f fps (ARKit delivers)", ar.captureFPS))
            statLine("WS", sender.status)
            statLine("Flow", "\(sender.framesSent) sent / \(sender.framesDropped) dropped  \(Int(sender.lastKbps)) kbps")
        }
    }

    private func statLine(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).bold().frame(width: 52, alignment: .leading)
            Text(value).font(.system(.caption, design: .monospaced)).lineLimit(1)
        }
    }
}

/// Renders the AR camera feed as a background layer.
struct CameraBackdrop: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        v.session = session
        v.automaticallyUpdatesLighting = false
        v.rendersContinuously = true
        return v
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - Settings sheet

private struct SettingsSheet: View {
    @Binding var formatDescriptor: String
    @Binding var sendFPS: Double
    @Binding var jpegQuality: Double

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Video format", selection: $formatDescriptor) {
                        ForEach(CaptureSettings.supportedFormats, id: \.self) { f in
                            Text(CaptureSettings.descriptor(for: f)).tag(CaptureSettings.descriptor(for: f))
                        }
                    }
                } header: {
                    Text("ARKit capture")
                } footer: {
                    Text("Rear camera, world-tracking config only. Changes restart the AR session immediately.")
                        .font(.footnote)
                }

                Section {
                    HStack {
                        Text("Send rate")
                        Spacer()
                        Text("\(Int(sendFPS)) fps").monospaced()
                    }
                    Slider(value: $sendFPS, in: 5...120, step: 1)
                } header: {
                    Text("Transmission rate")
                } footer: {
                    Text("Max frames per second delivered to the Mac. Clamped by the video format's native fps. Applies live.")
                        .font(.footnote)
                }

                Section {
                    HStack {
                        Text("JPEG quality")
                        Spacer()
                        Text(String(format: "%.2f", jpegQuality)).monospaced()
                    }
                    Slider(value: $jpegQuality, in: 0.2...1.0, step: 0.05)
                } header: {
                    Text("Compression")
                } footer: {
                    Text("0.2 = smallest and grainiest, 1.0 = best and largest. Format is JPEG.")
                        .font(.footnote)
                }
            }
            .navigationTitle("Capture settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
