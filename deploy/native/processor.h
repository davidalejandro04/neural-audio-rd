#pragma once
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <string>
#include <vector>

namespace nfx {
const char* runtimeVersion();
// Each channel/stream owns an instance. Construct and destroy outside audio callbacks.
class Processor {
public:
    explicit Processor(const std::filesystem::path& model);
    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;
    size_t blockSize() const noexcept { return block_; }
    int sampleRate() const noexcept { return rate_; }
    void reset() noexcept;
    bool process(const float* input, float* output, float drive, float mix) noexcept;
    const std::vector<float>& history() const noexcept;
    const std::vector<float>& hidden() const noexcept;
    const std::string& error() const noexcept { return error_; }
private:
    struct Slot { std::string name; std::vector<int64_t> shape; std::vector<float> data; };
    Ort::Env env_{nullptr};
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::RunOptions runOptions_{nullptr};
    std::vector<Slot> inputs_, outputs_;
    std::vector<Ort::Value> inputValues_, outputValues_;
    std::vector<const char*> inputNames_, outputNames_;
    std::vector<std::pair<size_t, size_t>> states_;
    std::vector<float> empty_;
    size_t audio_{0}, controls_{0}, wet_{0}, history_{0}, hidden_{0}, block_{0};
    bool hasHidden_{false};
    int rate_{0};
    std::string error_;
};

// Adapts arbitrary host callback lengths to the fixed ONNX block, adding exactly B samples latency.
class BufferedProcessor {
public:
    explicit BufferedProcessor(const std::filesystem::path& model);
    void reset() noexcept;
    bool process(float* audio, size_t samples, float drive, float mix) noexcept;
    size_t latency() const noexcept { return engine_.blockSize(); }
    int sampleRate() const noexcept { return engine_.sampleRate(); }
private:
    Processor engine_;
    std::vector<float> input_, output_;
    size_t fill_{0};
};
}
