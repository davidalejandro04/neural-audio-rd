#include "processor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace nfx {
namespace {
int moduleMarker = 0;
struct NativeRuntime {
#ifdef _WIN32
    HMODULE library{nullptr};
#endif
    const OrtApiBase* base{nullptr};
    NativeRuntime() {
#ifdef _WIN32
        HMODULE own = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&moduleMarker), &own)) throw std::runtime_error("Cannot locate runtime module");
        std::vector<wchar_t> path(32768);
        const auto length = GetModuleFileNameW(own, path.data(), static_cast<DWORD>(path.size()));
        if (!length || length == path.size()) throw std::runtime_error("Cannot locate runtime module path");
        auto dll = std::filesystem::path(std::wstring(path.data(), length)).parent_path() / L"neural_fx_ort_123.dll";
        library = LoadLibraryExW(dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) throw std::runtime_error("Could not load bundled neural_fx_ort_123.dll beside the executable/plugin");
        using GetBase = const OrtApiBase* (ORT_API_CALL*)();
        auto function = reinterpret_cast<GetBase>(GetProcAddress(library, "OrtGetApiBase"));
        if (!function) throw std::runtime_error("Bundled runtime has no OrtGetApiBase");
        base = function();
#else
        base = OrtGetApiBase();
#endif
        const auto* api = base->GetApi(ORT_API_VERSION);
        if (!api) throw std::runtime_error("Bundled runtime does not support the compiled API version");
        Ort::InitApi(api);
    }
    ~NativeRuntime() {
#ifdef _WIN32
        if (library) FreeLibrary(library);
#endif
    }
};
NativeRuntime& runtime() { static NativeRuntime instance; return instance; }
Ort::Env createEnvironment() { (void)runtime(); return Ort::Env(ORT_LOGGING_LEVEL_WARNING, "neural-fx"); }
}
const char* runtimeVersion() { return runtime().base->GetVersionString(); }
Processor::Processor(const std::filesystem::path& model) : env_(createEnvironment()) {
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, model.c_str(), options_);
    Ort::AllocatorWithDefaultOptions allocator;
    auto metadata = session_.GetModelMetadata();
    auto rate = metadata.LookupCustomMetadataMapAllocated("neural_fx_sample_rate", allocator);
    if (!rate) throw std::runtime_error("Missing model sample-rate metadata; re-export with the current exporter");
    rate_ = std::stoi(rate.get());
    auto effect = metadata.LookupCustomMetadataMapAllocated("neural_fx_effect", allocator);
    if (effect) effect_ = effect.get();
    auto order = metadata.LookupCustomMetadataMapAllocated("neural_fx_control_order", allocator);
    if (order && std::string(order.get()) != "drive,mix") throw std::runtime_error("Unsupported control order");
    auto mixContract = metadata.LookupCustomMetadataMapAllocated("neural_fx_mix_contract", allocator);
    if (mixContract && std::string(mixContract.get()) != "output_already_mixed") throw std::runtime_error("Unsupported mix contract");
    if (rate_ < 8000 || rate_ > 384000) throw std::runtime_error("Unsupported model sample rate");
    auto createSlots = [&](bool input, std::vector<Slot>& slots, std::vector<Ort::Value>& values, std::vector<const char*>& names) {
        const size_t count = input ? session_.GetInputCount() : session_.GetOutputCount();
        slots.reserve(count); values.reserve(count); names.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            auto name = input ? session_.GetInputNameAllocated(index, allocator) : session_.GetOutputNameAllocated(index, allocator);
            auto info = input ? session_.GetInputTypeInfo(index) : session_.GetOutputTypeInfo(index);
            auto tensor = info.GetTensorTypeAndShapeInfo();
            if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("Only float32 models are supported");
            auto shape = tensor.GetShape();
            size_t elements = 1;
            for (auto dimension : shape) {
                if (dimension < 1 || dimension > 1000000 || elements > 10000000 / static_cast<size_t>(dimension)) throw std::runtime_error("Invalid or dynamic model shape");
                elements *= static_cast<size_t>(dimension);
            }
            slots.push_back({name.get(), shape, std::vector<float>(elements, 0.f)});
        }
        for (auto& slot : slots) {
            names.push_back(slot.name.c_str());
            values.push_back(Ort::Value::CreateTensor<float>(memory_, slot.data.data(), slot.data.size(), slot.shape.data(), slot.shape.size()));
        }
    };
    createSlots(true, inputs_, inputValues_, inputNames_);
    createSlots(false, outputs_, outputValues_, outputNames_);
    auto indexOf = [](const std::vector<Slot>& slots, const std::string& name) {
        for (size_t i = 0; i < slots.size(); ++i) if (slots[i].name == name) return i;
        throw std::runtime_error("Missing model port: " + name);
    };
    audio_ = indexOf(inputs_, "audio"); controls_ = indexOf(inputs_, "controls");
    wet_ = indexOf(outputs_, "wet"); history_ = indexOf(inputs_, "history");
    if (inputs_[audio_].shape.size() != 3 || inputs_[audio_].shape[0] != 1 || inputs_[audio_].shape[1] != 1 || inputs_[controls_].data.size() != 2) throw std::runtime_error("Expected batch-one mono audio and two controls");
    block_ = inputs_[audio_].data.size();
    if (outputs_[wet_].shape != inputs_[audio_].shape) throw std::runtime_error("Audio input/output shape mismatch");
    states_.push_back({history_, indexOf(outputs_, "next_history")});
    hasHidden_ = inputs_.size() == 4;
    if (hasHidden_) { hidden_ = indexOf(inputs_, "hidden"); states_.push_back({hidden_, indexOf(outputs_, "next_hidden")}); }
    if (inputs_.size() != (hasHidden_ ? 4u : 3u) || outputs_.size() != (hasHidden_ ? 3u : 2u)) throw std::runtime_error("Unsupported model port contract");
    for (auto state : states_) if (inputs_[state.first].shape != outputs_[state.second].shape) throw std::runtime_error("State input/output shape mismatch");
    error_.reserve(512);
}

void Processor::reset() noexcept {
    for (auto& slot : inputs_) std::fill(slot.data.begin(), slot.data.end(), 0.f);
    for (auto& slot : outputs_) std::fill(slot.data.begin(), slot.data.end(), 0.f);
    error_.clear();
}

bool Processor::process(const float* input, float* output, float drive, float mix) noexcept {
    if (!std::isfinite(drive) || !std::isfinite(mix) || drive < 0 || drive > 1 || mix < 0 || mix > 1) {
        std::fill_n(output, block_, 0.f); return false;
    }
    std::copy_n(input, block_, inputs_[audio_].data.data());
    inputs_[controls_].data[0] = drive; inputs_[controls_].data[1] = mix;
    try {
        // All tensor backing buffers and Ort::Value handles are prepared before this call.
        session_.Run(runOptions_, inputNames_.data(), inputValues_.data(), inputValues_.size(),
                     outputNames_.data(), outputValues_.data(), outputValues_.size());
        for (const auto& slot : outputs_) for (float value : slot.data) if (!std::isfinite(value)) {
            std::fill_n(output, block_, 0.f); return false;
        }
        std::copy_n(outputs_[wet_].data.data(), block_, output);
        for (auto state : states_) std::copy(outputs_[state.second].data.begin(), outputs_[state.second].data.end(), inputs_[state.first].data.begin());
        return true;
    } catch (const Ort::Exception& exception) {
        error_ = exception.what();
        std::fill_n(output, block_, 0.f); return false;
    }
}
const std::vector<float>& Processor::history() const noexcept { return inputs_[history_].data; }
const std::vector<float>& Processor::hidden() const noexcept { return hasHidden_ ? inputs_[hidden_].data : empty_; }

BufferedProcessor::BufferedProcessor(const std::filesystem::path& model)
    : engine_(model), input_(engine_.blockSize(), 0.f), output_(engine_.blockSize(), 0.f) {}
void BufferedProcessor::reset() noexcept { engine_.reset(); fill_ = 0; std::fill(input_.begin(), input_.end(), 0.f); std::fill(output_.begin(), output_.end(), 0.f); }
bool BufferedProcessor::process(float* audio, size_t samples, float drive, float mix) noexcept {
    bool okay = true;
    for (size_t i = 0; i < samples; ++i) {
        input_[fill_] = audio[i]; audio[i] = output_[fill_];
        if (++fill_ == input_.size()) { okay = engine_.process(input_.data(), output_.data(), drive, mix) && okay; fill_ = 0; }
    }
    return okay;
}
}
