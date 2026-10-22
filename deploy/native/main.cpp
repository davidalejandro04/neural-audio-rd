#include "processor.h"
#include "allocation_probe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using Args = std::map<std::string, std::string>;
struct Pacer {
#ifdef _WIN32
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    ~Pacer() { if (timer) CloseHandle(timer); }
#endif
    void wait(Clock::time_point target) {
        const auto remaining = std::chrono::duration<double>(target - Clock::now()).count();
        if (remaining <= 0) return;
#ifdef _WIN32
        if (timer) {
            LARGE_INTEGER due; due.QuadPart = -std::max<LONGLONG>(1, static_cast<LONGLONG>(remaining * 1e7));
            if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) { WaitForSingleObject(timer, INFINITE); return; }
        }
#endif
        std::this_thread::sleep_until(target);
    }
};
Args arguments(int argc, char** argv) {
    Args result;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--paced" || key == "--buffered") result[key] = "1";
        else { if (key.rfind("--", 0) != 0 || i + 1 == argc) throw std::runtime_error("Expected --option value"); result[key] = argv[++i]; }
    }
    return result;
}
std::string required(const Args& a, const std::string& key) { auto it = a.find(key); if (it == a.end()) throw std::runtime_error("Missing " + key); return it->second; }
double number(const Args& a, const std::string& key, double fallback) {
    auto it = a.find(key); if (it == a.end()) return fallback;
    size_t end = 0; double value = std::stod(it->second, &end);
    if (end != it->second.size() || !std::isfinite(value)) throw std::runtime_error("Invalid " + key);
    return value;
}
template <typename T> T read(std::istream& f) { T v{}; if (!f.read(reinterpret_cast<char*>(&v), sizeof(v))) throw std::runtime_error("Truncated input"); return v; }
template <typename T> void write(std::ostream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void readFloats(std::istream& f, std::vector<float>& x) { if (!f.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(x.size() * sizeof(float)))) throw std::runtime_error("Truncated float data"); }
void writeFloats(std::ostream& f, const std::vector<float>& x) { f.write(reinterpret_cast<const char*>(x.data()), static_cast<std::streamsize>(x.size() * sizeof(float))); }
std::ofstream output(const std::string& path, bool binary = false) {
    auto p = std::filesystem::u8path(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p, binary ? std::ios::binary : std::ios::out);
    if (!f) throw std::runtime_error("Cannot write output");
    f << std::setprecision(12); return f;
}
struct Wav { int rate; std::vector<float> samples; };
Wav readWav(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    char text[4]; if (!f.read(text, 4) || std::memcmp(text, "RIFF", 4)) throw std::runtime_error("Expected RIFF WAV");
    (void)read<uint32_t>(f); f.read(text, 4); if (std::memcmp(text, "WAVE", 4)) throw std::runtime_error("Expected WAVE");
    uint16_t format = 0, channels = 0, bits = 0; uint32_t rate = 0;
    std::vector<char> bytes;
    while (f.read(text, 4)) {
        const auto size = read<uint32_t>(f);
        if (size > 512u * 1024u * 1024u) throw std::runtime_error("WAV chunk too large");
        if (!std::memcmp(text, "fmt ", 4)) {
            if (size < 16) throw std::runtime_error("Invalid WAV format chunk");
            format = read<uint16_t>(f); channels = read<uint16_t>(f); rate = read<uint32_t>(f);
            (void)read<uint32_t>(f); (void)read<uint16_t>(f); bits = read<uint16_t>(f);
            f.seekg(static_cast<std::streamoff>(size - 16), std::ios::cur);
        } else if (!std::memcmp(text, "data", 4)) {
            bytes.resize(size); if (!f.read(bytes.data(), size)) throw std::runtime_error("Truncated WAV data");
        } else f.seekg(size, std::ios::cur);
        if (size & 1u) f.seekg(1, std::ios::cur);
    }
    if (channels != 1 || !rate || bytes.empty() || (format != 1 && format != 3)) throw std::runtime_error("Use nonempty mono PCM or float WAV");
    if ((format == 3 && bits != 32) || (format == 1 && bits != 8 && bits != 16 && bits != 24 && bits != 32)) throw std::runtime_error("Unsupported WAV bit depth");
    const size_t stride = bits / 8;
    if (bytes.size() % stride) throw std::runtime_error("Incomplete WAV sample");
    Wav wav{static_cast<int>(rate), std::vector<float>(bytes.size() / stride)};
    for (size_t i = 0; i < wav.samples.size(); ++i) {
        const char* p = bytes.data() + i * stride;
        float v = 0;
        if (format == 3) std::memcpy(&v, p, 4);
        else if (bits == 8) v = (static_cast<unsigned char>(*p) - 128) / 128.f;
        else if (bits == 16) { int16_t value; std::memcpy(&value, p, 2); v = value / 32768.f; }
        else if (bits == 24) { int32_t value = static_cast<unsigned char>(p[0]) | (static_cast<unsigned char>(p[1]) << 8) | (static_cast<unsigned char>(p[2]) << 16); if (value & 0x800000) value |= ~0xffffff; v = value / 8388608.f; }
        else { int32_t value; std::memcpy(&value, p, 4); v = static_cast<float>(value / 2147483648.0); }
        if (!std::isfinite(v)) throw std::runtime_error("WAV contains NaN/Inf");
        wav.samples[i] = v;
    }
    return wav;
}
void writeWav(const std::string& path, const Wav& wav) {
    if (wav.samples.size() > (0xffffffffu - 48u) / 4u) throw std::runtime_error("Output too large for RIFF WAV");
    auto f = output(path, true); uint32_t bytes = static_cast<uint32_t>(wav.samples.size() * 4);
    f.write("RIFF", 4); write<uint32_t>(f, bytes + 48); f.write("WAVEfmt ", 8);
    write<uint32_t>(f, 16); write<uint16_t>(f, 3); write<uint16_t>(f, 1);
    write<uint32_t>(f, wav.rate); write<uint32_t>(f, wav.rate * 4); write<uint16_t>(f, 4); write<uint16_t>(f, 32);
    f.write("fact", 4); write<uint32_t>(f, 4); write<uint32_t>(f, static_cast<uint32_t>(wav.samples.size()));
    f.write("data", 4); write<uint32_t>(f, bytes); writeFloats(f, wav.samples);
    if (!f) throw std::runtime_error("WAV write failed");
}
void render(const Args& args) {
    const auto model = std::filesystem::u8path(required(args, "--model"));
    auto wav = readWav(required(args, "--input"));
    const float drive = static_cast<float>(number(args, "--drive", 0.5)), mix = static_cast<float>(number(args, "--mix", 1.0));
    nfx::Processor processor(model);
    if (processor.sampleRate() != wav.rate) throw std::runtime_error("Input sample rate differs from the model");
    if (args.count("--buffered")) {
        nfx::BufferedProcessor buffered(model);
        const auto chunk = static_cast<size_t>(number(args, "--host-block", 113));
        if (!chunk || chunk > 65536) throw std::runtime_error("Invalid host block");
        for (size_t pos = 0; pos < wav.samples.size(); pos += chunk)
            if (!buffered.process(wav.samples.data() + pos, std::min(chunk, wav.samples.size() - pos), drive, mix)) throw std::runtime_error("Buffered processing failed");
    } else {
        std::vector<float> input(processor.blockSize(), 0), wet(processor.blockSize(), 0);
        for (size_t pos = 0; pos < wav.samples.size(); pos += input.size()) {
            std::fill(input.begin(), input.end(), 0.f);
            const auto length = std::min(input.size(), wav.samples.size() - pos);
            std::copy_n(wav.samples.data() + pos, length, input.data());
            if (!processor.process(input.data(), wet.data(), drive, mix)) throw std::runtime_error("Model processing failed: " + processor.error());
            std::copy_n(wet.data(), length, wav.samples.data() + pos);
        }
    }
    writeWav(required(args, "--output"), wav);
    std::cout << "{\"samples\":" << wav.samples.size() << ",\"sample_rate\":" << wav.rate << "}\n";
}
void trace(const Args& args) {
    nfx::Processor processor(std::filesystem::u8path(required(args, "--model")));
    std::ifstream f(std::filesystem::u8path(required(args, "--input")), std::ios::binary);
    char magic[8]; if (!f.read(magic, 8) || std::memcmp(magic, "NFXTRACE", 8)) throw std::runtime_error("Invalid trace header");
    auto count = read<uint32_t>(f), block = read<uint32_t>(f);
    if (!count || count > 1000000 || block != processor.blockSize()) throw std::runtime_error("Invalid trace dimensions");
    auto out = output(required(args, "--output"), true);
    out.write("NFXOUT01", 8); write(out, count); write(out, block);
    write<uint32_t>(out, static_cast<uint32_t>(processor.history().size()));
    write<uint32_t>(out, static_cast<uint32_t>(processor.hidden().size()));
    std::vector<float> x(block), y(block);
    for (uint32_t i = 0; i < count; ++i) {
        if (read<uint32_t>(f)) processor.reset();
        float drive = read<float>(f), mix = read<float>(f); readFloats(f, x);
        if (!processor.process(x.data(), y.data(), drive, mix)) throw std::runtime_error("Trace processing failed");
        writeFloats(out, y); writeFloats(out, processor.history()); writeFloats(out, processor.hidden());
    }
    if (f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing trace bytes");
    std::cout << "{\"blocks\":" << count << "}\n";
}
double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end()); double index = (values.size() - 1) * q;
    size_t left = static_cast<size_t>(index), right = std::min(left + 1, values.size() - 1);
    return values[left] + (values[right] - values[left]) * (index - left);
}
void benchmark(const Args& args) {
    nfx::Processor processor(std::filesystem::u8path(required(args, "--model")));
    const double seconds = number(args, "--seconds", 0), requested = number(args, "--iterations", 2000);
    const bool paced = args.count("--paced") != 0;
    if (seconds < 0 || seconds > 3600 || requested < 20 || requested > 10000000 || (seconds > 0 && !paced)) throw std::runtime_error("Use 20+ iterations, or --seconds 1..3600 --paced");
    const double period = static_cast<double>(processor.blockSize()) / processor.sampleRate();
    const size_t capacity = seconds > 0 ? static_cast<size_t>(std::ceil(seconds / period)) + 100 : static_cast<size_t>(requested);
    std::vector<double> elapsed; elapsed.reserve(capacity);
    std::vector<float> input(processor.blockSize()), wet(processor.blockSize());
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.15f * std::sin(static_cast<float>(i) * 0.17f);
    for (int i = 0; i < 20; ++i) if (!processor.process(input.data(), wet.data(), 0.5f, 1.f)) throw std::runtime_error("Warmup failed");
    auto wallStart = Clock::now(), scheduled = wallStart;
    Pacer pacer;
    size_t misses = 0, schedulingMisses = 0; double total = 0;
    nfx::allocationCount = 0;
    while (elapsed.size() < capacity) {
        if (seconds > 0 && std::chrono::duration<double>(Clock::now() - wallStart).count() >= seconds) break;
        if (paced) pacer.wait(scheduled);
        const auto start = Clock::now();
        nfx::trackAllocations = true;
        const bool okay = processor.process(input.data(), wet.data(), 0.5f, 1.f);
        nfx::trackAllocations = false;
        const auto end = Clock::now();
        if (!okay) throw std::runtime_error("Benchmark processing failed");
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed.push_back(ms); total += ms;
        misses += ms > period * 1000;
        if (paced) schedulingMisses += std::chrono::duration<double>(end - scheduled).count() > period;
        scheduled += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period));
    }
    const double wall = std::chrono::duration<double>(Clock::now() - wallStart).count();
    const double p99 = quantile(elapsed, 0.99);
    auto report = output(required(args, "--output"));
    report << "{\"runtime\":\"native ONNX Runtime CPU\",\"onnxruntime\":\"" << nfx::runtimeVersion()
           << "\",\"sample_rate\":" << processor.sampleRate() << ",\"block_size\":" << processor.blockSize()
           << ",\"iterations\":" << elapsed.size() << ",\"paced\":" << (paced ? "true" : "false")
           << ",\"wall_seconds\":" << wall << ",\"simulated_audio_seconds\":" << elapsed.size() * period
           << ",\"mean_ms\":" << total / elapsed.size() << ",\"p95_ms\":" << quantile(elapsed, 0.95)
           << ",\"p99_ms\":" << p99 << ",\"worst_ms\":" << *std::max_element(elapsed.begin(), elapsed.end())
           << ",\"deadline_ms\":" << period * 1000 << ",\"p99_budget_fraction\":" << p99 / (period * 1000)
           << ",\"processing_deadline_misses\":" << misses << ",\"scheduling_deadline_misses\":" << schedulingMisses
           << ",\"application_new_calls\":" << nfx::allocationCount
           << ",\"allocation_scope\":\"executable operator new/new[] including aligned allocations; excludes ONNX Runtime DLL internals and malloc\""
           << ",\"screening_passed\":" << (p99 <= period * 700 && !misses && !nfx::allocationCount ? "true" : "false")
           << ",\"sustained_scheduling_passed\":" << (paced && wall >= 600 && !misses && !schedulingMisses && !nfx::allocationCount && p99 <= period * 700 ? "true" : "false")
           << ",\"production_ready\":false}\n";
    std::cout << "{\"measured_blocks\":" << elapsed.size() << ",\"p99_ms\":" << p99 << ",\"processing_misses\":" << misses << ",\"scheduling_misses\":" << schedulingMisses << "}\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: nafx-render render|trace|benchmark --model file.onnx [options]");
        auto args = arguments(argc, argv); std::string command = argv[1];
        if (command == "render") render(args);
        else if (command == "trace") trace(args);
        else if (command == "benchmark") benchmark(args);
        else throw std::runtime_error("Unknown command");
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 2; }
}
