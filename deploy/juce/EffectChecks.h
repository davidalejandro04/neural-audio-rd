#pragma once
#include "../native/allocation_probe.h"
#include <fstream>
#include <chrono>

inline void checkGuitarEffect(juce::AudioProcessor& processor, const juce::File& samples, const juce::File& output) {
    if (output.createDirectory().failed()) throw std::runtime_error("Cannot create effect-check folder");
    auto set = [&](const juce::String& name, float value) {
        for (auto* raw : processor.getParameters()) if (auto* p = dynamic_cast<juce::RangedAudioParameter*>(raw))
            if (p->paramID == name) { p->setValueNotifyingHost(p->convertTo0to1(value)); return; }
        throw std::runtime_error("Missing effect parameter");
    };
    auto preset = [&](float drive, float input, float trim, float tone = 5500) {
        set("engine", 1); set("drive", drive); set("inputDb", input); set("outputDb", trim); set("tone", tone); set("mix", 1);
        for (auto id : {"cabinet", "bass", "middle", "treble", "delayMix", "reverbMix"}) set(id, 0);
    };
    std::vector<double> times;
    std::uint64_t allocations = 0;
    auto render = [&](const std::vector<float>& source, int hostBlock, bool bypass = false) {
        processor.reset();
        juce::AudioBuffer<float> audio(2, hostBlock); juce::MidiBuffer midi;
        std::vector<float> result(source.size());
        for (size_t pos = 0; pos < source.size(); pos += static_cast<size_t>(hostBlock)) {
            const int count = static_cast<int>(std::min(source.size() - pos, static_cast<size_t>(hostBlock)));
            audio.setSize(2, count, false, false, true);
            for (int ch = 0; ch < 2; ++ch) std::copy_n(source.data() + pos, count, audio.getWritePointer(ch));
            nfx::allocationCount = 0; nfx::trackAllocations = true;
            const auto begin = std::chrono::steady_clock::now();
            if (bypass) processor.processBlockBypassed(audio, midi); else processor.processBlock(audio, midi);
            const auto end = std::chrono::steady_clock::now();
            nfx::trackAllocations = false; allocations += nfx::allocationCount;
            if (count == hostBlock && hostBlock == 128) times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            for (int i = 0; i < count; ++i) {
                const float a = audio.getSample(0, i), b = audio.getSample(1, i);
                if (!std::isfinite(a) || std::abs(a - b) > 2e-6f) throw std::runtime_error("Effect output nonfinite or stereo mismatch");
                result[pos + static_cast<size_t>(i)] = a;
            }
        }
        return result;
    };
    auto save = [&](const juce::String& name, const std::vector<float>& values) {
        std::ofstream f(std::filesystem::u8path(output.getChildFile(name + ".f32").getFullPathName().toStdString()), std::ios::binary);
        f.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!f) throw std::runtime_error("Could not write effect fixture");
    };
    const auto latency = static_cast<size_t>(processor.getLatencySamples());
    float maxBlockError = 0;
    auto manifest = juce::JSON::parse(samples.getChildFile("manifest.json"));
    auto* entries = manifest["clips"].getArray();
    if (!entries) throw std::runtime_error("Missing sample manifest");
    juce::AudioFormatManager readers; readers.registerBasicFormats();
    for (const auto& entry : *entries) {
        const auto file = samples.getChildFile(entry["dry"].toString());
        std::unique_ptr<juce::AudioFormatReader> reader(readers.createReaderFor(file));
        if (!reader || reader->sampleRate != 48000 || reader->numChannels != 1) throw std::runtime_error("Need mono 48 kHz guitar fixtures");
        juce::AudioBuffer<float> audio(1, static_cast<int>(reader->lengthInSamples));
        if (!reader->read(&audio, 0, audio.getNumSamples(), 0, true, false)) throw std::runtime_error("Cannot read guitar fixture");
        std::vector<float> source(static_cast<size_t>(audio.getNumSamples()) + latency, 0);
        std::copy_n(audio.getReadPointer(0), audio.getNumSamples(), source.data());
        const auto name = file.getFileNameWithoutExtension().replace("-dry", "");
        save(name + "-dry", source);
        preset(.1f, 6, -9); save(name + "-light", render(source, 128));
        preset(.65f, 6, -9); const auto crunch = render(source, 128); save(name + "-crunch", crunch);
        for (int block : {17, 64, 113, 256, 512}) {
            const auto actual = render(source, block);
            for (size_t i = 0; i < actual.size(); ++i) maxBlockError = juce::jmax(maxBlockError, std::abs(actual[i] - crunch[i]));
        }
        preset(.9f, 12, -12, 4500); save(name + "-heavy", render(source, 128));
        const auto dry = render(source, 113, true);
        set("mix", 0); set("outputDb", 0); const auto mixZero = render(source, 113);
        for (size_t i = 0; i < source.size(); ++i) {
            const float expected = i < latency ? 0.f : source[i - latency];
            if (std::abs(dry[i] - expected) > 2e-6f || std::abs(mixZero[i] - expected) > 2e-6f) throw std::runtime_error("Bypass/Mix=0 is not the delayed original dry sample");
        }
        // Real DI listening renders through the full rack; leave two seconds
        // of input silence so the exported WAV includes delay/reverb decay.
        source.resize(source.size() + 96000, 0);
        preset(.88f, 6, -9, 6000); set("engine", 2); set("cabinet", 1); set("bass", -1); set("middle", 3); set("treble", 1);
        set("delayMs", 340); set("feedback", .32f); set("delayMix", .18f); set("reverbMix", .12f); set("room", .45f);
        save(name + "-british-lead", render(source, 128));
        set("engine", 0); set("drive", .85f); save(name + "-neural-lead", render(source, 128));
    }
    if (maxBlockError > 2e-6f) throw std::runtime_error("Effect depends on host block size");
    std::vector<float> sine(48000 + latency, 0);
    for (size_t i = 0; i < 48000; ++i) sine[i] = .08f * std::sin(static_cast<float>(2 * juce::MathConstants<double>::pi * 1000 * i / 48000));
    preset(.1f, 6, -9); save("sine-light", render(sine, 128));
    preset(.9f, 6, -9); save("sine-heavy", render(sine, 128));
    preset(.65f, 6, -9); const auto quiet = render(std::vector<float>(2048, 0), 128);
    for (auto value : quiet) if (value != 0) throw std::runtime_error("Distortion generates noise on silence");
    if (allocations) throw std::runtime_error("Application new calls in pedal processing");
    std::sort(times.begin(), times.end());
    auto json = std::make_unique<juce::DynamicObject>();
    json->setProperty("latency_samples", static_cast<int>(latency)); json->setProperty("block_max_error", maxBlockError);
    json->setProperty("application_new_calls", static_cast<juce::int64>(allocations));
    json->setProperty("bypass_and_mix_zero_exact", true); json->setProperty("silence_exact", true);
    json->setProperty("stereo_consistent", true); json->setProperty("p99_stereo_128_ms", times[times.size() * 99 / 100]);
    json->setProperty("allocation_scope", "Harness C++ operator new/new[]; excludes malloc, external DLL internals and device hosting");
    output.getChildFile("processor-checks.json").replaceWithText(juce::JSON::toString(juce::var(json.release())));
}

inline void checkRig(juce::AudioProcessor& processor, const juce::File& output) {
    auto set = [&](const char* id, float value) {
        for (auto* raw : processor.getParameters()) if (auto* p = dynamic_cast<juce::RangedAudioParameter*>(raw))
            if (p->paramID == id) { p->setValueNotifyingHost(p->convertTo0to1(value)); return; }
        throw std::runtime_error("Missing rack parameter");
    };
    std::uint64_t allocations = 0; float blockError = 0;
    std::vector<float> source(96000, 0);
    for (size_t i = 0; i < 12000; ++i) source[i] = .08f * std::sin(static_cast<float>(2 * juce::MathConstants<double>::pi * 220 * i / 48000));
    auto render = [&](int block, bool bypass = false) {
        processor.reset();
        juce::AudioBuffer<float> buffer(2, block); juce::MidiBuffer midi;
        std::vector<float> result(source.size());
        for (size_t pos = 0; pos < source.size(); pos += static_cast<size_t>(block)) {
            const int count = static_cast<int>(std::min(static_cast<size_t>(block), source.size() - pos));
            buffer.setSize(2, count, false, false, true);
            for (int ch = 0; ch < 2; ++ch) std::copy_n(source.data() + pos, count, buffer.getWritePointer(ch));
            nfx::allocationCount = 0; nfx::trackAllocations = true;
            if (bypass) processor.processBlockBypassed(buffer, midi); else processor.processBlock(buffer, midi);
            nfx::trackAllocations = false; allocations += nfx::allocationCount;
            for (int i = 0; i < count; ++i) {
                const float a = buffer.getSample(0, i), b = buffer.getSample(1, i);
                if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > 2e-6f || std::abs(a) > 2) throw std::runtime_error("Invalid rack output");
                result[pos + static_cast<size_t>(i)] = a;
            }
        }
        return result;
    };
    auto difference = [](const auto& a, const auto& b) {
        float error = 0; for (size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(a[i] - b[i])); return error;
    };
    set("engine", 2); set("drive", .85f); set("inputDb", 6); set("outputDb", -9); set("mix", 1);
    set("bass", -1); set("middle", 3); set("treble", 1); set("delayMix", 0); set("reverbMix", 0);
    set("cabinet", 0); const auto direct = render(128);
    for (int cab = 1; cab <= 3; ++cab) {
        set("cabinet", static_cast<float>(cab)); const auto reference = render(128);
        if (difference(reference, direct) < .001f) throw std::runtime_error("Cabinet has no audible signal effect");
        for (int block : {17, 113, 512}) blockError = std::max(blockError, difference(reference, render(block)));
    }
    set("cabinet", 1); set("delayMs", 320); set("feedback", .4f); set("delayMix", .25f); set("reverbMix", .15f); set("room", .5f);
    const auto lead = render(128);
    for (int block : {17, 113, 512}) blockError = std::max(blockError, difference(lead, render(block)));
    double tailEnergy = 0; for (size_t i = 24000; i < lead.size(); ++i) tailEnergy += lead[i] * lead[i];
    if (tailEnergy < .001) throw std::runtime_error("Delay/reverb tail missing");
    const auto dry = render(113, true); const auto latency = static_cast<size_t>(processor.getLatencySamples());
    for (size_t i = 0; i < source.size(); ++i) if (std::abs(dry[i] - (i < latency ? 0.f : source[i-latency])) > 1e-6f) throw std::runtime_error("Rack bypass not exact");
    // Toggle bypass on a running, non-silent stream: no previous wet block or
    // reverb tail may leak into the latency-aligned original signal.
    processor.reset();
    juce::AudioBuffer<float> transition(2, 113); juce::MidiBuffer transitionMidi;
    for (int blockIndex = 0; blockIndex < 40; ++blockIndex) {
        const int offset = blockIndex * 113;
        for (int ch = 0; ch < 2; ++ch) std::copy_n(source.data() + offset, 113, transition.getWritePointer(ch));
        if (blockIndex < 20) processor.processBlock(transition, transitionMidi); else processor.processBlockBypassed(transition, transitionMidi);
        if (blockIndex >= 20) for (int i = 0; i < 113; ++i)
            if (std::abs(transition.getSample(0, i) - source[static_cast<size_t>(offset+i)-latency]) > 1e-6f) throw std::runtime_error("Wet signal leaks across live bypass transition");
    }
    juce::MemoryBlock state; processor.getStateInformation(state);
    set("cabinet", 0); set("delayMix", 0); processor.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    blockError = std::max(blockError, difference(lead, render(128)));
    set("mix", 0); set("outputDb", 0);
    if (difference(dry, render(113)) > 1e-6f) throw std::runtime_error("Rack mix zero not dry");
    // Mono and high sample rates use the same prepared chain and bounded delay.
    for (double rate : {44100., 96000.}) {
        processor.releaseResources(); processor.setPlayConfigDetails(1, 1, rate, 512); processor.prepareToPlay(rate, 512);
        set("mix", 1); set("delayMs", 1000); set("feedback", .65f); set("reverbMix", .5f); set("room", .85f);
        juce::AudioBuffer<float> mono(1, 512); juce::MidiBuffer midi;
        for (int block = 0; block < 100; ++block) {
            mono.clear(); if (block == 0) mono.setSample(0, 0, .5f); processor.processBlock(mono, midi);
            for (int i = 0; i < mono.getNumSamples(); ++i) if (!std::isfinite(mono.getSample(0, i))) throw std::runtime_error("Mono/rate check failed");
        }
    }
    processor.releaseResources(); processor.setPlayConfigDetails(2, 2, 48000, 512); processor.prepareToPlay(48000, 512);
    if (blockError > 5e-6f || allocations) throw std::runtime_error("Rack block/state parity or callback allocation check failed");
    auto json = std::make_unique<juce::DynamicObject>();
    json->setProperty("block_and_state_max_error", blockError); json->setProperty("application_new_calls", static_cast<juce::int64>(allocations));
    json->setProperty("tail_energy", tailEnergy); json->setProperty("cabinet_changes_signal", true); json->setProperty("bypass_mix_zero_exact", true);
    json->setProperty("mono_44100_96000_finite", true); json->setProperty("scope", "DSP rig; neural parity is tested separately by native-validate");
    output.getChildFile("rack-checks.json").replaceWithText(juce::JSON::toString(juce::var(json.release())));
}
