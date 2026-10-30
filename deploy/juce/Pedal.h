#pragma once
#include <juce_dsp/juce_dsp.h>
#include <array>

// A conventional DSP reference effect. This is explicitly separate from the learned model.
// All oversampling buffers are allocated in prepare(), never in process().
class Pedal final {
public:
    void prepare(double rate) {
        sampleRate = rate;
        for (auto& c : channels) {
            c.oversampling.initProcessing(512);
            c.drive.reset(rate * 4, .015); c.input.reset(rate * 4, .015);
            c.mix.reset(rate, .015); c.output.reset(rate, .015);
        }
        reset();
    }
    void reset() noexcept {
        for (auto& c : channels) {
            c.oversampling.reset(); c.dry.fill(0); c.wet.fill(0); c.cursor = 0;
            c.dcInput = c.dcOutput = c.tone1 = c.tone2 = c.preLow = 0; c.initial = true;
        }
    }
    void process(juce::AudioBuffer<float>& buffer, float drive, float mix, float inputDb,
                 float outputDb, float toneHz, int latency, bool british = false) noexcept {
        const float dc = static_cast<float>(std::exp(-2 * juce::MathConstants<double>::pi * 25 / sampleRate));
        const float pre = static_cast<float>(std::exp(-2 * juce::MathConstants<double>::pi * 450 / (sampleRate * 4)));
        const float tone = static_cast<float>(std::exp(-2 * juce::MathConstants<double>::pi * juce::jlimit(800., sampleRate * .45, static_cast<double>(toneHz)) / sampleRate));
        for (int ch = 0; ch < juce::jmin(2, buffer.getNumChannels()); ++ch) {
            auto& c = channels[static_cast<size_t>(ch)];
            const float gain = std::pow(10.f, 36.f * drive / 20.f);
            const float input = juce::Decibels::decibelsToGain(inputDb), output = juce::Decibels::decibelsToGain(outputDb);
            if (c.initial) {
                c.drive.setCurrentAndTargetValue(gain); c.input.setCurrentAndTargetValue(input);
                c.mix.setCurrentAndTargetValue(mix); c.output.setCurrentAndTargetValue(output); c.initial = false;
            }
            c.drive.setTargetValue(gain); c.input.setTargetValue(input);
            c.mix.setTargetValue(mix); c.output.setTargetValue(output);
            const int fullDelay = juce::jlimit(32, 8192, latency);
            const int wetDelay = fullDelay - juce::roundToInt(c.oversampling.getLatencyInSamples());
            for (int start = 0; start < buffer.getNumSamples(); start += 512) {
                const int count = juce::jmin(512, buffer.getNumSamples() - start);
                auto* audio = buffer.getWritePointer(ch) + start;
                std::copy_n(audio, count, c.source.data());
                float* pointers[]{audio};
                juce::dsp::AudioBlock<float> block(pointers, 1, static_cast<size_t>(count));
                auto up = c.oversampling.processSamplesUp(block);
                auto* high = up.getChannelPointer(0);
                for (size_t i = 0; i < up.getNumSamples(); ++i) {
                    float x = high[i] * c.input.getNextValue();
                    // Tight bass into cascaded asymmetric stages; synthetic amp voicing.
                    c.preLow = (1 - pre) * x + pre * c.preLow;
                    const float g = c.drive.getNextValue();
                    if (british) {
                        x = std::tanh((x - .65f * c.preLow) * g + .12f) - std::tanh(.12f);
                        high[i] = .65f * std::tanh((1.5f + 2.5f * drive) * x);
                    } else high[i] = std::tanh(x * g);
                }
                c.oversampling.processSamplesDown(block);
                for (int i = 0; i < count; ++i) {
                    const float cleaned = audio[i] - c.dcInput + dc * c.dcOutput;
                    c.dcInput = audio[i]; c.dcOutput = cleaned;
                    c.tone1 = (1 - tone) * cleaned + tone * c.tone1;
                    c.tone2 = (1 - tone) * c.tone1 + tone * c.tone2;
                    c.dry[c.cursor] = c.source[static_cast<size_t>(i)]; c.wet[c.cursor] = c.tone2;
                    const auto dryIndex = (c.cursor + c.dry.size() - static_cast<size_t>(fullDelay)) % c.dry.size();
                    const auto wetIndex = (c.cursor + c.wet.size() - static_cast<size_t>(wetDelay)) % c.wet.size();
                    const float amount = c.mix.getNextValue();
                    audio[i] = ((1 - amount) * c.dry[dryIndex] + amount * c.wet[wetIndex]) * c.output.getNextValue();
                    c.cursor = (c.cursor + 1) % c.dry.size();
                }
            }
        }
    }
private:
    struct Channel {
        juce::dsp::Oversampling<float> oversampling{1, 2, juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, true, true};
        juce::SmoothedValue<float> drive, input, mix, output;
        std::array<float, 16384> dry{}, wet{};
        std::array<float, 512> source{};
        size_t cursor{}; float dcInput{}, dcOutput{}, tone1{}, tone2{}, preLow{}; bool initial{true};
    };
    std::array<Channel, 2> channels;
    double sampleRate{48000};
};
