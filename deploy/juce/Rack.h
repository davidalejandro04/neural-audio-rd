#pragma once
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <vector>

// Algorithmic cabinet voicings, not measured IRs. All storage is prepared off
// the audio thread. Independent channel state also supports mono host layouts.
class GuitarRack final {
public:
    struct Settings {
        int cabinet{};
        float bass{}, middle{}, treble{}, delayMs{320}, feedback{.3f}, delayMix{}, reverbMix{}, room{.45f};
    };
    void prepare(double rate) {
        sampleRate = rate;
        for (auto& c : channels) {
            c.delay.assign(static_cast<size_t>(rate * 2) + 2, 0);
            c.reverb.setSampleRate(rate);
            for (auto* s : {&c.delayTime, &c.feedback, &c.delayMix, &c.reverbMix}) s->reset(rate, .04);
            for (auto& s : c.cabMix) s.reset(rate, .025);
            const float cuts[]{5200, 4200, 6500}, mids[]{1800, 1400, 2100};
            for (size_t i = 0; i < 3; ++i) {
                c.cabs[i][0].set(Coefficients::makeHighPass(rate, 80.f, .707f));
                c.cabs[i][1].set(Coefficients::makePeakFilter(rate, mids[i], .8f, juce::Decibels::decibelsToGain(i == 1 ? 3.f : 1.5f)));
                c.cabs[i][2].set(Coefficients::makeLowPass(rate, juce::jmin(cuts[i], static_cast<float>(rate * .4)), .707f));
                c.cabs[i][3].set(Coefficients::makeLowPass(rate, juce::jmin(cuts[i], static_cast<float>(rate * .4)), .707f));
            }
        }
        reset();
    }
    void reset() noexcept {
        for (auto& c : channels) {
            std::fill(c.delay.begin(), c.delay.end(), 0.f); c.cursor = 0; c.damped = 0; c.initial = true;
            c.reverb.reset();
            for (auto& f : c.eq) f.reset();
            for (auto& cab : c.cabs) for (auto& f : cab) f.reset();
        }
    }
    void process(juce::AudioBuffer<float>& audio, const Settings& s) noexcept {
        const float damping = static_cast<float>(std::exp(-2 * juce::MathConstants<double>::pi * 3500 / sampleRate));
        for (int ch = 0; ch < juce::jmin(2, audio.getNumChannels()); ++ch) {
            auto& c = channels[static_cast<size_t>(ch)];
            c.eq[0].set(Coefficients::makeLowShelf(sampleRate, 180.f, .707f, juce::Decibels::decibelsToGain(s.bass)));
            c.eq[1].set(Coefficients::makePeakFilter(sampleRate, 850.f, .75f, juce::Decibels::decibelsToGain(s.middle)));
            c.eq[2].set(Coefficients::makeHighShelf(sampleRate, juce::jmin(3200.f, static_cast<float>(sampleRate * .4)), .707f, juce::Decibels::decibelsToGain(s.treble)));
            auto target = [&](juce::SmoothedValue<float>& value, float next) {
                if (c.initial) value.setCurrentAndTargetValue(next); else value.setTargetValue(next);
            };
            target(c.delayTime, s.delayMs * static_cast<float>(sampleRate) / 1000);
            target(c.feedback, s.feedback); target(c.delayMix, s.delayMix); target(c.reverbMix, s.reverbMix);
            for (int i = 0; i < 4; ++i) target(c.cabMix[static_cast<size_t>(i)], s.cabinet == i ? 1.f : 0.f);
            juce::Reverb::Parameters rp;
            rp.roomSize = s.room; rp.damping = .65f; rp.wetLevel = 1; rp.dryLevel = 0; rp.width = 0; rp.freezeMode = 0;
            c.reverb.setParameters(rp); c.initial = false;
            auto* data = audio.getWritePointer(ch);
            for (int i = 0; i < audio.getNumSamples(); ++i) {
                float dry = data[i];
                for (auto& f : c.eq) dry = f.process(dry);
                float cabinet = dry * c.cabMix[0].getNextValue();
                for (size_t j = 0; j < 3; ++j) {
                    float filtered = dry;
                    for (auto& f : c.cabs[j]) filtered = f.process(filtered);
                    cabinet += filtered * c.cabMix[j + 1].getNextValue();
                }
                const float delay = c.delayTime.getNextValue();
                const auto whole = static_cast<size_t>(delay); const float fraction = delay - static_cast<float>(whole);
                const auto a = (c.cursor + c.delay.size() - whole) % c.delay.size();
                const auto b = (a + c.delay.size() - 1) % c.delay.size();
                const float echo = c.delay[a] + fraction * (c.delay[b] - c.delay[a]);
                c.damped = (1 - damping) * echo + damping * c.damped;
                c.delay[c.cursor] = cabinet + c.feedback.getNextValue() * c.damped;
                c.cursor = (c.cursor + 1) % c.delay.size();
                const float withDelay = cabinet + c.delayMix.getNextValue() * echo;
                float room = withDelay; c.reverb.processMono(&room, 1);
                data[i] = withDelay + c.reverbMix.getNextValue() * room;
            }
        }
    }
private:
    using Coefficients = juce::dsp::IIR::ArrayCoefficients<float>;
    struct Filter {
        std::array<float, 5> v{1, 0, 0, 0, 0}; float z1{}, z2{};
        void set(const std::array<float, 6>& a) noexcept { v = {a[0]/a[3], a[1]/a[3], a[2]/a[3], a[4]/a[3], a[5]/a[3]}; }
        void reset() noexcept { z1 = z2 = 0; }
        float process(float x) noexcept { const float y = v[0] * x + z1; z1 = v[1]*x - v[3]*y + z2; z2 = v[2]*x - v[4]*y; return y; }
    };
    struct Channel {
        std::array<Filter, 3> eq;
        std::array<std::array<Filter, 4>, 3> cabs;
        std::array<juce::SmoothedValue<float>, 4> cabMix;
        juce::SmoothedValue<float> delayTime, feedback, delayMix, reverbMix;
        std::vector<float> delay; size_t cursor{}; float damped{}; bool initial{true}; juce::Reverb reverb;
    };
    std::array<Channel, 2> channels;
    double sampleRate{48000};
};
