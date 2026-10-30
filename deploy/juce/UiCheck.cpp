#include <juce_audio_utils/juce_audio_utils.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "EffectChecks.h"
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();
int main() {
    try {
        juce::ScopedJuceInitialiser_GUI initialise;
        std::unique_ptr<juce::AudioProcessor> plugin(createPluginFilter());
        plugin->setPlayConfigDetails(2, 2, 48000, 512);
        plugin->prepareToPlay(48000, 512);
        std::unique_ptr<juce::AudioProcessorEditor> editor(plugin->createEditorIfNeeded());
        if (!editor) throw std::runtime_error("Missing editor");
        const int block = 113;
        juce::AudioBuffer<float> buffer(2, block); juce::MidiBuffer midi;
        bool auditionChecked = false;
        {
            auto* select = dynamic_cast<juce::ComboBox*>(editor->findChildWithID("sampleSelect"));
            auto* play = dynamic_cast<juce::TextButton*>(editor->findChildWithID("primary"));
            auto* stop = dynamic_cast<juce::TextButton*>(editor->findChildWithID("sampleStop"));
            auto* dry = dynamic_cast<juce::TextButton*>(editor->findChildWithID("dryCompare"));
            auto* loop = dynamic_cast<juce::ToggleButton*>(editor->findChildWithID("sampleLoop"));
            if (!select || !play || !stop || !dry || !loop || select->getNumItems() == 0) throw std::runtime_error("Sample controls/library missing");
            select->setSelectedId(0, juce::dontSendNotification);
            select->setSelectedId(1, juce::sendNotificationSync);
            auto folder = juce::File(juce::SystemStats::getEnvironmentVariable("NEURAL_FX_SAMPLES", ""));
            auto manifest = juce::JSON::parse(folder.getChildFile("manifest.json"));
            auto file = folder.getChildFile(manifest["clips"][0]["dry"].toString());
            juce::AudioFormatManager readers; readers.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(readers.createReaderFor(file));
            if (!reader || reader->numChannels != 1) throw std::runtime_error("Missing mono audition fixture");
            juce::AudioBuffer<float> clip(1, static_cast<int>(reader->lengthInSamples));
            if (!reader->read(&clip, 0, clip.getNumSamples(), 0, true, false)) throw std::runtime_error("Cannot read audition fixture");
            auto render = [&](const std::vector<float>& source) {
                std::vector<float> result(source.size());
                for (size_t pos = 0; pos < source.size(); pos += static_cast<size_t>(block)) {
                    const int count = static_cast<int>(std::min(static_cast<size_t>(block), source.size() - pos));
                    buffer.setSize(2, count, false, false, true);
                    for (int ch = 0; ch < 2; ++ch) std::copy_n(source.data() + pos, count, buffer.getWritePointer(ch));
                    plugin->processBlock(buffer, midi);
                    for (int i = 0; i < count; ++i) {
                        const auto a = buffer.getSample(0, i), b = buffer.getSample(1, i);
                        if (!std::isfinite(a) || std::abs(a - b) > 1e-6f) throw std::runtime_error("Audition stereo mismatch");
                        result[pos + static_cast<size_t>(i)] = a;
                    }
                }
                return result;
            };
            auto compare = [](const std::vector<float>& a, const std::vector<float>& b) {
                if (a.size() != b.size()) throw std::runtime_error("Audition length mismatch");
                for (size_t i = 0; i < a.size(); ++i) if (std::abs(a[i] - b[i]) > 1e-6f) throw std::runtime_error("Audition output differs from live-input path");
            };
            const auto n = static_cast<size_t>(clip.getNumSamples());
            std::vector<float> repeated(n * 2 + 37), silence(repeated.size(), 0);
            for (size_t i = 0; i < repeated.size(); ++i) repeated[i] = clip.getSample(0, static_cast<int>(i % n));
            plugin->reset(); const auto expected = render(repeated);
            plugin->reset(); play->onClick(); compare(expected, render(silence)); stop->onClick();
            dry->setToggleState(true, juce::dontSendNotification); dry->onClick();
            plugin->reset(); play->onClick(); const auto dryOutput = render(silence); stop->onClick();
            const auto latency = static_cast<size_t>(plugin->getLatencySamples());
            for (size_t i = 0; i < dryOutput.size(); ++i) {
                const auto expectedDry = i < latency ? 0.f : repeated[i - latency];
                if (std::abs(dryOutput[i] - expectedDry) > 1e-6f) throw std::runtime_error("Dry comparison is not latency aligned");
            }
            dry->setToggleState(false, juce::dontSendNotification); dry->onClick();
            loop->setToggleState(false, juce::dontSendNotification); loop->onClick();
            std::vector<float> once(n + 512, 0), onceSilence(once.size(), 0);
            std::copy_n(clip.getReadPointer(0), n, once.data());
            plugin->reset(); const auto expectedOnce = render(once);
            plugin->reset(); play->onClick(); compare(expectedOnce, render(onceSilence)); stop->onClick();
            plugin->reset(); compare(expected, render(repeated)); // Stop restores the live-input path.
            loop->setToggleState(true, juce::dontSendNotification); loop->onClick();
            auditionChecked = true;
        }
        auto screenshot = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_SCREENSHOT", "");
        // Rejecting a new selection must not leave the previous neural model active.
        if (juce::SystemStats::getEnvironmentVariable("NEURAL_FX_MODEL", "").isNotEmpty()) {
            juce::MemoryBlock saved; plugin->getStateInformation(saved);
            auto xml = juce::AudioProcessor::getXmlFromBinary(saved.getData(), static_cast<int>(saved.getSize()));
            if (!xml) throw std::runtime_error("Missing model state XML");
            auto tree = juce::ValueTree::fromXml(*xml);
            tree.setProperty("modelPath", juce::File::getCurrentWorkingDirectory().getChildFile("missing-test-model.onnx").getFullPathName(), nullptr);
            juce::MemoryBlock invalid;
            juce::AudioProcessor::copyXmlToBinary(*tree.createXml(), invalid);
            plugin->setStateInformation(invalid.getData(), static_cast<int>(invalid.getSize()));
            for (auto* raw : plugin->getParameters()) if (auto* p = dynamic_cast<juce::RangedAudioParameter*>(raw))
                if (p->paramID == "engine") p->setValueNotifyingHost(0);
            plugin->reset();
            const int latency = plugin->getLatencySamples();
            int offset = 0;
            for (int blockIndex = 0; blockIndex < 8; ++blockIndex) {
                buffer.clear();
                if (blockIndex == 0) for (int ch = 0; ch < 2; ++ch) buffer.setSample(ch, 0, .25f);
                plugin->processBlock(buffer, midi);
                for (int i = 0; i < buffer.getNumSamples(); ++i)
                    if (std::abs(buffer.getSample(0, i) - (offset+i == latency ? .25f : 0.f)) > 1e-6f) throw std::runtime_error("Failed model selection still processes with previous model");
                offset += buffer.getNumSamples();
            }
            plugin->setStateInformation(saved.getData(), static_cast<int>(saved.getSize())); plugin->reset();
        }
        auto effects = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_EFFECT_CHECK", "");
        if (effects.isNotEmpty()) {
            checkGuitarEffect(*plugin, juce::File(juce::SystemStats::getEnvironmentVariable("NEURAL_FX_SAMPLES", "")), juce::File(effects));
            checkRig(*plugin, juce::File(effects));
        }
        auto preset = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_SCREENSHOT_PRESET", "presetLead");
        if (screenshot.isNotEmpty()) {
            auto* button = dynamic_cast<juce::TextButton*>(editor->findChildWithID(preset));
            auto* play = dynamic_cast<juce::TextButton*>(editor->findChildWithID("primary"));
            if (!button || !play) throw std::runtime_error("Screenshot preset unavailable");
            button->onClick(); plugin->reset(); play->onClick();
            for (int i = 0; i < 500; ++i) { buffer.clear(); plugin->processBlock(buffer, midi); }
        }
        juce::Timer::callPendingTimersSynchronously();
        if (screenshot.isNotEmpty()) {
            auto image = editor->createComponentSnapshot(editor->getLocalBounds(), true, 1.f);
            juce::File destination(screenshot); destination.deleteFile();
            juce::FileOutputStream stream(destination); juce::PNGImageFormat png;
            if (!stream.openedOk() || !png.writeImageToStream(image, stream)) throw std::runtime_error("Cannot write editor screenshot");
        }
        std::cout << "{\"latency_samples\":" << plugin->getLatencySamples() << ",\"stereo_consistent\":true,\"reset_reproducible\":true,\"editor_created\":true,\"loop_parity\":true,\"dry_latency_aligned\":true,\"nonloop_parity\":true,\"stop_restores_live_input\":true,\"audition_checked\":" << (auditionChecked ? "true" : "false") << "}\n";
        editor.reset(); plugin->releaseResources(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
