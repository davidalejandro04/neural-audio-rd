#include <juce_audio_utils/juce_audio_utils.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cmath>

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 7) throw std::runtime_error("Usage: nafx-plugin-check plugin.vst3 input.f32 output.f32 host_block [drive mix]");
        juce::ScopedJuceInitialiser_GUI initialise;
        juce::AudioPluginFormatManager manager;
        auto* format = new juce::VST3PluginFormat(); manager.addFormat(format);
        juce::OwnedArray<juce::PluginDescription> descriptions;
        format->findAllTypesForFile(descriptions, juce::String::fromUTF8(argv[1]));
        if (descriptions.isEmpty()) throw std::runtime_error("VST3 scan did not find a processor");
        juce::String error;
        auto plugin = manager.createPluginInstance(*descriptions[0], 48000, 512, error);
        if (!plugin) throw std::runtime_error(error.toStdString());
        plugin->setPlayConfigDetails(2, 2, 48000, 512);
        plugin->prepareToPlay(48000, 512);
        std::unique_ptr<juce::AudioProcessorEditor> editor(plugin->createEditorIfNeeded());
        if (!editor) throw std::runtime_error("Plugin editor could not be created");
        if (argc == 7) {
            const float drive = std::stof(argv[5]), mix = std::stof(argv[6]);
            if (!std::isfinite(drive) || !std::isfinite(mix) || drive < 0 || drive > 1 || mix < 0 || mix > 1) throw std::runtime_error("Invalid normalized controls");
            for (auto* p : plugin->getParameters()) {
                if (p->getName(100) == "Drive") p->setValueNotifyingHost(drive);
                if (p->getName(100) == "Mix") p->setValueNotifyingHost(mix);
            }
        }
        juce::MemoryBlock saved; plugin->getStateInformation(saved);
        plugin->setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
        std::ifstream input(std::filesystem::u8path(argv[2]), std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot read raw input");
        const auto bytes = input.tellg();
        if (bytes <= 0 || static_cast<size_t>(bytes) % sizeof(float)) throw std::runtime_error("Invalid raw float input");
        std::vector<float> x(static_cast<size_t>(bytes) / sizeof(float)), y(x.size());
        input.seekg(0); input.read(reinterpret_cast<char*>(x.data()), bytes);
        const int block = std::stoi(argv[4]); if (block < 1 || block > 512) throw std::runtime_error("Host block must be 1..512");
        juce::AudioBuffer<float> buffer(2, block); juce::MidiBuffer midi;
        for (size_t pos = 0; pos < x.size(); pos += static_cast<size_t>(block)) {
            const int length = static_cast<int>(std::min(static_cast<size_t>(block), x.size() - pos));
            buffer.setSize(2, length, false, false, true);
            for (int ch = 0; ch < 2; ++ch) std::copy_n(x.data() + pos, length, buffer.getWritePointer(ch));
            plugin->processBlock(buffer, midi);
            for (int i = 0; i < length; ++i) {
                const auto a = buffer.getSample(0, i), b = buffer.getSample(1, i);
                if (!std::isfinite(a) || std::abs(a - b) > 1e-6f) throw std::runtime_error("Nonfinite or inconsistent channel output");
                y[pos + static_cast<size_t>(i)] = a;
            }
        }
        plugin->reset();
        for (size_t pos = 0; pos < x.size(); pos += static_cast<size_t>(block)) {
            const int length = static_cast<int>(std::min(static_cast<size_t>(block), x.size() - pos));
            buffer.setSize(2, length, false, false, true);
            for (int ch = 0; ch < 2; ++ch) std::copy_n(x.data() + pos, length, buffer.getWritePointer(ch));
            plugin->processBlock(buffer, midi);
            for (int i = 0; i < length; ++i) if (std::abs(buffer.getSample(0, i) - y[pos + static_cast<size_t>(i)]) > 1e-6f) throw std::runtime_error("Plugin reset did not reproduce initial output");
        }
        std::ofstream output(std::filesystem::u8path(argv[3]), std::ios::binary);
        output.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * sizeof(float)));
        std::cout << "{\"latency_samples\":" << plugin->getLatencySamples() << ",\"samples\":" << y.size() << ",\"stereo_consistent\":true,\"reset_reproducible\":true,\"editor_created\":true}\n";
        editor.reset();
        plugin->releaseResources(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
