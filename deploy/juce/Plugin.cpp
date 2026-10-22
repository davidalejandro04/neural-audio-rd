#include <juce_audio_utils/juce_audio_utils.h>
#include "processor.h"
#include <atomic>
#include <memory>
#include <vector>

class NeuralAudioFX final : public juce::AudioProcessor {
public:
    NeuralAudioFX() : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                                       .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
        parameters(*this, nullptr, "NeuralFX", makeParameters()) {
        drive = parameters.getRawParameterValue("drive"); mix = parameters.getRawParameterValue("mix");
        modelPath = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_MODEL", "");
    }
    static juce::AudioProcessorValueTreeState::ParameterLayout makeParameters() {
        juce::AudioProcessorValueTreeState::ParameterLayout p;
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("drive", 1), "Drive", 0.f, 1.f, 0.5f));
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("mix", 1), "Mix", 0.f, 1.f, 1.f));
        return p;
    }
    const juce::String getName() const override { return "NeuralAudioFX"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}
    bool isBusesLayoutSupported(const BusesLayout& layout) const override {
        const auto channels = layout.getMainOutputChannelSet();
        return (channels == juce::AudioChannelSet::mono() || channels == juce::AudioChannelSet::stereo()) && channels == layout.getMainInputChannelSet();
    }
    void prepareToPlay(double rate, int) override {
        const juce::ScopedLock lock(controlLock);
        active.store(nullptr, std::memory_order_release);
        retained.clear(); sampleRate = rate; prepared = true;
        setLatencySamples(0);
        loadLocked();
    }
    void releaseResources() override {
        const juce::ScopedLock lock(controlLock);
        active.store(nullptr, std::memory_order_release); retained.clear(); prepared = false;
    }
    void reset() override {
        // JUCE calls this outside active rendering; both channels must reset together.
        if (auto* engines = active.load(std::memory_order_acquire)) for (auto& channel : engines->channels) channel->reset();
    }
    void processBlock(juce::AudioBuffer<float>& audio, juce::MidiBuffer&) override { process(audio, mix->load()); }
    void processBlockBypassed(juce::AudioBuffer<float>& audio, juce::MidiBuffer&) override { process(audio, 0.f); }
    void process(juce::AudioBuffer<float>& audio, float amount) noexcept {
        juce::ScopedNoDenormals denormals;
        auto* engines = active.load(std::memory_order_acquire);
        if (!engines) return;  // Transparent when no compatible model is loaded.
        if (engines->fault.load()) { audio.clear(); return; }
        const auto gain = drive->load();
        for (int channel = 0; channel < audio.getNumChannels(); ++channel) {
            if (channel > 1 || !engines->channels[static_cast<size_t>(channel)]->process(audio.getWritePointer(channel), static_cast<size_t>(audio.getNumSamples()), gain, amount)) {
                engines->fault.store(true); audio.clear(); return;
            }
        }
    }
    void chooseModel(const juce::String& path) {
        const juce::ScopedLock lock(controlLock);
        modelPath = path;
        if (prepared) loadLocked(); else status = "Model selected";
    }
    void applyReferencePreset(const juce::File& file) {
        const auto data = juce::JSON::parse(file);
        if (!data.hasProperty("drive") || !data.hasProperty("mix") || !data.hasProperty("sample_rate")) return;
        const float a = static_cast<float>(data["drive"]), b = static_cast<float>(data["mix"]);
        if (!std::isfinite(a) || !std::isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1) return;
        { const juce::ScopedLock lock(controlLock);
          if (std::abs(static_cast<double>(data["sample_rate"]) - sampleRate) > 0.5) { status = "Reference and session sample rates differ"; return; }
        }
        for (auto pair : {std::make_pair("drive", a), std::make_pair("mix", b)}) {
            auto* parameter = parameters.getParameter(pair.first);
            parameter->beginChangeGesture(); parameter->setValueNotifyingHost(pair.second); parameter->endChangeGesture();
        }
    }
    juce::String modelStatus() const {
        const juce::ScopedLock lock(controlLock);
        if (auto* engines = active.load(std::memory_order_acquire)) if (engines->fault.load()) return "Processing stopped; reload model";
        return status;
    }
    void getStateInformation(juce::MemoryBlock& data) override {
        auto tree = parameters.copyState();
        { const juce::ScopedLock lock(controlLock); tree.setProperty("modelPath", modelPath, nullptr); }
        if (auto xml = tree.createXml()) copyXmlToBinary(*xml, data);
    }
    void setStateInformation(const void* data, int size) override {
        if (auto xml = getXmlFromBinary(data, size)) {
            auto tree = juce::ValueTree::fromXml(*xml);
            if (tree.hasType(parameters.state.getType())) { parameters.replaceState(tree); chooseModel(tree.getProperty("modelPath").toString()); }
        }
    }
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;
    juce::AudioProcessorValueTreeState parameters;
private:
    struct Engines { std::unique_ptr<nfx::BufferedProcessor> channels[2]; std::atomic<bool> fault{false}; };
    void loadLocked() {
        if (modelPath.isEmpty()) { active.store(nullptr, std::memory_order_release); status = "Choose a model"; setLatencySamples(0); return; }
        if (retained.size() >= 32) { status = "Reload the plug-in before selecting more models"; return; }
        try {
            auto engines = std::make_unique<Engines>();
            const auto path = std::filesystem::u8path(modelPath.toStdString());
            for (auto& channel : engines->channels) channel = std::make_unique<nfx::BufferedProcessor>(path);
            if (std::abs(engines->channels[0]->sampleRate() - sampleRate) > 0.5) { status = "Set the session to the model's sample rate"; return; }
            auto* pointer = engines.get();
            const auto latency = static_cast<int>(pointer->channels[0]->latency());
            retained.push_back(std::move(engines));
            // Retain previous engines until releaseResources: no deletion or refcounting on the audio thread.
            active.store(pointer, std::memory_order_release);
            setLatencySamples(latency); status = juce::File(modelPath).getFileName() + " ready";
        } catch (const std::exception& error) { status = "Could not load model: " + juce::String(error.what()); }
    }
    mutable juce::CriticalSection controlLock;
    juce::String modelPath, status{"Choose a model"};
    double sampleRate{48000}; bool prepared{false};
    std::atomic<float>* drive{}; std::atomic<float>* mix{};
    std::atomic<Engines*> active{nullptr};
    std::vector<std::unique_ptr<Engines>> retained;
};

class Editor final : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit Editor(NeuralAudioFX& p) : AudioProcessorEditor(p), processor(p), controls(p), load("Choose model"), reference("Load reference preset") {
        addAndMakeVisible(controls); addAndMakeVisible(load); addAndMakeVisible(reference); addAndMakeVisible(status);
        const juce::Component::SafePointer<Editor> safe(this);
        load.onClick = [safe] {
            if (!safe) return;
            safe->chooser = std::make_unique<juce::FileChooser>("Choose neural audio model", juce::File{}, "*.onnx");
            safe->chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                [safe](const juce::FileChooser& f) {
                    if (safe && f.getResult().existsAsFile()) safe->processor.chooseModel(f.getResult().getFullPathName());
                });
        };
        reference.onClick = [safe] {
            if (!safe) return;
            safe->chooser = std::make_unique<juce::FileChooser>("Choose reference preset", juce::File{}, "*.json");
            safe->chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                [safe](const juce::FileChooser& f) { if (safe && f.getResult().existsAsFile()) safe->processor.applyReferencePreset(f.getResult()); });
        };
        setSize(480, 250); startTimerHz(4);
    }
    void resized() override {
        auto area = getLocalBounds().reduced(12);
        auto buttons = area.removeFromTop(30);
        load.setBounds(buttons.removeFromLeft(buttons.getWidth() / 2)); reference.setBounds(buttons);
        status.setBounds(area.removeFromTop(36)); controls.setBounds(area);
    }
private:
    void timerCallback() override { status.setText(processor.modelStatus(), juce::dontSendNotification); }
    NeuralAudioFX& processor;
    juce::GenericAudioProcessorEditor controls;
    juce::TextButton load, reference; juce::Label status;
    std::unique_ptr<juce::FileChooser> chooser;
};
juce::AudioProcessorEditor* NeuralAudioFX::createEditor() { return new Editor(*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new NeuralAudioFX(); }
