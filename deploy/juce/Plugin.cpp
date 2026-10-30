#include <juce_audio_utils/juce_audio_utils.h>
#include "processor.h"
#include <atomic>
#include <memory>
#include <vector>
#include <array>
#include "Pedal.h"
#include "Rack.h"

class NeuralAudioFX final : public juce::AudioProcessor {
public:
    NeuralAudioFX() : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                                       .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
        parameters(*this, nullptr, "NeuralFX", makeParameters()) {
        drive = parameters.getRawParameterValue("drive"); mix = parameters.getRawParameterValue("mix");
        engine = parameters.getRawParameterValue("engine"); inputDb = parameters.getRawParameterValue("inputDb");
        outputDb = parameters.getRawParameterValue("outputDb"); tone = parameters.getRawParameterValue("tone");
        for (size_t i = 0; i < rackIds.size(); ++i) rackValues[i] = parameters.getRawParameterValue(rackIds[i]);
        modelPath = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_MODEL", "");
        formats.registerBasicFormats();
        if (juce::SystemStats::getEnvironmentVariable("NEURAL_FX_PEDAL", "") == "1") {
            for (auto pair : {std::make_pair("engine", 1.f), std::make_pair("drive", .65f), std::make_pair("inputDb", 6.f), std::make_pair("outputDb", -9.f)})
                parameters.getParameter(pair.first)->setValueNotifyingHost(parameters.getParameter(pair.first)->convertTo0to1(pair.second));
        }
        if (juce::SystemStats::getEnvironmentVariable("NEURAL_FX_RIG", "") == "1") {
            for (auto pair : {std::make_pair("engine", 2.f), std::make_pair("drive", .72f), std::make_pair("inputDb", 6.f),
                              std::make_pair("outputDb", -9.f), std::make_pair("tone", 6000.f), std::make_pair("cabinet", 1.f),
                              std::make_pair("bass", -1.f), std::make_pair("middle", 3.f), std::make_pair("treble", 1.f), std::make_pair("reverbMix", .06f)})
                parameters.getParameter(pair.first)->setValueNotifyingHost(parameters.getParameter(pair.first)->convertTo0to1(pair.second));
        }
    }
    static juce::AudioProcessorValueTreeState::ParameterLayout makeParameters() {
        juce::AudioProcessorValueTreeState::ParameterLayout p;
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("drive", 1), "Drive", 0.f, 1.f, 0.5f));
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("mix", 1), "Mix", 0.f, 1.f, 1.f));
        p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID("engine", 1), "Effect engine", juce::StringArray{"Neural model", "Pedal distortion (4x)", "British amp (4x)"}, 0));
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("inputDb", 1), "Pedal input", -12.f, 24.f, 0.f));
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("outputDb", 1), "Output", -24.f, 0.f, 0.f));
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("tone", 1), "Pedal tone", 1000.f, 10000.f, 5500.f));
        p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID("cabinet", 1), "Cabinet", juce::StringArray{"Off", "British 4x12", "Vintage 4x12", "Modern 4x12"}, 0));
        auto add = [&](const char* id, const char* label, float low, float high, float initial) {
            p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID(id, 1), label, low, high, initial));
        };
        add("bass", "Bass", -12, 12, 0); add("middle", "Middle", -12, 12, 0); add("treble", "Treble", -12, 12, 0);
        add("delayMs", "Delay time", 40, 1000, 320); add("feedback", "Delay feedback", 0, .65f, .3f);
        add("delayMix", "Delay level", 0, .5f, 0); add("reverbMix", "Reverb level", 0, .5f, 0); add("room", "Room size", 0, .85f, .45f);
        return p;
    }
    const juce::String getName() const override { return "NeuralAudioFX"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 16; }
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
        retained.clear(); sampleRate = rate; playbackRate.store(rate); prepared = true;
        pedal.prepare(rate); rack.prepare(rate); master.reset(rate, .02); masterInitial = true;
        for (auto& c : dryHistory) c.fill(0); dryCursor = 0;
        currentEngine = -1; reportedLatency.store(128); setLatencySamples(128);
        loadLocked();
    }
    void releaseResources() override {
        const juce::ScopedLock lock(controlLock);
        active.store(nullptr, std::memory_order_release); retained.clear(); prepared = false;
        playing.store(false);
    }
    void reset() override {
        // JUCE calls this outside active rendering; both channels must reset together.
        if (auto* engines = active.load(std::memory_order_acquire)) for (auto& channel : engines->channels) channel->reset();
        pedal.reset();
        rack.reset(); masterInitial = true;
        for (auto& c : dryHistory) c.fill(0); dryCursor = 0;
    }
    void processBlock(juce::AudioBuffer<float>& audio, juce::MidiBuffer&) override { const bool dry = dryCompare.load(); process(audio, dry ? 0.f : mix->load(), dry); }
    void processBlockBypassed(juce::AudioBuffer<float>& audio, juce::MidiBuffer&) override { process(audio, 0.f, true); }
    void process(juce::AudioBuffer<float>& audio, float amount, bool bypass) noexcept {
        juce::ScopedNoDenormals denormals;
        fillAudition(audio);
        inputPeak.store(audio.getMagnitude(0, audio.getNumSamples()));
        auto* engines = active.load(std::memory_order_acquire);
        const bool dryOnly = bypass || amount == 0;
        const auto latency = static_cast<size_t>(reportedLatency.load());
        for (int i = 0; i < audio.getNumSamples(); ++i) {
            const auto read = (dryCursor + dryHistory[0].size() - latency) % dryHistory[0].size();
            for (int ch = 0; ch < juce::jmin(2, audio.getNumChannels()); ++ch) {
                auto& c = dryHistory[static_cast<size_t>(ch)]; c[dryCursor] = audio.getSample(ch, i);
                if (dryOnly) audio.setSample(ch, i, c[read]);
            }
            dryCursor = (dryCursor + 1) % dryHistory[0].size();
        }
        if (dryOnly != wasDry) { pedal.reset(); rack.reset(); if (engines) for (auto& channel : engines->channels) channel->reset(); wasDry = dryOnly; }
        if (dryOnly) { finish(audio, bypass, 0); return; }
        const int selected = juce::roundToInt(engine->load());
        if (selected != currentEngine) { pedal.reset(); rack.reset(); if (engines) for (auto& channel : engines->channels) channel->reset(); currentEngine = selected; }
        if (selected > 0) {
            pedal.process(audio, drive->load(), amount, inputDb->load(), 0, tone->load(), reportedLatency.load(), selected == 2);
            finish(audio, bypass, amount); return;
        }
        if (!engines) { pedal.process(audio, 0, 0, 0, 0, 5500, reportedLatency.load()); outputPeak.store(audio.getMagnitude(0, audio.getNumSamples())); return; }
        if (engines->fault.load()) { audio.clear(); return; }
        const auto gain = drive->load();
        for (int channel = 0; channel < audio.getNumChannels(); ++channel) {
            if (channel > 1 || !engines->channels[static_cast<size_t>(channel)]->process(audio.getWritePointer(channel), static_cast<size_t>(audio.getNumSamples()), gain, amount)) {
                engines->fault.store(true); audio.clear(); return;
            }
        }
        finish(audio, bypass, amount);
    }
    void finish(juce::AudioBuffer<float>& audio, bool bypass, float amount) noexcept {
        const bool dry = bypass || amount == 0;
        if (dry != wasDry) { rack.reset(); wasDry = dry; }
        if (!dry) {
            GuitarRack::Settings s;
            s.cabinet = juce::roundToInt(rackValues[0]->load());
            s.bass = rackValues[1]->load(); s.middle = rackValues[2]->load(); s.treble = rackValues[3]->load();
            s.delayMs = rackValues[4]->load(); s.feedback = rackValues[5]->load(); s.delayMix = rackValues[6]->load();
            s.reverbMix = rackValues[7]->load(); s.room = rackValues[8]->load();
            rack.process(audio, s);
        }
        const float gain = bypass ? 1.f : juce::Decibels::decibelsToGain(outputDb->load());
        if (masterInitial || bypass != wasBypassed) { master.setCurrentAndTargetValue(gain); masterInitial = false; }
        master.setTargetValue(gain); wasBypassed = bypass;
        for (int i = 0; i < audio.getNumSamples(); ++i) {
            const float g = master.getNextValue();
            for (int ch = 0; ch < audio.getNumChannels(); ++ch) audio.getWritePointer(ch)[i] *= g;
        }
        outputPeak.store(audio.getMagnitude(0, audio.getNumSamples()));
    }
    struct Clip {
        juce::AudioBuffer<float> audio;
        double rate{};
        juce::String name, path;
        std::array<float, 180> peaks{};
    };
    bool loadSample(const juce::File& file) {
        const juce::ScopedLock lock(controlLock);
        for (auto& clip : clips) if (clip->path == file.getFullPathName()) {
            playing.store(false); audition.store(clip.get()); rewind.store(true); sampleMessage = ""; return true;
        }
        if (clips.size() >= 32) { sampleMessage = "32 files loaded; reopen the app to load more"; return false; }
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        if (!reader || reader->lengthInSamples <= 0 || reader->lengthInSamples > reader->sampleRate * 60 ||
            reader->numChannels < 1 || reader->numChannels > 2) {
            sampleMessage = "Choose a mono/stereo audio file up to 60 seconds"; return false;
        }
        auto clip = std::make_unique<Clip>();
        clip->rate = reader->sampleRate; clip->name = file.getFileNameWithoutExtension(); clip->path = file.getFullPathName();
        clip->audio.setSize(static_cast<int>(reader->numChannels), static_cast<int>(reader->lengthInSamples));
        if (!reader->read(&clip->audio, 0, clip->audio.getNumSamples(), 0, true, true)) {
            sampleMessage = "Could not read audio"; return false;
        }
        for (int ch = 0; ch < clip->audio.getNumChannels(); ++ch)
            for (int i = 0; i < clip->audio.getNumSamples(); ++i)
                if (!std::isfinite(clip->audio.getSample(ch, i))) { sampleMessage = "Audio contains nonfinite samples"; return false; }
        for (size_t i = 0; i < clip->peaks.size(); ++i) {
            const int begin = static_cast<int>(i * clip->audio.getNumSamples() / clip->peaks.size());
            const int end = static_cast<int>((i + 1) * clip->audio.getNumSamples() / clip->peaks.size());
            clip->peaks[i] = clip->audio.getMagnitude(begin, juce::jmax(1, end - begin));
        }
        playing.store(false); auto* ptr = clip.get(); clips.push_back(std::move(clip));
        audition.store(ptr); rewind.store(true); sampleMessage = ""; return true;
    }
    void playSample() {
        const auto* clip = audition.load();
        const juce::ScopedLock lock(controlLock);
        if (!clip) { sampleMessage = "Select a sample or open a WAV first"; return; }
        if (std::abs(clip->rate - playbackRate.load()) > .5) { sampleMessage = "Set the audio device to " + juce::String(clip->rate, 0) + " Hz"; return; }
        sampleMessage = ""; rewind.store(true); playing.store(true);
    }
    void stopSample() { playing.store(false); rewind.store(true); }
    juce::String sampleStatus() const { const juce::ScopedLock lock(controlLock); return sampleMessage; }
    const Clip* selectedSample() const { return audition.load(); }
    std::atomic<bool> playing{false}, looping{true}, dryCompare{false};
    std::atomic<float> inputPeak{0}, outputPeak{0}, sampleProgress{0};
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
        for (auto pair : {std::make_pair("drive", a), std::make_pair("mix", b), std::make_pair("engine", 0.f), std::make_pair("inputDb", 0.f), std::make_pair("outputDb", 0.f), std::make_pair("cabinet", 0.f), std::make_pair("bass", 0.f), std::make_pair("middle", 0.f), std::make_pair("treble", 0.f), std::make_pair("delayMix", 0.f), std::make_pair("reverbMix", 0.f)}) {
            auto* parameter = parameters.getParameter(pair.first);
            parameter->beginChangeGesture(); parameter->setValueNotifyingHost(parameter->convertTo0to1(pair.second)); parameter->endChangeGesture();
        }
    }
    juce::String modelStatus() const {
        const juce::ScopedLock lock(controlLock);
        if (engine->load() > .5f) return (engine->load() > 1.5f ? "British amp DSP" : "Pedal DSP") + juce::String(" / 4x oversampling / ") + juce::String(sampleRate, 0) + " Hz";
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
    void fillAudition(juce::AudioBuffer<float>& audio) noexcept {
        auto* clip = audition.load();
        if (rewind.exchange(false) || clip != audioClip) { sampleCursor = 0; sampleProgress.store(0); audioClip = clip; }
        if (!playing.load() || !clip) return;
        if (std::abs(clip->rate - playbackRate.load()) > .5) { playing.store(false); return; }
        const int length = clip->audio.getNumSamples();
        for (int i = 0; i < audio.getNumSamples(); ++i) {
            if (sampleCursor >= length) {
                if (looping.load()) sampleCursor = 0;
                else { for (int ch = 0; ch < audio.getNumChannels(); ++ch) audio.clear(ch, i, audio.getNumSamples() - i); playing.store(false); break; }
            }
            for (int ch = 0; ch < audio.getNumChannels(); ++ch)
                audio.setSample(ch, i, clip->audio.getSample(juce::jmin(ch, clip->audio.getNumChannels() - 1), sampleCursor));
            ++sampleCursor;
        }
        sampleProgress.store(static_cast<float>(sampleCursor) / length);
    }
    struct Engines { std::unique_ptr<nfx::BufferedProcessor> channels[2]; std::atomic<bool> fault{false}; };
    void loadLocked() {
        // A rejected selection must never leave an unrelated previous model audible.
        active.store(nullptr, std::memory_order_release);
        if (modelPath.isEmpty()) { active.store(nullptr, std::memory_order_release); status = "Choose a neural model"; reportedLatency.store(128); setLatencySamples(128); return; }
        if (retained.size() >= 32) { status = "Reload the plug-in before selecting more models"; return; }
        try {
            auto engines = std::make_unique<Engines>();
            const auto path = std::filesystem::u8path(modelPath.toStdString());
            for (auto& channel : engines->channels) channel = std::make_unique<nfx::BufferedProcessor>(path);
            if (std::abs(engines->channels[0]->sampleRate() - sampleRate) > 0.5) { status = "Set the session to the model's sample rate"; return; }
            auto* pointer = engines.get();
            const auto latency = static_cast<int>(pointer->channels[0]->latency());
            if (latency < 32 || latency > 8192) { status = "Plugin supports model blocks from 32 to 8192 samples"; return; }
            retained.push_back(std::move(engines));
            // Retain previous engines until releaseResources: no deletion or refcounting on the audio thread.
            active.store(pointer, std::memory_order_release);
            reportedLatency.store(latency); setLatencySamples(latency);
            status = juce::String(pointer->channels[0]->effect()) + " / " + juce::String(sampleRate, 0) + " Hz / " + juce::String(latency) + " samples / " + juce::File(modelPath).getFileName();
        } catch (const std::exception& error) { status = "Could not load model: " + juce::String(error.what()); }
    }
    mutable juce::CriticalSection controlLock;
    juce::String modelPath, status{"Choose a model"};
    double sampleRate{48000}; bool prepared{false};
    std::atomic<float>* drive{}; std::atomic<float>* mix{};
    std::atomic<float>* engine{}; std::atomic<float>* inputDb{}; std::atomic<float>* outputDb{}; std::atomic<float>* tone{};
    Pedal pedal; int currentEngine{-1}; std::atomic<int> reportedLatency{128};
    GuitarRack rack;
    juce::SmoothedValue<float> master; bool masterInitial{true}, wasDry{false}, wasBypassed{false};
    const std::array<const char*, 9> rackIds{"cabinet", "bass", "middle", "treble", "delayMs", "feedback", "delayMix", "reverbMix", "room"};
    std::array<std::atomic<float>*, 9> rackValues{};
    std::array<std::array<float, 16384>, 2> dryHistory{}; size_t dryCursor{};
    std::atomic<Engines*> active{nullptr};
    std::vector<std::unique_ptr<Engines>> retained;
    juce::AudioFormatManager formats;
    std::vector<std::unique_ptr<Clip>> clips; // Immutable, retained until processor destruction; never freed by callback.
    std::atomic<Clip*> audition{nullptr};
    std::atomic<bool> rewind{false};
    std::atomic<double> playbackRate{48000};
    Clip* audioClip{}; int sampleCursor{}; // Audio-thread-owned cursor.
    juce::String sampleMessage;
};

#include "Editor.h"
juce::AudioProcessorEditor* NeuralAudioFX::createEditor() { return new Editor(*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new NeuralAudioFX(); }
