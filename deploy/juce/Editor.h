#pragma once

namespace palette {
const juce::Colour background{0xff101619}, panel{0xff192226}, line{0xff2c393f};
const juce::Colour text{0xffedf3ee}, muted{0xff9cacae}, green{0xffb7ed97}, orange{0xfff5ad70};
}

class LabLook final : public juce::LookAndFeel_V4 {
public:
    LabLook() {
        setColour(juce::Label::textColourId, palette::text);
        setColour(juce::ComboBox::backgroundColourId, palette::background);
        setColour(juce::ComboBox::textColourId, palette::text);
        setColour(juce::ComboBox::outlineColourId, palette::line);
        setColour(juce::PopupMenu::backgroundColourId, palette::panel);
        setColour(juce::PopupMenu::textColourId, palette::text);
        setColour(juce::PopupMenu::highlightedBackgroundColourId, palette::line);
        setColour(juce::Slider::textBoxTextColourId, palette::text);
        setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour(juce::ToggleButton::textColourId, palette::muted);
        setColour(juce::ToggleButton::tickColourId, palette::green);
    }
    void drawButtonBackground(juce::Graphics& g, juce::Button& b, const juce::Colour&, bool over, bool down) override {
        auto bounds = b.getLocalBounds().toFloat().reduced(.5f);
        auto colour = b.getToggleState() || b.getComponentID() == "primary" ? palette::green : palette::line;
        if (over) colour = colour.brighter(.1f);
        if (down) colour = colour.darker(.15f);
        g.setColour(colour); g.fillRoundedRectangle(bounds, 8.f);
    }
    void drawButtonText(juce::Graphics& g, juce::TextButton& b, bool, bool) override {
        g.setColour(b.getToggleState() || b.getComponentID() == "primary" ? palette::background : palette::text);
        g.setFont(juce::FontOptions(14.f, juce::Font::bold));
        g.drawFittedText(b.getButtonText(), b.getLocalBounds().reduced(8, 0), juce::Justification::centred, 1);
    }
    void drawRotarySlider(juce::Graphics& g, int x, int y, int w, int h, float value, float start, float end, juce::Slider& slider) override {
        auto bounds = juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y), static_cast<float>(w), static_cast<float>(h)).reduced(12);
        auto centre = bounds.getCentre(); auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) * .5f;
        juce::Path track, progress;
        track.addCentredArc(centre.x, centre.y, radius, radius, 0, start, end, true);
        progress.addCentredArc(centre.x, centre.y, radius, radius, 0, start, start + value * (end - start), true);
        g.setColour(palette::line); g.strokePath(track, juce::PathStrokeType(7.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        auto accent = slider.getName() == "Drive" ? palette::orange : palette::green;
        g.setColour(accent); g.strokePath(progress, juce::PathStrokeType(7.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        g.setColour(palette::background); g.fillEllipse(centre.x - radius + 12, centre.y - radius + 12, (radius - 12) * 2, (radius - 12) * 2);
        const float angle = start + value * (end - start) - juce::MathConstants<float>::halfPi;
        g.setColour(accent); g.drawLine(centre.x + std::cos(angle) * (radius - 29), centre.y + std::sin(angle) * (radius - 29),
                                       centre.x + std::cos(angle) * (radius - 17), centre.y + std::sin(angle) * (radius - 17), 3.f);
    }
};

class Editor final : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit Editor(NeuralAudioFX& p) : AudioProcessorEditor(p), processor(p) {
        setLookAndFeel(&look);
        for (auto* c : std::initializer_list<juce::Component*>{&drive, &mix, &load, &reference, &status, &samples, &open, &play, &stop, &dry, &loop, &sampleInfo, &engine, &inputGain, &outputGain, &tone, &crunch, &heavy}) addAndMakeVisible(c);
        driveAttachment = std::make_unique<Attachment>(p.parameters, "drive", drive);
        mixAttachment = std::make_unique<Attachment>(p.parameters, "mix", mix);
        configureKnob(drive, "Drive"); configureKnob(mix, "Mix");
        engine.addItem("Neural model", 1); engine.addItem("Pedal distortion (4x)", 2); engine.setComponentID("effectEngine");
        engineAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.parameters, "engine", engine);
        for (auto* s : {&inputGain, &outputGain, &tone}) { s->setSliderStyle(juce::Slider::LinearHorizontal); s->setTextBoxStyle(juce::Slider::TextBoxRight, false, 75, 26); }
        inputAttachment = std::make_unique<Attachment>(p.parameters, "inputDb", inputGain);
        outputAttachment = std::make_unique<Attachment>(p.parameters, "outputDb", outputGain);
        toneAttachment = std::make_unique<Attachment>(p.parameters, "tone", tone);
        inputGain.setTextValueSuffix(""); outputGain.setTextValueSuffix(""); tone.setTextValueSuffix("");
        for (auto* slider : {&inputGain, &outputGain}) {
            slider->textFromValueFunction = [](double value) { return (value > 0 ? "+" : "") + juce::String(value, 1) + " dB"; };
            slider->valueFromTextFunction = [](const juce::String& text) { return text.getDoubleValue(); }; slider->updateText();
        }
        tone.textFromValueFunction = [](double value) { return juce::String(juce::roundToInt(value)) + " Hz"; }; tone.updateText();
        inputGain.setComponentID("pedalInput"); outputGain.setComponentID("outputTrim"); tone.setComponentID("pedalTone");
        load.setButtonText("Choose model"); reference.setButtonText("Reference preset");
        open.setButtonText("Open WAV..."); play.setButtonText("Play sample"); stop.setButtonText("Stop"); dry.setButtonText("Dry compare");
        play.setComponentID("primary"); dry.setClickingTogglesState(true);
        crunch.setButtonText("Crunch"); heavy.setButtonText("Heavy");
        samples.setComponentID("sampleSelect"); stop.setComponentID("sampleStop"); dry.setComponentID("dryCompare");
        loop.setButtonText("Loop"); loop.setComponentID("sampleLoop"); loop.setToggleState(true, juce::dontSendNotification);
        status.setColour(juce::Label::textColourId, palette::muted); status.setFont(juce::FontOptions(12.f));
        status.setJustificationType(juce::Justification::centredRight);
        sampleInfo.setColour(juce::Label::textColourId, palette::muted); sampleInfo.setFont(juce::FontOptions(12.f));
        samples.setTextWhenNothingSelected("Select a dataset sample or open a WAV");
        load.setTooltip("Load an exported ONNX model matching your audio-device sample rate.");
        dry.setTooltip("Compare the dry input through the same latency buffer. Your Mix setting is retained.");
        play.setTooltip("Replace the live input with the selected sample while it plays. Click again to restart.");
        drive.setTooltip("Normalized distortion amount. Double-click to reset to 50%.");
        mix.setTooltip("Blend dry and distorted audio. Double-click to reset to 100%.");
        inputGain.setTooltip("Boost the clean guitar before pedal saturation. Applies to Pedal distortion.");
        tone.setTooltip("Pedal low-pass tone cutoff; lower values reduce high-frequency fizz.");
        outputGain.setTooltip("Output trim after the effect. Dry compare and host bypass audition the original level.");
        const juce::Component::SafePointer<Editor> safe(this);
        load.onClick = [safe] { if (safe) safe->chooseFile("Choose neural audio model", "*.onnx", 0); };
        reference.onClick = [safe] { if (safe) safe->chooseFile("Choose reference preset", "*.json", 1); };
        open.onClick = [safe] { if (safe) safe->chooseFile("Open sample (up to 60 seconds)", "*.wav", 2); };
        play.onClick = [safe] { if (safe) safe->processor.playSample(); };
        stop.onClick = [safe] { if (safe) safe->processor.stopSample(); };
        dry.onClick = [safe] { if (safe) safe->processor.dryCompare.store(safe->dry.getToggleState()); };
        loop.onClick = [safe] { if (safe) safe->processor.looping.store(safe->loop.getToggleState()); };
        samples.onChange = [safe] { if (safe) safe->selectSample(); };
        crunch.onClick = [safe] { if (safe) safe->setPedalPreset(.65f, 6.f, -9.f, 5500.f); };
        heavy.onClick = [safe] { if (safe) safe->setPedalPreset(.90f, 12.f, -12.f, 4500.f); };
        dry.setToggleState(processor.dryCompare.load(), juce::dontSendNotification);
        loop.setToggleState(processor.looping.load(), juce::dontSendNotification);
        setSize(800, 790); loadLibrary(); timerCallback(); startTimerHz(30);
    }
    ~Editor() override { stopTimer(); setLookAndFeel(nullptr); }
    void paint(juce::Graphics& g) override {
        g.fillAll(palette::background);
        text(g, "NEURAL / AUDIO LAB", {28, 22, 480, 35}, 26, palette::text, true);
        text(g, "ELECTRIC GUITAR / DISTORTION WORKBENCH", {29, 60, 520, 20}, 12, palette::muted, true);
        g.setColour(palette::green.withAlpha(.12f)); g.fillRoundedRectangle(658, 28, 114, 26, 13);
        text(g, "EFFECT PREVIEW", {665, 31, 104, 20}, 10, palette::green, true);
        card(g, {24, 142, 752, 207}); card(g, {24, 363, 752, 90}); card(g, {24, 471, 752, 238});
        text(g, "PEDAL INPUT", {44, 374, 200, 18}, 11, palette::muted, true);
        text(g, "PEDAL TONE", {291, 374, 200, 18}, 11, palette::muted, true);
        text(g, "OUTPUT", {538, 374, 200, 18}, 11, palette::muted, true);
        text(g, "DRIVE", {57, 158, 180, 22}, 12, palette::orange, true, juce::Justification::centred);
        text(g, "MIX", {278, 158, 180, 22}, 12, palette::green, true, juce::Justification::centred);
        text(g, "Harmonic intensity", {57, 318, 180, 20}, 12, palette::muted, false, juce::Justification::centred);
        text(g, "Dry / wet blend", {278, 318, 180, 20}, 12, palette::muted, false, juce::Justification::centred);
        g.setColour(palette::line); g.drawVerticalLine(500, 163, 328);
        text(g, "SIGNAL LEVEL", {528, 160, 220, 23}, 12, palette::muted, true);
        meter(g, 211, "IN", inLevel); meter(g, 263, "OUT", outLevel);
        text(g, processor.dryCompare.load() ? "LISTENING: DRY" : "LISTENING: PROCESSED", {528, 308, 220, 20}, 11,
             processor.dryCompare.load() ? palette::orange : palette::green, true);
        text(g, "SAMPLE BENCH", {43, 483, 250, 21}, 12, palette::muted, true);
        waveform(g);
        const auto* clip = processor.selectedSample();
        if (clip) {
            auto seconds = static_cast<double>(clip->audio.getNumSamples()) / clip->rate;
            text(g, juce::String(seconds * processor.sampleProgress.load(), 2) + " / " + juce::String(seconds, 2) + " s", {594, 673, 160, 20}, 12, palette::muted, false, juce::Justification::centredRight);
        }
        const bool clipping = outLevel > 1.f;
        text(g, clipping ? "Output exceeds 0 dBFS - reduce input or output gain." : "Dry input -> selected effect -> output  /  Stop restores live input", {27, 754, 744, 20}, 12,
             clipping ? palette::orange : palette::muted);
    }
    void resized() override {
        load.setBounds(28, 95, 132, 32); reference.setBounds(170, 95, 150, 32); engine.setBounds(340, 95, 432, 32);
        status.setBounds(27, 718, 744, 25);
        drive.setBounds(60, 182, 174, 132); mix.setBounds(281, 182, 174, 132);
        inputGain.setBounds(40, 402, 220, 32); tone.setBounds(287, 402, 220, 32); outputGain.setBounds(534, 402, 220, 32);
        samples.setBounds(44, 513, 520, 34); open.setBounds(578, 513, 176, 34);
        play.setBounds(44, 559, 130, 32); stop.setBounds(184, 559, 78, 32); loop.setBounds(270, 559, 75, 32); dry.setBounds(603, 559, 151, 32);
        crunch.setBounds(356, 559, 104, 32); heavy.setBounds(470, 559, 104, 32);
        sampleInfo.setBounds(43, 673, 547, 22);
    }
private:
    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    static void configureKnob(juce::Slider& slider, const juce::String& name) {
        slider.setName(name); slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 100, 24);
        slider.setDoubleClickReturnValue(true, name == "Drive" ? .5 : 1.);
        slider.textFromValueFunction = [](double value) { return juce::String(juce::roundToInt(value * 100)) + "%"; };
        slider.valueFromTextFunction = [](const juce::String& value) { return value.getDoubleValue() / 100.; };
        slider.updateText();
    }
    static void text(juce::Graphics& g, const juce::String& value, juce::Rectangle<int> rect, float size, juce::Colour colour,
                     bool bold = false, juce::Justification justification = juce::Justification::centredLeft) {
        g.setColour(colour); g.setFont(juce::FontOptions(size, bold ? juce::Font::bold : juce::Font::plain)); g.drawFittedText(value, rect, justification, 1);
    }
    static void card(juce::Graphics& g, juce::Rectangle<float> rect) {
        g.setColour(palette::panel); g.fillRoundedRectangle(rect, 14); g.setColour(palette::line); g.drawRoundedRectangle(rect, 14, 1);
    }
    static void meter(juce::Graphics& g, int y, const juce::String& name, float value) {
        const float db = juce::Decibels::gainToDecibels(value, -60.f);
        text(g, name, {528, y - 18, 60, 17}, 11, palette::muted, true);
        text(g, juce::String(db, 1) + " dBFS", {623, y - 18, 125, 17}, 11, palette::muted, false, juce::Justification::centredRight);
        g.setColour(palette::background); g.fillRoundedRectangle(528, static_cast<float>(y + 5), 220, 8, 4);
        g.setColour(value > 1 ? palette::orange : palette::green); g.fillRoundedRectangle(528, static_cast<float>(y + 5), 220 * juce::jlimit(0.f, 1.f, (db + 60) / 60), 8, 4);
    }
    void waveform(juce::Graphics& g) const {
        const juce::Rectangle<float> area(44, 606, 710, 57);
        g.setColour(palette::background); g.fillRoundedRectangle(area, 6);
        auto* clip = processor.selectedSample();
        if (!clip) { text(g, "Open a WAV to audition the distortion", area.toNearestInt(), 12, palette::muted, false, juce::Justification::centred); return; }
        g.setColour(palette::green.withAlpha(.7f));
        const float maximum = juce::jmax(.001f, *std::max_element(clip->peaks.begin(), clip->peaks.end()));
        for (size_t i = 0; i < clip->peaks.size(); ++i) {
            const float x = area.getX() + 8 + static_cast<float>(i) * (area.getWidth() - 16) / clip->peaks.size();
            const float height = juce::jmax(1.f, juce::jmin(1.f, clip->peaks[i] / maximum) * 23);
            g.drawVerticalLine(static_cast<int>(x), area.getCentreY() - height, area.getCentreY() + height);
        }
        g.setColour(palette::orange); const float x = area.getX() + processor.sampleProgress.load() * area.getWidth();
        g.drawLine(x, area.getY() + 3, x, area.getBottom() - 3, 2);
    }
    void chooseFile(const juce::String& title, const juce::String& extension, int action) {
        chooser = std::make_unique<juce::FileChooser>(title, libraryRoot, extension);
        const juce::Component::SafePointer<Editor> safe(this);
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [safe, action](const juce::FileChooser& f) {
                if (!safe || !f.getResult().existsAsFile()) return;
                if (action == 0) safe->processor.chooseModel(f.getResult().getFullPathName());
                else if (action == 1) safe->processor.applyReferencePreset(f.getResult());
                else if (safe->processor.loadSample(f.getResult())) { safe->samples.setSelectedId(0, juce::dontSendNotification); safe->samples.setText(f.getResult().getFileName(), juce::dontSendNotification); }
            });
    }
    void loadLibrary() {
        auto configured = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_SAMPLES", "");
        libraryRoot = configured.isEmpty() ? juce::File::getCurrentWorkingDirectory().getChildFile("samples/guitar") : juce::File(configured);
        if (configured.isEmpty() && !libraryRoot.getChildFile("manifest.json").existsAsFile()) libraryRoot = juce::File::getCurrentWorkingDirectory().getChildFile("samples/distortion");
        if (!libraryRoot.getChildFile("manifest.json").existsAsFile()) {
            auto model = juce::SystemStats::getEnvironmentVariable("NEURAL_FX_MODEL", "");
            if (model.isNotEmpty()) libraryRoot = juce::File(model).getParentDirectory().getParentDirectory().getChildFile("samples");
        }
        auto manifest = juce::JSON::parse(libraryRoot.getChildFile("manifest.json"));
        if (auto* entries = manifest["clips"].getArray()) {
            sampleEntries = *entries;
            for (int i = 0; i < sampleEntries.size(); ++i) samples.addItem(sampleEntries[i]["name"].toString(), i + 1);
            if (!sampleEntries.isEmpty() && !processor.selectedSample()) {
                samples.setSelectedId(1, juce::dontSendNotification);
                auto file = libraryRoot.getChildFile(sampleEntries[0]["dry"].toString());
                if (file.isAChildOf(libraryRoot)) processor.loadSample(file);
            }
        }
    }
    void selectSample() {
        const int index = samples.getSelectedId() - 1;
        if (index < 0 || index >= sampleEntries.size()) return;
        const auto& entry = sampleEntries[index];
        auto file = libraryRoot.getChildFile(entry["dry"].toString());
        if (!file.isAChildOf(libraryRoot) || !processor.loadSample(file)) return;
        for (auto id : {"drive", "mix", "engine", "inputDb", "outputDb", "tone"}) {
            const float value = entry.hasProperty(id) ? static_cast<float>(entry[id]) : (juce::String(id) == "tone" ? 5500.f : 0.f);
            setParameter(id, value);
        }
    }
    void setParameter(const char* id, float value) {
        if (!std::isfinite(value)) return;
        auto* param = processor.parameters.getParameter(id);
        param->beginChangeGesture(); param->setValueNotifyingHost(param->convertTo0to1(value)); param->endChangeGesture();
    }
    void setPedalPreset(float driveValue, float input, float output, float cutoff) {
        setParameter("engine", 1); setParameter("drive", driveValue); setParameter("mix", 1);
        setParameter("inputDb", input); setParameter("outputDb", output); setParameter("tone", cutoff);
        dry.setToggleState(false, juce::dontSendNotification); processor.dryCompare.store(false);
    }
    void timerCallback() override {
        status.setText(processor.modelStatus(), juce::dontSendNotification);
        auto message = processor.sampleStatus();
        if (message.isEmpty()) {
            const auto* clip = processor.selectedSample();
            message = clip ? clip->name + "  /  " + juce::String(clip->rate, 0) + " Hz" : "No sample loaded";
        }
        sampleInfo.setText(message, juce::dontSendNotification);
        inLevel = juce::jmax(processor.inputPeak.load(), inLevel * .90f);
        outLevel = juce::jmax(processor.outputPeak.load(), outLevel * .90f);
        play.setButtonText(processor.playing.load() ? "Restart sample" : "Play sample");
        const bool isPedal = processor.parameters.getRawParameterValue("engine")->load() > .5f;
        inputGain.setEnabled(isPedal); tone.setEnabled(isPedal);
        repaint();
    }
    NeuralAudioFX& processor;
    LabLook look;
    juce::Slider drive, mix, inputGain, outputGain, tone;
    juce::TextButton load, reference, open, play, stop, dry, crunch, heavy;
    juce::ToggleButton loop;
    juce::ComboBox samples, engine;
    juce::Label status, sampleInfo;
    std::unique_ptr<Attachment> driveAttachment, mixAttachment, inputAttachment, outputAttachment, toneAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> engineAttachment;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::File libraryRoot;
    juce::Array<juce::var> sampleEntries;
    juce::TooltipWindow tooltips{this, 700};
    float inLevel{}, outLevel{};
};
