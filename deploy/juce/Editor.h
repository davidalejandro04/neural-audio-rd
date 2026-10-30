#pragma once

namespace palette {
const juce::Colour background{0xff111315}, panel{0xff1c1f21}, line{0xff363a3c};
const juce::Colour text{0xfff4eee2}, muted{0xffa9aaa5}, green{0xffdcb875}, orange{0xffed8757};
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
        setColour(juce::Slider::thumbColourId, palette::green);
        setColour(juce::Slider::trackColourId, palette::green.darker(.5f));
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
        for (auto* c : std::initializer_list<juce::Component*>{&drive, &mix, &load, &reference, &status, &samples, &open, &play, &stop, &dry, &loop, &sampleInfo, &engine, &inputGain, &outputGain, &tone, &crunch, &heavy, &clean, &rhythm, &neural, &cabinet}) addAndMakeVisible(c);
        driveAttachment = std::make_unique<Attachment>(p.parameters, "drive", drive);
        mixAttachment = std::make_unique<Attachment>(p.parameters, "mix", mix);
        configureKnob(drive, "Drive"); configureKnob(mix, "Mix");
        engine.addItem("Neural preamp / loaded model", 1); engine.addItem("Pedal distortion / DSP 4x", 2); engine.addItem("British amp / DSP 4x", 3); engine.setComponentID("effectEngine");
        engineAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.parameters, "engine", engine);
        for (auto* s : {&inputGain, &outputGain, &tone}) { s->setSliderStyle(juce::Slider::LinearHorizontal); s->setTextBoxStyle(juce::Slider::TextBoxRight, false, 75, 26); }
        inputAttachment = std::make_unique<Attachment>(p.parameters, "inputDb", inputGain);
        outputAttachment = std::make_unique<Attachment>(p.parameters, "outputDb", outputGain);
        toneAttachment = std::make_unique<Attachment>(p.parameters, "tone", tone);
        cabinet.addItemList({"Off / direct", "British 4x12 / focused", "Vintage 4x12 / warm", "Modern 4x12 / bright"}, 1);
        cabinet.setComponentID("cabinet");
        cabinetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(p.parameters, "cabinet", cabinet);
        for (size_t i = 0; i < rackSliders.size(); ++i) {
            auto& slider = rackSliders[i]; addAndMakeVisible(slider);
            slider.setName(rackNames[i]); slider.setComponentID(rackIds[i]);
            slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
            slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 92, 22);
            rackAttachments[i] = std::make_unique<Attachment>(p.parameters, rackIds[i], slider);
            slider.setTextValueSuffix(i < 3 ? " dB" : i == 3 ? " ms" : "");
            slider.setNumDecimalPlacesToDisplay(i < 3 ? 1 : i == 3 ? 0 : 2);
            slider.textFromValueFunction = [i](double value) { return juce::String(value, i < 3 ? 1 : i == 3 ? 0 : 2) + (i < 3 ? " dB" : i == 3 ? " ms" : ""); };
            slider.updateText();
            slider.setDoubleClickReturnValue(true, i < 3 ? 0 : i == 3 ? 320 : i == 4 ? .3 : i == 7 ? .45 : 0);
        }
        configureKnob(inputGain, "Input"); configureKnob(outputGain, "Master");
        inputGain.setDoubleClickReturnValue(true, 0); outputGain.setDoubleClickReturnValue(true, -9);
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
        clean.setButtonText("01  Clean"); crunch.setButtonText("02  Crunch"); rhythm.setButtonText("03  Hard rock"); heavy.setButtonText("04  Sunset lead"); neural.setButtonText("05  Neural lead");
        clean.setComponentID("presetClean"); crunch.setComponentID("presetCrunch"); rhythm.setComponentID("presetRock"); heavy.setComponentID("presetLead"); neural.setComponentID("presetNeural");
        samples.setComponentID("sampleSelect"); stop.setComponentID("sampleStop"); dry.setComponentID("dryCompare");
        loop.setButtonText("Loop"); loop.setComponentID("sampleLoop"); loop.setToggleState(true, juce::dontSendNotification);
        status.setColour(juce::Label::textColourId, palette::muted); status.setFont(juce::FontOptions(12.f));
        status.setComponentID("modelStatus");
        status.setJustificationType(juce::Justification::centredRight);
        sampleInfo.setColour(juce::Label::textColourId, palette::muted); sampleInfo.setFont(juce::FontOptions(12.f));
        samples.setTextWhenNothingSelected("Select a dataset sample or open a WAV");
        load.setTooltip("Load an exported ONNX model matching your audio-device sample rate.");
        dry.setTooltip("Compare the dry input through the same latency buffer. Your Mix setting is retained.");
        play.setTooltip("Replace the live input with the selected sample while it plays. Click again to restart.");
        drive.setTooltip("Normalized distortion amount. Double-click to reset to 50%.");
        mix.setTooltip("Amp dry/wet blend; the model applies this once. At zero, the rack is bypassed. Double-click: 100%.");
        inputGain.setTooltip("Boost before DSP saturation. Disabled for neural models to preserve their training input level.");
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
        clean.onClick = [safe] { if (safe) safe->setRigPreset(0); };
        crunch.onClick = [safe] { if (safe) safe->setRigPreset(1); };
        rhythm.onClick = [safe] { if (safe) safe->setRigPreset(2); };
        heavy.onClick = [safe] { if (safe) safe->setRigPreset(3); };
        neural.onClick = [safe] { if (safe) safe->setRigPreset(4); };
        neural.setTooltip("Use the currently loaded neural model with a lead rack. Load the high_gain export for the new preamp.");
        cabinet.setTooltip("Algorithmic cabinet voicings, not recorded speaker impulse responses.");
        dry.setToggleState(processor.dryCompare.load(), juce::dontSendNotification);
        loop.setToggleState(processor.looping.load(), juce::dontSendNotification);
        setSize(1100, 830); loadLibrary(); timerCallback(); startTimerHz(30);
    }
    ~Editor() override { stopTimer(); setLookAndFeel(nullptr); }
    void paint(juce::Graphics& g) override {
        g.fillAll(palette::background);
        text(g, "NEURAL AUDIO", {26, 16, 420, 28}, 24, palette::text, true);
        text(g, "BRITISH HIGH GAIN  /  GUITAR RIG", {27, 47, 530, 20}, 11, palette::green, true);
        text(g, "INPUT  >  AMP  >  CABINET  >  DELAY  >  REVERB", {590, 26, 482, 25}, 11, palette::muted, true, juce::Justification::centredRight);
        card(g, {24, 176, 730, 247}); card(g, {770, 176, 306, 247});
        text(g, "01 / AMPLIFIER", {44, 190, 340, 23}, 12, palette::green, true);
        text(g, "GAIN & TONE", {550, 190, 180, 23}, 10, palette::muted, true, juce::Justification::centredRight);
        const char* names[]{"DRIVE", "INPUT", "BASS", "MIDDLE", "TREBLE", "MASTER"};
        for (int i = 0; i < 6; ++i) text(g, names[i], {39 + i * 118, 226, 112, 20}, 11, i == 0 ? palette::orange : palette::muted, true, juce::Justification::centred);
        text(g, "AMP MIX", {44, 377, 75, 22}, 10, palette::muted, true);
        text(g, "DSP TONE", {396, 377, 80, 22}, 10, palette::muted, true);
        text(g, "02 / CABINET", {790, 190, 266, 23}, 12, palette::green, true);
        // Vector speaker grille: this is part of the actual native editor.
        g.setColour(palette::background); g.fillRoundedRectangle(790, 258, 122, 142, 5);
        g.setColour(palette::line);
        for (int x = 794; x < 909; x += 5) g.drawVerticalLine(x, 263, 395);
        for (int y = 263; y < 397; y += 5) g.drawHorizontalLine(y, 794, 908);
        for (int row = 0; row < 2; ++row) for (int col = 0; col < 2; ++col) {
            auto r = juce::Rectangle<float>(800.f + col * 55.f, 275.f + row * 61.f, 46, 46);
            g.setColour(palette::background.withAlpha(.8f)); g.fillEllipse(r);
            g.setColour(palette::green.withAlpha(.4f)); g.drawEllipse(r, 1);
            g.drawEllipse(r.reduced(9), 1);
        }
        text(g, "4 x 12", {924, 265, 125, 25}, 22, palette::text, true);
        text(g, "Closed-back voice", {924, 295, 134, 20}, 11, palette::muted);
        text(g, "Algorithmic cabinet", {924, 318, 134, 20}, 10, palette::muted);
        text(g, "Tight lows.", {924, 357, 130, 18}, 11, palette::green);
        text(g, "Forward mids.", {924, 377, 130, 18}, 11, palette::green);
        card(g, {24, 437, 402, 179}); card(g, {442, 437, 312, 179}); card(g, {770, 437, 306, 179});
        text(g, "03 / TAPE-STYLE DELAY", {44, 449, 355, 22}, 12, palette::green, true);
        const char* delayNames[]{"TIME", "FEEDBACK", "LEVEL"};
        for (int i = 0; i < 3; ++i) text(g, delayNames[i], {45 + i * 124, 475, 112, 18}, 10, palette::muted, true, juce::Justification::centred);
        text(g, "04 / ROOM REVERB", {462, 449, 270, 22}, 12, palette::green, true);
        text(g, "LEVEL", {469, 475, 112, 18}, 10, palette::muted, true, juce::Justification::centred);
        text(g, "SIZE", {610, 475, 112, 18}, 10, palette::muted, true, juce::Justification::centred);
        text(g, "OUTPUT MONITOR", {790, 449, 266, 22}, 12, palette::muted, true);
        meter(g, 505, "IN", inLevel); meter(g, 551, "OUT", outLevel);
        text(g, processor.dryCompare.load() ? "DRY COMPARE / LATENCY ALIGNED" : "PROCESSED / LIVE SIGNAL PATH", {790, 582, 266, 18}, 10, palette::green, true);
        card(g, {24, 630, 1052, 148});
        text(g, "AUDITION", {44, 643, 100, 20}, 11, palette::green, true);
        waveform(g);
        const bool clipping = outLevel > 1.f;
        text(g, clipping ? "Output above 0 dBFS: lower Master." : "Stop restores live input. Dry compare bypasses the complete rig.", {28, 796, 550, 18}, 11, clipping ? palette::orange : palette::muted);
    }
    void resized() override {
        engine.setBounds(28, 83, 420, 32); load.setBounds(462, 83, 140, 32); reference.setBounds(614, 83, 156, 32);
        status.setBounds(782, 76, 290, 46); status.setJustificationType(juce::Justification::centredRight);
        const std::array<juce::TextButton*, 5> presets{&clean, &crunch, &rhythm, &heavy, &neural};
        for (int i = 0; i < 5; ++i) presets[static_cast<size_t>(i)]->setBounds(28 + i * 211, 132, 199, 30);
        drive.setBounds(39, 247, 112, 116); inputGain.setBounds(157, 247, 112, 116);
        for (int i = 0; i < 3; ++i) rackSliders[static_cast<size_t>(i)].setBounds(275 + i * 118, 247, 112, 116);
        outputGain.setBounds(629, 247, 112, 116);
        mix.setSliderStyle(juce::Slider::LinearHorizontal); mix.setTextBoxStyle(juce::Slider::TextBoxRight, false, 58, 24);
        mix.setBounds(119, 374, 252, 28); tone.setBounds(476, 374, 254, 28);
        cabinet.setBounds(790, 223, 266, 29);
        for (int i = 0; i < 3; ++i) rackSliders[static_cast<size_t>(i + 3)].setBounds(45 + i * 124, 493, 112, 112);
        rackSliders[6].setBounds(469, 493, 112, 112); rackSliders[7].setBounds(610, 493, 112, 112);
        samples.setBounds(149, 638, 475, 30); open.setBounds(636, 638, 130, 30);
        play.setBounds(44, 686, 132, 30); stop.setBounds(186, 686, 72, 30); loop.setBounds(268, 686, 70, 30); dry.setBounds(790, 638, 266, 30);
        sampleInfo.setBounds(43, 741, 995, 23);
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
        text(g, name, {790, y - 18, 60, 17}, 11, palette::muted, true);
        text(g, juce::String(db, 1) + " dBFS", {931, y - 18, 125, 17}, 11, palette::muted, false, juce::Justification::centredRight);
        g.setColour(palette::background); g.fillRoundedRectangle(790, static_cast<float>(y + 5), 266, 8, 4);
        g.setColour(value > 1 ? palette::orange : palette::green); g.fillRoundedRectangle(790, static_cast<float>(y + 5), 266 * juce::jlimit(0.f, 1.f, (db + 60) / 60), 8, 4);
    }
    void waveform(juce::Graphics& g) const {
        const juce::Rectangle<float> area(356, 681, 700, 48);
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
                if (action == 0) { safe->processor.chooseModel(f.getResult().getFullPathName()); safe->setParameter("engine", 0); }
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
    }

    void setParameter(const char* id, float value) {
        if (!std::isfinite(value)) return;
        auto* param = processor.parameters.getParameter(id);
        param->beginChangeGesture(); param->setValueNotifyingHost(param->convertTo0to1(value)); param->endChangeGesture();
    }
    void setRigPreset(int index) {
        const float drives[]{.02f, .38f, .72f, .88f, .85f};
        setParameter("engine", index == 4 ? 0.f : index == 0 ? 1.f : 2.f);
        setParameter("drive", drives[index]); setParameter("mix", 1);
        setParameter("inputDb", index == 0 ? -9.f : index == 1 ? 0.f : 6.f);
        setParameter("outputDb", index == 0 ? -3.f : -9.f); setParameter("tone", 6000);
        setParameter("cabinet", index == 0 ? 2.f : 1.f);
        setParameter("bass", index == 0 ? 0.f : -1.f); setParameter("middle", index >= 2 ? 3.f : 1.f); setParameter("treble", index >= 2 ? 1.f : 0.f);
        setParameter("delayMs", index >= 3 ? 340.f : 120.f); setParameter("feedback", index >= 3 ? .32f : .15f);
        setParameter("delayMix", index >= 3 ? .18f : 0.f); setParameter("reverbMix", index >= 3 ? .12f : .06f); setParameter("room", .45f);
        dry.setToggleState(false, juce::dontSendNotification); processor.dryCompare.store(false);
        currentPreset = index; presetValues.clear();
        for (auto* p : processor.getParameters()) presetValues.push_back(p->getValue());
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
        bool matches = presetValues.size() == static_cast<size_t>(processor.getParameters().size());
        if (matches) for (int i = 0; i < processor.getParameters().size(); ++i)
            matches = matches && std::abs(processor.getParameters()[i]->getValue() - presetValues[static_cast<size_t>(i)]) < 1e-5f;
        const std::array<juce::TextButton*, 5> presets{&clean, &crunch, &rhythm, &heavy, &neural};
        for (int i = 0; i < 5; ++i) presets[static_cast<size_t>(i)]->setToggleState(matches && i == currentPreset, juce::dontSendNotification);
        repaint();
    }
    NeuralAudioFX& processor;
    LabLook look;
    juce::Slider drive, mix, inputGain, outputGain, tone;
    juce::TextButton load, reference, open, play, stop, dry, crunch, heavy, clean, rhythm, neural;
    juce::ToggleButton loop;
    juce::ComboBox samples, engine, cabinet;
    const std::array<const char*, 8> rackIds{"bass", "middle", "treble", "delayMs", "feedback", "delayMix", "reverbMix", "room"};
    const std::array<const char*, 8> rackNames{"Bass", "Middle", "Treble", "Time", "Feedback", "Delay level", "Reverb level", "Size"};
    std::array<juce::Slider, 8> rackSliders;
    std::array<std::unique_ptr<Attachment>, 8> rackAttachments;
    juce::Label status, sampleInfo;
    std::unique_ptr<Attachment> driveAttachment, mixAttachment, inputAttachment, outputAttachment, toneAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> engineAttachment, cabinetAttachment;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::File libraryRoot;
    juce::Array<juce::var> sampleEntries;
    juce::TooltipWindow tooltips{this, 700};
    float inLevel{}, outLevel{};
    int currentPreset{-1}; std::vector<float> presetValues;
};
