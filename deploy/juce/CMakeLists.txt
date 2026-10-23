juce_add_plugin(NeuralAudioFX
  COMPANY_NAME "Neural Audio R&D" BUNDLE_ID "org.neuralfx.research"
  PLUGIN_MANUFACTURER_CODE NfxL PLUGIN_CODE Nfxr
  FORMATS VST3 Standalone PRODUCT_NAME "NeuralAudioFX"
  IS_SYNTH FALSE NEEDS_MIDI_INPUT FALSE NEEDS_MIDI_OUTPUT FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE)
target_sources(NeuralAudioFX PRIVATE Plugin.cpp)
target_link_libraries(NeuralAudioFX PRIVATE neural_fx_core juce::juce_audio_utils juce::juce_dsp
  PUBLIC juce::juce_recommended_config_flags)
target_compile_definitions(NeuralAudioFX PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
if(WIN32)
  foreach(target NeuralAudioFX_VST3 NeuralAudioFX_Standalone)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:${target}>/${NFX_ORT_RUNTIME_FILENAME}")
  endforeach()
endif()
juce_add_console_app(nafx-plugin-check PRODUCT_NAME "nafx-plugin-check")
target_sources(nafx-plugin-check PRIVATE HostCheck.cpp)
target_link_libraries(nafx-plugin-check PRIVATE juce::juce_audio_utils)
target_compile_definitions(nafx-plugin-check PRIVATE JUCE_PLUGINHOST_VST3=1 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
if(MSVC)
  target_compile_options(nafx-plugin-check PRIVATE /EHsc)
endif()

# Headless editor/transport regression harness; no audio device or visible window.
juce_add_console_app(nafx-ui-check PRODUCT_NAME "nafx-ui-check")
target_sources(nafx-ui-check PRIVATE UiCheck.cpp ../native/allocation_probe.cpp)
target_link_libraries(nafx-ui-check PRIVATE NeuralAudioFX juce::juce_audio_utils juce::juce_dsp)
target_compile_definitions(nafx-ui-check PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
if(MSVC)
  target_compile_options(nafx-ui-check PRIVATE /EHsc)
endif()
if(WIN32)
  add_custom_command(TARGET nafx-ui-check POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:nafx-ui-check>/${NFX_ORT_RUNTIME_FILENAME}")
endif()
