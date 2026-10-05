# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.16)
project(SDL2G19 C)
option(G19_HOST_TEST "Run real SDL2 with a host EGL contract double" OFF)
option(G42_INPUT_VALIDATION "Select the bounded physical input consumer" OFF)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_STATIC_PIC ON CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL2_DISABLE_INSTALL ${G19_HOST_TEST} CACHE BOOL "" FORCE)
set(SDL_VIRTUAL_JOYSTICK ${G19_HOST_TEST} CACHE BOOL "" FORCE)
set(SDL_HIDAPI ${G19_HOST_TEST} CACHE BOOL "" FORCE)
set(SDL_HIDAPI_JOYSTICK OFF CACHE BOOL "" FORCE)
set(SDL_PTHREADS ON CACHE BOOL "" FORCE)
set(SDL_SOURCE_VERSION "2.30.12-g8c56053f13ca-g26")
set(SDL2_DISABLE_SDL2MAIN ON CACHE BOOL "" FORCE)
foreach(feature RENDER HAPTIC SENSOR OPENGL OPENGLES VULKAN X11
                WAYLAND KMSDRM OFFSCREEN JACK PULSEAUDIO DBUS IBUS FCITX LIBUDEV)
    set(SDL_${feature} OFF CACHE BOOL "" FORCE)
endforeach()
# Sound: the pinned fork's PS5 driver (src/audio/ps5), sceAudioOut on the main port
# at 48 kHz, as the console's other titles open it; the host test keeps it off.
# Without it a consumer (EmulationStation) opened no device and played nothing.
if(G19_HOST_TEST)
    set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
else()
    set(SDL_AUDIO ON CACHE BOOL "" FORCE)
endif()
if(G19_HOST_TEST)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()
if(NOT G19_HOST_TEST)
    set(PS5 ON CACHE BOOL "" FORCE)
    # The native folder's libc does not export these header-declared symbols.
    # Use SDL's existing implementations; compile-only probes cannot check imports.
    foreach(symbol WCSLCPY WCSLCAT WCSCASECMP WCSNCASECMP)
        set(HAVE_${symbol} 0 CACHE INTERNAL "native libc contract" FORCE)
    endforeach()
endif()
add_subdirectory("${SDL_SOURCE}" sdl)

# Use SDL's own video-device extension point. Exclude its OSMesa/scanout and
# keyboard/IME source files; retain core SDL and the upstream PS5 pad backend.
get_target_property(sources SDL2-static SOURCES)
list(FILTER sources EXCLUDE REGEX "/video/ps5/")
set_property(TARGET SDL2-static PROPERTY SOURCES "${sources}")
target_sources(SDL2-static PRIVATE SDL_ps5g19.c)
target_include_directories(SDL2-static PRIVATE "${SDL_SOURCE}/src" "${PS5_OPENGL_PREFIX}/include")
target_compile_definitions(SDL2-static PRIVATE SDL_VIDEO_DRIVER_PS5=1 SDL_VIDEO_OPENGL=1)
add_library(g19-example OBJECT example.c)
if(G42_INPUT_VALIDATION AND NOT G19_HOST_TEST)
    # Keep the original object/receipt path; the receipt records this selector.
    configure_file(input_validation.c "${CMAKE_CURRENT_BINARY_DIR}/example.c" COPYONLY)
    set_property(TARGET g19-example PROPERTY SOURCES "${CMAKE_CURRENT_BINARY_DIR}/example.c")
endif()
target_link_libraries(g19-example PRIVATE SDL2-static)
target_include_directories(g19-example PRIVATE "${PS5_OPENGL_PREFIX}/include")
target_compile_definitions(g19-example PRIVATE SDL_MAIN_HANDLED=1)
target_compile_options(g19-example PRIVATE -Wall -Wextra -Werror)
if(G19_HOST_TEST)
    target_sources(SDL2-static PRIVATE "${SDL_SOURCE}/src/joystick/ps5/SDL_ps5joystick.c")
    target_compile_definitions(SDL2-static PRIVATE SDL_JOYSTICK_PS5=1)
    # Keep the mock outside the consumer and driver: no SDL API is mocked.
    add_executable(g19-contract test_contract.c $<TARGET_OBJECTS:g19-example>)
    target_sources(g19-contract PRIVATE input_validation.c test_input.c)
    set_source_files_properties(input_validation.c PROPERTIES COMPILE_DEFINITIONS
        "main=g42_input_main;G42_HOST_CONTRACT=1;G42_TIMEOUT_MS=250")
    target_include_directories(g19-contract PRIVATE "${SDL_SOURCE}/src/joystick/ps5")
    target_compile_definitions(g19-example PRIVATE main=g19_example_main)
    target_compile_options(g19-contract PRIVATE -UNDEBUG -Wall -Wextra -Werror)
    target_include_directories(g19-contract PRIVATE "${PS5_OPENGL_PREFIX}/include")
    target_link_libraries(g19-contract PRIVATE SDL2-static)
    enable_testing()
    add_test(NAME g19-contract COMMAND g19-contract)
else()
    # Reuse upstream's install/export rules; only this adapter's dependencies differ.
    set_property(TARGET SDL2-static PROPERTY INTERFACE_LINK_LIBRARIES
        "$<INSTALL_INTERFACE:PS5OpenGLCore33::OpenGL>;ScePad;SceUserService;SceSystemService")
    target_compile_definitions(SDL2-static INTERFACE SDL_MAIN_HANDLED=1)
    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/sdl/SDL2Config.cmake"
        "\ninclude(CMakeFindDependencyMacro)\nfind_dependency(PS5OpenGLCore33 CONFIG)\n")
    configure_file(sdl2.pc.in "${CMAKE_CURRENT_BINARY_DIR}/sdl/sdl2.pc" @ONLY)
endif()
