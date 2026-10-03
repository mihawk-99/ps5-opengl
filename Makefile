# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

SHELL := /bin/bash
PS5_NATIVE_APP_TEMPLATE ?= $(abspath ../ps5-native-app-boilerplate)
PS5_PAYLOAD_SDK ?= $(PS5_NATIVE_APP_TEMPLATE)/.deps/native/ps5-payload-sdk
export PS5_NATIVE_APP_TEMPLATE PS5_PAYLOAD_SDK
PS5_OPENGL_SDK_PREFIX ?= build/sdk/ps5-opengl-gl46

.PHONY: help source-fetch cts-fetch sdk sdk-gl46 imgui-demo nanovg sokol sokol-cube cubes gl46-demo showcase demo test test-imgui test-sokol-cube test-cubes test-glsl test-compiler test-multidraw test-layered-mip test-depth-targets test-staging
help:
	@printf '%s\n' 'source-fetch: pinned graphics/example sources' \
	  'sdk / sdk-gl46: build the compiler, Mesa and installed OpenGL 4.6 SDK' \
	  'imgui-demo / nanovg / sokol: package one example using the installed SDK' \
	  'sokol-cube: package the upstream 3D sample; see examples/core33-sokol-cube' \
	  'cubes: package the 3D frame benchmark using the current source runtime' \
	  'gl46-demo: package the compute-driven OpenGL 4.6 cube animation' \
	  'showcase: package the OpenGL 4.6 showcase (4K, compute, indirect count, bloom)' \
	  'demo: build the SDK if needed, then the showcase demo app and its release zip' \
	  'test: dependency-free host tests and published validation audit' \
	  'test-imgui: software-Mesa renderer and TV-demo input checks' \
	  'cts-fetch: also fetch pinned optional CTS sources; see docs/testing.md'

source-fetch:
	python3 tools/fetch-sources.py
cts-fetch:
	python3 tools/fetch-sources.py --cts
sdk: sdk-gl46
sdk-gl46:
	bash toolchain/build-opengnm-psbc.sh
	bash toolchain/build-opengnm-psbc-ps5.sh
	PS5_MESA_CROSS_FILE="$(PS5_PAYLOAD_SDK)/toolchain/prospero.ini" bash toolchain/build-mesa-ps5.sh
	$(MAKE) test-compiler
	bash toolchain/install-ps5-opengl-gl46.sh $(PS5_OPENGL_SDK_PREFIX)
	python3 tests/ps5/verify_gl46_link_surface.py
	python3 tests/ps5/verify_gl33_capability_audit.py
imgui-demo:
	bash tools/build-native-test-app.sh egl_public_core33_imgui_tv
nanovg:
	bash tools/build-native-test-app.sh egl_public_core33_nanovg
sokol:
	bash tools/build-native-test-app.sh egl_public_core33_sokol
sokol-cube:
	bash tools/build-native-test-app.sh egl_public_core33_sokol_cube
test-sokol-cube:
	bash tools/test-sokol-cube-host.sh
cubes:
	bash tools/build-native-test-app.sh egl_public_core33_cubes
gl46-demo:
	bash tools/build-native-test-app.sh egl_public_gl46_compute_cubes
showcase:
	bash tools/build-native-test-app.sh egl_public_gl46_showcase

# The demo app: the showcase built against an SDK (the one built here unless
# PS5_OPENGL_PREFIX names another), with its pinned boilerplate fetched below
# build/. DEMO_VERSION names the zip in build/demo/.
PS5_OPENGL_PREFIX ?= $(CURDIR)/build/sdk/ps5-opengl-gl46
DEMO_VERSION ?= dev
# The fetched boilerplate's payload SDK is used unless PS5_PAYLOAD_SDK is set
# explicitly (the default above names a sibling checkout that may not exist).
DEMO_PAYLOAD_SDK = $(if $(filter file,$(origin PS5_PAYLOAD_SDK)),$$template/.deps/native/ps5-payload-sdk,$(PS5_PAYLOAD_SDK))
demo:
	template=$$(bash tools/fetch-native-boilerplate.sh) && sdk="$(DEMO_PAYLOAD_SDK)" && \
	  { test -f "$(PS5_OPENGL_PREFIX)/manifest.sha256" || \
	    $(MAKE) sdk-gl46 PS5_NATIVE_APP_TEMPLATE="$$template" PS5_PAYLOAD_SDK="$$sdk"; } && \
	  PS5_OPENGL_PREFIX="$(PS5_OPENGL_PREFIX)" PS5_NATIVE_APP_TEMPLATE="$$template" \
	  PS5_PAYLOAD_SDK="$$sdk" bash tools/build-native-test-app.sh egl_public_gl46_showcase
	bash tools/package-demo-app.sh "$(DEMO_VERSION)"
test-cubes:
	bash tools/test-cubes-host.sh
test-glsl:
	bash tools/test-glsl-host.sh
test-multidraw:
	bash tools/test-multidraw-host.sh
test-layered-mip:
	bash tools/test-layered-mip-host.sh
test-depth-targets:
	python3 tests/ps5/test_framebuffer_layer_query.py
	bash tools/test-depth-targets-host.sh
test-staging:
	python3 tests/ps5/test_depth_clear_fill.py
	python3 tests/ps5/test_depth_layer_layout.py
	python3 tests/ps5/test_color_msaa_layer_layout.py
	python3 tests/ps5/test_depth_subresources.py
	python3 tests/ps5/test_generate_mipmap.py
	python3 tests/ps5/test_depth_staging_alignment.py
	python3 tests/ps5/test_depth_blit_layers.py
	python3 tests/ps5/test_depth_msaa_array.py
	bash tools/test-depth-array-samples-host.sh
	bash tools/test-msaa-depth-array-host.sh
	bash tools/test-depth-mip-blit-host.sh
	bash tools/test-gpu-blit-host.sh
	bash tools/test-gpu-blit-extended-host.sh
	bash tools/test-gpu-clear-extended-host.sh
	bash tools/test-native-color-formats-host.sh
	bash tools/test-gpu-mipmap-host.sh
	bash tools/test-gpu-transfer-regression-host.sh
	bash tools/test-staging-host.sh
test:
	python3 tests/ps5/test_draw_gpu_timing.py
	python3 -m unittest discover -s tools -p 'test_*.py'
	python3 tests/ps5/test_egl_drawable.py
	python3 tests/ps5/test_egl_context_attributes.py
	python3 tools/test_sdl_sdk.py
	python3 tests/ps5/test_gpu_clear_state.py
	python3 tests/ps5/test_vertex_layout_state.py
	python3 tests/ps5/test_vertex_buffer_references.py
	python3 tests/ps5/test_gpu_blit.py
	python3 tests/ps5/test_linear_color_targets.py
	python3 tests/ps5/test_render_target_extents.py
	python3 tests/ps5/test_draw_profile.py
	python3 tests/ps5/test_display_modes.py
	python3 tests/ps5/test_framebuffer_fallbacks.py
	python3 tests/ps5/test_gpu_present.py
	python3 tests/ps5/test_submit_batch_probe.py
	python3 tests/ps5/test_native_work_pool.py
	python3 tests/ps5/test_multidraw_lifetime.py
	python3 tests/ps5/test_command_groups.py
	python3 tests/ps5/test_submit_retirement.py
	python3 tests/ps5/test_present_shutdown.py
	python3 tests/ps5/test_transfer_staging.py
	python3 tests/ps5/test_color_staging.py
	python3 tests/ps5/test_resource_release.py
	python3 tests/ps5/test_buffer_arena.py
	python3 tests/ps5/test_draw_flush_ranges.py
	python3 tests/ps5/test_buffer_publication.py
	python3 tests/ps5/test_compute_publication.py
	python3 tests/ps5/test_resource_gpu_access.py
	python3 tests/ps5/test_index_bounds.py
	python3 tests/ps5/test_descriptor_reuse.py
	python3 tests/ps5/test_used_buffer_retention.py
	python3 tests/ps5/test_query_completion_scope.py
	python3 tests/ps5/test_identical_image_copy.py
	python3 tests/ps5/test_staging_reuse.py
	python3 tests/ps5/test_batch_texture_flush.py
	python3 tests/ps5/test_native_color_layout.py
	python3 tests/ps5/test_native_srgb_transfer.py
	python3 tools/test_staging_profile.py
	python3 tests/ps5/test_gpu_memory.py
	python3 tests/ps5/test_app_heap.py
	python3 tools/summarize-imgui-profile.py --self-test
	python3 tools/summarize-app-heap.py --self-test
	python3 tools/summarize-cubes.py --self-test
	python3 tools/verify-cts-candidate.py --self-test
	python3 tools/verify-published-validation.py
	python3 tools/verify-cts-official-run.py validation/2026-10-01-gl46-conformance
test-imgui:
	bash tools/test-imgui-host.sh
	bash tools/test-imgui-host.sh --tv-demo
	python3 tests/ps5/test_imgui_egl_cleanup.py
test-compiler:
	python3 tests/ps5/test_shader_cache.py
	python3 tools/fetch-sources.py --verify-psbc
	python3 tests/ps5/test_fragment_exports.py
	python3 tests/ps5/test_shader_ballot.py
	python3 tests/ps5/test_meta_vertex_inputs.py
	python3 tests/ps5/test_unused_primitive_export.py
	python3 tests/ps5/test_vertex_constants.py
	python3 tests/ps5/test_geometry_texture_bindings.py
	python3 tests/ps5/test_buffer_array_lowering.py
	python3 tests/ps5/test_compute_metadata.py
	python3 tests/ps5/test_descriptor_snapshot.py
	python3 tests/ps5/test_compute_bindings.py
	python3 tests/ps5/test_compute_sampler_lod.py
	python3 tests/ps5/test_compute_render.py
	python3 tests/ps5/test_compute_api_caps.py
	python3 tests/ps5/test_compiler_descriptor_sizes.py
	python3 tests/ps5/test_varying_clone_swizzles.py
	python3 tests/ps5/test_tessellation_link.py
	python3 tests/ps5/test_tess_capture.py
