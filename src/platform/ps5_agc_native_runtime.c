// PS5 OpenGL - OpenGL implementation for PlayStation 5.
// Copyright (C) 2026 BlackBearReloaded
// SPDX-License-Identifier: GPL-3.0-or-later

/* Clean Mesa/ACO -> AGC triangle proof.
 *
 * The default build constructs and captures the complete draw command stream
 * but never flips or submits it. Define AGC_TRIANGLE_SUBMIT for the one-frame
 * hardware proof.
 */

#include <dlfcn.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#if defined(PS5_ASYNC_NATIVE_PREP) && \
    (!defined(PS5_NATIVE_TITLE_RUNTIME) || !defined(PS5_MULTIDRAW_BATCH) || \
     !defined(AGC_RUNTIME_PACKAGES) || !defined(AGC_TRIANGLE_SUBMIT))
#error "Async native preparation requires native multidraw package submission"
#endif
#ifdef PS5_ASYNC_NATIVE_PREP
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/param.h>
#include <sys/cpuset.h>
#endif

#include <ps5/kernel.h>
#ifdef PS5_NATIVE_TITLE_RUNTIME
#include "util/cache_ops.h"
#endif

#if defined(PS5_DRAW_PROFILE) && (!defined(AGC_RUNTIME_PACKAGES) || !defined(AGC_TRIANGLE_SUBMIT))
#error "Draw profiling requires the submitting runtime backend"
#endif
#if defined(PS5_DRAW_GPU_TIMESTAMPS) && (!defined(PS5_DRAW_PROFILE) || !defined(PS5_MULTIDRAW_BATCH))
#error "GPU timestamp diagnostic requires profiling and native batching"
#endif
#ifndef PS5_DRAW_PROFILE
#define PS5_PROFILE_MARK(i) ((void)0)
#endif
#if defined(PS5_MULTIDRAW_BATCH) && (!defined(PS5_NATIVE_TITLE_RUNTIME) || !defined(AGC_RUNTIME_PACKAGES) || !defined(AGC_TRIANGLE_SUBMIT) || defined(PS5_DRAW_BATCH_PROBE))
#error "Multi-draw batching requires the native runtime without the repeat probe"
#endif
#if defined(PS5_GPU_PRESENT_BATCH) && !defined(PS5_DEFERRED_DRAW_BATCH)
#error "GPU presentation batching requires the deferred native draw queue"
#endif
#ifdef PS5_DRAW_BATCH_PROBE
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || !defined(AGC_RUNTIME_PACKAGES) || !defined(AGC_TRIANGLE_SUBMIT)
#error "Batch probe requires the native submitting runtime"
#endif
#include "util/os_time.h"
/* Diagnostic only: repeated opaque draws, never a general GL optimization. */
static unsigned runtime_batch_probe_sequence;
static unsigned runtime_batch_probe_repeats(void)
{
    static const unsigned counts[] = {1, 2, 8};
    return counts[runtime_batch_probe_sequence++ % 3];
}
#endif

static _Thread_local uint32_t runtime_ngg_ge_pc_alloc;
static _Thread_local uint32_t runtime_ngg_ge_pc_alloc_valid;

int ps5_agc_gate2_set_ngg_control(uint32_t valid, uint32_t ge_pc_alloc)
{
    if (valid > 1u || ge_pc_alloc > UINT32_C(0x7ff) ||
        (!valid && ge_pc_alloc))
        return -1;
    runtime_ngg_ge_pc_alloc_valid = valid;
    runtime_ngg_ge_pc_alloc = ge_pc_alloc;
    return 0;
}

#if defined(AGC_RUNTIME_PACKAGES)
#define RUNTIME_MAX_VIEWPORTS 16
struct runtime_draw_inputs {
    const uint8_t *vs_package, *ps_package, *hs_package;
    unsigned vs_package_len, ps_package_len, hs_package_len;
    uint8_t *framebuffer;
    size_t framebuffer_size;
    uint32_t vertex_user_data[32], hull_user_data[32], pixel_user_data[32];
    unsigned vertex_user_data_count, hull_user_data_count, pixel_user_data_count;
    const void *index_buffer;
    unsigned index_count, index_size;
    uint32_t primitive_type;
    unsigned draw_count;
    void *depth_buffer, *stencil_buffer;
    size_t depth_buffer_size, stencil_buffer_size;
    unsigned depth_samples;
    uint32_t depth_control, depth_view, stencil_control;
    uint32_t stencil_refmask, stencil_refmask_bf;
    uint32_t blend_control, target_mask, color_control, color_control_valid;
    uint32_t blend_color[4];
    uint32_t viewport[RUNTIME_MAX_VIEWPORTS][8];
    uint32_t scissor[RUNTIME_MAX_VIEWPORTS][2], generic_scissor[2];
    unsigned viewport_count;
    uint32_t rasterizer_control, rasterizer_valid;
    uint32_t point_line[3], point_line_valid;
    uint32_t interp_control, interp_control_valid, point_coord_input;
    uint32_t polygon_offset[6], polygon_offset_valid;
    uint32_t hs_rsrc2, ls_hs_config, tf_param;
    unsigned patch_vertices;
};
static _Thread_local struct runtime_draw_inputs runtime_draw_state = {
    .index_size = sizeof(uint16_t), .primitive_type = 4, .draw_count = 3,
    .depth_samples = 1, .target_mask = 0xf,
    .color_control = 0x00cc0010,
    .viewport = {{0x44700000, 0x44700000, 0xc4070000, 0x44070000,
                  0x3f800000, 0, 0, 0x3f800000}},
    .scissor = {{0x80000000, 0x04380780}},
    .generic_scissor = {0x80000000, 0x04380780},
    .viewport_count = 1,
    .point_line = {0x00080008, 0x00080008, 0x00000008},
};
#define runtime_vs_package runtime_draw_state.vs_package
#define runtime_ps_package runtime_draw_state.ps_package
#define runtime_hs_package runtime_draw_state.hs_package
#define runtime_vs_package_len runtime_draw_state.vs_package_len
#define runtime_ps_package_len runtime_draw_state.ps_package_len
#define runtime_hs_package_len runtime_draw_state.hs_package_len
#define runtime_framebuffer runtime_draw_state.framebuffer
#define runtime_framebuffer_size runtime_draw_state.framebuffer_size
#define runtime_vertex_user_data runtime_draw_state.vertex_user_data
#define runtime_hull_user_data runtime_draw_state.hull_user_data
#define runtime_pixel_user_data runtime_draw_state.pixel_user_data
#define runtime_vertex_user_data_count runtime_draw_state.vertex_user_data_count
#define runtime_hull_user_data_count runtime_draw_state.hull_user_data_count
#define runtime_pixel_user_data_count runtime_draw_state.pixel_user_data_count
#define runtime_index_buffer runtime_draw_state.index_buffer
#define runtime_index_count runtime_draw_state.index_count
#define runtime_index_size runtime_draw_state.index_size
#define runtime_primitive_type runtime_draw_state.primitive_type
#define runtime_draw_count runtime_draw_state.draw_count
#define runtime_depth_buffer runtime_draw_state.depth_buffer
#define runtime_depth_buffer_size runtime_draw_state.depth_buffer_size
#define runtime_depth_samples runtime_draw_state.depth_samples
#define runtime_stencil_buffer runtime_draw_state.stencil_buffer
#define runtime_stencil_buffer_size runtime_draw_state.stencil_buffer_size
#define runtime_depth_control runtime_draw_state.depth_control
#define runtime_depth_view runtime_draw_state.depth_view
#define runtime_stencil_control runtime_draw_state.stencil_control
#define runtime_stencil_refmask runtime_draw_state.stencil_refmask
#define runtime_stencil_refmask_bf runtime_draw_state.stencil_refmask_bf
#define runtime_blend_control runtime_draw_state.blend_control
#define runtime_target_mask runtime_draw_state.target_mask
#define runtime_color_control runtime_draw_state.color_control
#define runtime_color_control_valid runtime_draw_state.color_control_valid
#define runtime_blend_color runtime_draw_state.blend_color
#define runtime_viewport runtime_draw_state.viewport
#define runtime_scissor runtime_draw_state.scissor
#define runtime_generic_scissor runtime_draw_state.generic_scissor
#define runtime_viewport_count runtime_draw_state.viewport_count
#define runtime_rasterizer_control runtime_draw_state.rasterizer_control
#define runtime_rasterizer_valid runtime_draw_state.rasterizer_valid
#define runtime_point_line runtime_draw_state.point_line
#define runtime_point_line_valid runtime_draw_state.point_line_valid
#define runtime_interp_control runtime_draw_state.interp_control
#define runtime_interp_control_valid runtime_draw_state.interp_control_valid
#define runtime_point_coord_input runtime_draw_state.point_coord_input
#define runtime_polygon_offset runtime_draw_state.polygon_offset
#define runtime_polygon_offset_valid runtime_draw_state.polygon_offset_valid
#define runtime_hs_rsrc2 runtime_draw_state.hs_rsrc2
#define runtime_ls_hs_config runtime_draw_state.ls_hs_config
#define runtime_tf_param runtime_draw_state.tf_param
#define runtime_patch_vertices runtime_draw_state.patch_vertices

int ps5_agc_gate2_set_packages(const void *vs, size_t vs_size,
                               const void *ps, size_t ps_size)
{
    if (!vs || !ps || !vs_size || !ps_size ||
        vs_size > UINT32_MAX || ps_size > UINT32_MAX)
        return -1;
    runtime_vs_package = vs;
    runtime_vs_package_len = (unsigned int)vs_size;
    runtime_ps_package = ps;
    runtime_ps_package_len = (unsigned int)ps_size;
    return 0;
}

int ps5_agc_gate2_set_tessellation(const void *hs, size_t hs_size,
                                   uint32_t hs_rsrc2,
                                   uint32_t ls_hs_config,
                                   uint32_t tf_param,
                                   unsigned int patch_vertices)
{
    if ((!hs != !hs_size) || hs_size > UINT32_MAX ||
        (hs && (!patch_vertices || patch_vertices > 32)) ||
        (!hs && patch_vertices))
        return -1;
    runtime_hs_package = hs;
    runtime_hs_package_len = (unsigned int)hs_size;
    runtime_hs_rsrc2 = hs_rsrc2;
    runtime_ls_hs_config = ls_hs_config;
    runtime_tf_param = tf_param;
    runtime_patch_vertices = patch_vertices;
    return 0;
}

/* Whether the registered scanout pool may still hold dirty CPU cache lines from
 * before the GPU first wrote it: set for each new pool or registration, cleared
 * by its one whole-pool flush. Gallium's CPU writers (the clear fallback,
 * transfers) flush the ranges they modify, and reads invalidate theirs, so an
 * unbatched GPU write to a pool already flushed needs no second 2-buffer flush.
 * Without this, every unbatched scanout write (each glClear of the window)
 * flushed the whole pool: about 8 ms a 1080p clear on the console. */
static int runtime_scanout_flush_needed = 1;

int ps5_agc_gate2_set_framebuffer(void *framebuffer, size_t size)
{
    if (!framebuffer || !size)
        return -1;
    if (framebuffer != runtime_framebuffer || size != runtime_framebuffer_size)
        runtime_scanout_flush_needed = 1;
    runtime_framebuffer = framebuffer;
    runtime_framebuffer_size = size;
    return 0;
}

int ps5_agc_gate2_set_vertex_user_data(const uint32_t *values,
                                       unsigned int count)
{
    if (count > sizeof(runtime_vertex_user_data) /
                    sizeof(runtime_vertex_user_data[0]) ||
        (count && !values))
        return -1;
    memset(runtime_vertex_user_data, 0, sizeof(runtime_vertex_user_data));
    if (count)
        memcpy(runtime_vertex_user_data, values, count * sizeof(values[0]));
    runtime_vertex_user_data_count = count;
    return 0;
}

int ps5_agc_gate2_set_hull_user_data(const uint32_t *values,
                                     unsigned int count)
{
    if (count > sizeof(runtime_hull_user_data) /
                    sizeof(runtime_hull_user_data[0]) ||
        (count && !values))
        return -1;
    memset(runtime_hull_user_data, 0, sizeof(runtime_hull_user_data));
    if (count)
        memcpy(runtime_hull_user_data, values, count * sizeof(values[0]));
    runtime_hull_user_data_count = count;
    return 0;
}

int ps5_agc_gate2_set_index_buffer_typed(const void *indices,
                                         unsigned int index_count,
                                         unsigned int index_size)
{
    if ((!indices && index_count) ||
        (indices && index_count != runtime_draw_count) ||
        (index_size != sizeof(uint16_t) &&
         index_size != sizeof(uint32_t)))
        return -1;
    runtime_index_buffer = indices;
    runtime_index_count = index_count;
    runtime_index_size = index_size;
    return 0;
}

int ps5_agc_gate2_set_index_buffer(const void *indices,
                                   unsigned int index_count)
{
    return ps5_agc_gate2_set_index_buffer_typed(
        indices, index_count, sizeof(uint16_t));
}

int ps5_agc_gate2_set_draw_state(uint32_t primitive_type,
                                 unsigned int draw_count)
{
    switch (primitive_type) {
    case 1: /* POINTLIST */
        if (draw_count < 1)
            return -1;
        break;
    case 2: /* LINELIST */
        if (draw_count < 2 || (draw_count & 1))
            return -1;
        break;
    case 3: /* LINESTRIP */
        if (draw_count < 2)
            return -1;
        break;
    case 4: /* TRILIST */
        if (draw_count < 3 || draw_count % 3)
            return -1;
        break;
    case 5: /* TRIFAN */
    case 6: /* TRISTRIP */
        if (draw_count < 3)
            return -1;
        break;
    case 9: /* PATCH */
        if (!runtime_patch_vertices || draw_count < runtime_patch_vertices ||
            draw_count % runtime_patch_vertices)
            return -1;
        break;
    case 10: /* LINELIST_ADJ */
        if (draw_count < 4 || draw_count % 4)
            return -1;
        break;
    case 11: /* LINESTRIP_ADJ */
        if (draw_count < 4)
            return -1;
        break;
    case 12: /* TRILIST_ADJ */
        if (draw_count < 6 || draw_count % 6)
            return -1;
        break;
    case 13: /* TRISTRIP_ADJ */
        if (draw_count < 6 || (draw_count & 1))
            return -1;
        break;
    default:
        return -1;
    }
    runtime_primitive_type = primitive_type;
    runtime_draw_count = draw_count;
    return 0;
}

int ps5_agc_gate2_set_pixel_user_data(const uint32_t *values,
                                      unsigned int count)
{
    if (count > sizeof(runtime_pixel_user_data) /
                    sizeof(runtime_pixel_user_data[0]) ||
        (count && !values))
        return -1;
    memset(runtime_pixel_user_data, 0, sizeof(runtime_pixel_user_data));
    if (count)
        memcpy(runtime_pixel_user_data, values, count * sizeof(values[0]));
    runtime_pixel_user_data_count = count;
    return 0;
}

int ps5_agc_gate2_set_depth_buffer(void *depth, size_t size,
                                   uint32_t depth_control)
{
    if (!depth) {
        if (size || depth_control)
            return -1;
        runtime_depth_buffer = NULL;
        runtime_depth_buffer_size = 0;
        runtime_stencil_buffer = NULL;
        runtime_stencil_buffer_size = 0;
        runtime_depth_control = 0;
        runtime_depth_view = 0;
        runtime_stencil_control = 0;
        runtime_stencil_refmask = 0;
        runtime_stencil_refmask_bf = 0;
        return 0;
    }
    if (size < 0xa00000u || ((uintptr_t)depth & 0x1fffffu))
        return -1;
    runtime_depth_buffer = depth;
    runtime_depth_buffer_size = size;
    runtime_stencil_buffer = NULL;
    runtime_stencil_buffer_size = 0;
    runtime_depth_control = depth_control;
    runtime_stencil_control = 0;
    runtime_stencil_refmask = 0;
    runtime_stencil_refmask_bf = 0;
    return 0;
}

int ps5_agc_gate2_set_depth_target_view(uint32_t view)
{
    const uint32_t allowed = UINT32_C(0x000007ff) |
                             (UINT32_C(0x000007ff) << 13);
    const uint32_t first = view & UINT32_C(0x000007ff);
    const uint32_t last = (view >> 13) & UINT32_C(0x000007ff);

    if ((view & ~allowed) || last < first)
        return -1;
    runtime_depth_view = view;
    return 0;
}

int ps5_agc_gate2_set_depth_stencil_buffer(
    void *depth, size_t depth_size, void *stencil, size_t stencil_size,
    uint32_t depth_control, uint32_t stencil_control,
    uint32_t stencil_refmask, uint32_t stencil_refmask_bf)
{
    if (!depth || !stencil || depth_size < 0xa00000u ||
        stencil_size < 0x280000u ||
        ((uintptr_t)depth & 0x1fffffu) ||
        ((uintptr_t)stencil & 0xffffu))
        return -1;
    runtime_depth_buffer = depth;
    runtime_depth_buffer_size = depth_size;
    runtime_stencil_buffer = stencil;
    runtime_stencil_buffer_size = stencil_size;
    runtime_depth_control = depth_control;
    runtime_stencil_control = stencil_control;
    runtime_stencil_refmask = stencil_refmask;
    runtime_stencil_refmask_bf = stencil_refmask_bf;
    return 0;
}

int ps5_agc_gate2_set_graphics_state(
    uint32_t blend_control, uint32_t target_mask, uint32_t color_control,
    uint32_t color_control_valid,
    const uint32_t blend_color[4], const uint32_t viewport[8],
    const uint32_t scissor[2], uint32_t rasterizer_control,
    uint32_t rasterizer_valid, const uint32_t polygon_offset[6],
    uint32_t polygon_offset_valid)
{
    if (!blend_color || !viewport || !scissor || !polygon_offset ||
        (target_mask & ~0xfu) || color_control_valid > 1u ||
        rasterizer_valid > 1u ||
        polygon_offset_valid > 1u)
        return -1;
    runtime_blend_control = blend_control;
    runtime_target_mask = target_mask;
    runtime_color_control = color_control;
    runtime_color_control_valid = color_control_valid;
    memcpy(runtime_blend_color, blend_color, sizeof(runtime_blend_color));
    memcpy(runtime_viewport[0], viewport, sizeof(runtime_viewport[0]));
    memcpy(runtime_scissor[0], scissor, sizeof(runtime_scissor[0]));
    if (runtime_viewport_count == 1)
        memcpy(runtime_generic_scissor, scissor,
               sizeof(runtime_generic_scissor));
    runtime_rasterizer_control = rasterizer_control;
    runtime_rasterizer_valid = rasterizer_valid;
    memcpy(runtime_polygon_offset, polygon_offset,
           sizeof(runtime_polygon_offset));
    runtime_polygon_offset_valid = polygon_offset_valid;
    return 0;
}

int ps5_agc_gate2_set_viewport_states(
    const uint32_t viewport[RUNTIME_MAX_VIEWPORTS][8],
    const uint32_t scissor[RUNTIME_MAX_VIEWPORTS][2], unsigned count)
{
    unsigned minx = UINT16_MAX, miny = UINT16_MAX, maxx = 0, maxy = 0;

    if (!viewport || !scissor || !count || count > RUNTIME_MAX_VIEWPORTS)
        return -1;
    memcpy(runtime_viewport, viewport, count * sizeof(runtime_viewport[0]));
    memcpy(runtime_scissor, scissor, count * sizeof(runtime_scissor[0]));
    for (unsigned i = 0; i < count; ++i) {
        unsigned x = scissor[i][0] & UINT32_C(0x7fff);
        unsigned y = (scissor[i][0] >> 16) & UINT32_C(0x7fff);
        unsigned right = scissor[i][1] & UINT32_C(0x7fff);
        unsigned bottom = (scissor[i][1] >> 16) & UINT32_C(0x7fff);

        if (x < minx) minx = x;
        if (y < miny) miny = y;
        if (right > maxx) maxx = right;
        if (bottom > maxy) maxy = bottom;
    }
    runtime_generic_scissor[0] = UINT32_C(0x80000000) | minx | (miny << 16);
    runtime_generic_scissor[1] = maxx | (maxy << 16);
    runtime_viewport_count = count;
    return 0;
}

int ps5_agc_gate2_set_point_line_state(const uint32_t values[3],
                                       uint32_t valid)
{
    if (!values || valid > 1u)
        return -1;
    memcpy(runtime_point_line, values, sizeof(runtime_point_line));
    runtime_point_line_valid = valid;
    return 0;
}

int ps5_agc_gate2_set_interp_control(uint32_t control, uint32_t valid)
{
    if (valid > 1u)
        return -1;
    runtime_interp_control = control;
    runtime_interp_control_valid = valid;
    return 0;
}

int ps5_agc_gate2_set_point_coord_input(uint32_t enabled)
{
    /* Zero disables; otherwise one-based PS attribute index (32 inputs). */
    if (enabled > 32u)
        return -1;
    runtime_point_coord_input = enabled;
    return 0;
}

#define VS_PACKAGE runtime_vs_package
#define VS_PACKAGE_LEN runtime_vs_package_len
#define PS_PACKAGE runtime_ps_package
#define PS_PACKAGE_LEN runtime_ps_package_len
#define LOG_PREFIX "[agc_triangle_runtime]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-triangle-runtime-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-triangle-runtime-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-triangle-runtime-target.bgra"
#elif defined(AGC_DEPTH_TEST) || defined(AGC_CLEAR_TEST)
#include "opengl_gate4_depth_vs.inc"
#include "opengl_gate4_depth_ps.inc"
#include "opengl_gate4_depth_bindings.inc"
#define VS_PACKAGE gate4_depth_vs_package
#define VS_PACKAGE_LEN gate4_depth_vs_package_len
#define PS_PACKAGE gate4_depth_ps_package
#define PS_PACKAGE_LEN gate4_depth_ps_package_len
#define GATE3_BIND_ADDRESS32_HI GATE4_DEPTH_ADDRESS32_HI
#define GATE3_BIND_USER_SGPR_COUNT GATE4_DEPTH_USER_SGPR_COUNT
#define GATE3_BIND_VERTEX_BUFFER_USER_DWORD \
    GATE4_DEPTH_VERTEX_BUFFER_USER_DWORD
#define AGC_DEPTH_TEST_VARIANT 1
#define AGC_OFFSCREEN_COLOR_VARIANT 1
#define AGC_TRIANGLE_EXPECTS_VARIATION 1
#ifdef AGC_CLEAR_TEST
#define AGC_CLEAR_TEST_VARIANT 1
#define LOG_PREFIX "[agc_clear]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-clear-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-clear-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-clear-target.bgra"
#define DEPTH_DUMP_PATH "/data/VdecHello/opengl33-clear-depth.raw"
#else
#define LOG_PREFIX "[agc_depth]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-depth-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-depth-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-depth-target.bgra"
#define DEPTH_DUMP_PATH "/data/VdecHello/opengl33-depth-target.raw"
#endif
#elif defined(AGC_TEXTURE_QUAD) || defined(AGC_TEXTURE_NEAREST) || \
    defined(AGC_TEXTURE_VIEWPORT_HALF) || \
    defined(AGC_TEXTURE_SCISSOR_HALF) || \
    defined(AGC_BLEND_DISABLED) || defined(AGC_BLEND_SRC_ALPHA) || \
    defined(AGC_BLEND_ADDITIVE) || defined(AGC_FRAME_SLOTS) || \
    defined(AGC_OFFSCREEN_COLOR) || defined(AGC_RENDER_TO_TEXTURE)
#include "opengl_gate3_texture_vs.inc"
#include "opengl_gate3_texture_ps.inc"
#include "opengl_gate3_texture_vs_bindings.inc"
#include "opengl_gate3_texture_ps_bindings.inc"
#define VS_PACKAGE gate3_texture_vs_package
#define VS_PACKAGE_LEN gate3_texture_vs_package_len
#define PS_PACKAGE gate3_texture_ps_package
#define PS_PACKAGE_LEN gate3_texture_ps_package_len
#define GATE3_BIND_ADDRESS32_HI GATE3_TEXTURE_VS_ADDRESS32_HI
#define GATE3_BIND_USER_SGPR_COUNT GATE3_TEXTURE_VS_USER_SGPR_COUNT
#define GATE3_BIND_VERTEX_BUFFER_USER_DWORD \
    GATE3_TEXTURE_VS_VERTEX_BUFFER_USER_DWORD
#if defined(AGC_FRAME_SLOTS)
#define AGC_FRAME_SLOTS_VARIANT 1
#ifdef AGC_4K
#define LOG_PREFIX "[agc_frame_slots_4k]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-frame-slots-4k-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-frame-slots-4k-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-frame-slots-4k-target.bgra"
#else
#define LOG_PREFIX "[agc_frame_slots]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-frame-slots-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-frame-slots-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-frame-slots-target.bgra"
#endif
#else
#define AGC_TEXTURE_VARIANT 1
#endif
#if defined(AGC_FRAME_SLOTS)
#elif defined(AGC_OFFSCREEN_COLOR)
#define AGC_OFFSCREEN_COLOR_VARIANT 1
#define LOG_PREFIX "[agc_offscreen_color]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-offscreen-color-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-offscreen-color-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-offscreen-color-target.bgra"
#elif defined(AGC_RENDER_TO_TEXTURE)
#define AGC_OFFSCREEN_COLOR_VARIANT 1
#define AGC_RENDER_TO_TEXTURE_VARIANT 1
#define LOG_PREFIX "[agc_render_to_texture]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-render-to-texture-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-render-to-texture-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-render-to-texture-target.bgra"
#define SOURCE_DUMP_PATH "/data/VdecHello/opengl33-render-to-texture-source.bgra"
#elif defined(AGC_BLEND_DISABLED)
#define AGC_BLEND_VARIANT 1
#define BLEND_SECOND_CONTROL UINT32_C(0x00000000)
#define LOG_PREFIX "[agc_blend_disabled]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-blend-disabled-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-blend-disabled-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-blend-disabled-target.bgra"
#elif defined(AGC_BLEND_SRC_ALPHA)
#define AGC_BLEND_VARIANT 1
#define BLEND_SECOND_CONTROL UINT32_C(0x40000504)
#ifdef AGC_4K
#define LOG_PREFIX "[agc_blend_src_alpha_4k]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-blend-src-alpha-4k-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-blend-src-alpha-4k-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-blend-src-alpha-4k-target.bgra"
#else
#define LOG_PREFIX "[agc_blend_src_alpha]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-blend-src-alpha-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-blend-src-alpha-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-blend-src-alpha-target.bgra"
#endif
#elif defined(AGC_BLEND_ADDITIVE)
#define AGC_BLEND_VARIANT 1
#define BLEND_SECOND_CONTROL UINT32_C(0x40000101)
#define LOG_PREFIX "[agc_blend_additive]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-blend-additive-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-blend-additive-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-blend-additive-target.bgra"
#elif defined(AGC_TEXTURE_VIEWPORT_HALF)
#define LOG_PREFIX "[agc_texture_viewport_half]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-texture-viewport-half-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-texture-viewport-half-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-texture-viewport-half-target.bgra"
#elif defined(AGC_TEXTURE_SCISSOR_HALF)
#define LOG_PREFIX "[agc_texture_scissor_half]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-texture-scissor-half-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-texture-scissor-half-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-texture-scissor-half-target.bgra"
#elif defined(AGC_TEXTURE_NEAREST)
#define LOG_PREFIX "[agc_texture_nearest]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-texture-nearest-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-texture-nearest-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-texture-nearest-target.bgra"
#else
#define LOG_PREFIX "[agc_texture_quad]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-texture-quad-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-texture-quad-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-texture-quad-target.bgra"
#endif
#define AGC_TRIANGLE_EXPECTS_VARIATION 1
#elif defined(AGC_UNIFORM_QUAD) || defined(AGC_UNIFORM_SHIFTED_QUAD)
#include "opengl_gate3_uniform_vs.inc"
#include "opengl_gate3_uniform_ps.inc"
#include "opengl_gate3_uniform_bindings.inc"
#define VS_PACKAGE gate3_uniform_vs_package
#define VS_PACKAGE_LEN gate3_uniform_vs_package_len
#define PS_PACKAGE gate3_uniform_ps_package
#define PS_PACKAGE_LEN gate3_uniform_ps_package_len
#define GATE3_BIND_ADDRESS32_HI GATE3_UNIFORM_ADDRESS32_HI
#define GATE3_BIND_USER_SGPR_COUNT GATE3_UNIFORM_USER_SGPR_COUNT
#define GATE3_BIND_VERTEX_BUFFER_USER_DWORD \
    GATE3_UNIFORM_VERTEX_BUFFER_USER_DWORD
#ifdef AGC_UNIFORM_SHIFTED_QUAD
#define LOG_PREFIX "[agc_uniform_shifted]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-uniform-shifted-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-uniform-shifted-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-uniform-shifted-target.bgra"
#else
#define LOG_PREFIX "[agc_uniform_identity]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-uniform-identity-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-uniform-identity-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-uniform-identity-target.bgra"
#endif
#define AGC_TRIANGLE_EXPECTS_VARIATION 1
#elif defined(AGC_INDEXED_QUAD) || defined(AGC_TRIANGLE_INDEX_BUFFER) || \
    defined(AGC_TRIANGLE_VERTEX_BUFFER)
#include "opengl_gate3_vb_vs.inc"
#include "opengl_gate3_vb_ps.inc"
#include "opengl_gate3_vb_bindings.inc"
#define VS_PACKAGE gate3_vb_vs_package
#define VS_PACKAGE_LEN gate3_vb_vs_package_len
#define PS_PACKAGE gate3_vb_ps_package
#define PS_PACKAGE_LEN gate3_vb_ps_package_len
#define GATE3_BIND_ADDRESS32_HI GATE3_VB_ADDRESS32_HI
#define GATE3_BIND_USER_SGPR_COUNT GATE3_VB_USER_SGPR_COUNT
#define GATE3_BIND_VERTEX_BUFFER_USER_DWORD \
    GATE3_VB_VERTEX_BUFFER_USER_DWORD
#ifdef AGC_INDEXED_QUAD
#define LOG_PREFIX "[agc_indexed_quad]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-indexed-quad-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-indexed-quad-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-indexed-quad-target.bgra"
#elif defined(AGC_TRIANGLE_INDEX_BUFFER)
#define LOG_PREFIX "[agc_index_buffer]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-index-buffer-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-index-buffer-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-index-buffer-target.bgra"
#else
#define LOG_PREFIX "[agc_vertex_buffer]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-vertex-buffer-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-vertex-buffer-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-vertex-buffer-target.bgra"
#endif
#define AGC_TRIANGLE_EXPECTS_VARIATION 1
#elif defined(AGC_TRIANGLE_VARYING)
#include "opengl_gate3_vs.inc"
#include "opengl_gate3_ps.inc"
#define VS_PACKAGE gate3_vs_package
#define VS_PACKAGE_LEN gate3_vs_package_len
#define PS_PACKAGE gate3_ps_package
#define PS_PACKAGE_LEN gate3_ps_package_len
#define LOG_PREFIX "[agc_varying]"
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-varying-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-varying-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-varying-target.bgra"
#define AGC_TRIANGLE_EXPECTS_VARIATION 1
#else
#include "opengl_gate1_vs.inc"
#include "opengl_gate1_ps.inc"
#define VS_PACKAGE gate1_vs_package
#define VS_PACKAGE_LEN gate1_vs_package_len
#define PS_PACKAGE gate1_ps_package
#define PS_PACKAGE_LEN gate1_ps_package_len
#ifdef AGC_INVALID_FRAMEBUFFER_TEST
#define LOG_PREFIX "[agc_invalid_framebuffer]"
#elif defined(AGC_DEPTH_DEFAULTS_PROBE)
#define LOG_PREFIX "[agc_depth_defaults]"
#else
#define LOG_PREFIX "[agc_triangle]"
#endif
#define DCB_DUMP_PATH "/data/VdecHello/opengl33-triangle-build.dcb"
#define MEMORY_DUMP_PATH "/data/VdecHello/opengl33-triangle-memory.bin"
#define TARGET_DUMP_PATH "/data/VdecHello/opengl33-triangle-target.bgra"
#endif

#include "ps5_scanout.h"
#define DISPLAY_WIDTH PS5_SCANOUT_WIDTH
#define DISPLAY_HEIGHT PS5_SCANOUT_HEIGHT
#define FRAMEBUFFER_BYTES PS5_SCANOUT_BYTES
#define FRAMEBUFFER_POOL_BYTES PS5_SCANOUT_POOL_BYTES
#define FRAMEBUFFER_ALIGNMENT PS5_SCANOUT_ALIGNMENT
#define OFFSCREEN_BYTES FRAMEBUFFER_BYTES
#define DEPTH_BYTES 0xa00000u
#define DEPTH_ALIGNMENT 0x200000u
#if defined(AGC_FRAME_SLOTS_VARIANT)
#define WORK_BYTES 0x30000u
#else
#define WORK_BYTES 0x10000u
#endif
#define COMMAND_BYTES 0x4000u
#define DIRECT_MEMORY_TYPE 12
#define MAP_PROTECTION 0x33
#define VIDEO_OUT_PIXEL_FORMAT UINT64_C(0x8000000000000000)
#define RENDER_MARKER INT64_C(0x474c3301)
#define AGC_INDEX_SIZE_16 0u
#define AGC_INDEX_SIZE_32 1u
#define TEXTURE_WIDTH 64u
#define TEXTURE_HEIGHT 64u
#define TESS_OFFCHIP_WORKGROUPS 144u
/* Driver-assigned slots can exceed the requested buffering count (CTS used
 * slot 166 with 160 requested). Back the full GFX10.3 ten-bit slot range. */
#define TESS_OFFCHIP_SLOT_COUNT 1024u
#define TESS_OFFCHIP_SLOT_BYTES 32768u
#define TESS_OFFCHIP_BYTES (TESS_OFFCHIP_SLOT_COUNT * TESS_OFFCHIP_SLOT_BYTES)
#define TESS_FACTOR_BYTES 0x1b000u
#define NATIVE_COLOR_FORMAT_RGBA8_UNORM 1u
#define NATIVE_COLOR_SWIZZLE_64KB_R_X 27u

typedef struct native_color_target_spec {
    uintptr_t address;
    size_t allocation_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t samples;
    uint32_t swizzle;
    uint32_t compressed;
} native_color_target_spec_t;

static int validate_color_target_spec(const native_color_target_spec_t *spec)
{
    size_t required_bytes;

    if (!spec)
        return -1;
    if (spec->width != DISPLAY_WIDTH || spec->height != DISPLAY_HEIGHT)
        return -2;
    if (spec->format != NATIVE_COLOR_FORMAT_RGBA8_UNORM)
        return -3;
    if (spec->samples != 1)
        return -4;
    if (spec->swizzle != NATIVE_COLOR_SWIZZLE_64KB_R_X ||
        spec->compressed)
        return -5;
    if (!spec->address || (spec->address & (FRAMEBUFFER_ALIGNMENT - 1u)))
        return -6;
    required_bytes = (size_t)((spec->width + 127u) >> 7) *
                     ((spec->height + 127u) >> 7) * UINT32_C(0x10000);
    if (spec->allocation_bytes < required_bytes)
        return -7;
    return 0;
}

int64_t sceKernelGetDirectMemorySize(void);
int32_t sceKernelAllocateDirectMemory(int64_t, int64_t, size_t, size_t, int,
                                      int64_t *);
int32_t sceKernelMapDirectMemory(void **, size_t, int, int, int64_t, size_t);
int32_t sceKernelReleaseDirectMemory(int64_t, size_t);
int32_t sceKernelUsleep(uint32_t);

typedef struct agc_register {
    uint16_t offset;
    uint16_t padding;
    uint32_t value;
} agc_register_t;

#define RUNTIME_GRAPHICS_STATE_OFFSET 0x6800u
#define RUNTIME_GRAPHICS_STATE_MAX_RECORDS 256u
#define RUNTIME_CONTEXT_STATE_OFFSET 0x7000u
_Static_assert(sizeof(agc_register_t) == 8,
               "unexpected AGC indirect-register record size");
_Static_assert(RUNTIME_GRAPHICS_STATE_OFFSET +
                   RUNTIME_GRAPHICS_STATE_MAX_RECORDS *
                       sizeof(agc_register_t) <=
                   RUNTIME_CONTEXT_STATE_OFFSET,
               "runtime graphics state overlaps context state");

typedef struct agc_command_buffer {
    uint32_t *bottom;
    uint32_t *top;
    uint32_t *up;
    uint32_t *down;
    uintptr_t callback;
    void *user_data;
    uint32_t reserved_dwords;
    uint32_t padding;
} agc_command_buffer_t;

typedef struct agc_submit_description {
    void *words;
    uint32_t word_count;
    uint8_t flag;
    uint8_t padding[3];
} agc_submit_description_t;

typedef struct video_buffer {
    void *data;
    void *metadata;
    void *reserved0;
    void *reserved1;
} video_buffer_t;

typedef struct video_attribute {
    uint8_t bytes[80];
} video_attribute_t;

typedef struct agc_api {
    int (*init)(uint32_t);
    int (*create_shader)(void **, void *, void *);
    int (*link_shaders)(void *, void *, void *, void *, void *, uint32_t);
    void *(*get_defaults)(void);
    uint32_t *(*set_cx)(void *, const void *, uint32_t);
    uint32_t *(*set_sh)(void *, const void *, uint32_t);
    uint32_t *(*set_uc)(void *, const void *, uint32_t);
    uint32_t *(*set_sh_direct)(void *, uint32_t, const uint32_t *, uint32_t);
    uint32_t *(*draw_auto)(void *, uint32_t, uint64_t);
    uint32_t *(*set_index_size)(void *, uint8_t, uint8_t);
    uint32_t *(*set_index_buffer)(void *, void *);
    uint32_t *(*set_index_count)(void *, uint32_t);
    uint32_t *(*draw_index)(void *, uint32_t, void *, uint64_t);
    uint32_t *(*release_mem)(void *, uint8_t, int16_t, uint64_t, int8_t,
                             void *, uint32_t, uint64_t, uint16_t,
                             uint16_t, int8_t, int32_t);
    uint32_t *(*set_flip)(void *, uint32_t, int, uint32_t, int64_t);
    int (*suspend_point)(void);
    uint32_t (*wait_size)(void);
    uint32_t (*wait_rendering)(uint32_t **, uint32_t, uint32_t, uint32_t,
                               int);
    int (*submit)(void *);
    int (*set_tf_ring)(uintptr_t, uint32_t);
    int (*get_tf_ring)(uintptr_t *, uint32_t *);
} agc_api_t;

static uint32_t *set_linkage_uc_state(const agc_api_t *agc,
                                      agc_command_buffer_t *command,
                                      uint8_t *memory)
{
    agc_register_t *uc = (agc_register_t *)(memory + 0x6000);
    uint32_t count = 3;

    if (runtime_ngg_ge_pc_alloc_valid) {
        uc[count++] = (agc_register_t){
            0x0260, 0, runtime_ngg_ge_pc_alloc
        };
    }
    if (runtime_hs_package) {
        /* Mesa si_emit_vgt_flush: reset ring pointers before rewriting VGT
         * UCONFIG state, even after the previous draw has retired. */
        static const uint32_t reset[] = {
            UINT32_C(0xc0004600), UINT32_C(0x0000040f), /* VS_PARTIAL_FLUSH */
            UINT32_C(0xc0004600), UINT32_C(0x00000024), /* VGT_FLUSH */
        };
        if (command->up < command->bottom || command->up > command->down ||
            command->down > command->top || command->down - command->up < 4)
            return NULL;
        memcpy(command->up, reset, sizeof(reset));
        command->up += 4;
        uintptr_t factor = (uintptr_t)(memory + WORK_BYTES +
                                       TESS_OFFCHIP_BYTES);
        uc[count++] = (agc_register_t){0x024e, 0,
                                       TESS_FACTOR_BYTES / 4u};
        uc[count++] = (agc_register_t){0x024f, 0,
                                       TESS_OFFCHIP_WORKGROUPS - 1u};
        uc[count++] = (agc_register_t){0x0250, 0,
                                       (uint32_t)(factor >> 8)};
        uc[count++] = (agc_register_t){0x0261, 0,
                                       (uint32_t)(factor >> 40)};
    }
    return agc->set_uc(command, uc, count);
}

typedef void *(*agc_get_register_defaults2_fn)(void);

typedef struct video_api {
    int (*open)(int32_t, int32_t, int32_t, const void *);
    int (*close)(int32_t);
    int (*set_flip_rate)(int32_t, int32_t);
    void (*set_attribute2)(video_attribute_t *, uint64_t, uint32_t, uint32_t,
                           uint32_t, uint64_t, uint32_t, uint64_t);
    int (*register_buffers2)(int32_t, int32_t, int32_t, video_buffer_t *,
                             int32_t, video_attribute_t *, int32_t, void *);
    int (*unregister_buffers)(int32_t, int32_t);
    int (*submit_flip)(int32_t, int32_t, uint32_t, int64_t);
    int (*is_flip_pending)(int32_t);
    int (*wait_vblank)(int32_t);
    int (*get_flip_status)(int32_t, void *);
} video_api_t;

static int load_apis(void *agc_module, void *driver_module,
                     void *video_module, agc_api_t *agc, video_api_t *video);

static _Thread_local volatile unsigned out_of_space;
static uint8_t command_out_of_space(agc_command_buffer_t *, uint32_t, void *);

#if defined(AGC_RUNTIME_PACKAGES)
static void flush_gpu_data(const void *address, size_t bytes);
#include "ps5_cache_profile.h"
#ifdef PS5_DRAW_PROFILE
#define flush_gpu_data(address, bytes) PS5_CACHE_MEASURE(flush_gpu_data, address, bytes)
#endif

static video_api_t runtime_video_api;
static int runtime_video_handle = -1;
static uint8_t *runtime_video_framebuffer;
static size_t runtime_video_framebuffer_size;
static int runtime_video_registered;
static uint64_t runtime_render_marker = (uint64_t)RENDER_MARKER;
static unsigned runtime_present_count;
static int runtime_agc_initialized;
#ifdef PS5_GPU_PRESENT_BATCH
static int runtime_gpu_present_buffer = -1;
static int runtime_gpu_present_deferred;
static int runtime_gpu_present_is_cpu;
static uint64_t runtime_gpu_present_marker;
static unsigned runtime_gpu_present_count;
#endif

#ifdef PS5_NATIVE_TITLE_RUNTIME
static void runtime_require_retirement(int completed)
{
    if (completed)
        return;
    /* ponytail: fail-stop until device-loss recovery owns every GPU reference.
     * Returning or running atexit handlers could free/reuse in-flight memory. */
    fputs("[ps5-agc] submission retirement failed; terminating application before cleanup\n",
          stderr);
    fflush(stderr);
    _Exit(EXIT_FAILURE);
}
#endif

#include "util/os_time.h"
/* Completion publication is now prompt. Avoid yielding for a short GPU tail;
 * long waits still sleep, and never restart their 200us spin budget. */
static unsigned runtime_completion_pause(int64_t *spin_deadline)
{
    int64_t now = os_time_get_nano();
    if (!*spin_deadline)
        *spin_deadline = now + 200000;
    if (now < *spin_deadline) {
        __builtin_ia32_pause();
        return 0;
    }
    sceKernelUsleep(UINT32_C(1000));
    return 1;
}

/* Publish/read the full 64-bit value in the aligned marker slot. Native
 * publication follows the qualified platform cache policy below. */
static uint32_t *runtime_release_completion(const agc_api_t *agc,
                                            agc_command_buffer_t *command,
                                            volatile uint64_t *marker,
                                            uint32_t value)
{
#ifdef PS5_NATIVE_TITLE_RUNTIME
    extern uint32_t *sceAgcDcbAcquireMem(void *, uint8_t, uint32_t, uint32_t,
                                        uintptr_t, uint64_t, uint32_t);
    /* Flush shader outputs before publishing the CPU fence. The native GL
     * completion path acquires the marker range, then uses cache policy 3
     * through the L2 destination so publication does not retain a cached line. */
    if (!agc->release_mem(command, 40, 0x30c, 0, 0, NULL, 0,
                          0, 0, 0, 0, 0) ||
        !sceAgcDcbAcquireMem(command, 1, 0, 0x9000, (uintptr_t)marker, 32, 400))
        return NULL;
    return agc->release_mem(command, 40, 0, 1, 3, (void *)marker, 2,
                            value, 0, 1, 0, 0);
#else
    return agc->release_mem(command, 40, 0x30c, 0, 0, (void *)marker, 2,
                            value, 0, 0, 3, 0);
#endif
}

/* SuspendPoint ends a native submission period and stages the next period's
 * prologue. It is a frame/lifecycle boundary, not a draw completion operation.
 * GPU release markers still govern every retirement and resource lifetime. */
#ifdef PS5_FRAME_SUSPEND
static int (*runtime_frame_suspend)(void);
static int runtime_defer_suspend(int (*callback)(void))
{
    if (!callback || (runtime_frame_suspend && runtime_frame_suspend != callback))
        return -1;
    runtime_frame_suspend = callback;
    return 0;
}
static int runtime_end_submit_period(void)
{
    if (!runtime_frame_suspend)
        return 0;
    int result = runtime_frame_suspend();
    if (!result)
        runtime_frame_suspend = NULL;
    return result;
}
#endif

#ifdef PS5_MULTIDRAW_BATCH
#ifdef PS5_DRAW_PROFILE
#include "util/os_time.h"
static void runtime_batch_profile_record(const int64_t ticks[5], unsigned sleeps,
                                         int result);
#endif
#ifdef PS5_DRAW_GPU_TIMESTAMPS
/* Diagnostic page follows all shader/tessellation storage; never shares state. */
static void runtime_draw_gpu_observe(void *memory, size_t bytes)
{
    volatile uint64_t *t = (volatile uint64_t *)((uint8_t *)memory + bytes - 0x4000);
    flush_gpu_data((const void *)t, 64);
    if (t[0] && !t[9]) t[9] = os_time_get_nano();
    if (t[2] && !t[10]) t[10] = os_time_get_nano();
}
static void runtime_draw_gpu_timing(void *memory, size_t bytes)
{
    static unsigned samples;
    volatile uint64_t *t = (volatile uint64_t *)((uint8_t *)memory + bytes - 0x4000);
    runtime_draw_gpu_observe(memory, bytes);
    if (++samples <= 64 || samples % 128 == 0)
        printf("[ps5-draw-gpu-time] sample=%u begin=%" PRIu64 " draw=%" PRIu64
               " end=%" PRIu64 " submit_ns=%" PRIu64 " first_ns=%" PRIu64
               " end_seen_ns=%" PRIu64 " cpu_ns=%" PRId64 " valid=%u\n",
               samples, t[0], t[1], t[2], t[8], t[9], t[10], os_time_get_nano(),
               (unsigned)(t[0] != 0 && t[2] >= t[1] && t[1] >= t[0]));
}

#endif
/* The caller holds the queue lock and retains descriptors/resources until
 * the matching FIFO retirement. Queue-full rejection never transfers ownership. */
#ifdef PS5_NATIVE_TITLE_RUNTIME
/* The draw worker takes allocations while the producer retires older batches. */
#ifdef PS5_ASYNC_NATIVE_PREP
static atomic_flag runtime_work_lock = ATOMIC_FLAG_INIT;
#define RUNTIME_WORK_LOCK() do { \
    while (atomic_flag_test_and_set_explicit(&runtime_work_lock, memory_order_acquire)) \
        __builtin_ia32_pause(); \
} while (0)
#define RUNTIME_WORK_UNLOCK() \
    atomic_flag_clear_explicit(&runtime_work_lock, memory_order_release)
#else
#define RUNTIME_WORK_LOCK() ((void)0)
#define RUNTIME_WORK_UNLOCK() ((void)0)
#endif
/* Only confirmed-retired work may enter this cache. */
static struct runtime_work_allocation {
    void *memory;
    int64_t direct;
    size_t bytes;
} runtime_work_free[PS5_MULTIDRAW_BATCH_CAPACITY * PS5_INFLIGHT_BATCH_CAPACITY];
static unsigned runtime_work_free_count;
static size_t runtime_work_free_bytes;

/* Queue-lock protected, immutable after publication. Keep entries until all
 * submissions retire at shutdown; a full cache falls back to the original path. */
static struct runtime_shader_pair {
    const void *vs_key, *ps_key;
    size_t vs_size, ps_size, bytes;
    uint32_t primitive;
    uint8_t *source, *memory;
    int64_t direct;
    void *vertex, *pixel;
} runtime_shader_pairs[2048];
static struct runtime_shader_pair *runtime_shader_pair_hot[4096];
static unsigned runtime_shader_pair_count;
static size_t runtime_shader_pair_bytes;
#ifdef PS5_ASYNC_NATIVE_PREP
static pthread_mutex_t runtime_shader_mutex = PTHREAD_MUTEX_INITIALIZER;
#define RUNTIME_SHADER_LOCK() do { if (pthread_mutex_lock(&runtime_shader_mutex)) abort(); } while (0)
#define RUNTIME_SHADER_UNLOCK() do { if (pthread_mutex_unlock(&runtime_shader_mutex)) abort(); } while (0)
#else
#define RUNTIME_SHADER_LOCK() ((void)0)
#define RUNTIME_SHADER_UNLOCK() ((void)0)
#endif
static uint64_t runtime_pair_lookups, runtime_pair_hot_hits;
static uint64_t runtime_pair_scan_hits, runtime_pair_full;

/* Prepared draws write every emitted command and own only these mutable state
 * pages. Shader code lives in the pair allocation; unused command capacity is
 * neither executed nor an upload. Downward scratch is published separately. */
static void runtime_prepared_work_clear(uint8_t *memory)
{
    memset(memory + 0x4000, 0, 0x4000);
}

static int runtime_prepared_work_publish(uint8_t *memory,
                                          const agc_command_buffer_t *command)
{
    if (command->bottom != (uint32_t *)(memory + 0x8000) ||
        command->top != (uint32_t *)(memory + 0xc000) ||
        command->up < command->bottom || command->down < command->up ||
        command->down > command->top)
        return -1;
    flush_gpu_data(memory + 0x4000, 0x4000);
    flush_gpu_data(command->bottom, (size_t)(command->up - command->bottom) * 4);
    flush_gpu_data(command->down, (size_t)(command->top - command->down) * 4);
    return 0;
}

static struct runtime_shader_pair *runtime_shader_pair_get_unlocked(
    const agc_api_t *agc, int64_t direct_limit,
    const uint8_t *vs_header, size_t vs_header_size,
    const uint8_t *vs_code, size_t vs_code_size,
    const uint8_t *ps_header, size_t ps_header_size,
    const uint8_t *ps_code, size_t ps_code_size)
{
    const size_t limit = 256u * 1024u * 1024u;
    uintptr_t key = ((uintptr_t)runtime_vs_package >> 4) ^
                    (((uintptr_t)runtime_ps_package >> 4) * UINT64_C(0x9e3779b185ebca87)) ^
                    ((uintptr_t)runtime_primitive_type * UINT64_C(0xc2b2ae3d27d4eb4f));
    unsigned slot = (unsigned)((key ^ (key >> 17) ^ (key >> 31)) & 4095u);
    struct runtime_shader_pair *hot = runtime_shader_pair_hot[slot];
    if (++runtime_pair_lookups % 100000 == 0)
        printf("[ps5-pair-cache] lookups=%" PRIu64 " hot=%" PRIu64
               " scanned=%" PRIu64 " full=%" PRIu64 " entries=%u bytes=%zu\n",
               runtime_pair_lookups, runtime_pair_hot_hits,
               runtime_pair_scan_hits, runtime_pair_full,
               runtime_shader_pair_count, runtime_shader_pair_bytes);
    if (hot && hot->vs_key == runtime_vs_package &&
        hot->ps_key == runtime_ps_package &&
        hot->vs_size == runtime_vs_package_len &&
        hot->ps_size == runtime_ps_package_len &&
        hot->primitive == runtime_primitive_type &&
        !memcmp(hot->source, runtime_vs_package, hot->vs_size) &&
        !memcmp(hot->source + hot->vs_size, runtime_ps_package, hot->ps_size)) {
        ++runtime_pair_hot_hits;
        return hot;
    }
    for (unsigned i = 0; i < runtime_shader_pair_count; ++i) {
        struct runtime_shader_pair *p = &runtime_shader_pairs[i];
        if (p->vs_key == runtime_vs_package && p->ps_key == runtime_ps_package &&
            p->vs_size == runtime_vs_package_len && p->ps_size == runtime_ps_package_len &&
            p->primitive == runtime_primitive_type &&
            !memcmp(p->source, runtime_vs_package, p->vs_size) &&
            !memcmp(p->source + p->vs_size, runtime_ps_package, p->ps_size)) {
            runtime_shader_pair_hot[slot] = p;
            ++runtime_pair_scan_hits;
            return p; /* Byte comparison also covers freed/reused package addresses. */
        }
    }
    if (runtime_shader_pair_count == sizeof(runtime_shader_pairs) / sizeof(runtime_shader_pairs[0])) {
        ++runtime_pair_full;
        return NULL;
    }
    const size_t vs_header_at = 0x2000;
    const size_t vs_code_at = (vs_header_at + vs_header_size + 0xfffu) & ~(size_t)0xfffu;
    const size_t ps_header_at = (vs_code_at + vs_code_size + 0xfffu) & ~(size_t)0xfffu;
    const size_t ps_code_at = (ps_header_at + ps_header_size + 0xfffu) & ~(size_t)0xfffu;
    const size_t bytes = (ps_code_at + ps_code_size + 0x3fffu) & ~(size_t)0x3fffu;
    const size_t source_bytes = (size_t)runtime_vs_package_len + runtime_ps_package_len;
    if (bytes > limit || source_bytes > limit - bytes ||
        runtime_shader_pair_bytes > limit - bytes - source_bytes) {
        ++runtime_pair_full;
        return NULL;
    }
    struct runtime_shader_pair p = {.vs_key = runtime_vs_package, .ps_key = runtime_ps_package,
        .vs_size = runtime_vs_package_len, .ps_size = runtime_ps_package_len,
        .bytes = bytes, .primitive = runtime_primitive_type, .direct = -1};
    p.source = malloc(source_bytes);
    if (!p.source)
        return NULL;
    if (sceKernelAllocateDirectMemory(0, direct_limit, bytes, 0x4000,
                                     DIRECT_MEMORY_TYPE, &p.direct) != 0)
        goto fail;
    if (sceKernelMapDirectMemory((void **)&p.memory, bytes, MAP_PROTECTION, 0,
                                p.direct, 0x4000) != 0) {
        p.memory = NULL;
        goto fail;
    }
    memset(p.memory, 0, bytes);
    memcpy(p.source, runtime_vs_package, p.vs_size);
    memcpy(p.source + p.vs_size, runtime_ps_package, p.ps_size);
    memcpy(p.memory + vs_header_at, vs_header, vs_header_size);
    memcpy(p.memory + vs_code_at, vs_code, vs_code_size);
    memcpy(p.memory + ps_header_at, ps_header, ps_header_size);
    memcpy(p.memory + ps_code_at, ps_code, ps_code_size);
    if (agc->create_shader(&p.vertex, p.memory + vs_header_at, p.memory + vs_code_at) ||
        agc->create_shader(&p.pixel, p.memory + ps_header_at, p.memory + ps_code_at) ||
        agc->link_shaders(p.memory, p.memory + 0x1000, NULL, p.vertex, p.pixel, p.primitive))
        goto fail;
    flush_gpu_data(p.memory, bytes);
    runtime_shader_pair_bytes += bytes + source_bytes;
    runtime_shader_pairs[runtime_shader_pair_count] = p;
    runtime_shader_pair_hot[slot] = &runtime_shader_pairs[runtime_shader_pair_count];
    return &runtime_shader_pairs[runtime_shader_pair_count++];
fail:
    if (p.memory && munmap(p.memory, bytes) != 0)
        abort(); /* Never release a still-mapped allocation. */
    if (p.direct >= 0 && sceKernelReleaseDirectMemory(p.direct, bytes) != 0)
        abort();
    free(p.source);
    return NULL;
}

static struct runtime_shader_pair *runtime_shader_pair_get(
    const agc_api_t *agc, int64_t direct_limit,
    const uint8_t *vs_header, size_t vs_header_size,
    const uint8_t *vs_code, size_t vs_code_size,
    const uint8_t *ps_header, size_t ps_header_size,
    const uint8_t *ps_code, size_t ps_code_size)
{
    RUNTIME_SHADER_LOCK();
    struct runtime_shader_pair *pair = runtime_shader_pair_get_unlocked(
        agc, direct_limit, vs_header, vs_header_size, vs_code, vs_code_size,
        ps_header, ps_header_size, ps_code, ps_code_size);
    RUNTIME_SHADER_UNLOCK();
    return pair; /* Entries remain immutable until workers stop and GPU retires. */
}

static int runtime_shader_pair_clear(void)
{
    while (runtime_shader_pair_count) {
        struct runtime_shader_pair *p = &runtime_shader_pairs[runtime_shader_pair_count - 1];
        if (munmap(p->memory, p->bytes) != 0 ||
            sceKernelReleaseDirectMemory(p->direct, p->bytes) != 0)
            return -1;
        free(p->source);
        --runtime_shader_pair_count;
    }
    runtime_shader_pair_bytes = 0;
    memset(runtime_shader_pair_hot, 0, sizeof(runtime_shader_pair_hot));
    return 0;
}

static int runtime_work_take(size_t bytes, void **memory, int64_t *direct)
{
    RUNTIME_WORK_LOCK();
    for (unsigned i = 0; i < runtime_work_free_count; ++i) {
        if (runtime_work_free[i].bytes != bytes)
            continue;
        *memory = runtime_work_free[i].memory;
        *direct = runtime_work_free[i].direct;
        runtime_work_free_bytes -= bytes;
        runtime_work_free[i] = runtime_work_free[--runtime_work_free_count];
        RUNTIME_WORK_UNLOCK();
        return 1;
    }
    RUNTIME_WORK_UNLOCK();
    return 0;
}

static int runtime_work_put(void *memory, int64_t direct, size_t bytes)
{
    const size_t limit = 64u * 1024u * 1024u;
    RUNTIME_WORK_LOCK();
    if (runtime_work_free_count < sizeof(runtime_work_free) / sizeof(runtime_work_free[0]) &&
        bytes <= limit && runtime_work_free_bytes <= limit - bytes) {
        runtime_work_free[runtime_work_free_count++] =
            (struct runtime_work_allocation){memory, direct, bytes};
        runtime_work_free_bytes += bytes;
        RUNTIME_WORK_UNLOCK();
        return 0;
    }
    RUNTIME_WORK_UNLOCK();
    return munmap(memory, bytes) != 0 ||
           sceKernelReleaseDirectMemory(direct, bytes) != 0 ? -1 : 0;
}

static uint32_t *runtime_acquire_cpu_uploads(agc_command_buffer_t *command,
                                             void *memory, size_t bytes)
{
    extern uint32_t *sceAgcDcbAcquireMem(void *, uint8_t, uint32_t, uint32_t,
                                        uintptr_t, uint64_t, uint32_t);
    /* Mesa gfx10_emit_barrier: instruction/scalar/vector/L1 invalidation,
     * and L2/metadata writeback+invalidation. CPU uploads include descriptors
     * outside the command allocation; writeback preserves GPU-produced data. */
    return sceAgcDcbAcquireMem(command, 0, 0, UINT32_C(0xc3b1),
                               (uintptr_t)memory, bytes, 0xa0);
}


static int runtime_work_cache_clear(void)
{
    while (runtime_work_free_count) {
        struct runtime_work_allocation *entry = &runtime_work_free[runtime_work_free_count - 1];
        if (munmap(entry->memory, entry->bytes) != 0 ||
            sceKernelReleaseDirectMemory(entry->direct, entry->bytes) != 0)
            return -1;
        runtime_work_free_bytes -= entry->bytes;
        --runtime_work_free_count;
    }
    return 0;
}
#endif

#ifdef PS5_NATIVE_TITLE_RUNTIME
/* Queued compute keeps one immutable header/code allocation per package,
 * found by package bytes like shader pairs. Entries stay until shutdown;
 * a full cache leaves the caller on its synchronous path. */
static struct runtime_compute_program {
    uint8_t *package, *memory;
    size_t package_size, bytes;
    int64_t direct;
    agc_register_t *registers;
} runtime_compute_programs[256];
static unsigned runtime_compute_program_count;

static struct runtime_compute_program *runtime_compute_program_get(
    const agc_api_t *agc, int64_t direct_limit,
    const uint8_t *package, size_t package_size,
    const uint8_t *header, size_t header_size,
    const uint8_t *code, size_t code_size)
{
    for (unsigned i = 0; i < runtime_compute_program_count; ++i) {
        struct runtime_compute_program *p = &runtime_compute_programs[i];
        if (p->package_size == package_size && !memcmp(p->package, package, package_size))
            return p;
    }
    if (runtime_compute_program_count == sizeof(runtime_compute_programs) /
                                         sizeof(runtime_compute_programs[0]) ||
        header_size > 0x1000 || code_size > 16u * 1024u * 1024u)
        return NULL;
    struct runtime_compute_program p = {.package_size = package_size, .direct = -1,
        .bytes = (0x1000 + code_size + 0x3fffu) & ~(size_t)0x3fffu};
    void *program = NULL;
    p.package = malloc(package_size);
    if (!p.package)
        return NULL;
    if (sceKernelAllocateDirectMemory(0, direct_limit, p.bytes, 0x4000,
                                     DIRECT_MEMORY_TYPE, &p.direct) != 0)
        goto fail;
    if (sceKernelMapDirectMemory((void **)&p.memory, p.bytes, MAP_PROTECTION, 0,
                                p.direct, 0x4000) != 0) {
        p.memory = NULL;
        goto fail;
    }
    memset(p.memory, 0, p.bytes);
    memcpy(p.package, package, package_size);
    memcpy(p.memory, header, header_size);
    memcpy(p.memory + 0x1000, code, code_size);
    /* Same header-relative layout the synchronous path verifies. */
    if (agc->create_shader(&program, p.memory, p.memory + 0x1000) ||
        program != p.memory)
        goto fail;
    p.registers = *(agc_register_t **)(p.memory + 0x20);
    if (p.registers != (agc_register_t *)(p.memory + 0x60) || p.memory[0x5c] != 8)
        goto fail;
    flush_gpu_data(p.memory, p.bytes);
    runtime_compute_programs[runtime_compute_program_count] = p;
    return &runtime_compute_programs[runtime_compute_program_count++];
fail:
    if (p.memory && munmap(p.memory, p.bytes) != 0)
        abort(); /* Never release a still-mapped allocation. */
    if (p.direct >= 0 && sceKernelReleaseDirectMemory(p.direct, p.bytes) != 0)
        abort();
    free(p.package);
    return NULL;
}

static int runtime_compute_program_clear(void)
{
    while (runtime_compute_program_count) {
        struct runtime_compute_program *p =
            &runtime_compute_programs[runtime_compute_program_count - 1];
        if (munmap(p->memory, p->bytes) != 0 ||
            sceKernelReleaseDirectMemory(p->direct, p->bytes) != 0)
            return -1;
        free(p->package);
        --runtime_compute_program_count;
    }
    return 0;
}
#endif

static struct runtime_batch_entry {
    agc_submit_description_t submit;
    void *memory;
    int64_t direct;
    size_t bytes;
    volatile uint64_t *marker;
    uint32_t expected;
    uint32_t completion_offset, command_capacity, upload_prefix_words;
} runtime_batch_entries[PS5_MULTIDRAW_BATCH_CAPACITY];
static struct runtime_pending_batch {
    struct runtime_batch_entry entries[PS5_MULTIDRAW_BATCH_CAPACITY];
    unsigned count, attempted;
#ifdef PS5_DRAW_PROFILE
    int profile;
    int64_t start_ns, submit_ns, suspend_ns;
#endif
} runtime_pending[PS5_INFLIGHT_BATCH_CAPACITY];
static unsigned runtime_pending_head, runtime_pending_batches, runtime_pending_peak;
static agc_api_t runtime_batch_api;
static unsigned runtime_batch_count;
static int runtime_batch_active, runtime_batch_faulted;
static int runtime_batch_fixed_framebuffer;
static int runtime_batch_append(const agc_api_t *, const struct runtime_batch_entry *);

#ifdef PS5_ASYNC_NATIVE_PREP
/* The producer keeps translating GL draws while this worker prepares native
 * commands. Submission/retirement stays ordered at the existing batch edge. */
static int64_t runtime_next_render_marker(void);
static int runtime_async_prepare_draw(void);
static int runtime_async_drain(void);
#ifndef PS5_NATIVE_PREP_WORKERS
#define PS5_NATIVE_PREP_WORKERS 2
#endif
#if PS5_NATIVE_PREP_WORKERS < 1 || PS5_NATIVE_PREP_WORKERS > 2
#error "Native preparation supports one or two workers"
#endif
#if PS5_NATIVE_PREP_WORKERS > 1 && (defined(PS5_DRAW_PROFILE) || defined(PS5_DRAW_GPU_TIMESTAMPS) || defined(AGC_RUNTIME_DIAGNOSTICS))
#error "Multiworker preparation requires profile0; shared diagnostic counters are not concurrent"
#endif
#define RUNTIME_ASYNC_JOBS PS5_MULTIDRAW_BATCH_CAPACITY
struct runtime_async_job {
    struct runtime_draw_inputs draw;
    struct ps5_agc_backend_draw_state backend;
    uint32_t ngg_valid, ngg_value;
    int64_t marker;
    agc_api_t api;
    struct runtime_batch_entry entry;
    int has_entry;
    _Atomic uint64_t ready;
};
static struct runtime_async_job runtime_async_jobs[RUNTIME_ASYNC_JOBS];
static _Thread_local struct runtime_async_job *runtime_async_output;
static pthread_t runtime_async_threads[PS5_NATIVE_PREP_WORKERS];
static _Atomic uint64_t runtime_async_produced, runtime_async_consumed;
static uint64_t runtime_async_collected;
static _Atomic int runtime_async_stop_requested, runtime_async_error;
static _Atomic int runtime_async_ready[PS5_NATIVE_PREP_WORKERS];
static int runtime_async_started;

int ps5_agc_gate2_run_sync(void);

static int runtime_async_pin(unsigned worker)
{
#ifdef PS5_ASYNC_HOST_TEST
    (void)worker;
    return 0;
#else
    /* Keep preparation on distinct physical cores, separate from producer8. */
    unsigned cpu = 10u + worker * 2u;
    cpuset_t one = {0}, verified = {0};
    CPU_SET(cpu, &one);
    if (cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &one) ||
        cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, 8, &verified) ||
        memcmp(&one, &verified, 8))
        return -1;
    printf("[ps5-async-prep] cpu=%u verified=1 worker=%u\n", cpu, worker);
    return 0;
#endif
}

static void *runtime_async_main(void *argument)
{
    unsigned worker = (unsigned)(uintptr_t)argument;
#ifdef PS5_DRAW_PROFILE
    int64_t first_job_ns = 0;
#endif
    int pin = runtime_async_pin(worker);
    atomic_store_explicit(&runtime_async_ready[worker], pin ? -1 : 1, memory_order_release);
    if (pin)
        return NULL;
    for (;;) {
        uint64_t consumed = atomic_load_explicit(&runtime_async_consumed, memory_order_relaxed);
        if (consumed >= atomic_load_explicit(&runtime_async_produced, memory_order_acquire)) {
            if (atomic_load_explicit(&runtime_async_stop_requested, memory_order_acquire))
                return NULL;
            /* ponytail: this development worker busy-spins on a spare core;
             * add an idle event only if the lock-free path proves useful. */
            __builtin_ia32_pause();
            continue;
        }
        if (!atomic_compare_exchange_weak_explicit(&runtime_async_consumed,
                &consumed, consumed + 1, memory_order_relaxed, memory_order_relaxed))
            continue;
        struct runtime_async_job *job = &runtime_async_jobs[consumed % RUNTIME_ASYNC_JOBS];

        runtime_draw_state = job->draw;
        ps5_agc_draw_state = job->backend;
        runtime_ngg_ge_pc_alloc_valid = job->ngg_valid;
        runtime_ngg_ge_pc_alloc = job->ngg_value;
#ifdef PS5_DRAW_PROFILE
        if (!first_job_ns)
            first_job_ns = os_time_get_nano();
#endif
        runtime_async_output = job;
        int result = ps5_agc_gate2_run_sync();
        runtime_async_output = NULL;

        if (result) {
            int expected = 0;
            atomic_compare_exchange_strong_explicit(&runtime_async_error, &expected,
                result, memory_order_relaxed, memory_order_relaxed);
        }
        atomic_store_explicit(&job->ready, consumed + 1, memory_order_release);
#ifdef PS5_DRAW_PROFILE
        if ((consumed + 1) % 100000 == 0)
            printf("[ps5-async-elapsed] completed=%" PRIu64 " elapsed_ns=%" PRId64 "\n",
                   consumed + 1, os_time_get_nano() - first_job_ns);
#endif
    }
}

/* Only the producer appends results. A later completed draw cannot publish
 * past an earlier one, and a ring slot remains owned until collection. */
static void runtime_async_collect_one(void)
{
    struct runtime_async_job *job =
        &runtime_async_jobs[runtime_async_collected % RUNTIME_ASYNC_JOBS];
    while (atomic_load_explicit(&job->ready, memory_order_acquire) != runtime_async_collected + 1)
        __builtin_ia32_pause();
    if (job->has_entry && runtime_batch_append(&job->api, &job->entry) != 0)
        abort(); /* Accepted allocation must never lose its retirement owner. */
    job->has_entry = 0;
    atomic_store_explicit(&job->ready, 0, memory_order_relaxed);
    ++runtime_async_collected;
}

int ps5_agc_gate2_run(void)
{
    if (!runtime_batch_active)
        return ps5_agc_gate2_run_sync();
    /* Tessellation changes the process-wide factor ring. Finish workers first. */
    if (runtime_draw_state.hs_package) {
        if (runtime_async_drain() != 0)
            return -1;
        return ps5_agc_gate2_run_sync();
    }
    if (runtime_async_prepare_draw() != 0)
        return -1;
    if (!runtime_async_started) {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0)
            return -1;
        unsigned created = 0;
        int failed = pthread_attr_setstacksize(&attr, 8u * 1024u * 1024u);
        while (!failed && created < PS5_NATIVE_PREP_WORKERS) {
            failed = pthread_create(&runtime_async_threads[created], &attr,
                                     runtime_async_main, (void *)(uintptr_t)created);
            if (!failed)
                ++created;
        }
        pthread_attr_destroy(&attr);
        for (unsigned i = 0; i < created; ++i) {
            int ready;
            unsigned spins = 0;
            while (!(ready = atomic_load_explicit(&runtime_async_ready[i], memory_order_acquire))) {
                if ((++spins & 1023u) == 0)
                    sched_yield();
                else
                    __builtin_ia32_pause();
            }
            failed |= ready < 0;
        }
        if (failed) {
            atomic_store_explicit(&runtime_async_stop_requested, 1, memory_order_release);
            for (unsigned i = 0; i < created; ++i) {
                if (pthread_join(runtime_async_threads[i], NULL) != 0)
                    abort();
                atomic_store(&runtime_async_ready[i], 0);
            }
            atomic_store(&runtime_async_stop_requested, 0);
            return -1;
        }
        runtime_async_started = 1;
    }
    uint64_t produced = atomic_load_explicit(&runtime_async_produced, memory_order_relaxed);
    if (produced - runtime_async_collected == RUNTIME_ASYNC_JOBS)
        runtime_async_collect_one();
    if (atomic_load_explicit(&runtime_async_error, memory_order_acquire))
        return -1;
    struct runtime_async_job *job =
        &runtime_async_jobs[produced % RUNTIME_ASYNC_JOBS];
    job->draw = runtime_draw_state;
    job->backend = ps5_agc_draw_state;
    job->ngg_valid = runtime_ngg_ge_pc_alloc_valid;
    job->ngg_value = runtime_ngg_ge_pc_alloc;
    job->marker = runtime_next_render_marker();
    job->has_entry = 0;
    atomic_store_explicit(&runtime_async_produced, produced + 1, memory_order_release);
    return 0;
}

static int runtime_async_drain(void)
{
    if (!runtime_async_started)
        return 0;
    uint64_t target = atomic_load_explicit(&runtime_async_produced, memory_order_relaxed);
    while (runtime_async_collected != target)
        runtime_async_collect_one();
    return atomic_load_explicit(&runtime_async_error, memory_order_acquire);
}

static void runtime_async_stop(void)
{
    if (!runtime_async_started)
        return;
    runtime_async_drain();
    atomic_store_explicit(&runtime_async_stop_requested, 1, memory_order_release);
    for (unsigned i = 0; i < PS5_NATIVE_PREP_WORKERS; ++i) {
        if (pthread_join(runtime_async_threads[i], NULL) != 0)
            abort();
        atomic_store(&runtime_async_ready[i], 0);
    }
    runtime_async_started = 0;
    atomic_store(&runtime_async_stop_requested, 0);
    atomic_store(&runtime_async_error, 0);
    atomic_store(&runtime_async_produced, 0);
    atomic_store(&runtime_async_consumed, 0);
    runtime_async_collected = 0;
}
#endif

int ps5_agc_gate2_batch_begin(void)
{
    if (runtime_batch_active || runtime_batch_count || runtime_batch_faulted)
        return -1;
    runtime_batch_active = 1;
    runtime_batch_fixed_framebuffer = 0;
    return 0;
}

/* Opt-in contract: no attachment changes or attachment sampling within a batch.
 * Gallium splits at framebuffer changes and rejects attachment feedback. */
int ps5_agc_gate2_batch_begin_framebuffer(void)
{
    int result = ps5_agc_gate2_batch_begin();
    if (!result)
        runtime_batch_fixed_framebuffer = 1;
    return result;
}

static int runtime_batch_append(const agc_api_t *api,
                               const struct runtime_batch_entry *entry)
{
    if (!runtime_batch_active || runtime_batch_faulted ||
        runtime_batch_count == PS5_MULTIDRAW_BATCH_CAPACITY ||
        !entry || !entry->memory || entry->direct < 0 || !entry->bytes ||
        !entry->marker || !api || !api->submit || !api->suspend_point ||
        !entry->submit.words || !entry->submit.word_count)
        return -1;
    if (runtime_batch_count &&
        (runtime_batch_api.submit != api->submit ||
         runtime_batch_api.suspend_point != api->suspend_point))
        return -1;
    runtime_batch_api = *api;
    runtime_batch_entries[runtime_batch_count++] = *entry;
    return 0; /* Ownership transfers only on success. No GPU work yet. */
}

static int runtime_batch_queue(const agc_api_t *api,
                               const struct runtime_batch_entry *entry)
{
#ifdef PS5_ASYNC_NATIVE_PREP
    if (runtime_async_output) {
        if (!api || !entry || runtime_async_output->has_entry)
            return -1;
        runtime_async_output->api = *api;
        runtime_async_output->entry = *entry;
        runtime_async_output->has_entry = 1;
        return 0;
    }
#endif
    return runtime_batch_append(api, entry);
}

/* Keep draw bodies and their dependency barriers. Fixed-framebuffer batches
 * may also move their post-draw color/depth releases to the group boundary;
 * all other batches retain those releases per draw. No allocation is freed. */
static unsigned runtime_batch_combine(void)
{
    unsigned groups = 0;
    struct runtime_batch_entry *leader = NULL;
    for (unsigned i = 0; i < runtime_batch_count; ++i) {
        struct runtime_batch_entry *entry = &runtime_batch_entries[i];
        if (!entry->completion_offset ||
            entry->completion_offset >= entry->submit.word_count ||
            entry->submit.word_count > entry->command_capacity) {
            leader = NULL;
            ++groups;
            continue;
        }
        /* Every upload is published before this group is submitted. Its first
         * acquire covers all retained allocations; preserve subsequent draw
         * bodies and their explicit GPU-to-GPU dependency barriers. */
        uint32_t skip = leader && leader->upload_prefix_words &&
            entry->upload_prefix_words < entry->completion_offset
                ? entry->upload_prefix_words : 0;
        uint32_t append_words = entry->submit.word_count - skip;
        if (!leader || entry->submit.flag != leader->submit.flag ||
            append_words > leader->command_capacity - leader->completion_offset) {
            leader = entry;
            ++groups;
            continue;
        }
        uint32_t offset = leader->completion_offset;
        memcpy((uint32_t *)leader->submit.words + offset,
               (const uint32_t *)entry->submit.words + skip,
               (size_t)append_words * sizeof(uint32_t));
        leader->submit.word_count = offset + append_words;
        leader->completion_offset = offset + entry->completion_offset - skip;
        leader->marker = entry->marker;
        leader->expected = entry->expected;
        entry->submit.word_count = 0; /* Retain allocation; leader retires it. */
    }
    if (groups < runtime_batch_count)
        for (unsigned i = 0; i < runtime_batch_count; ++i)
            if (runtime_batch_entries[i].submit.word_count)
                flush_gpu_data(runtime_batch_entries[i].submit.words,
                    (size_t)runtime_batch_entries[i].submit.word_count * sizeof(uint32_t));
    return groups;
}

int ps5_agc_gate2_batch_submit(void)
{
    unsigned attempted = 0;
    int result = 0;
#ifdef PS5_DRAW_PROFILE
    int64_t ticks[3] = {0};
#define BATCH_SUBMIT_MARK(i) ticks[i] = os_time_get_nano()
#else
#define BATCH_SUBMIT_MARK(i) ((void)0)
#endif
#ifdef PS5_ASYNC_NATIVE_PREP
    if (runtime_async_drain() != 0) {
        runtime_batch_faulted = 1;
        return -1;
    }
#endif
    if (!runtime_batch_active || runtime_batch_faulted ||
        runtime_pending_batches == PS5_INFLIGHT_BATCH_CAPACITY)
        return -1;
    runtime_batch_active = 0;
    if (!runtime_batch_count)
        return 0;
#ifdef PS5_DRAW_PROFILE
    const int profile = runtime_present_count >= 30;
#endif
    BATCH_SUBMIT_MARK(0);
    unsigned groups = runtime_batch_combine();
    (void)groups;
#ifdef PS5_DRAW_GPU_TIMESTAMPS
    for (unsigned i = 0; i < runtime_batch_count; ++i) {
        struct runtime_batch_entry *e = &runtime_batch_entries[i];
        /* CPU fields occupy a separate cache line from GPU-written timestamps. */
        ((volatile uint64_t *)((uint8_t *)e->memory + e->bytes - 0x4000))[8] = os_time_get_nano();
    }
#endif
    for (unsigned i = 0; i < runtime_batch_count; ++i) {
        ++attempted; /* A failed submit is conservatively treated as in flight. */
        if (!runtime_batch_entries[i].submit.word_count)
            continue;
        if (runtime_batch_api.submit(&runtime_batch_entries[i].submit) != 0) {
            result = 1;
            break;
        }
    }
    BATCH_SUBMIT_MARK(1);
#ifdef PS5_FRAME_SUSPEND
    if (attempted && runtime_defer_suspend(runtime_batch_api.suspend_point) != 0)
#else
    if (attempted && runtime_batch_api.suspend_point() != 0)
#endif
        result = 1;
    BATCH_SUBMIT_MARK(2);
    runtime_require_retirement(result == 0);
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
    printf("[ps5-command-groups] draws=%u submissions=%u\n", runtime_batch_count, groups);
#endif
    struct runtime_pending_batch *pending = &runtime_pending[
        (runtime_pending_head + runtime_pending_batches) % PS5_INFLIGHT_BATCH_CAPACITY];
    memcpy(pending->entries, runtime_batch_entries,
           runtime_batch_count * sizeof(runtime_batch_entries[0]));
    memset(runtime_batch_entries, 0,
           runtime_batch_count * sizeof(runtime_batch_entries[0]));
    pending->count = runtime_batch_count;
    pending->attempted = attempted;
    ++runtime_pending_batches;
    if (runtime_pending_batches > runtime_pending_peak) {
        runtime_pending_peak = runtime_pending_batches;
        printf("[ps5-inflight] peak=%u capacity=%u\n", runtime_pending_peak, PS5_INFLIGHT_BATCH_CAPACITY);
    }
    runtime_batch_count = 0;
#ifdef PS5_DRAW_PROFILE
    pending->profile = profile;
    pending->start_ns = ticks[0];
    pending->submit_ns = ticks[1] - ticks[0];
    pending->suspend_ns = ticks[2] - ticks[1];
#endif
#undef BATCH_SUBMIT_MARK
    return 0;
}

int ps5_agc_gate2_batch_retire(int wait)
{
    struct runtime_pending_batch *pending = &runtime_pending[runtime_pending_head];
    unsigned waits = 0;
    int64_t spin_deadline = 0;
    int complete = 0;
#ifdef PS5_DRAW_PROFILE
    int64_t poll_start = 0, poll_end = 0, cleanup_end = 0;
    if (pending->profile)
        poll_start = os_time_get_nano();
#endif
    if (runtime_batch_faulted)
        return -1;
    if (!pending->count)
        return 1;
    do {
        complete = 1;
        for (unsigned i = 0; i < pending->attempted; ++i) {
            struct runtime_batch_entry *entry = &pending->entries[i];
#ifdef PS5_DRAW_GPU_TIMESTAMPS
            runtime_draw_gpu_observe(entry->memory, entry->bytes);
#endif
            if (!entry->submit.word_count)
                continue;
            flush_gpu_data((const void *)entry->marker, sizeof(*entry->marker));
            complete &= *entry->marker == entry->expected;
        }
        if (complete || !wait)
            break;
        waits += runtime_completion_pause(&spin_deadline);
    } while (waits < 2000);
#ifdef PS5_DRAW_PROFILE
    if (pending->profile)
        poll_end = os_time_get_nano();
#endif
    if (!complete) {
        if (wait)
            runtime_require_retirement(0);
        return 0;
    }
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
    printf("[ps5-multidraw-batch] draws=%u attempted=%u waits=%u result=0\n",
           pending->count, pending->attempted, waits);
#endif
    for (unsigned i = 0; i < pending->count; ++i) {
        struct runtime_batch_entry *entry = &pending->entries[i];
#ifdef PS5_DRAW_GPU_TIMESTAMPS
        runtime_draw_gpu_timing(entry->memory, entry->bytes);
#endif
#ifdef PS5_NATIVE_TITLE_RUNTIME
        if (runtime_work_put(entry->memory, entry->direct, entry->bytes) != 0) {
#else
        if (munmap(entry->memory, entry->bytes) != 0 ||
            sceKernelReleaseDirectMemory(entry->direct, entry->bytes) != 0) {
#endif
            runtime_batch_faulted = 1;
            return -1;
        }
        memset(entry, 0, sizeof(*entry));
    }
#ifdef PS5_DRAW_PROFILE
    if (pending->profile) {
        int64_t ticks[5] = {0};
        cleanup_end = os_time_get_nano();
        ticks[0] = pending->start_ns;
        ticks[1] = ticks[0] + pending->submit_ns;
        ticks[2] = ticks[1] + pending->suspend_ns;
        ticks[3] = ticks[2] + poll_end - poll_start;
        ticks[4] = ticks[3] + cleanup_end - poll_end;
        runtime_batch_profile_record(ticks, waits, 0);
    }
    pending->profile = 0;
#endif
    memset(pending, 0, sizeof(*pending));
    runtime_pending_head = (runtime_pending_head + 1) % PS5_INFLIGHT_BATCH_CAPACITY;
    --runtime_pending_batches;
    return 1;
}

int ps5_agc_gate2_batch_end(void)
{
    if (ps5_agc_gate2_batch_submit() != 0)
        return -1;
    while (runtime_pending_batches)
        if (ps5_agc_gate2_batch_retire(1) != 1)
            return -1;
    return 0;
}
#endif

#ifdef PS5_DRAW_PROFILE
#include "util/os_time.h"
static uint64_t runtime_profile_ns[9], runtime_profile_sleeps;
static unsigned runtime_profile_calls, runtime_profile_failures;
static uint64_t runtime_present_profile_ns[3];
static unsigned runtime_present_profile_calls, runtime_present_profile_failures;
static uint64_t runtime_batch_profile_ns[4], runtime_batch_profile_sleeps;
static unsigned runtime_batch_profile_calls, runtime_batch_profile_failures;
static uint64_t runtime_prepare_profile_ns[5];
static unsigned runtime_prepare_profile_calls, runtime_prepare_profile_failures;

/* Reuse existing timestamps: startup stays outside the steady-state averages,
 * but inside the application's unchanged startup-inclusive timing window. */
static void runtime_prepare_startup_report(const int64_t ticks[6],
                                           unsigned attempts, int result)
{
    int valid = 1;
    double ms[5] = {0};
    for (unsigned i = 0; i < 6; ++i)
        valid &= ticks[i] > 0 && (!i || ticks[i] >= ticks[i - 1]);
    if (valid)
        for (unsigned i = 0; i < 5; ++i)
            ms[i] = (ticks[i + 1] - ticks[i]) * 1e-6;
    printf("[ps5-prepare-startup] attempts=%u result=%d clock_valid=%d "
           "setup_ms=%.6f scanout_flush_ms=%.6f video_ms=%.6f "
           "command_ms=%.6f command_flush_ms=%.6f total_ms=%.6f\n",
           attempts, result, valid, ms[0], ms[1], ms[2], ms[3], ms[4],
           ms[0] + ms[1] + ms[2] + ms[3] + ms[4]);
}

static void runtime_prepare_profile_record(const int64_t ticks[6], int result)
{
    for (unsigned i = 0; i < 6; ++i) {
        if (result || ticks[i] <= 0 || (i && ticks[i] < ticks[i - 1])) {
            ++runtime_prepare_profile_failures;
            return;
        }
    }
    for (unsigned i = 0; i < 5; ++i)
        runtime_prepare_profile_ns[i] += ticks[i + 1] - ticks[i];
    ++runtime_prepare_profile_calls;
}

static void runtime_batch_profile_record(const int64_t ticks[5], unsigned sleeps,
                                         int result)
{
    for (unsigned i = 0; i < 5; ++i) {
        if (result || ticks[i] <= 0 || (i && ticks[i] < ticks[i - 1])) {
            ++runtime_batch_profile_failures;
            return;
        }
    }
    for (unsigned i = 0; i < 4; ++i)
        runtime_batch_profile_ns[i] += ticks[i + 1] - ticks[i];
    runtime_batch_profile_sleeps += sleeps;
    ++runtime_batch_profile_calls;
}

static void runtime_present_profile_record(const int64_t ticks[4], int result)
{
    for (unsigned i = 0; i < 4; ++i) {
        if (result || ticks[i] <= 0 || (i && ticks[i] < ticks[i - 1])) {
            ++runtime_present_profile_failures;
            return;
        }
    }
    for (unsigned i = 0; i < 3; ++i)
        runtime_present_profile_ns[i] += ticks[i + 1] - ticks[i];
    ++runtime_present_profile_calls;
}

static void runtime_profile_record(const int64_t ticks[10], unsigned sleeps, int result)
{
    for (unsigned i = 0; i < 10; ++i) {
        if (result || ticks[i] <= 0 || (i && ticks[i] < ticks[i - 1])) {
            ++runtime_profile_failures;
            return;
        }
    }
    for (unsigned i = 0; i < 9; ++i)
        runtime_profile_ns[i] += ticks[i + 1] - ticks[i];
    runtime_profile_sleeps += sleeps;
    ++runtime_profile_calls;
}

static void runtime_profile_report(void)
{
    if (runtime_profile_calls || runtime_profile_failures) {
        const double scale = runtime_profile_calls ? 1e-6 / runtime_profile_calls : 0;
        uint64_t total = 0;
        for (unsigned i = 0; i < 9; ++i)
            total += runtime_profile_ns[i];
        printf("[ps5-submit-perf] calls=%u failures=%u warmup_frames=30 sleeps=%" PRIu64 " "
               "setup_ms=%.6f scanout_flush_ms=%.6f video_ms=%.6f "
               "command_ms=%.6f command_flush_ms=%.6f submit_ms=%.6f "
               "suspend_ms=%.6f poll_ms=%.6f cleanup_ms=%.6f total_ms=%.6f\n",
               runtime_profile_calls, runtime_profile_failures, runtime_profile_sleeps,
               runtime_profile_ns[0] * scale, runtime_profile_ns[1] * scale,
               runtime_profile_ns[2] * scale, runtime_profile_ns[3] * scale,
               runtime_profile_ns[4] * scale, runtime_profile_ns[5] * scale,
               runtime_profile_ns[6] * scale, runtime_profile_ns[7] * scale,
               runtime_profile_ns[8] * scale, total * scale);
    }
    memset(runtime_profile_ns, 0, sizeof(runtime_profile_ns));
    runtime_profile_calls = runtime_profile_failures = 0;
    runtime_profile_sleeps = 0;
    if (runtime_present_profile_calls || runtime_present_profile_failures) {
        const double scale = runtime_present_profile_calls ?
            1e-6 / runtime_present_profile_calls : 0;
        printf("[ps5-present-perf] calls=%u failures=%u warmup_frames=30 "
               "idle_ms=%.6f flip_ms=%.6f vblank_ms=%.6f total_ms=%.6f\n",
               runtime_present_profile_calls, runtime_present_profile_failures,
               runtime_present_profile_ns[0] * scale,
               runtime_present_profile_ns[1] * scale,
               runtime_present_profile_ns[2] * scale,
               (runtime_present_profile_ns[0] + runtime_present_profile_ns[1] +
                runtime_present_profile_ns[2]) * scale);
    }
    memset(runtime_present_profile_ns, 0, sizeof(runtime_present_profile_ns));
    runtime_present_profile_calls = runtime_present_profile_failures = 0;
    if (runtime_batch_profile_calls || runtime_batch_profile_failures) {
        const double scale = runtime_batch_profile_calls ?
            1e-6 / runtime_batch_profile_calls : 0;
        uint64_t total = 0;
        for (unsigned i = 0; i < 4; ++i)
            total += runtime_batch_profile_ns[i];
        printf("[ps5-batch-perf] calls=%u failures=%u warmup_frames=30 sleeps=%" PRIu64 " "
               "submit_ms=%.6f suspend_ms=%.6f poll_ms=%.6f cleanup_ms=%.6f total_ms=%.6f\n",
               runtime_batch_profile_calls, runtime_batch_profile_failures,
               runtime_batch_profile_sleeps, runtime_batch_profile_ns[0] * scale,
               runtime_batch_profile_ns[1] * scale, runtime_batch_profile_ns[2] * scale,
               runtime_batch_profile_ns[3] * scale, total * scale);
    }
    memset(runtime_batch_profile_ns, 0, sizeof(runtime_batch_profile_ns));
    runtime_batch_profile_calls = runtime_batch_profile_failures = 0;
    runtime_batch_profile_sleeps = 0;
    if (runtime_prepare_profile_calls || runtime_prepare_profile_failures) {
        const double scale = runtime_prepare_profile_calls ?
            1e-6 / runtime_prepare_profile_calls : 0;
        uint64_t total = 0;
        for (unsigned i = 0; i < 5; ++i)
            total += runtime_prepare_profile_ns[i];
        printf("[ps5-prepare-perf] calls=%u failures=%u warmup_frames=30 "
               "setup_ms=%.6f scanout_flush_ms=%.6f video_ms=%.6f "
               "command_ms=%.6f command_flush_ms=%.6f total_ms=%.6f\n",
               runtime_prepare_profile_calls, runtime_prepare_profile_failures,
               runtime_prepare_profile_ns[0] * scale, runtime_prepare_profile_ns[1] * scale,
               runtime_prepare_profile_ns[2] * scale, runtime_prepare_profile_ns[3] * scale,
               runtime_prepare_profile_ns[4] * scale, total * scale);
    }
    memset(runtime_prepare_profile_ns, 0, sizeof(runtime_prepare_profile_ns));
    runtime_prepare_profile_calls = runtime_prepare_profile_failures = 0;
}
#define PS5_PROFILE_MARK(i) profile_ticks[i] = os_time_get_nano()
#endif

static int64_t runtime_next_render_marker(void)
{
    uint64_t marker = runtime_render_marker;

    runtime_render_marker += UINT64_C(0x100);
    if (runtime_render_marker < (uint64_t)RENDER_MARKER)
        runtime_render_marker = (uint64_t)RENDER_MARKER;
    return (int64_t)marker;
}

static int runtime_video_wait_idle(void);

#ifdef PS5_GPU_PRESENT_BATCH
int ps5_agc_gate2_wait_present(void);
int ps5_agc_gate2_batch_present(unsigned buffer_index)
{
#ifdef PS5_ASYNC_NATIVE_PREP
    /* The flip patches the final entry, so its producer must finish first. */
    if (runtime_async_drain() != 0)
        return -1;
#endif
    if (ps5_agc_gate2_wait_present() != 0 ||
        buffer_index > 1 || runtime_gpu_present_buffer >= 0 ||
        !runtime_video_registered || runtime_video_handle < 0 ||
        !runtime_batch_active || !runtime_batch_count || runtime_batch_faulted ||
        !runtime_batch_api.set_flip || !runtime_batch_api.release_mem ||
        !runtime_video_api.get_flip_status || runtime_video_wait_idle() != 0)
        return -1;
    struct runtime_batch_entry *entry = &runtime_batch_entries[runtime_batch_count - 1];
    uint32_t *words = entry->submit.words;
    if (!words || (uintptr_t)words < (uintptr_t)entry->memory ||
        (uintptr_t)words - (uintptr_t)entry->memory > entry->bytes ||
        entry->bytes - ((uintptr_t)words - (uintptr_t)entry->memory) < COMMAND_BYTES ||
        entry->submit.word_count > COMMAND_BYTES / sizeof(*words) - 128)
        return -1;
    agc_command_buffer_t command = {0};
    command.bottom = words;
    command.top = words + COMMAND_BYTES / sizeof(*words);
    command.up = words + entry->submit.word_count;
    command.down = command.top;
    command.callback = (uintptr_t)command_out_of_space;
    uint64_t marker = (uint64_t)runtime_next_render_marker();
    out_of_space = 0;
    /* Queue the flip after all frame draws; a new completion marker after it
     * keeps this command allocation alive until the GPU consumed the tail. */
    if (!runtime_batch_api.set_flip(&command, (uint32_t)runtime_video_handle,
                                    (int)buffer_index, 1, (int64_t)marker) ||
        !runtime_release_completion(&runtime_batch_api, &command, entry->marker,
                                     (uint32_t)marker) ||
        out_of_space || command.up <= words + entry->submit.word_count ||
        command.up > command.top)
        return -1;
    entry->submit.word_count = (uint32_t)(command.up - words);
    entry->expected = (uint32_t)marker;
    flush_gpu_data(words, entry->submit.word_count * sizeof(*words));
    runtime_gpu_present_buffer = (int)buffer_index;
    runtime_gpu_present_marker = marker;
    runtime_gpu_present_is_cpu = 0;
    return 0;
}

static int runtime_gpu_present_finish(unsigned buffer_index)
{
    if (runtime_gpu_present_buffer != (int)buffer_index)
        return -1;
    for (unsigned waits = 0; waits <= 2000; ++waits) {
        uint64_t status[16] = {0};
        int result = runtime_video_api.get_flip_status(runtime_video_handle, status);
        if (result != 0)
            return result;
        int pending = runtime_video_api.is_flip_pending(runtime_video_handle);
        if (pending < 0)
            return pending;
        if (!pending && status[3] == runtime_gpu_present_marker) {
            runtime_gpu_present_buffer = -1;
            runtime_gpu_present_deferred = 0;
            if (!runtime_gpu_present_is_cpu)
                ++runtime_gpu_present_count;
            runtime_gpu_present_is_cpu = 0;
            return 0;
        }
        if (waits == 2000)
            return -1;
        /* Vblank can wake before VideoOut publishes flip status. Sleeping until
         * another vblank then adds a whole refresh despite completed work. */
        result = sceKernelUsleep(UINT32_C(1000));
        if (result != 0)
            return result;
    }
    return -1;
}

/* Both GPU-tail and CPU-submitted swaps have retired their rendering before
 * deferring display completion. They share the same scanout ownership gate. */
int ps5_agc_gate2_wait_present(void)
{
    return runtime_gpu_present_deferred
        ? runtime_gpu_present_finish((unsigned)runtime_gpu_present_buffer) : 0;
}

#endif

#if defined(PS5_DRAW_PROFILE) && defined(PS5_NATIVE_TITLE_RUNTIME)
/* Read-only ABI subset also exercised by the independently authored
 * ProsperoLight presenter. Raw status is evidence, not a requested mode. */
typedef struct runtime_resolution_status {
    uint32_t full_width, full_height, pane_width, pane_height;
    uint64_t refresh_rate;
    float screen_inches;
    uint32_t reserved[4];
} runtime_resolution_status_t;
typedef struct runtime_output_status {
    uint32_t resolution_class, output_class;
    uint64_t refresh_rate, flags;
    uint32_t mode, reserved[5];
} runtime_output_status_t;
_Static_assert(sizeof(runtime_resolution_status_t) == 48 &&
               offsetof(runtime_resolution_status_t, refresh_rate) == 16 &&
               sizeof(runtime_output_status_t) == 48 &&
               offsetof(runtime_output_status_t, refresh_rate) == 8,
               "VideoOut status ABI size/offset mismatch");
extern int sceVideoOutGetResolutionStatus(int32_t, runtime_resolution_status_t *);
extern int sceVideoOutGetOutputStatus(int32_t, runtime_output_status_t *);

static void runtime_video_report(const char *stage)
{
    runtime_resolution_status_t resolution = {0};
    runtime_output_status_t output = {0};
    int resolution_rc = sceVideoOutGetResolutionStatus(runtime_video_handle, &resolution);
    int output_rc = sceVideoOutGetOutputStatus(runtime_video_handle, &output);
    printf("[ps5-output] stage=%s render_width=%u render_height=%u buffer_bytes=%u "
           "resolution_rc=%08" PRIx32 " full_width=%u full_height=%u pane_width=%u pane_height=%u "
           "refresh_id=%" PRIu64 " output_rc=%08" PRIx32 " output_refresh_id=%" PRIu64 "\n",
           stage, DISPLAY_WIDTH, (unsigned)DISPLAY_HEIGHT, FRAMEBUFFER_BYTES,
           (uint32_t)resolution_rc, resolution.full_width, resolution.full_height,
           resolution.pane_width, resolution.pane_height, resolution.refresh_rate,
           (uint32_t)output_rc, output.refresh_rate);
}
#endif

#if PS5_SCANOUT_FPS > 60
#ifndef PS5_NATIVE_TITLE_RUNTIME
#error High refresh requires the native title runtime
#endif
extern int sceVideoOutIsOutputSupported(int32_t, uint32_t, const void *, const void *, const void *);
extern int sceVideoOutConfigureOutput(int32_t, uint32_t, const void *, const void *, const void *);
static int runtime_output_needs_restore;
/* Survives EGL teardown; only a successful HFR port close arms this guard. */
static int runtime_output_reopen_pending;
/* The open port was switched to the high-refresh output mode. */
static int runtime_output_hfr_active;

static int runtime_video_configure_output(void)
{
    int support = sceVideoOutIsOutputSupported(runtime_video_handle, 15, NULL, NULL, NULL);
    int preset = -1, result = support > 0 ? 0 : -1;
    if (result == 0) {
        /* Even a partially failed request must be restored before closing. */
        runtime_output_needs_restore = 1;
        result = preset = sceVideoOutConfigureOutput(runtime_video_handle, 15, NULL, NULL, NULL);
    }
    runtime_output_hfr_active = result == 0;
    printf("[ps5-output-mode] target=%u support=%08" PRIx32 " preset=%08" PRIx32
           " vrr=%08" PRIx32 " result=%08" PRIx32 "\n", (unsigned)PS5_SCANOUT_FPS,
           (uint32_t)support, (uint32_t)preset, UINT32_MAX, (uint32_t)result);
    return result;
}

static int runtime_video_restore_output(void)
{
    if (!runtime_output_needs_restore)
        return 0;
    int result = sceVideoOutConfigureOutput(runtime_video_handle, 1, NULL, NULL, NULL);
    int wait = 0;
    for (unsigned i = 0; result == 0 && wait == 0 && i < 2; ++i)
        wait = runtime_video_api.wait_vblank(runtime_video_handle);
    printf("[ps5-output-restore] result=%08" PRIx32 " wait=%08" PRIx32 "\n",
           (uint32_t)result, (uint32_t)wait);
    if (result != 0 || wait != 0)
        return result != 0 ? result : wait;
    runtime_output_needs_restore = 0;
#ifdef PS5_DRAW_PROFILE
    runtime_video_report("restored");
#endif
    return 0;
}
#endif

int ps5_agc_gate2_shutdown_present(void)
{
    int close_rc = 0;
#ifdef PS5_GPU_PRESENT_BATCH
    if (ps5_agc_gate2_wait_present() != 0)
        return -1;
    if (runtime_gpu_present_buffer >= 0)
        return -1; /* Unconfirmed presentation still owns scanout. */
#endif
#ifdef PS5_MULTIDRAW_BATCH
    if (runtime_batch_faulted || runtime_batch_active || runtime_batch_count ||
        runtime_pending_batches)
        return -1;
#endif
#ifdef PS5_ASYNC_NATIVE_PREP
    runtime_async_stop();
#endif

#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    runtime_require_retirement(runtime_work_cache_clear() == 0);
    runtime_require_retirement(runtime_shader_pair_clear() == 0);
    runtime_require_retirement(runtime_compute_program_clear() == 0);
#endif

#ifdef PS5_FRAME_SUSPEND
    runtime_require_retirement(runtime_end_submit_period() == 0);
#endif
#ifdef PS5_DRAW_PROFILE
    runtime_profile_report();
#endif
    if (runtime_video_registered) {
        int drain_rc = runtime_video_wait_idle();
        printf("[ps5-agc] present-drain result=%08" PRIx32 "\n", (uint32_t)drain_rc);
        if (drain_rc != 0)
            return drain_rc; /* Preserve scanout ownership if retirement is unknown. */
#if defined(PS5_DRAW_PROFILE) && defined(PS5_NATIVE_TITLE_RUNTIME)
        runtime_video_report("end");
#endif
        /* An idle flip queue can still scan out its last registered buffer.
         * This path always closes the whole port; close releases its buffers.
         * Retain registration/allocation ownership until that close succeeds. */
    }
#if PS5_SCANOUT_FPS > 60
    if (runtime_video_handle >= 0) {
        int restore_rc = runtime_video_restore_output();
        if (restore_rc != 0)
            return restore_rc;
    }
#endif
    if (runtime_video_handle >= 0)
        close_rc = runtime_video_api.close(runtime_video_handle);
    if (runtime_video_handle >= 0)
        printf("[ps5-agc] present-shutdown method=close"
               " close=%08" PRIx32 " frames=%u\n",
               (uint32_t)close_rc,
               runtime_present_count);
    /* A failed close does not release ownership of the scanout allocation. */
    if (close_rc != 0)
        return close_rc;
#if PS5_SCANOUT_FPS > 60
    if (runtime_video_handle >= 0 && runtime_output_hfr_active)
        runtime_output_reopen_pending = 1;
    runtime_output_hfr_active = 0;
#endif
    memset(&runtime_video_api, 0, sizeof(runtime_video_api));
    runtime_video_handle = -1;
    runtime_video_framebuffer = NULL;
    runtime_video_framebuffer_size = 0;
    runtime_video_registered = 0;
    runtime_scanout_flush_needed = 1;
    runtime_present_count = 0;
#ifdef PS5_GPU_PRESENT_BATCH
    if (runtime_gpu_present_count)
        printf("[ps5-gpu-present] frames=%u\n", runtime_gpu_present_count);
    runtime_gpu_present_count = 0;
#endif
    runtime_render_marker = (uint64_t)RENDER_MARKER;
    return close_rc;
}

#ifdef PS5_DYNAMIC_SCANOUT
/* Display modes change only with no port, registration or queued GPU work. */
int ps5_agc_gate2_display_idle(void)
{
    int idle = runtime_video_handle < 0 && !runtime_video_registered;
#ifdef PS5_GPU_PRESENT_BATCH
    idle = idle && runtime_gpu_present_buffer < 0;
#endif
#ifdef PS5_MULTIDRAW_BATCH
    idle = idle && !runtime_batch_active && !runtime_batch_count &&
           !runtime_batch_faulted && !runtime_pending_batches;
#endif
    return idle;
}
#endif

static int runtime_video_acquire(const video_api_t *video,
                                 uint8_t *framebuffer,
                                 size_t framebuffer_size,
                                 int *attempts)
{
    video_buffer_t buffers[2] = {
        {framebuffer, NULL, NULL, NULL},
        {framebuffer + (framebuffer_size >= FRAMEBUFFER_POOL_BYTES
                            ? FRAMEBUFFER_BYTES : 0),
         NULL, NULL, NULL}
    };
    video_attribute_t attribute = {{0}};

    *attempts = 0;
    if (runtime_video_registered &&
        runtime_video_framebuffer == framebuffer &&
        runtime_video_framebuffer_size >= framebuffer_size)
        return 0;
    if (runtime_video_handle >= 0 && ps5_agc_gate2_shutdown_present() != 0)
        return -1;
#if PS5_SCANOUT_FPS > 60
    if (runtime_output_reopen_pending) {
        /* ponytail: conservative process-local settling, not sink readiness.
         * Keep the proven five-second interval until a shorter one is qualified.
         * First open, live reuse and final close never pay this reopen wait. */
        for (unsigned i = 0; i < 10; ++i) {
            int wait = sceKernelUsleep(UINT32_C(500000));
            if (wait != 0) {
                printf("[ps5-output-reopen] settle_ms=5000 result=%08" PRIx32 "\n",
                       (uint32_t)wait);
                return wait; /* Keep the guard armed; no open/register occurred. */
            }
        }
        runtime_output_reopen_pending = 0;
        printf("[ps5-output-reopen] settle_ms=5000 result=00000000\n");
    }
#endif
    for (int attempt = 1; attempt <= 3; ++attempt) {
        *attempts = attempt;
        runtime_video_handle = video->open(0xff, 0, 0, NULL);
        if (runtime_video_handle >= 0)
            break;
        if (attempt != 3)
            sceKernelUsleep(UINT32_C(500000));
    }
    runtime_video_api = *video;
    if (runtime_video_handle < 0)
        goto fail;
#if PS5_SCANOUT_FPS > 60
#ifdef PS5_DYNAMIC_SCANOUT
    ps5_scanout_active_fps = 60u;
    if (PS5_SCANOUT_HFR_REQUESTED) {
        /* A display without 120 Hz keeps the port at 60 Hz instead of failing. */
        if (runtime_video_configure_output() == 0)
            ps5_scanout_active_fps = 120u;
        else if (runtime_video_restore_output() != 0)
            goto fail;
    }
#else
    if (runtime_video_configure_output() != 0)
        goto fail;
#endif
#endif
    if (video->set_flip_rate(runtime_video_handle, 0) != 0)
        goto fail;
    video->set_attribute2(&attribute, VIDEO_OUT_PIXEL_FORMAT, 0,
                          DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0, 0);
    int register_rc = video->register_buffers2(runtime_video_handle, 0, 0, buffers, 2,
                                               &attribute, 0, NULL);
#ifdef PS5_DRAW_PROFILE
    printf("[ps5-output-register] width=%u height=%u offset=%u result=%08" PRIx32 "\n",
           DISPLAY_WIDTH, (unsigned)DISPLAY_HEIGHT,
           framebuffer_size >= FRAMEBUFFER_POOL_BYTES ? FRAMEBUFFER_BYTES : 0u,
           (uint32_t)register_rc);
#endif
    if (register_rc != 0)
        goto fail;
    runtime_video_framebuffer = framebuffer;
    runtime_video_framebuffer_size = framebuffer_size;
    runtime_video_registered = 1;
    runtime_scanout_flush_needed = 1;
    printf("[ps5-agc] present-open handle=%08" PRIx32
           " framebuffer=%p/%zu buffers=2 alias=%u\n",
           (uint32_t)runtime_video_handle, framebuffer,
           framebuffer_size,
           framebuffer_size < FRAMEBUFFER_POOL_BYTES ? 1u : 0u);
    return 0;

fail: {
    /* Reuse shutdown's ownership rule even when setup failed before registration. */
    int close_rc = ps5_agc_gate2_shutdown_present();
    return close_rc != 0 ? close_rc : -1;
}
}

int ps5_agc_gate2_prepare_present(void *framebuffer, size_t framebuffer_size)
{
    agc_api_t agc = {0};
    video_api_t video = {0};
    int attempts;

    if (!framebuffer || framebuffer_size < FRAMEBUFFER_BYTES ||
        ((uintptr_t)framebuffer & (FRAMEBUFFER_ALIGNMENT - 1u)))
        return -1;
    if (runtime_video_registered &&
        runtime_video_framebuffer == framebuffer &&
        runtime_video_framebuffer_size >= framebuffer_size)
        return 0;
    if (load_apis(NULL, NULL, NULL, &agc, &video) != 0)
        return -1;
    flush_gpu_data(framebuffer, framebuffer_size);
    return runtime_video_acquire(&video, framebuffer, framebuffer_size,
                                 &attempts);
}

static int runtime_video_wait_idle(void)
{
    for (unsigned waits = 0; ; ++waits) {
        int pending = runtime_video_api.is_flip_pending(runtime_video_handle);
        if (pending <= 0)
            return pending;
        if (waits == 120)
            return -1;
        int result = runtime_video_api.wait_vblank(runtime_video_handle);
        if (result != 0)
            return result;
    }
}

static int runtime_video_prepare_draw(void)
{
#ifdef PS5_GPU_PRESENT_BATCH
    /* Offscreen work can overlap a pending flip. Scanout must be reusable
     * before any command capable of writing it enters the GPU queue. */
#ifdef PS5_RUNTIME_WRITES_SCANOUT
    if (PS5_RUNTIME_WRITES_SCANOUT())
#endif
        if (ps5_agc_gate2_wait_present() != 0)
            return -1;
#endif
    return runtime_video_registered ? 0 : -1;
}

#ifdef PS5_ASYNC_NATIVE_PREP
static int runtime_async_prepare_draw(void)
{
    /* Initialization and scanout ownership belong to the ordered producer.
     * Workers only build commands against the already registered pool. */
    if (!runtime_framebuffer || runtime_framebuffer_size < FRAMEBUFFER_BYTES ||
        ((uintptr_t)runtime_framebuffer & (FRAMEBUFFER_ALIGNMENT - 1u)))
        return -1;
    if (!runtime_agc_initialized) {
        agc_api_t agc = {0};
        video_api_t video = {0};
        if (load_apis(NULL, NULL, NULL, &agc, &video) != 0 || agc.init(8) != 0)
            return -1;
        runtime_agc_initialized = 1;
    }
    size_t bytes = runtime_framebuffer_size >= FRAMEBUFFER_POOL_BYTES
        ? FRAMEBUFFER_POOL_BYTES : FRAMEBUFFER_BYTES;
    if (!runtime_video_registered || runtime_video_framebuffer != runtime_framebuffer ||
        runtime_video_framebuffer_size < bytes) {
        if (runtime_async_drain() != 0 ||
            ps5_agc_gate2_prepare_present(runtime_framebuffer, bytes) != 0)
            return -1;
    }
    return runtime_video_prepare_draw();
}
#endif

int ps5_agc_gate2_present(unsigned buffer_index, unsigned swap_interval)
{
    int result;
    int64_t marker;
#ifdef PS5_DRAW_PROFILE
    int64_t profile_ticks[4] = {0};
    const int profile_this_present = runtime_present_count >= 30;
#endif
#ifdef PS5_MULTIDRAW_BATCH
    if (runtime_batch_faulted || runtime_batch_active || runtime_batch_count ||
        (runtime_pending_batches && swap_interval != 0))
        return -1;
#endif

#ifdef PS5_GPU_PRESENT_BATCH
    /* An empty next frame may reach swap without touching its back buffer. */
    if (ps5_agc_gate2_wait_present() != 0)
        return -1;
#endif
    if (buffer_index > 1 || swap_interval > 1 || !runtime_video_registered ||
        runtime_video_handle < 0 ||
        !runtime_video_api.submit_flip ||
        !runtime_video_api.is_flip_pending ||
        !runtime_video_api.wait_vblank)
        return -1;
#ifdef PS5_FRAME_SUSPEND
    runtime_require_retirement(runtime_end_submit_period() == 0);
#endif
    PS5_PROFILE_MARK(0);
    /* The GPU-tail path checks both marker and idle state together below. */
    if (
#ifdef PS5_GPU_PRESENT_BATCH
        runtime_gpu_present_buffer < 0 &&
#endif
        runtime_video_wait_idle() != 0) {
#ifdef PS5_DRAW_PROFILE
        if (profile_this_present)
            runtime_present_profile_record(profile_ticks, -1);
#endif
        return -1;
    }
    PS5_PROFILE_MARK(1);
#ifdef PS5_GPU_PRESENT_BATCH
    if (runtime_gpu_present_buffer >= 0) {
        PS5_PROFILE_MARK(2);
        if (runtime_gpu_present_buffer != (int)buffer_index)
            return -1;
        if (swap_interval == 0 ||
            runtime_video_framebuffer_size >= FRAMEBUFFER_POOL_BYTES) {
            runtime_gpu_present_deferred = 1;
            result = 0;
        } else {
            result = runtime_gpu_present_finish(buffer_index);
        }
    } else
#endif
    {
        marker = runtime_next_render_marker();
        result = runtime_video_api.submit_flip(
            runtime_video_handle, (int)buffer_index, 1, marker);
        PS5_PROFILE_MARK(2);
        if (result == 0) {
#ifdef PS5_GPU_PRESENT_BATCH
            if ((swap_interval == 0 ||
                 runtime_video_framebuffer_size >= FRAMEBUFFER_POOL_BYTES) &&
                runtime_video_api.get_flip_status) {
                /* Explicit GL flush/readback can leave no GPU tail to append
                 * a flip to. It must still overlap the next offscreen frame. */
                runtime_gpu_present_buffer = (int)buffer_index;
                runtime_gpu_present_marker = (uint64_t)marker;
                runtime_gpu_present_is_cpu = 1;
                runtime_gpu_present_deferred = 1;
            } else
#endif
                result = runtime_video_api.wait_vblank(runtime_video_handle);
        }
    }
    PS5_PROFILE_MARK(3);
#ifdef PS5_DRAW_PROFILE
    if (profile_this_present)
        runtime_present_profile_record(profile_ticks, result);
#endif
    if (result == 0) {
        ++runtime_present_count;
#if defined(PS5_DRAW_PROFILE) && defined(PS5_NATIVE_TITLE_RUNTIME)
        if (runtime_present_count == 30)
            runtime_video_report("warmup");
#endif
    }
    return result;
}
#endif

static uint8_t command_out_of_space(agc_command_buffer_t *buffer,
                                    uint32_t words, void *user_data)
{
    (void)buffer;
    (void)words;
    (void)user_data;
    out_of_space = 1;
    return 0;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_to_float(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t fnv1a32(const void *data, size_t bytes)
{
    const uint8_t *at = data;
    uint32_t hash = UINT32_C(2166136261);
    while (bytes--) {
        hash ^= *at++;
        hash *= UINT32_C(16777619);
    }
    return hash;
}

/* sceAgcDcbSet{Cx,Sh,Uc}RegistersIndirect retains the supplied table
 * address until GPU execution.  Reject malformed DCB framing and any
 * indirect register table that escaped the persistent direct-memory work
 * allocation.  This deliberately covers only the three indexed load
 * packets emitted by the AGC helpers used below. */
static int validate_indirect_register_tables(
    const uint32_t *words, uint32_t word_count,
    const void *work, size_t work_bytes,
    uint32_t *bad_packet, uint32_t *bad_opcode,
    uint64_t *bad_address, uint32_t *bad_count)
{
    const uint64_t work_begin = (uint64_t)(uintptr_t)work;
    const uint64_t work_end = work_begin + work_bytes;
    uint32_t at = 0;
    uint32_t packet = 0;

    if (!words || !work || work_end < work_begin)
        return -1;
    while (at < word_count) {
        const uint32_t header = words[at];
        const uint32_t type = header >> 30;
        uint32_t packet_words;

        if (type == 2) {
            packet_words = 1;
        } else if (type == 0 || type == 3) {
            packet_words = ((header >> 16) & UINT32_C(0x3fff)) + 2u;
        } else {
            return -1;
        }
        if (packet_words > word_count - at)
            return -1;

        if (type == 3) {
            const uint32_t opcode = (header >> 8) & 0xffu;
            if (opcode == 0x63u || opcode == 0x64u || opcode == 0x9fu) {
                uint64_t address;
                uint64_t table_bytes;
                uint32_t count;

                if (packet_words < 5u)
                    return -1;
                address = (uint64_t)words[at + 1u] |
                          ((uint64_t)words[at + 2u] << 32);
                count = words[at + 4u];
                table_bytes = (uint64_t)count * sizeof(agc_register_t);
                if (address < work_begin || address > work_end ||
                    table_bytes > work_end - address) {
                    if (bad_packet)
                        *bad_packet = packet;
                    if (bad_opcode)
                        *bad_opcode = opcode;
                    if (bad_address)
                        *bad_address = address;
                    if (bad_count)
                        *bad_count = count;
                    return -2;
                }
            }
        }
        at += packet_words;
        packet++;
    }
    return 0;
}

#ifdef AGC_DEPTH_DEFAULTS_PROBE
static int dump_depth_register_template(void *agc_module)
{
    static const uint32_t depth_template_hash = UINT32_C(0x67096014);
    agc_get_register_defaults2_fn get_defaults2 = NULL;
    uint8_t *defaults;
    uint8_t *metadata;
    uint64_t count;
    uint64_t index;

    *(void **)(&get_defaults2) =
        dlsym(agc_module, "sceAgcGetRegisterDefaults2");
    if (!get_defaults2) {
        printf(LOG_PREFIX " sceAgcGetRegisterDefaults2 unavailable: %s\n",
               dlerror());
        return 1;
    }
    defaults = get_defaults2();
    if (!defaults)
        return 1;
    memcpy(&metadata, defaults + 0x30, sizeof(metadata));
    memcpy(&count, defaults + 0x38, sizeof(count));
    printf(LOG_PREFIX " defaults2=%p metadata=%p count=%" PRIu64 "\n",
           (void *)defaults, (void *)metadata, count);
    if (!metadata || count > UINT64_C(0x100000))
        return 1;

    for (index = 0; index < count; ++index) {
        uint32_t hash, encoded, metadata2;
        uint8_t *page;
        agc_register_t *records;
        uint32_t record;

        memcpy(&hash, metadata + index * 12, sizeof(hash));
        if (hash != depth_template_hash)
            continue;
        memcpy(&encoded, metadata + index * 12 + 4, sizeof(encoded));
        memcpy(&metadata2, metadata + index * 12 + 8,
               sizeof(metadata2));
        memcpy(&page, defaults + (encoded & 3u) * sizeof(page),
               sizeof(page));
        if (!page)
            return 1;
        memcpy(&records,
               page + ((encoded & 0x3fcu) >> 2) * sizeof(records),
               sizeof(records));
        printf(LOG_PREFIX
               " match index=%" PRIu64 " encoded=0x%08" PRIx32
               " metadata2=0x%08" PRIx32 " page=%u slot=%u records=%p\n",
               index, encoded, metadata2, encoded & 3u,
               (encoded & 0x3fcu) >> 2, (void *)records);
        if (!records)
            return 1;
        for (record = 0; record < 16; ++record)
            printf(LOG_PREFIX
                   " depth_record[%02u] offset=0x%04" PRIx16
                   " padding=0x%04" PRIx16 " value=0x%08" PRIx32 "\n",
                   record, records[record].offset,
                   records[record].padding, records[record].value);
        return 0;
    }
    printf(LOG_PREFIX " hash=0x%08" PRIx32 " not found\n",
           depth_template_hash);
    return 1;
}
#endif

static void (flush_gpu_data)(const void *address, size_t bytes)
{
#ifdef PS5_NATIVE_TITLE_RUNTIME
    if (bytes)
        util_flush_inval_range((void *)address, bytes);
#else
    const uintptr_t end = (uintptr_t)address + bytes;
    uintptr_t at = (uintptr_t)address & ~(uintptr_t)63;
    for (; bytes && at < end; at += 64)
        __asm__ volatile("clflush (%0)" : : "r"(at) : "memory");
    __asm__ volatile("mfence" ::: "memory");
#endif
}

#if (defined(AGC_RENDER_TO_TEXTURE_VARIANT) || \
     defined(AGC_CLEAR_TEST_VARIANT)) && defined(AGC_TRIANGLE_SUBMIT)
static size_t tiled_rgba8_offset(uint32_t x, uint32_t y,
                                 uint32_t width)
{
    size_t offset =
        ((size_t)(y << 4) & 0x70u) ^
        ((size_t)(y << 5) & 0xf00u) ^
        ((size_t)(y << 9) & 0x1000u) ^
        ((size_t)(y << 8) & 0x4000u) ^
        ((size_t)(x << 2) & 0x0cu) ^
        ((size_t)(x << 5) & 0x380u) ^
        ((size_t)(x << 4) & 0x400u) ^
        ((size_t)(x << 6) & 0x800u) ^
        ((size_t)(x << 9) & 0xa000u);
    size_t blocks_per_row = (width + 127u) >> 7;

    return ((((size_t)y >> 7) * blocks_per_row + (x >> 7)) << 16) +
           offset;
}

static int read_tiled_rgba8_region(const uint8_t *source,
                                   size_t source_bytes,
                                   uint32_t image_width,
                                   uint32_t image_height,
                                   uint32_t x, uint32_t y,
                                   uint32_t width, uint32_t height,
                                   uint32_t *destination,
                                   uint32_t destination_pitch)
{
    uint32_t row;

    if (!source || !destination || !width || !height ||
        destination_pitch < width || x >= image_width || y >= image_height ||
        width > image_width - x || height > image_height - y)
        return -1;
    for (row = 0; row < height; ++row) {
        uint32_t column;
        for (column = 0; column < width; ++column) {
            size_t offset = tiled_rgba8_offset(x + column, y + row,
                                               image_width);
            if (offset > source_bytes || source_bytes - offset < 4)
                return -1;
            memcpy(destination + (size_t)row * destination_pitch + column,
                   source + offset, 4);
        }
    }
    return 0;
}
#endif

static int shader_sections(const uint8_t *elf, size_t elf_size,
                           const uint8_t **header, size_t *header_size,
                           const uint8_t **code, size_t *code_size)
{
    uint64_t section_offset;
    uint16_t entry_size, count, names_index, index;
    const uint8_t *names;
    size_t names_size;

    if (elf_size < 64 || memcmp(elf, "\x7f" "ELF", 4) != 0)
        return -1;
    memcpy(&section_offset, elf + 40, sizeof(section_offset));
    memcpy(&entry_size, elf + 58, sizeof(entry_size));
    memcpy(&count, elf + 60, sizeof(count));
    memcpy(&names_index, elf + 62, sizeof(names_index));
    if (!section_offset || entry_size < 64 || !count || names_index >= count ||
        section_offset > elf_size ||
        (size_t)count > (elf_size - (size_t)section_offset) / entry_size)
        return -1;
    {
        const uint8_t *record = elf + section_offset + names_index * entry_size;
        uint64_t offset, size;
        memcpy(&offset, record + 24, sizeof(offset));
        memcpy(&size, record + 32, sizeof(size));
        if (offset > elf_size || size > elf_size - (size_t)offset)
            return -1;
        names = elf + offset;
        names_size = (size_t)size;
    }

    *header = NULL;
    *code = NULL;
    for (index = 0; index < count; ++index) {
        const uint8_t *record = elf + section_offset + index * entry_size;
        uint32_t name_offset;
        uint64_t offset, size;
        const char *name;
        memcpy(&name_offset, record, sizeof(name_offset));
        memcpy(&offset, record + 24, sizeof(offset));
        memcpy(&size, record + 32, sizeof(size));
        if (name_offset >= names_size || offset > elf_size ||
            size > elf_size - (size_t)offset ||
            !memchr(names + name_offset, 0, names_size - name_offset))
            return -1;
        name = (const char *)names + name_offset;
        if (strcmp(name, ".shader_header") == 0) {
            *header = elf + offset;
            *header_size = (size_t)size;
        } else if (strcmp(name, ".shader_text") == 0) {
            *code = elf + offset;
            *code_size = (size_t)size;
        }
    }
    return *header && *code && *header_size >= 96 && *code_size ? 0 : -1;
}

static int validate_shader_header(const uint8_t *header, size_t header_size,
                                  size_t code_size, uint8_t expected_stage)
{
    uint32_t magic, version, declared_header, declared_code;
    uint64_t resource_offset, serialized_code;

    if (header_size < 96)
        return -1;
    memcpy(&magic, header, sizeof(magic));
    memcpy(&version, header + 4, sizeof(version));
    memcpy(&resource_offset, header + 8, sizeof(resource_offset));
    memcpy(&serialized_code, header + 16, sizeof(serialized_code));
    memcpy(&declared_header, header + 64, sizeof(declared_header));
    memcpy(&declared_code, header + 68, sizeof(declared_code));
    if (magic != UINT32_C(0x34333231))
        return -2;
    if (version != 24)
        return -3;
    if (!resource_offset || serialized_code)
        return -4;
    if (declared_header != header_size || declared_code != code_size)
        return -5;
    if (header[0x5a] != expected_stage)
        return -6;
    return 0;
}

static int load_apis(void *agc_module, void *driver_module,
                     void *video_module, agc_api_t *agc, video_api_t *video)
{
#ifdef PS5_NATIVE_TITLE_RUNTIME
    extern int sceAgcInit(uint32_t);
    extern int sceAgcCreateShader(void **, void *, void *);
    extern int sceAgcLinkShaders(void *, void *, void *, void *, void *,
                                 uint32_t);
    extern void *sceAgcGetRegisterDefaults(void);
    extern uint32_t *sceAgcDcbSetCxRegistersIndirect(void *, const void *,
                                                      uint32_t);
    extern uint32_t *sceAgcDcbSetShRegistersIndirect(void *, const void *,
                                                      uint32_t);
    extern uint32_t *sceAgcDcbSetUcRegistersIndirect(void *, const void *,
                                                      uint32_t);
    extern uint32_t *sceAgcCbSetShRegisterRangeDirect(
        void *, uint32_t, const uint32_t *, uint32_t);
    extern uint32_t *sceAgcDcbDrawIndexAuto(void *, uint32_t, uint64_t);
    extern uint32_t *sceAgcDcbSetNumInstances(void *, uint32_t);
    extern uint32_t *sceAgcDcbSetIndexSize(void *, uint8_t, uint8_t);
    extern uint32_t *sceAgcDcbSetIndexBuffer(void *, void *);
    extern uint32_t *sceAgcDcbSetIndexCount(void *, uint32_t);
    extern uint32_t *sceAgcDcbDrawIndex(void *, uint32_t, void *, uint64_t);
    extern uint32_t *sceAgcCbReleaseMem(
        void *, uint8_t, int16_t, uint64_t, int8_t, void *, uint32_t,
        uint64_t, uint16_t, uint16_t, int8_t, int32_t);
    extern uint32_t *sceAgcDcbSetFlip(void *, uint32_t, int, uint32_t,
                                      int64_t);
    extern int sceAgcSuspendPoint(void);
    extern uint32_t sceAgcDriverGetWaitRenderingPacketSizeInDwords(void);
    extern uint32_t sceAgcDriverWaitUntilSafeForRendering(
        uint32_t **, uint32_t, uint32_t, uint32_t, int);
    extern int sceAgcDriverSubmitDcb(void *);
    extern int sceAgcDriverSetTFRing(uintptr_t, uint32_t);
    extern int sceAgcDriverGetTFRing(uintptr_t *, uint32_t *);
    extern int sceVideoOutOpen(int32_t, int32_t, int32_t, const void *);
    extern int sceVideoOutClose(int32_t);
    extern int sceVideoOutSetFlipRate(int32_t, int32_t);
    extern void sceVideoOutSetBufferAttribute2(
        video_attribute_t *, uint64_t, uint32_t, uint32_t, uint32_t,
        uint64_t, uint32_t, uint64_t);
    extern int sceVideoOutRegisterBuffers2(
        int32_t, int32_t, int32_t, video_buffer_t *, int32_t,
        video_attribute_t *, int32_t, void *);
    extern int sceVideoOutUnregisterBuffers(int32_t, int32_t);
    extern int sceVideoOutSubmitFlip(int32_t, int32_t, uint32_t, int64_t);
    extern int sceVideoOutIsFlipPending(int32_t);
    extern int sceVideoOutWaitVblank(int32_t);
    extern int sceVideoOutGetFlipStatus(int32_t, void *);

    (void)agc_module;
    (void)driver_module;
    (void)video_module;
    *agc = (agc_api_t){
        sceAgcInit,
        sceAgcCreateShader,
        sceAgcLinkShaders,
        sceAgcGetRegisterDefaults,
        sceAgcDcbSetCxRegistersIndirect,
        sceAgcDcbSetShRegistersIndirect,
        sceAgcDcbSetUcRegistersIndirect,
        sceAgcCbSetShRegisterRangeDirect,
        sceAgcDcbDrawIndexAuto,
        sceAgcDcbSetIndexSize,
        sceAgcDcbSetIndexBuffer,
        sceAgcDcbSetIndexCount,
        sceAgcDcbDrawIndex,
        sceAgcCbReleaseMem,
        sceAgcDcbSetFlip,
        sceAgcSuspendPoint,
        sceAgcDriverGetWaitRenderingPacketSizeInDwords,
        sceAgcDriverWaitUntilSafeForRendering,
        sceAgcDriverSubmitDcb,
        sceAgcDriverSetTFRing,
        sceAgcDriverGetTFRing,
    };
    *video = (video_api_t){
        sceVideoOutOpen,
        sceVideoOutClose,
        sceVideoOutSetFlipRate,
        sceVideoOutSetBufferAttribute2,
        sceVideoOutRegisterBuffers2,
        sceVideoOutUnregisterBuffers,
        sceVideoOutSubmitFlip,
        sceVideoOutIsFlipPending,
        sceVideoOutWaitVblank,
        sceVideoOutGetFlipStatus,
    };
#ifdef PS5_AGC_BIND_NATIVE_API
    if (PS5_AGC_BIND_NATIVE_API(agc, sceAgcDcbSetNumInstances) != 0)
        return -1;
#endif
    return 0;
#else
#define LOAD(dst, module, name)                                               \
    do {                                                                      \
        *(void **)(&(dst)) = dlsym((module), (name));                         \
        if (!(dst))                                                           \
            return -1;                                                        \
    } while (0)
    LOAD(agc->init, agc_module, "sceAgcInit");
    LOAD(agc->create_shader, agc_module, "sceAgcCreateShader");
    LOAD(agc->link_shaders, agc_module, "sceAgcLinkShaders");
    LOAD(agc->get_defaults, agc_module, "sceAgcGetRegisterDefaults");
    LOAD(agc->set_cx, agc_module, "sceAgcDcbSetCxRegistersIndirect");
    LOAD(agc->set_sh, agc_module, "sceAgcDcbSetShRegistersIndirect");
    LOAD(agc->set_uc, agc_module, "sceAgcDcbSetUcRegistersIndirect");
    LOAD(agc->set_sh_direct, agc_module,
         "sceAgcCbSetShRegisterRangeDirect");
    LOAD(agc->draw_auto, agc_module, "sceAgcDcbDrawIndexAuto");
    LOAD(agc->set_index_size, agc_module, "sceAgcDcbSetIndexSize");
    LOAD(agc->set_index_buffer, agc_module, "sceAgcDcbSetIndexBuffer");
    LOAD(agc->set_index_count, agc_module, "sceAgcDcbSetIndexCount");
    LOAD(agc->draw_index, agc_module, "sceAgcDcbDrawIndex");
    LOAD(agc->release_mem, agc_module, "sceAgcCbReleaseMem");
    LOAD(agc->set_flip, agc_module, "sceAgcDcbSetFlip");
    LOAD(agc->suspend_point, agc_module, "sceAgcSuspendPoint");
    LOAD(agc->wait_size, driver_module,
         "sceAgcDriverGetWaitRenderingPacketSizeInDwords");
    LOAD(agc->wait_rendering, driver_module,
         "sceAgcDriverWaitUntilSafeForRendering");
    LOAD(agc->submit, driver_module, "sceAgcDriverSubmitDcb");
    LOAD(agc->set_tf_ring, driver_module, "sceAgcDriverSetTFRing");
    LOAD(agc->get_tf_ring, driver_module, "sceAgcDriverGetTFRing");

    LOAD(video->open, video_module, "sceVideoOutOpen");
    LOAD(video->close, video_module, "sceVideoOutClose");
    LOAD(video->set_flip_rate, video_module, "sceVideoOutSetFlipRate");
    LOAD(video->set_attribute2, video_module,
         "sceVideoOutSetBufferAttribute2");
    LOAD(video->register_buffers2, video_module,
         "sceVideoOutRegisterBuffers2");
    LOAD(video->unregister_buffers, video_module,
         "sceVideoOutUnregisterBuffers");
    LOAD(video->submit_flip, video_module, "sceVideoOutSubmitFlip");
    LOAD(video->is_flip_pending, video_module,
         "sceVideoOutIsFlipPending");
    LOAD(video->wait_vblank, video_module, "sceVideoOutWaitVblank");
    LOAD(video->get_flip_status, video_module, "sceVideoOutGetFlipStatus");
#undef LOAD
    return 0;
#endif
}

static int append_target_state(agc_register_t *cx, uint32_t *cx_count,
                               void *defaults, void *target)
{
    static const uint16_t target_offsets[16] = {
        0x318, 0x31b, 0x31c, 0x31d, 0x31e, 0x31f, 0x321, 0x323,
        0x324, 0x325, 0x390, 0x398, 0x3a0, 0x3a8, 0x3b0, 0x3b8
    };
    agc_register_t **blocks = *(agc_register_t ***)defaults;
    uint32_t default_count = *(uint32_t *)((uint8_t *)defaults + 0x20);
    uint32_t index;

    if (!blocks || !blocks[0])
        return -1;
    for (index = 0; index < 16; ++index)
        cx[index] = (agc_register_t){target_offsets[index], 0, 0};
    /* Read each default once instead of rescanning the table for every target
     * register. Keep the first occurrence, matching the original lookup. */
    uint32_t found = 0;
    for (uint32_t candidate = 0; candidate < default_count && found != 0xffffu; ++candidate) {
        switch (blocks[0][candidate].offset) {
        case 0x318: index = 0; break;
        case 0x31b: index = 1; break;
        case 0x31c: index = 2; break;
        case 0x31d: index = 3; break;
        case 0x31e: index = 4; break;
        case 0x31f: index = 5; break;
        case 0x321: index = 6; break;
        case 0x323: index = 7; break;
        case 0x324: index = 8; break;
        case 0x325: index = 9; break;
        case 0x390: index = 10; break;
        case 0x398: index = 11; break;
        case 0x3a0: index = 12; break;
        case 0x3a8: index = 13; break;
        case 0x3b0: index = 14; break;
        case 0x3b8: index = 15; break;
        default: continue;
        }
        if (!(found & (1u << index))) {
            cx[index].value = blocks[0][candidate].value;
            found |= 1u << index;
        }
    }
    if (found != 0xffffu)
        return -1;

    cx[0].value = (uint32_t)((uintptr_t)target >> 8);
    cx[1].value &= 0xfc001fffu;
    cx[2].value = (cx[2].value &
                   ~(0x7cu | 0x700u | 0x1800u | 0x10000000u |
                     0x10000u | 0x8000u | 0x40000u | 0x4000u)) |
                  0x28u | 0x8000u;
    cx[3].value &= ~(0x7000u | 0x18000u);
    cx[4].value = (cx[4].value &
                   ~(0x60u | 0x0cu | 0x00100200u | 0x80000u)) |
                  0x48u;
    cx[5].value = cx[6].value = cx[9].value = 0;
    cx[10].value = (cx[10].value & 0xffffff00u) |
                   (uint32_t)((uintptr_t)target >> 40);
    cx[11].value &= 0xffffff00u;
    cx[12].value &= 0xffffff00u;
    cx[13].value &= 0xffffff00u;
    cx[14].value = (DISPLAY_HEIGHT - 1u) |
                   ((DISPLAY_WIDTH - 1u) << 14);
    cx[15].value = (cx[15].value &
                    ~(0x1fffu | 0x7c000u | 0x03000000u | 0x44000000u)) |
                   0x6c000u | 0x01000000u | 0x44000000u;
    *cx_count = 16;

#define ADD_CX(offset_, value_)                                               \
    do {                                                                      \
        cx[(*cx_count)++] =                                                   \
            (agc_register_t){(offset_), 0, (value_)};                         \
    } while (0)
#ifdef AGC_TEXTURE_VIEWPORT_HALF
    ADD_CX(0x10f, float_bits(DISPLAY_WIDTH * .25f));
    ADD_CX(0x110, float_bits(DISPLAY_WIDTH * .25f));
#else
    ADD_CX(0x10f, float_bits(DISPLAY_WIDTH * .5f));
    ADD_CX(0x110, float_bits(DISPLAY_WIDTH * .5f));
#endif
    ADD_CX(0x111, float_bits(DISPLAY_HEIGHT * -.5f));
    ADD_CX(0x112, float_bits(DISPLAY_HEIGHT * .5f));
    ADD_CX(0x113, float_bits(1));
    ADD_CX(0x114, 0);
    ADD_CX(0x0b4, 0);
    ADD_CX(0x0b5, float_bits(1));
    ADD_CX(0x2fa, float_bits(1));
    ADD_CX(0x2fb, float_bits(1));
    ADD_CX(0x2fc, float_bits(1));
    ADD_CX(0x2fd, float_bits(1));
    ADD_CX(0x090, 0x80000000u);
#ifdef AGC_TEXTURE_SCISSOR_HALF
    ADD_CX(0x091, (DISPLAY_WIDTH / 2u) | (DISPLAY_HEIGHT << 16));
#else
    ADD_CX(0x091, DISPLAY_WIDTH | (DISPLAY_HEIGHT << 16));
#endif
    ADD_CX(0x08e, 0x0f);
#undef ADD_CX
    return 0;
}

#if defined(AGC_DEPTH_TEST_VARIANT) || defined(AGC_RUNTIME_PACKAGES)
static void append_depth_target_state(agc_register_t *cx,
                                      uint32_t *cx_count, void *depth,
                                      void *stencil)
{
    static const agc_register_t template[16] = {
        {0x0010, 0, UINT32_C(0x80000183)}, /* DB_Z_INFO: D32F, 64KB_Z_X */
        {0x0011, 0, UINT32_C(0x20000180)}, /* DB_STENCIL_INFO: disabled */
        {0x0012, 0, 0}, {0x0013, 0, 0},
        {0x0014, 0, 0}, {0x0015, 0, 0},
        {0x001a, 0, 0}, {0x001b, 0, 0},
        {0x001c, 0, 0}, {0x001d, 0, 0},
        {0x001e, 0, 0}, {0x0002, 0, 0},
        {0x0005, 0, 0}, {0x0007, 0, 0},
        {0x000b, 0, 0}, {0x000a, 0, 0},
    };
    agc_register_t *records = cx + *cx_count;
    uintptr_t address = (uintptr_t)depth;
    uintptr_t stencil_address = (uintptr_t)stencil;

    memcpy(records, template, sizeof(template));
#ifdef AGC_RUNTIME_PACKAGES
    if (runtime_depth_samples == 4)
        records[0].value |= UINT32_C(2) << 2; /* NUM_SAMPLES = log2(4). */
    records[11].value = runtime_depth_view;
#endif
    if (stencil) {
        records[1].value = UINT32_C(0x20000181); /* S8, 64KB_Z_X */
        records[3].value = (uint32_t)(stencil_address >> 8);
        records[5].value = (uint32_t)(stencil_address >> 8);
        records[7].value = (uint32_t)(stencil_address >> 40);
        records[9].value = (uint32_t)(stencil_address >> 40);
    }
    records[2].value = (uint32_t)(address >> 8);  /* Z read base */
    records[4].value = (uint32_t)(address >> 8);  /* Z write base */
    records[6].value = (uint32_t)(address >> 40); /* Z read base high */
    records[8].value = (uint32_t)(address >> 40); /* Z write base high */
    records[13].value = (DISPLAY_WIDTH - 1u) |
                        ((DISPLAY_HEIGHT - 1u) << 16);
    *cx_count += 16;
}
#endif

static int append_shader_state(uint8_t *memory, void *vertex, void *pixel,
                               agc_register_t *cx, uint32_t *cx_count,
                               agc_register_t *sh, uint32_t *sh_count)
{
    agc_register_t *vs_cx = *(agc_register_t **)((uint8_t *)vertex + 24);
    agc_register_t *ps_cx = *(agc_register_t **)((uint8_t *)pixel + 24);
    agc_register_t *vs_sh = *(agc_register_t **)((uint8_t *)vertex + 32);
    agc_register_t *ps_sh = *(agc_register_t **)((uint8_t *)pixel + 32);
    uint32_t vs_cx_count = *((uint8_t *)vertex + 91);
    uint32_t ps_cx_count = *((uint8_t *)pixel + 91);
    uint32_t vs_sh_count = *((uint8_t *)vertex + 92);
    uint32_t ps_sh_count = *((uint8_t *)pixel + 92);

    if (!vs_cx || !ps_cx || !vs_sh || !ps_sh || vs_cx_count > 32 ||
        ps_cx_count > 32 || vs_sh_count > 16 || ps_sh_count > 16 ||
        *cx_count + 34 + vs_cx_count + ps_cx_count > 128)
        return -1;
    memcpy(cx + *cx_count, memory + 0x5000, 34 * sizeof(*cx));
    *cx_count += 34;
    memcpy(cx + *cx_count, vs_cx, vs_cx_count * sizeof(*cx));
    *cx_count += vs_cx_count;
    memcpy(cx + *cx_count, ps_cx, ps_cx_count * sizeof(*cx));
    *cx_count += ps_cx_count;
    memcpy(sh, vs_sh, vs_sh_count * sizeof(*sh));
    memcpy(sh + vs_sh_count, ps_sh, ps_sh_count * sizeof(*sh));
    *sh_count = vs_sh_count + ps_sh_count;
    return 0;
}

static int append_hull_shader_state(void *hull, agc_register_t *sh,
                                    uint32_t *sh_count)
{
    agc_register_t *hull_sh = *(agc_register_t **)((uint8_t *)hull + 32);
    uint32_t hull_cx_count = *((uint8_t *)hull + 91);
    uint32_t hull_sh_count = *((uint8_t *)hull + 92);

    if (!hull_sh || hull_cx_count || hull_sh_count != 2 ||
        *sh_count + hull_sh_count + 1 > 32)
        return -1;
    memcpy(sh + *sh_count, hull_sh, hull_sh_count * sizeof(*sh));
    *sh_count += hull_sh_count;
    sh[(*sh_count)++] = (agc_register_t){0x010b, 0, runtime_hs_rsrc2};
    return 0;
}

static uint32_t last_register_value(const agc_register_t *registers,
                                    uint32_t count, uint16_t offset)
{
    while (count) {
        --count;
        if (registers[count].offset == offset)
            return registers[count].value;
    }
    return UINT32_MAX;
}

static int capture_commands(const uint32_t *words, uint32_t word_count,
                            const uint8_t *memory)
{
    FILE *dcb = fopen(DCB_DUMP_PATH, "wb");
    FILE *refs = fopen(MEMORY_DUMP_PATH, "wb");
    int ok = dcb && refs;
    if (dcb) {
        ok &= fwrite(words, sizeof(*words), word_count, dcb) == word_count;
        ok &= fclose(dcb) == 0;
    }
    if (refs) {
        ok &= fwrite(memory, 1, WORK_BYTES, refs) == WORK_BYTES;
        ok &= fclose(refs) == 0;
    }
    return ok ? 0 : -1;
}

#ifdef AGC_TRIANGLE_SUBMIT
static unsigned wait_for_flip_marker(const video_api_t *video, int handle,
                                     uint64_t marker, unsigned maximum,
                                     uint64_t status[16])
{
    unsigned waits;
    memset(status, 0, 16 * sizeof(*status));
    for (waits = 0; waits < maximum; ++waits) {
        if (video->get_flip_status(handle, status) == 0 &&
            status[3] == marker)
            break;
        video->wait_vblank(handle);
    }
    return waits;
}
#endif

#ifdef AGC_FRAME_SLOTS_VARIANT
#define FRAME_SLOT_BYTES 0x10000u
#define FRAME_SLOT_COUNT 2u
#define FRAME_TEST_COUNT 4u
#ifdef AGC_4K
#define FRAME_SLOT_DCB_FORMAT \
    "/data/VdecHello/opengl33-frame-slots-4k-f%u.dcb"
#define FRAME_SLOT_TARGET0_PATH \
    "/data/VdecHello/opengl33-frame-slots-4k-target0.bgra"
#define FRAME_SLOT_TARGET1_PATH \
    "/data/VdecHello/opengl33-frame-slots-4k-target1.bgra"
#else
#define FRAME_SLOT_DCB_FORMAT \
    "/data/VdecHello/opengl33-frame-slots-f%u.dcb"
#define FRAME_SLOT_TARGET0_PATH \
    "/data/VdecHello/opengl33-frame-slots-target0.bgra"
#define FRAME_SLOT_TARGET1_PATH \
    "/data/VdecHello/opengl33-frame-slots-target1.bgra"
#endif

static int write_blob(const char *path, const void *data, size_t bytes)
{
    FILE *file = fopen(path, "wb");
    int ok = file && fwrite(data, 1, bytes, file) == bytes;
    if (file)
        ok &= fclose(file) == 0;
    return ok ? 0 : -1;
}

static int initialize_frame_slot(uint8_t *slot, uint32_t color)
{
    static const float vertices[16] = {
        -0.75f, -0.75f, 0.0f, 1.0f,
         0.75f, -0.75f, 1.0f, 1.0f,
         0.75f,  0.75f, 1.0f, 0.0f,
        -0.75f,  0.75f, 0.0f, 0.0f,
    };
    static const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    uint32_t *vertex_descriptor = (uint32_t *)(slot + 0x4000);
    void *vertex_data = slot + 0x4100;
    uint32_t *combined_descriptor = (uint32_t *)(slot + 0x4300);
    uint32_t *texels = (uint32_t *)(slot + 0xc000);
    uintptr_t vertex_address = (uintptr_t)vertex_data;
    uintptr_t texture_address = (uintptr_t)texels;
    uint32_t texel;

    if ((uint32_t)(vertex_address >> 32) != GATE3_BIND_ADDRESS32_HI ||
        (uint32_t)(texture_address >> 32) != GATE3_TEXTURE_PS_ADDRESS32_HI ||
        GATE3_BIND_VERTEX_BUFFER_USER_DWORD >=
            GATE3_BIND_USER_SGPR_COUNT ||
        GATE3_TEXTURE_PS_DESCRIPTOR_SET0_USER_DWORD >=
            GATE3_TEXTURE_PS_USER_SGPR_COUNT ||
        GATE3_TEXTURE_PS_SET0_BINDING0_OFFSET != 0 ||
        GATE3_TEXTURE_PS_SET0_BINDING0_STRIDE != 48)
        return -1;

    memset(slot, 0, FRAME_SLOT_BYTES);
    memcpy(vertex_data, vertices, sizeof(vertices));
    memcpy(slot + 0x4200, indices, sizeof(indices));
    vertex_descriptor[0] = (uint32_t)vertex_address;
    vertex_descriptor[1] =
        (uint32_t)(vertex_address >> 32) | (16u << 16);
    vertex_descriptor[2] = 4;
    vertex_descriptor[3] = 0x5204;

    for (texel = 0; texel < TEXTURE_WIDTH * TEXTURE_HEIGHT; ++texel)
        texels[texel] = color;
    combined_descriptor[0] = (uint32_t)(texture_address >> 8);
    combined_descriptor[1] =
        UINT32_C(0xc3800000) | (uint32_t)(texture_address >> 40);
    combined_descriptor[2] = UINT32_C(0x000fc00f);
    combined_descriptor[3] = UINT32_C(0x90000fac);
    combined_descriptor[5] = UINT32_C(0x00400000);
    combined_descriptor[8] = UINT32_C(0x00000092);
    combined_descriptor[9] = UINT32_C(0x00fff000);
    combined_descriptor[10] = UINT32_C(0x08000000);
    return 0;
}

static int build_frame_slot_command(const agc_api_t *agc, int video_handle,
                                    uint32_t buffer_index,
                                    uint8_t *shared_memory, uint8_t *slot,
                                    void *vertex, void *pixel, void *target,
                                    uint64_t marker, uint32_t *word_count,
                                    uint32_t *high_water)
{
    agc_register_t *cx = (agc_register_t *)(slot + 0x7000);
    agc_register_t *sh = (agc_register_t *)(slot + 0x7800);
    uint32_t *words = (uint32_t *)(slot + 0x8000);
    agc_command_buffer_t command = {0};
    uint32_t cx_count = 0, sh_count = 0;
    uint32_t vertex_user_data[GATE3_BIND_USER_SGPR_COUNT];
    uint32_t pixel_user_data[GATE3_TEXTURE_PS_USER_SGPR_COUNT];

    if (append_target_state(cx, &cx_count, agc->get_defaults(), target) != 0 ||
        append_shader_state(shared_memory, vertex, pixel, cx, &cx_count,
                            sh, &sh_count) != 0)
        return -1;

    command.bottom = words;
    command.top = words + COMMAND_BYTES / sizeof(*words);
    command.up = words;
    command.down = command.top;
    command.callback = (uintptr_t)command_out_of_space;
    agc->wait_rendering(&command.up, agc->wait_size(), 0,
                        (uint32_t)video_handle, (int)buffer_index);
    agc->set_cx(&command, cx, cx_count);
    if (!set_linkage_uc_state(agc, &command, shared_memory))
        return -1;
    agc->set_sh(&command, sh, sh_count);

    memset(vertex_user_data, 0, sizeof(vertex_user_data));
    vertex_user_data[GATE3_BIND_VERTEX_BUFFER_USER_DWORD] =
        (uint32_t)(uintptr_t)(slot + 0x4000);
    agc->set_sh_direct(&command, 0x8c, vertex_user_data,
                       GATE3_BIND_USER_SGPR_COUNT);
    memset(pixel_user_data, 0, sizeof(pixel_user_data));
    pixel_user_data[GATE3_TEXTURE_PS_DESCRIPTOR_SET0_USER_DWORD] =
        (uint32_t)(uintptr_t)(slot + 0x4300);
    agc->set_sh_direct(&command, 0x0c, pixel_user_data,
                       GATE3_TEXTURE_PS_USER_SGPR_COUNT);
    agc->set_index_size(&command, AGC_INDEX_SIZE_16, 0);
    agc->set_index_buffer(&command, slot + 0x4200);
    agc->set_index_count(&command, 6);
    agc->draw_index(&command, 6, slot + 0x4200, 0);
#ifdef AGC_TRIANGLE_SUBMIT
    agc->set_flip(&command, (uint32_t)video_handle, (int)buffer_index, 1,
                  (int64_t)marker);
#else
    (void)marker;
#endif
    *word_count = (uint32_t)(command.up - words);
    if (*word_count > *high_water)
        *high_water = *word_count;
    if (out_of_space || command.up < command.bottom ||
        command.up > command.top || *word_count == 0)
        return -1;
    return 0;
}

static int run_frame_slot_test(const agc_api_t *agc, const video_api_t *video,
                               int video_handle, uint8_t *memory,
                               uint8_t *framebuffer, void *vertex, void *pixel)
{
    static const uint32_t colors[FRAME_TEST_COUNT] = {
        UINT32_C(0xff0000ff), UINT32_C(0xff00ff00),
        UINT32_C(0xffff0000), UINT32_C(0xffffffff),
    };
    uint32_t completed_generation[FRAME_SLOT_COUNT] = {UINT32_MAX, UINT32_MAX};
    uint64_t completed_marker[FRAME_SLOT_COUNT] = {0, 0};
    uint32_t high_water = 0;
    uint32_t frame;
    int result = 0;

#ifndef AGC_TRIANGLE_SUBMIT
    (void)video;
    (void)completed_marker;
#endif
    for (frame = 0; frame < FRAME_TEST_COUNT && result == 0; ++frame) {
        uint32_t slot_index = frame & 1u;
        uint8_t *slot = memory + (slot_index + 1u) * FRAME_SLOT_BYTES;
        uint8_t *target = framebuffer + slot_index * FRAMEBUFFER_BYTES;
        uint64_t marker = (uint64_t)RENDER_MARKER + frame;
        uint32_t words = 0;
        uint32_t old_texture_hash = fnv1a32(
            slot + 0xc000, TEXTURE_WIDTH * TEXTURE_HEIGHT * 4u);
        uint32_t expected_previous = frame - FRAME_SLOT_COUNT;
        char path[96];

#ifdef AGC_TRIANGLE_SUBMIT
        if (frame >= FRAME_SLOT_COUNT &&
            (completed_generation[slot_index] != expected_previous ||
             completed_marker[slot_index] !=
                 (uint64_t)RENDER_MARKER + expected_previous)) {
            printf(LOG_PREFIX " refuse_reuse frame=%u slot=%u generation=%u"
                   " marker=%016" PRIx64 "\n", frame, slot_index,
                   completed_generation[slot_index],
                   completed_marker[slot_index]);
            result = 1;
            break;
        }
#else
        (void)expected_previous;
#endif
        if (initialize_frame_slot(slot, colors[frame]) != 0 ||
            build_frame_slot_command(agc, video_handle, slot_index, memory,
                                     slot, vertex, pixel, target, marker,
                                     &words, &high_water) != 0) {
            result = 1;
            break;
        }
        snprintf(path, sizeof(path), FRAME_SLOT_DCB_FORMAT, frame);
        if (write_blob(path, slot + 0x8000,
                       words * sizeof(uint32_t)) != 0) {
            result = 1;
            break;
        }
        printf(LOG_PREFIX " build frame=%u slot=%u reuse=%u old_tex=%08" PRIx32
               " new_tex=%08" PRIx32 " words=%u/%u marker=%016" PRIx64
               "\n", frame, slot_index,
               frame >= FRAME_SLOT_COUNT ? 1u : 0u, old_texture_hash,
               fnv1a32(slot + 0xc000,
                       TEXTURE_WIDTH * TEXTURE_HEIGHT * 4u),
               words, COMMAND_BYTES / 4u, marker);

#ifdef AGC_TRIANGLE_SUBMIT
        {
            agc_submit_description_t submit = {
                slot + 0x8000, words, 0, {0, 0, 0}
            };
            uint64_t status[16];
            unsigned waits;
            int submit_rc;
            int suspend_rc;
            size_t nonzero = 0, matches = 0;
            const uint32_t *pixels;
            size_t pixel;

            flush_gpu_data(slot, FRAME_SLOT_BYTES);
            submit_rc = agc->submit(&submit);
            suspend_rc = submit_rc == 0 ? agc->suspend_point() : -1;
            waits = submit_rc == 0
                        ? wait_for_flip_marker(video, video_handle, marker,
                                               120, status)
                        : 120;
            flush_gpu_data(target,
                           (size_t)DISPLAY_WIDTH * DISPLAY_HEIGHT * 4u);
            pixels = (const uint32_t *)target;
            for (pixel = 0;
                 pixel < (size_t)DISPLAY_WIDTH * DISPLAY_HEIGHT; ++pixel) {
                nonzero += pixels[pixel] != 0;
                matches += pixels[pixel] == colors[frame];
            }
            printf(LOG_PREFIX " complete frame=%u slot=%u submit=%08" PRIx32
                   " suspend=%08" PRIx32 " waits=%u status=%016" PRIx64
                   " nonzero=%zu matches=%zu expected=%08" PRIx32 "\n",
                   frame, slot_index, (uint32_t)submit_rc,
                   (uint32_t)suspend_rc, waits, status[3], nonzero, matches,
                   colors[frame]);
            if (submit_rc != 0 || suspend_rc != 0 || waits >= 120 ||
                status[3] != marker ||
                nonzero != (DISPLAY_WIDTH * 3u / 4u) *
                               (DISPLAY_HEIGHT * 3u / 4u) ||
                matches != (DISPLAY_WIDTH * 3u / 4u) *
                               (DISPLAY_HEIGHT * 3u / 4u)) {
                result = 1;
                break;
            }
            completed_generation[slot_index] = frame;
            completed_marker[slot_index] = marker;
        }
#endif
    }

    if (write_blob(MEMORY_DUMP_PATH, memory, WORK_BYTES) != 0)
        result = 1;
#ifdef AGC_TRIANGLE_SUBMIT
    if (write_blob(FRAME_SLOT_TARGET0_PATH, framebuffer,
                   FRAMEBUFFER_BYTES) != 0 ||
        write_blob(FRAME_SLOT_TARGET1_PATH,
                   framebuffer + FRAMEBUFFER_BYTES,
                   FRAMEBUFFER_BYTES) != 0)
        result = 1;
#endif
    printf(LOG_PREFIX " summary frames=%u slots=%u allocations=2"
           " work_bytes=%u framebuffer_bytes=%u high_water=%u/%u"
           " completed=%u,%u result=%d\n",
           frame, FRAME_SLOT_COUNT, WORK_BYTES, FRAMEBUFFER_POOL_BYTES,
           high_water, COMMAND_BYTES / 4u,
           completed_generation[0], completed_generation[1], result);
    return result;
}
#endif

int main(void)
{
#ifdef PS5_MULTIDRAW_BATCH
    if (runtime_batch_faulted)
        return 1;
#endif
#ifdef PS5_DRAW_BATCH_PROBE
    const unsigned batch_repeats = runtime_batch_probe_repeats();
    int64_t batch_wait_ns = 0;
#endif
#ifdef PS5_DRAW_PROFILE
    int64_t profile_ticks[10] = {0};
    unsigned profile_sleeps = 0;
#ifdef PS5_MULTIDRAW_BATCH
    const int profile_preparation = runtime_present_count >= 30 && runtime_batch_active;
#else
    const int profile_preparation = 0;
#endif
    const int profile_this_draw = runtime_present_count >= 30
#ifdef PS5_MULTIDRAW_BATCH
                                 && !runtime_batch_active
#endif
                                 ;
    PS5_PROFILE_MARK(0);
#endif
    const uint8_t *vs_header, *vs_code, *ps_header, *ps_code;
    const uint8_t *hs_header = NULL, *hs_code = NULL;
    size_t vs_header_size = 0, vs_code_size = 0;
    size_t ps_header_size = 0, ps_code_size = 0;
    size_t hs_header_size = 0, hs_code_size = 0;
    size_t hs_header_at = 0, hs_code_at = 0;
    size_t vs_header_at = WORK_BYTES, vs_code_at = 0;
    size_t ps_header_at = 0, ps_code_at = 0;
    size_t tess_descriptors_at = 0;
    size_t work_bytes = WORK_BYTES;
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    struct runtime_shader_pair *prepared = NULL;
#endif
#ifndef PS5_NATIVE_TITLE_RUNTIME
    void *agc_module = NULL, *driver_module = NULL, *video_module = NULL;
#endif
    agc_api_t agc = {0};
    video_api_t video = {0};
    int64_t work_start = -1, framebuffer_start = -1;
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    int64_t offscreen_start = -1;
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    int64_t depth_start = -1;
#endif
    uint8_t *memory = NULL;
    uint8_t *framebuffer = NULL;
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    uint8_t *offscreen = NULL;
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    uint8_t *depth = NULL;
#endif
    void *vertex = NULL, *pixel = NULL, *hull = NULL;
    int video_handle = -1, video_open_attempts = 0, buffers_registered = 0;
    int64_t render_marker = RENDER_MARKER;
#ifdef AGC_RUNTIME_PACKAGES
    volatile uint64_t *completion_marker = NULL;
#endif
    size_t framebuffer_pool_bytes = FRAMEBUFFER_POOL_BYTES;
#ifndef AGC_RUNTIME_PACKAGES
    int framebuffer_buffer_count = 2;
    int framebuffer_alias_second = 0;
#endif
    int unregister_rc = 0, close_rc = 0;
    int framebuffer_unmap_rc = 0, framebuffer_release_rc = 0;
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    int offscreen_unmap_rc = 0, offscreen_release_rc = 0;
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    int depth_unmap_rc = 0, depth_release_rc = 0;
#endif
    int work_unmap_rc = 0, work_release_rc = 0;
    int init_rc = -1, vertex_rc = -1, pixel_rc = -1, hull_rc = -1;
    int link_rc = -1;
    int tf_ring_set_rc = -1, tf_ring_get_rc = -1, tf_ring_unset_rc = -1;
    uintptr_t tf_ring_address = 0;
    uint32_t tf_ring_bytes = 0;
    int work_alloc_rc = -1, work_map_rc = -1;
    int framebuffer_alloc_rc = -1, framebuffer_map_rc = -1;
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    int offscreen_alloc_rc = -1, offscreen_map_rc = -1;
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    int depth_alloc_rc = -1, depth_map_rc = -1;
#ifdef AGC_TRIANGLE_SUBMIT
    int depth_dump_rc = -1;
    size_t depth_nonzero = 0;
    uint32_t depth_hash = 0;
#endif
#endif
    int result = 1;
    const char *failure_phase = "packages";
    int64_t direct_limit = -1;
    agc_register_t *cx;
    agc_register_t *sh;
    uint32_t cx_count = 0, sh_count = 0;
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
    agc_register_t *scanout_cx;
    uint32_t scanout_cx_count = 0;
    uint32_t barrier_begin_words = 0, barrier_end_words = 0;
#endif
    uint32_t *words;
    agc_command_buffer_t command = {0};
    agc_submit_description_t submit = {0};
    uint32_t draw_words, final_words;
#ifdef PS5_MULTIDRAW_BATCH
    uint32_t completion_offset = 0, upload_prefix_words = 0;
#endif
#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
    uint32_t draw_hash, final_hash;
#endif
    uint32_t bad_indirect_packet = UINT32_MAX;
    uint32_t bad_indirect_opcode = 0;
    uint32_t bad_indirect_count = 0;
    uint64_t bad_indirect_address = 0;
    int indirect_validation_rc = -1;

#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
#ifdef AGC_INVALID_FRAMEBUFFER_TEST
    const char *mode = "invalid-framebuffer";
#elif defined(AGC_TRIANGLE_NEGATIVE_PACKAGE)
    const char *mode = "negative-package";
#elif defined(AGC_TRIANGLE_TIMEOUT_TEST)
    const char *mode = "timeout-test";
#elif defined(AGC_TRIANGLE_SUBMIT)
    const char *mode = "submit";
#else
    const char *mode = "capture-only";
#endif
    printf(LOG_PREFIX " mode=%s package_vs=%u package_ps=%u\n", mode,
           VS_PACKAGE_LEN, PS_PACKAGE_LEN);
    fflush(stdout);
#endif

#ifdef AGC_INVALID_FRAMEBUFFER_TEST
    {
        native_color_target_spec_t spec = {
            (uintptr_t)UINT64_C(0x201600000), FRAMEBUFFER_BYTES,
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
            NATIVE_COLOR_FORMAT_RGBA8_UNORM, 1,
            NATIVE_COLOR_SWIZZLE_64KB_R_X, 0
        };
        int valid_rc = validate_color_target_spec(&spec);
        int format_rc, samples_rc, swizzle_rc, compressed_rc;
        int alignment_rc, allocation_rc, dimensions_rc;

        spec.format = 99;
        format_rc = validate_color_target_spec(&spec);
        spec.format = NATIVE_COLOR_FORMAT_RGBA8_UNORM;
        spec.samples = 4;
        samples_rc = validate_color_target_spec(&spec);
        spec.samples = 1;
        spec.swizzle = 0;
        swizzle_rc = validate_color_target_spec(&spec);
        spec.swizzle = NATIVE_COLOR_SWIZZLE_64KB_R_X;
        spec.compressed = 1;
        compressed_rc = validate_color_target_spec(&spec);
        spec.compressed = 0;
        spec.address += 0x1000;
        alignment_rc = validate_color_target_spec(&spec);
        spec.address -= 0x1000;
        spec.allocation_bytes = 0x800000;
        allocation_rc = validate_color_target_spec(&spec);
        spec.allocation_bytes = FRAMEBUFFER_BYTES;
        spec.width = 1024;
        dimensions_rc = validate_color_target_spec(&spec);
        printf(LOG_PREFIX " valid=%d invalid=format:%d,samples:%d,"
               "swizzle:%d,compressed:%d,alignment:%d,allocation:%d,"
               "dimensions:%d agc_opened=0 submitted=0\n",
               valid_rc, format_rc, samples_rc, swizzle_rc, compressed_rc,
               alignment_rc, allocation_rc, dimensions_rc);
        return valid_rc == 0 && format_rc == -3 && samples_rc == -4 &&
                       swizzle_rc == -5 && compressed_rc == -5 &&
                       alignment_rc == -6 && allocation_rc == -7 &&
                       dimensions_rc == -2
                   ? 0 : 1;
    }
#endif

    if (shader_sections(VS_PACKAGE, VS_PACKAGE_LEN,
                        &vs_header, &vs_header_size,
                        &vs_code, &vs_code_size) != 0 ||
        shader_sections(PS_PACKAGE, PS_PACKAGE_LEN,
                        &ps_header, &ps_header_size,
                        &ps_code, &ps_code_size) != 0) {
        printf(LOG_PREFIX " invalid embedded shader package\n");
        goto cleanup;
    }
    if (runtime_hs_package &&
        shader_sections(runtime_hs_package, runtime_hs_package_len,
                        &hs_header, &hs_header_size,
                        &hs_code, &hs_code_size) != 0) {
        printf(LOG_PREFIX " invalid hull shader package\n");
        goto cleanup;
    }
    if (runtime_hs_package) {
        tess_descriptors_at = (WORK_BYTES + TESS_OFFCHIP_BYTES +
                               TESS_FACTOR_BYTES + 0xfffu) & ~0xfffu;
        hs_header_at = tess_descriptors_at + 0x1000u;
        hs_code_at = (hs_header_at + hs_header_size + 0xfffu) & ~0xfffu;
        vs_header_at = (hs_code_at + hs_code_size + 0xfffu) & ~0xfffu;
    }
    vs_code_at = (vs_header_at + vs_header_size + 0xfffu) & ~0xfffu;
    ps_header_at = (vs_code_at + vs_code_size + 0xfffu) & ~0xfffu;
    ps_code_at = (ps_header_at + ps_header_size + 0xfffu) & ~0xfffu;
    work_bytes = (ps_code_at + ps_code_size + 0x3fffu) & ~0x3fffu;
    if (validate_shader_header(vs_header, vs_header_size, vs_code_size, 2) ||
        validate_shader_header(ps_header, ps_header_size, ps_code_size, 1)) {
        printf(LOG_PREFIX " embedded shader header validation failed\n");
        goto cleanup;
    }
    if (runtime_hs_package &&
        validate_shader_header(hs_header, hs_header_size, hs_code_size, 3)) {
        printf(LOG_PREFIX " hull shader header validation failed\n");
        goto cleanup;
    }
    failure_phase = "api";

#ifdef AGC_TRIANGLE_NEGATIVE_PACKAGE
    {
        uint8_t malformed[0x200];
        int negative_rc;
        if (vs_header_size > sizeof(malformed))
            goto cleanup;
        memcpy(malformed, vs_header, vs_header_size);
        malformed[0] ^= 1;
        negative_rc = validate_shader_header(
            malformed, vs_header_size, vs_code_size, 2);
        printf(LOG_PREFIX " negative validation=%d agc_opened=0 submitted=0\n",
               negative_rc);
        return negative_rc == -2 ? 0 : 1;
    }
#endif

#ifdef PS5_NATIVE_TITLE_RUNTIME
    if (load_apis(NULL, NULL, NULL, &agc, &video) != 0) {
        printf(LOG_PREFIX " linked API binding failed\n");
        goto cleanup;
    }
#else
    agc_module = dlopen("libSceAgc.sprx", RTLD_NOW | RTLD_LOCAL);
    if (agc_module)
        *(void **)(&agc.init) = dlsym(agc_module, "sceAgcInit");
    if (!agc_module || !agc.init) {
        printf(LOG_PREFIX " module/symbol load failed: %s\n", dlerror());
        goto cleanup;
    }
#endif

#ifdef AGC_RUNTIME_PACKAGES
    if (runtime_agc_initialized)
        init_rc = 0;
    else {
        init_rc = agc.init(8);
        if (init_rc == 0)
            runtime_agc_initialized = 1;
    }
#else
    init_rc = agc.init(8);
#endif
    if (init_rc == 0)
        direct_limit = sceKernelGetDirectMemorySize();
    /* libkernel_web reports zero in the payload-loader context; the proven
     * allocation contract accepts (search_start, search_end) == (0, 0). */
    if (init_rc != 0)
        goto receipt;
    failure_phase = "work";
#ifdef AGC_DEPTH_DEFAULTS_PROBE
    result = dump_depth_register_template(agc_module);
    goto cleanup;
#endif
#ifndef PS5_NATIVE_TITLE_RUNTIME
    driver_module = dlopen("libSceAgcDriver.sprx", RTLD_NOW | RTLD_LOCAL);
    video_module = dlopen("libSceVideoOut.sprx", RTLD_NOW | RTLD_LOCAL);
    if (!driver_module || !video_module ||
        load_apis(agc_module, driver_module, video_module, &agc, &video) != 0) {
        printf(LOG_PREFIX " module/symbol load failed: %s\n", dlerror());
        goto cleanup;
    }
#endif
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    if (!runtime_hs_package) {
        prepared = runtime_shader_pair_get(&agc, direct_limit,
            vs_header, vs_header_size, vs_code, vs_code_size,
            ps_header, ps_header_size, ps_code, ps_code_size);
        if (prepared)
            work_bytes = WORK_BYTES;
    }
#endif
#ifdef PS5_DRAW_GPU_TIMESTAMPS
    work_bytes += 0x4000;
#endif
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    if (runtime_work_take(work_bytes, (void **)&memory, &work_start)) {
        work_alloc_rc = work_map_rc = 0;
    } else
#endif
    {
    work_alloc_rc = sceKernelAllocateDirectMemory(
        0, direct_limit, work_bytes, 0x4000, DIRECT_MEMORY_TYPE, &work_start);
    if (work_alloc_rc == 0)
        work_map_rc = sceKernelMapDirectMemory(
            (void **)&memory, work_bytes, MAP_PROTECTION, 0, work_start,
            0x4000);
    }
    if (work_alloc_rc != 0 || work_map_rc != 0 || !memory)
        goto receipt;
    if (runtime_hs_package) {
        uintptr_t factor = (uintptr_t)(memory + WORK_BYTES +
                                       TESS_OFFCHIP_BYTES);

        tf_ring_set_rc = agc.set_tf_ring(factor, TESS_FACTOR_BYTES);
        if (tf_ring_set_rc == 0)
            tf_ring_get_rc = agc.get_tf_ring(&tf_ring_address,
                                             &tf_ring_bytes);
        if (tf_ring_set_rc != 0 || tf_ring_get_rc != 0 ||
            tf_ring_address != factor || tf_ring_bytes != TESS_FACTOR_BYTES)
            goto receipt;
    }
    failure_phase = "shaders";
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH) && \
    !defined(PS5_DRAW_GPU_TIMESTAMPS) && !defined(AGC_RUNTIME_DIAGNOSTICS)
    if (prepared)
        runtime_prepared_work_clear(memory);
    else
#endif
        memset(memory, 0, work_bytes);
#ifdef AGC_RUNTIME_PACKAGES
    completion_marker = (volatile uint64_t *)(memory + 0x6ff0);
#endif
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    if (prepared) {
        vertex = prepared->vertex;
        pixel = prepared->pixel;
        memcpy(memory + 0x5000, prepared->memory, 34 * sizeof(agc_register_t));
        memcpy(memory + 0x6000, prepared->memory + 0x1000, 3 * sizeof(agc_register_t));
        hull_rc = vertex_rc = pixel_rc = link_rc = 0;
    } else
#endif
    {
    memcpy(memory + vs_header_at, vs_header, vs_header_size);
    memcpy(memory + vs_code_at, vs_code, vs_code_size);
    memcpy(memory + ps_header_at, ps_header, ps_header_size);
    memcpy(memory + ps_code_at, ps_code, ps_code_size);
    if (runtime_hs_package) {
        uint32_t *descriptors = (uint32_t *)(memory + tess_descriptors_at);
        uintptr_t offchip = (uintptr_t)(memory + WORK_BYTES);
        uintptr_t factor = offchip + TESS_OFFCHIP_BYTES;

        memcpy(memory + hs_header_at, hs_header, hs_header_size);
        memcpy(memory + hs_code_at, hs_code, hs_code_size);
        descriptors[20] = (uint32_t)factor;
        descriptors[21] = (uint32_t)(factor >> 32);
        descriptors[22] = TESS_FACTOR_BYTES;
        descriptors[23] = UINT32_C(0x3004dfac);
        descriptors[24] = (uint32_t)offchip;
        descriptors[25] = (uint32_t)(offchip >> 32);
        descriptors[26] = TESS_OFFCHIP_BYTES;
        descriptors[27] = UINT32_C(0x3004dfac);
    }
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    RUNTIME_SHADER_LOCK();
#endif
    if (runtime_hs_package)
        hull_rc = agc.create_shader(&hull, memory + hs_header_at,
                                    memory + hs_code_at);
    else
        hull_rc = 0;
    vertex_rc = agc.create_shader(&vertex, memory + vs_header_at,
                                  memory + vs_code_at);
    if (hull_rc == 0 && vertex_rc == 0)
        pixel_rc = agc.create_shader(&pixel, memory + ps_header_at,
                                     memory + ps_code_at);
    if (pixel_rc == 0)
        link_rc = agc.link_shaders(memory + 0x5000, memory + 0x6000, NULL,
                                   vertex, pixel,
#ifdef AGC_RUNTIME_PACKAGES
                                   runtime_primitive_type);
#else
                                   4);
#endif
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    RUNTIME_SHADER_UNLOCK();
#endif
    }
    if (link_rc != 0)
        goto receipt;
    failure_phase = "framebuffer";
#ifdef AGC_RUNTIME_PACKAGES
    if (runtime_point_coord_input) {
        agc_register_t *input = (agc_register_t *)(memory + 0x5000);
        unsigned attribute = runtime_point_coord_input - 1u;

        if (input[attribute].offset != 0x0191 + attribute)
            goto receipt;
        /* Radeon GFX10: point coordinates select the generated S/T pair.
         * Clear the parameter-export offset and set PT_SPRITE_TEX. */
        input[attribute].value = (input[attribute].value & ~UINT32_C(0x3f)) |
                         UINT32_C(1) << 17;
    }
#endif

#if defined(AGC_DEPTH_TEST_VARIANT) || defined(AGC_TEXTURE_VARIANT) || \
    defined(AGC_UNIFORM_QUAD) || \
    defined(AGC_UNIFORM_SHIFTED_QUAD) || \
    defined(AGC_INDEXED_QUAD) || defined(AGC_TRIANGLE_INDEX_BUFFER) || \
    defined(AGC_TRIANGLE_VERTEX_BUFFER)
    {
#if defined(AGC_DEPTH_TEST_VARIANT)
        static const float vertices[63] = {
            -1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
             3.0f, -1.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            -1.0f,  3.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f,

            -0.65f, -0.65f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f,
             0.65f, -0.65f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f,
             0.00f,  0.65f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f,

            -0.20f, -0.55f,  0.5f, 1.0f, 0.0f, 0.0f, 1.0f,
             0.85f, -0.55f,  0.5f, 1.0f, 0.0f, 0.0f, 1.0f,
             0.33f,  0.55f,  0.5f, 1.0f, 0.0f, 0.0f, 1.0f,
        };
#elif defined(AGC_TEXTURE_VARIANT)
        static const float vertices[16] = {
            -0.75f, -0.75f, 0.0f, 1.0f,
             0.75f, -0.75f, 1.0f, 1.0f,
             0.75f,  0.75f, 1.0f, 0.0f,
            -0.75f,  0.75f, 0.0f, 0.0f,
        };
#elif defined(AGC_UNIFORM_QUAD) || defined(AGC_UNIFORM_SHIFTED_QUAD) || \
    defined(AGC_INDEXED_QUAD)
        static const float vertices[24] = {
            -0.75f, -0.75f, 1.0f, 0.0f, 0.0f, 1.0f,
             0.75f, -0.75f, 0.0f, 1.0f, 0.0f, 1.0f,
             0.75f,  0.75f, 0.0f, 0.0f, 1.0f, 1.0f,
            -0.75f,  0.75f, 1.0f, 1.0f, 0.0f, 1.0f,
        };
#else
        static const float vertices[18] = {
            -0.75f, -0.75f, 1.0f, 0.0f, 0.0f, 1.0f,
             0.75f, -0.75f, 0.0f, 1.0f, 0.0f, 1.0f,
             0.00f,  0.75f, 0.0f, 0.0f, 1.0f, 1.0f,
        };
#endif
        uint32_t *descriptor = (uint32_t *)(memory + 0x4000);
        void *vertex_data = memory + 0x4100;
        uintptr_t vertex_address = (uintptr_t)vertex_data;
        if ((uint32_t)(vertex_address >> 32) != GATE3_BIND_ADDRESS32_HI ||
            GATE3_BIND_VERTEX_BUFFER_USER_DWORD >=
                GATE3_BIND_USER_SGPR_COUNT)
            goto receipt;
        memcpy(vertex_data, vertices, sizeof(vertices));
#if defined(AGC_DEPTH_TEST_VARIANT)
        {
            uint32_t draw;
#ifdef AGC_CLEAR_TEST_VARIANT
            static const float clear_color[4] = {
                1.0f, 1.0f, 1.0f, 1.0f
            };
            for (draw = 0; draw < 3; ++draw)
                memcpy((float *)vertex_data + draw * 7u + 3u,
                       clear_color, sizeof(clear_color));
#endif
            for (draw = 0; draw < 3; ++draw) {
                uintptr_t draw_address =
                    vertex_address + draw * 3u * 7u * sizeof(float);
                descriptor[draw * 4u] = (uint32_t)draw_address;
                descriptor[draw * 4u + 1u] =
                    (uint32_t)(draw_address >> 32) | (28u << 16);
                descriptor[draw * 4u + 2u] = 3;
                descriptor[draw * 4u + 3u] = UINT32_C(0x5204);
            }
        }
#else
        descriptor[0] = (uint32_t)vertex_address;
#if defined(AGC_TEXTURE_VARIANT)
        descriptor[1] = (uint32_t)(vertex_address >> 32) | (16u << 16);
        descriptor[2] = 4;
#else
        descriptor[1] = (uint32_t)(vertex_address >> 32) | (24u << 16);
        descriptor[2] = sizeof(vertices) / (6u * sizeof(float));
#endif
        descriptor[3] = 0x5204;
#endif
#if defined(AGC_BLEND_VARIANT)
        {
            static const float flipped_vertices[16] = {
                -0.75f, -0.75f, 1.0f, 1.0f,
                 0.75f, -0.75f, 0.0f, 1.0f,
                 0.75f,  0.75f, 0.0f, 0.0f,
                -0.75f,  0.75f, 1.0f, 0.0f,
            };
            uint32_t *flipped_descriptor =
                (uint32_t *)(memory + 0x4400);
            void *flipped_vertex_data = memory + 0x4500;
            uintptr_t flipped_vertex_address =
                (uintptr_t)flipped_vertex_data;
            agc_register_t *blend_disabled =
                (agc_register_t *)(memory + 0x4700);
            agc_register_t *blend_second =
                (agc_register_t *)(memory + 0x4710);

            if ((uint32_t)(flipped_vertex_address >> 32) !=
                    GATE3_BIND_ADDRESS32_HI)
                goto receipt;
            memcpy(flipped_vertex_data, flipped_vertices,
                   sizeof(flipped_vertices));
            flipped_descriptor[0] = (uint32_t)flipped_vertex_address;
            flipped_descriptor[1] =
                (uint32_t)(flipped_vertex_address >> 32) | (16u << 16);
            flipped_descriptor[2] = 4;
            flipped_descriptor[3] = 0x5204;
            *blend_disabled =
                (agc_register_t){0x1e0, 0, UINT32_C(0x00000000)};
            *blend_second =
                (agc_register_t){0x1e0, 0, BLEND_SECOND_CONTROL};
        }
#endif
#if defined(AGC_DEPTH_TEST_VARIANT) || defined(AGC_TEXTURE_VARIANT) || \
    defined(AGC_UNIFORM_QUAD) || \
    defined(AGC_UNIFORM_SHIFTED_QUAD) || \
    defined(AGC_INDEXED_QUAD)
        {
            static const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
            memcpy(memory + 0x4200, indices, sizeof(indices));
        }
#elif defined(AGC_TRIANGLE_INDEX_BUFFER)
        {
            static const uint16_t indices[3] = {0, 1, 2};
            memcpy(memory + 0x4200, indices, sizeof(indices));
        }
#endif
#if defined(AGC_TEXTURE_VARIANT)
        {
            uint32_t *combined_descriptor = (uint32_t *)(memory + 0x4300);
            uint32_t *texels = (uint32_t *)(memory + 0xc000);
            uintptr_t texture_address = (uintptr_t)texels;
            uint32_t y;
            if ((uint32_t)(texture_address >> 32) !=
                    GATE3_TEXTURE_PS_ADDRESS32_HI ||
                GATE3_TEXTURE_PS_DESCRIPTOR_SET0_USER_DWORD >=
                    GATE3_TEXTURE_PS_USER_SGPR_COUNT ||
                GATE3_TEXTURE_PS_SET0_BINDING0_OFFSET != 0 ||
                GATE3_TEXTURE_PS_SET0_BINDING0_STRIDE != 48)
                goto receipt;
            for (y = 0; y < TEXTURE_HEIGHT; ++y) {
                uint32_t x;
                for (x = 0; x < TEXTURE_WIDTH; ++x) {
                    const unsigned right = x >= TEXTURE_WIDTH / 2u;
                    const unsigned bottom = y >= TEXTURE_HEIGHT / 2u;
#if defined(AGC_BLEND_VARIANT)
                    static const uint32_t colors_half_alpha[4] = {
                        UINT32_C(0x800000ff), UINT32_C(0x8000ff00),
                        UINT32_C(0x80ff0000), UINT32_C(0x80ffffff),
                    };
                    texels[y * TEXTURE_WIDTH + x] =
                        colors_half_alpha[(bottom << 1) | right];
#else
                    static const uint32_t colors_opaque[4] = {
                        UINT32_C(0xff0000ff), UINT32_C(0xff00ff00),
                        UINT32_C(0xffff0000), UINT32_C(0xffffffff),
                    };
                    texels[y * TEXTURE_WIDTH + x] =
                        colors_opaque[(bottom << 1) | right];
#endif
                }
            }
            combined_descriptor[0] = (uint32_t)(texture_address >> 8);
            combined_descriptor[1] =
                UINT32_C(0xc3800000) |
                (uint32_t)(texture_address >> 40);
            combined_descriptor[2] = UINT32_C(0x000fc00f);
            combined_descriptor[3] = UINT32_C(0x90000fac);
            combined_descriptor[4] = 0;
            combined_descriptor[5] = UINT32_C(0x00400000);
            combined_descriptor[6] = 0;
            combined_descriptor[7] = 0;
            combined_descriptor[8] = UINT32_C(0x00000092);
            combined_descriptor[9] = UINT32_C(0x00fff000);
#if defined(AGC_TEXTURE_NEAREST) || defined(AGC_BLEND_VARIANT)
            combined_descriptor[10] = UINT32_C(0x08000000);
#else
            combined_descriptor[10] = UINT32_C(0x09500000);
#endif
            combined_descriptor[11] = 0;
        }
#endif
#if defined(AGC_UNIFORM_QUAD) || defined(AGC_UNIFORM_SHIFTED_QUAD)
        {
#ifdef AGC_UNIFORM_SHIFTED_QUAD
            static const float params[8] = {
                0.5f, 1.0f, 0.25f, 1.0f, 0.1f, -0.1f, 0.0f, 0.0f,
            };
#else
            static const float params[8] = {
                1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            };
#endif
            uint32_t *uniform_descriptor = (uint32_t *)(memory + 0x4300);
            void *uniform_data = memory + 0x4400;
            uintptr_t uniform_address = (uintptr_t)uniform_data;
            if ((uint32_t)(uniform_address >> 32) !=
                    GATE3_UNIFORM_ADDRESS32_HI ||
                GATE3_UNIFORM_DESCRIPTOR_SET0_USER_DWORD >=
                    GATE3_UNIFORM_USER_SGPR_COUNT ||
                GATE3_UNIFORM_SET0_BINDING0_OFFSET != 0 ||
                GATE3_UNIFORM_SET0_BINDING0_STRIDE != 16)
                goto receipt;
            memcpy(uniform_data, params, sizeof(params));
            uniform_descriptor[0] = (uint32_t)uniform_address;
            uniform_descriptor[1] =
                (uint32_t)(uniform_address >> 32) | (16u << 16);
            uniform_descriptor[2] = 2;
            uniform_descriptor[3] = 0xfacu | (77u << 12);
        }
#endif
    }
#endif

#ifdef AGC_RUNTIME_PACKAGES
    framebuffer = runtime_framebuffer;
    framebuffer_alloc_rc = 0;
    framebuffer_map_rc = 0;
    if (!framebuffer || runtime_framebuffer_size < FRAMEBUFFER_BYTES ||
        ((uintptr_t)framebuffer & (FRAMEBUFFER_ALIGNMENT - 1u)))
        goto receipt;
    if (runtime_framebuffer_size < FRAMEBUFFER_POOL_BYTES)
        framebuffer_pool_bytes = FRAMEBUFFER_BYTES;
#else
    framebuffer_alloc_rc = sceKernelAllocateDirectMemory(
        0, direct_limit, FRAMEBUFFER_POOL_BYTES, FRAMEBUFFER_ALIGNMENT,
        DIRECT_MEMORY_TYPE, &framebuffer_start);
    if (framebuffer_alloc_rc == 0)
        framebuffer_map_rc = sceKernelMapDirectMemory(
            (void **)&framebuffer, FRAMEBUFFER_POOL_BYTES, MAP_PROTECTION, 0,
            framebuffer_start, FRAMEBUFFER_ALIGNMENT);
    if (framebuffer_alloc_rc != 0 || framebuffer_map_rc != 0 || !framebuffer)
        goto receipt;
#endif
#ifndef AGC_RUNTIME_PACKAGES
    memset(framebuffer, 0, FRAMEBUFFER_POOL_BYTES);
#endif
    PS5_PROFILE_MARK(1);
#ifdef PS5_GPU_PRESENT_BATCH
    /* The first queued draw only initializes a new scanout pool. Gallium's
     * CPU clear/transfer/blit writers flush the ranges they modify. A query or
     * fence splitting a GPU batch does not dirty this GPU-written pool. Keep
     * the conservative flush for new pools and nonbatched scanout writers.
     * Offscreen draws do not access this display pool. */
    if ((!runtime_batch_active
#ifdef PS5_RUNTIME_WRITES_SCANOUT
         && PS5_RUNTIME_WRITES_SCANOUT()
#endif
         ) || !runtime_video_registered ||
        framebuffer != runtime_video_framebuffer ||
        framebuffer_pool_bytes > runtime_video_framebuffer_size) {
        const int registered_pool = runtime_video_registered &&
            framebuffer == runtime_video_framebuffer &&
            framebuffer_pool_bytes <= runtime_video_framebuffer_size;
        if (!registered_pool || runtime_scanout_flush_needed)
            flush_gpu_data(framebuffer, framebuffer_pool_bytes);
        if (registered_pool)
            runtime_scanout_flush_needed = 0;
    }
#else
    flush_gpu_data(framebuffer, framebuffer_pool_bytes);
#endif
    PS5_PROFILE_MARK(2);
#ifdef AGC_RUNTIME_PACKAGES
#ifdef PS5_ASYNC_NATIVE_PREP
    if (!runtime_async_output) {
#endif
    if (runtime_video_acquire(&video, framebuffer, framebuffer_pool_bytes,
                              &video_open_attempts) != 0)
        goto receipt;
    if (runtime_video_prepare_draw() != 0)
        goto receipt;
#ifdef PS5_ASYNC_NATIVE_PREP
    }
#endif
    video_handle = runtime_video_handle;
    buffers_registered = runtime_video_registered;
#ifdef PS5_ASYNC_NATIVE_PREP
    render_marker = runtime_async_output ? runtime_async_output->marker :
                                          runtime_next_render_marker();
#else
    render_marker = runtime_next_render_marker();
#endif
    failure_phase = "state";
    PS5_PROFILE_MARK(3);
#else
    for (int attempt = 1; attempt <= 3; ++attempt) {
        video_open_attempts = attempt;
        video_handle = video.open(0xff, 0, 0, NULL);
        if (video_handle >= 0)
            break;
        if (attempt != 3)
            sceKernelUsleep(UINT32_C(500000));
    }
    if (video_handle < 0 || video.set_flip_rate(video_handle, 0) != 0)
        goto receipt;
    {
        video_buffer_t buffers[2] = {
            {framebuffer, NULL, NULL, NULL},
            {framebuffer + (framebuffer_alias_second ? 0 : FRAMEBUFFER_BYTES),
             NULL, NULL, NULL}
        };
        video_attribute_t attribute = {{0}};
        video.set_attribute2(&attribute, VIDEO_OUT_PIXEL_FORMAT, 0,
                             DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0, 0);
        if (video.register_buffers2(video_handle, 0, 0, buffers,
                                    framebuffer_buffer_count,
                                    &attribute, 0, NULL) != 0)
            goto receipt;
        buffers_registered = 1;
    }
#endif
#if defined(AGC_RUNTIME_PACKAGES) && defined(AGC_RUNTIME_DIAGNOSTICS)
    printf(LOG_PREFIX " framebuffer-pool bytes=%zu buffers=2 persistent=1\n",
           framebuffer_pool_bytes);
#endif

#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    offscreen_alloc_rc = sceKernelAllocateDirectMemory(
        0, direct_limit, OFFSCREEN_BYTES, FRAMEBUFFER_ALIGNMENT,
        DIRECT_MEMORY_TYPE, &offscreen_start);
    if (offscreen_alloc_rc == 0)
        offscreen_map_rc = sceKernelMapDirectMemory(
            (void **)&offscreen, OFFSCREEN_BYTES, MAP_PROTECTION, 0,
            offscreen_start, FRAMEBUFFER_ALIGNMENT);
    if (offscreen_alloc_rc != 0 || offscreen_map_rc != 0 || !offscreen)
        goto receipt;
    memset(offscreen, 0, OFFSCREEN_BYTES);
    flush_gpu_data(offscreen, OFFSCREEN_BYTES);
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
    {
        uint32_t *target_descriptor = (uint32_t *)(memory + 0x4400);
        uintptr_t target_address = (uintptr_t)offscreen;

        target_descriptor[0] = (uint32_t)(target_address >> 8);
        target_descriptor[1] =
            UINT32_C(0xc3800000) | (uint32_t)(target_address >> 40);
        target_descriptor[2] = UINT32_C(0x010dc1df);
        target_descriptor[3] = UINT32_C(0x91b00fac);
        target_descriptor[4] = 0;
        target_descriptor[5] = UINT32_C(0x00400000);
        target_descriptor[6] = 0;
        target_descriptor[7] = 0;
        target_descriptor[8] = UINT32_C(0x00000092);
        target_descriptor[9] = UINT32_C(0x00fff000);
        target_descriptor[10] = UINT32_C(0x08000000);
        target_descriptor[11] = 0;
    }
#endif
#endif

#ifdef AGC_DEPTH_TEST_VARIANT
    depth_alloc_rc = sceKernelAllocateDirectMemory(
        0, direct_limit, DEPTH_BYTES, DEPTH_ALIGNMENT,
        DIRECT_MEMORY_TYPE, &depth_start);
    if (depth_alloc_rc == 0)
        depth_map_rc = sceKernelMapDirectMemory(
            (void **)&depth, DEPTH_BYTES, MAP_PROTECTION, 0,
            depth_start, DEPTH_ALIGNMENT);
    if (depth_alloc_rc != 0 || depth_map_rc != 0 || !depth)
        goto receipt;
    memset(depth, 0, DEPTH_BYTES);
    flush_gpu_data(depth, DEPTH_BYTES);
#endif

#ifdef AGC_FRAME_SLOTS_VARIANT
    result = run_frame_slot_test(&agc, &video, video_handle, memory,
                                 framebuffer, vertex, pixel);
    goto receipt;
#endif

    cx = (agc_register_t *)(memory + 0x7000);
    sh = (agc_register_t *)(memory + 0x7800);
    words = (uint32_t *)(memory + 0x8000);
    if (append_target_state(cx, &cx_count, agc.get_defaults(),
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
                            offscreen) != 0 ||
#else
                            framebuffer) != 0 ||
#endif
        append_shader_state(memory, vertex, pixel, cx, &cx_count,
                            sh, &sh_count) != 0 ||
        (runtime_hs_package &&
         append_hull_shader_state(hull, sh, &sh_count) != 0))
        goto receipt;
    if (runtime_hs_package) {
        uint32_t stages = last_register_value(cx, cx_count, 0x02d5);

        /* GFX10.3 tessellation preamble defaults from Mesa ac_cmdbuf.c.
         * CLEAR_STATE does not supply a usable tessellation distribution. */
        if ((stages & UINT32_C(0x211f)) != UINT32_C(0x210d))
            goto receipt;
        cx[cx_count++] = (agc_register_t){0x0286, 0,
                                         UINT32_C(0x42800000)};
        cx[cx_count++] = (agc_register_t){0x0287, 0, 0};
        cx[cx_count++] = (agc_register_t){0x02d4, 0,
                                         UINT32_C(0xd8181e0c)};
        cx[cx_count++] = (agc_register_t){0x02d6, 0,
                                         runtime_ls_hs_config};
        cx[cx_count++] = (agc_register_t){0x02db, 0, runtime_tf_param};
    }
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
    scanout_cx = (agc_register_t *)(memory + 0x4800);
    if (append_target_state(scanout_cx, &scanout_cx_count,
                            agc.get_defaults(), framebuffer) != 0)
        goto receipt;
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    append_depth_target_state(cx, &cx_count, depth, NULL);
#endif
#ifdef AGC_RUNTIME_PACKAGES
    if (runtime_depth_buffer &&
        runtime_depth_buffer_size >= DEPTH_BYTES)
        append_depth_target_state(cx, &cx_count, runtime_depth_buffer,
                                  runtime_stencil_buffer);
    if (getenv("PSBC_DEBUG_IO"))
        printf("[ps5-linkage] vs-out=%08x ps-in=%08x inputs=%08x/%08x/%08x/%08x\n",
               last_register_value(cx, cx_count, 0x1b1),
               last_register_value(cx, cx_count, 0x1b6),
               last_register_value(cx, cx_count, 0x191),
               last_register_value(cx, cx_count, 0x192),
               last_register_value(cx, cx_count, 0x193),
               last_register_value(cx, cx_count, 0x194));
#endif

    command.bottom = words;
    command.top = words + COMMAND_BYTES / sizeof(*words);
    command.up = words;
    command.down = command.top;
    command.callback = (uintptr_t)command_out_of_space;
#ifndef AGC_RUNTIME_PACKAGES
    agc.wait_rendering(&command.up, agc.wait_size(), 0,
                       (uint32_t)video_handle, 0);
#endif
#ifdef PS5_DRAW_GPU_TIMESTAMPS
    if (!agc.release_mem(&command, 40, 0, 0, 0,
                         memory + work_bytes - 0x4000, 3, 0, 0, 0, 3, 0))
        goto receipt;
#endif
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH)
    if (!runtime_acquire_cpu_uploads(&command, memory, work_bytes))
        goto receipt;
#ifndef PS5_DRAW_GPU_TIMESTAMPS
    upload_prefix_words = (uint32_t)(command.up - words);
#endif
#endif
    agc.set_cx(&command, cx, cx_count);
#if defined(AGC_BLEND_VARIANT)
    agc.set_cx(&command, memory + 0x4700, 1);
#endif
    if (!set_linkage_uc_state(&agc, &command, memory))
        goto receipt;
    agc.set_sh(&command, sh, sh_count);
#ifdef AGC_RUNTIME_PACKAGES
    if (runtime_hs_package) {
        uintptr_t descriptors = (uintptr_t)(memory + tess_descriptors_at);
        uint32_t pointer[2] = {
            (uint32_t)descriptors, (uint32_t)(descriptors >> 32)
        };

        agc.set_sh_direct(&command, 0x102, pointer, 2);
        agc.set_sh_direct(&command, 0x082, pointer, 2);
        if (runtime_hull_user_data_count)
            agc.set_sh_direct(&command, 0x10c, runtime_hull_user_data,
                              runtime_hull_user_data_count);
    }
    if (runtime_vertex_user_data_count)
        agc.set_sh_direct(&command, 0x8c, runtime_vertex_user_data,
                          runtime_vertex_user_data_count);
    if (runtime_pixel_user_data_count)
        agc.set_sh_direct(&command, 0x0c, runtime_pixel_user_data,
                          runtime_pixel_user_data_count);
    /* A color-only pass must disable the preceding pass's depth/stencil
     * writes, including when both passes share a deferred command stream. */
    {
        agc_register_t *depth_state =
            (agc_register_t *)(memory + 0x4700);
        depth_state[0] = (agc_register_t){
            0x0200, 0, runtime_depth_buffer ? runtime_depth_control : 0
        };
        if (runtime_stencil_buffer) {
            depth_state[1] = (agc_register_t){
                0x010b, 0, runtime_stencil_control
            };
            depth_state[2] = (agc_register_t){
                0x010c, 0, runtime_stencil_refmask
            };
            depth_state[3] = (agc_register_t){
                0x010d, 0, runtime_stencil_refmask_bf
            };
        }
        agc.set_cx(&command, depth_state,
                   runtime_stencil_buffer ? 4 : 1);
    }
    {
        agc_register_t *graphics_state =
            (agc_register_t *)(memory + RUNTIME_GRAPHICS_STATE_OFFSET);
        uint32_t graphics_count = 0;
        uint32_t index;

        graphics_state[graphics_count++] = (agc_register_t){
            0x01e0, 0, runtime_blend_control
        };
        graphics_state[graphics_count++] = (agc_register_t){
            0x008e, 0, runtime_target_mask
        };
        if (runtime_color_control_valid)
            graphics_state[graphics_count++] = (agc_register_t){
                0x0202, 0, runtime_color_control
            };
        for (index = 0; index < 4; ++index)
            graphics_state[graphics_count++] = (agc_register_t){
                (uint16_t)(0x0105u + index), 0,
                runtime_blend_color[index]
            };
        for (unsigned viewport = 0; viewport < runtime_viewport_count;
             ++viewport) {
            for (index = 0; index < 6; ++index)
                graphics_state[graphics_count++] = (agc_register_t){
                    (uint16_t)(0x010fu + viewport * 6u + index), 0,
                    runtime_viewport[viewport][index]
                };
            graphics_state[graphics_count++] = (agc_register_t){
                (uint16_t)(0x00b4u + viewport * 2u), 0,
                runtime_viewport[viewport][6]
            };
            graphics_state[graphics_count++] = (agc_register_t){
                (uint16_t)(0x00b5u + viewport * 2u), 0,
                runtime_viewport[viewport][7]
            };
            graphics_state[graphics_count++] = (agc_register_t){
                (uint16_t)(0x0094u + viewport * 2u), 0,
                runtime_scissor[viewport][0]
            };
            graphics_state[graphics_count++] = (agc_register_t){
                (uint16_t)(0x0095u + viewport * 2u), 0,
                runtime_scissor[viewport][1]
            };
        }
        graphics_state[graphics_count++] = (agc_register_t){
            0x0090, 0, runtime_generic_scissor[0]
        };
        graphics_state[graphics_count++] = (agc_register_t){
            0x0091, 0, runtime_generic_scissor[1]
        };
        if (runtime_rasterizer_valid)
            graphics_state[graphics_count++] = (agc_register_t){
                0x0205, 0, runtime_rasterizer_control
            };
        if (runtime_point_line_valid) {
            for (index = 0; index < 3; ++index)
                graphics_state[graphics_count++] = (agc_register_t){
                    (uint16_t)(0x0280u + index), 0,
                    runtime_point_line[index]
                };
        }
        {
            /* Guard band as in radeonsi: clip only where screen coordinates
             * would leave the rasterizer range, and keep wide points/lines
             * whose centre lies outside the viewport (discard band grows by
             * the largest half point size / line width). */
            const float range = 16383.0f;
            const float point_half =
                (float)(runtime_point_line[1] >> 16) / 16.0f;
            const float line_half =
                (float)(runtime_point_line[2] & 0xffffu) / 16.0f;
            const float pixels = point_half > line_half ? point_half
                                                        : line_half;
            float clip_x = range, clip_y = range;
            float disc_x = range, disc_y = range;

            for (unsigned viewport = 0; viewport < runtime_viewport_count;
                 ++viewport) {
                const float sx = fabsf(bits_to_float(runtime_viewport[viewport][0]));
                const float tx = fabsf(bits_to_float(runtime_viewport[viewport][1]));
                const float sy = fabsf(bits_to_float(runtime_viewport[viewport][2]));
                const float ty = fabsf(bits_to_float(runtime_viewport[viewport][3]));
                float gx = 1.0f, gy = 1.0f;

                if (sx > 0.0f && tx < range)
                    gx = (range - tx) / sx;
                if (sy > 0.0f && ty < range)
                    gy = (range - ty) / sy;
                if (gx < 1.0f)
                    gx = 1.0f;
                if (gy < 1.0f)
                    gy = 1.0f;
                clip_x = gx < clip_x ? gx : clip_x;
                clip_y = gy < clip_y ? gy : clip_y;
                gx = sx > 0.0f ? 1.0f + pixels / sx : 1.0f;
                gy = sy > 0.0f ? 1.0f + pixels / sy : 1.0f;
                disc_x = gx < disc_x ? gx : disc_x;
                disc_y = gy < disc_y ? gy : disc_y;
            }
            disc_x = disc_x < clip_x ? disc_x : clip_x;
            disc_y = disc_y < clip_y ? disc_y : clip_y;
            graphics_state[graphics_count++] =
                (agc_register_t){0x02fa, 0, float_bits(clip_y)};
            graphics_state[graphics_count++] =
                (agc_register_t){0x02fb, 0, float_bits(disc_y)};
            graphics_state[graphics_count++] =
                (agc_register_t){0x02fc, 0, float_bits(clip_x)};
            graphics_state[graphics_count++] =
                (agc_register_t){0x02fd, 0, float_bits(disc_x)};
        }
        if (runtime_interp_control_valid)
            graphics_state[graphics_count++] = (agc_register_t){
                0x01b5, 0, runtime_interp_control
            };
        if (runtime_polygon_offset_valid) {
            for (index = 0; index < 6; ++index)
                graphics_state[graphics_count++] = (agc_register_t){
                    (uint16_t)(0x02deu + index), 0,
                    runtime_polygon_offset[index]
                };
        }
        agc.set_cx(&command, graphics_state, graphics_count);
    }
#endif
#if defined(AGC_DEPTH_TEST_VARIANT)
    {
        agc_register_t *clear_state =
            (agc_register_t *)(memory + 0x4700);
        agc_register_t *draw_state =
            (agc_register_t *)(memory + 0x4710);
        uint32_t user_data[GATE3_BIND_USER_SGPR_COUNT];

        clear_state[0] = (agc_register_t){
            0x008e, 0,
#ifdef AGC_CLEAR_TEST_VARIANT
            UINT32_C(0x0000000f)
#else
            0
#endif
        };
        clear_state[1] =
            (agc_register_t){0x0200, 0, UINT32_C(0x00000076)};
        draw_state[0] =
            (agc_register_t){0x008e, 0, UINT32_C(0x0000000f)};
        draw_state[1] =
            (agc_register_t){0x0200, 0, UINT32_C(0x00000016)};
        memset(user_data, 0, sizeof(user_data));
        user_data[GATE3_BIND_VERTEX_BUFFER_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4000);
        agc.set_sh_direct(&command, 0x8c, user_data,
                          GATE3_BIND_USER_SGPR_COUNT);
        agc.set_cx(&command, clear_state, 2);
        agc.draw_auto(&command, 3, 2);
        agc.set_cx(&command, draw_state, 2);
        user_data[GATE3_BIND_VERTEX_BUFFER_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4010);
        agc.set_sh_direct(&command, 0x8c, user_data,
                          GATE3_BIND_USER_SGPR_COUNT);
        agc.draw_auto(&command, 3, 2);
        user_data[GATE3_BIND_VERTEX_BUFFER_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4020);
        agc.set_sh_direct(&command, 0x8c, user_data,
                          GATE3_BIND_USER_SGPR_COUNT);
        agc.draw_auto(&command, 3, 2);
    }
#elif defined(AGC_TEXTURE_VARIANT) || defined(AGC_UNIFORM_QUAD) || \
    defined(AGC_UNIFORM_SHIFTED_QUAD) || \
    defined(AGC_INDEXED_QUAD) || defined(AGC_TRIANGLE_INDEX_BUFFER) || \
    defined(AGC_TRIANGLE_VERTEX_BUFFER)
    {
        uint32_t user_data[GATE3_BIND_USER_SGPR_COUNT];
        memset(user_data, 0, sizeof(user_data));
        user_data[GATE3_BIND_VERTEX_BUFFER_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4000);
#if defined(AGC_UNIFORM_QUAD) || defined(AGC_UNIFORM_SHIFTED_QUAD)
        user_data[GATE3_UNIFORM_DESCRIPTOR_SET0_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4300);
#endif
        agc.set_sh_direct(&command, 0x8c, user_data,
                          GATE3_BIND_USER_SGPR_COUNT);
    }
#endif
#if defined(AGC_TEXTURE_VARIANT)
    {
        uint32_t pixel_user_data[GATE3_TEXTURE_PS_USER_SGPR_COUNT];
        memset(pixel_user_data, 0, sizeof(pixel_user_data));
        pixel_user_data[GATE3_TEXTURE_PS_DESCRIPTOR_SET0_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4300);
        agc.set_sh_direct(&command, 0x0c, pixel_user_data,
                          GATE3_TEXTURE_PS_USER_SGPR_COUNT);
    }
#endif
#if defined(AGC_DEPTH_TEST_VARIANT)
    /* The clear, near, and far draws were emitted with their state above. */
#elif defined(AGC_TEXTURE_VARIANT) || defined(AGC_UNIFORM_QUAD) || \
    defined(AGC_UNIFORM_SHIFTED_QUAD) || \
    defined(AGC_INDEXED_QUAD)
    agc.set_index_size(&command, AGC_INDEX_SIZE_16, 0);
    agc.set_index_buffer(&command, memory + 0x4200);
    agc.set_index_count(&command, 6);
    agc.draw_index(&command, 6, memory + 0x4200, 0);
#if defined(AGC_RENDER_TO_TEXTURE_VARIANT)
    {
        uint32_t pixel_user_data[GATE3_TEXTURE_PS_USER_SGPR_COUNT];

        /* SlimGL sub_E56F0(a3=0x811): uncompressed CB data flush plus
         * GLV/GL1 invalidation.  Event 45 is
         * FLUSH_AND_INV_CB_DATA_TS and release GCR control is 0x0c. */
        barrier_begin_words = (uint32_t)(command.up - words);
        if (!agc.release_mem(&command, 45, 12, 1, 0, NULL, 0, 0,
                             0, 1, 0, 0))
            goto receipt;
        barrier_end_words = (uint32_t)(command.up - words);
        agc.set_cx(&command, scanout_cx, scanout_cx_count);
        memset(pixel_user_data, 0, sizeof(pixel_user_data));
        pixel_user_data[GATE3_TEXTURE_PS_DESCRIPTOR_SET0_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4400);
        agc.set_sh_direct(&command, 0x0c, pixel_user_data,
                          GATE3_TEXTURE_PS_USER_SGPR_COUNT);
        agc.draw_index(&command, 6, memory + 0x4200, 0);
    }
#endif
#if defined(AGC_BLEND_VARIANT)
    {
        uint32_t user_data[GATE3_BIND_USER_SGPR_COUNT];
        memset(user_data, 0, sizeof(user_data));
        user_data[GATE3_BIND_VERTEX_BUFFER_USER_DWORD] =
            (uint32_t)(uintptr_t)(memory + 0x4400);
        agc.set_cx(&command, memory + 0x4710, 1);
        agc.set_sh_direct(&command, 0x8c, user_data,
                          GATE3_BIND_USER_SGPR_COUNT);
        agc.draw_index(&command, 6, memory + 0x4200, 0);
    }
#endif
#elif defined(AGC_TRIANGLE_INDEX_BUFFER)
    agc.set_index_size(&command, AGC_INDEX_SIZE_16, 0);
    agc.set_index_buffer(&command, memory + 0x4200);
    agc.set_index_count(&command, 3);
    agc.draw_index(&command, 3, memory + 0x4200, 0);
#elif defined(AGC_RUNTIME_PACKAGES)
#ifdef PS5_DRAW_GPU_TIMESTAMPS
    if (!agc.release_mem(&command, 40, 0, 0, 0,
                         memory + work_bytes - 0x4000 + 8, 3, 0, 0, 0, 3, 0))
        goto receipt;
#endif
#ifdef PS5_DRAW_BATCH_PROBE
    for (unsigned repetition = 0; repetition < batch_repeats; ++repetition)
#endif
    {
    if (runtime_index_buffer) {
        uint8_t agc_index_size = runtime_index_size == sizeof(uint32_t) ?
                                     AGC_INDEX_SIZE_32 : AGC_INDEX_SIZE_16;

        agc.set_index_size(&command, agc_index_size, 0);
        agc.set_index_buffer(&command, (void *)runtime_index_buffer);
        agc.set_index_count(&command, runtime_index_count);
        agc.draw_index(&command, runtime_index_count,
                       (void *)runtime_index_buffer, 0);
    } else {
        agc.draw_auto(&command, runtime_draw_count, 2);
    }
    }
#else
    agc.draw_auto(&command, 3, 2);
#endif
#ifdef PS5_DRAW_GPU_TIMESTAMPS
    if (!agc.release_mem(&command, 40, 0, 0, 0,
                         memory + work_bytes - 0x4000 + 16, 3, 0, 0, 0, 3, 0))
        goto receipt;
#endif
    draw_words = (uint32_t)(command.up - words);
#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
    draw_hash = fnv1a32(words, draw_words * sizeof(*words));
#endif
    if (out_of_space || command.up < command.bottom ||
        command.up > command.top || draw_words == 0)
        goto receipt;
    failure_phase = "release";

#ifdef AGC_TRIANGLE_SUBMIT
#ifdef AGC_RUNTIME_PACKAGES
#if defined(PS5_MULTIDRAW_BATCH) && defined(PS5_AGC_POST_DRAW_BARRIER_WORDS) && !defined(PS5_DRAW_GPU_TIMESTAMPS)
    if (runtime_batch_active && runtime_batch_fixed_framebuffer &&
        PS5_AGC_POST_DRAW_BARRIER_WORDS < draw_words)
        completion_offset = draw_words - PS5_AGC_POST_DRAW_BARRIER_WORDS;
#endif
    if (!agc.release_mem(&command, 45, 12, 1, 0, NULL, 0, 0,
                         0, 1, 0, 0))
        goto receipt;
#ifdef PS5_MULTIDRAW_BATCH
    if (!completion_offset)
        completion_offset = (uint32_t)(command.up - words);
#endif
    /* Completion is what CPU clears and readbacks of the attachments wait for.
     * Event 43 is FLUSH_AND_INV_DB_DATA_TS: without it depth and stencil written
     * by these draws can stay in the DB cache and replace a later CPU clear. */
    if (!agc.release_mem(&command, 43, 12, 1, 0, NULL, 0, 0,
                         0, 1, 0, 0))
        goto receipt;
    if (!runtime_release_completion(&agc, &command, completion_marker,
                                     (uint32_t)render_marker))
        goto receipt;
#else
    agc.set_flip(&command, (uint32_t)video_handle, 0, 1, render_marker);
#endif
#endif
    final_words = (uint32_t)(command.up - words);
    failure_phase = "indirect";
#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
    final_hash = fnv1a32(words, final_words * sizeof(*words));
#endif
    submit.words = words;
    submit.word_count = final_words;
#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
    printf(LOG_PREFIX " state cx=%u/%08" PRIx32
           " sh=%u/%08" PRIx32 " draw=%u/%08" PRIx32
           " final=%u/%08" PRIx32 " oos=%u\n",
           cx_count, fnv1a32(cx, cx_count * sizeof(*cx)),
           sh_count, fnv1a32(sh, sh_count * sizeof(*sh)),
           draw_words, draw_hash, final_words, final_hash, out_of_space);
    printf(LOG_PREFIX " link stages=%04x:%08" PRIx32
           " ge_cntl=%04x:%08" PRIx32
           " user_vgpr=%04x:%08" PRIx32
           " primitive=%04x:%08" PRIx32
           " pc_alloc=%u/%04x:%08" PRIx32 "\n",
           ((agc_register_t *)(memory + 0x5000))[32].offset,
           ((agc_register_t *)(memory + 0x5000))[32].value,
           ((agc_register_t *)(memory + 0x6000))[0].offset,
           ((agc_register_t *)(memory + 0x6000))[0].value,
           ((agc_register_t *)(memory + 0x6000))[1].offset,
           ((agc_register_t *)(memory + 0x6000))[1].value,
           ((agc_register_t *)(memory + 0x6000))[2].offset,
           ((agc_register_t *)(memory + 0x6000))[2].value,
           runtime_ngg_ge_pc_alloc_valid,
           ((agc_register_t *)(memory + 0x6000))[3].offset,
           ((agc_register_t *)(memory + 0x6000))[3].value);
#ifdef AGC_RUNTIME_PACKAGES
    printf(LOG_PREFIX " link-input0=%04x:%08" PRIx32
           " ps-input=%08" PRIx32 "/%08" PRIx32
           "/%08" PRIx32 " baryc=%08" PRIx32
           " raster=%u/%08" PRIx32
           " interp=%u/%08" PRIx32 " pcoord=%u"
           " depth=%u/%08" PRIx32 "/%08" PRIx32 "\n",
           ((agc_register_t *)(memory + 0x5000))[0].offset,
           ((agc_register_t *)(memory + 0x5000))[0].value,
           last_register_value(cx, cx_count, UINT16_C(0x1b3)),
           last_register_value(cx, cx_count, UINT16_C(0x1b4)),
           last_register_value(cx, cx_count, UINT16_C(0x1b6)),
           last_register_value(cx, cx_count, UINT16_C(0x1b8)),
           runtime_rasterizer_valid, runtime_rasterizer_control,
           runtime_interp_control_valid, runtime_interp_control,
           runtime_point_coord_input, runtime_depth_samples,
           last_register_value(cx, cx_count, UINT16_C(0x0010)),
           last_register_value(cx, cx_count, UINT16_C(0x0011)));
#endif
#ifdef AGC_RUNTIME_PACKAGES
    printf(LOG_PREFIX " draw primitive=%" PRIu32
           " count=%u indexed=%u index_size=%u agc_index_size=%u"
           " index=%016" PRIxPTR "\n",
           runtime_primitive_type, runtime_draw_count,
           runtime_index_buffer != NULL, runtime_index_size,
           runtime_index_size == sizeof(uint32_t) ?
               AGC_INDEX_SIZE_32 : AGC_INDEX_SIZE_16,
           (uintptr_t)runtime_index_buffer);
#endif
    printf(LOG_PREFIX " capture_rc=%08" PRIx32 "\n",
           (uint32_t)capture_commands(words, final_words, memory));
#endif
    indirect_validation_rc = validate_indirect_register_tables(
        words, final_words, memory, WORK_BYTES,
        &bad_indirect_packet, &bad_indirect_opcode,
        &bad_indirect_address, &bad_indirect_count);
#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
    printf(LOG_PREFIX " indirect_validation=%d packet=%" PRIu32
           " opcode=%02" PRIx32 " address=%016" PRIx64
           " count=%" PRIu32 "\n",
           indirect_validation_rc, bad_indirect_packet,
           bad_indirect_opcode, bad_indirect_address,
           bad_indirect_count);
#endif
    if (indirect_validation_rc != 0) {
#if defined(AGC_RUNTIME_PACKAGES) && !defined(AGC_RUNTIME_DIAGNOSTICS)
        printf(LOG_PREFIX " indirect_validation=%d packet=%" PRIu32
               " opcode=%02" PRIx32 " address=%016" PRIx64
               " count=%" PRIu32 "\n",
               indirect_validation_rc, bad_indirect_packet,
               bad_indirect_opcode, bad_indirect_address,
               bad_indirect_count);
#endif
        goto receipt;
    }
    failure_phase = "submit";
#if defined(AGC_TEXTURE_VARIANT)
    {
        const uint32_t *texture_descriptor =
            (const uint32_t *)(memory + 0x4300);
        printf(LOG_PREFIX " texture_desc="
               "%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32
               " sampler=%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32
               " texels=%08" PRIx32 "\n",
               texture_descriptor[0], texture_descriptor[1],
               texture_descriptor[2], texture_descriptor[3],
               texture_descriptor[4], texture_descriptor[5],
               texture_descriptor[6], texture_descriptor[7],
               texture_descriptor[8], texture_descriptor[9],
               texture_descriptor[10], texture_descriptor[11],
               fnv1a32(memory + 0xc000,
                       TEXTURE_WIDTH * TEXTURE_HEIGHT * 4u));
#if defined(AGC_RENDER_TO_TEXTURE_VARIANT)
        {
            const uint32_t *target_descriptor =
                (const uint32_t *)(memory + 0x4400);
            printf(LOG_PREFIX " sampled_target_desc="
                   "%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
                   ",%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
                   ",%08" PRIx32 ",%08" PRIx32
                   " sampler=%08" PRIx32 ",%08" PRIx32
                   ",%08" PRIx32 ",%08" PRIx32
                   " barrier=%u-%u event=45 gcr=0000000c"
                   " scanout_cx=%u/%08" PRIx32 "\n",
                   target_descriptor[0], target_descriptor[1],
                   target_descriptor[2], target_descriptor[3],
                   target_descriptor[4], target_descriptor[5],
                   target_descriptor[6], target_descriptor[7],
                   target_descriptor[8], target_descriptor[9],
                   target_descriptor[10], target_descriptor[11],
                   barrier_begin_words, barrier_end_words,
                   scanout_cx_count,
                   fnv1a32(scanout_cx,
                           scanout_cx_count * sizeof(*scanout_cx)));
        }
#endif
#if defined(AGC_BLEND_VARIANT)
        printf(LOG_PREFIX " blend=%08" PRIx32 "->%08" PRIx32
               " vb=%08" PRIx32 "->%08" PRIx32 "\n",
               ((const agc_register_t *)(memory + 0x4700))->value,
               ((const agc_register_t *)(memory + 0x4710))->value,
               (uint32_t)(uintptr_t)(memory + 0x4000),
               (uint32_t)(uintptr_t)(memory + 0x4400));
#endif
#if defined(AGC_RUNTIME_PACKAGES)
        printf(LOG_PREFIX " graphics blend=%08" PRIx32
               " mask=%08" PRIx32
               " color_control=%u/%08" PRIx32
               " color=%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32
               " viewport=%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32
               " z=%08" PRIx32 ",%08" PRIx32
               " scissor=%08" PRIx32 ",%08" PRIx32
               " raster=%u/%08" PRIx32
               " point_line=%u/%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32
               " interp=%u/%08" PRIx32
               " polygon_offset=%u/%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
               ",%08" PRIx32 "\n",
               runtime_blend_control, runtime_target_mask,
               runtime_color_control_valid, runtime_color_control,
               runtime_blend_color[0], runtime_blend_color[1],
               runtime_blend_color[2], runtime_blend_color[3],
               runtime_viewport[0][0], runtime_viewport[0][1],
               runtime_viewport[0][2], runtime_viewport[0][3],
               runtime_viewport[0][4], runtime_viewport[0][5],
               runtime_viewport[0][6], runtime_viewport[0][7],
               runtime_scissor[0][0], runtime_scissor[0][1],
               runtime_rasterizer_valid, runtime_rasterizer_control,
               runtime_point_line_valid, runtime_point_line[0],
               runtime_point_line[1], runtime_point_line[2],
               runtime_interp_control_valid, runtime_interp_control,
               runtime_polygon_offset_valid, runtime_polygon_offset[0],
               runtime_polygon_offset[1], runtime_polygon_offset[2],
               runtime_polygon_offset[3], runtime_polygon_offset[4],
               runtime_polygon_offset[5]);
#endif
    }
#endif
#if !defined(AGC_RUNTIME_PACKAGES) || defined(AGC_RUNTIME_DIAGNOSTICS)
    fflush(stdout);
#endif

#ifdef AGC_TRIANGLE_SUBMIT
    PS5_PROFILE_MARK(4);
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_MULTIDRAW_BATCH) && \
    !defined(PS5_DRAW_GPU_TIMESTAMPS) && !defined(AGC_RUNTIME_DIAGNOSTICS)
    if (prepared) {
        if (runtime_prepared_work_publish(memory, &command) != 0)
            goto receipt;
    } else
#endif
        flush_gpu_data(memory, work_bytes);
    PS5_PROFILE_MARK(5);
#ifdef PS5_MULTIDRAW_BATCH
    if (runtime_batch_active) {
        /* Finish all metadata before transferring ownership to the batch. */
        struct runtime_batch_entry entry = {
            .submit = submit, .memory = memory, .direct = work_start,
            .bytes = work_bytes, .marker = completion_marker,
            .expected = (uint32_t)render_marker,
        };
        if (command.down >= command.up && command.down <= command.top) {
            entry.completion_offset = completion_offset;
            entry.upload_prefix_words = upload_prefix_words;
            entry.command_capacity = (uint32_t)(command.down - words);
        }
        if (runtime_batch_queue(&agc, &entry) != 0)
            goto receipt;
        memory = NULL;
        work_start = -1;
        result = 0;
        goto receipt;
    }
#endif
    {
        uint64_t status[16] = {0};
        unsigned waits;
#ifdef PS5_DRAW_BATCH_PROBE
        const int64_t batch_start = os_time_get_nano();
#endif
#ifdef PS5_DRAW_GPU_TIMESTAMPS
        ((volatile uint64_t *)(memory + work_bytes - 0x4000))[8] = os_time_get_nano();
#endif
        int submit_rc = agc.submit(&submit);
        PS5_PROFILE_MARK(6);
#ifdef PS5_FRAME_SUSPEND
        int suspend_rc = submit_rc == 0 ? runtime_defer_suspend(agc.suspend_point) : -1;
#else
        int suspend_rc = submit_rc == 0 ? agc.suspend_point() : -1;
#endif
        PS5_PROFILE_MARK(7);
#ifdef AGC_RUNTIME_PACKAGES
        waits = 2000;
        if (submit_rc == 0) {
            int64_t spin_deadline = 0;
            for (waits = 0; waits < 2000;) {
#ifdef PS5_DRAW_GPU_TIMESTAMPS
                runtime_draw_gpu_observe(memory, work_bytes);
#endif
                flush_gpu_data((const void *)completion_marker,
                               sizeof(*completion_marker));
                if (*completion_marker == (uint32_t)render_marker)
                    break;
                waits += runtime_completion_pause(&spin_deadline);
            }
        }
        status[3] = *completion_marker;
#ifdef PS5_NATIVE_TITLE_RUNTIME
        runtime_require_retirement(submit_rc == 0 && suspend_rc == 0 && waits < 2000);
#endif
#ifdef PS5_DRAW_BATCH_PROBE
        batch_wait_ns = os_time_get_nano() - batch_start;
#endif
#ifdef PS5_DRAW_GPU_TIMESTAMPS
        runtime_draw_gpu_timing(memory, work_bytes);
#endif
        PS5_PROFILE_MARK(8);
#ifdef PS5_DRAW_PROFILE
        profile_sleeps = waits;
#endif
#else
        waits = submit_rc == 0
                    ? wait_for_flip_marker(&video, video_handle,
                                           (uint64_t)render_marker, 120,
                                           status)
                    : 120;
#endif
#if defined(AGC_RUNTIME_PACKAGES) && !defined(AGC_RUNTIME_DIAGNOSTICS)
        /* Gallium performs resource readback and pixel validation itself.
         * Avoid multi-megabyte command/framebuffer dumps on every GL draw. */
        result = submit_rc == 0 && suspend_rc == 0 && waits < 2000 ? 0 : 1;
        if (result)
            printf(LOG_PREFIX " submit=%08" PRIx32
                   " suspend=%08" PRIx32 " waits=%u\n",
                   (uint32_t)submit_rc, (uint32_t)suspend_rc, waits);
#else
        flush_gpu_data(
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
            framebuffer,
#elif defined(AGC_OFFSCREEN_COLOR_VARIANT)
            offscreen,
#else
            framebuffer,
#endif
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
            OFFSCREEN_BYTES);
#else
            (size_t)DISPLAY_WIDTH * DISPLAY_HEIGHT * 4u);
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
        flush_gpu_data(depth, DEPTH_BYTES);
        {
            const uint32_t *depth_words = (const uint32_t *)depth;
            size_t word;
            FILE *depth_file;
            for (word = 0; word < DEPTH_BYTES / sizeof(uint32_t); ++word)
                depth_nonzero += depth_words[word] != 0;
            depth_hash = fnv1a32(depth, DEPTH_BYTES);
            depth_file = fopen(DEPTH_DUMP_PATH, "wb");
            if (depth_file) {
                size_t written = fwrite(depth, 1, DEPTH_BYTES, depth_file);
                int depth_close_rc = fclose(depth_file);
                depth_dump_rc =
                    written == DEPTH_BYTES && depth_close_rc == 0 ? 0 : -1;
            }
            printf(LOG_PREFIX " depth_nonzero=%zu depth_hash=%08" PRIx32
                   " depth_dump=%08" PRIx32 "\n",
                   depth_nonzero, depth_hash, (uint32_t)depth_dump_rc);
        }
#endif
        {
            const uint32_t *pixels = (const uint32_t *)
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
                framebuffer;
#elif defined(AGC_OFFSCREEN_COLOR_VARIANT)
                offscreen;
#else
                framebuffer;
#endif
            size_t nonzero = 0;
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
            size_t scanout_nonzero = 0;
#endif
#if defined(AGC_RENDER_TO_TEXTURE_VARIANT) || \
    defined(AGC_CLEAR_TEST_VARIANT)
            uint32_t readback[5] = {0};
            int readback_rc = 0;
#endif
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
            size_t source_nonzero = 0;
            uint32_t source_hash = 0;
            int source_dump_rc = -1;
            int readback_oob_rc = 0;
#endif
            uint32_t min_x = DISPLAY_WIDTH, min_y = DISPLAY_HEIGHT;
            uint32_t max_x = 0, max_y = 0;
            uint32_t first_values[4] = {0};
            uint32_t first_indices[4] = {0};
            uint32_t min_value = UINT32_MAX;
            uint32_t max_value = 0;
            unsigned first_count = 0;
            uint32_t y;
            uint32_t center = pixels[(DISPLAY_HEIGHT / 2u) * DISPLAY_WIDTH +
                                     DISPLAY_WIDTH / 2u];
            uint32_t corner = pixels[16u * DISPLAY_WIDTH + 16u];
            uint32_t image_hash = fnv1a32(
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
                framebuffer,
#elif defined(AGC_OFFSCREEN_COLOR_VARIANT)
                offscreen,
#else
                framebuffer,
#endif
                (size_t)DISPLAY_WIDTH * DISPLAY_HEIGHT * 4u);
            int dump_rc = -1;
            for (y = 0; y < DISPLAY_HEIGHT; ++y) {
                uint32_t x;
                for (x = 0; x < DISPLAY_WIDTH; ++x) {
                    uint32_t value = pixels[(size_t)y * DISPLAY_WIDTH + x];
                    if (!value)
                        continue;
                    if (first_count < 4) {
                        first_indices[first_count] = y * DISPLAY_WIDTH + x;
                        first_values[first_count++] = value;
                    }
                    ++nonzero;
                    if (value < min_value) min_value = value;
                    if (value > max_value) max_value = value;
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
            flush_gpu_data(framebuffer, FRAMEBUFFER_POOL_BYTES);
            {
                const uint32_t *scanout = (const uint32_t *)framebuffer;
                size_t scanout_pixel;
                for (scanout_pixel = 0;
                     scanout_pixel < FRAMEBUFFER_POOL_BYTES / sizeof(uint32_t);
                     ++scanout_pixel)
                    scanout_nonzero += scanout[scanout_pixel] != 0;
            }
#endif
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
            flush_gpu_data(offscreen, OFFSCREEN_BYTES);
            {
                const uint32_t *source = (const uint32_t *)offscreen;
                size_t source_pixel;
                FILE *source_file;

                for (source_pixel = 0;
                     source_pixel < FRAMEBUFFER_BYTES / sizeof(uint32_t);
                     ++source_pixel)
                    source_nonzero += source[source_pixel] != 0;
                source_hash = fnv1a32(offscreen, FRAMEBUFFER_BYTES);
                source_file = fopen(SOURCE_DUMP_PATH, "wb");
                if (source_file) {
                    size_t written = fwrite(offscreen, 1,
                                            FRAMEBUFFER_BYTES, source_file);
                    int source_close_rc = fclose(source_file);
                    source_dump_rc =
                        written == FRAMEBUFFER_BYTES &&
                        source_close_rc == 0 ? 0 : -1;
                }
            }
            readback_rc |= read_tiled_rgba8_region(
                framebuffer, FRAMEBUFFER_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 100, 100, 1, 1, &readback[0], 1);
            readback_rc |= read_tiled_rgba8_region(
                framebuffer, FRAMEBUFFER_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 600, 400, 1, 1, &readback[1], 1);
            readback_rc |= read_tiled_rgba8_region(
                framebuffer, FRAMEBUFFER_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 1300, 400, 1, 1, &readback[2], 1);
            readback_rc |= read_tiled_rgba8_region(
                framebuffer, FRAMEBUFFER_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 600, 700, 1, 1, &readback[3], 1);
            readback_rc |= read_tiled_rgba8_region(
                framebuffer, FRAMEBUFFER_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 1300, 700, 1, 1, &readback[4], 1);
            readback_oob_rc = read_tiled_rgba8_region(
                framebuffer, FRAMEBUFFER_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, DISPLAY_WIDTH - 1u, 0, 2, 1,
                &readback[0], 2);
#endif
#ifdef AGC_CLEAR_TEST_VARIANT
            readback_rc |= read_tiled_rgba8_region(
                offscreen, OFFSCREEN_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 100, 100, 1, 1, &readback[0], 1);
            readback_rc |= read_tiled_rgba8_region(
                offscreen, OFFSCREEN_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, DISPLAY_WIDTH - 1u, DISPLAY_HEIGHT - 1u,
                1, 1, &readback[1], 1);
            readback_rc |= read_tiled_rgba8_region(
                offscreen, OFFSCREEN_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 960, 540, 1, 1, &readback[2], 1);
            readback_rc |= read_tiled_rgba8_region(
                offscreen, OFFSCREEN_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 600, 700, 1, 1, &readback[3], 1);
            readback_rc |= read_tiled_rgba8_region(
                offscreen, OFFSCREEN_BYTES, DISPLAY_WIDTH,
                DISPLAY_HEIGHT, 1500, 700, 1, 1, &readback[4], 1);
#endif
            {
                FILE *target = fopen(
                    TARGET_DUMP_PATH, "wb");
                if (target) {
                    size_t written = fwrite(
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
                                            framebuffer,
#else
                                            offscreen,
#endif
#else
                                            framebuffer,
#endif
                                            1,
                                            FRAMEBUFFER_BYTES, target);
                    int close_rc = fclose(target);
                    dump_rc = written == FRAMEBUFFER_BYTES && close_rc == 0
                                  ? 0 : -1;
                }
            }
            printf(LOG_PREFIX " submit=%08" PRIx32
                   " suspend=%08" PRIx32 " waits=%u marker=%016" PRIx64
                   " center=%08" PRIx32 " corner=%08" PRIx32
                   " image=%08" PRIx32 "\n",
                   (uint32_t)submit_rc, (uint32_t)suspend_rc, waits,
                   status[3], center, corner, image_hash);
            printf(LOG_PREFIX " nonzero=%zu bounds=%u,%u-%u,%u"
                   " values=%08" PRIx32 "-%08" PRIx32
                   " first=%u:%08" PRIx32 ",%u:%08" PRIx32
                   ",%u:%08" PRIx32 ",%u:%08" PRIx32
                   " dump=%08" PRIx32 "\n",
                   nonzero, min_x, min_y, max_x, max_y,
                   min_value, max_value,
                   first_indices[0], first_values[0],
                   first_indices[1], first_values[1],
                   first_indices[2], first_values[2],
                   first_indices[3], first_values[3], (uint32_t)dump_rc);
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
            printf(LOG_PREFIX " offscreen=%p/%u scanout_nonzero=%zu"
                   " target_base=%08" PRIx32 ":%02" PRIx32 "\n",
                   offscreen, OFFSCREEN_BYTES, scanout_nonzero,
                   cx[0].value, cx[10].value & 0xffu);
#endif
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
            printf(LOG_PREFIX " render_to_texture source_nonzero=%zu"
                   " source_hash=%08" PRIx32 " source_dump=%08" PRIx32
                   " sampled_base=%08" PRIx32 ":%02" PRIx32 "\n",
                   source_nonzero, source_hash, (uint32_t)source_dump_rc,
                   ((const uint32_t *)(memory + 0x4400))[0],
                   ((const uint32_t *)(memory + 0x4400))[1] & 0xffu);
            printf(LOG_PREFIX " readback rc=%d oob=%d"
                   " pixels=%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
                   ",%08" PRIx32 ",%08" PRIx32 "\n",
                   readback_rc, readback_oob_rc, readback[0], readback[1],
                   readback[2], readback[3], readback[4]);
#endif
#ifdef AGC_CLEAR_TEST_VARIANT
            printf(LOG_PREFIX " readback rc=%d"
                   " pixels=%08" PRIx32 ",%08" PRIx32 ",%08" PRIx32
                   ",%08" PRIx32 ",%08" PRIx32 "\n",
                   readback_rc, readback[0], readback[1], readback[2],
                   readback[3], readback[4]);
            result = submit_rc == 0 && suspend_rc == 0 && waits < 120 &&
                      readback_rc == 0 &&
                      readback[0] == UINT32_C(0xffffffff) &&
                      readback[1] == UINT32_C(0xffffffff) &&
                      readback[2] == UINT32_C(0xff00ff00) &&
                      readback[3] == UINT32_C(0xff00ff00) &&
                      readback[4] == UINT32_C(0xff0000ff)
                         ? 0 : 1;
#else
#ifdef AGC_RUNTIME_PACKAGES
            /* Gallium owns draw semantics and validates pixels separately.
             * A legal draw may cover the old corner sentinel or produce no
             * fragments, so the reusable bridge reports execution only. */
            result = submit_rc == 0 && suspend_rc == 0 && waits < 2000
                         ? 0 : 1;
#else
#ifdef AGC_EXPECT_EMPTY_TARGET
            result = submit_rc == 0 && suspend_rc == 0 && waits < 120 &&
                      nonzero == 0 && corner == 0 ? 0 : 1;
#else
            result = submit_rc == 0 && suspend_rc == 0 && waits < 120 &&
                      nonzero != 0 && corner == 0 ? 0 : 1;
#endif
#endif
#endif
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
#ifndef AGC_RENDER_TO_TEXTURE_VARIANT
            if (scanout_nonzero != 0)
                result = 1;
#endif
#endif
#ifdef AGC_RENDER_TO_TEXTURE_VARIANT
            if (source_nonzero == 0 || source_dump_rc != 0 ||
                readback_rc != 0 || readback_oob_rc != -1 ||
                readback[0] != 0 ||
                readback[1] != UINT32_C(0xff0000ff) ||
                readback[2] != UINT32_C(0xff00ff00) ||
                readback[3] != UINT32_C(0xffff0000) ||
                readback[4] != UINT32_C(0xffffffff))
                result = 1;
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
            if (depth_nonzero == 0 || depth_dump_rc != 0)
                result = 1;
#endif
#ifdef AGC_TRIANGLE_EXPECTS_VARIATION
            if (min_value == max_value)
                result = 1;
#endif
#if (defined(AGC_TEXTURE_VARIANT) && \
     !defined(AGC_RENDER_TO_TEXTURE_VARIANT)) || \
    defined(AGC_UNIFORM_QUAD) || \
    defined(AGC_UNIFORM_SHIFTED_QUAD) || \
    defined(AGC_INDEXED_QUAD)
#if defined(AGC_TEXTURE_VIEWPORT_HALF) || \
    defined(AGC_TEXTURE_SCISSOR_HALF)
            if (nonzero != 720u * 810u)
                result = 1;
#else
            if (nonzero != (DISPLAY_WIDTH * 3u / 4u) *
                               (DISPLAY_HEIGHT * 3u / 4u))
                result = 1;
#endif
#endif
        }
#endif
#ifdef AGC_TRIANGLE_TIMEOUT_TEST
        {
            uint64_t timeout_status[16];
            unsigned timeout_waits = wait_for_flip_marker(
                &video, video_handle, UINT64_C(0x474c33ff), 3,
                timeout_status);
            printf(LOG_PREFIX " timeout_test waits=%u limit=3"
                   " observed=%016" PRIx64 " result=%s\n",
                   timeout_waits, timeout_status[3],
                   timeout_waits == 3 ? "bounded" : "unexpected-match");
            if (timeout_waits != 3)
                result = 1;
        }
#endif
#ifndef AGC_RUNTIME_PACKAGES
        for (waits = 0; waits < 120; ++waits)
            video.wait_vblank(video_handle);
#endif
    }
#else
    result = 0;
#endif

receipt:
#if defined(AGC_RUNTIME_PACKAGES) && !defined(AGC_RUNTIME_DIAGNOSTICS)
    if (result)
#endif
    {
    printf(LOG_PREFIX " phase=%s init=%08" PRIx32 " vertex=%08" PRIx32
           " hull=%08" PRIx32 " pixel=%08" PRIx32 " link=%08" PRIx32
           " video=%08" PRIx32 " video_attempts=%d registered=%d result=%d\n",
           failure_phase, (uint32_t)init_rc, (uint32_t)vertex_rc, (uint32_t)hull_rc,
           (uint32_t)pixel_rc, (uint32_t)link_rc, (uint32_t)video_handle,
           video_open_attempts, buffers_registered, result);
    printf(LOG_PREFIX " dmem_limit=%016" PRIx64
           " work=%08" PRIx32 "/%08" PRIx32 "/%016" PRIx64
           " framebuffer=%08" PRIx32 "/%08" PRIx32
           "/%016" PRIx64 " mapped=%p/%p\n",
           (uint64_t)direct_limit, (uint32_t)work_alloc_rc,
           (uint32_t)work_map_rc, (uint64_t)work_start,
           (uint32_t)framebuffer_alloc_rc, (uint32_t)framebuffer_map_rc,
           (uint64_t)framebuffer_start, memory, framebuffer);
    if (runtime_hs_package)
        printf(LOG_PREFIX " tf_ring=%08" PRIx32 "/%08" PRIx32
               " address=%016" PRIxPTR " bytes=%u\n",
               (uint32_t)tf_ring_set_rc, (uint32_t)tf_ring_get_rc,
               tf_ring_address, tf_ring_bytes);
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    printf(LOG_PREFIX " offscreen=%08" PRIx32 "/%08" PRIx32
           "/%016" PRIx64 " mapped=%p bytes=%u\n",
           (uint32_t)offscreen_alloc_rc, (uint32_t)offscreen_map_rc,
           (uint64_t)offscreen_start, offscreen, OFFSCREEN_BYTES);
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    printf(LOG_PREFIX " depth=%08" PRIx32 "/%08" PRIx32
           "/%016" PRIx64 " mapped=%p bytes=%u\n",
           (uint32_t)depth_alloc_rc, (uint32_t)depth_map_rc,
           (uint64_t)depth_start, depth, DEPTH_BYTES);
#endif
    fflush(stdout);
    }

cleanup:
    if (tf_ring_set_rc == 0)
        tf_ring_unset_rc = agc.set_tf_ring(0, 0);
#ifdef AGC_DEPTH_TEST_VARIANT
    if (depth)
        depth_unmap_rc = munmap(depth, DEPTH_BYTES);
    if (depth_start >= 0)
        depth_release_rc = sceKernelReleaseDirectMemory(
            depth_start, DEPTH_BYTES);
#endif
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    if (offscreen)
        offscreen_unmap_rc = munmap(offscreen, OFFSCREEN_BYTES);
    if (offscreen_start >= 0)
        offscreen_release_rc = sceKernelReleaseDirectMemory(
            offscreen_start, OFFSCREEN_BYTES);
#endif
#ifndef AGC_RUNTIME_PACKAGES
    if (buffers_registered)
        unregister_rc = video.unregister_buffers(video_handle, 0);
    if (video_handle >= 0 && video.close)
        close_rc = video.close(video_handle);
#endif
#ifndef AGC_RUNTIME_PACKAGES
    if (framebuffer)
        framebuffer_unmap_rc = munmap(framebuffer, FRAMEBUFFER_POOL_BYTES);
#endif
    if (framebuffer_start >= 0)
        framebuffer_release_rc = sceKernelReleaseDirectMemory(
            framebuffer_start, FRAMEBUFFER_POOL_BYTES);
    if (memory)
        work_unmap_rc = munmap(memory, work_bytes);
    if (work_start >= 0 && work_unmap_rc == 0)
        work_release_rc = sceKernelReleaseDirectMemory(work_start, work_bytes);
#if defined(AGC_RUNTIME_PACKAGES) && !defined(AGC_RUNTIME_DIAGNOSTICS)
    if (work_unmap_rc != 0 || work_release_rc != 0)
#endif
    {
    printf(LOG_PREFIX " cleanup unregister=%08" PRIx32
           " close=%08" PRIx32 " framebuffer=%08" PRIx32
           "/%08" PRIx32 " work=%08" PRIx32 "/%08" PRIx32
           " tf_ring=%08" PRIx32 "\n",
           (uint32_t)unregister_rc, (uint32_t)close_rc,
           (uint32_t)framebuffer_unmap_rc,
           (uint32_t)framebuffer_release_rc, (uint32_t)work_unmap_rc,
           (uint32_t)work_release_rc, (uint32_t)tf_ring_unset_rc);
#ifdef AGC_OFFSCREEN_COLOR_VARIANT
    printf(LOG_PREFIX " cleanup offscreen=%08" PRIx32 "/%08" PRIx32 "\n",
           (uint32_t)offscreen_unmap_rc, (uint32_t)offscreen_release_rc);
#endif
#ifdef AGC_DEPTH_TEST_VARIANT
    printf(LOG_PREFIX " cleanup depth=%08" PRIx32 "/%08" PRIx32 "\n",
           (uint32_t)depth_unmap_rc, (uint32_t)depth_release_rc);
#endif
    fflush(stdout);
    }
#ifndef PS5_NATIVE_TITLE_RUNTIME
    if (video_module)
        dlclose(video_module);
    if (driver_module)
        dlclose(driver_module);
    if (agc_module)
        dlclose(agc_module);
#endif
#ifdef PS5_DRAW_PROFILE
    PS5_PROFILE_MARK(9);
    if (video_open_attempts > 0)
        runtime_prepare_startup_report(profile_ticks, video_open_attempts,
                                       result || work_unmap_rc || work_release_rc);
    if (profile_preparation)
        runtime_prepare_profile_record(profile_ticks,
                                       result || work_unmap_rc || work_release_rc);
    if (profile_this_draw)
        runtime_profile_record(profile_ticks, profile_sleeps,
                               result || work_unmap_rc || work_release_rc);
    if (runtime_prepare_profile_calls + runtime_profile_calls >= 10000u)
        runtime_profile_report();
#endif
#ifdef PS5_DRAW_BATCH_PROBE
    printf("[ps5-batch-probe] repeats=%u wait_ns=%" PRId64 " result=%d\n",
           batch_repeats, batch_wait_ns,
           result || work_unmap_rc || work_release_rc || batch_wait_ns <= 0);
#endif
    return result || work_unmap_rc || work_release_rc;
}
