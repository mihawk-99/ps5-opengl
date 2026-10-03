// PS5 OpenGL - OpenGL implementation for PlayStation 5.
// Copyright (C) 2026 BlackBearReloaded
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ps5_screen.h"

#include <inttypes.h>
#include <stdbool.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(__SSE2__)
#include <emmintrin.h>
#endif

#ifdef PS5_RUNTIME_QUIET
static int
ps5_runtime_printf(const char *format, ...)
{
   (void)format;
   return 0;
}
#define printf ps5_runtime_printf
#endif

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "nir/tgsi_to_nir.h"
#include "amd/common/amdgfxregs.h"
#include "amd/common/ac_descriptors.h"
#include "amd/common/ac_formats.h"
#include "amd/common/ac_shader_util.h"
#include "amd/common/gfx10_format_table.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "psbc_compile.h"
#include "ps5_agc_package.h"
#include "util/ralloc.h"
#include "util/cache_ops.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/os_time.h"
#include "util/u_sampler.h"
#include "util/u_sample_positions.h"
#include "util/simple_mtx.h"
#include "util/u_draw.h"
#include "util/u_blitter.h"
#include "util/u_framebuffer.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_memset.h"
#include "util/u_prim.h"
#include "util/u_prim_restart.h"
#include "indices/u_primconvert.h"
#include "util/u_surface.h"
#include "util/u_upload_mgr.h"
#include "mesa/main/sse_minmax.h"
#include "util/u_cpu_detect.h"
#include "ps5_shader_cache.h"

#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
__attribute__((weak)) void ps5_opengl_register_gl_consumer(void) {}
#endif

/* This driver passes Mesa-owned NIR directly into the standalone backend. */
_Static_assert(sizeof(nir_instr_type) == 1, "NIR enums must be packed");
_Static_assert(sizeof(nir_intrinsic_op) == 4, "unexpected NIR intrinsic enum size");
_Static_assert(offsetof(nir_intrinsic_instr, intrinsic) == 56,
               "PSBC/Mesa NIR layout mismatch");
_Static_assert(MESA_SHADER_VERTEX == 0 && MESA_SHADER_TESS_CTRL == 1 &&
               MESA_SHADER_TESS_EVAL == 2, "pre-raster buffer bank indices");
_Static_assert(PIPE_FUNC_NEVER == 0 && PIPE_FUNC_LESS == 1 &&
               PIPE_FUNC_EQUAL == 2 && PIPE_FUNC_LEQUAL == 3 &&
               PIPE_FUNC_GREATER == 4 && PIPE_FUNC_NOTEQUAL == 5 &&
               PIPE_FUNC_GEQUAL == 6 && PIPE_FUNC_ALWAYS == 7,
               "Gallium and GFX10 compare encodings must match");
_Static_assert(PIPE_STENCIL_OP_KEEP == 0 && PIPE_STENCIL_OP_ZERO == 1 &&
               PIPE_STENCIL_OP_REPLACE == 2 && PIPE_STENCIL_OP_INCR == 3 &&
               PIPE_STENCIL_OP_DECR == 4 &&
               PIPE_STENCIL_OP_INCR_WRAP == 5 &&
               PIPE_STENCIL_OP_DECR_WRAP == 6 &&
               PIPE_STENCIL_OP_INVERT == 7,
               "unexpected Gallium stencil operation order");
_Static_assert(PIPE_LOGICOP_CLEAR == 0 && PIPE_LOGICOP_COPY == 12 &&
               PIPE_LOGICOP_SET == 15,
               "unexpected Gallium logic-operation order");

#define PS5_DIRECT_ALIGNMENT 0x4000u
#ifndef PS5_GPU_CLEAR_MIN_PIXELS
#define PS5_GPU_CLEAR_MIN_PIXELS 16384u
#endif
#ifndef PS5_GPU_BLIT_MIN_PIXELS
#define PS5_GPU_BLIT_MIN_PIXELS (64u * 64u)
#endif
#define PS5_RENDER_TARGET_BYTES PS5_SCANOUT_BYTES
#ifndef PS5_RENDER_ARENA_BYTES
#define PS5_RENDER_ARENA_BYTES 0u
#endif
#ifndef PS5_RENDER_POOL_BYTES
#define PS5_RENDER_POOL_BYTES (PS5_SCANOUT_POOL_BYTES + PS5_RENDER_ARENA_BYTES)
#endif
#define PS5_RENDER_ARENA_OFFSET (2u * PS5_RENDER_TARGET_BYTES)
#define PS5_RENDER_ARENA_SLOT_BYTES PS5_DIRECT_ALIGNMENT
#define PS5_RENDER_ARENA_SLOT_COUNT \
   ((PS5_RENDER_POOL_BYTES - PS5_RENDER_ARENA_OFFSET) / \
    PS5_RENDER_ARENA_SLOT_BYTES)
#define PS5_RENDER_ARENA_BITMAP_WORDS \
   ((PS5_RENDER_ARENA_SLOT_COUNT + 63u) / 64u)

struct ps5_screen {
   struct pipe_screen base;
   struct pipe_resource *render_pool;
   uint64_t render_arena_bitmap[PS5_RENDER_ARENA_BITMAP_WORDS];
   simple_mtx_t resource_mutex;
   simple_mtx_t submit_mutex;
};

#ifdef PS5_DEFERRED_DRAW_BATCH
#ifndef PS5_MULTIDRAW_BATCH
#error "Deferred draws require the validated multi-draw runtime"
#endif
/* Runtime setters/queue are process-global, including across pipe_screens. */
static simple_mtx_t ps5_deferred_mutex = SIMPLE_MTX_INITIALIZER;
static void ps5_draw_batch_flush_locked(void);
static void ps5_draw_batch_retire_locked(bool wait);
static uint64_t ps5_draw_batch_fence_submit(void);
static bool ps5_draw_batch_fence_finish(uint64_t sequence, uint64_t timeout);
struct ps5_query;
static void ps5_draw_batch_drain_query(struct ps5_query *query);
static bool ps5_draw_batch_query_ready(const struct ps5_query *query, bool wait);
static void ps5_draw_batch_drain_buffer(struct pipe_resource *resource);
static bool ps5_draw_batch_compute_only(void);
#else
#define ps5_draw_batch_drain_buffer(resource) \
   ps5_invalidate_gpu_writes((struct ps5_resource *)(resource))
#define ps5_draw_batch_drain_query(query) ((void)(query))
#define ps5_draw_batch_query_ready(query, wait) true
#endif
static int ps5_storage_image_view_descriptor(const struct pipe_image_view *view,
                                             uint32_t descriptor[8]);

void
ps5_screen_submit_lock(struct pipe_screen *base)
{
#ifdef PS5_DEFERRED_DRAW_BATCH
   (void)base;
   simple_mtx_lock(&ps5_deferred_mutex);
   ps5_draw_batch_flush_locked();
   ps5_draw_batch_retire_locked(true);
#else
   simple_mtx_lock(&((struct ps5_screen *)base)->submit_mutex);
#endif
}

void
ps5_screen_present_lock(struct pipe_screen *base)
{
#ifdef PS5_DEFERRED_DRAW_BATCH
   (void)base;
   simple_mtx_lock(&ps5_deferred_mutex);
   /* Another context can enqueue after the EGL flush releases this mutex. */
   ps5_draw_batch_flush_locked();
   ps5_draw_batch_retire_locked(false);
#else
   simple_mtx_lock(&((struct ps5_screen *)base)->submit_mutex);
#endif
}

void
ps5_screen_submit_unlock(struct pipe_screen *base)
{
#ifdef PS5_DEFERRED_DRAW_BATCH
   (void)base;
   simple_mtx_unlock(&ps5_deferred_mutex);
#else
   simple_mtx_unlock(&((struct ps5_screen *)base)->submit_mutex);
#endif
}

#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
/* Raw invariant-TSC cycles: no per-draw OS clock calls. Phases overlap:
 * 0=draw preparation, 1=descriptor copy, 2=whole deferred draw entry. */
static uint64_t ps5_prepare_cycles[9], ps5_prepare_calls[9];
static uint64_t ps5_draw_first_cycle;
#define PS5_MAP_CALLER_SLOTS 32
static struct {
   uintptr_t pc;
   uint64_t calls, reads, textures;
} ps5_map_callers[PS5_MAP_CALLER_SLOTS];
static uint64_t ps5_map_caller_overflow;

static void
ps5_map_caller_record(uintptr_t pc, unsigned usage, unsigned target)
{
   for (unsigned i = 0; i < PS5_MAP_CALLER_SLOTS; ++i) {
      uintptr_t current = __atomic_load_n(&ps5_map_callers[i].pc, __ATOMIC_RELAXED);
      if (!current) {
         uintptr_t empty = 0;
         if (__atomic_compare_exchange_n(&ps5_map_callers[i].pc, &empty, pc,
                                         false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            current = pc;
         else
            current = empty;
      }
      if (current != pc)
         continue;
      __atomic_fetch_add(&ps5_map_callers[i].calls, 1, __ATOMIC_RELAXED);
      if (usage & PIPE_MAP_READ)
         __atomic_fetch_add(&ps5_map_callers[i].reads, 1, __ATOMIC_RELAXED);
      if (target != PIPE_BUFFER)
         __atomic_fetch_add(&ps5_map_callers[i].textures, 1, __ATOMIC_RELAXED);
      return;
   }
   __atomic_fetch_add(&ps5_map_caller_overflow, 1, __ATOMIC_RELAXED);
}
static uint64_t ps5_prepare_clock(void)
{
   uint32_t low, high;
   __asm__ volatile("lfence; rdtsc" : "=a"(low), "=d"(high) : : "memory");
   return (uint64_t)high << 32 | low;
}
static void ps5_prepare_report(void)
{
   for (unsigned phase = 0; phase < 9; ++phase)
      printf("[ps5-driver-cycles] phase=%u calls=%" PRIu64 " cycles=%" PRIu64 "\n",
             phase, __atomic_load_n(&ps5_prepare_calls[phase], __ATOMIC_RELAXED),
             __atomic_load_n(&ps5_prepare_cycles[phase], __ATOMIC_RELAXED));
   printf("[ps5-map-caller-anchor] report=%" PRIxPTR " overflow=%" PRIu64 "\n",
          (uintptr_t)ps5_prepare_report,
          __atomic_load_n(&ps5_map_caller_overflow, __ATOMIC_RELAXED));
   for (unsigned i = 0; i < PS5_MAP_CALLER_SLOTS; ++i) {
      uintptr_t pc = __atomic_load_n(&ps5_map_callers[i].pc, __ATOMIC_RELAXED);
      if (pc)
         printf("[ps5-map-caller] pc=%" PRIxPTR " calls=%" PRIu64
                " reads=%" PRIu64 " textures=%" PRIu64 "\n", pc,
                __atomic_load_n(&ps5_map_callers[i].calls, __ATOMIC_RELAXED),
                __atomic_load_n(&ps5_map_callers[i].reads, __ATOMIC_RELAXED),
                __atomic_load_n(&ps5_map_callers[i].textures, __ATOMIC_RELAXED));
   }
}
struct ps5_prepare_scope { unsigned phase; uint64_t start; };
static void ps5_prepare_scope_end(struct ps5_prepare_scope *scope)
{
   uint64_t elapsed = ps5_prepare_clock() - scope->start;
   __atomic_fetch_add(&ps5_prepare_cycles[scope->phase], elapsed, __ATOMIC_RELAXED);
   uint64_t calls = __atomic_add_fetch(&ps5_prepare_calls[scope->phase], 1, __ATOMIC_RELAXED);
   if (scope->phase == 0) {
      if (calls == 1)
         ps5_draw_first_cycle = scope->start;
      if (calls % 10000 == 0) {
         ps5_opengl_register_gl_consumer();
         printf("[ps5-driver-elapsed] calls=%" PRIu64 " cycles=%" PRIu64 "\n",
                calls, ps5_prepare_clock() - ps5_draw_first_cycle);
         ps5_prepare_report();
      }
   }
}
#endif

static uint64_t ps5_texture_publication_epoch = 1;

static void
ps5_draw_batch_drain(void)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {3, ps5_prepare_clock()};
#endif
#ifdef PS5_DEFERRED_DRAW_BATCH
   ps5_screen_submit_lock(NULL);
   ++ps5_texture_publication_epoch;
   ps5_screen_submit_unlock(NULL);
#endif
}

static void
ps5_draw_batch_submit(void)
{
#ifdef PS5_DEFERRED_DRAW_BATCH
   simple_mtx_lock(&ps5_deferred_mutex);
   ps5_draw_batch_flush_locked();
   simple_mtx_unlock(&ps5_deferred_mutex);
#endif
}


#define PS5_MAX_TEXTURE_UNITS 16u
#define PS5_MERGED_TEXTURE_UNITS (2u * PS5_MAX_TEXTURE_UNITS)
#define PS5_MAX_CONSTANT_BUFFERS 15u
#define PS5_DESCRIPTOR_STAGE_COUNT 2u
#define PS5_TEXTURE_STAGE_COUNT 5u
#define PS5_CONSTANT_STAGE_COUNT 5u
#define PS5_GEOMETRY_CONSTANT_SLOT 2u
#define PS5_TESS_CTRL_CONSTANT_SLOT 3u
#define PS5_TESS_EVAL_CONSTANT_SLOT 4u
#define PS5_GEOMETRY_TEXTURE_SLOT 2u
#define PS5_TESS_CTRL_TEXTURE_SLOT 3u
#define PS5_TESS_EVAL_TEXTURE_SLOT 4u
#define PS5_TESSELLATION_TEXTURE_BINDING 16u
#define PS5_MAX_RENDER_TARGETS 8u
#define PS5_TEXTURE_DESCRIPTOR_STRIDE 48u
#define PS5_TEXTURE_DESCRIPTOR_BYTES \
   (PS5_MERGED_TEXTURE_UNITS * PS5_TEXTURE_DESCRIPTOR_STRIDE)
#define PS5_STREAMOUT_DESCRIPTOR_OFFSET (PIPE_MAX_ATTRIBS * 16u)
#define PS5_STREAMOUT_CONTROL_DESCRIPTOR 4u
#define PS5_STREAMOUT_CONTROL_OFFSET \
   (PS5_STREAMOUT_DESCRIPTOR_OFFSET + 8u * 16u)
#define PS5_STREAMOUT_CONTROL_BYTES 64u
#define PS5_OCCLUSION_MAX_RBS 16u
#define PS5_OCCLUSION_QUERY_BYTES (PS5_OCCLUSION_MAX_RBS * 16u)
#define PS5_OCCLUSION_VALID_BIT (UINT64_C(1) << 63)
#define PS5_BORDER_COLOR_COUNT 4096u
#define PS5_BORDER_COLOR_BYTES \
   (PS5_BORDER_COLOR_COUNT * sizeof(union pipe_color_union))

struct ps5_sampler_state {
   struct pipe_sampler_state base;
   uint16_t border_color_ptr;
   uint8_t border_color_type;
   bool words_valid;
   uint32_t words[3];
};

struct ps5_constant_state {
   struct pipe_resource *buffer;
   unsigned offset;
   unsigned size;
   bool copied;
   bool valid;
};

#define PS5_COMPUTE_STORAGE_SLOTS PIPE_MAX_SHADER_BUFFERS
#ifndef PS5_ENABLE_COMPUTE_API_TEST
#define PS5_ENABLE_COMPUTE_API_TEST 0
#endif
#define PS5_COMPUTE_CONSTANT_SLOTS 15 /* CB0 plus fourteen user UBOs. */
#define PS5_COMPUTE_BUFFER_SLOTS (PS5_COMPUTE_STORAGE_SLOTS + PS5_COMPUTE_CONSTANT_SLOTS)
#define PS5_COMPUTE_IMAGE_SLOTS PS5_AGC_COMPUTE_MAX_IMAGES
#define PS5_COMPUTE_TEXTURE_SLOTS PS5_AGC_COMPUTE_MAX_TEXTURES
#define PS5_COMPUTE_TEXTURE_OFFSET (PS5_COMPUTE_BUFFER_SLOTS * 16 + PS5_COMPUTE_IMAGE_SLOTS * 32)
#define PS5_COMPUTE_DESCRIPTOR_BYTES (PS5_COMPUTE_TEXTURE_OFFSET + PS5_COMPUTE_TEXTURE_SLOTS * 48)
#define PS5_MAX_VIEWPORTS 16
struct ps5_compute_shader {
   PsbcShaderOutput output;
   unsigned ssbos;
   unsigned ubos;
   unsigned images;
   unsigned textures;
   unsigned buffer_textures;
   unsigned filtered_textures;
   unsigned texture_lod[PS5_COMPUTE_TEXTURE_SLOTS];
   unsigned array_textures;
   uint8_t texture_binding_size[PS5_COMPUTE_TEXTURE_SLOTS];
};

struct ps5_context {
   struct pipe_context base;
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DEFERRED_DRAW_BATCH)
   struct pipe_resource *descriptor_cache[3][PS5_MULTIDRAW_BATCH_CAPACITY];
   unsigned descriptor_cache_count[3];
   uint64_t descriptor_cache_hits, descriptor_cache_misses;
#endif


   struct blitter_context *blitter;
   bool deferred_blitter_draw;
   int last_draw_status;
   unsigned draw_calls;
   bool reported_large_color_fallback;
   bool legacy_primitive_conversion;
   unsigned legacy_first_index;
   int legacy_flat_input_vertex;
   struct ps5_shader *vs;
   struct ps5_shader *tcs;
   struct ps5_shader *default_tcs;
   uint64_t default_tcs_outputs;
   uint8_t default_tcs_vertices;
   float tess_levels[6], default_tcs_levels[6];
   struct ps5_shader *tes;
   struct ps5_shader *gs;
   struct ps5_shader *fs;
   struct ps5_shader *tessellation_vs;
   struct ps5_shader *tessellation_tcs;
   struct ps5_shader *tessellation_tes;
   struct ps5_shader *tessellation_gs;
   bool tessellation_streamout;
   unsigned tessellation_patch_vertices;
   struct ps5_vertex_layout *tessellation_layout;
   PsbcTessellationOutput tessellation_output;
   uint8_t *tessellation_hs_package;
   size_t tessellation_hs_package_size;
   uint8_t *tessellation_tes_package;
   size_t tessellation_tes_package_size;
   uint8_t patch_vertices;
   /* Two-pass tessellation + geometry, see ps5_draw_tess_gs_two_pass. The
    * derived shaders are keyed by the serials of the application's stages. */
   uint64_t tess_capture_tes_serial;
   uint64_t tess_capture_gs_serial;
   struct ps5_shader *tess_capture_tes;
   struct ps5_shader *tess_capture_vs;
   struct ps5_shader *tess_capture_fs;
   struct ps5_vertex_elements *tess_capture_elements;
   struct pipe_resource *tess_capture_buffer;
   bool tess_capture_active;
   struct ps5_shader *geometry_vs;
   struct ps5_shader *geometry_gs;
   struct ps5_vertex_layout *geometry_layout;
   uint32_t geometry_primitive_type;
   bool geometry_provoking_vtx_last;
   PsbcShaderOutput geometry_output;
   uint8_t *geometry_package;
   size_t geometry_package_size;
   PsbcShaderOutput geometry_streamout_output;
   uint8_t *geometry_streamout_package;
   size_t geometry_streamout_package_size;
   struct pipe_framebuffer_state framebuffer;
   bool framebuffer_valid;
   struct ps5_vertex_elements *vertex_elements;
   struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
   unsigned vertex_buffer_count;
   struct pipe_resource *vertex_descriptor_table;
   struct pipe_resource *descriptor_storage[PS5_DESCRIPTOR_STAGE_COUNT];
   struct pipe_resource *border_color_storage;
   unsigned border_color_count;
   struct pipe_stream_output_target *stream_output_targets[PIPE_MAX_SO_BUFFERS];
   struct pipe_resource *streamout_records;
   struct pipe_resource *primitive_query_storage;
   struct pipe_resource *streamout_staging[PIPE_MAX_SO_BUFFERS];
   unsigned stream_output_target_count;
   unsigned split_instance_id;
   enum mesa_prim stream_output_primitive;
   struct ps5_constant_state
      constants[PS5_CONSTANT_STAGE_COUNT][PS5_MAX_CONSTANT_BUFFERS];
   void *samplers[PS5_TEXTURE_STAGE_COUNT][PS5_MAX_TEXTURE_UNITS];
   struct pipe_sampler_view
      *sampler_views[PS5_TEXTURE_STAGE_COUNT][PS5_MAX_TEXTURE_UNITS];
   struct pipe_blend_state *blend;
   bool logicop_used;
   struct pipe_rasterizer_state *rasterizer;
   struct pipe_blend_color blend_color;
   struct pipe_viewport_state viewport[PS5_MAX_VIEWPORTS];
   struct pipe_scissor_state scissor[PS5_MAX_VIEWPORTS];
   uint16_t viewport_valid;
   uint16_t scissor_valid;
   struct pipe_depth_stencil_alpha_state *depth_stencil_alpha;
   struct pipe_stencil_ref stencil_ref;
   struct ps5_query *active_occlusion_query;
   struct ps5_query
      *active_primitives_generated_query[PIPE_MAX_VERTEX_STREAMS];
   struct ps5_query
      *active_primitives_emitted_query[PIPE_MAX_VERTEX_STREAMS];
   struct ps5_query
      *active_streamout_overflow_query[PIPE_MAX_VERTEX_STREAMS + 1];
   struct ps5_query *render_condition_query;
   unsigned sample_mask;
   unsigned min_samples; /* pipe_context::set_min_samples */
   void *fbfetch_sampler; /* nearest sampler for the framebuffer-fetch unit */
   bool queries_enabled;
   bool render_condition_inverted;
   struct ps5_compute_shader *cs;
   struct pipe_resource *compute_descriptors;
   struct pipe_sampler_view *compute_views[PS5_COMPUTE_TEXTURE_SLOTS];
   bool compute_views_invalid;
   uint32_t compute_samplers[PS5_COMPUTE_TEXTURE_SLOTS][4];
   unsigned compute_sampler_mask;
   bool compute_samplers_invalid;
   struct pipe_shader_buffer compute_buffers[PS5_COMPUTE_BUFFER_SLOTS];
   struct pipe_shader_buffer fragment_buffers[PS5_COMPUTE_STORAGE_SLOTS];
   struct pipe_shader_buffer geometry_buffers[PS5_COMPUTE_STORAGE_SLOTS];
   struct pipe_shader_buffer preraster_buffers[3][PS5_COMPUTE_STORAGE_SLOTS];
   bool preraster_bindings_invalid[3];
   bool fragment_bindings_invalid;
   bool geometry_bindings_invalid;
   struct pipe_image_view compute_images[PS5_COMPUTE_IMAGE_SLOTS];
   struct pipe_image_view fragment_images[PS5_COMPUTE_IMAGE_SLOTS];
   struct pipe_image_view preraster_images[4][PS5_COMPUTE_IMAGE_SLOTS];
   bool preraster_images_invalid[4];
   bool fragment_images_invalid;
   bool compute_images_invalid;
   bool compute_bindings_invalid;
   uint32_t compute_constants_invalid;
   int last_compute_status;
   unsigned dispatches;
   uint64_t batch_eligible;
   uint64_t batch_reject[7];
#ifdef PS5_DRAW_PROFILE
   uint64_t batch_checks;
   struct {
      unsigned key[16];
      uint64_t count;
   } framebuffer_fallbacks[32];
   uint64_t framebuffer_fallback_overflow;
#endif
};

struct ps5_shader {
   PsbcStage stage;
   nir_shader *nir;
   /* Never reused, unlike the address of a deleted shader. */
   uint64_t serial;
   /* Framebuffer fetch reads color buffer 0 through this texture unit + 1. */
   unsigned fbfetch_unit;
   struct pipe_stream_output_info stream_output;
   struct ps5_shader_variant *variants;
   struct ps5_shader_variant *active;
};

struct ps5_vertex_layout {
   uint32_t count;
   PsbcVertexAttribute attributes[PSBC_MAX_VERTEX_ATTRIBUTES];
};

struct ps5_fragment_exports {
   uint32_t formats;
   uint32_t int8_mask;
   uint32_t int10_mask;
   uint32_t color_mask;
   unsigned rasterization_samples;
   unsigned ps_iter_samples; /* fragment invocations per pixel; 1 = per pixel */
   bool ignore_sample_mask;
};

struct ps5_shader_variant {
   struct ps5_vertex_layout layout;
   struct ps5_fragment_exports exports;
   uint32_t primitive_type;
   bool provoking_vtx_last;
   bool alpha_to_one;
   bool poly_line_smooth;
   bool omit_implicit_primitive_id;
   bool primitive_id_per_primitive;
   int8_t flat_input_vertex;
   PsbcShaderOutput output;
   PsbcShaderOutput streamout_output;
   uint8_t *package;
   size_t package_size;
   uint8_t *streamout_package;
   size_t streamout_package_size;
   struct ps5_shader_variant *next;
};

struct ps5_stream_output_target {
   struct pipe_stream_output_target base;
   unsigned offset;
   unsigned vertex_count;
};

struct ps5_streamout_control {
   uint32_t buffer_offsets[4];
   uint32_t generated_primitives[4];
   uint32_t emitted_primitives[4];
   uint32_t reserved[4];
};

struct ps5_streamout_record {
   uint32_t primitive, invocation, reserved[2];
   uint32_t offsets[4], generated[4], emitted[4];
};
/* Native receipts cross 1023 -> 0; the generic ABI field is wider. */
#define PS5_TESS_STREAMOUT_ORDINAL_COUNT 1024u
_Static_assert(sizeof(struct ps5_streamout_record) == 64, "streamout record ABI");

_Static_assert(sizeof(struct ps5_streamout_control) ==
               PS5_STREAMOUT_CONTROL_BYTES,
               "PS5 streamout control ABI");

struct ps5_resource {
   struct pipe_resource base;
   uint64_t texture_publication_epoch, stencil_publication_epoch;
   size_t texture_published_bytes, stencil_published_bytes;
   bool external_cpu_access;
   bool gpu_written; /* CPU-cached lines may predate a compute write */
   struct pipe_resource *render_pool_owner;
   struct pipe_resource *stencil_sample;
   uint8_t *data;
   uint8_t *stencil_data;
   size_t size;
   size_t allocation_size;
   size_t stencil_allocation_size;
   unsigned stride;
   size_t level_offset[PIPE_MAX_TEXTURE_LEVELS];
   unsigned level_stride[PIPE_MAX_TEXTURE_LEVELS];
   size_t layer_stride;
   size_t render_staging_offset;
   size_t render_staging_size;
   size_t depth_staging_offset;
   size_t depth_staging_size;
   int64_t direct_start;
   int64_t stencil_direct_start;
   unsigned render_arena_first_slot;
   unsigned render_arena_slot_count;
};

struct ps5_transfer {
   struct pipe_transfer base;
   void *staging;
   size_t staging_mapping_size;
};

struct ps5_vertex_elements {
   unsigned count;
   struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
};

struct ps5_fence {
   unsigned references;
   uint64_t sequence;
};

struct ps5_query {
   unsigned type;
   unsigned index;
   uint64_t start;
   uint64_t end;
   uint64_t value;
   struct pipe_resource *buffer;
   bool active;
   bool ready;
};

static struct ps5_query **
ps5_active_primitive_query(struct ps5_context *context, unsigned type,
                           unsigned index)
{
   if (index >= PIPE_MAX_VERTEX_STREAMS)
      return NULL;
   if (type == PIPE_QUERY_PRIMITIVES_GENERATED)
      return &context->active_primitives_generated_query[index];
   if (type == PIPE_QUERY_PRIMITIVES_EMITTED)
      return &context->active_primitives_emitted_query[index];
   return NULL;
}

static bool
ps5_any_primitive_query(const struct ps5_context *context)
{
   for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS; ++stream)
      if (context->active_primitives_generated_query[stream] ||
          context->active_primitives_emitted_query[stream])
         return true;
   return false;
}

static bool
ps5_any_generated_query(const struct ps5_context *context)
{
   for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS; ++stream)
      if (context->active_primitives_generated_query[stream])
         return true;
   return false;
}

static struct ps5_query **
ps5_active_streamout_overflow_query(struct ps5_context *context,
                                    unsigned type, unsigned index)
{
   if (type == PIPE_QUERY_SO_OVERFLOW_PREDICATE &&
       index < PIPE_MAX_VERTEX_STREAMS)
      return &context->active_streamout_overflow_query[index];
   if (type == PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE)
      return &context->active_streamout_overflow_query[PIPE_MAX_VERTEX_STREAMS];
   return NULL;
}

static bool
ps5_vertex_format(enum pipe_format format, PsbcVertexFormat *out);
static unsigned
ps5_vertex_format_size(enum pipe_format format);
static bool
ps5_vertex_layout_from_state(const struct ps5_shader *shader,
                             const struct ps5_vertex_elements *elements,
                             struct ps5_vertex_layout *layout);
static bool
ps5_lower_sample_interpolation(nir_builder *builder,
                                nir_intrinsic_instr *intrinsic, void *data)
{
   if (intrinsic->intrinsic != nir_intrinsic_load_barycentric_at_sample)
      return false;

   const unsigned samples = MAX2(*(const unsigned *)data, 1u);
   const unsigned mode = nir_intrinsic_interp_mode(intrinsic);
   builder->cursor = nir_before_instr(&intrinsic->instr);
   nir_def *value;
   if (samples == 1) {
      value = nir_load_barycentric_pixel(builder, 32, .interp_mode = mode);
   } else {
      /* Match get_sample_position and the fixed hardware sample pattern;
       * RADV's sample-position ring table is not part of our runtime ABI. */
      nir_def *offset = nir_imm_vec2(builder, 0, 0);
      for (unsigned sample = 0; sample < samples; ++sample) {
         float position[2];
         u_default_get_sample_position(NULL, samples, sample, position);
         offset = nir_bcsel(builder,
            nir_ieq_imm(builder, intrinsic->src[0].ssa, sample),
            nir_imm_vec2(builder, position[0] - 0.5f, position[1] - 0.5f), offset);
      }
      value = nir_load_barycentric_at_offset(builder, 32, offset,
                                             .interp_mode = mode);
   }
   nir_def_replace(&intrinsic->def, value);
   return true;
}

static bool
ps5_select_shader_variant(struct ps5_shader *shader, uint32_t address32_hi,
                          const struct ps5_vertex_layout *layout,
                          uint32_t primitive_type,
                          bool provoking_vtx_last, bool alpha_to_one,
                          bool poly_line_smooth,
                          bool omit_implicit_primitive_id,
                          bool primitive_id_per_primitive,
                          int flat_input_vertex,
                          const struct ps5_fragment_exports *exports);
static bool
ps5_select_geometry_pipeline(struct ps5_context *context,
                             uint32_t address32_hi,
                             const struct ps5_vertex_layout *layout,
                             uint32_t primitive_type);
static bool
ps5_select_tessellation_pipeline(struct ps5_context *context,
                                 uint32_t address32_hi,
                                 const struct ps5_vertex_layout *layout);
static bool
ps5_geometry_ring_itemsize(const PsbcShaderMetadata *metadata,
                           uint32_t *itemsize);
static void ps5_context_profile_report(struct ps5_context *context);

static bool
ps5_render_condition_passes(const struct ps5_context *context);
static unsigned
ps5_streamout_buffer_mask(unsigned stream_buffer_mask);
static unsigned
ps5_streamout_buffer_stream(unsigned stream_buffer_mask, unsigned buffer);

#define PS5_DIRECT_MEMORY_TYPE 12
#define PS5_MAP_PROTECTION 0x33
#define PS5_RENDER_ALIGNMENT 0x200000u
#define PS5_COLOR_TARGET_ALIGNMENT 0x10000u
#define PS5_DEPTH_TARGET_BYTES 0xa00000u
#define PS5_D32_SURFACE_BYTES 0x870000u
#define PS5_STENCIL_TARGET_BYTES 0x280000u
#define PS5_STENCIL_ALIGNMENT 0x10000u
#define PS5_MAX_CONSTANT_BUFFER_SIZE 0x10000u
/* Mesa reserves eight vec4 slots in VS/GS CB0 for lowered clip planes.
 * Advertise enough backing storage for the 1024 user components required by
 * OpenGL 3.3 after that reservation. */
#define PS5_MAX_DEFAULT_CONSTANT_BUFFER_SIZE 0x1080u
#define PS5_VERTEX_STORAGE_OFFSET 0x800u
#define PS5_VERTEX_IMAGE_OFFSET 0xa00u
#define PS5_GEOMETRY_STORAGE_OFFSET 0xc00u
#define PS5_GEOMETRY_IMAGE_OFFSET 0xe00u
#define PS5_CONSTANT_DATA_OFFSET 0x1000u
#define PS5_FRAGMENT_UBO_OFFSET (PS5_COMPUTE_STORAGE_SLOTS * 16u + PS5_COMPUTE_IMAGE_SLOTS * 32u)
#define PS5_FRAGMENT_TEXTURE_OFFSET (PS5_FRAGMENT_UBO_OFFSET + PS5_MAX_CONSTANT_BUFFERS * 16u)
#define PS5_TESSELLATION_BUFFER_OFFSET \
   (PS5_CONSTANT_DATA_OFFSET + 4u * PS5_MAX_CONSTANT_BUFFER_SIZE)
#define PS5_TESSELLATION_BUFFER_STRIDE \
   ((PS5_MAX_CONSTANT_BUFFERS + PS5_COMPUTE_STORAGE_SLOTS) * 16u + \
    PS5_COMPUTE_IMAGE_SLOTS * 32u)
#define PS5_TESSELLATION_TEXTURE_OFFSET \
   (PS5_TESSELLATION_BUFFER_OFFSET + 4u * PS5_TESSELLATION_BUFFER_STRIDE)
#define PS5_DESCRIPTOR_STORAGE_BYTES \
   (PS5_TESSELLATION_TEXTURE_OFFSET + 4u * PS5_MAX_TEXTURE_UNITS * PS5_TEXTURE_DESCRIPTOR_STRIDE)
#define PS5_MAX_TEXTURE_2D_SIZE PS5_MAX_RENDER_SIZE
#define PS5_MAX_TEXTURE_CUBE_LEVELS 15u
#define PS5_MAX_TEXTURE_CUBE_SIZE (1u << (PS5_MAX_TEXTURE_CUBE_LEVELS - 1u))
#define PS5_MAX_TEXTURE_ARRAY_LAYERS 2048u
#define PS5_MAX_TEXTURE_3D_LEVELS 12u
#define PS5_MAX_TEXTURE_3D_SIZE (1u << (PS5_MAX_TEXTURE_3D_LEVELS - 1u))
#define PS5_MAX_TEXEL_BUFFER_ELEMENTS (1u << 20)
#define PS5_MIN_POINT_LINE_SIZE 1.0f
#define PS5_MAX_POINT_LINE_SIZE 64.0f
#ifndef PS5_ENABLE_PACKED_DEPTH_STENCIL
#define PS5_ENABLE_PACKED_DEPTH_STENCIL 0
#endif
#ifndef PS5_ENABLE_SRGB_CANDIDATE
#define PS5_ENABLE_SRGB_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE
#define PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TEXTURE_RG_CANDIDATE
#define PS5_ENABLE_TEXTURE_RG_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_SWIZZLE_CANDIDATE
#define PS5_ENABLE_TEXTURE_SWIZZLE_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_SNORM_CANDIDATE
#define PS5_ENABLE_TEXTURE_SNORM_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_FLOAT_CANDIDATE
#define PS5_ENABLE_TEXTURE_FLOAT_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_SHARED_EXPONENT_CANDIDATE
#define PS5_ENABLE_SHARED_EXPONENT_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_PACKED_FLOAT_CANDIDATE
#define PS5_ENABLE_PACKED_FLOAT_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE
#define PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_NARROW_TEXTURE_INTEGER_CANDIDATE
#define PS5_ENABLE_NARROW_TEXTURE_INTEGER_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_RGB10_A2UI_CANDIDATE
#define PS5_ENABLE_RGB10_A2UI_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE
#define PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_SHADER_TEXTURE_LOD_CANDIDATE
#define PS5_ENABLE_SHADER_TEXTURE_LOD_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE
#define PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_CUBE_CANDIDATE
#define PS5_ENABLE_TEXTURE_CUBE_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_CUBE_ARRAY_CANDIDATE
#define PS5_ENABLE_TEXTURE_CUBE_ARRAY_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TEXTURE_ARRAY_CANDIDATE
#define PS5_ENABLE_TEXTURE_ARRAY_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE
#define PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TEXTURE_3D_CANDIDATE
#define PS5_ENABLE_TEXTURE_3D_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TEXTURE_1D_CANDIDATE
#define PS5_ENABLE_TEXTURE_1D_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_UBO_CANDIDATE
#define PS5_ENABLE_UBO_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_TIMER_QUERY_CANDIDATE
#define PS5_ENABLE_TIMER_QUERY_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE
#define PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_DEPTH_CLAMP_CANDIDATE
#define PS5_ENABLE_DEPTH_CLAMP_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_330_CANDIDATE
#define PS5_ENABLE_GLSL_330_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_400_CANDIDATE
#define PS5_ENABLE_GLSL_400_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_410_CANDIDATE
#define PS5_ENABLE_GLSL_410_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_420_CANDIDATE
#define PS5_ENABLE_GLSL_420_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_430_CANDIDATE
#define PS5_ENABLE_GLSL_430_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_440_CANDIDATE
#define PS5_ENABLE_GLSL_440_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_450_CANDIDATE
#define PS5_ENABLE_GLSL_450_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GLSL_460_CANDIDATE
#define PS5_ENABLE_GLSL_460_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_FP64_CANDIDATE
#define PS5_ENABLE_FP64_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_VIEWPORT_ARRAY_CANDIDATE
#define PS5_ENABLE_VIEWPORT_ARRAY_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_PACKED_VERTEX_CANDIDATE
#define PS5_ENABLE_PACKED_VERTEX_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_INTEGER_VERTEX_CANDIDATE
#define PS5_ENABLE_INTEGER_VERTEX_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_PADDED_FBO_CANDIDATE
#define PS5_ENABLE_PADDED_FBO_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE
#define PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE
#define PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_SOFTWARE_BLIT_CANDIDATE
#define PS5_ENABLE_SOFTWARE_BLIT_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE
#define PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE
#define PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE 0
#endif
#define PS5_MAX_COLOR_WIDTH (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE \
   ? PS5_MAX_RENDER_SIZE : PS5_RENDER_WIDTH)
#define PS5_MAX_COLOR_HEIGHT (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE \
   ? PS5_MAX_RENDER_SIZE : PS5_RENDER_HEIGHT)
#define PS5_MAX_DEPTH_WIDTH (PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE \
   ? PS5_MAX_RENDER_SIZE : PS5_RENDER_WIDTH)
#define PS5_MAX_DEPTH_HEIGHT (PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE \
   ? PS5_MAX_RENDER_SIZE : PS5_RENDER_HEIGHT)
#ifndef PS5_ENABLE_SHARED_RENDER_POOL_CANDIDATE
#define PS5_ENABLE_SHARED_RENDER_POOL_CANDIDATE 1
#endif
#ifndef PS5_ENABLE_FAKE_SW_MSAA_CANDIDATE
#define PS5_ENABLE_FAKE_SW_MSAA_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_MSAA4_CANDIDATE
#define PS5_ENABLE_MSAA4_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_MSAA_ARRAY_CANDIDATE
#define PS5_ENABLE_MSAA_ARRAY_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_SEAMLESS_CUBE_CANDIDATE
#define PS5_ENABLE_SEAMLESS_CUBE_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_GEOMETRY_CANDIDATE
#define PS5_ENABLE_GEOMETRY_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TESSELLATION_CANDIDATE
#define PS5_ENABLE_TESSELLATION_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_DRAW_INDIRECT_CANDIDATE
#define PS5_ENABLE_DRAW_INDIRECT_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_MRT_CANDIDATE
#define PS5_ENABLE_MRT_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_INDEPENDENT_BLEND_CANDIDATE
#define PS5_ENABLE_INDEPENDENT_BLEND_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE
#define PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE
#define PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE
#define PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_POINT_COORD_CANDIDATE
#define PS5_ENABLE_POINT_COORD_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_BORDER_COLOR_CANDIDATE
#define PS5_ENABLE_BORDER_COLOR_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE
#define PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_CORE_TEXTURE_FORMATS_CANDIDATE
#define PS5_ENABLE_CORE_TEXTURE_FORMATS_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_CORE_RENDER_FORMATS_CANDIDATE
#define PS5_ENABLE_CORE_RENDER_FORMATS_CANDIDATE 0
#endif
#ifndef PS5_ENABLE_SMOOTH_RASTER_CANDIDATE
#define PS5_ENABLE_SMOOTH_RASTER_CANDIDATE 0
#endif
#ifndef PS5_PUBLIC_TEXTURE_RG_RENDER_TEST
#define PS5_PUBLIC_TEXTURE_RG_RENDER_TEST 0
#endif
#ifndef PS5_PUBLIC_TEXTURE_RG_TILE_TEST
#define PS5_PUBLIC_TEXTURE_RG_TILE_TEST 0
#endif

_Static_assert(PS5_DEPTH_TARGET_BYTES >= PS5_D32_SURFACE_BYTES,
               "D32 allocation is smaller than the swizzled surface");
_Static_assert(PS5_RENDER_POOL_BYTES >= 2u * PS5_RENDER_TARGET_BYTES,
               "render pool cannot hold two VideoOut buffers");
_Static_assert(PS5_RENDER_POOL_BYTES <= UINT32_MAX,
               "render pool exceeds one shader address32 window");
_Static_assert((PS5_RENDER_POOL_BYTES - PS5_RENDER_ARENA_OFFSET) %
                  PS5_RENDER_ARENA_SLOT_BYTES == 0,
               "render arena is not slot aligned");
_Static_assert((PS5_RENDER_ARENA_SLOT_COUNT + 63u) / 64u ==
                  PS5_RENDER_ARENA_BITMAP_WORDS,
               "render arena bitmap does not cover its slots");
_Static_assert(PS5_MAX_CONSTANT_BUFFERS * 16u <= PS5_CONSTANT_DATA_OFFSET,
               "constant descriptors overlap copied data");
_Static_assert(PS5_TEXTURE_DESCRIPTOR_BYTES +
                  2u * PS5_MAX_CONSTANT_BUFFERS * 16u <=
               PS5_VERTEX_STORAGE_OFFSET,
               "merged VS/GS UBO descriptors overlap vertex storage");
_Static_assert(PS5_VERTEX_STORAGE_OFFSET +
                  PS5_COMPUTE_STORAGE_SLOTS * 16u <=
               PS5_VERTEX_IMAGE_OFFSET,
               "vertex storage descriptors overlap vertex images");
_Static_assert(PS5_VERTEX_IMAGE_OFFSET + PS5_COMPUTE_IMAGE_SLOTS * 32u <=
               PS5_GEOMETRY_STORAGE_OFFSET,
               "vertex images overlap geometry storage descriptors");
_Static_assert(PS5_GEOMETRY_STORAGE_OFFSET +
                  PS5_COMPUTE_STORAGE_SLOTS * 16u <=
               PS5_GEOMETRY_IMAGE_OFFSET,
               "geometry storage descriptors overlap geometry images");
_Static_assert(PS5_GEOMETRY_IMAGE_OFFSET + PS5_COMPUTE_IMAGE_SLOTS * 32u <=
               PS5_CONSTANT_DATA_OFFSET,
               "geometry images overlap copied data");
_Static_assert(PS5_FRAGMENT_TEXTURE_OFFSET +
                  PS5_MAX_TEXTURE_UNITS * PS5_TEXTURE_DESCRIPTOR_STRIDE <=
               PS5_CONSTANT_DATA_OFFSET,
               "fragment descriptors overlap copied data");
_Static_assert(PS5_MERGED_TEXTURE_UNITS <= PSBC_GALLIUM_UBO_BINDING_BASE,
               "merged VS/GS samplers overlap UBO bindings");
_Static_assert(PSBC_GALLIUM_UBO_BINDING_BASE +
                  2u * PS5_MAX_CONSTANT_BUFFERS <=
               PSBC_MAX_DESCRIPTOR_BINDINGS,
               "PSBC descriptor ABI is too small for merged VS/GS UBOs");

struct ps5_native_depth_stencil_state {
   uint32_t depth_control;
   uint32_t stencil_control;
   uint32_t stencil_refmask;
   uint32_t stencil_refmask_bf;
};

struct ps5_native_graphics_state {
   uint32_t blend_control[PS5_MAX_RENDER_TARGETS];
   uint32_t target_mask;
   uint32_t color_control;
   uint32_t color_control_valid;
   uint32_t dual_source_blend;
   uint32_t alpha_to_coverage;
   uint32_t alpha_to_one;
   uint32_t blend_color[4];
   uint32_t viewport[PS5_MAX_VIEWPORTS][8];
   uint32_t scissor[PS5_MAX_VIEWPORTS][2];
   uint32_t clip_control;
   uint32_t clip_control_valid;
   uint32_t rasterizer_control;
   uint32_t rasterizer_valid;
   uint32_t point_line[3];
   uint32_t point_line_valid;
   uint32_t interp_control;
   uint32_t interp_control_valid;
   uint32_t polygon_offset[6];
   uint32_t polygon_offset_valid;
};

static uint32_t
ps5_float_bits(float value)
{
   uint32_t bits;

   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static size_t
ps5_tiled_depth_layer_xor(unsigned layer)
{
   /* GFX10 16-pipe 64KB_Z_X: address bits 8..11 contain Z3..Z0.
    * See Mesa addrlib's GFX10_SW_PATTERN_NIBBLE2[74]. */
   return ((layer & 1u) << 11) | ((layer & 2u) << 9) |
          ((layer & 4u) << 7) | ((layer & 8u) << 5);
}

static size_t
ps5_tiled_surface_size(unsigned width, unsigned height)
{
   return (size_t)((width + 127u) >> 7) *
          ((height + 127u) >> 7) * UINT32_C(0x10000);
}

static size_t
ps5_tiled_stencil_surface_size(unsigned width, unsigned height)
{
   return (size_t)((width + 255u) >> 8) *
          ((height + 255u) >> 8) * UINT32_C(0x10000);
}

static size_t
ps5_tiled_stencil_surface_size_samples(unsigned width, unsigned height,
                                       unsigned samples)
{
   return samples == 4
             ? (size_t)((width + 127u) >> 7) *
                  ((height + 127u) >> 7) * UINT32_C(0x10000)
             : ps5_tiled_stencil_surface_size(width, height);
}

static size_t
ps5_tiled_rgba8_msaa4_surface_size(unsigned width, unsigned height)
{
   return (size_t)((width + 63u) >> 6) *
          ((height + 63u) >> 6) * UINT32_C(0x10000);
}

static bool
ps5_tiled_color_msaa4_tile(enum pipe_format format, unsigned *width,
                           unsigned *height)
{
   switch (util_format_get_blocksize(format)) {
   case 1: *width = 128; *height = 128; return true;
   case 2: *width = 128; *height = 64; return true;
   case 4: *width = 64; *height = 64; return true;
   case 8: *width = 64; *height = 32; return true;
   case 16: *width = 32; *height = 32; return true;
   default: return false;
   }
}

static size_t
ps5_tiled_color_msaa4_surface_size(enum pipe_format format, unsigned width,
                                   unsigned height)
{
   unsigned tile_width;
   unsigned tile_height;

   if (!ps5_tiled_color_msaa4_tile(format, &tile_width, &tile_height))
      return 0;
   return (size_t)((width + tile_width - 1u) / tile_width) *
          ((height + tile_height - 1u) / tile_height) *
          UINT32_C(0x10000);
}

static size_t
ps5_tiled_depth_surface_size(unsigned width, unsigned height,
                             unsigned samples)
{
   return samples == 4
             ? ps5_tiled_rgba8_msaa4_surface_size(width, height)
             : ps5_tiled_surface_size(width, height);
}

static bool
ps5_integer_texture_format(enum pipe_format format)
{
   if (format == PIPE_FORMAT_R32G32B32A32_UINT ||
       format == PIPE_FORMAT_R32G32B32A32_SINT)
      return true;
   return PS5_ENABLE_NARROW_TEXTURE_INTEGER_CANDIDATE &&
          (format == PIPE_FORMAT_R8G8B8A8_UINT ||
           format == PIPE_FORMAT_R8G8B8A8_SINT ||
           format == PIPE_FORMAT_R16G16B16A16_UINT ||
           format == PIPE_FORMAT_R16G16B16A16_SINT);
}

static bool
ps5_core_render_target_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}

static bool
ps5_render_target_format(enum pipe_format format)
{
   if (PS5_ENABLE_CORE_RENDER_FORMATS_CANDIDATE)
      return ps5_core_render_target_format(format);
   return format == PIPE_FORMAT_R8G8B8A8_UNORM ||
          (PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE &&
           format == PIPE_FORMAT_R8G8B8A8_SRGB) ||
          (PS5_ENABLE_TEXTURE_RG_CANDIDATE &&
           (format == PIPE_FORMAT_R8_UNORM ||
            format == PIPE_FORMAT_R8G8_UNORM)) ||
          (PS5_ENABLE_PACKED_FLOAT_CANDIDATE &&
           format == PIPE_FORMAT_R11G11B10_FLOAT) ||
          (PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE &&
           ps5_integer_texture_format(format)) ||
          (PS5_ENABLE_RGB10_A2UI_CANDIDATE &&
           format == PIPE_FORMAT_R10G10B10A2_UINT);
}

static bool
ps5_msaa4_color_format(enum pipe_format format)
{
   unsigned tile_width;
   unsigned tile_height;

   return ps5_render_target_format(format) &&
          ps5_tiled_color_msaa4_tile(format, &tile_width, &tile_height);
}

static bool
ps5_msaa4_color_support(enum pipe_format format,
                        enum pipe_texture_target target,
                        unsigned sample_count,
                        unsigned storage_sample_count,
                        unsigned bindings)
{
   const unsigned allowed = PIPE_BIND_RENDER_TARGET |
      PIPE_BIND_SAMPLER_VIEW |
      (PS5_ENABLE_GLSL_420_CANDIDATE ? PIPE_BIND_SHADER_IMAGE : 0);

   return PS5_ENABLE_MSAA4_CANDIDATE &&
          (target == PIPE_TEXTURE_2D ||
           (PS5_ENABLE_MSAA_ARRAY_CANDIDATE &&
            target == PIPE_TEXTURE_2D_ARRAY)) &&
          sample_count == 4 && storage_sample_count == 4 && bindings &&
          ps5_msaa4_color_format(format) && (bindings & allowed) &&
          !(bindings & ~allowed);
}

static bool
ps5_msaa4_depth_support(enum pipe_format format,
                        enum pipe_texture_target target,
                        unsigned sample_count,
                        unsigned storage_sample_count,
                        unsigned bindings)
{
   const unsigned allowed = PIPE_BIND_DEPTH_STENCIL |
      (PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE ? PIPE_BIND_SAMPLER_VIEW : 0);

   return PS5_ENABLE_MSAA4_CANDIDATE &&
          (target == PIPE_TEXTURE_2D ||
           (PS5_ENABLE_MSAA_ARRAY_CANDIDATE &&
            PS5_ENABLE_TEXTURE_ARRAY_CANDIDATE &&
            PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
            PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE &&
            target == PIPE_TEXTURE_2D_ARRAY)) &&
          sample_count == 4 && storage_sample_count == 4 && bindings &&
          (format == PIPE_FORMAT_Z32_FLOAT ||
           (PS5_ENABLE_PACKED_DEPTH_STENCIL &&
            format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)) &&
          (bindings & allowed) && !(bindings & ~allowed);
}

static size_t
ps5_tiled_color_surface_size(enum pipe_format format, unsigned width,
                             unsigned height)
{
   unsigned tile_width = 128;
   unsigned tile_height = 128;
   unsigned bytes_per_pixel = util_format_get_blocksize(format);

   if (bytes_per_pixel == 1) {
      tile_width = 256;
      tile_height = 256;
   } else if (bytes_per_pixel == 2) {
      tile_width = 256;
      tile_height = 128;
   } else if (bytes_per_pixel == 8) {
      tile_width = 128;
      tile_height = 64;
   } else if (bytes_per_pixel == 16) {
      tile_width = 64;
      tile_height = 64;
   }
   return (size_t)((width + tile_width - 1u) / tile_width) *
          ((height + tile_height - 1u) / tile_height) *
          UINT32_C(0x10000);
}

static uint32_t
ps5_color_target_info(enum pipe_format format)
{
   if (PS5_ENABLE_CORE_RENDER_FORMATS_CANDIDATE &&
       ps5_core_render_target_format(format)) {
      unsigned cb_format = ac_get_cb_format(GFX10, format);
      unsigned number_type = ac_get_cb_number_type(format);
      unsigned swap = ac_translate_colorswap(GFX10, format, false);
      uint32_t info;

      if (!cb_format || swap == ~0u)
         return 0;
      info = (cb_format << 2) | (number_type << 8) | (swap << 11);
      if (number_type <= 1u || number_type == 6u)
         info |= UINT32_C(1) << 15;
      else
         info |= (UINT32_C(1) << 17) | (UINT32_C(1) << 18);
      if (number_type == 4u || number_type == 5u)
         info |= UINT32_C(1) << 16;
      return info;
   }
   switch (format) {
   case PIPE_FORMAT_R8_UNORM: return UINT32_C(0x00008004);
   case PIPE_FORMAT_R8G8_UNORM: return UINT32_C(0x0000800c);
   case PIPE_FORMAT_R11G11B10_FLOAT: return UINT32_C(0x00060718);
   case PIPE_FORMAT_R8G8B8A8_UINT: return UINT32_C(0x00070428);
   case PIPE_FORMAT_R8G8B8A8_SINT: return UINT32_C(0x00070528);
   case PIPE_FORMAT_R16G16B16A16_UINT: return UINT32_C(0x00070430);
   case PIPE_FORMAT_R16G16B16A16_SINT: return UINT32_C(0x00070530);
   case PIPE_FORMAT_R32G32B32A32_UINT: return UINT32_C(0x00070438);
   case PIPE_FORMAT_R32G32B32A32_SINT: return UINT32_C(0x00070538);
   case PIPE_FORMAT_R10G10B10A2_UINT: return UINT32_C(0x00070424);
   case PIPE_FORMAT_R8G8B8A8_SRGB: return UINT32_C(0x00008628);
   default: return UINT32_C(0x00008028);
   }
}

static struct ps5_fragment_exports
ps5_fragment_exports_for_framebuffer(const struct pipe_framebuffer_state *fb)
{
   struct ps5_fragment_exports exports = {.formats = UINT32_C(0x99999999)};
   /* Zero means non-multisample; a one-sample MS texture still uses the mask. */
   exports.ignore_sample_mask = fb->samples == 0;
   exports.rasterization_samples = fb->samples;

   for (unsigned i = 0; i < fb->nr_cbufs; ++i) {
      enum pipe_format format = fb->cbufs[i].format;
      struct ac_spi_color_formats formats;
      uint32_t info;

      if (!fb->cbufs[i].texture)
         continue;
      exports.color_mask |= 1u << i;
      info = ps5_color_target_info(format);
      /* Always retain alpha and blending support. This avoids variants for
       * blend enable/alpha-to-coverage while preserving float32 precision. */
      ac_choose_spi_color_formats(G_028C70_FORMAT_GFX6(info),
                                   G_028C70_COMP_SWAP(info),
                                   G_028C70_NUMBER_TYPE(info), false, false,
                                   &formats);
      exports.formats = (exports.formats & ~(UINT32_C(0xf) << (4 * i))) |
                        (formats.blend_alpha << (4 * i));
      if (util_format_is_pure_integer(format)) {
         unsigned bits = util_format_description(format)->channel[0].size;
         exports.int8_mask |= (bits == 8 ? 1u : 0u) << i;
         exports.int10_mask |= (bits == 10 ? 1u : 0u) << i;
      }
   }
   return exports;
}

static bool
ps5_core_sampled_texture_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16X16_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return true;
   default:
      return false;
   }
}

static bool
ps5_sampled_texture_format(enum pipe_format format)
{
   return format == PIPE_FORMAT_R8G8B8A8_UNORM ||
          (PS5_ENABLE_SRGB_CANDIDATE &&
           format == PIPE_FORMAT_R8G8B8A8_SRGB) ||
          (PS5_ENABLE_TEXTURE_RG_CANDIDATE &&
           (format == PIPE_FORMAT_R8_UNORM ||
            format == PIPE_FORMAT_R8G8_UNORM)) ||
          (PS5_ENABLE_TEXTURE_SNORM_CANDIDATE &&
           (format == PIPE_FORMAT_R8_SNORM ||
            format == PIPE_FORMAT_R8G8_SNORM ||
            format == PIPE_FORMAT_R8G8B8A8_SNORM)) ||
          (PS5_ENABLE_TEXTURE_FLOAT_CANDIDATE &&
           (format == PIPE_FORMAT_R16G16B16A16_FLOAT ||
            format == PIPE_FORMAT_R32G32B32A32_FLOAT)) ||
          (PS5_ENABLE_SHARED_EXPONENT_CANDIDATE &&
           format == PIPE_FORMAT_R9G9B9E5_FLOAT) ||
          (PS5_ENABLE_PACKED_FLOAT_CANDIDATE &&
           format == PIPE_FORMAT_R11G11B10_FLOAT) ||
          (PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE &&
           ps5_integer_texture_format(format)) ||
          (PS5_ENABLE_RGB10_A2UI_CANDIDATE &&
           format == PIPE_FORMAT_R10G10B10A2_UINT) ||
          (PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE &&
            (format == PIPE_FORMAT_Z32_FLOAT ||
             (PS5_ENABLE_PACKED_DEPTH_STENCIL &&
              format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) ||
             (PS5_ENABLE_GLSL_430_CANDIDATE &&
              (format == PIPE_FORMAT_X32_S8X24_UINT ||
               format == PIPE_FORMAT_S8_UINT)))) ||
          (PS5_ENABLE_CORE_TEXTURE_FORMATS_CANDIDATE &&
           ps5_core_sampled_texture_format(format));
}

static bool
ps5_packed_vertex_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_SNORM:
   case PIPE_FORMAT_B10G10R10A2_SNORM:
   case PIPE_FORMAT_R10G10B10A2_USCALED:
   case PIPE_FORMAT_B10G10R10A2_USCALED:
   case PIPE_FORMAT_R10G10B10A2_SSCALED:
   case PIPE_FORMAT_B10G10R10A2_SSCALED:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}

static bool
ps5_integer_vertex_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32_SINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32B32_UINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
      return true;
   default:
      return false;
   }
}

static bool
ps5_texel_buffer_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32_FLOAT:
   case PIPE_FORMAT_R32G32B32_UINT:
   case PIPE_FORMAT_R32G32B32_SINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   /* Buffer images also use the SNORM image formats (rgba8/rgba16_snorm). */
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
      return true;
   default:
      return false;
   }
}

static bool
ps5_cube_texture_target(enum pipe_texture_target target)
{
   return target == PIPE_TEXTURE_CUBE ||
          (PS5_ENABLE_TEXTURE_CUBE_ARRAY_CANDIDATE &&
           target == PIPE_TEXTURE_CUBE_ARRAY);
}

static bool
ps5_sampled_texture_target(enum pipe_texture_target target)
{
   return (PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE && target == PIPE_BUFFER) ||
          (PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
           (target == PIPE_TEXTURE_1D ||
            target == PIPE_TEXTURE_1D_ARRAY)) ||
          target == PIPE_TEXTURE_2D ||
          (PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE &&
           target == PIPE_TEXTURE_RECT) ||
          (PS5_ENABLE_TEXTURE_CUBE_CANDIDATE &&
           ps5_cube_texture_target(target)) ||
          (PS5_ENABLE_TEXTURE_ARRAY_CANDIDATE &&
           target == PIPE_TEXTURE_2D_ARRAY) ||
          (PS5_ENABLE_TEXTURE_3D_CANDIDATE &&
           target == PIPE_TEXTURE_3D);
}

static bool
ps5_texture_view_target_compatible(enum pipe_texture_target storage,
                                   enum pipe_texture_target view)
{
   if (storage == view)
      return true;
   if ((storage == PIPE_TEXTURE_1D || storage == PIPE_TEXTURE_1D_ARRAY) &&
       (view == PIPE_TEXTURE_1D || view == PIPE_TEXTURE_1D_ARRAY))
      return true;
   if ((storage == PIPE_TEXTURE_2D || storage == PIPE_TEXTURE_2D_ARRAY ||
        ps5_cube_texture_target(storage)) &&
       (view == PIPE_TEXTURE_2D || view == PIPE_TEXTURE_2D_ARRAY ||
        ps5_cube_texture_target(view)))
      return true;
   return false;
}

static bool
ps5_texture_view_format_compatible(enum pipe_format storage,
                                   enum pipe_format view)
{
   if (storage == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
       view == PIPE_FORMAT_X32_S8X24_UINT)
      return PS5_ENABLE_GLSL_430_CANDIDATE;
   return ps5_sampled_texture_format(storage) &&
          ps5_sampled_texture_format(view) &&
          util_format_get_blocksize(storage) == util_format_get_blocksize(view) &&
          util_format_get_blockwidth(storage) == util_format_get_blockwidth(view) &&
          util_format_get_blockheight(storage) == util_format_get_blockheight(view);
}

static bool
ps5_linear_sampled_layout(const struct pipe_resource *resource)
{
   /* Reuse the native render/sample layout for single-mip color images. CPU
    * access already converts through transfer_map/unmap; draws need no copy.
    * ponytail: other formats, mip chains and layers keep existing staging. */
   if (PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE &&
       PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE && resource &&
       resource->target == PIPE_TEXTURE_2D &&
       (resource->format == PIPE_FORMAT_R8G8B8A8_UNORM ||
        resource->format == PIPE_FORMAT_R8G8B8A8_SRGB ||
        resource->format == PIPE_FORMAT_R8_UNORM ||
        resource->format == PIPE_FORMAT_R8G8_UNORM ||
        resource->format == PIPE_FORMAT_R16G16B16A16_FLOAT) &&
       ps5_sampled_texture_format(resource->format) &&
       ps5_render_target_format(resource->format) &&
       resource->nr_samples <= 1 && resource->nr_storage_samples <= 1 &&
       resource->array_size == 1 && resource->depth0 == 1 &&
       !resource->last_level && resource->width0 && resource->height0 &&
       resource->width0 <= PS5_MAX_COLOR_WIDTH && resource->height0 <= PS5_MAX_COLOR_HEIGHT &&
       (resource->bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW)) ==
          (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW))
      return false;
   return resource && ps5_sampled_texture_format(resource->format) &&
          resource->nr_samples <= 1 &&
          (!(resource->bind & PIPE_BIND_DEPTH_STENCIL) ||
           resource->target == PIPE_TEXTURE_RECT ||
           ((resource->format == PIPE_FORMAT_Z32_FLOAT ||
             resource->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) &&
            resource->last_level > 0)) &&
          (resource->target != PIPE_TEXTURE_2D ||
           !(resource->bind & PIPE_BIND_RENDER_TARGET) ||
           (resource->bind & PIPE_BIND_SAMPLER_VIEW) ||
           resource->last_level > 0);
}

static bool
ps5_color_render_target(enum pipe_texture_target target)
{
   return (PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
           (target == PIPE_TEXTURE_1D ||
            target == PIPE_TEXTURE_1D_ARRAY)) ||
          target == PIPE_TEXTURE_2D ||
          (PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE &&
           target == PIPE_TEXTURE_RECT) ||
          (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
           (target == PIPE_TEXTURE_2D_ARRAY ||
            ps5_cube_texture_target(target) ||
            target == PIPE_TEXTURE_3D));
}

static bool
ps5_depth_render_target(enum pipe_texture_target target)
{
   return (PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
           (target == PIPE_TEXTURE_1D ||
            target == PIPE_TEXTURE_1D_ARRAY)) ||
          target == PIPE_TEXTURE_2D ||
          (PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE &&
           target == PIPE_TEXTURE_RECT) ||
          (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
           (target == PIPE_TEXTURE_2D_ARRAY ||
            ps5_cube_texture_target(target) ||
            target == PIPE_TEXTURE_3D));
}

static unsigned
ps5_texture_level_layers(const struct pipe_resource *resource,
                         unsigned level)
{
   return resource->target == PIPE_TEXTURE_3D
             ? MAX2(resource->depth0 >> level, 1u)
             : resource->array_size;
}

static unsigned
ps5_linear_mip_storage_extent(unsigned extent, unsigned level)
{
   return MAX2((extent + BITFIELD_BIT(level) - 1u) >> level, 1u);
}

static bool
ps5_packed_depth_sample_layout(const struct pipe_resource *resource,
                               size_t *layer_size, unsigned *base_stride)
{
   size_t total = 0;

   if (!resource ||
       resource->format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
       !layer_size || !base_stride)
      return false;
   for (unsigned level = resource->last_level + 1; level-- > 0;) {
      const unsigned width =
         ps5_linear_mip_storage_extent(resource->width0, level);
      const unsigned height =
         ps5_linear_mip_storage_extent(resource->height0, level);
      const size_t stride = ((size_t)width * sizeof(float) + 255u) &
                            ~(size_t)255u;

      if (stride > SIZE_MAX / height ||
          total > SIZE_MAX - stride * height)
         return false;
      if (!level)
         *base_stride = (unsigned)stride;
      total += stride * height;
   }
   *layer_size = total;
   return true;
}

static bool
ps5_render_staging_required(const struct pipe_resource *resource)
{
   return PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE && resource &&
          (resource->bind & PIPE_BIND_RENDER_TARGET) &&
          ps5_color_render_target(resource->target) &&
          ps5_linear_sampled_layout(resource);
}

static bool
ps5_depth_staging_required(const struct pipe_resource *resource)
{
   return PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE && resource &&
          (resource->format == PIPE_FORMAT_Z32_FLOAT ||
           resource->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) &&
          (resource->last_level > 0 || resource->target == PIPE_TEXTURE_RECT) &&
          (resource->bind & PIPE_BIND_DEPTH_STENCIL) &&
          ps5_depth_render_target(resource->target);
}

static bool
ps5_sampled_resource_bind(const struct pipe_resource *resource)
{
   /* Mesa derives mutable texture bindings from 2D format support, so cube,
    * array, and 3D resources can carry an advisory render-target bit even
    * though this backend only exposes those targets for sampling.
    */
   if (!resource || !(resource->bind & PIPE_BIND_SAMPLER_VIEW))
      return false;
   if (!(resource->bind & ~(PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET)))
      return true;
   return (resource->format == PIPE_FORMAT_Z32_FLOAT ||
           resource->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) &&
          !(resource->bind & ~(PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_DEPTH_STENCIL));
}

static unsigned
ps5_storage_image_texel_size(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM: case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8_UINT: case PIPE_FORMAT_R8_SINT: return 1;
   case PIPE_FORMAT_R8G8_UNORM: case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8_UINT: case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R16_UNORM: case PIPE_FORMAT_R16_SNORM: case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16_UINT: case PIPE_FORMAT_R16_SINT: return 2;
   case PIPE_FORMAT_R16G16_UNORM: case PIPE_FORMAT_R16G16_SNORM: case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16_UINT: case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R11G11B10_FLOAT: case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM: case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SNORM: case PIPE_FORMAT_R8G8B8A8_SINT: return 4;
   case PIPE_FORMAT_R32_FLOAT: case PIPE_FORMAT_R32_UINT: case PIPE_FORMAT_R32_SINT: return 4;
   case PIPE_FORMAT_R32G32_FLOAT: case PIPE_FORMAT_R32G32_UINT: case PIPE_FORMAT_R32G32_SINT: return 8;
   case PIPE_FORMAT_R16G16B16A16_FLOAT: case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT: case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM: return 8;
   case PIPE_FORMAT_R32G32B32A32_FLOAT: case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT: return 16;
   default: return 0;
   }
}

static unsigned
ps5_storage_image_channels(enum pipe_format format)
{
   return ps5_storage_image_texel_size(format) ? util_format_get_nr_components(format) : 0;
}

static bool
ps5_compute_image_array_resource(const struct pipe_resource *resource)
{
   return resource && resource->target == PIPE_TEXTURE_2D_ARRAY &&
          resource->array_size > 0 && resource->array_size <= 8 &&
          resource->nr_samples <= 1 && resource->nr_storage_samples <= 1 &&
          resource->bind == (PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE) &&
          ps5_storage_image_texel_size(resource->format);
}

static unsigned
ps5_texture_format_size(enum pipe_format format)
{
   if (PS5_ENABLE_CORE_TEXTURE_FORMATS_CANDIDATE &&
       ps5_core_sampled_texture_format(format))
      return util_format_get_blocksize(format);
   switch (format) {
   case PIPE_FORMAT_R8_UNORM: return PS5_ENABLE_TEXTURE_RG_CANDIDATE ? 1 : 0;
   case PIPE_FORMAT_R8G8_UNORM: return PS5_ENABLE_TEXTURE_RG_CANDIDATE ? 2 : 0;
   case PIPE_FORMAT_R8_SNORM:
      return PS5_ENABLE_TEXTURE_SNORM_CANDIDATE ? 1 : 0;
   case PIPE_FORMAT_R8G8_SNORM:
      return PS5_ENABLE_TEXTURE_SNORM_CANDIDATE ? 2 : 0;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      return PS5_ENABLE_TEXTURE_SNORM_CANDIDATE ? 4 : 0;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      return PS5_ENABLE_TEXTURE_FLOAT_CANDIDATE ? 8 : 0;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return PS5_ENABLE_TEXTURE_FLOAT_CANDIDATE ? 16 : 0;
   case PIPE_FORMAT_R9G9B9E5_FLOAT:
      return PS5_ENABLE_SHARED_EXPONENT_CANDIDATE ? 4 : 0;
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return PS5_ENABLE_PACKED_FLOAT_CANDIDATE ? 4 : 0;
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
      return PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE &&
             PS5_ENABLE_NARROW_TEXTURE_INTEGER_CANDIDATE ? 4 : 0;
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
      return PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE &&
             PS5_ENABLE_NARROW_TEXTURE_INTEGER_CANDIDATE ? 8 : 0;
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
      return PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE ? 16 : 0;
   case PIPE_FORMAT_R10G10B10A2_UINT:
      return PS5_ENABLE_RGB10_A2UI_CANDIDATE ? 4 : 0;
   case PIPE_FORMAT_R8G8B8A8_SRGB: return PS5_ENABLE_SRGB_CANDIDATE ? 4 : 0;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_Z32_FLOAT: return 4;
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return PS5_ENABLE_PACKED_DEPTH_STENCIL ? 8 : 0;
   case PIPE_FORMAT_X32_S8X24_UINT:
      return PS5_ENABLE_GLSL_430_CANDIDATE ? 8 : 0;
   case PIPE_FORMAT_S8_UINT:
      return PS5_ENABLE_GLSL_430_CANDIDATE ? 1 : 0;
   default: return 0;
   }
}

static bool
ps5_texture_descriptor_format(enum pipe_format format, uint32_t *word1)
{
   if (PS5_ENABLE_CORE_TEXTURE_FORMATS_CANDIDATE &&
       ps5_core_sampled_texture_format(format)) {
      const struct gfx10_format *native = &gfx10_format_table[format];

      if (!native->img_format || native->buffers_only)
         return false;
      *word1 = native->img_format << 20;
      return true;
   }
   switch (format) {
   case PIPE_FORMAT_R8_UNORM: *word1 = UINT32_C(0x00100000); return true;
   case PIPE_FORMAT_R8_SNORM: *word1 = UINT32_C(0x00200000); return true;
   case PIPE_FORMAT_R8G8_UNORM: *word1 = UINT32_C(0x00e00000); return true;
   case PIPE_FORMAT_R8G8_SNORM: *word1 = UINT32_C(0x00f00000); return true;
   case PIPE_FORMAT_R8G8B8A8_UNORM: *word1 = UINT32_C(0x03800000); return true;
   case PIPE_FORMAT_R8G8B8A8_SNORM: *word1 = UINT32_C(0x03900000); return true;
   case PIPE_FORMAT_R16G16B16A16_FLOAT: *word1 = UINT32_C(0x04700000); return true;
   case PIPE_FORMAT_R32G32B32A32_FLOAT: *word1 = UINT32_C(0x04d00000); return true;
   case PIPE_FORMAT_R9G9B9E5_FLOAT: *word1 = UINT32_C(0x08400000); return true;
   case PIPE_FORMAT_R11G11B10_FLOAT: *word1 = UINT32_C(0x02400000); return true;
   case PIPE_FORMAT_R8G8B8A8_UINT: *word1 = UINT32_C(0x03c00000); return true;
   case PIPE_FORMAT_R8G8B8A8_SINT: *word1 = UINT32_C(0x03d00000); return true;
   case PIPE_FORMAT_R16G16B16A16_UINT: *word1 = UINT32_C(0x04500000); return true;
   case PIPE_FORMAT_R16G16B16A16_SINT: *word1 = UINT32_C(0x04600000); return true;
   case PIPE_FORMAT_R32G32B32A32_UINT: *word1 = UINT32_C(0x04b00000); return true;
   case PIPE_FORMAT_R32G32B32A32_SINT: *word1 = UINT32_C(0x04c00000); return true;
   case PIPE_FORMAT_R10G10B10A2_UINT: *word1 = UINT32_C(0x03600000); return true;
   case PIPE_FORMAT_R8G8B8A8_SRGB: *word1 = UINT32_C(0x08200000); return true;
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      if (PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE) {
         *word1 = UINT32_C(0x01600000);
         return true;
      }
      return false;
   case PIPE_FORMAT_X32_S8X24_UINT:
   case PIPE_FORMAT_S8_UINT:
      if (PS5_ENABLE_GLSL_430_CANDIDATE) {
         *word1 = gfx10_format_table[PIPE_FORMAT_S8_UINT].img_format << 20;
         return *word1 != 0;
      }
      return false;
   default: return false;
   }
}

static unsigned
ps5_format_swizzle(enum pipe_format format, unsigned swizzle)
{
   if (swizzle < 4 && ps5_storage_image_texel_size(format) &&
       swizzle >= ps5_storage_image_channels(format))
      return util_format_description(format)->swizzle[swizzle];
   return swizzle;
}

static bool
ps5_texture_descriptor_swizzle(unsigned swizzle, enum pipe_format format, uint32_t *selector)
{
   static const uint8_t selectors[] = {4, 5, 6, 7, 0, 1};

   swizzle = ps5_format_swizzle(format, swizzle);
   if (swizzle >= ARRAY_SIZE(selectors))
      return false;
   *selector = selectors[swizzle];
   return true;
}

static bool
ps5_texture_descriptor_wrap(unsigned wrap, uint32_t *clamp)
{
   switch (wrap) {
   case PIPE_TEX_WRAP_REPEAT: *clamp = 0; return true;
   case PIPE_TEX_WRAP_MIRROR_REPEAT: *clamp = 1; return true;
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE: *clamp = 2; return true;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
      *clamp = 3;
      return PS5_ENABLE_BORDER_COLOR_CANDIDATE;
   case PIPE_TEX_WRAP_CLAMP:
      *clamp = 4;
      return PS5_ENABLE_BORDER_COLOR_CANDIDATE;
   case PIPE_TEX_WRAP_MIRROR_CLAMP:
      *clamp = 5;
      return PS5_ENABLE_BORDER_COLOR_CANDIDATE;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      *clamp = 6;
      return PS5_ENABLE_BORDER_COLOR_CANDIDATE;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER:
      *clamp = 7;
      return PS5_ENABLE_BORDER_COLOR_CANDIDATE;
   default: return false;
   }
}

static uint32_t
ps5_texture_descriptor_anisotropy(unsigned max_anisotropy)
{
   return max_anisotropy < 2 ? 0 : max_anisotropy < 4 ? 1 :
          max_anisotropy < 8 ? 2 : max_anisotropy < 16 ? 3 : 4;
}

static bool
ps5_texture_descriptor_filter(unsigned filter, unsigned max_anisotropy,
                              uint32_t *native)
{
   switch (filter) {
   case PIPE_TEX_FILTER_NEAREST:
      *native = max_anisotropy > 1 ? 2 : 0;
      return true;
   case PIPE_TEX_FILTER_LINEAR:
      *native = max_anisotropy > 1 ? 3 : 1;
      return true;
   default: return false;
   }
}

static bool
ps5_texture_descriptor_mip_filter(unsigned filter, uint32_t *native)
{
   switch (filter) {
   case PIPE_TEX_MIPFILTER_NONE: *native = 0; return true;
   case PIPE_TEX_MIPFILTER_NEAREST: *native = 1; return true;
   case PIPE_TEX_MIPFILTER_LINEAR: *native = 2; return true;
   default: return false;
   }
}

static uint32_t
ps5_texture_descriptor_unsigned_lod(float value)
{
   if (value <= 0.0f)
      return 0;
   if (value >= 15.0f)
      return 15u << 8;
   return (uint32_t)(value * 256.0f);
}

static uint32_t
ps5_texture_descriptor_lod_bias(float value)
{
   int32_t fixed;

   if (value <= -32.0f)
      fixed = -32 * 256;
   else if (value >= 31.0f)
      fixed = 31 * 256;
   else
      fixed = (int32_t)(value * 256.0f);
   return (uint32_t)fixed & UINT32_C(0x3fff);
}

static bool
ps5_float_is_finite(float value)
{
   return (ps5_float_bits(value) & UINT32_C(0x7f800000)) !=
          UINT32_C(0x7f800000);
}

static uint32_t
ps5_pack_float_12p4(float value)
{
   if (value <= 0.0f)
      return 0;
   if (value >= 4096.0f)
      return UINT32_C(0xffff);
   return (uint32_t)(value * 16.0f);
}

static bool
ps5_dual_source_blend_factor(unsigned factor)
{
   return factor == PIPE_BLENDFACTOR_SRC1_COLOR ||
          factor == PIPE_BLENDFACTOR_INV_SRC1_COLOR ||
          factor == PIPE_BLENDFACTOR_SRC1_ALPHA ||
          factor == PIPE_BLENDFACTOR_INV_SRC1_ALPHA;
}

static bool
ps5_translate_blend_factor(unsigned factor, uint32_t *native)
{
   if (!native)
      return false;

   switch (factor) {
   case PIPE_BLENDFACTOR_ZERO: *native = 0; return true;
   case PIPE_BLENDFACTOR_ONE: *native = 1; return true;
   case PIPE_BLENDFACTOR_SRC_COLOR: *native = 2; return true;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR: *native = 3; return true;
   case PIPE_BLENDFACTOR_SRC_ALPHA: *native = 4; return true;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA: *native = 5; return true;
   case PIPE_BLENDFACTOR_DST_ALPHA: *native = 6; return true;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA: *native = 7; return true;
   case PIPE_BLENDFACTOR_DST_COLOR: *native = 8; return true;
   case PIPE_BLENDFACTOR_INV_DST_COLOR: *native = 9; return true;
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE: *native = 10; return true;
   case PIPE_BLENDFACTOR_CONST_COLOR: *native = 13; return true;
   case PIPE_BLENDFACTOR_INV_CONST_COLOR: *native = 14; return true;
   case PIPE_BLENDFACTOR_SRC1_COLOR:
      if (!PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE)
         return false;
      *native = 15;
      return true;
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      if (!PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE)
         return false;
      *native = 16;
      return true;
   case PIPE_BLENDFACTOR_SRC1_ALPHA:
      if (!PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE)
         return false;
      *native = 17;
      return true;
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      if (!PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE)
         return false;
      *native = 18;
      return true;
   case PIPE_BLENDFACTOR_CONST_ALPHA: *native = 19; return true;
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA: *native = 20; return true;
   default: return false;
   }
}

static bool
ps5_translate_blend_function(unsigned function, uint32_t *native)
{
   static const uint8_t functions[] = {
      0, /* ADD */
      1, /* SUBTRACT */
      4, /* REVERSE_SUBTRACT */
      2, /* MIN */
      3, /* MAX */
   };

   if (!native || function > PIPE_BLEND_MAX)
      return false;
   *native = functions[function];
   return true;
}

static bool
ps5_encode_rt_blend(const struct pipe_rt_blend_state *rt, uint32_t *control)
{
   uint32_t rgb_source;
   uint32_t rgb_destination;
   uint32_t rgb_function;
   uint32_t alpha_source;
   uint32_t alpha_destination;
   uint32_t alpha_function;

   if (!rt || !control)
      return false;
   *control = 0;
   if (!rt->blend_enable || !rt->colormask)
      return true;
   if (!ps5_translate_blend_factor(rt->rgb_src_factor, &rgb_source) ||
       !ps5_translate_blend_factor(rt->rgb_dst_factor, &rgb_destination) ||
       !ps5_translate_blend_function(rt->rgb_func, &rgb_function) ||
       !ps5_translate_blend_factor(rt->alpha_src_factor, &alpha_source) ||
       !ps5_translate_blend_factor(rt->alpha_dst_factor,
                                   &alpha_destination) ||
       !ps5_translate_blend_function(rt->alpha_func, &alpha_function))
      return false;

   *control = UINT32_C(1) << 30;
   *control |= rgb_source | (rgb_function << 5) | (rgb_destination << 8);
   if (alpha_source != rgb_source || alpha_destination != rgb_destination ||
       alpha_function != rgb_function) {
      *control |= UINT32_C(1) << 29;
      *control |= (alpha_source << 16) | (alpha_function << 21) |
                  (alpha_destination << 24);
   }
   return true;
}

static bool
ps5_encode_blend_state(const struct pipe_blend_state *state,
                       unsigned target_count,
                       uint32_t control[PS5_MAX_RENDER_TARGETS],
                       uint32_t *target_mask, uint32_t *color_control,
                       uint32_t *dual_source_blend)
{
   if (target_count > PS5_MAX_RENDER_TARGETS || !control ||
       !target_mask || !color_control || !dual_source_blend)
      return false;
   memset(control, 0, PS5_MAX_RENDER_TARGETS * sizeof(control[0]));
   *target_mask = 0;
   *dual_source_blend = 0;
   if (!target_count) {
      *color_control = 0;
      return true;
   }
   if (!state) {
      for (unsigned i = 0; i < target_count; ++i)
         *target_mask |= UINT32_C(0xf) << (4u * i);
      *color_control = UINT32_C(0x00cc0011);
      return true;
   }
   /* Advanced blend equations are computed by the fragment shader through
    * framebuffer fetch (KHR_blend_equation_advanced lowering); the state
    * tracker leaves fixed-function blending disabled for them. */
   if (state->max_rt >= target_count)
      return false;

   for (unsigned i = 0; i < target_count; ++i) {
      const struct pipe_rt_blend_state *rt =
         &state->rt[state->independent_blend_enable ? i : 0];
      const bool dual_source = rt->blend_enable && rt->colormask &&
         (ps5_dual_source_blend_factor(rt->rgb_src_factor) ||
          ps5_dual_source_blend_factor(rt->rgb_dst_factor) ||
          ps5_dual_source_blend_factor(rt->alpha_src_factor) ||
          ps5_dual_source_blend_factor(rt->alpha_dst_factor));

      if (dual_source &&
          (!PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE || i ||
           target_count != 1 || rt->rgb_func == PIPE_BLEND_MIN ||
           rt->rgb_func == PIPE_BLEND_MAX ||
           rt->alpha_func == PIPE_BLEND_MIN ||
           rt->alpha_func == PIPE_BLEND_MAX))
         return false;
      *dual_source_blend |= dual_source;

      *target_mask |= (uint32_t)rt->colormask << (4u * i);
      if (!ps5_encode_rt_blend(rt, &control[i]))
         return false;
   }
   *color_control = (*target_mask ? UINT32_C(0x10) : 0) |
      (((state->logicop_enable &&
         state->logicop_func != PIPE_LOGICOP_COPY) ?
           (state->logicop_func | (state->logicop_func << 4)) :
           UINT32_C(0xcc)) << 16);
   /* ponytail: keep RB+ off until matching SX conversion state is emitted.
    * This also preserves the required disable for ROP3 and dual-source blend. */
   *color_control |= 1;
   return true;
}

static bool
ps5_polygon_offset_for_fill(const struct pipe_rasterizer_state *state,
                            unsigned fill)
{
   switch (fill) {
   case PIPE_POLYGON_MODE_FILL:
      return state->offset_tri;
   case PIPE_POLYGON_MODE_LINE:
      return state->offset_line;
   case PIPE_POLYGON_MODE_POINT:
      return state->offset_point;
   default:
      return false;
   }
}

static bool
ps5_encode_rasterizer_state(const struct pipe_rasterizer_state *state,
                            uint32_t *control, uint32_t *valid)
{
   static const uint8_t fill[] = {2, 1, 0};
   bool polygon_mode;

   if (!control || !valid)
      return false;
   *control = 0;
   *valid = 0;
   if (!state)
      return true;
   if (state->cull_face > PIPE_FACE_FRONT_AND_BACK ||
       state->fill_front > PIPE_POLYGON_MODE_POINT ||
       state->fill_back > PIPE_POLYGON_MODE_POINT ||
       state->offset_units_unscaled ||
       ((state->poly_smooth || state->line_smooth) &&
        !PS5_ENABLE_SMOOTH_RASTER_CANDIDATE) ||
       state->poly_stipple_enable ||
       state->point_smooth ||
       (state->multisample && !PS5_ENABLE_MSAA4_CANDIDATE) ||
       state->line_stipple_enable || state->conservative_raster_mode)
      return false;
   if ((state->fill_front == PIPE_POLYGON_MODE_LINE ||
        state->fill_back == PIPE_POLYGON_MODE_LINE) &&
       !PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE && state->line_width != 1.0f)
      return false;
   if ((state->fill_front == PIPE_POLYGON_MODE_POINT ||
        state->fill_back == PIPE_POLYGON_MODE_POINT) &&
       !PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE &&
       (state->point_size != 1.0f || state->point_size_per_vertex))
      return false;

   polygon_mode =
      (state->fill_front != PIPE_POLYGON_MODE_FILL &&
       !(state->cull_face & PIPE_FACE_FRONT)) ||
      (state->fill_back != PIPE_POLYGON_MODE_FILL &&
       !(state->cull_face & PIPE_FACE_BACK));
   *control = (state->cull_face & PIPE_FACE_FRONT ? 1u : 0u) |
              (state->cull_face & PIPE_FACE_BACK ? 2u : 0u) |
              (!state->front_ccw ? 4u : 0u) |
              (polygon_mode ? 1u << 3 : 0u) |
              ((uint32_t)fill[state->fill_front] << 5) |
              ((uint32_t)fill[state->fill_back] << 8) |
              (ps5_polygon_offset_for_fill(state, state->fill_front) ?
                  1u << 11 : 0u) |
              (ps5_polygon_offset_for_fill(state, state->fill_back) ?
                  1u << 12 : 0u) |
              (state->offset_point || state->offset_line ?
                  1u << 13 : 0u) |
              (!state->flatshade_first ? 1u << 19 : 0u) | /* PROVOKING_VTX_LAST */
              (polygon_mode ? 1u << 24 : 0u);
   *valid = 1;
   return true;
}

static bool
ps5_encode_graphics_state(const struct ps5_context *context,
                          struct ps5_native_graphics_state *native)
{
   const struct pipe_viewport_state *viewport;
   const struct pipe_scissor_state *scissor;
   float z_extent;
   float zmin;
   float zmax;
   unsigned i;
   unsigned minx;
   unsigned miny;
   unsigned maxx;
   unsigned maxy;
   unsigned viewport_index;
   float polygon_scale;
   float point_min;
   float point_max;
   uint32_t fixed_size;

   if (!context || !native ||
       !ps5_encode_blend_state(context->blend,
                                context->framebuffer.nr_cbufs,
                                native->blend_control,
                                &native->target_mask,
                                &native->color_control,
                                &native->dual_source_blend) ||
       !ps5_encode_rasterizer_state(context->rasterizer,
                                    &native->rasterizer_control,
                                    &native->rasterizer_valid))
      return false;
   native->clip_control_valid = context->rasterizer != NULL;
   native->clip_control = context->rasterizer
      ? ((!context->rasterizer->depth_clip_near ? 1u << 26 : 0u) |
         (!context->rasterizer->depth_clip_far ? 1u << 27 : 0u) |
         (context->rasterizer->rasterizer_discard ? 1u << 22 : 0u) |
         (context->rasterizer->clip_halfz ? 1u << 19 : 0u) |
         (1u << 24))
      : 0;
   native->point_line[0] = 0;
   native->point_line[1] = 0;
   native->point_line[2] = 0;
   native->point_line_valid = 0;
   if (PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE && context->rasterizer) {
      /* Mesa's clear/meta triangle rasterizers leave these unused sizes zero.
       * Program safe defaults without rejecting the unrelated triangle draw. */
      const float point_size = context->rasterizer->point_size == 0.0f
                                  ? 1.0f : context->rasterizer->point_size;
      const float line_width = context->rasterizer->line_width == 0.0f
                                  ? 1.0f : context->rasterizer->line_width;
      if (!ps5_float_is_finite(point_size) ||
          !ps5_float_is_finite(line_width) ||
          point_size < PS5_MIN_POINT_LINE_SIZE ||
          point_size > PS5_MAX_POINT_LINE_SIZE ||
          line_width < PS5_MIN_POINT_LINE_SIZE ||
          line_width > PS5_MAX_POINT_LINE_SIZE)
         return false;
      fixed_size = (uint32_t)(point_size * 8.0f);
      native->point_line[0] = fixed_size | (fixed_size << 16);
      if (context->rasterizer->point_size_per_vertex) {
         point_min = PS5_MIN_POINT_LINE_SIZE;
         point_max = PS5_MAX_POINT_LINE_SIZE;
      } else {
         point_min = point_size;
         point_max = point_size;
      }
      native->point_line[1] = ps5_pack_float_12p4(point_min * 0.5f) |
         (ps5_pack_float_12p4(point_max * 0.5f) << 16);
      native->point_line[2] =
         ps5_pack_float_12p4(line_width * 0.5f);
      native->point_line_valid = 1;
   }
   native->interp_control = 0;
   native->interp_control_valid = 0;
   if (PS5_ENABLE_POINT_COORD_CANDIDATE && context->rasterizer) {
      native->interp_control =
         1u | /* FLAT_SHADE_ENA */
         (context->rasterizer->point_quad_rasterization ? 1u << 1 : 0u) |
         (2u << 2) |  /* point X selects S */
         (3u << 5) |  /* point Y selects T */
         (1u << 11) | /* point W selects 1 */
         (context->rasterizer->sprite_coord_mode !=
               PIPE_SPRITE_COORD_UPPER_LEFT ? 1u << 14 : 0u);
      native->interp_control_valid = 1;
   }
   native->color_control_valid = 1;
   native->alpha_to_coverage =
      context->blend && context->blend->alpha_to_coverage;
   native->alpha_to_one = context->blend && context->blend->alpha_to_one;
   for (i = 0; i < context->framebuffer.nr_cbufs; ++i) {
      if (!context->framebuffer.cbufs[i].texture)
         native->target_mask &= ~(UINT32_C(0xf) << (4u * i));
   }
   native->color_control = (native->color_control & ~UINT32_C(0x10)) |
                           (native->target_mask ? UINT32_C(0x10) : 0);
   for (i = 0; i < 4; ++i)
      native->blend_color[i] = ps5_float_bits(context->blend_color.color[i]);
   native->polygon_offset_valid = 0;
   for (i = 0; i < 6; ++i)
      native->polygon_offset[i] = 0;
   if (context->rasterizer &&
       (context->rasterizer->offset_point ||
        context->rasterizer->offset_line ||
        context->rasterizer->offset_tri)) {
      polygon_scale = context->rasterizer->offset_scale * 16.0f;

      /* GFX10 D32F: -23 depth bits plus floating-point format. */
      native->polygon_offset[0] = UINT32_C(0x000001e9);
      native->polygon_offset[1] =
         ps5_float_bits(context->rasterizer->offset_clamp);
      native->polygon_offset[2] = ps5_float_bits(polygon_scale);
      native->polygon_offset[3] =
         ps5_float_bits(context->rasterizer->offset_units);
      native->polygon_offset[4] = native->polygon_offset[2];
      native->polygon_offset[5] = native->polygon_offset[3];
      native->polygon_offset_valid =
         context->framebuffer.zsbuf.texture != NULL;
   }

   for (viewport_index = 0; viewport_index < PS5_MAX_VIEWPORTS;
        ++viewport_index) {
      const unsigned source =
         context->viewport_valid & (UINT16_C(1) << viewport_index)
            ? viewport_index : 0;

      if (context->viewport_valid & (UINT16_C(1) << source)) {
         viewport = &context->viewport[source];
         for (i = 0; i < 3; ++i) {
            if (!ps5_float_is_finite(viewport->scale[i]) ||
                !ps5_float_is_finite(viewport->translate[i]))
               return false;
            native->viewport[viewport_index][2 * i] =
               ps5_float_bits(viewport->scale[i]);
            native->viewport[viewport_index][2 * i + 1] =
               ps5_float_bits(viewport->translate[i]);
         }
         z_extent = viewport->scale[2] < 0.0f ? -viewport->scale[2] :
                                                      viewport->scale[2];
         zmin = viewport->translate[2] - z_extent;
         zmax = viewport->translate[2] + z_extent;
         native->viewport[viewport_index][6] = ps5_float_bits(zmin);
         native->viewport[viewport_index][7] = ps5_float_bits(zmax);
      } else {
         native->viewport[viewport_index][0] = UINT32_C(0x44700000);
         native->viewport[viewport_index][1] = UINT32_C(0x44700000);
         native->viewport[viewport_index][2] = UINT32_C(0xc4070000);
         native->viewport[viewport_index][3] = UINT32_C(0x44070000);
         native->viewport[viewport_index][4] = UINT32_C(0x3f800000);
         native->viewport[viewport_index][5] = 0;
         native->viewport[viewport_index][6] = 0;
         native->viewport[viewport_index][7] = UINT32_C(0x3f800000);
      }

      scissor = context->rasterizer && context->rasterizer->scissor &&
                (context->scissor_valid &
                 (UINT16_C(1) << viewport_index))
                   ? &context->scissor[viewport_index] : NULL;
      minx = scissor ? MIN2(scissor->minx, context->framebuffer.width) : 0;
      miny = scissor ? MIN2(scissor->miny, context->framebuffer.height) : 0;
      maxx = scissor ? MIN2(scissor->maxx, context->framebuffer.width)
                    : context->framebuffer.width;
      maxy = scissor ? MIN2(scissor->maxy, context->framebuffer.height)
                    : context->framebuffer.height;
      if (maxx < minx)
         maxx = minx;
      if (maxy < miny)
         maxy = miny;
      native->scissor[viewport_index][0] =
         UINT32_C(0x80000000) | minx | (miny << 16);
      native->scissor[viewport_index][1] = maxx | (maxy << 16);
   }
   return true;
}

static bool
ps5_stencil_uses_unit_op_value(const struct pipe_stencil_state *state)
{
   /* GFX10 ADD/SUB stencil operations consume STENCILOPVAL. */
   const unsigned add_sub_ops =
      BITFIELD_BIT(PIPE_STENCIL_OP_INCR) |
      BITFIELD_BIT(PIPE_STENCIL_OP_DECR) |
      BITFIELD_BIT(PIPE_STENCIL_OP_INCR_WRAP) |
      BITFIELD_BIT(PIPE_STENCIL_OP_DECR_WRAP);

   return (add_sub_ops & BITFIELD_BIT(state->fail_op)) ||
          (add_sub_ops & BITFIELD_BIT(state->zpass_op)) ||
          (add_sub_ops & BITFIELD_BIT(state->zfail_op));
}

static bool
ps5_encode_depth_stencil_state(
   const struct pipe_depth_stencil_alpha_state *dsa,
   const struct pipe_stencil_ref *ref,
   struct ps5_native_depth_stencil_state *native)
{
   static const uint8_t stencil_op[] = {
      0, /* KEEP */
      1, /* ZERO */
      3, /* REPLACE_TEST */
      5, /* ADD_CLAMP */
      6, /* SUB_CLAMP */
      8, /* ADD_WRAP */
      9, /* SUB_WRAP */
      7, /* INVERT */
   };
   const struct pipe_stencil_state *front;
   const struct pipe_stencil_state *back;

   if (!dsa || !ref || !native || dsa->alpha_enabled ||
       dsa->depth_bounds_test || dsa->depth_func > PIPE_FUNC_ALWAYS)
      return false;
   front = &dsa->stencil[0];
   back = dsa->stencil[1].enabled ? &dsa->stencil[1] : front;
   if ((!front->enabled && dsa->stencil[1].enabled) ||
       front->func > PIPE_FUNC_ALWAYS || back->func > PIPE_FUNC_ALWAYS ||
       front->fail_op > PIPE_STENCIL_OP_INVERT ||
       front->zpass_op > PIPE_STENCIL_OP_INVERT ||
       front->zfail_op > PIPE_STENCIL_OP_INVERT ||
       back->fail_op > PIPE_STENCIL_OP_INVERT ||
       back->zpass_op > PIPE_STENCIL_OP_INVERT ||
       back->zfail_op > PIPE_STENCIL_OP_INVERT)
      return false;

   memset(native, 0, sizeof(*native));
   if (dsa->depth_enabled)
      native->depth_control = (dsa->depth_func << 4) | 2u |
                              (dsa->depth_writemask ? 4u : 0u);
   if (!front->enabled)
      return true;

   native->depth_control |= 1u | (front->func << 8);
   native->stencil_control =
      stencil_op[front->fail_op] |
      (stencil_op[front->zpass_op] << 4) |
      (stencil_op[front->zfail_op] << 8);
   native->stencil_refmask = ref->ref_value[0] |
      (front->valuemask << 8) | (front->writemask << 16) |
      (ps5_stencil_uses_unit_op_value(front) ? UINT32_C(1) << 24 : 0);
   if (dsa->stencil[1].enabled) {
      native->depth_control |= 1u << 7;
      native->depth_control |= back->func << 20;
      native->stencil_control |=
         (stencil_op[back->fail_op] << 12) |
         (stencil_op[back->zpass_op] << 16) |
         (stencil_op[back->zfail_op] << 20);
      native->stencil_refmask_bf = ref->ref_value[1] |
         (back->valuemask << 8) | (back->writemask << 16) |
         (ps5_stencil_uses_unit_op_value(back) ? UINT32_C(1) << 24 : 0);
   } else {
      native->stencil_refmask_bf = native->stencil_refmask;
   }
   return true;
}

static uint32_t
ps5_hash32(const void *data, size_t size)
{
   const uint8_t *bytes = data;
   uint32_t hash = UINT32_C(2166136261);
   size_t i;

   for (i = 0; i < size; ++i)
      hash = (hash ^ bytes[i]) * UINT32_C(16777619);
   return hash;
}

#ifdef PS5_DRAW_PROFILE
static uint64_t ps5_cpu_flush_calls, ps5_cpu_flush_bytes;
#endif

static void
ps5_flush_gpu_data(const void *address, size_t bytes)
{
   if (!bytes)
      return;
   util_flush_inval_range((void *)address, bytes);
#ifdef PS5_DRAW_PROFILE
   __atomic_fetch_add(&ps5_cpu_flush_calls, 1, __ATOMIC_RELAXED);
   __atomic_fetch_add(&ps5_cpu_flush_bytes, bytes, __ATOMIC_RELAXED);
#endif
}

#include "../../platform/ps5_cache_profile.h"
#ifdef PS5_DRAW_PROFILE
#define ps5_flush_gpu_data(address, bytes) PS5_CACHE_MEASURE(ps5_flush_gpu_data, address, bytes)
#endif

/* A draw may publish the completed private table after its last writer.
 * Resource-data publication is separate and must never be deferred here. */
static void
ps5_publish_descriptor_prefix(const void *data, size_t bytes, size_t *pending)
{
   if (pending)
      *pending = MAX2(*pending, bytes);
   else
      ps5_flush_gpu_data(data, bytes);
}

static bool
ps5_texel_buffer_descriptor(const struct pipe_sampler_view *view,
                            uint32_t address32_hi, uint32_t descriptor[4])
{
   const struct ps5_resource *resource = view && view->texture ?
      (const struct ps5_resource *)view->texture : NULL;
   const unsigned texel_size = view ? util_format_get_blocksize(view->format) : 0;
   const size_t offset = view ? view->u.buf.offset : 0;
   const size_t size = view ? view->u.buf.size : 0;
   struct ac_buffer_state state;

   if (!PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE || !resource ||
       resource->base.target != PIPE_BUFFER || view->target != PIPE_BUFFER ||
       !ps5_texel_buffer_format(view->format) || !texel_size ||
       offset > resource->size || size > resource->size - offset ||
       /* RGB32 ranges need component alignment, not 12-byte alignment.
        * The descriptor count below excludes any incomplete trailing texel. */
       (offset % (texel_size == 12 ? 4 : texel_size)) ||
       size / texel_size > PS5_MAX_TEXEL_BUFFER_ELEMENTS ||
       (uint32_t)(((uintptr_t)resource->data + offset) >> 32) != address32_hi)
      return false;
   memset(&state, 0, sizeof(state));
   state.va = (uintptr_t)resource->data + offset;
   state.size = size / texel_size;
   state.format = view->format;
   state.swizzle[0] = ps5_format_swizzle(view->format, view->swizzle_r);
   state.swizzle[1] = ps5_format_swizzle(view->format, view->swizzle_g);
   state.swizzle[2] = ps5_format_swizzle(view->format, view->swizzle_b);
   state.swizzle[3] = ps5_format_swizzle(view->format, view->swizzle_a);
   state.stride = texel_size;
   state.gfx10_oob_select = V_008F0C_OOB_SELECT_STRUCTURED_WITH_OFFSET;
   state.has_desc_resource_level = true;
   ac_build_buffer_descriptor(GFX10_3, &state, descriptor);
   return true;
}

static bool
ps5_image_buffer_descriptor(const struct pipe_image_view *view,
                            uint32_t address32_hi, uint32_t descriptor[8])
{
   const struct util_format_description *format =
      view ? util_format_description(view->format) : NULL;
   struct pipe_sampler_view sampled = {
      .texture = view ? view->resource : NULL,
      .format = view ? view->format : PIPE_FORMAT_NONE,
      .target = PIPE_BUFFER,
      .swizzle_r = format ? format->swizzle[0] : PIPE_SWIZZLE_NONE,
      .swizzle_g = format ? format->swizzle[1] : PIPE_SWIZZLE_NONE,
      .swizzle_b = format ? format->swizzle[2] : PIPE_SWIZZLE_NONE,
      .swizzle_a = format ? format->swizzle[3] : PIPE_SWIZZLE_NONE,
   };

   if (!view || !view->resource || view->resource->target != PIPE_BUFFER)
      return false;
   sampled.u.buf.offset = view->u.buf.offset;
   sampled.u.buf.size = view->u.buf.size;
   memset(descriptor, 0, 8 * sizeof(*descriptor));
   return ps5_texel_buffer_descriptor(&sampled, address32_hi, descriptor);
}

int
ps5_resource_texel_buffer_descriptor_owned(struct pipe_resource *base,
                                            const uint32_t descriptor[4])
{
   const struct ps5_resource *resource = (const struct ps5_resource *)base;
   const uint64_t address = descriptor ? descriptor[0] |
      ((uint64_t)(descriptor[1] & UINT32_C(0xffff)) << 32) : 0;
   const unsigned stride = descriptor ? descriptor[1] >> 16 : 0;
   const uint64_t bytes = descriptor ? (uint64_t)descriptor[2] * stride : 0;
   const uintptr_t start = resource ? (uintptr_t)resource->data : 0;

   if (!resource || !descriptor || base->target != PIPE_BUFFER ||
       (stride != 1 && stride != 2 && stride != 4 && stride != 8 &&
        stride != 12 && stride != 16) || !descriptor[2] ||
       descriptor[2] > PS5_MAX_TEXEL_BUFFER_ELEMENTS ||
       !(descriptor[3] & UINT32_C(0x01000000)) || address < start ||
       address - start > resource->size ||
       bytes > resource->size - (address - start))
      return -1;
   return 0;
}

struct ps5_batch_flush_cache {
   const void *data[3 + PS5_MAX_TEXTURE_UNITS + PIPE_MAX_ATTRIBS];
   size_t size[3 + PS5_MAX_TEXTURE_UNITS + PIPE_MAX_ATTRIBS];
};

static void
ps5_flush_batch_backing(struct ps5_batch_flush_cache *batch, unsigned slot,
                         const void *data, size_t bytes)
{
   /* Only unsubmitted, retained batch resources may reuse a CPU flush. Ordinary
    * CPU access drains the batch; unsynchronized writes publish at the write.
    * Persistent mappings bypass this cache because writes have no callback.
    * ponytail: one published interval per plane/texture unit, reset at batch drain. */
#ifdef PS5_GPU_PRESENT_BATCH
   if (batch && data && bytes && batch->data[slot] && batch->size[slot]) {
      uintptr_t start = (uintptr_t)data, old = (uintptr_t)batch->data[slot];
      if (bytes <= UINTPTR_MAX - start && batch->size[slot] <= UINTPTR_MAX - old) {
         uintptr_t end = start + bytes, old_end = old + batch->size[slot];
         if (start <= old_end && old <= end) {
            if (start < old)
               ps5_flush_gpu_data(data, old - start);
            if (end > old_end)
               ps5_flush_gpu_data((const void *)old_end, end - old_end);
            uintptr_t first = start < old ? start : old;
            batch->data[slot] = (const void *)first;
            batch->size[slot] = (end > old_end ? end : old_end) - first;
            return;
         }
      }
   }
#else
   (void)batch;
   (void)slot;
#endif
#ifdef PS5_DRAW_PROFILE
   /* Keep distinct call sites in the cache report: an uncached persistent
    * vertex upload needs a different fix from a retained texture publication. */
   if (slot < 2) {
      if (batch) ps5_flush_gpu_data(data, bytes); /* retained depth */
      else ps5_flush_gpu_data(data, bytes); /* uncached depth */
   } else if (slot < 2 + PS5_MAX_TEXTURE_UNITS) {
      if (batch) ps5_flush_gpu_data(data, bytes); /* retained texture */
      else ps5_flush_gpu_data(data, bytes); /* uncached texture */
   } else if (slot < 2 + PS5_MAX_TEXTURE_UNITS + PIPE_MAX_ATTRIBS) {
      if (batch) ps5_flush_gpu_data(data, bytes); /* retained vertex */
      else ps5_flush_gpu_data(data, bytes); /* uncached vertex */
   } else {
      if (batch) ps5_flush_gpu_data(data, bytes); /* retained index */
      else ps5_flush_gpu_data(data, bytes); /* uncached index */
   }
#else
   ps5_flush_gpu_data(data, bytes);
#endif
#ifdef PS5_GPU_PRESENT_BATCH
   if (batch) {
      batch->data[slot] = data;
      batch->size[slot] = bytes;
   }
#endif
}

static void
ps5_flush_texture_backing(struct ps5_batch_flush_cache *batch, unsigned slot,
                           struct ps5_resource *texture, size_t bytes, bool stencil)
{
   uint64_t *epoch = stencil ? &texture->stencil_publication_epoch : &texture->texture_publication_epoch;
   size_t *published = stencil ? &texture->stencil_published_bytes : &texture->texture_published_bytes;
   const void *data = stencil ? texture->stencil_data : texture->data;
   /* Publication belongs to the resource, not a draw batch or shader stage.
    * CPU access invalidates it through the resource drain. Raw-pointer exports,
    * persistent mappings and CPU-staged depth keep their per-use publication. */
   const bool reusable =
#ifndef PS5_DEFERRED_DRAW_BATCH
      batch &&
#endif
      !texture->external_cpu_access &&
      texture->base.target != PIPE_BUFFER &&
      !(texture->base.bind & PIPE_BIND_DISPLAY_TARGET) &&
      !texture->depth_staging_size;
   if (reusable && *epoch == ps5_texture_publication_epoch &&
       *published >= bytes)
      return;
   ps5_flush_batch_backing(texture->external_cpu_access ? NULL : batch, slot, data, bytes);
   if (reusable) {
      *epoch = ps5_texture_publication_epoch;
      *published = bytes;
   }
}

static bool
ps5_stage_packed_depth_samples(struct ps5_resource *resource,
                               unsigned *base_stride)
{
   const unsigned layers = ps5_texture_level_layers(&resource->base, 0);
   uint8_t *staging = resource->data + resource->depth_staging_offset;
   size_t sample_layer_size;
   size_t sample_size;
   size_t sample_level_offset = 0;

   if (!ps5_packed_depth_sample_layout(&resource->base,
                                       &sample_layer_size, base_stride) ||
       sample_layer_size > SIZE_MAX / layers)
      return false;
   sample_size = sample_layer_size * layers;
   if (!resource->depth_staging_size ||
       sample_size > resource->depth_staging_size ||
       resource->depth_staging_offset >= resource->allocation_size ||
       resource->depth_staging_size >
          resource->allocation_size - resource->depth_staging_offset)
      return false;

   for (unsigned level = resource->base.last_level + 1; level-- > 0;) {
      const unsigned width = MAX2(resource->base.width0 >> level, 1u);
      const unsigned height = MAX2(resource->base.height0 >> level, 1u);
      const unsigned storage_width =
         ps5_linear_mip_storage_extent(resource->base.width0, level);
      const unsigned storage_height =
         ps5_linear_mip_storage_extent(resource->base.height0, level);
      const size_t sample_stride =
         ((size_t)storage_width * sizeof(float) + 255u) & ~(size_t)255u;

      for (unsigned layer = 0; layer < layers; ++layer) {
         const size_t source_base = (size_t)layer * resource->layer_stride +
                                    resource->level_offset[level];
         const size_t sample_base = (size_t)layer * sample_layer_size +
                                    sample_level_offset;

         for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
               const size_t source = source_base +
                  (size_t)y * resource->level_stride[level] +
                  (size_t)x * 8u;
               const size_t destination = sample_base +
                  (size_t)y * sample_stride + (size_t)x * sizeof(float);

               if (source > resource->size ||
                   resource->size - source < sizeof(float) ||
                   destination > sample_size ||
                   sample_size - destination < sizeof(float))
                  return false;
               memcpy(staging + destination, resource->data + source,
                      sizeof(float));
            }
         }
      }
      sample_level_offset += sample_stride * storage_height;
   }
   ps5_flush_gpu_data(staging, sample_size);
   return true;
}

static size_t ps5_tiled_stencil_msaa4_offset(unsigned x, unsigned y,
                                              unsigned sample,
                                              unsigned width,
                                              unsigned layer);
static size_t ps5_tiled_stencil_offset(unsigned x, unsigned y,
                                        unsigned width, unsigned layer);
static size_t ps5_tiled_color_msaa4_offset(enum pipe_format format,
                                            unsigned x, unsigned y,
                                            unsigned sample,
                                            unsigned width,
                                            unsigned layer);

static struct ps5_resource *
ps5_stage_packed_stencil_samples(struct ps5_resource *resource)
{
   const unsigned layers = ps5_texture_level_layers(&resource->base, 0);
   const bool multisampled = resource->base.nr_samples == 4 &&
                             resource->base.nr_storage_samples == 4;
   if (!resource->stencil_sample) {
      struct pipe_resource templ = resource->base;
      templ.format = PIPE_FORMAT_R8_UINT;
      templ.bind = PIPE_BIND_SAMPLER_VIEW |
                   (multisampled ? PIPE_BIND_RENDER_TARGET : 0);
      templ.flags = 0;
      if (!multisampled) {
         templ.nr_samples = 0;
         templ.nr_storage_samples = 0;
      }
      resource->stencil_sample = resource->base.screen->resource_create(
         resource->base.screen, &templ);
      if (!resource->stencil_sample)
         return NULL;
   }
   struct ps5_resource *sample =
      (struct ps5_resource *)resource->stencil_sample;
   if ((resource->base.bind & PIPE_BIND_DEPTH_STENCIL) &&
       !resource->depth_staging_size) {
      const unsigned samples = multisampled ? 4 : 1;
      const size_t source_layer = ps5_tiled_stencil_surface_size_samples(
         resource->base.width0, resource->base.height0, samples);

      for (unsigned layer = 0; layer < layers; ++layer) {
         for (unsigned y = 0; y < resource->base.height0; ++y) {
            for (unsigned x = 0; x < resource->base.width0; ++x) {
               for (unsigned sample_index = 0; sample_index < samples;
                    ++sample_index) {
                  const size_t source = (size_t)layer * source_layer +
                     (multisampled
                        ? ps5_tiled_stencil_msaa4_offset(
                             x, y, sample_index, resource->base.width0, layer)
                        : ps5_tiled_stencil_offset(
                             x, y, resource->base.width0, layer));
                  const size_t sample_base =
                     (size_t)layer * sample->layer_stride;
                  const size_t destination = sample_base +
                     (multisampled
                        ? ps5_tiled_color_msaa4_offset(
                             PIPE_FORMAT_R8_UINT, x, y, sample_index,
                             resource->base.width0, layer)
                        : (size_t)y * sample->level_stride[0] + x);

                  if (source >= resource->stencil_allocation_size ||
                      destination >= sample->allocation_size)
                     return NULL;
                  sample->data[destination] = resource->stencil_data[source];
               }
            }
         }
      }
      ps5_flush_gpu_data(sample->data, sample->allocation_size);
      return sample;
   }
   for (unsigned level = 0; level <= resource->base.last_level; ++level) {
      const unsigned width = MAX2(resource->base.width0 >> level, 1u);
      const unsigned height = MAX2(resource->base.height0 >> level, 1u);

      for (unsigned layer = 0; layer < layers; ++layer) {
         const size_t source_base = (size_t)layer * resource->layer_stride +
                                    resource->level_offset[level];
         const size_t sample_base = (size_t)layer * sample->layer_stride +
                                    sample->level_offset[level];

         for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
               const size_t source = source_base +
                  (size_t)y * resource->level_stride[level] +
                  (size_t)x * 8u + sizeof(float);
               const size_t destination = sample_base +
                  (size_t)y * sample->level_stride[level] + x;

               if (source >= resource->size || destination >= sample->size)
                  return NULL;
               sample->data[destination] = resource->data[source];
            }
         }
      }
   }
   ps5_flush_gpu_data(sample->data, sample->size);
   return sample;
}

#if PS5_PUBLIC_TEXTURE_RG_TILE_TEST
static bool
ps5_record_public_rg_tile(struct ps5_context *context)
{
   static const struct {
      enum pipe_format format;
      unsigned width;
      unsigned height;
      const char *name;
      const char *path;
   } cases[] = {
      { PIPE_FORMAT_R8_UNORM, 256, 256, "r8-x",
        "/data/VdecHello/opengl33-public-r8-x.raw" },
      { PIPE_FORMAT_R8_UNORM, 256, 256, "r8-y",
        "/data/VdecHello/opengl33-public-r8-y.raw" },
      { PIPE_FORMAT_R8G8_UNORM, 256, 128, "rg8-xy",
        "/data/VdecHello/opengl33-public-rg8-xy.raw" },
   };
   const unsigned index = context->draw_calls - 1;
   struct ps5_resource *target;
   FILE *file;
   size_t written = 0;
   size_t nonzero = 0;
   int close_status = -1;

   if (index >= ARRAY_SIZE(cases) ||
       !context->framebuffer.cbufs[0].texture)
      return false;
   target = (struct ps5_resource *)context->framebuffer.cbufs[0].texture;
   if (target->base.format != cases[index].format ||
       target->base.width0 != cases[index].width ||
       target->base.height0 != cases[index].height ||
       target->allocation_size < UINT32_C(0x10000))
      return false;

   ps5_flush_gpu_data(target->data, UINT32_C(0x10000));
   for (size_t i = 0; i < UINT32_C(0x10000); ++i)
      nonzero += target->data[i] != 0;
   file = fopen(cases[index].path, "wb");
   if (file) {
      written = fwrite(target->data, 1, UINT32_C(0x10000), file);
      close_status = fclose(file);
   }
   printf("[ps5-gallium] rg-tile-evidence case=%s format=%u size=%ux%u "
          "bytes=%zu hash=%08x nonzero=%zu dump=%d\n",
          cases[index].name, target->base.format, target->base.width0,
          target->base.height0, written,
          ps5_hash32(target->data, UINT32_C(0x10000)), nonzero,
          written == UINT32_C(0x10000) && close_status == 0 ? 0 : -1);
   return written == UINT32_C(0x10000) && close_status == 0;
}
#endif

#ifdef PS5_PUBLIC_STENCIL_TEST
static bool
ps5_record_public_stencil(struct ps5_resource *depth)
{
   size_t depth_nonzero = 0;
   size_t stencil_matches = 0;
   size_t stencil_unexpected = 0;
   int depth_dump_status = -1;
   int stencil_dump_status = -1;
   FILE *file;

   if (!depth ||
       depth->base.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
       !depth->data || !depth->stencil_data) {
      printf("[ps5-gallium] public-stencil-evidence invalid-resource\n");
      return false;
   }

   for (size_t i = 0;
        i < depth->allocation_size / sizeof(uint32_t); ++i)
      depth_nonzero += ((const uint32_t *)depth->data)[i] != 0;
   for (size_t i = 0; i < depth->stencil_allocation_size; ++i) {
      if (depth->stencil_data[i] == UINT8_C(0x5a))
         stencil_matches++;
      else if (depth->stencil_data[i])
         stencil_unexpected++;
   }

   file = fopen("/data/VdecHello/opengl33-public-stencil-depth.raw", "wb");
   if (file) {
      const size_t written = fwrite(
         depth->data, 1, depth->allocation_size, file);
      const int close_status = fclose(file);
      depth_dump_status =
         written == depth->allocation_size && close_status == 0 ? 0 : -1;
   }
   file = fopen("/data/VdecHello/opengl33-public-stencil.raw", "wb");
   if (file) {
      const size_t written = fwrite(
         depth->stencil_data, 1, depth->stencil_allocation_size, file);
      const int close_status = fclose(file);
      stencil_dump_status = written == depth->stencil_allocation_size &&
                            close_status == 0 ? 0 : -1;
   }

   printf("[ps5-gallium] public-stencil-evidence depth=%p/%zu nonzero=%zu dump=%d stencil=%p/%zu matches=%zu unexpected=%zu ref=5a dump=%d\n",
          depth->data, depth->allocation_size, depth_nonzero,
          depth_dump_status, depth->stencil_data,
          depth->stencil_allocation_size, stencil_matches,
          stencil_unexpected, stencil_dump_status);
   return depth_nonzero == 0 && stencil_matches == 16320 &&
          stencil_unexpected == 0 && depth_dump_status == 0 &&
          stencil_dump_status == 0;
}
#endif

static bool
ps5_uses_merged_geometry_metadata(const struct ps5_context *context,
                                  const struct ps5_shader *shader,
                                  const PsbcShaderMetadata *metadata)
{
   return context->gs && shader == context->vs &&
          (metadata == &context->geometry_output.metadata ||
           metadata == &context->geometry_streamout_output.metadata);
}

static bool
ps5_uses_tessellation_metadata(const struct ps5_context *context,
                               const PsbcShaderMetadata *metadata)
{
   return metadata && (metadata == &context->tessellation_output.hs.metadata ||
                       metadata == &context->tessellation_output.tes.metadata);
}

static bool
ps5_texture_used(const struct ps5_context *context,
                 const struct ps5_shader *shader,
                 const PsbcShaderMetadata *metadata, unsigned unit)
{
   if (ps5_uses_tessellation_metadata(context, metadata)) {
      const struct ps5_shader *stages[] = {context->vs, context->tcs, context->tes, context->gs};
      if (unit < PS5_TESSELLATION_TEXTURE_BINDING ||
          unit >= PS5_TESSELLATION_TEXTURE_BINDING + 4u * PS5_MAX_TEXTURE_UNITS)
         return false;
      unit -= PS5_TESSELLATION_TEXTURE_BINDING;
      const struct ps5_shader *stage = stages[unit / PS5_MAX_TEXTURE_UNITS];
      return stage && BITSET_TEST(stage->nir->info.textures_used, unit % PS5_MAX_TEXTURE_UNITS);
   }
   if (unit < PS5_MAX_TEXTURE_UNITS)
      return BITSET_TEST(shader->nir->info.textures_used, unit);
   return unit < PS5_MERGED_TEXTURE_UNITS &&
      ps5_uses_merged_geometry_metadata(context, shader, metadata) &&
      BITSET_TEST(context->gs->nir->info.textures_used,
                  unit - PS5_MAX_TEXTURE_UNITS);
}

static unsigned
ps5_shader_texture_count(const struct ps5_shader *shader)
{
   /* Only the first 16 units belong to one stage's descriptor bank. */
   STATIC_ASSERT(PS5_MAX_TEXTURE_UNITS == 16);
   return shader ? util_bitcount(shader->nir->info.textures_used[0] & 0xffffu) : 0;
}

static unsigned
ps5_texture_count(const struct ps5_context *context,
                  const struct ps5_shader *shader,
                  const PsbcShaderMetadata *metadata)
{
   if (ps5_uses_tessellation_metadata(context, metadata))
      return ps5_shader_texture_count(context->vs) +
             ps5_shader_texture_count(context->tcs) +
             ps5_shader_texture_count(context->tes) +
             ps5_shader_texture_count(context->gs);
   return ps5_shader_texture_count(shader) +
      (ps5_uses_merged_geometry_metadata(context, shader, metadata)
         ? ps5_shader_texture_count(context->gs) : 0);
}

static unsigned
ps5_constant_state_binding(const struct ps5_shader *shader,
                           unsigned ubo_index)
{
   return ubo_index + !shader->nir->info.first_ubo_is_default_ubo;
}

static size_t
ps5_copied_constant_offset(unsigned state_slot)
{
   return PS5_CONSTANT_DATA_OFFSET +
          (state_slot >= PS5_GEOMETRY_CONSTANT_SLOT
              ? (state_slot - 1u) * PS5_MAX_CONSTANT_BUFFER_SIZE : 0u);
}

/* Snapshots need only the ordinary descriptor bank and copied CB0. Geometry
 * and tessellation retain their full multi-stage layout. */
static size_t
ps5_descriptor_snapshot_size(const struct ps5_context *context, unsigned slot)
{
   if (slot == 0 && (context->gs || context->tcs || context->tes))
      return PS5_ENABLE_UBO_CANDIDATE ? PS5_DESCRIPTOR_STORAGE_BYTES
                                      : PS5_DIRECT_ALIGNMENT;
   const struct ps5_constant_state *state = &context->constants[slot][0];
   return state->valid && state->copied
      ? MAX2(PS5_CONSTANT_DATA_OFFSET,
             PS5_CONSTANT_DATA_OFFSET + ((state->size + 15u) & ~15u))
      : PS5_CONSTANT_DATA_OFFSET;
}

static unsigned
ps5_shader_storage_count(const struct ps5_shader *shader)
{
   return shader ? shader->nir->info.num_ssbos : 0;
}

static bool
ps5_shader_uses_storage(const struct ps5_shader *shader)
{
   return shader && (shader->nir->info.num_ssbos || shader->nir->info.num_images);
}


static bool
ps5_image_view_incomplete(const struct pipe_image_view *view)
{
   return !view || !view->resource ||
      (view->resource->target != PIPE_BUFFER &&
       view->u.tex.level > view->resource->last_level);
}

static bool
ps5_prepare_preraster_images(struct ps5_context *context,
                             const struct ps5_shader *shader,
                             const PsbcShaderMetadata *metadata,
                             uint32_t *user_data, unsigned user_data_count,
                             unsigned expected_offset)
{
   const unsigned count = shader ? shader->nir->info.num_images : 0;
   const unsigned stage = shader ? shader->nir->info.stage : MESA_SHADER_STAGES;
   struct ps5_resource *table =
      (struct ps5_resource *)context->descriptor_storage[0];
   const PsbcDescriptorBinding *bank = NULL;

   if (!count)
      return true;
   if (stage > MESA_SHADER_GEOMETRY || count > PS5_COMPUTE_IMAGE_SLOTS ||
       context->preraster_images_invalid[stage] || !table || !table->data ||
       table->size < expected_offset + PS5_COMPUTE_IMAGE_SLOTS * 32u ||
       !metadata->descriptor_set0_valid ||
       metadata->descriptor_set0_user_data_dword >= user_data_count ||
       (uintptr_t)table->data >> 32 != metadata->address32_hi)
      return false;
   for (unsigned i = 0; i < metadata->descriptor_binding_count; ++i) {
      const PsbcDescriptorBinding *candidate = &metadata->descriptor_bindings[i];
      if (candidate->binding == PSBC_GALLIUM_IMAGE_ARRAY_BINDING(shader->stage)) {
         if (bank)
            return false;
         bank = candidate;
      }
   }
   if (!bank || bank->set || bank->type != PSBC_DESCRIPTOR_STORAGE_IMAGE ||
       bank->array_size != PS5_COMPUTE_IMAGE_SLOTS ||
       bank->offset != expected_offset || bank->stride != 32)
      return false;

   memset(table->data + bank->offset, 0, bank->array_size * bank->stride);
   for (unsigned i = 0; i < count; ++i) {
      const struct pipe_image_view *view = &context->preraster_images[stage][i];
      const struct ps5_resource *resource = (const struct ps5_resource *)view->resource;
      uint32_t *descriptor =
         (uint32_t *)(table->data + bank->offset + i * bank->stride);
      if (ps5_image_view_incomplete(view))
         continue;
      if (!resource || (resource->base.target == PIPE_BUFFER
             ? !ps5_image_buffer_descriptor(view, metadata->address32_hi,
                                             descriptor)
             : ps5_storage_image_view_descriptor(view, descriptor)))
         return false;
      if (resource->base.target == PIPE_BUFFER)
         ps5_flush_gpu_data(resource->data + view->u.buf.offset,
                            view->u.buf.size);
      else
         ps5_flush_gpu_data(resource->data, resource->size);
   }
   user_data[metadata->descriptor_set0_user_data_dword] =
      (uintptr_t)table->data;
   ps5_flush_gpu_data(table->data + bank->offset,
                      bank->array_size * bank->stride);
   return true;
}

static bool
ps5_prepare_fragment_storage(struct ps5_context *context, uint32_t *user_data,
                             unsigned user_data_count)
{
   const unsigned count = ps5_shader_storage_count(context->fs);
   if (!ps5_shader_uses_storage(context->fs))
      return true;
   const unsigned images = context->fs->nir->info.num_images;
   const unsigned table_bytes = PS5_COMPUTE_STORAGE_SLOTS * 16 + (images ? PS5_COMPUTE_IMAGE_SLOTS * 32 : 0);
   const PsbcShaderMetadata *metadata = &context->fs->active->output.metadata;
   struct ps5_resource *table = (struct ps5_resource *)context->descriptor_storage[1];
   if (count > PS5_COMPUTE_STORAGE_SLOTS || images > PS5_COMPUTE_IMAGE_SLOTS ||
       (count && context->fragment_bindings_invalid) || (images && context->fragment_images_invalid) ||
       !table || !table->data || table->size < table_bytes ||
       !metadata->descriptor_set0_valid || metadata->descriptor_binding_count !=
          1u + (images != 0) + (context->fs->nir->info.num_ubos != 0) + ps5_shader_texture_count(context->fs) ||
       metadata->descriptor_set0_user_data_dword >= user_data_count ||
       (uintptr_t)table->data >> 32 != metadata->address32_hi)
      return false;
   const PsbcDescriptorBinding *bank = &metadata->descriptor_bindings[0];
   if (bank->set || bank->binding != PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_FRAGMENT) ||
       bank->type != PSBC_DESCRIPTOR_STORAGE_BUFFER || bank->array_size != PS5_COMPUTE_STORAGE_SLOTS ||
       bank->offset || bank->stride != 16)
      return false;
   if (images) {
      const PsbcDescriptorBinding *image_bank = &metadata->descriptor_bindings[1];
      if (image_bank->set || image_bank->binding != PSBC_GALLIUM_IMAGE_ARRAY_BINDING(PSBC_STAGE_FRAGMENT) ||
          image_bank->type != PSBC_DESCRIPTOR_STORAGE_IMAGE || image_bank->array_size != PS5_COMPUTE_IMAGE_SLOTS ||
          image_bank->offset != PS5_COMPUTE_STORAGE_SLOTS * 16 || image_bank->stride != 32)
         return false;
   }
   memset(table->data, 0, table_bytes);
   for (unsigned i = 0; i < count; ++i) {
      const struct pipe_shader_buffer *bound = &context->fragment_buffers[i];
      const struct ps5_resource *resource = (const struct ps5_resource *)bound->buffer;
      /* An unbound slot keeps its zeroed (null) descriptor, as in the vertex
       * path: GL leaves such access undefined, it is not a draw error. */
      if (!resource)
         continue;
      if (resource->base.screen != context->base.screen ||
          !resource->data || bound->buffer_offset > resource->size ||
          bound->buffer_size > resource->size - bound->buffer_offset)
         return false;
      uintptr_t address = (uintptr_t)resource->data + bound->buffer_offset;
      uint32_t *srd = (uint32_t *)table->data + i * 4;
      srd[0] = address; srd[1] = address >> 32;
      srd[2] = bound->buffer_size; srd[3] = UINT32_C(0x31016fac);
      ps5_flush_gpu_data((void *)address, bound->buffer_size);
   }
   for (unsigned i = 0; i < images; ++i) {
      const struct pipe_image_view *view = &context->fragment_images[i];
      const struct ps5_resource *resource = (const struct ps5_resource *)view->resource;
      uint32_t *descriptor = (uint32_t *)(table->data +
         PS5_COMPUTE_STORAGE_SLOTS * 16 + i * 32);
      if (ps5_image_view_incomplete(view)) {
         memset(descriptor, 0, 32);
         continue;
      }
      if (!resource || (resource->base.target == PIPE_BUFFER ?
          !ps5_image_buffer_descriptor(view, metadata->address32_hi, descriptor) :
          ps5_storage_image_view_descriptor(view, descriptor)))
         return false;
      if (resource->base.target == PIPE_BUFFER)
         ps5_flush_gpu_data(resource->data + view->u.buf.offset, view->u.buf.size);
      else
         ps5_flush_gpu_data(resource->data, resource->size);
   }
   user_data[metadata->descriptor_set0_user_data_dword] = (uintptr_t)table->data;
   ps5_flush_gpu_data(table->data, table_bytes);
   return true;
}

static bool
ps5_prepare_vertex_storage(struct ps5_context *context,
                           const PsbcShaderMetadata *metadata,
                           uint32_t *user_data, unsigned user_data_count)
{
   const unsigned count = ps5_shader_storage_count(context->vs);
   struct ps5_resource *table =
      (struct ps5_resource *)context->descriptor_storage[0];
   const PsbcDescriptorBinding *bank = NULL;

   if (!count)
      return true;
   if (count > PS5_COMPUTE_STORAGE_SLOTS ||
       context->preraster_bindings_invalid[MESA_SHADER_VERTEX] ||
       !table || !table->data ||
       table->size < PS5_VERTEX_STORAGE_OFFSET +
                        PS5_COMPUTE_STORAGE_SLOTS * 16u ||
       !metadata->descriptor_set0_valid ||
       metadata->descriptor_set0_user_data_dword >= user_data_count ||
       (uintptr_t)table->data >> 32 != metadata->address32_hi)
      return false;
   for (unsigned i = 0; i < metadata->descriptor_binding_count; ++i) {
      const PsbcDescriptorBinding *candidate = &metadata->descriptor_bindings[i];
      if (candidate->binding ==
          PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_VERTEX)) {
         if (bank)
            return false;
         bank = candidate;
      }
   }
   if (!bank || bank->set || bank->type != PSBC_DESCRIPTOR_STORAGE_BUFFER ||
       bank->array_size != PS5_COMPUTE_STORAGE_SLOTS ||
       bank->offset != PS5_VERTEX_STORAGE_OFFSET || bank->stride != 16)
      return false;

   memset(table->data + bank->offset, 0, bank->array_size * bank->stride);
   for (unsigned i = 0; i < count; ++i) {
      const struct pipe_shader_buffer *bound =
         &context->preraster_buffers[MESA_SHADER_VERTEX][i];
      const struct ps5_resource *resource =
         (const struct ps5_resource *)bound->buffer;
      if (!resource)
         continue;
      if (resource->base.screen != context->base.screen || !resource->data ||
          resource->base.target != PIPE_BUFFER || !bound->buffer_size ||
          bound->buffer_offset > resource->size ||
          bound->buffer_size > resource->size - bound->buffer_offset)
         return false;
      uintptr_t address = (uintptr_t)resource->data + bound->buffer_offset;
      uint32_t *srd = (uint32_t *)(table->data + bank->offset) + i * 4;
      srd[0] = address;
      srd[1] = address >> 32;
      srd[2] = bound->buffer_size;
      srd[3] = UINT32_C(0x31016fac);
      ps5_flush_gpu_data((void *)address, bound->buffer_size);
   }
   user_data[metadata->descriptor_set0_user_data_dword] =
      (uintptr_t)table->data;
   ps5_flush_gpu_data(table->data + bank->offset,
                      bank->array_size * bank->stride);
   return true;
}

static bool
ps5_prepare_geometry_storage(struct ps5_context *context,
                             const PsbcShaderMetadata *metadata,
                             uint32_t *user_data, unsigned user_data_count)
{
   const unsigned count = ps5_shader_storage_count(context->gs);
   struct ps5_resource *table =
      (struct ps5_resource *)context->descriptor_storage[0];
   const PsbcDescriptorBinding *bank = NULL;

   if (!count)
      return true;
   if (count > PS5_COMPUTE_STORAGE_SLOTS || context->geometry_bindings_invalid ||
       !table || !table->data ||
       table->size < PS5_GEOMETRY_STORAGE_OFFSET +
                        PS5_COMPUTE_STORAGE_SLOTS * 16u ||
       !metadata->descriptor_set0_valid ||
       metadata->descriptor_set0_user_data_dword >= user_data_count ||
       (uintptr_t)table->data >> 32 != metadata->address32_hi)
      return false;
   for (unsigned i = 0; i < metadata->descriptor_binding_count; ++i) {
      const PsbcDescriptorBinding *candidate = &metadata->descriptor_bindings[i];
      if (candidate->binding ==
          PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_GEOMETRY)) {
         if (bank)
            return false;
         bank = candidate;
      }
   }
   if (!bank || bank->set || bank->type != PSBC_DESCRIPTOR_STORAGE_BUFFER ||
       bank->array_size != PS5_COMPUTE_STORAGE_SLOTS ||
       bank->offset != PS5_GEOMETRY_STORAGE_OFFSET || bank->stride != 16)
      return false;

   memset(table->data + bank->offset, 0, bank->array_size * bank->stride);
   for (unsigned i = 0; i < count; ++i) {
      const struct pipe_shader_buffer *bound = &context->geometry_buffers[i];
      const struct ps5_resource *resource =
         (const struct ps5_resource *)bound->buffer;
      if (!resource)
         continue;
      if (resource->base.screen != context->base.screen ||
          !resource->data || bound->buffer_offset > resource->size ||
          bound->buffer_size > resource->size - bound->buffer_offset)
         return false;
      uintptr_t address = (uintptr_t)resource->data + bound->buffer_offset;
      uint32_t *srd = (uint32_t *)(table->data + bank->offset) + i * 4;
      srd[0] = address;
      srd[1] = address >> 32;
      srd[2] = bound->buffer_size;
      srd[3] = UINT32_C(0x31016fac);
      ps5_flush_gpu_data((void *)address, bound->buffer_size);
   }
   user_data[metadata->descriptor_set0_user_data_dword] =
      (uintptr_t)table->data;
   ps5_flush_gpu_data(table->data + bank->offset,
                      bank->array_size * bank->stride);
   return true;
}

static bool
ps5_metadata_has_indirect_ubo(const PsbcShaderMetadata *metadata,
                              unsigned stage)
{
   for (unsigned index = 0; index < metadata->descriptor_binding_count;
        ++index) {
      const PsbcDescriptorBinding *binding =
         &metadata->descriptor_bindings[index];
      if (binding->type == PSBC_DESCRIPTOR_UNIFORM_BUFFER &&
          binding->binding == PSBC_GALLIUM_UBO_ARRAY_BINDING(stage))
         return true;
   }
   return false;
}

static bool
ps5_prepare_constant(struct ps5_context *context,
                     const struct ps5_shader *shader, unsigned slot,
                     uint32_t *user_data, unsigned user_data_count,
                     const PsbcShaderMetadata *metadata_override,
                     size_t *pending_descriptor_bytes)
{
   const PsbcShaderMetadata *metadata;
   struct ps5_resource *storage;
   uintptr_t descriptor_address;
   unsigned index;
   unsigned ubo_count = 0;
   unsigned expected_ubo_count;
   unsigned expected_ubo_bindings;
   unsigned expected_texture_count;
   bool merged_geometry;
   bool merged_geometry_storage;
   bool indirect_ubo;
   unsigned resource_bindings;
   const bool storage_fs = slot == 1 && ps5_shader_uses_storage(shader);

   if (!shader || !shader->active || slot >= 2)
      return false;
   metadata = metadata_override ? metadata_override :
                                   &shader->active->output.metadata;
   merged_geometry = ps5_uses_merged_geometry_metadata(context, shader,
                                                       metadata);
   merged_geometry_storage = merged_geometry &&
      (ps5_shader_uses_storage(context->vs) ||
       ps5_shader_uses_storage(context->gs) ||
       ps5_metadata_has_indirect_ubo(metadata, PSBC_STAGE_VERTEX) ||
       ps5_metadata_has_indirect_ubo(metadata, PSBC_STAGE_GEOMETRY));
   resource_bindings = (shader->nir->info.num_ssbos != 0) +
                       (shader->nir->info.num_images != 0);
   if (merged_geometry)
      resource_bindings += (context->gs->nir->info.num_ssbos != 0) +
                           (context->gs->nir->info.num_images != 0);
   indirect_ubo = !storage_fs && !merged_geometry &&
      ps5_metadata_has_indirect_ubo(
         metadata, slot ? PSBC_STAGE_FRAGMENT : PSBC_STAGE_VERTEX);
   expected_ubo_count = shader->nir->info.num_ubos +
      (merged_geometry ? context->gs->nir->info.num_ubos : 0u);
   expected_ubo_bindings = merged_geometry_storage
      ? (shader->nir->info.num_ubos != 0) +
           (context->gs->nir->info.num_ubos != 0)
      : indirect_ubo ? 1u : expected_ubo_count;
   expected_texture_count = ps5_texture_count(context, shader, metadata);
   if (!expected_ubo_count)
      return true;

   storage = (struct ps5_resource *)context->descriptor_storage[slot];
   if ((!PS5_ENABLE_UBO_CANDIDATE &&
        (shader->nir->info.num_ubos != 1 ||
         !shader->nir->info.first_ubo_is_default_ubo)) ||
       !storage ||
       storage->base.target != PIPE_BUFFER ||
       storage->size < ps5_descriptor_snapshot_size(context, slot) ||
      metadata->descriptor_binding_count !=
          (storage_fs ? 2u + (shader->nir->info.num_images != 0) :
                        expected_ubo_bindings + resource_bindings) +
             expected_texture_count ||
       !metadata->descriptor_set0_valid ||
       metadata->descriptor_set0_user_data_dword >= user_data_count)
      return false;

   descriptor_address = (uintptr_t)storage->data;
   if ((uint32_t)(descriptor_address >> 32) != metadata->address32_hi ||
       metadata->descriptor_binding_count > PSBC_MAX_DESCRIPTOR_BINDINGS)
      return false;

   if (!storage_fs)
      memset(storage->data, 0, PS5_CONSTANT_DATA_OFFSET);
   for (index = 0; index < metadata->descriptor_binding_count; ++index) {
      const PsbcDescriptorBinding *binding =
         &metadata->descriptor_bindings[index];
      const struct ps5_constant_state *state;
      struct ps5_resource *buffer;
      uintptr_t data_address;
      uint32_t *descriptor;
      unsigned state_binding;
      unsigned state_slot = slot;
      unsigned ubo_index;
      unsigned binding_ubo_count = 1;
      const struct ps5_shader *binding_shader = shader;

      if (binding->type != PSBC_DESCRIPTOR_UNIFORM_BUFFER)
         continue;
      if (storage_fs) {
         if (binding->set || binding->binding != PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_FRAGMENT) ||
             binding->array_size != PS5_MAX_CONSTANT_BUFFERS || binding->stride != 16 ||
             binding->offset != PS5_FRAGMENT_UBO_OFFSET || expected_ubo_count > PS5_MAX_CONSTANT_BUFFERS)
            return false;
      } else if (merged_geometry_storage) {
         if (binding->set || binding->stride != 16)
            return false;
         if (binding->binding ==
             PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_VERTEX)) {
            binding_ubo_count = shader->nir->info.num_ubos;
            if (!binding_ubo_count || binding->array_size != binding_ubo_count ||
                binding->offset != PS5_TEXTURE_DESCRIPTOR_BYTES)
               return false;
         } else if (binding->binding ==
                    PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_GEOMETRY)) {
            binding_shader = context->gs;
            binding_ubo_count = binding_shader->nir->info.num_ubos;
            state_slot = PS5_GEOMETRY_CONSTANT_SLOT;
            if (!binding_ubo_count || binding->array_size != binding_ubo_count ||
                binding->offset != PS5_TEXTURE_DESCRIPTOR_BYTES +
                                      shader->nir->info.num_ubos * 16u)
               return false;
         } else {
            return false;
         }
      } else if (indirect_ubo) {
         binding_ubo_count = shader->nir->info.num_ubos;
         if (binding->set ||
             binding->binding != PSBC_GALLIUM_UBO_ARRAY_BINDING(
                                    slot ? PSBC_STAGE_FRAGMENT : PSBC_STAGE_VERTEX) ||
             binding->array_size != binding_ubo_count || binding->stride != 16 ||
             binding->offset != PS5_TEXTURE_DESCRIPTOR_BYTES)
            return false;
      } else if (binding->set ||
          binding->binding < PSBC_GALLIUM_UBO_BINDING_BASE ||
          binding->binding >= PSBC_GALLIUM_UBO_BINDING_BASE +
                                 2u * PS5_MAX_CONSTANT_BUFFERS ||
          binding->array_size != 1 || binding->stride != 16 ||
          binding->offset != PS5_TEXTURE_DESCRIPTOR_BYTES +
                                (binding->binding -
                                 PSBC_GALLIUM_UBO_BINDING_BASE) * 16u)
         return false;
      for (unsigned array_index = 0;
           array_index < (storage_fs ? expected_ubo_count : binding_ubo_count);
           ++array_index) {
         state_binding = (storage_fs || merged_geometry_storage || indirect_ubo)
            ? array_index
            : binding->binding - PSBC_GALLIUM_UBO_BINDING_BASE;
         if (state_binding >= expected_ubo_count)
            return false;
         ubo_index = state_binding;
         if (merged_geometry_storage) {
            state_binding = ps5_constant_state_binding(binding_shader,
                                                        ubo_index);
         } else if (merged_geometry &&
                    state_binding >= shader->nir->info.num_ubos) {
            state_slot = PS5_GEOMETRY_CONSTANT_SLOT;
            ubo_index -= shader->nir->info.num_ubos;
            if (ubo_index >= context->gs->nir->info.num_ubos)
               return false;
            state_binding = ps5_constant_state_binding(context->gs, ubo_index);
         } else {
            if (ubo_index >= shader->nir->info.num_ubos)
               return false;
            state_binding = ps5_constant_state_binding(shader, ubo_index);
         }
         if (state_binding >= PS5_MAX_CONSTANT_BUFFERS)
            return false;
         state = &context->constants[state_slot][state_binding];
         descriptor = (uint32_t *)(storage->data + binding->offset + array_index * 16u);
         ubo_count++;
         memset(descriptor, 0, 16);
         if (!state->valid) {
            if (storage_fs)
               return false;
            continue;
         }
         if (!state->size || state->size > PS5_MAX_CONSTANT_BUFFER_SIZE)
            return false;
         if (state->copied) {
            const size_t copied_offset =
               ps5_copied_constant_offset(state_slot);

            if (copied_offset > storage->size ||
                state->size > storage->size - copied_offset)
               return false;
            data_address = (uintptr_t)storage->data + copied_offset;
         } else {
            buffer = (struct ps5_resource *)state->buffer;
            if (!buffer || buffer->base.target != PIPE_BUFFER ||
                state->offset > buffer->size ||
                state->size > buffer->size - state->offset)
               return false;
            data_address = (uintptr_t)buffer->data + state->offset;
            ps5_flush_gpu_data((void *)data_address, state->size);
         }
         descriptor[0] = (uint32_t)data_address;
         descriptor[1] = (uint32_t)(data_address >> 32);
         descriptor[2] = state->size;
         descriptor[3] = UINT32_C(0x0004dfac) |
            S_008F0C_OOB_SELECT(V_008F0C_OOB_SELECT_RAW);
      }
   }
   if (ubo_count != expected_ubo_count)
      return false;
   user_data[metadata->descriptor_set0_user_data_dword] =
      (uint32_t)descriptor_address;
   ps5_publish_descriptor_prefix(storage->data,
      ps5_descriptor_snapshot_size(context, slot), pending_descriptor_bytes);
   return true;
}

static bool
ps5_tessellation_buffer_layout(const struct ps5_context *context,
                               PsbcCompileOptions *options)
{
   const struct ps5_shader *stages[] = {
      context->vs, context->tcs, context->tes, context->gs,
   };
   options->descriptor_binding_count = 0;
   options->gallium_buffer_arrays = true;
   for (unsigned stage = 0; stage < 4; ++stage) {
      const struct ps5_shader *shader = stages[stage];
      if (!shader)
         continue;
      if (shader->stage != PSBC_STAGE_VERTEX + stage ||
          shader->nir->info.num_images > PS5_COMPUTE_IMAGE_SLOTS ||
          shader->nir->info.num_ubos > PS5_MAX_CONSTANT_BUFFERS ||
          shader->nir->info.num_ssbos > PS5_COMPUTE_STORAGE_SLOTS)
         return false;
      const unsigned offset = PS5_TESSELLATION_BUFFER_OFFSET +
                              stage * PS5_TESSELLATION_BUFFER_STRIDE;
      if (shader->nir->info.num_ubos)
         options->descriptor_bindings[options->descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PSBC_GALLIUM_UBO_ARRAY_BINDING(shader->stage),
               .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER,
               .array_size = shader->nir->info.num_ubos,
               .offset = offset, .stride = 16,
            };
      if (shader->nir->info.num_ssbos)
         options->descriptor_bindings[options->descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PSBC_GALLIUM_SSBO_ARRAY_BINDING(shader->stage),
               .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
               .array_size = shader->nir->info.num_ssbos,
               .offset = offset + PS5_MAX_CONSTANT_BUFFERS * 16u, .stride = 16,
            };
      if (shader->nir->info.num_images)
         options->descriptor_bindings[options->descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PSBC_GALLIUM_IMAGE_ARRAY_BINDING(shader->stage),
               .type = PSBC_DESCRIPTOR_STORAGE_IMAGE,
               .array_size = PS5_COMPUTE_IMAGE_SLOTS,
               .offset = offset +
                  (PS5_MAX_CONSTANT_BUFFERS + PS5_COMPUTE_STORAGE_SLOTS) * 16u,
               .stride = 32,
            };
      for (unsigned unit = 0; unit < PS5_MAX_TEXTURE_UNITS; ++unit) {
         if (!BITSET_TEST(shader->nir->info.textures_used, unit))
            continue;
         if (options->descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
            return false;
         const unsigned index = stage * PS5_MAX_TEXTURE_UNITS + unit;
         options->descriptor_bindings[options->descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PS5_TESSELLATION_TEXTURE_BINDING + index,
               .type = PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER,
               .array_size = 1, .stride = PS5_TEXTURE_DESCRIPTOR_STRIDE,
               .offset = PS5_TESSELLATION_TEXTURE_OFFSET + index * PS5_TEXTURE_DESCRIPTOR_STRIDE,
            };
      }
   }
   return true;
}

static bool
ps5_prepare_tessellation_buffers(struct ps5_context *context,
                                 const PsbcShaderMetadata *metadata,
                                 uint32_t *user_data, unsigned user_data_count)
{
   PsbcCompileOptions expected = {0};
   struct ps5_resource *table =
      (struct ps5_resource *)context->descriptor_storage[0];
   const struct ps5_shader *stages[] = {
      context->vs, context->tcs, context->tes, context->gs,
   };
   const unsigned slots[] = {0, PS5_TESS_CTRL_CONSTANT_SLOT,
      PS5_TESS_EVAL_CONSTANT_SLOT, PS5_GEOMETRY_CONSTANT_SLOT};
   if (!ps5_tessellation_buffer_layout(context, &expected) ||
       metadata->descriptor_binding_count != expected.descriptor_binding_count) {
      printf("[ps5-gallium] tess-buffer reject=layout actual=%u expected=%u\n",
             metadata->descriptor_binding_count,
             expected.descriptor_binding_count);
      return false;
   }
   if (!expected.descriptor_binding_count)
      return true;
   if (!table || !table->data || table->base.target != PIPE_BUFFER ||
       table->base.screen != context->base.screen ||
       table->size < PS5_DESCRIPTOR_STORAGE_BYTES ||
       !metadata->descriptor_set0_valid ||
       metadata->descriptor_set0_user_data_dword >= user_data_count ||
       (uintptr_t)table->data >> 32 != metadata->address32_hi)
      return false;
   for (unsigned b = 0; b < expected.descriptor_binding_count; ++b) {
      const PsbcDescriptorBinding *bank = &expected.descriptor_bindings[b];
      const PsbcDescriptorBinding *actual = &metadata->descriptor_bindings[b];
      if (actual->set != bank->set || actual->binding != bank->binding ||
          actual->type != bank->type || actual->array_size != bank->array_size ||
          actual->offset != bank->offset || actual->stride != bank->stride) {
         printf("[ps5-gallium] tess-buffer reject=bank index=%u binding=%u/%u type=%u/%u array=%u/%u offset=%u/%u stride=%u/%u\n",
                b, actual->binding, bank->binding, actual->type, bank->type,
                actual->array_size, bank->array_size, actual->offset,
                bank->offset, actual->stride, bank->stride);
         return false;
      }
      if (bank->type == PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
         continue; /* Encoded by the shared texture preparation path below. */
      const unsigned stage = (bank->binding - 1u) / 4u;
      const bool uniform = bank->type == PSBC_DESCRIPTOR_UNIFORM_BUFFER;
      if (bank->type == PSBC_DESCRIPTOR_STORAGE_IMAGE) {
         if (stage > MESA_SHADER_GEOMETRY ||
             context->preraster_images_invalid[stage]) {
            fprintf(stderr, "[ps5-gallium] tess-buffer reject=image-state stage=%u invalid=%u\n",
                    stage, stage <= MESA_SHADER_GEOMETRY &&
                              context->preraster_images_invalid[stage]);
            return false;
         }
         memset(table->data + bank->offset, 0,
                bank->array_size * bank->stride);
         for (unsigned i = 0; i < stages[stage]->nir->info.num_images; ++i) {
            const struct pipe_image_view *view =
               &context->preraster_images[stage][i];
            const struct ps5_resource *resource =
               (const struct ps5_resource *)view->resource;
            uint32_t *descriptor =
               (uint32_t *)(table->data + bank->offset + i * bank->stride);
            if (ps5_image_view_incomplete(view))
               continue;
            if (!resource || (resource->base.target == PIPE_BUFFER
                   ? !ps5_image_buffer_descriptor(view, metadata->address32_hi,
                                                   descriptor)
                   : ps5_storage_image_view_descriptor(view, descriptor))) {
               fprintf(stderr, "[ps5-gallium] tess-buffer reject=image-resource stage=%u index=%u target=%u\n",
                       stage, i, resource ? resource->base.target : 0u);
               return false;
            }
            if (resource->base.target == PIPE_BUFFER)
               ps5_flush_gpu_data(resource->data + view->u.buf.offset,
                                  view->u.buf.size);
            else
               ps5_flush_gpu_data(resource->data, resource->size);
         }
         continue;
      }
      if (!uniform && (stage == 3 ? context->geometry_bindings_invalid :
                                   context->preraster_bindings_invalid[stage]))
         return false;
      for (unsigned i = 0; i < bank->array_size; ++i) {
         uint32_t *srd =
            (uint32_t *)(table->data + bank->offset + i * 16u);
         uintptr_t address;
         unsigned size;
         if (uniform) {
            const unsigned index = ps5_constant_state_binding(stages[stage], i);
            if (index >= PS5_MAX_CONSTANT_BUFFERS)
               return false;
            const struct ps5_constant_state *bound =
               &context->constants[slots[stage]][index];
            size = bound->size;
            if (!bound->valid || !size || size > PS5_MAX_CONSTANT_BUFFER_SIZE)
               return false;
            if (bound->copied) {
               address = (uintptr_t)table->data + ps5_copied_constant_offset(slots[stage]);
            } else {
               const struct ps5_resource *resource = (const struct ps5_resource *)bound->buffer;
               if (!resource || !resource->data || resource->base.target != PIPE_BUFFER ||
                   resource->base.screen != context->base.screen ||
                   bound->offset > resource->size || size > resource->size - bound->offset)
                  return false;
               address = (uintptr_t)resource->data + bound->offset;
            }
         } else {
            const struct pipe_shader_buffer *bound = stage == 3
               ? &context->geometry_buffers[i] : &context->preraster_buffers[stage][i];
            const struct ps5_resource *resource = (const struct ps5_resource *)bound->buffer;
            if (!resource) {
               memset(srd, 0, 16);
               continue;
            }
            size = bound->buffer_size;
            if (!resource->data || resource->base.target != PIPE_BUFFER ||
                resource->base.screen != context->base.screen || !size ||
                bound->buffer_offset > resource->size || size > resource->size - bound->buffer_offset) {
               printf("[ps5-gallium] tess-buffer reject=resource stage=%u index=%u bound=%u size=%u invalid=%u\n",
                      stage, i, resource != NULL, size,
                      stage == 3 ? context->geometry_bindings_invalid :
                                   context->preraster_bindings_invalid[stage]);
               return false;
            }
            address = (uintptr_t)resource->data + bound->buffer_offset;
         }
         srd[0] = address;
         srd[1] = address >> 32;
         srd[2] = size;
         srd[3] = uniform ? UINT32_C(0x0004dfac) |
            S_008F0C_OOB_SELECT(V_008F0C_OOB_SELECT_RAW) : UINT32_C(0x31016fac);
         ps5_flush_gpu_data((void *)address, size);
      }
   }
   user_data[metadata->descriptor_set0_user_data_dword] = (uintptr_t)table->data;
   ps5_flush_gpu_data(table->data + PS5_TESSELLATION_BUFFER_OFFSET,
                      4u * PS5_TESSELLATION_BUFFER_STRIDE);
   return true;
}

static bool
ps5_encode_sampler_words(const struct pipe_sampler_state *sampler, uint32_t words[3])
{
   uint32_t wrap[3], filter[2], mip_filter;
   uint32_t min_lod, max_lod, lod_bias, anisotropy;
   if (sampler->unnormalized_coords ||
       sampler->max_anisotropy > 16 ||
       !ps5_float_is_finite(sampler->min_lod) ||
       !ps5_float_is_finite(sampler->max_lod) ||
       !ps5_float_is_finite(sampler->lod_bias) ||
       !ps5_texture_descriptor_wrap(sampler->wrap_s, &wrap[0]) ||
       !ps5_texture_descriptor_wrap(sampler->wrap_t, &wrap[1]) ||
       !ps5_texture_descriptor_wrap(sampler->wrap_r, &wrap[2]) ||
       !ps5_texture_descriptor_filter(sampler->min_img_filter,
                                      sampler->max_anisotropy,
                                      &filter[0]) ||
       !ps5_texture_descriptor_filter(sampler->mag_img_filter,
                                      sampler->max_anisotropy,
                                      &filter[1]) ||
       !ps5_texture_descriptor_mip_filter(sampler->min_mip_filter,
                                          &mip_filter))
      return false;
   min_lod = ps5_texture_descriptor_unsigned_lod(sampler->min_lod);
   max_lod = ps5_texture_descriptor_unsigned_lod(sampler->max_lod);
   lod_bias = ps5_texture_descriptor_lod_bias(sampler->lod_bias);
   anisotropy = ps5_texture_descriptor_anisotropy(
      sampler->max_anisotropy);
   words[0] = wrap[0] | (wrap[1] << 3) | (wrap[2] << 6) |
                   (anisotropy << 9) | ((anisotropy >> 1) << 16) |
                   (anisotropy << 21) |
                   (sampler->compare_mode ?
                      sampler->compare_func << 12 : 0) |
                   ((PS5_ENABLE_SEAMLESS_CUBE_CANDIDATE &&
                     !sampler->seamless_cube_map) << 28);
   words[1] = min_lod | (max_lod << 12) |
                   ((anisotropy ? anisotropy + 6u : 0u) << 24);
   words[2] = lod_bias | (filter[1] << 20) | (filter[0] << 22) |
                    (mip_filter << 26) |
                    (anisotropy ? 1u << 29 : 0u);
   return true;
}

static bool
ps5_prepare_texture(struct ps5_context *context,
                    const struct ps5_shader *shader, unsigned slot,
                    uint32_t *user_data, unsigned user_data_count,
                    const PsbcShaderMetadata *metadata_override,
                    struct ps5_batch_flush_cache *flush_cache,
                    size_t *pending_descriptor_bytes)
{
   const PsbcShaderMetadata *metadata;
   struct ps5_resource *table;
   uintptr_t table_address;
   size_t flush_size = 0;
   unsigned texture_count = 0;
   unsigned expected_texture_count;
   unsigned expected_ubo_count;
   unsigned expected_ubo_bindings;
   bool indirect_ubo;
   bool merged_geometry;
   bool merged_geometry_storage;
   unsigned resource_bindings;
   const bool storage_fs = slot == 1 && ps5_shader_uses_storage(shader);

   if (!shader || (!shader->active && !metadata_override) || slot >= PS5_DESCRIPTOR_STAGE_COUNT) {
      printf("[ps5-gallium] texture-prepare reject=shader slot=%u shader=%u active=%u override=%u\n",
             slot, shader != NULL, shader && shader->active,
             metadata_override != NULL);
      return false;
   }
   metadata = metadata_override ? metadata_override :
                                   &shader->active->output.metadata;
   const bool merged_tessellation = ps5_uses_tessellation_metadata(context, metadata);
   merged_geometry = ps5_uses_merged_geometry_metadata(context, shader,
                                                       metadata);
   merged_geometry_storage = merged_geometry &&
      (ps5_shader_uses_storage(context->vs) ||
       ps5_shader_uses_storage(context->gs) ||
       ps5_metadata_has_indirect_ubo(metadata, PSBC_STAGE_VERTEX) ||
       ps5_metadata_has_indirect_ubo(metadata, PSBC_STAGE_GEOMETRY));
   resource_bindings = (shader->nir->info.num_ssbos != 0) +
                       (shader->nir->info.num_images != 0);
   if (merged_geometry)
      resource_bindings += (context->gs->nir->info.num_ssbos != 0) +
                           (context->gs->nir->info.num_images != 0);
   expected_texture_count = ps5_texture_count(context, shader, metadata);
   if (!expected_texture_count)
      return true;
   expected_ubo_count = shader->nir->info.num_ubos +
      (merged_geometry ? context->gs->nir->info.num_ubos : 0u);
   indirect_ubo = !storage_fs && !merged_geometry &&
      ps5_metadata_has_indirect_ubo(metadata, shader->stage);
   expected_ubo_bindings = merged_geometry_storage
      ? (shader->nir->info.num_ubos != 0) +
           (context->gs->nir->info.num_ubos != 0)
      : indirect_ubo ? 1u : expected_ubo_count;
   table = (struct ps5_resource *)context->descriptor_storage[slot];
   unsigned expected_binding_count =
      (storage_fs ? 1u + (shader->nir->info.num_images != 0) +
                       (expected_ubo_count != 0) :
                    expected_ubo_bindings + resource_bindings) +
         expected_texture_count;
   if (merged_tessellation) {
      PsbcCompileOptions layout = {0};
      if (!ps5_tessellation_buffer_layout(context, &layout))
         return false;
      expected_binding_count = layout.descriptor_binding_count;
   }
   if ((shader->stage == PSBC_STAGE_VERTEX && slot != 0) ||
       (shader->stage == PSBC_STAGE_FRAGMENT && slot != 1) ||
       (shader->stage != PSBC_STAGE_VERTEX &&
        shader->stage != PSBC_STAGE_FRAGMENT) ||
       expected_texture_count > (merged_tessellation ? 4u * PS5_MAX_TEXTURE_UNITS : merged_geometry ? PS5_MERGED_TEXTURE_UNITS
                                                 : PS5_MAX_TEXTURE_UNITS) ||
       !metadata->descriptor_set0_valid ||
       metadata->descriptor_set0_user_data_dword >= user_data_count ||
       metadata->descriptor_binding_count != expected_binding_count ||
       !table) {
      printf("[ps5-gallium] texture-prepare reject=layout stage=%u slot=%u expected=%u bindings=%u set0=%u dword=%u/%u table=%u merged=%u/%u\n",
             shader->stage, slot, expected_binding_count,
             metadata->descriptor_binding_count,
             metadata->descriptor_set0_valid,
             metadata->descriptor_set0_user_data_dword, user_data_count,
             table != NULL, merged_geometry, merged_tessellation);
      return false;
   }

   table_address = (uintptr_t)table->data;
   if ((uint32_t)(table_address >> 32) != metadata->address32_hi)
      return false;

   for (unsigned index = 0; index < metadata->descriptor_binding_count;
        ++index) {
      const PsbcDescriptorBinding *binding =
         &metadata->descriptor_bindings[index];
      const unsigned unit = binding->binding % PS5_MAX_TEXTURE_UNITS;
      unsigned state_slot = slot;
      const struct ps5_sampler_state *sampler_state;
      const struct pipe_sampler_state *sampler;
      const struct pipe_sampler_view *view;
      struct ps5_resource *texture;
      struct ps5_resource *stencil_sample = NULL;
      uintptr_t texture_address;
      uint32_t *descriptor;
      uint32_t format_word;
      uint32_t swizzle[4];
      unsigned format_size;
      unsigned descriptor_format_size;
      unsigned descriptor_stride;
      unsigned descriptor_last_level;
      unsigned view_layers;
      enum pipe_texture_target descriptor_target;
      bool tiled_render_target;
      bool tiled_depth_target;
      bool depth_texture;
      bool stencil_texture;
      bool staged_packed_depth;
      bool staged_stencil;
      bool multisampled;

      if (binding->type != PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER)
         continue;
      if (binding->set ||
          binding->binding >= (merged_tessellation ? PS5_TESSELLATION_TEXTURE_BINDING + 4u * PS5_MAX_TEXTURE_UNITS : merged_geometry ? PS5_MERGED_TEXTURE_UNITS
                                               : PS5_MAX_TEXTURE_UNITS) ||
          binding->array_size != 1 || binding->stride != 48 ||
          (merged_tessellation && binding->binding < PS5_TESSELLATION_TEXTURE_BINDING) ||
          binding->offset != (merged_tessellation ? PS5_TESSELLATION_TEXTURE_OFFSET +
             (binding->binding - PS5_TESSELLATION_TEXTURE_BINDING) * PS5_TEXTURE_DESCRIPTOR_STRIDE :
             (storage_fs ? PS5_FRAGMENT_TEXTURE_OFFSET : 0u) + binding->binding * PS5_TEXTURE_DESCRIPTOR_STRIDE) ||
          binding->offset + binding->stride > table->size ||
          !ps5_texture_used(context, shader, metadata, binding->binding)) {
         printf("[ps5-gallium] texture-prepare reject=binding index=%u type=%u set=%u binding=%u array=%u stride=%u offset=%u/%zu used=%u\n",
                index, binding->type, binding->set, binding->binding,
                binding->array_size, binding->stride, binding->offset,
                table->size,
                ps5_texture_used(context, shader, metadata,
                                 binding->binding));
         return false;
      }
      texture_count++;
      if (merged_geometry && binding->binding >= PS5_MAX_TEXTURE_UNITS)
         state_slot = PS5_GEOMETRY_TEXTURE_SLOT;
      if (merged_tessellation) {
         const unsigned slots[] = {0, PS5_TESS_CTRL_TEXTURE_SLOT,
            PS5_TESS_EVAL_TEXTURE_SLOT, PS5_GEOMETRY_TEXTURE_SLOT};
         state_slot = slots[(binding->binding - PS5_TESSELLATION_TEXTURE_BINDING) / PS5_MAX_TEXTURE_UNITS];
      }
      view = context->sampler_views[state_slot][unit];
      if (!view) {
         /* Match radeonsi's null sampler view: an unused/absent descriptor
          * must not reject the entire draw. Invalid SRDs suppress accesses. */
         memset(table->data + binding->offset, 0, binding->stride);
         flush_size = MAX2(flush_size, binding->offset + binding->stride);
         continue;
      }
      if (!view->texture) {
         printf("[ps5-gallium] texture-prepare reject=view slot=%u unit=%u view=%u texture=%u\n",
                state_slot, unit, view != NULL,
                view && view->texture);
         return false;
      }
      texture = (struct ps5_resource *)view->texture;
      descriptor = (uint32_t *)(table->data + binding->offset);
      if (texture->base.target == PIPE_BUFFER) {
         const size_t offset = view->u.buf.offset;
         const size_t size = view->u.buf.size;
         memset(descriptor, 0, binding->stride);
         if (!ps5_texel_buffer_descriptor(view, metadata->address32_hi, descriptor))
            return false;
         ps5_flush_gpu_data((uint8_t *)texture->data + offset, size);
         flush_size = MAX2(flush_size, binding->offset + binding->stride);
         continue;
      }
      view_layers = view->u.tex.last_layer - view->u.tex.first_layer + 1;
      sampler_state = context->samplers[state_slot][unit];
      sampler = sampler_state ? &sampler_state->base : NULL;
      if (!sampler) {
         printf("[ps5-gallium] texture-prepare reject=sampler slot=%u unit=%u\n",
                state_slot, unit);
         return false;
      }
      format_size = ps5_texture_format_size(texture->base.format);
      depth_texture = texture->base.format == PIPE_FORMAT_Z32_FLOAT ||
                      texture->base.format ==
                         PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      stencil_texture =
         texture->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
         view->format == PIPE_FORMAT_X32_S8X24_UINT;
      staged_stencil = stencil_texture;
      multisampled = PS5_ENABLE_MSAA4_CANDIDATE &&
                     texture->base.nr_samples == 4 &&
                     texture->base.nr_storage_samples == 4;
      tiled_render_target = (multisampled && !depth_texture) ||
                            (PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE &&
                             (texture->base.target == PIPE_TEXTURE_2D ||
                              (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
                               texture->base.target ==
                                  PIPE_TEXTURE_2D_ARRAY)) &&
                             (texture->base.bind & PIPE_BIND_RENDER_TARGET) &&
                             !ps5_linear_sampled_layout(&texture->base));
      tiled_depth_target = PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE &&
                           depth_texture &&
                           ps5_depth_render_target(texture->base.target) &&
                           ((texture->base.bind & PIPE_BIND_DEPTH_STENCIL) ||
                            multisampled) &&
                           !texture->depth_staging_size;
      descriptor_format_size = stencil_texture ? 1u : format_size;
      descriptor_stride = stencil_texture ? texture->base.width0
                                           : texture->level_stride[0];
      descriptor_last_level = texture->base.last_level;
      staged_packed_depth =
         texture->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
         !stencil_texture && !tiled_depth_target &&
         texture->depth_staging_size;
      if (!ps5_sampled_texture_target(texture->base.target) ||
          !ps5_sampled_texture_format(texture->base.format) ||
          !texture->base.width0 || !texture->base.height0 ||
          texture->base.width0 > PS5_MAX_TEXTURE_2D_SIZE ||
          texture->base.height0 > PS5_MAX_TEXTURE_2D_SIZE ||
          (texture->base.target == PIPE_TEXTURE_1D &&
           (texture->base.height0 != 1 || texture->base.array_size != 1)) ||
          (texture->base.target == PIPE_TEXTURE_1D_ARRAY &&
           (texture->base.height0 != 1 || !texture->base.array_size ||
            texture->base.array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS)) ||
          (ps5_cube_texture_target(texture->base.target) &&
           ((texture->base.target == PIPE_TEXTURE_CUBE &&
             texture->base.array_size != 6) ||
            (texture->base.target == PIPE_TEXTURE_CUBE_ARRAY &&
             (!texture->base.array_size || texture->base.array_size % 6 ||
              texture->base.array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS)) ||
            texture->base.width0 > PS5_MAX_TEXTURE_CUBE_SIZE ||
            texture->base.width0 != texture->base.height0)) ||
          (texture->base.target == PIPE_TEXTURE_2D_ARRAY &&
           (!texture->base.array_size ||
            texture->base.array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS)) ||
          (texture->base.target == PIPE_TEXTURE_3D &&
           (texture->base.array_size != 1 || !texture->base.depth0 ||
            texture->base.depth0 > PS5_MAX_TEXTURE_3D_SIZE ||
            texture->base.width0 > PS5_MAX_TEXTURE_3D_SIZE ||
            texture->base.height0 > PS5_MAX_TEXTURE_3D_SIZE)) ||
          (texture->base.last_level && !PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE) ||
          (texture->base.target == PIPE_TEXTURE_RECT &&
           texture->base.last_level) ||
          texture->base.last_level > 15 ||
          ((tiled_render_target ||
            (tiled_depth_target && (!staged_stencil || stencil_texture)))
              ? (tiled_render_target
              ? ((multisampled
                    ? !ps5_msaa4_color_format(texture->base.format)
                    : (texture->base.format != PIPE_FORMAT_R8G8B8A8_UNORM &&
                       texture->base.format != PIPE_FORMAT_R8G8B8A8_SRGB &&
                       texture->base.format != PIPE_FORMAT_R8_UNORM &&
                       texture->base.format != PIPE_FORMAT_R8G8_UNORM &&
                       texture->base.format != PIPE_FORMAT_R16G16B16A16_FLOAT &&
                       texture->base.format !=
                          PIPE_FORMAT_R11G11B10_FLOAT &&
                       !(PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE &&
                         ps5_integer_texture_format(
                            texture->base.format)) &&
                       texture->base.format !=
                          PIPE_FORMAT_R10G10B10A2_UINT)) ||
                       (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE
                           ? (texture->base.width0 > PS5_MAX_COLOR_WIDTH ||
                              texture->base.height0 > PS5_MAX_COLOR_HEIGHT)
                           : (texture->base.width0 != PS5_RENDER_WIDTH ||
                              texture->base.height0 != PS5_RENDER_HEIGHT)))
                    : ((texture->base.format != PIPE_FORMAT_Z32_FLOAT &&
                        texture->base.format !=
                           PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) ||
                       (PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE
                           ? (texture->base.width0 > PS5_MAX_DEPTH_WIDTH ||
                              texture->base.height0 > PS5_MAX_DEPTH_HEIGHT)
                           : (texture->base.width0 != PS5_RENDER_WIDTH ||
                              texture->base.height0 != PS5_RENDER_HEIGHT)))) ||
                 texture->base.last_level ||
                 texture->level_stride[0] !=
                    texture->base.width0 * format_size
              : texture->level_stride[0] !=
                   ((format_size * texture->base.width0 + 255u) & ~255u)) ||
          !ps5_texture_view_target_compatible(texture->base.target,
                                              view->target) ||
          !ps5_texture_view_format_compatible(texture->base.format,
                                              view->format) ||
          view->u.tex.first_level > view->u.tex.last_level ||
          view->u.tex.last_level > texture->base.last_level ||
          view->u.tex.first_layer > view->u.tex.last_layer ||
          view->u.tex.last_layer >= ps5_texture_level_layers(
             &texture->base, view->u.tex.first_level) ||
          ((view->target == PIPE_TEXTURE_1D ||
            view->target == PIPE_TEXTURE_2D ||
            view->target == PIPE_TEXTURE_RECT) && view_layers != 1) ||
          (view->target == PIPE_TEXTURE_CUBE && view_layers != 6) ||
          (view->target == PIPE_TEXTURE_CUBE_ARRAY && view_layers % 6) ||
          (view->target == PIPE_TEXTURE_3D &&
            (view->u.tex.first_layer || view->u.tex.last_layer)) ||
          (stencil_texture && !texture->stencil_data) ||
          (sampler->min_mip_filter != PIPE_TEX_MIPFILTER_NONE &&
           !PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE) ||
          (sampler->compare_mode &&
           (!depth_texture || stencil_texture ||
            sampler->compare_func > PIPE_FUNC_ALWAYS)) ||
          !sampler_state->words_valid ||
          !ps5_texture_descriptor_format(view->format, &format_word) ||
          !ps5_texture_descriptor_swizzle(view->swizzle_r, view->format, &swizzle[0]) ||
          !ps5_texture_descriptor_swizzle(view->swizzle_g, view->format, &swizzle[1]) ||
          !ps5_texture_descriptor_swizzle(view->swizzle_b, view->format, &swizzle[2]) ||
          !ps5_texture_descriptor_swizzle(view->swizzle_a, view->format, &swizzle[3])) {
         printf("[ps5-gallium] texture-prepare reject=state slot=%u unit=%u target=%u/%u format=%u/%u size=%ux%ux%u array=%u levels=%u view-levels=%u:%u layers=%u:%u stride=%u alloc=%zu bind=%x samples=%u:%u sampler=%u/%u/%u mip=%u lod=%g:%g bias=%g aniso=%u compare=%u/%u unnormalized=%u\n",
                state_slot, unit, texture->base.target, view->target,
                texture->base.format, view->format, texture->base.width0,
                texture->base.height0, texture->base.depth0,
                texture->base.array_size, texture->base.last_level,
                view->u.tex.first_level, view->u.tex.last_level,
                view->u.tex.first_layer, view->u.tex.last_layer,
                texture->level_stride[0], texture->allocation_size,
                texture->base.bind, texture->base.nr_samples,
                texture->base.nr_storage_samples, sampler->wrap_s,
                sampler->wrap_t, sampler->wrap_r,
                sampler->min_mip_filter, sampler->min_lod,
                sampler->max_lod, sampler->lod_bias,
                sampler->max_anisotropy, sampler->compare_mode,
                sampler->compare_func, sampler->unnormalized_coords);
         return false;
      }

      if (staged_stencil) {
         stencil_sample = ps5_stage_packed_stencil_samples(texture);
         if (!stencil_sample)
            return false;
         descriptor_stride = stencil_sample->level_stride[0];
         if (multisampled) {
            if (!ps5_texture_descriptor_format(PIPE_FORMAT_R8_UINT,
                                               &format_word))
               return false;
            tiled_render_target = true;
            tiled_depth_target = false;
         }
      } else if (staged_packed_depth) {
         descriptor_format_size = sizeof(float);
         if (!ps5_stage_packed_depth_samples(texture, &descriptor_stride))
            return false;
      } else if (texture->base.format ==
                    PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
                 !tiled_depth_target) {
         format_word = gfx10_format_table[view->format].img_format << 20;
      }

      descriptor_target = texture->base.target;
      if (ps5_cube_texture_target(view->target))
         descriptor_target = view->target;
      else if (ps5_cube_texture_target(descriptor_target))
         descriptor_target = PIPE_TEXTURE_2D_ARRAY;

      texture_address = (uintptr_t)(staged_stencil ? stencil_sample->data
                                      : texture->data) +
         (staged_packed_depth ? texture->depth_staging_offset : 0);
      if ((texture_address & 0xffu) || texture_address >> 48) {
         printf("[ps5-gallium] texture-prepare reject=address slot=%u unit=%u address=%" PRIxPTR " high=%08x align=%u overflow=%u\n",
                state_slot, unit, texture_address,
                (uint32_t)(texture_address >> 32),
                (unsigned)(texture_address & 0xffu),
                (unsigned)(texture_address >> 48 != 0));
         return false;
      }
      memset(descriptor, 0, binding->stride);
      descriptor[0] = (uint32_t)(texture_address >> 8);
      descriptor[1] = format_word |
                      (((texture->base.width0 - 1u) & 3u) << 30) |
                      (uint32_t)(texture_address >> 40);
      descriptor[2] = ((texture->base.width0 - 1u) >> 2) |
                      ((texture->base.height0 - 1u) << 14) |
                      (UINT32_C(1) << 31); /* GFX10 RESOURCE_LEVEL. */
      descriptor[3] = ((tiled_depth_target && !staged_stencil) && multisampled
                           ? (descriptor_target == PIPE_TEXTURE_2D_ARRAY
                                 ? UINT32_C(0xf1820000)
                                 : UINT32_C(0xe1820000))
                       : tiled_depth_target && !staged_stencil
                           ? (descriptor_target == PIPE_TEXTURE_1D
                                 ? UINT32_C(0x81800000)
                              : descriptor_target ==
                                   PIPE_TEXTURE_1D_ARRAY
                                 ? UINT32_C(0xc1800000)
                              : ps5_cube_texture_target(descriptor_target)
                                 ? UINT32_C(0xb1800000)
                              : descriptor_target ==
                                   PIPE_TEXTURE_2D_ARRAY
                                 ? UINT32_C(0xd1800000)
                              : descriptor_target == PIPE_TEXTURE_3D
                                 ? UINT32_C(0xa1800000)
                                 : UINT32_C(0x91800000))
                       : multisampled
                           ? (descriptor_target == PIPE_TEXTURE_2D_ARRAY
                                 ? UINT32_C(0xf1b20000)
                                 : UINT32_C(0xe1b20000))
                       : tiled_render_target
                           ? (descriptor_target == PIPE_TEXTURE_2D_ARRAY
                                 ? UINT32_C(0xd1b00000)
                                 : UINT32_C(0x91b00000))
                        : descriptor_target == PIPE_TEXTURE_1D
                           ? UINT32_C(0x80000000)
                       : descriptor_target == PIPE_TEXTURE_1D_ARRAY
                          ? UINT32_C(0xc0000000)
                       : ps5_cube_texture_target(descriptor_target)
                           ? UINT32_C(0xb0000000)
                       : descriptor_target == PIPE_TEXTURE_2D_ARRAY
                          ? UINT32_C(0xd0000000)
                       : descriptor_target == PIPE_TEXTURE_3D
                          ? UINT32_C(0xa0000000)
                          : UINT32_C(0x90000000)) |
                      swizzle[0] |
                      (swizzle[1] << 3) | (swizzle[2] << 6) |
                      (swizzle[3] << 9) |
                      (view->u.tex.first_level << 12) |
                      (view->u.tex.last_level << 16);
      /* GFX10 sampler views keep the allocation base address and select their
       * layer range with BASE_ARRAY and the absolute last accessible layer. */
      descriptor[4] = ps5_cube_texture_target(descriptor_target)
                         ? view->u.tex.last_layer
                      : descriptor_target == PIPE_TEXTURE_3D
                         ? texture->base.depth0 - 1
                      : descriptor_target == PIPE_TEXTURE_1D_ARRAY ||
                        descriptor_target == PIPE_TEXTURE_2D_ARRAY
                         ? view->u.tex.last_layer : 0;
      if (descriptor_target != PIPE_TEXTURE_3D)
         descriptor[4] |= view->u.tex.first_layer << 16;
      if (view->target == PIPE_TEXTURE_2D &&
          !texture->base.last_level &&
          !tiled_render_target && !tiled_depth_target && !staged_stencil &&
          !multisampled &&
          !util_format_is_compressed(texture->base.format)) {
         unsigned pitch = descriptor_stride / descriptor_format_size;

         if (pitch > texture->base.width0 && pitch <= UINT32_C(0x4000))
            descriptor[4] = pitch - 1u; /* GFX10.3 custom linear pitch. */
      }
      descriptor[5] = UINT32_C(0x00400000) |
                      ((multisampled ? 2u : descriptor_last_level) << 4);
      memcpy(&descriptor[8], sampler_state->words, sizeof(sampler_state->words));
      descriptor[11] = sampler_state->border_color_ptr |
                       ((uint32_t)sampler_state->border_color_type << 30);
#ifdef AGC_RUNTIME_DIAGNOSTICS
      if (sampler->compare_mode) {
         uint32_t first_depth;

         memcpy(&first_depth,
                texture->data + texture->level_offset[0],
                sizeof(first_depth));
         printf("[ps5-gallium] shadow-sample format=%u size=%ux%u last=%u depth=%08x descriptor=%08x/%08x/%08x/%08x/%08x/%08x sampler=%08x/%08x/%08x\n",
                texture->base.format, texture->base.width0,
                texture->base.height0, texture->base.last_level,
                first_depth, descriptor[0], descriptor[1], descriptor[2],
                descriptor[3], descriptor[4], descriptor[5], descriptor[8],
                descriptor[9], descriptor[10]);
      }
#endif
      flush_size = MAX2(flush_size, binding->offset + binding->stride);
      if (stencil_texture) {
          ps5_flush_texture_backing(
             slot == 1 && !merged_geometry ? flush_cache : NULL, 2 + unit,
             texture, texture->stencil_allocation_size, true);
      } else if (tiled_render_target || tiled_depth_target) {
         size_t tiled_size = multisampled
            ? tiled_depth_target
               ? ps5_tiled_rgba8_msaa4_surface_size(texture->base.width0,
                                                    texture->base.height0)
               : ps5_tiled_color_msaa4_surface_size(
                    texture->base.format, texture->base.width0,
                    texture->base.height0)
            : tiled_render_target
               ? ps5_tiled_color_surface_size(texture->base.format,
                                              texture->base.width0,
                                              texture->base.height0)
            : ps5_tiled_surface_size(texture->base.width0,
                                     texture->base.height0);

         /* Batch eligibility excludes attachment aliases and per-draw staging.
          * Array/MSAA backings obey the same retention and CPU-access drains. */
         ps5_flush_texture_backing(
            slot == 1 && !merged_geometry ? flush_cache : NULL, 2 + unit,
            texture,
            !multisampled && texture->base.target == PIPE_TEXTURE_2D
               ? tiled_size
               : tiled_depth_target || texture->base.target == PIPE_TEXTURE_2D_ARRAY
                  ? texture->allocation_size : tiled_size, false);
      } else {
         /* Eligible batches retain read-only linear fragment textures. Any CPU
          * texture access drains the batch, invalidating this flush cache. */
         ps5_flush_texture_backing(
            slot == 1 && !merged_geometry ? flush_cache : NULL, 2 + unit,
            texture, texture->size, false);
      }
   }
   if (texture_count != expected_texture_count) {
      printf("[ps5-gallium] texture-prepare reject=count actual=%u expected=%u bindings=%u\n",
             texture_count, expected_texture_count,
             metadata->descriptor_binding_count);
      return false;
   }
   user_data[metadata->descriptor_set0_user_data_dword] =
      (uint32_t)table_address;
   ps5_publish_descriptor_prefix(table->data, flush_size, pending_descriptor_bytes);
   return true;
}

int64_t sceKernelGetDirectMemorySize(void);
int32_t sceKernelAllocateDirectMemory(int64_t search_start, int64_t search_end,
                                     size_t length, size_t alignment,
                                     int memory_type, int64_t *physical_start);
int32_t sceKernelMapDirectMemory(void **address, size_t length, int protection,
                                int flags, int64_t physical_start,
                                size_t alignment);
int32_t sceKernelReleaseDirectMemory(int64_t physical_start, size_t length);
int ps5_agc_gate2_run(void) __attribute__((weak));
int ps5_agc_gate2_set_packages(const void *vs, size_t vs_size,
                               const void *ps, size_t ps_size)
   __attribute__((weak));
int ps5_agc_gate2_set_tessellation(const void *hs, size_t hs_size,
                                   uint32_t hs_rsrc2,
                                   uint32_t ls_hs_config,
                                   uint32_t tf_param,
                                   unsigned patch_vertices)
   __attribute__((weak));
int ps5_agc_gate2_set_ngg_control(uint32_t valid, uint32_t ge_pc_alloc)
   __attribute__((weak));
int ps5_agc_gate2_set_framebuffer(void *framebuffer, size_t size)
   __attribute__((weak));
int ps5_agc_gate2_set_framebuffers(void *const *framebuffers,
                                   const size_t *sizes, unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_scanout(void *framebuffer, size_t size)
   __attribute__((weak));
int ps5_agc_gate2_prepare_present(void *framebuffer, size_t size)
   __attribute__((weak));
int ps5_agc_gate2_wait_present(void) __attribute__((weak));
int ps5_agc_gate2_set_vertex_user_data(const uint32_t *values, unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_hull_user_data(const uint32_t *values, unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_index_buffer(const void *indices, unsigned index_count)
   __attribute__((weak));
int ps5_agc_gate2_set_index_buffer_typed(const void *indices,
                                         unsigned index_count,
                                         unsigned index_size)
   __attribute__((weak));
int ps5_agc_gate2_set_draw_state(uint32_t primitive_type,
                                 unsigned draw_count)
   __attribute__((weak));
int ps5_agc_gate2_set_instance_count(unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_pixel_user_data(const uint32_t *values, unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_depth_buffer(void *depth, size_t size,
                                   uint32_t depth_control)
   __attribute__((weak));
int ps5_agc_gate2_set_depth_stencil_buffer(
   void *depth, size_t depth_size, void *stencil, size_t stencil_size,
   uint32_t depth_control, uint32_t stencil_control,
   uint32_t stencil_refmask, uint32_t stencil_refmask_bf)
   __attribute__((weak));
int ps5_agc_gate2_set_graphics_state(
   uint32_t blend_control, uint32_t target_mask, uint32_t color_control,
   uint32_t color_control_valid,
   const uint32_t blend_color[4], const uint32_t viewport[8],
   const uint32_t scissor[2], uint32_t rasterizer_control,
   uint32_t rasterizer_valid, const uint32_t polygon_offset[6],
   uint32_t polygon_offset_valid) __attribute__((weak));
int ps5_agc_gate2_set_graphics_state_mrt(
   const uint32_t *blend_control, unsigned count, uint32_t target_mask,
   uint32_t color_control, uint32_t color_control_valid,
   const uint32_t blend_color[4], const uint32_t viewport[8],
   const uint32_t scissor[2], uint32_t rasterizer_control,
   uint32_t rasterizer_valid, const uint32_t polygon_offset[6],
   uint32_t polygon_offset_valid) __attribute__((weak));
int ps5_agc_gate2_set_viewport_states(
   const uint32_t viewport[PS5_MAX_VIEWPORTS][8],
   const uint32_t scissor[PS5_MAX_VIEWPORTS][2], unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_point_line_state(const uint32_t values[3],
                                       uint32_t valid)
   __attribute__((weak));
int ps5_agc_gate2_set_interp_control(uint32_t control, uint32_t valid)
   __attribute__((weak));
int ps5_agc_gate2_set_point_coord_input(uint32_t enabled)
   __attribute__((weak));
int ps5_agc_gate2_set_dual_source_blend(uint32_t enabled)
   __attribute__((weak));
int ps5_agc_gate2_set_multisample_state(unsigned samples,
                                        uint32_t sample_mask,
                                        uint32_t enabled,
                                        uint32_t alpha_to_coverage,
                                        uint32_t poly_line_smooth,
                                        uint32_t sample_shading)
   __attribute__((weak));
int ps5_agc_gate2_set_streamout(const void *vertex_shader,
                                 size_t vertex_shader_size,
                                uint32_t enabled_mask,
                                const uint32_t size_dwords[4],
                                const uint32_t stride_dwords[4],
                                const uint32_t offset_dwords[4])
   __attribute__((weak));
int ps5_agc_gate2_set_occlusion_query(void *query, size_t size, bool precise)
   __attribute__((weak));
int ps5_agc_gate2_set_clip_control(uint32_t control, uint32_t valid)
   __attribute__((weak));
int ps5_agc_gate2_set_vs_out_control(uint32_t control, uint32_t valid)
   __attribute__((weak));
int ps5_agc_gate2_set_color_to_texture_barrier(uint32_t enabled)
   __attribute__((weak));
int ps5_agc_gate2_set_depth_to_texture_barrier(uint32_t enabled)
   __attribute__((weak));
int ps5_agc_gate2_set_depth_target_extents(uint32_t width, uint32_t height)
   __attribute__((weak));
int ps5_agc_gate2_set_depth_target_view(uint32_t view)
   __attribute__((weak));
int ps5_agc_gate2_set_color_target_info(const uint32_t *values,
                                        unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_color_target_extents(const uint32_t *widths,
                                           const uint32_t *heights,
                                           unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_color_target_views(const uint32_t *views,
                                         unsigned count)
   __attribute__((weak));
int ps5_agc_gate2_set_color_target_layouts(void *const *targets, const size_t *sizes,
   const uint32_t *infos, const uint32_t *widths, const uint32_t *heights,
   const uint32_t *pitches, unsigned count) __attribute__((weak));
int ps5_agc_gate2_set_border_color_table(const void *table, size_t size)
   __attribute__((weak));

static const char *
ps5_get_name(struct pipe_screen *screen)
{
   (void)screen;
   return "PS5 AGC";
}

static const char *
ps5_get_vendor(struct pipe_screen *screen)
{
   (void)screen;
   return "PS5 homebrew";
}

static const char *
ps5_get_device_vendor(struct pipe_screen *screen)
{
   (void)screen;
   return "AMD";
}

static bool
ps5_resource_layout(const struct pipe_resource *templ, size_t *size,
                    unsigned *stride)
{
   size_t total = 0;
   unsigned width;
   unsigned height;
   unsigned level;
   bool linear_sampled;
   bool cube;
   bool cube_array;
   bool texture_1d;
   bool array_1d;
   bool array_2d;
   bool texture_3d;
   bool rectangle;
   bool multisample;
   bool depth_target;
   unsigned storage_layers;

   if (!templ || !templ->width0 ||
       templ->last_level >= PIPE_MAX_TEXTURE_LEVELS)
      return false;

   multisample = templ->nr_samples > 1 || templ->nr_storage_samples > 1;
   if (multisample &&
       (!PS5_ENABLE_MSAA4_CANDIDATE ||
        (templ->target != PIPE_TEXTURE_2D &&
         !(PS5_ENABLE_MSAA_ARRAY_CANDIDATE &&
           templ->target == PIPE_TEXTURE_2D_ARRAY)) ||
        templ->nr_samples != 4 || templ->nr_storage_samples != 4 ||
        templ->last_level || templ->depth0 != 1 ||
        (templ->target == PIPE_TEXTURE_2D && templ->array_size != 1) ||
        !(ps5_msaa4_color_support(
             templ->format, templ->target, templ->nr_samples,
             templ->nr_storage_samples, templ->bind) ||
          ps5_msaa4_depth_support(
             templ->format, templ->target, templ->nr_samples,
             templ->nr_storage_samples, templ->bind))))
      return false;

   if (templ->target == PIPE_BUFFER) {
      if (templ->height0 != 1 || templ->depth0 != 1 || templ->array_size != 1)
         return false;
      *stride = templ->width0;
      *size = templ->width0;
      return true;
   }

   texture_1d = PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
                templ->target == PIPE_TEXTURE_1D;
   array_1d = PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
              templ->target == PIPE_TEXTURE_1D_ARRAY;
   cube = PS5_ENABLE_TEXTURE_CUBE_CANDIDATE &&
          templ->target == PIPE_TEXTURE_CUBE;
   cube_array = PS5_ENABLE_TEXTURE_CUBE_CANDIDATE &&
                PS5_ENABLE_TEXTURE_CUBE_ARRAY_CANDIDATE &&
                templ->target == PIPE_TEXTURE_CUBE_ARRAY;
   array_2d = PS5_ENABLE_TEXTURE_ARRAY_CANDIDATE &&
              templ->target == PIPE_TEXTURE_2D_ARRAY;
   texture_3d = PS5_ENABLE_TEXTURE_3D_CANDIDATE &&
                templ->target == PIPE_TEXTURE_3D;
   rectangle = PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE &&
               templ->target == PIPE_TEXTURE_RECT;
   depth_target =
      (ps5_depth_render_target(templ->target) &&
       (templ->format == PIPE_FORMAT_Z32_FLOAT ||
        templ->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) &&
       (templ->bind & PIPE_BIND_DEPTH_STENCIL) &&
       !(templ->bind & ~(PIPE_BIND_DEPTH_STENCIL |
                         PIPE_BIND_SAMPLER_VIEW))) ||
      ps5_msaa4_depth_support(
         templ->format, templ->target, templ->nr_samples,
         templ->nr_storage_samples, templ->bind);
   if ((templ->target != PIPE_TEXTURE_2D && !texture_1d && !array_1d &&
        !rectangle && !cube && !cube_array && !array_2d && !texture_3d) ||
       !templ->height0 ||
       templ->width0 > PS5_MAX_TEXTURE_2D_SIZE ||
       templ->height0 > PS5_MAX_TEXTURE_2D_SIZE ||
       (!texture_3d && templ->depth0 != 1) ||
       !ps5_texture_format_size(templ->format))
      return false;
   if ((texture_1d && (templ->height0 != 1 ||
                       templ->array_size != 1 ||
                       (!depth_target &&
                        (!ps5_sampled_texture_format(templ->format) ||
                         !ps5_sampled_resource_bind(templ))))) ||
       (array_1d && (templ->height0 != 1 || !templ->array_size ||
                     templ->array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS ||
                     (!depth_target &&
                      (!ps5_sampled_texture_format(templ->format) ||
                       !ps5_sampled_resource_bind(templ))))) ||
       (!texture_1d && !array_1d && !cube && !cube_array && !array_2d &&
        !texture_3d &&
        templ->array_size != 1) ||
       (cube && (templ->array_size != 6 ||
                 templ->width0 > PS5_MAX_TEXTURE_CUBE_SIZE ||
                 templ->width0 != templ->height0 ||
                 (!depth_target &&
                  (!ps5_sampled_texture_format(templ->format) ||
                   !ps5_sampled_resource_bind(templ))))) ||
       (cube_array &&
        (!templ->array_size || templ->array_size % 6 ||
         templ->array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS ||
         templ->width0 > PS5_MAX_TEXTURE_CUBE_SIZE ||
         templ->width0 != templ->height0 ||
         !ps5_sampled_texture_format(templ->format) ||
         !ps5_sampled_resource_bind(templ))) ||
       (array_2d && (!templ->array_size ||
                     templ->array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS ||
                     !ps5_sampled_texture_format(templ->format) ||
                     (!depth_target &&
                      !ps5_msaa4_color_support(
                         templ->format, templ->target, templ->nr_samples,
                         templ->nr_storage_samples, templ->bind) &&
                      !ps5_sampled_resource_bind(templ) &&
                      !ps5_compute_image_array_resource(templ)))) ||
       (texture_3d && (templ->array_size != 1 || !templ->depth0 ||
                       templ->depth0 > PS5_MAX_TEXTURE_3D_SIZE ||
                       templ->width0 > PS5_MAX_TEXTURE_3D_SIZE ||
                       templ->height0 > PS5_MAX_TEXTURE_3D_SIZE ||
                       (!depth_target &&
                        (!ps5_sampled_texture_format(templ->format) ||
                         !ps5_sampled_resource_bind(templ))))))
      return false;

   if (templ->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
       !PS5_ENABLE_PACKED_DEPTH_STENCIL)
      return false;

   if ((templ->format == PIPE_FORMAT_Z32_FLOAT ||
        templ->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) &&
       (templ->bind & PIPE_BIND_DEPTH_STENCIL) && templ->last_level &&
       !ps5_depth_staging_required(templ))
      return false;
   if (rectangle && templ->last_level)
      return false;

   linear_sampled = ps5_linear_sampled_layout(templ);
   if (linear_sampled &&
       (templ->last_level > 15 ||
        (templ->last_level && !PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE)))
      return false;

   if (linear_sampled) {
      for (level = templ->last_level + 1; level-- > 0;) {
         size_t row;
         size_t level_size;
         unsigned rows;

         width = ps5_linear_mip_storage_extent(templ->width0, level);
         height = ps5_linear_mip_storage_extent(templ->height0, level);
         row = util_format_get_stride(templ->format, width);
         row = (row + 255u) & ~(size_t)255u;
         rows = util_format_get_nblocksy(templ->format, height);
         if (row > UINT32_MAX || row > SIZE_MAX / rows)
            return false;
         level_size = row * rows;
         if (total > SIZE_MAX - level_size)
            return false;
         if (!level)
            *stride = (unsigned)row;
         total += level_size;
      }
      storage_layers = texture_3d ? templ->depth0 : templ->array_size;
      if (total > SIZE_MAX / storage_layers)
         return false;
      *size = total * storage_layers;
      return true;
   }

   width = templ->width0;
   height = templ->height0;
   for (level = 0; level <= templ->last_level; ++level) {
      size_t row = util_format_get_stride(templ->format, width);
      size_t level_size;
      unsigned rows = util_format_get_nblocksy(templ->format, height);

      if (row > UINT32_MAX || row > SIZE_MAX / rows)
         return false;
      level_size = row * rows;
      if (total > SIZE_MAX - level_size)
         return false;
      if (!level)
         *stride = (unsigned)row;
      total += level_size;
      width = width > 1 ? width >> 1 : 1;
      height = height > 1 ? height >> 1 : 1;
   }

   if (array_2d || depth_target) {
      storage_layers = ps5_texture_level_layers(templ, 0);
      if (total > SIZE_MAX / storage_layers)
         return false;
      total *= storage_layers;
   }

   if (multisample) {
      if (total > SIZE_MAX / 4u)
         return false;
      total *= 4u;
   }
   *size = total;
   return true;
}

static bool
ps5_can_create_resource(struct pipe_screen *screen,
                        const struct pipe_resource *templ)
{
   struct pipe_resource proxy;
   size_t size;
   unsigned stride;

   (void)screen;
   if (ps5_resource_layout(templ, &size, &stride))
      return true;

   /* Mesa's mutable-texture proxy supplies bind=0. Validate its sampled
    * dimensions with the binding real OpenGL texture allocation will carry,
    * rounding a requested minimum sample count to native 4x as allocation does.
    * It can also guess a complete mip chain even when the application has
    * clamped the real texture to level 0, so retry the valid base image.
    */
   if (!templ || templ->bind)
      return false;
   proxy = *templ;
   if (proxy.nr_samples > 1 && proxy.nr_samples <= 4 &&
       proxy.nr_samples == proxy.nr_storage_samples) {
      proxy.nr_samples = 4;
      proxy.nr_storage_samples = 4;
   }
   proxy.bind = PIPE_BIND_SAMPLER_VIEW;
   if (ps5_resource_layout(&proxy, &size, &stride))
      return true;
   if (!proxy.last_level)
      return false;
   proxy.last_level = 0;
   return ps5_resource_layout(&proxy, &size, &stride);
}

static bool
ps5_render_arena_range_is_free(const struct ps5_screen *screen,
                               unsigned first, unsigned count)
{
   for (unsigned slot = first; slot < first + count; ++slot) {
      if (screen->render_arena_bitmap[slot / 64u] &
          (UINT64_C(1) << (slot % 64u)))
         return false;
   }
   return true;
}

static void
ps5_render_arena_mark(struct ps5_screen *screen, unsigned first,
                      unsigned count, bool used)
{
   for (unsigned slot = first; slot < first + count; ++slot) {
      uint64_t mask = UINT64_C(1) << (slot % 64u);

      if (used)
         screen->render_arena_bitmap[slot / 64u] |= mask;
      else
         screen->render_arena_bitmap[slot / 64u] &= ~mask;
   }
}

static bool
ps5_render_arena_allocate(struct ps5_screen *screen,
                          struct ps5_resource *resource, size_t size,
                          size_t alignment)
{
   struct ps5_resource *pool =
      (struct ps5_resource *)screen->render_pool;
   unsigned slots;
   unsigned slot_alignment;

   if (!pool || !pool->data || !size ||
       size > PS5_RENDER_POOL_BYTES - PS5_RENDER_ARENA_OFFSET ||
       size % PS5_RENDER_ARENA_SLOT_BYTES ||
       alignment < PS5_RENDER_ARENA_SLOT_BYTES ||
       alignment % PS5_RENDER_ARENA_SLOT_BYTES)
      return false;
   slots = (unsigned)(size / PS5_RENDER_ARENA_SLOT_BYTES);
   slot_alignment = (unsigned)(alignment / PS5_RENDER_ARENA_SLOT_BYTES);
   for (unsigned first = 0;
        first + slots <= PS5_RENDER_ARENA_SLOT_COUNT; ++first) {
      if (screen->render_arena_bitmap[first / 64u] == UINT64_MAX) {
         first |= 63u;
         continue;
      }
      if (first % slot_alignment ||
          !ps5_render_arena_range_is_free(screen, first, slots))
         continue;
      ps5_render_arena_mark(screen, first, slots, true);
      resource->render_arena_first_slot = first;
      resource->render_arena_slot_count = slots;
      resource->data = pool->data + PS5_RENDER_ARENA_OFFSET +
                       first * PS5_RENDER_ARENA_SLOT_BYTES;
      resource->allocation_size = size;
      pipe_resource_reference(&resource->render_pool_owner,
                              screen->render_pool);
      memset(resource->data, 0, size);
#ifdef AGC_RUNTIME_DIAGNOSTICS
      printf("[ps5-gallium] shared-resource offset=%zu bytes=%zu first=%u count=%u alignment=%zu\n",
             PS5_RENDER_ARENA_OFFSET +
                first * (size_t)PS5_RENDER_ARENA_SLOT_BYTES,
             size, first, slots, alignment);
#endif
      return true;
   }
   return false;
}

static void
ps5_release_resource_memory(void *address, size_t bytes, int64_t direct)
{
   int unmap_status = address ? munmap(address, bytes) : 0;
   int32_t release_status = 0;

   if (!unmap_status && direct >= 0)
      release_status = sceKernelReleaseDirectMemory(direct, bytes);
   if (unmap_status || release_status) {
      /* ponytail: fail-stop until failed ownership transfers have a recovery
       * owner; resource_destroy cannot report failure or safely permit reuse. */
      fprintf(stderr, "[ps5-gallium] resource-release failed unmap=%08x release=%08x; terminating before cleanup\n",
              (unsigned)unmap_status, (unsigned)release_status);
      fflush(stderr);
      _Exit(EXIT_FAILURE);
   }
}

static struct pipe_resource *
ps5_resource_create_unlocked(struct pipe_screen *screen,
                             const struct pipe_resource *templ)
{
   struct ps5_screen *ps5 = (struct ps5_screen *)screen;
   struct ps5_resource *resource;
   int64_t direct_limit;
   int allocation_status;
   int map_status;
   size_t size;
   size_t allocation_size;
   size_t allocation_alignment = PS5_DIRECT_ALIGNMENT;
   size_t render_staging_offset = 0;
   size_t render_staging_size = 0;
   size_t depth_staging_offset = 0;
   size_t depth_staging_size = 0;
   bool render_staging;
   bool depth_staging;
   unsigned stride;

   if (!ps5_resource_layout(templ, &size, &stride)) {
      printf("[ps5-gallium] reject-resource target=%u format=%u size=%ux%ux%u array=%u last=%u samples=%u/%u bind=%08x flags=%08x\n",
             templ ? templ->target : 0, templ ? templ->format : 0,
             templ ? templ->width0 : 0, templ ? templ->height0 : 0,
             templ ? templ->depth0 : 0, templ ? templ->array_size : 0,
             templ ? templ->last_level : 0, templ ? templ->nr_samples : 0,
             templ ? templ->nr_storage_samples : 0,
             templ ? templ->bind : 0, templ ? templ->flags : 0);
      return NULL;
   }

   resource = calloc(1, sizeof(*resource));
   if (!resource) {
#ifdef PS5_PUBLIC_STENCIL_TEST
      printf("[ps5-gallium] public-stencil-resource-failure stage=metadata format=%u\n",
             templ->format);
#endif
      return NULL;
   }
   resource->direct_start = -1;
   resource->stencil_direct_start = -1;
   render_staging = ps5_render_staging_required(templ);
   depth_staging = ps5_depth_staging_required(templ);

   if ((templ->bind & PIPE_BIND_DISPLAY_TARGET) &&
       ((PS5_ENABLE_SHARED_RENDER_POOL_CANDIDATE && ps5->render_pool) ||
        templ->width0 != PS5_RENDER_WIDTH ||
        templ->height0 != PS5_RENDER_HEIGHT)) {
      free(resource);
      return NULL;
   }

   if (size > SIZE_MAX - (PS5_DIRECT_ALIGNMENT - 1)) {
      free(resource);
      return NULL;
   }
   if ((templ->target == PIPE_TEXTURE_2D ||
        (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
         templ->target == PIPE_TEXTURE_2D_ARRAY)) &&
       ((templ->bind & PIPE_BIND_RENDER_TARGET) ||
        (templ->nr_samples == 4 &&
         ps5_msaa4_color_format(templ->format))) &&
       !ps5_linear_sampled_layout(templ)) {
      if ((!(PS5_ENABLE_PADDED_FBO_CANDIDATE ||
             PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) &&
           (templ->width0 != PS5_RENDER_WIDTH ||
            templ->height0 != PS5_RENDER_HEIGHT)) ||
          ((PS5_ENABLE_PADDED_FBO_CANDIDATE ||
            PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) &&
           (templ->width0 > PS5_MAX_COLOR_WIDTH ||
            templ->height0 > PS5_MAX_COLOR_HEIGHT))) {
         free(resource);
         return NULL;
      }
      allocation_size = templ->bind & PIPE_BIND_DISPLAY_TARGET
                           ? PS5_RENDER_POOL_BYTES
                           : PS5_RENDER_TARGET_BYTES;
      if (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE &&
          !(templ->bind & PIPE_BIND_DISPLAY_TARGET)) {
         size_t tiled_size = templ->nr_samples == 4
            ? ps5_tiled_color_msaa4_surface_size(
                 templ->format, templ->width0, templ->height0)
            : ps5_tiled_color_surface_size(templ->format, templ->width0,
                                           templ->height0);

         if (templ->target == PIPE_TEXTURE_2D_ARRAY) {
            if (tiled_size > SIZE_MAX / templ->array_size) {
               free(resource);
               return NULL;
            }
            tiled_size *= templ->array_size;
         }

         allocation_size =
            (tiled_size + PS5_RENDER_ALIGNMENT - 1u) &
            ~(size_t)(PS5_RENDER_ALIGNMENT - 1u);
      }
#ifdef PS5_PUBLIC_STENCIL_TEST
      if (!(templ->bind & PIPE_BIND_DISPLAY_TARGET))
         printf("[ps5-gallium] public-stencil-offscreen-color bytes=%zu buffers=1\n",
                allocation_size);
#endif
      allocation_alignment = PS5_RENDER_ALIGNMENT;
   } else if (render_staging) {
      const unsigned layers = ps5_texture_level_layers(templ, 0);
      size_t combined;

      render_staging_size = ps5_tiled_color_surface_size(
         templ->format, templ->width0, templ->height0);
      if (!render_staging_size || render_staging_size > SIZE_MAX / layers) {
         free(resource);
         return NULL;
      }
      render_staging_size *= layers;
      render_staging_offset =
         (size + PS5_COLOR_TARGET_ALIGNMENT - 1u) &
         ~(size_t)(PS5_COLOR_TARGET_ALIGNMENT - 1u);
      if (render_staging_offset < size ||
          render_staging_size > SIZE_MAX - render_staging_offset) {
         free(resource);
         return NULL;
      }
      combined = render_staging_offset + render_staging_size;
      if (combined > SIZE_MAX - (PS5_COLOR_TARGET_ALIGNMENT - 1u)) {
         free(resource);
         return NULL;
      }
      allocation_size =
         (combined + PS5_COLOR_TARGET_ALIGNMENT - 1u) &
         ~(size_t)(PS5_COLOR_TARGET_ALIGNMENT - 1u);
      allocation_alignment = PS5_COLOR_TARGET_ALIGNMENT;
   } else if (depth_staging) {
      const unsigned layers = ps5_texture_level_layers(templ, 0);
      size_t combined;

      depth_staging_size = ps5_tiled_depth_surface_size(
         templ->width0, templ->height0, templ->nr_samples);
      if (depth_staging_size > SIZE_MAX / layers) {
         free(resource);
         return NULL;
      }
      depth_staging_size *= layers;
      if (templ->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
         size_t sample_layer_size;
         unsigned sample_stride;

         if (!ps5_packed_depth_sample_layout(
                templ, &sample_layer_size, &sample_stride) ||
             sample_layer_size > SIZE_MAX / layers) {
            free(resource);
            return NULL;
         }
         depth_staging_size = MAX2(depth_staging_size,
                                   sample_layer_size * layers);
      }
      depth_staging_offset =
         (size + PS5_RENDER_ALIGNMENT - 1u) &
         ~(size_t)(PS5_RENDER_ALIGNMENT - 1u);
      if (depth_staging_offset < size ||
          depth_staging_size > SIZE_MAX - depth_staging_offset) {
         free(resource);
         return NULL;
      }
      combined = depth_staging_offset + depth_staging_size;
      if (combined > SIZE_MAX - (PS5_RENDER_ALIGNMENT - 1u)) {
         free(resource);
         return NULL;
      }
      allocation_size =
         (combined + PS5_RENDER_ALIGNMENT - 1u) &
         ~(size_t)(PS5_RENDER_ALIGNMENT - 1u);
      allocation_alignment = PS5_RENDER_ALIGNMENT;
   } else if ((ps5_depth_render_target(templ->target) &&
               (templ->bind & PIPE_BIND_DEPTH_STENCIL)) ||
              ps5_msaa4_depth_support(
                 templ->format, templ->target, templ->nr_samples,
                 templ->nr_storage_samples, templ->bind)) {
      const unsigned layers = ps5_texture_level_layers(templ, 0);
      size_t tiled_size = PS5_DEPTH_TARGET_BYTES;

      if ((!PS5_ENABLE_PADDED_FBO_CANDIDATE &&
           (templ->width0 != PS5_RENDER_WIDTH ||
            templ->height0 != PS5_RENDER_HEIGHT)) ||
          (PS5_ENABLE_PADDED_FBO_CANDIDATE &&
           (templ->width0 > PS5_MAX_DEPTH_WIDTH ||
            templ->height0 > PS5_MAX_DEPTH_HEIGHT))) {
         free(resource);
         return NULL;
      }
      if (PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE)
         tiled_size = ps5_tiled_depth_surface_size(
            templ->width0, templ->height0, templ->nr_samples);
      if (tiled_size > SIZE_MAX / layers ||
          tiled_size * layers > SIZE_MAX - (PS5_RENDER_ALIGNMENT - 1u)) {
         free(resource);
         return NULL;
      }
      allocation_size =
         (tiled_size * layers + PS5_RENDER_ALIGNMENT - 1u) &
         ~(size_t)(PS5_RENDER_ALIGNMENT - 1u);
      allocation_alignment = PS5_RENDER_ALIGNMENT;
   } else {
      allocation_size = (size + PS5_DIRECT_ALIGNMENT - 1) &
                        ~(size_t)(PS5_DIRECT_ALIGNMENT - 1);
   }
   /* Keep persistent textures from displacing transient buffer allocations. */
   if (PS5_ENABLE_SHARED_RENDER_POOL_CANDIDATE && ps5->render_pool &&
       templ->target == PIPE_BUFFER &&
       !(templ->bind & PIPE_BIND_DISPLAY_TARGET) &&
       templ->format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
      if (!ps5_render_arena_allocate(ps5, resource, allocation_size,
                                     allocation_alignment)) {
         printf("[ps5-gallium] shared-resource exhausted bytes=%zu alignment=%zu\n",
                allocation_size, allocation_alignment);
         /* The arena is an allocation fast path, not a resource-size limit.
          * Fall through to the same checked direct allocation as staging/MSAA. */
      } else {
         goto primary_ready;
      }
   }
   direct_limit = sceKernelGetDirectMemorySize();
#ifdef PS5_PUBLIC_STENCIL_TEST
   if (templ->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
      const unsigned layers = ps5_texture_level_layers(templ, 0);
      const size_t layer_size = PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE
         ? ps5_tiled_stencil_surface_size_samples(
              templ->width0, templ->height0, templ->nr_samples)
         : PS5_STENCIL_TARGET_BYTES;

      if (layer_size > SIZE_MAX / layers) {
         free(resource);
         return NULL;
      }
      resource->stencil_allocation_size = layer_size * layers;
      printf("[ps5-gallium] public-stencil-allocation-order stencil-first\n");
      allocation_status = sceKernelAllocateDirectMemory(
         0, direct_limit, resource->stencil_allocation_size,
         PS5_STENCIL_ALIGNMENT, PS5_DIRECT_MEMORY_TYPE,
         &resource->stencil_direct_start);
      if (allocation_status != 0) {
         printf("[ps5-gallium] public-stencil-resource-failure stage=stencil-allocate rc=%08x format=%u bytes=%zu alignment=%u limit=%016llx\n",
                (unsigned)allocation_status, templ->format,
                resource->stencil_allocation_size, PS5_STENCIL_ALIGNMENT,
                (unsigned long long)direct_limit);
         free(resource);
         return NULL;
      }
      map_status = sceKernelMapDirectMemory(
         (void **)&resource->stencil_data,
         resource->stencil_allocation_size, PS5_MAP_PROTECTION, 0,
         resource->stencil_direct_start, PS5_STENCIL_ALIGNMENT);
      if (map_status != 0 || !resource->stencil_data) {
         printf("[ps5-gallium] public-stencil-resource-failure stage=stencil-map rc=%08x format=%u direct=%016llx bytes=%zu alignment=%u address=%p\n",
                (unsigned)map_status, templ->format,
                (unsigned long long)resource->stencil_direct_start,
                resource->stencil_allocation_size, PS5_STENCIL_ALIGNMENT,
                resource->stencil_data);
         ps5_release_resource_memory(resource->stencil_data,
            resource->stencil_allocation_size, resource->stencil_direct_start);
         free(resource);
         return NULL;
      }
      memset(resource->stencil_data, 0,
             resource->stencil_allocation_size);
   }
#endif
   allocation_status = sceKernelAllocateDirectMemory(
      0, direct_limit, allocation_size, allocation_alignment,
      PS5_DIRECT_MEMORY_TYPE, &resource->direct_start);
   if (allocation_status != 0) {
      static unsigned failures;
      if (failures++ < 8)
         printf("[ps5-gallium] resource-allocate-failed rc=%08x target=%u format=%u bytes=%zu alignment=%zu limit=%016llx\n",
                (unsigned)allocation_status, templ->target, templ->format,
                allocation_size, allocation_alignment, (unsigned long long)direct_limit);
#ifdef PS5_PUBLIC_STENCIL_TEST
      printf("[ps5-gallium] public-stencil-resource-failure stage=depth-allocate rc=%08x format=%u bytes=%zu alignment=%zu limit=%016llx\n",
             (unsigned)allocation_status, templ->format, allocation_size,
             allocation_alignment, (unsigned long long)direct_limit);
#endif
#ifdef PS5_PUBLIC_STENCIL_TEST
      ps5_release_resource_memory(resource->stencil_data,
         resource->stencil_allocation_size, resource->stencil_direct_start);
#endif
      free(resource);
      return NULL;
   }

   map_status = sceKernelMapDirectMemory(
      (void **)&resource->data, allocation_size, PS5_MAP_PROTECTION, 0,
      resource->direct_start, allocation_alignment);
   if (map_status != 0 || !resource->data) {
      static unsigned failures;
      if (failures++ < 8)
         printf("[ps5-gallium] resource-map-failed rc=%08x target=%u format=%u bytes=%zu direct=%016llx\n",
                (unsigned)map_status, templ->target, templ->format,
                allocation_size, (unsigned long long)resource->direct_start);
#ifdef PS5_PUBLIC_STENCIL_TEST
      printf("[ps5-gallium] public-stencil-resource-failure stage=depth-map rc=%08x format=%u direct=%016llx bytes=%zu alignment=%zu address=%p\n",
             (unsigned)map_status, templ->format,
             (unsigned long long)resource->direct_start, allocation_size,
             allocation_alignment, resource->data);
#endif
      ps5_release_resource_memory(resource->data, allocation_size,
                                  resource->direct_start);
#ifdef PS5_PUBLIC_STENCIL_TEST
      ps5_release_resource_memory(resource->stencil_data,
         resource->stencil_allocation_size, resource->stencil_direct_start);
#endif
      free(resource);
      return NULL;
   }
   memset(resource->data, 0, allocation_size);
primary_ready:
   if (templ->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
#ifndef PS5_PUBLIC_STENCIL_TEST
      const unsigned layers = ps5_texture_level_layers(templ, 0);
      const size_t layer_size = PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE
         ? ps5_tiled_stencil_surface_size_samples(
              templ->width0, templ->height0, templ->nr_samples)
         : PS5_STENCIL_TARGET_BYTES;

      if (layer_size > SIZE_MAX / layers)
         goto fail_primary;
      resource->stencil_allocation_size = layer_size * layers;
      allocation_status = sceKernelAllocateDirectMemory(
         0, direct_limit, resource->stencil_allocation_size,
         PS5_STENCIL_ALIGNMENT, PS5_DIRECT_MEMORY_TYPE,
         &resource->stencil_direct_start);
      if (allocation_status != 0) {
#ifdef PS5_PUBLIC_STENCIL_TEST
         printf("[ps5-gallium] public-stencil-resource-failure stage=stencil-allocate rc=%08x format=%u bytes=%zu alignment=%u limit=%016llx\n",
                (unsigned)allocation_status, templ->format,
                resource->stencil_allocation_size, PS5_STENCIL_ALIGNMENT,
                (unsigned long long)direct_limit);
#endif
         goto fail_primary;
      }
      map_status = sceKernelMapDirectMemory(
         (void **)&resource->stencil_data,
         resource->stencil_allocation_size, PS5_MAP_PROTECTION, 0,
         resource->stencil_direct_start, PS5_STENCIL_ALIGNMENT);
      if (map_status != 0 || !resource->stencil_data) {
#ifdef PS5_PUBLIC_STENCIL_TEST
         printf("[ps5-gallium] public-stencil-resource-failure stage=stencil-map rc=%08x format=%u direct=%016llx bytes=%zu alignment=%u address=%p\n",
                (unsigned)map_status, templ->format,
                (unsigned long long)resource->stencil_direct_start,
                resource->stencil_allocation_size, PS5_STENCIL_ALIGNMENT,
                resource->stencil_data);
#endif
         goto fail_stencil;
      }
      memset(resource->stencil_data, 0, resource->stencil_allocation_size);
#endif
#ifdef PS5_PUBLIC_STENCIL_TEST
      printf("[ps5-gallium] public-stencil-resource depth=%p/%zu direct=%016llx stencil=%p/%zu direct=%016llx\n",
             resource->data, allocation_size,
             (unsigned long long)resource->direct_start,
             resource->stencil_data, resource->stencil_allocation_size,
             (unsigned long long)resource->stencil_direct_start);
#endif
   }
   resource->base = *templ;
   resource->base.reference.count = 1;
   resource->base.screen = screen;
   resource->size = size;
   resource->allocation_size = allocation_size;
   resource->stride = stride;
   resource->render_staging_offset = render_staging_offset;
   resource->render_staging_size = render_staging_size;
   resource->depth_staging_offset = depth_staging_offset;
   resource->depth_staging_size = depth_staging_size;
   if (PS5_ENABLE_SHARED_RENDER_POOL_CANDIDATE &&
       (templ->bind & PIPE_BIND_DISPLAY_TARGET)) {
      ps5->render_pool = &resource->base;
      printf("[ps5-gallium] shared-render-pool bytes=%zu arena-offset=%u arena-bytes=%zu\n",
             allocation_size, PS5_RENDER_ARENA_OFFSET,
             (size_t)(PS5_RENDER_POOL_BYTES - PS5_RENDER_ARENA_OFFSET));
   }
   if (templ->target == PIPE_BUFFER) {
      resource->level_stride[0] = stride;
      resource->layer_stride = size;
   } else {
      const unsigned bytes_per_pixel = ps5_texture_format_size(templ->format);
      unsigned level;
      size_t offset = 0;
      bool linear_sampled = ps5_linear_sampled_layout(templ);

      if (linear_sampled) {
         for (level = templ->last_level + 1; level-- > 0;) {
            unsigned width =
               ps5_linear_mip_storage_extent(templ->width0, level);
            unsigned height =
               ps5_linear_mip_storage_extent(templ->height0, level);

            resource->level_offset[level] = offset;
            resource->level_stride[level] =
               (util_format_get_stride(templ->format, width) + 255u) &
               ~255u;
            offset += util_format_get_2d_size(
               templ->format, resource->level_stride[level], height);
         }
      } else {
         unsigned width = templ->width0;
         unsigned height = templ->height0;

         for (level = 0; level <= templ->last_level; ++level) {
            resource->level_offset[level] = offset;
            resource->level_stride[level] =
               util_format_get_stride(templ->format, width);
            offset += util_format_get_2d_size(
               templ->format, resource->level_stride[level], height);
            width = width > 1 ? width >> 1 : 1;
            height = height > 1 ? height >> 1 : 1;
         }
      }
      resource->layer_stride = offset;
      if (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
          templ->target == PIPE_TEXTURE_2D_ARRAY &&
          ((templ->bind & PIPE_BIND_RENDER_TARGET) ||
           ps5_msaa4_color_support(
              templ->format, templ->target, templ->nr_samples,
              templ->nr_storage_samples, templ->bind)) &&
          !linear_sampled)
         resource->layer_stride = templ->nr_samples == 4
            ? ps5_tiled_color_msaa4_surface_size(
                 templ->format, templ->width0, templ->height0)
            : ps5_tiled_color_surface_size(templ->format, templ->width0,
                                           templ->height0);
      if (ps5_depth_render_target(templ->target) &&
          ((templ->bind & PIPE_BIND_DEPTH_STENCIL) ||
           ps5_msaa4_depth_support(
              templ->format, templ->target, templ->nr_samples,
              templ->nr_storage_samples, templ->bind)) &&
          !depth_staging)
         resource->layer_stride = ps5_tiled_depth_surface_size(
            templ->width0, templ->height0, templ->nr_samples);
      if (PS5_ENABLE_PADDED_FBO_CANDIDATE &&
          !PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE &&
          templ->target == PIPE_TEXTURE_2D &&
          (templ->bind & PIPE_BIND_RENDER_TARGET)) {
         resource->stride = PS5_RENDER_WIDTH * bytes_per_pixel;
         resource->level_stride[0] = resource->stride;
         resource->layer_stride =
            (size_t)resource->stride * PS5_RENDER_HEIGHT;
      }
   }
   return &resource->base;

#ifndef PS5_PUBLIC_STENCIL_TEST
fail_stencil:
   ps5_release_resource_memory(resource->stencil_data,
      resource->stencil_allocation_size, resource->stencil_direct_start);
fail_primary:
   ps5_release_resource_memory(resource->data, allocation_size,
                               resource->direct_start);
   free(resource);
   return NULL;
#endif
}

static struct pipe_resource *
ps5_resource_create(struct pipe_screen *screen,
                    const struct pipe_resource *templ)
{
   struct ps5_screen *ps5 = (struct ps5_screen *)screen;
   struct pipe_resource *resource;

   simple_mtx_lock(&ps5->resource_mutex);
   resource = ps5_resource_create_unlocked(screen, templ);
   simple_mtx_unlock(&ps5->resource_mutex);
   return resource;
}

/* A compute write leaves CPU-cached lines stale, and a partial CPU write to a
 * stale line would write old neighbouring bytes back over GPU results. Every
 * CPU access drains first, so invalidate there instead of after each dispatch. */
static void
ps5_invalidate_gpu_writes(struct ps5_resource *resource)
{
   if (!resource || !resource->gpu_written)
      return;
   resource->gpu_written = false;
   ps5_flush_gpu_data(resource->data, resource->allocation_size);
}

static int
ps5_resource_get_info(struct pipe_resource *base, void **address,
                  size_t *logical_size, size_t *allocation_size, bool cpu_export)
{
   struct ps5_resource *resource = (struct ps5_resource *)base;

   /* GPU users are ordered by the queue: compute submits behind queued
    * graphics, so only a CPU export waits and invalidates. */
   if (cpu_export) {
      ps5_draw_batch_drain();
      if (base && (base->bind & PIPE_BIND_DISPLAY_TARGET))
         ps5_draw_batch_drain_buffer(base);
      ps5_invalidate_gpu_writes(resource);
   }
   if (!resource)
      return -1;
   if (address) {
      if (cpu_export)
         resource->external_cpu_access = true;
      *address = resource->data;
   }
   if (logical_size)
      *logical_size = resource->size;
   if (allocation_size)
      *allocation_size = resource->allocation_size;
   return 0;
}
int
ps5_resource_info(struct pipe_resource *base, void **address,
                  size_t *logical_size, size_t *allocation_size)
{
   return ps5_resource_get_info(base, address, logical_size, allocation_size, true);
}

/* Internal GPU access does not export a writable CPU pointer. CPU writes must
 * still use transfers/subdata; existing persistent/exported flags remain set. */
int
ps5_resource_gpu_info(struct pipe_resource *base, void **address,
                      size_t *logical_size, size_t *allocation_size)
{
   return ps5_resource_get_info(base, address, logical_size, allocation_size, false);
}

int
ps5_screen_prepare_present(struct pipe_screen *base)
{
   struct ps5_screen *screen = (struct ps5_screen *)base;
   struct ps5_resource *pool = screen && screen->render_pool
      ? (struct ps5_resource *)screen->render_pool : NULL;

   /* Only the two scanouts are registered, not the trailing offscreen arena. */
   return pool && pool->data && pool->allocation_size >= PS5_SCANOUT_POOL_BYTES &&
          ps5_agc_gate2_prepare_present
      ? ps5_agc_gate2_prepare_present(pool->data, PS5_SCANOUT_POOL_BYTES)
      : -1;
}

static int
ps5_packed_depth_sampled_descriptor(struct pipe_resource *base,
                                    enum pipe_format view_format,
                                    unsigned first_level,
                                    unsigned last_level,
                                    uint32_t descriptor[8])
{
   if (!base || !descriptor)
      return -1;
   struct ps5_resource *resource = (struct ps5_resource *)base;
   const bool stencil = view_format == PIPE_FORMAT_X32_S8X24_UINT;
   if (stencil) {
      if (base->format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
          first_level > last_level || last_level > base->last_level)
         return -1;
      struct ps5_resource *sample =
         ps5_stage_packed_stencil_samples(resource);
      return sample ? ps5_resource_sampled_image_descriptor(
                         &sample->base, first_level, last_level, descriptor)
                    : -1;
   }
   const bool tiled_depth = PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE &&
      ps5_depth_render_target(base->target) &&
      (base->bind & PIPE_BIND_DEPTH_STENCIL) && !resource->depth_staging_size;
   if (base->format == PIPE_FORMAT_Z32_FLOAT && !tiled_depth)
      return ps5_resource_sampled_image_descriptor(
         base, first_level, last_level, descriptor);
   const bool staged_depth = base->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
                             !tiled_depth &&
                             resource->depth_staging_size;
   unsigned texel_size = base->format == PIPE_FORMAT_Z32_FLOAT || staged_depth
                            ? 4u : 8u;
   unsigned stride = staged_depth ? base->width0 * texel_size :
                                    resource->level_stride[0];
   size_t layer_size = resource->layer_stride;
   uint32_t format;

   if ((base->format != PIPE_FORMAT_Z32_FLOAT &&
        base->format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) ||
       view_format != base->format ||
       !ps5_sampled_texture_target(base->target) ||
       first_level > last_level || last_level > base->last_level ||
       base->nr_samples > 1 ||
       base->nr_storage_samples > 1 || !base->width0 || !base->height0 ||
       !ps5_texture_descriptor_format(view_format, &format) ||
       (staged_depth && !ps5_stage_packed_depth_samples(resource, &stride)))
      return -1;

   if (staged_depth &&
       !ps5_packed_depth_sample_layout(base, &layer_size, &stride))
      return -1;
   uintptr_t address = (uintptr_t)resource->data +
      (staged_depth ? resource->depth_staging_offset : 0);
   if ((address & 255u) || address >> 48)
      return -1;
   if (!tiled_depth && !staged_depth)
      format = gfx10_format_table[view_format].img_format << 20;
   memset(descriptor, 0, 8u * sizeof(*descriptor));
   descriptor[0] = address >> 8;
   descriptor[1] = format | (((base->width0 - 1u) & 3u) << 30) |
                   (uint32_t)(address >> 40);
   descriptor[2] = ((base->width0 - 1u) >> 2) |
                   ((base->height0 - 1u) << 14) | UINT32_C(0x80000000);
   const bool tiled = tiled_depth;
   descriptor[3] = (tiled ?
      (ps5_cube_texture_target(base->target) ? UINT32_C(0xb1800000) :
       base->target == PIPE_TEXTURE_2D_ARRAY ? UINT32_C(0xd1800000) :
       base->target == PIPE_TEXTURE_3D ? UINT32_C(0xa1800000) :
       UINT32_C(0x91800000)) :
      (ps5_cube_texture_target(base->target) ? UINT32_C(0xb0000000) :
       base->target == PIPE_TEXTURE_2D_ARRAY ? UINT32_C(0xd0000000) :
       base->target == PIPE_TEXTURE_3D ? UINT32_C(0xa0000000) :
       UINT32_C(0x90000000))) |
      (first_level << 12) | (last_level << 16);
   descriptor[4] = ps5_cube_texture_target(base->target) ||
                   base->target == PIPE_TEXTURE_2D_ARRAY
                      ? base->array_size - 1u
                   : base->target == PIPE_TEXTURE_3D
                      ? base->depth0 - 1u : 0;
   if (base->target == PIPE_TEXTURE_2D && !base->last_level && !tiled &&
       stride / texel_size > base->width0)
      descriptor[4] = stride / texel_size - 1u;
   descriptor[5] = UINT32_C(0x00400000) | (base->last_level << 4);
   return 0;
}

static int
ps5_resource_image_descriptor(struct pipe_resource *base, uint32_t descriptor[8],
                                     unsigned required_bind, unsigned first_level, unsigned last_level)
{
   const struct ps5_resource *resource = (const struct ps5_resource *)base;
   uint32_t format;
   const bool sampled = required_bind == PIPE_BIND_SAMPLER_VIEW;
   const bool srgb = base && sampled && base->format == PIPE_FORMAT_R8G8B8A8_SRGB;
   const bool depth = base && sampled && base->format == PIPE_FORMAT_Z32_FLOAT &&
                      ps5_linear_sampled_layout(base);
   const unsigned texel_size = depth || srgb ? 4 : base ? ps5_storage_image_texel_size(base->format) : 0;
   const bool multisampled = base && base->nr_samples == 4 && base->nr_storage_samples == 4;
   const bool one_d = base &&
      (base->target == PIPE_TEXTURE_1D || base->target == PIPE_TEXTURE_1D_ARRAY);
   const bool volume = base && base->target == PIPE_TEXTURE_3D;
   const bool rectangle = base && base->target == PIPE_TEXTURE_RECT;
   const bool cube = base && ps5_cube_texture_target(base->target);
   const bool array = base && (base->target == PIPE_TEXTURE_2D_ARRAY ||
                               (one_d && base->target == PIPE_TEXTURE_1D_ARRAY) || cube);
   if (!base || !descriptor || (!one_d && !volume && !rectangle && !cube &&
       base->target != PIPE_TEXTURE_2D && base->target != PIPE_TEXTURE_2D_ARRAY) ||
       !texel_size || !base->width0 || !base->height0 ||
       base->width0 > PS5_MAX_TEXTURE_2D_SIZE || base->height0 > PS5_MAX_TEXTURE_2D_SIZE ||
       (one_d && base->height0 != 1) ||
       (volume ? (!base->depth0 || base->depth0 > PS5_MAX_TEXTURE_3D_SIZE ||
                  base->width0 > PS5_MAX_TEXTURE_3D_SIZE || base->height0 > PS5_MAX_TEXTURE_3D_SIZE) : base->depth0 != 1) ||
       !base->array_size || base->array_size > PS5_MAX_TEXTURE_ARRAY_LAYERS ||
       (cube && (base->width0 != base->height0 || base->array_size % 6 ||
                 (base->target == PIPE_TEXTURE_CUBE && base->array_size != 6))) ||
       (!array && base->array_size != 1) || base->last_level >= PIPE_MAX_TEXTURE_LEVELS ||
       (rectangle && base->last_level) ||
       base->last_level > 15 || first_level > last_level || last_level > base->last_level ||
       (!multisampled && (base->nr_samples > 1 || base->nr_storage_samples > 1)) ||
       (!(base->bind & required_bind) &&
        !(required_bind == PIPE_BIND_SHADER_IMAGE && (base->bind & PIPE_BIND_SAMPLER_VIEW))) ||
       (base->bind & PIPE_BIND_DISPLAY_TARGET) ||
       ((base->bind & PIPE_BIND_DEPTH_STENCIL) && !depth) ||
       (resource->depth_staging_size && !depth) || !resource->data ||
       resource->size > resource->allocation_size ||
       ((uintptr_t)resource->data & 255u) || (uintptr_t)resource->data >> 48 ||
       !ps5_texture_descriptor_format(base->format, &format))
      return -1;
   const bool tiled = !ps5_linear_sampled_layout(base);
   if (multisampled) {
      const size_t logical_layer = (size_t)base->width0 * base->height0 * texel_size;
      const bool format_supported = sampled
         ? (base->format == PIPE_FORMAT_R32_FLOAT ||
            base->format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
            base->format == PIPE_FORMAT_R8G8B8A8_UNORM ||
            base->format == PIPE_FORMAT_R8G8B8A8_SINT ||
            base->format == PIPE_FORMAT_R8G8B8A8_UINT)
         : ps5_storage_image_texel_size(base->format) != 0;
      const unsigned sampled_binds = PIPE_BIND_SAMPLER_VIEW |
                                     PIPE_BIND_RENDER_TARGET;
      const bool binds_supported = sampled
         ? (base->bind & sampled_binds) == sampled_binds &&
           !(base->bind & ~sampled_binds)
         : ps5_msaa4_color_support(
              base->format, base->target, base->nr_samples,
              base->nr_storage_samples, base->bind);
      /* 4x MSAA halves both axes of the format-sized 64 KiB color tile.
       * Keep the same checked footprint. */
      const size_t physical_layer = ps5_tiled_color_surface_size(base->format, base->width0 * 2, base->height0 * 2);
      if (!tiled || (base->target != PIPE_TEXTURE_2D && base->target != PIPE_TEXTURE_2D_ARRAY) ||
          !format_supported ||
          base->last_level || !binds_supported ||
          resource->render_staging_size || resource->render_staging_offset ||
          ((uintptr_t)resource->data & (PS5_COLOR_TARGET_ALIGNMENT - 1u)) ||
          resource->level_offset[0] || resource->level_stride[0] != base->width0 * texel_size ||
          resource->size != logical_layer * 4 * base->array_size ||
          resource->layer_stride != (array ? physical_layer : logical_layer) ||
          !physical_layer || physical_layer > resource->allocation_size / base->array_size)
         return -1;
   } else if (tiled) {
      /* Reuse the single-level R8/RG8/RGBA8/RGBA16F layout used by graphics.
       * No depth, MSAA, compression metadata, aliases or staging are admitted. */
      const unsigned binds = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET;
      const size_t logical = (size_t)base->width0 * base->height0 * texel_size;
      const size_t physical = ps5_tiled_color_surface_size(base->format, base->width0, base->height0);
      if ((base->format != PIPE_FORMAT_R8G8B8A8_UNORM && !srgb &&
           base->format != PIPE_FORMAT_R8_UNORM && base->format != PIPE_FORMAT_R8G8_UNORM &&
           base->format != PIPE_FORMAT_R16G16B16A16_FLOAT) || base->target != PIPE_TEXTURE_2D ||
          base->last_level || base->array_size != 1 || (base->bind & binds) != binds ||
          (base->bind & ~(binds | PIPE_BIND_SHADER_IMAGE)) ||
          base->width0 > PS5_MAX_COLOR_WIDTH || base->height0 > PS5_MAX_COLOR_HEIGHT ||
          resource->render_staging_size || resource->render_staging_offset ||
          ((uintptr_t)resource->data & (PS5_COLOR_TARGET_ALIGNMENT - 1u)) ||
          resource->size != logical || resource->layer_stride != logical ||
          resource->level_offset[0] || resource->level_stride[0] != base->width0 * texel_size ||
          !physical || physical > resource->allocation_size)
         return -1;
   } else {
      if (depth && resource->depth_staging_size &&
          (resource->depth_staging_offset < resource->size ||
           resource->depth_staging_offset > resource->allocation_size ||
           resource->depth_staging_size > resource->allocation_size - resource->depth_staging_offset))
         return -1;
      /* Mesa's ordinary textures carry SAMPLER_VIEW|RENDER_TARGET, not an image
       * hint. Their linear data remains authoritative: draws stage it in/out,
       * and compute submission drains pending graphics before using this SRD.
       * Never resolve staging here: it may predate the latest compute write. */
      if (ps5_render_staging_required(base)) {
         if (!(base->bind & PIPE_BIND_SAMPLER_VIEW) || !resource->render_staging_size ||
             resource->render_staging_offset < resource->size ||
             (resource->render_staging_offset & (PS5_COLOR_TARGET_ALIGNMENT - 1u)) ||
             resource->render_staging_offset > resource->allocation_size ||
             resource->render_staging_size > resource->allocation_size - resource->render_staging_offset)
            return -1;
      } else if (resource->render_staging_size || resource->render_staging_offset) {
         return -1;
      }
      size_t offset = 0;
      for (unsigned level = base->last_level + 1; level-- > 0;) {
         const unsigned width = ps5_linear_mip_storage_extent(base->width0, level);
         const unsigned height = ps5_linear_mip_storage_extent(base->height0, level);
         const unsigned stride = (width * texel_size + 255u) & ~255u;
         const size_t span = (size_t)stride * height;
         if (resource->level_offset[level] != offset || resource->level_stride[level] != stride ||
             offset > resource->size || span > resource->size - offset)
            return -1;
         offset += span;
      }
      const unsigned layers = volume ? base->depth0 : base->array_size;
      if ((array || volume) &&
          (resource->layer_stride != offset || offset > resource->size / layers))
         return -1;
   }
   /* Same reverse-ordered linear mip storage and layer encoding as graphics.
    * Sampled and storage arrays share the advertised layer limit; storage
    * views validate sublayers below. */
   const uintptr_t address = (uintptr_t)resource->data;
   const unsigned pitch = resource->level_stride[0] / texel_size;
   const unsigned channels = depth ? 1 : srgb ? 4 : ps5_storage_image_channels(base->format);
   const uint32_t swizzle = channels == 1 ? 0x204u : channels == 2 ? 0x22cu :
                            channels == 3 ? 0x3acu : 0xfacu;
   const uint32_t srd[8] = {
      address >> 8,
      format | (((base->width0 - 1u) & 3u) << 30) | (uint32_t)(address >> 40),
      ((base->width0 - 1u) >> 2) | ((base->height0 - 1u) << 14) | UINT32_C(0x80000000),
      (multisampled ? (array ? UINT32_C(0xf1b20000) : UINT32_C(0xe1b20000)) :
       tiled ? UINT32_C(0x91b00000) : volume ? UINT32_C(0xa0000000) :
       sampled && cube ? UINT32_C(0xb0000000) :
       one_d ? (array ? UINT32_C(0xc0000000) : UINT32_C(0x80000000)) :
       array ? UINT32_C(0xd0000000) : UINT32_C(0x90000000)) | swizzle |
         (first_level << 12) | (last_level << 16),
      multisampled ? (array ? base->array_size - 1 : 0) : tiled ? 0 : volume ? base->depth0 - 1 :
         array ? base->array_size - 1 :
         !one_d && !base->last_level && pitch > base->width0 ? pitch - 1u : 0,
      UINT32_C(0x00400000) | ((multisampled ? 2u : base->last_level) << 4), 0, 0,
   };
   memcpy(descriptor, srd, sizeof(srd));
   return 0;
}

int
ps5_resource_storage_image_descriptor(struct pipe_resource *base, unsigned level, uint32_t descriptor[8])
{
   return ps5_resource_image_descriptor(base, descriptor, PIPE_BIND_SHADER_IMAGE, level, level);
}

static int
ps5_storage_image_view_descriptor(const struct pipe_image_view *view,
                                  uint32_t descriptor[8])
{
   struct pipe_resource *base = view ? view->resource : NULL;
   uint32_t format;
   if (!base || !descriptor || view->u.tex.level >= PIPE_MAX_TEXTURE_LEVELS ||
       view->u.tex.level > base->last_level ||
       !ps5_storage_image_texel_size(view->format) ||
       ps5_storage_image_texel_size(view->format) != ps5_storage_image_texel_size(base->format) ||
       !ps5_texture_descriptor_format(view->format, &format))
      return -1;
   const unsigned first = view->u.tex.first_layer, last = view->u.tex.last_layer;
   const bool volume = base->target == PIPE_TEXTURE_3D;
   const unsigned layers = volume ? MAX2(base->depth0 >> view->u.tex.level, 1u) : base->array_size;
   if (first > last || last >= layers ||
       (view->u.tex.single_layer_view && first != last) ||
       (volume && (view->u.tex.single_layer_view
          ? !view->u.tex.is_2d_view_of_3d
          : first || last != layers - 1)) ||
       ps5_resource_storage_image_descriptor(base, view->u.tex.level, descriptor))
      return -1;
   /* Validate the allocation using its storage format, then reinterpret only
    * equal-size texels. Addressing, pitch, tiling and mip/layer bounds stay intact. */
   const unsigned channels = ps5_storage_image_channels(view->format);
   descriptor[1] = (descriptor[1] & ~UINT32_C(0x1ff00000)) | format;
   descriptor[3] = (descriptor[3] & ~UINT32_C(0xfff)) |
      (channels == 1 ? 0x204u : channels == 2 ? 0x22cu : channels == 3 ? 0x3acu : 0xfacu);
   if (volume && !view->u.tex.single_layer_view)
      return 0;
   const struct ps5_resource *resource = (const struct ps5_resource *)base;
   const bool array = volume || base->target == PIPE_TEXTURE_1D_ARRAY ||
                      base->target == PIPE_TEXTURE_2D_ARRAY || ps5_cube_texture_target(base->target);
   if (!array)
      return first || last ? -1 : 0;
   const size_t offset = (size_t)first * resource->layer_stride;
   const size_t backing_size = base->nr_samples == 4
      ? resource->allocation_size : resource->size;
   if (offset >= backing_size)
      return -1;
   const uintptr_t address = (uintptr_t)resource->data + offset;
   if ((address & 255u) || address >> 48)
      return -1;
   descriptor[0] = address >> 8;
   descriptor[1] = (descriptor[1] & UINT32_C(0xffffff00)) | (uint32_t)(address >> 40);
   descriptor[4] = last - first;
   if (view->u.tex.single_layer_view) {
      const bool one_d = base->target == PIPE_TEXTURE_1D_ARRAY;
      descriptor[3] = (descriptor[3] & UINT32_C(0x0fffffff)) |
                     (base->nr_samples == 4 ? UINT32_C(0xe0000000) :
                      one_d ? UINT32_C(0x80000000) : UINT32_C(0x90000000));
      const unsigned pitch = resource->level_stride[0] / ps5_storage_image_texel_size(base->format);
      descriptor[4] = !one_d && !base->last_level && pitch > base->width0 ? pitch - 1u : 0;
   }
   return 0;
}

int
ps5_resource_storage_image_descriptor_owned(struct pipe_resource *base,
                                            const uint32_t descriptor[8])
{
   const struct ps5_resource *resource = (const struct ps5_resource *)base;
   if (!base || !descriptor)
      return -1;
   struct pipe_image_view view = {.resource = base, .format = base->format};
   /* Recover only a supported, equal-size image view; the full descriptor is
    * still reconstructed and compared below, including every address/bound. */
   for (unsigned f = 0; f < PIPE_FORMAT_COUNT; ++f) {
      uint32_t format;
      if (ps5_storage_image_texel_size(f) &&
          ps5_storage_image_texel_size(f) == ps5_storage_image_texel_size(base->format) &&
          ps5_texture_descriptor_format(f, &format) &&
          format == (descriptor[1] & UINT32_C(0x1ff00000))) {
         view.format = f;
         break;
      }
   }
   view.u.tex.level = (descriptor[3] >> 12) & 15u;
   view.u.tex.last_layer = base->target == PIPE_TEXTURE_3D
      ? MAX2(base->depth0 >> view.u.tex.level, 1u) - 1u : 0;
   if (base->target == PIPE_TEXTURE_1D_ARRAY || base->target == PIPE_TEXTURE_2D_ARRAY ||
       ps5_cube_texture_target(base->target)) {
      const uintptr_t address = ((uint64_t)descriptor[0] << 8) |
                                ((uint64_t)(descriptor[1] & 255u) << 40);
      const uintptr_t start = (uintptr_t)resource->data;
      const size_t backing_size = base->nr_samples == 4
         ? resource->allocation_size : resource->size;
      if (address < start || !resource->layer_stride ||
          address - start >= backing_size || (address - start) % resource->layer_stride)
         return -1;
      const size_t first = (address - start) / resource->layer_stride;
      if (first >= base->array_size)
         return -1;
      view.u.tex.first_layer = first;
      const unsigned type = descriptor[3] >> 28;
      view.u.tex.single_layer_view = type == 8 || type == 9 || type == 14;
      if (!view.u.tex.single_layer_view && descriptor[4] >= base->array_size - first)
         return -1;
      view.u.tex.last_layer = first + (view.u.tex.single_layer_view ? 0 : descriptor[4]);
   } else if (base->target == PIPE_TEXTURE_3D && (descriptor[3] >> 28) == 9) {
      /* A 2D view of one 3D slice (glBindImageTexture layered=GL_FALSE). */
      const uintptr_t address = ((uint64_t)descriptor[0] << 8) |
                                ((uint64_t)(descriptor[1] & 255u) << 40);
      const uintptr_t start = (uintptr_t)resource->data;
      if (address < start || !resource->layer_stride ||
          address - start >= resource->size || (address - start) % resource->layer_stride)
         return -1;
      const size_t first = (address - start) / resource->layer_stride;
      if (first > view.u.tex.last_layer)
         return -1;
      view.u.tex.first_layer = view.u.tex.last_layer = first;
      view.u.tex.single_layer_view = true;
      view.u.tex.is_2d_view_of_3d = true;
   }
   uint32_t expected[8];
   return ps5_storage_image_view_descriptor(&view, expected) ||
          memcmp(descriptor, expected, sizeof(expected)) ? -1 : 0;
}

int
ps5_resource_sampled_image_descriptor(struct pipe_resource *base, unsigned first_level,
                                      unsigned last_level, uint32_t descriptor[8])
{
   return ps5_resource_image_descriptor(base, descriptor, PIPE_BIND_SAMPLER_VIEW, first_level, last_level);
}

int
ps5_resource_sampled_image_descriptor_owned(struct pipe_resource *base,
                                            const uint32_t descriptor[8])
{
   uint32_t expected[8];

   if (!descriptor)
      return -1;
   for (unsigned channel = 0; channel < 4; ++channel) {
      const unsigned selector = (descriptor[3] >> (channel * 3)) & 7u;
      if (selector == 2 || selector == 3)
         return -1;
   }
   const unsigned first = (descriptor[3] >> 28) >= 14 ? 0 :
                          (descriptor[3] >> 12) & 15u;
   const unsigned last = (descriptor[3] >> 28) >= 14 ? 0 :
                         (descriptor[3] >> 16) & 15u;

   if (!ps5_resource_sampled_image_descriptor(base, first, last, expected)) {
      expected[3] = (expected[3] & ~0xfffu) | (descriptor[3] & 0xfffu);
      if (!memcmp(descriptor, expected, sizeof(expected)))
         return 0;
   }
   if (base && (base->format == PIPE_FORMAT_Z32_FLOAT ||
                base->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)) {
      const enum pipe_format views[] = {
         PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, PIPE_FORMAT_X32_S8X24_UINT,
      };
      const unsigned count = base->format == PIPE_FORMAT_Z32_FLOAT ? 1 :
                             ARRAY_SIZE(views);
      for (unsigned i = 0; i < count; ++i) {
         const enum pipe_format view = base->format == PIPE_FORMAT_Z32_FLOAT
            ? PIPE_FORMAT_Z32_FLOAT : views[i];
         if (!ps5_packed_depth_sampled_descriptor(base, view, first, last,
                                                  expected)) {
            expected[3] = (expected[3] & ~0xfffu) | (descriptor[3] & 0xfffu);
            if (!memcmp(descriptor, expected, sizeof(expected)))
               return 0;
         }
      }
   }
   return -1;
}

struct pipe_resource *
ps5_display_target_alias(struct pipe_resource *base, unsigned buffer_index)
{
   struct ps5_resource *owner = (struct ps5_resource *)base;
   struct ps5_resource *alias;
   size_t offset;

   if (!owner || !(owner->base.bind & PIPE_BIND_DISPLAY_TARGET) ||
       buffer_index != 1 ||
       owner->allocation_size < 2u * PS5_RENDER_TARGET_BYTES)
      return NULL;
   alias = calloc(1, sizeof(*alias));
   if (!alias)
      return NULL;
   offset = buffer_index * (size_t)PS5_RENDER_TARGET_BYTES;
   alias->base = owner->base;
   alias->base.reference.count = 1;
   alias->data = owner->data + offset;
   alias->size = owner->size;
   alias->allocation_size = PS5_RENDER_TARGET_BYTES;
   alias->stride = owner->stride;
   memcpy(alias->level_offset, owner->level_offset,
          sizeof(alias->level_offset));
   memcpy(alias->level_stride, owner->level_stride,
          sizeof(alias->level_stride));
   alias->layer_stride = owner->layer_stride;
   alias->direct_start = -1;
   alias->stencil_direct_start = -1;
   pipe_resource_reference(&alias->render_pool_owner, base);
   printf("[ps5-gallium] display-alias index=%u offset=%zu bytes=%zu\n",
          buffer_index, offset, alias->allocation_size);
   return &alias->base;
}

int
ps5_resource_stencil_info(struct pipe_resource *base, void **address,
                          size_t *allocation_size)
{
   struct ps5_resource *resource = (struct ps5_resource *)base;

   ps5_draw_batch_drain();
   if (!resource ||
       resource->base.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
       !resource->stencil_data)
      return -1;
   if (address) {
      resource->external_cpu_access = true;
      *address = resource->stencil_data;
   }
   if (allocation_size)
      *allocation_size = resource->stencil_allocation_size;
   return 0;
}

static void
ps5_resource_destroy(struct pipe_screen *screen, struct pipe_resource *base)
{
   struct ps5_screen *ps5 = (struct ps5_screen *)screen;
   struct ps5_resource *resource = (struct ps5_resource *)base;
   struct pipe_resource *render_pool_owner = resource->render_pool_owner;
   struct pipe_resource *stencil_sample = resource->stencil_sample;
#ifdef PS5_PUBLIC_STENCIL_TEST
   const bool record_prior_depth =
      resource->base.format == PIPE_FORMAT_Z32_FLOAT;
#endif

   simple_mtx_lock(&ps5->resource_mutex);
   if (render_pool_owner) {
      ps5_render_arena_mark(ps5, resource->render_arena_first_slot,
                            resource->render_arena_slot_count, false);
      resource->data = NULL;
      resource->direct_start = -1;
   }
   if (ps5->render_pool == base) {
      ps5->render_pool = NULL;
      memset(ps5->render_arena_bitmap, 0,
             sizeof(ps5->render_arena_bitmap));
   }
   simple_mtx_unlock(&ps5->resource_mutex);
   ps5_release_resource_memory(resource->stencil_data,
      resource->stencil_allocation_size, resource->stencil_direct_start);
   ps5_release_resource_memory(resource->data, resource->allocation_size,
                               resource->direct_start);
#ifdef PS5_PUBLIC_STENCIL_TEST
   if (record_prior_depth)
      printf("[ps5-gallium] public-stencil-prior-depth-destroy format=%u address=%p bytes=%zu direct=%016llx unmap=%08x release=%08x\n",
             resource->base.format, resource->data, resource->allocation_size,
             (unsigned long long)resource->direct_start,
             0u, 0u);
#endif
   free(resource);
   if (stencil_sample)
      pipe_resource_reference(&stencil_sample, NULL);
   if (render_pool_owner)
      pipe_resource_reference(&render_pool_owner, NULL);
}

static bool
ps5_is_format_supported(struct pipe_screen *screen, enum pipe_format format,
                        enum pipe_texture_target target, unsigned sample_count,
                        unsigned storage_sample_count, unsigned bindings)
{
   const unsigned supported_bindings =
      PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET |
      PIPE_BIND_SAMPLER_VIEW;

   (void)screen;

   if (format == PIPE_FORMAT_NONE)
      return PS5_ENABLE_GLSL_430_CANDIDATE &&
             target == PIPE_TEXTURE_2D &&
             bindings == PIPE_BIND_RENDER_TARGET &&
             ((sample_count <= 1 && storage_sample_count <= 1) ||
              (PS5_ENABLE_MSAA4_CANDIDATE && sample_count == 4 &&
               storage_sample_count == 4));

   if (sample_count > 1 || storage_sample_count > 1)
      return ps5_msaa4_color_support(format, target, sample_count,
                                     storage_sample_count, bindings) ||
               ps5_msaa4_depth_support(format, target, sample_count,
                                       storage_sample_count, bindings);

   if (PS5_ENABLE_GLSL_430_CANDIDATE &&
       (format == PIPE_FORMAT_X24S8_UINT ||
        format == PIPE_FORMAT_X32_S8X24_UINT ||
        format == PIPE_FORMAT_S8_UINT))
      return ps5_sampled_texture_target(target) && sample_count <= 1 &&
             storage_sample_count <= 1 &&
             bindings == PIPE_BIND_SAMPLER_VIEW;

   /* R64_FLOAT vertex formats require numeric conversion to float inputs.
    * Let u_vbuf convert them; real double inputs use raw UINT formats.
    */
   if (target == PIPE_BUFFER)
      return sample_count <= 1 && storage_sample_count <= 1 &&
             (((format == PIPE_FORMAT_R32_FLOAT ||
                format == PIPE_FORMAT_R32G32_FLOAT ||
                format == PIPE_FORMAT_R32G32B32_FLOAT ||
                format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
                (PS5_ENABLE_INTEGER_VERTEX_CANDIDATE &&
                 ps5_integer_vertex_format(format)) ||
                (PS5_ENABLE_PACKED_VERTEX_CANDIDATE &&
                 ps5_packed_vertex_format(format))) &&
               !(bindings & ~PIPE_BIND_VERTEX_BUFFER)) ||
              (PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE &&
               ps5_texel_buffer_format(format) &&
               bindings == PIPE_BIND_SAMPLER_VIEW));

   if (format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
       !PS5_ENABLE_PACKED_DEPTH_STENCIL)
      return false;

   /* BGRA8 renders (the display order) but is not a sampled format: Mesa
    * stores sampled BGRA textures as RGBA8 and swizzles on upload. */
   if (PS5_ENABLE_CORE_RENDER_FORMATS_CANDIDATE &&
       !(bindings & PIPE_BIND_DISPLAY_TARGET) &&
       !(format == PIPE_FORMAT_B8G8R8A8_UNORM && (bindings & PIPE_BIND_SAMPLER_VIEW)) &&
       ps5_core_render_target_format(format)) {
      const unsigned allowed = PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      const bool render = bindings & PIPE_BIND_RENDER_TARGET;

      return (render ? ps5_color_render_target(target) :
                       ps5_sampled_texture_target(target)) &&
             sample_count <= 1 && storage_sample_count <= 1 && bindings &&
             !(bindings & ~allowed) &&
             (!(render && (bindings & PIPE_BIND_SAMPLER_VIEW)) ||
              PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE);
   }

   if (PS5_ENABLE_CORE_TEXTURE_FORMATS_CANDIDATE &&
       ps5_core_sampled_texture_format(format))
      return ps5_sampled_texture_target(target) && sample_count <= 1 &&
             storage_sample_count <= 1 &&
             (bindings & PIPE_BIND_SAMPLER_VIEW) &&
             !(bindings & ~PIPE_BIND_SAMPLER_VIEW);

   if (format == PIPE_FORMAT_Z32_FLOAT ||
       format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
      const unsigned allowed = PIPE_BIND_DEPTH_STENCIL |
         (PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE ? PIPE_BIND_SAMPLER_VIEW : 0);
      const bool depth_target = bindings & PIPE_BIND_DEPTH_STENCIL;
      const bool sampled = bindings & PIPE_BIND_SAMPLER_VIEW;

      if (format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
          !PS5_ENABLE_PACKED_DEPTH_STENCIL)
         return false;
      return sample_count <= 1 && storage_sample_count <= 1 &&
             bindings && (depth_target || sampled) &&
             (!depth_target || ps5_depth_render_target(target)) &&
             (!sampled || PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE) &&
             (!sampled || depth_target ||
              (target != PIPE_BUFFER &&
               ps5_sampled_texture_target(target))) &&
             !(bindings & ~allowed);
   }

   if (format == PIPE_FORMAT_R8G8B8A8_SRGB) {
      const unsigned allowed = PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      const bool render = bindings & PIPE_BIND_RENDER_TARGET;

      return PS5_ENABLE_SRGB_CANDIDATE &&
             (render ? ps5_color_render_target(target) :
                       ps5_sampled_texture_target(target)) &&
             sample_count <= 1 && storage_sample_count <= 1 && bindings &&
             !(bindings & ~allowed) &&
             (!render || PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE) &&
             (!(render && (bindings & PIPE_BIND_SAMPLER_VIEW)) ||
              PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE);
   }

   if (format == PIPE_FORMAT_R8_UNORM ||
       format == PIPE_FORMAT_R8G8_UNORM) {
      const unsigned allowed = PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      const bool render = bindings & PIPE_BIND_RENDER_TARGET;

      return PS5_ENABLE_TEXTURE_RG_CANDIDATE &&
             (render ? (target == PIPE_TEXTURE_2D &&
                        PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) :
                       ps5_sampled_texture_target(target)) &&
             sample_count <= 1 && storage_sample_count <= 1 && bindings &&
             !(bindings & ~allowed) &&
             (!(render && (bindings & PIPE_BIND_SAMPLER_VIEW)) ||
              PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE);
   }

   if (format == PIPE_FORMAT_R8_SNORM ||
       format == PIPE_FORMAT_R8G8_SNORM ||
       format == PIPE_FORMAT_R8G8B8A8_SNORM)
      return PS5_ENABLE_TEXTURE_SNORM_CANDIDATE &&
             ps5_sampled_texture_target(target) && sample_count <= 1 &&
             storage_sample_count <= 1 &&
             (bindings & PIPE_BIND_SAMPLER_VIEW) &&
             !(bindings & ~PIPE_BIND_SAMPLER_VIEW);

   if (format == PIPE_FORMAT_R16G16B16A16_FLOAT ||
       format == PIPE_FORMAT_R32G32B32A32_FLOAT)
      return PS5_ENABLE_TEXTURE_FLOAT_CANDIDATE &&
             ps5_sampled_texture_target(target) && sample_count <= 1 &&
             storage_sample_count <= 1 &&
             (bindings & PIPE_BIND_SAMPLER_VIEW) &&
             !(bindings & ~PIPE_BIND_SAMPLER_VIEW);

   if (format == PIPE_FORMAT_R9G9B9E5_FLOAT)
      return PS5_ENABLE_SHARED_EXPONENT_CANDIDATE &&
             ps5_sampled_texture_target(target) && sample_count <= 1 &&
             storage_sample_count <= 1 &&
             (bindings & PIPE_BIND_SAMPLER_VIEW) &&
             !(bindings & ~PIPE_BIND_SAMPLER_VIEW);

   if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
      const unsigned allowed = PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      const bool render = bindings & PIPE_BIND_RENDER_TARGET;

      return PS5_ENABLE_PACKED_FLOAT_CANDIDATE &&
             (render ? (target == PIPE_TEXTURE_2D &&
                        PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) :
                       ps5_sampled_texture_target(target)) &&
             sample_count <= 1 && storage_sample_count <= 1 && bindings &&
             !(bindings & ~allowed) &&
             (!(render && (bindings & PIPE_BIND_SAMPLER_VIEW)) ||
              PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE);
   }

   if (ps5_integer_texture_format(format)) {
      const unsigned allowed = PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      const bool render = bindings & PIPE_BIND_RENDER_TARGET;

      return PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE &&
             (render ? (target == PIPE_TEXTURE_2D &&
                        PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) :
                       ps5_sampled_texture_target(target)) &&
             sample_count <= 1 && storage_sample_count <= 1 && bindings &&
             !(bindings & ~allowed) &&
             (!(render && (bindings & PIPE_BIND_SAMPLER_VIEW)) ||
              PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE);
   }

   if (format == PIPE_FORMAT_R10G10B10A2_UINT) {
      const unsigned allowed = PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      const bool render = bindings & PIPE_BIND_RENDER_TARGET;

      return PS5_ENABLE_RGB10_A2UI_CANDIDATE &&
             (render ? (target == PIPE_TEXTURE_2D &&
                        PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) :
                       ps5_sampled_texture_target(target)) &&
             sample_count <= 1 && storage_sample_count <= 1 && bindings &&
             !(bindings & ~allowed) &&
             (!(render && (bindings & PIPE_BIND_SAMPLER_VIEW)) ||
              PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE);
   }

   if ((target == PIPE_TEXTURE_2D_ARRAY ||
        ps5_cube_texture_target(target) || target == PIPE_TEXTURE_3D) &&
       PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE)
      return format == PIPE_FORMAT_R8G8B8A8_UNORM && sample_count <= 1 &&
             storage_sample_count <= 1 && bindings &&
             (bindings & PIPE_BIND_SAMPLER_VIEW) &&
             !(bindings & ~(PIPE_BIND_RENDER_TARGET |
                            PIPE_BIND_SAMPLER_VIEW));

   if (target != PIPE_TEXTURE_2D)
      return ps5_sampled_texture_target(target) &&
             format == PIPE_FORMAT_R8G8B8A8_UNORM && sample_count <= 1 &&
             storage_sample_count <= 1 &&
             (bindings & PIPE_BIND_SAMPLER_VIEW) &&
             !(bindings & ~PIPE_BIND_SAMPLER_VIEW);

   return (format == PIPE_FORMAT_R8G8B8A8_UNORM ||
           (format == PIPE_FORMAT_B8G8R8A8_UNORM &&
            !(bindings & PIPE_BIND_SAMPLER_VIEW))) &&
          target == PIPE_TEXTURE_2D && sample_count <= 1 &&
          storage_sample_count <= 1 &&
          (PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE ||
           !((bindings & PIPE_BIND_RENDER_TARGET) &&
             (bindings & PIPE_BIND_SAMPLER_VIEW))) &&
          !(bindings & ~supported_bindings);
}

static bool
ps5_map_bounds(const struct ps5_resource *resource, unsigned level,
               const struct pipe_box *box, size_t *offset)
{
   uint64_t end;
   unsigned block_size;
   unsigned block_width;
   unsigned block_height;
   unsigned layers;

   if (!box || box->x < 0 || box->y < 0 || box->z < 0 || box->width < 0 ||
       box->height < 0 || box->depth < 1)
      return false;
   if (resource->base.nr_samples > 1)
      return false;

   block_size = ps5_texture_format_size(resource->base.format);
   block_width = util_format_get_blockwidth(resource->base.format);
   block_height = util_format_get_blockheight(resource->base.format);

   if (resource->base.target == PIPE_BUFFER) {
      if (level)
         return false;
      end = (uint64_t)(unsigned)box->x + (unsigned)box->width;
      if (box->y || box->z || box->height != 1 || box->depth != 1 ||
          end > resource->size)
         return false;
      *offset = (unsigned)box->x;
      return true;
   }

   if (level > resource->base.last_level)
      return false;
   layers = resource->base.target == PIPE_TEXTURE_3D
               ? MAX2(resource->base.depth0 >> level, 1u)
               : resource->base.array_size;
   if ((uint64_t)(unsigned)box->x + (unsigned)box->width >
          (resource->base.width0 >> level ? resource->base.width0 >> level : 1) ||
       (uint64_t)(unsigned)box->y + (unsigned)box->height >
          (resource->base.height0 >> level ? resource->base.height0 >> level : 1) ||
       (uint64_t)(unsigned)box->z + (unsigned)box->depth >
          layers)
      return false;

   if ((unsigned)box->x % block_width ||
       (unsigned)box->y % block_height)
      return false;

   *offset = (size_t)(unsigned)box->z * resource->layer_stride +
              resource->level_offset[level] +
              (size_t)((unsigned)box->y / block_height) *
                 resource->level_stride[level] +
              (size_t)((unsigned)box->x / block_width) * block_size;
   return true;
}

static size_t
ps5_tiled_affine_offset(unsigned x, unsigned y,
                        const uint16_t *x_masks, unsigned x_bits,
                        const uint16_t *y_masks, unsigned y_bits)
{
   size_t offset = 0;

   for (unsigned bit = 0; bit < x_bits; ++bit) {
      if (x & BITFIELD_BIT(bit))
         offset ^= x_masks[bit];
   }
   for (unsigned bit = 0; bit < y_bits; ++bit) {
      if (y & BITFIELD_BIT(bit))
         offset ^= y_masks[bit];
   }
   return offset;
}

static size_t
ps5_tiled_depth_offset(unsigned x, unsigned y, unsigned width, unsigned layer)
{
   static const uint16_t x_masks[7] = {
      0x0004, 0x0010, 0x0040, 0x0100, 0x2200, 0x0800, 0x8400,
   };
   static const uint16_t y_masks[7] = {
      0x0008, 0x0020, 0x0080, 0x1100, 0x0200, 0x0400, 0x4800,
   };
   size_t local = ps5_tiled_affine_offset(
      x & 127u, y & 127u, x_masks, ARRAY_SIZE(x_masks),
      y_masks, ARRAY_SIZE(y_masks));

   return ((((size_t)y >> 7) * ((width + 127u) >> 7) + (x >> 7)) << 16) +
          (local ^ ps5_tiled_depth_layer_xor(layer));
}

static size_t
ps5_tiled_depth_msaa4_offset(unsigned x, unsigned y, unsigned sample,
                             unsigned width, unsigned layer)
{
   static const uint16_t x_masks[7] = {
      0x0010, 0x0040, 0x8000, 0x0100, 0x2200, 0x0800, 0x0400,
   };
   static const uint16_t y_masks[8] = {
      0x0020, 0x0080, 0x4000, 0x1100,
      0x0200, 0x0400, 0x4800, 0x8000,
   };
   static const uint16_t sample_masks[2] = {0x0004, 0x0008};
   size_t local;

   if (sample >= 4)
      return SIZE_MAX;
   local = ps5_tiled_affine_offset(x, y, x_masks, ARRAY_SIZE(x_masks),
                                   y_masks, ARRAY_SIZE(y_masks));
   for (unsigned bit = 0; bit < ARRAY_SIZE(sample_masks); ++bit) {
      if (sample & BITFIELD_BIT(bit))
         local ^= sample_masks[bit];
   }
   return ((((size_t)y >> 6) * ((width + 63u) >> 6) + (x >> 6)) << 16) +
          (local ^ ps5_tiled_depth_layer_xor(layer));
}

static size_t
ps5_tiled_stencil_offset(unsigned x, unsigned y, unsigned width, unsigned layer)
{
   static const uint16_t x_masks[8] = {
      0x0001, 0x0004, 0x0010, 0x0140,
      0x0200, 0x0800, 0x2400, 0x8000,
   };
   static const uint16_t y_masks[8] = {
      0x0002, 0x0008, 0x0020, 0x0100,
      0x0280, 0x0400, 0x1800, 0x4000,
   };
   size_t local = ps5_tiled_affine_offset(
      x & 255u, y & 255u, x_masks, ARRAY_SIZE(x_masks),
      y_masks, ARRAY_SIZE(y_masks));

   return ((((size_t)y >> 8) * ((width + 255u) >> 8) + (x >> 8)) << 16) +
          (local ^ ps5_tiled_depth_layer_xor(layer));
}

static size_t
ps5_tiled_stencil_msaa4_offset(unsigned x, unsigned y, unsigned sample,
                               unsigned width, unsigned layer)
{
   static const uint16_t x_masks[7] = {
      0x0004, 0x0010, 0x0040, 0x0100, 0x2200, 0x0800, 0x8400,
   };
   static const uint16_t y_masks[7] = {
      0x0008, 0x0020, 0x0080, 0x1100, 0x0200, 0x0400, 0x4800,
   };
   static const uint16_t sample_masks[2] = {0x0001, 0x0002};
   size_t local;

   if (sample >= 4)
      return SIZE_MAX;
   local = ps5_tiled_affine_offset(x, y, x_masks, ARRAY_SIZE(x_masks),
                                   y_masks, ARRAY_SIZE(y_masks));
   for (unsigned bit = 0; bit < ARRAY_SIZE(sample_masks); ++bit) {
      if (sample & BITFIELD_BIT(bit))
         local ^= sample_masks[bit];
   }
   return ((((size_t)y >> 7) * ((width + 127u) >> 7) + (x >> 7)) << 16) +
          (local ^ ps5_tiled_depth_layer_xor(layer));
}

static size_t
ps5_tiled_color_offset(enum pipe_format format, unsigned x, unsigned y,
                       unsigned width, unsigned layer)
{
   /* Exhaustive GPU coordinate ramps in the 2026-08-30 tile-layout receipt
    * validated every address in one 64 KiB swizzle-27 tile. */
   static const uint16_t r8_x_masks[8] = {
      0x0001, 0x0002, 0x0004, 0x0140,
      0x0200, 0x0800, 0x2400, 0x8000,
   };
   static const uint16_t r8_y_masks[8] = {
      0x0010, 0x0008, 0x0020, 0x0100,
      0x0280, 0x0400, 0x1800, 0x4000,
   };
   static const uint16_t rg8_x_masks[8] = {
      0x0001, 0x0002, 0x0004, 0x00c0,
      0x0100, 0x0400, 0x1200, 0x4000,
   };
   static const uint16_t rg8_y_masks[7] = {
      0x0008, 0x0010, 0x0020, 0x0080,
      0x0900, 0x0200, 0x2400,
   };
   static const uint16_t rgba8_x_masks[7] = {
      0x0004, 0x0008, 0x0080, 0x0100, 0x2200, 0x0800, 0x8400,
   };
   static const uint16_t rgba8_y_masks[7] = {
      0x0010, 0x0020, 0x0040, 0x1100, 0x0200, 0x0400, 0x4800,
   };
   /* AMD GFX10 AddressLib's non-RBPlus 16-pipe SW_64K_R_X equation
    * uniquely reproduces the hardware-proven R8, RG8 and RGBA8 ramps.
    * The corresponding 128-bpp equation locally addresses each 64x64 tile;
    * its seventh X/Y terms apply the inter-tile XOR. */
   static const uint16_t rgba32_x_masks[7] = {
      0x0010, 0x0040, 0x2000, 0x0100, 0x8200, 0x0800, 0x0400,
   };
   static const uint16_t rgba32_y_masks[7] = {
      0x0020, 0x0080, 0x1000, 0x4100, 0x0200, 0x0400, 0x0800,
   };
   static const uint16_t rgba16_x_masks[7] = {
      0x0008, 0x0020, 0x0040, 0x2100, 0x0200, 0x0800, 0x8400,
   };
   static const uint16_t rgba16_y_masks[7] = {
      0x0010, 0x0080, 0x1000, 0x0100, 0x4200, 0x0400, 0x0800,
   };
   unsigned tile_width;
   unsigned tile_height;
   size_t local;

   switch (util_format_get_blocksize(format)) {
   case 1:
      tile_width = 256;
      tile_height = 256;
      local = ps5_tiled_affine_offset(x & 255u, y & 255u,
                                      r8_x_masks, 8, r8_y_masks, 8);
      break;
   case 2:
      tile_width = 256;
      tile_height = 128;
      local = ps5_tiled_affine_offset(x & 255u, y & 127u,
                                      rg8_x_masks, 8, rg8_y_masks, 7) << 1;
      break;
   case 4:
      tile_width = 128;
      tile_height = 128;
      local = ps5_tiled_affine_offset(x & 127u, y & 127u,
                                      rgba8_x_masks, 7, rgba8_y_masks, 7);
      break;
   case 8:
      tile_width = 128;
      tile_height = 64;
      local = ps5_tiled_affine_offset(x, y, rgba16_x_masks, 7,
                                      rgba16_y_masks, 7);
      break;
   case 16:
      tile_width = 64;
      tile_height = 64;
      local = ps5_tiled_affine_offset(x, y, rgba32_x_masks, 7,
                                      rgba32_y_masks, 7);
      break;
   default:
      return SIZE_MAX;
   }
   /* GFX10 16-pipe 64KB_R_X puts Z3..Z0 in address bits 8..11
    * for every supported single-sample texel size. */
   local ^= ((layer & 1u) << 11) | ((layer & 2u) << 9) |
            ((layer & 4u) << 7) | ((layer & 8u) << 5);
   return ((((size_t)y / tile_height) *
               ((width + tile_width - 1u) / tile_width) +
            x / tile_width) << 16) + local;
}

static size_t
ps5_tiled_color_msaa4_offset(enum pipe_format format, unsigned x, unsigned y,
                             unsigned sample, unsigned width, unsigned layer)
{
   static const uint16_t bpe1_x_masks[7] = {
      0x0001, 0x0002, 0x0004, 0x0140, 0x0200, 0x0800, 0x2400,
   };
   static const uint16_t bpe1_y_masks[9] = {
      0x0010, 0x0008, 0x0020, 0x0100, 0x0280,
      0x0400, 0x1800, 0x4000, 0x8000,
   };
   static const uint16_t bpe1_sample_masks[2] = {0x4000, 0x8000};
   static const uint16_t bpe2_x_masks[7] = {
      0x0002, 0x0004, 0x0008, 0x0180, 0x0200, 0x0800, 0x2400,
   };
   static const uint16_t bpe2_y_masks[8] = {
      0x0010, 0x0020, 0x0040, 0x0100,
      0x1200, 0x0400, 0x4800, 0x8000,
   };
   static const uint16_t bpe2_sample_masks[2] = {0x4000, 0x8000};
   static const uint16_t bpe4_x_masks[7] = {
      0x0004, 0x0008, 0x0080, 0x0100, 0x2200, 0x0800, 0x0400,
   };
   static const uint16_t bpe4_y_masks[8] = {
      0x0010, 0x0020, 0x0040, 0x1100,
      0x0200, 0x0400, 0x4800, 0x8000,
   };
   static const uint16_t bpe4_sample_masks[2] = {0x4000, 0x8000};
   static const uint16_t bpe8_x_masks[7] = {
      0x0008, 0x0020, 0x0040, 0x2100, 0x0200, 0x0800, 0x0400,
   };
   static const uint16_t bpe8_y_masks[7] = {
      0x0010, 0x0080, 0x1000, 0x0100, 0x4200, 0x0400, 0x8800,
   };
   static const uint16_t bpe8_sample_masks[2] = {0x8000, 0x0400};
   static const uint16_t bpe16_x_masks[7] = {
      0x0010, 0x0040, 0x2000, 0x0100, 0x8200, 0x0800, 0x0400,
   };
   static const uint16_t bpe16_y_masks[7] = {
      0x0020, 0x0080, 0x1000, 0x4100, 0x0200, 0x0400, 0x0800,
   };
   static const uint16_t bpe16_sample_masks[2] = {0x0800, 0x0400};
   /* GFX10 16-pipe R_X 4x array-slice terms from AddressLib nibble2
    * entries 74, 117 and 118. Sample bits displace some slice bits. */
   static const uint16_t layer_masks[3][4] = {
      {0x0800, 0x0400, 0x0200, 0x0100},
      {0x0800, 0x0200, 0x0100, 0},
      {0x0200, 0x0100, 0, 0},
   };
   const uint16_t *x_masks;
   const uint16_t *y_masks;
   const uint16_t *sample_masks;
   unsigned x_bits;
   unsigned y_bits;
   unsigned tile_width;
   unsigned tile_height;
   size_t local;

   if (sample >= 4 ||
       !ps5_tiled_color_msaa4_tile(format, &tile_width, &tile_height))
      return SIZE_MAX;
   switch (util_format_get_blocksize(format)) {
   case 1:
      x_masks = bpe1_x_masks;
      y_masks = bpe1_y_masks;
      sample_masks = bpe1_sample_masks;
      x_bits = ARRAY_SIZE(bpe1_x_masks);
      y_bits = ARRAY_SIZE(bpe1_y_masks);
      break;
   case 2:
      x_masks = bpe2_x_masks;
      y_masks = bpe2_y_masks;
      sample_masks = bpe2_sample_masks;
      x_bits = ARRAY_SIZE(bpe2_x_masks);
      y_bits = ARRAY_SIZE(bpe2_y_masks);
      break;
   case 4:
      x_masks = bpe4_x_masks;
      y_masks = bpe4_y_masks;
      sample_masks = bpe4_sample_masks;
      x_bits = ARRAY_SIZE(bpe4_x_masks);
      y_bits = ARRAY_SIZE(bpe4_y_masks);
      break;
   case 8:
      x_masks = bpe8_x_masks;
      y_masks = bpe8_y_masks;
      sample_masks = bpe8_sample_masks;
      x_bits = ARRAY_SIZE(bpe8_x_masks);
      y_bits = ARRAY_SIZE(bpe8_y_masks);
      break;
   case 16:
      x_masks = bpe16_x_masks;
      y_masks = bpe16_y_masks;
      sample_masks = bpe16_sample_masks;
      x_bits = ARRAY_SIZE(bpe16_x_masks);
      y_bits = ARRAY_SIZE(bpe16_y_masks);
      break;
   default:
      return SIZE_MAX;
   }
   local = ps5_tiled_affine_offset(x, y, x_masks, x_bits,
                                   y_masks, y_bits);
   for (unsigned bit = 0; bit < 2; ++bit) {
      if (sample & BITFIELD_BIT(bit))
         local ^= sample_masks[bit];
   }
   const unsigned bpe = util_format_get_blocksize(format);
   for (unsigned bit = 0; bit < 4; ++bit)
      if (layer & BITFIELD_BIT(bit))
         local ^= layer_masks[bpe == 16 ? 2 : bpe == 8 ? 1 : 0][bit];
   return ((((size_t)y / tile_height) *
               ((width + tile_width - 1u) / tile_width) +
            x / tile_width) << 16) + local;
}

static unsigned
ps5_tiled_rgba8_width(const struct ps5_resource *resource)
{
   return PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE
             ? resource->base.width0 : PS5_RENDER_WIDTH;
}

static unsigned
ps5_surface_width(const struct pipe_surface *surface)
{
   return MAX2(surface->texture->width0 >> surface->level, 1u);
}

static unsigned
ps5_surface_height(const struct pipe_surface *surface)
{
   return MAX2(surface->texture->height0 >> surface->level, 1u);
}

static unsigned
ps5_surface_layer_count(const struct pipe_surface *surface)
{
   return ps5_texture_level_layers(surface->texture, surface->level);
}

static unsigned
ps5_color_surface_first_layer(const struct pipe_surface *surface)
{
   /* Native array tiling includes the absolute layer in its address XOR.
    * Staged surfaces are deliberately copied to a layer-zero allocation. */
   return surface->texture &&
          surface->texture->target == PIPE_TEXTURE_2D_ARRAY &&
          !((const struct ps5_resource *)surface->texture)->render_staging_size
             ? surface->first_layer : 0;
}

static unsigned
ps5_linear_color_pitch(const struct pipe_surface *surface)
{
   /* Render one existing linear mip/layer directly. Keep allocated staging for
    * layered draws and unsupported formats; sampler/CPU storage is unchanged. */
   if (!PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE ||
       !ps5_agc_gate2_set_color_target_layouts || !surface || !surface->texture)
      return 0;
   const struct ps5_resource *r = (const struct ps5_resource *)surface->texture;
   if ((r->base.target != PIPE_TEXTURE_2D && r->base.target != PIPE_TEXTURE_2D_ARRAY) ||
       !r->base.width0 || !r->base.height0 || r->base.depth0 != 1 ||
       r->base.last_level >= ARRAY_SIZE(r->level_stride) || r->base.last_level >= 16 ||
       r->base.width0 > PS5_MAX_COLOR_WIDTH || r->base.height0 > PS5_MAX_COLOR_HEIGHT ||
       (r->base.target == PIPE_TEXTURE_2D && r->base.array_size != 1) ||
       (r->base.bind & PIPE_BIND_DISPLAY_TARGET) || !(r->base.bind & PIPE_BIND_RENDER_TARGET) ||
       r->base.nr_samples > 1 || r->base.nr_storage_samples > 1 ||
       /* A legal texture view reinterprets texels, not their linear addresses. */
       util_format_get_blockwidth(r->base.format) != 1 ||
       util_format_get_blockheight(r->base.format) != 1 ||
       util_format_get_blocksize(surface->format) != util_format_get_blocksize(r->base.format) ||
       !r->render_staging_size || r->depth_staging_size ||
       !ps5_linear_sampled_layout(&r->base) || !r->data || r->size > r->allocation_size ||
       surface->level > r->base.last_level || surface->level >= ARRAY_SIZE(r->level_stride) ||
       surface->level >= 32 || surface->first_layer != surface->last_layer ||
       surface->last_layer >= ps5_surface_layer_count(surface) || !r->layer_stride ||
       surface->last_layer >= r->size / r->layer_stride ||
       (surface->format != PIPE_FORMAT_R8G8B8A8_UNORM &&
        surface->format != PIPE_FORMAT_R8G8B8A8_SNORM && surface->format != PIPE_FORMAT_R8_UNORM &&
        surface->format != PIPE_FORMAT_R8G8_UNORM && surface->format != PIPE_FORMAT_R16G16B16A16_FLOAT &&
        surface->format != PIPE_FORMAT_R11G11B10_FLOAT && surface->format != PIPE_FORMAT_R8G8B8A8_SRGB &&
        surface->format != PIPE_FORMAT_R10G10B10A2_UNORM && surface->format != PIPE_FORMAT_R16_FLOAT &&
        surface->format != PIPE_FORMAT_R32_FLOAT && surface->format != PIPE_FORMAT_R16G16_FLOAT &&
        surface->format != PIPE_FORMAT_R32G32B32A32_UINT))
      return 0;
   unsigned width = ps5_surface_width(surface), height = ps5_surface_height(surface);
   unsigned stride = r->level_stride[surface->level];
   unsigned bpp = ps5_texture_format_size(surface->format);
   if (!width || !height || width > PS5_MAX_COLOR_WIDTH || height > PS5_MAX_COLOR_HEIGHT ||
       !stride || (stride & 255u) || !bpp || stride < (uint64_t)width * bpp ||
       stride / bpp > PS5_MAX_COLOR_WIDTH || stride > SIZE_MAX / height)
      return 0;
   size_t span = (size_t)(height - 1u) * stride + (size_t)width * bpp;
   size_t level = r->level_offset[surface->level];
   if (level > r->layer_stride || span > r->layer_stride - level)
      return 0;
   size_t offset = (size_t)surface->first_layer * r->layer_stride + level;
   uintptr_t address = (uintptr_t)r->data;
   if (offset > r->size || offset > UINTPTR_MAX - address || span > r->size - offset)
      return 0;
   address += offset;
   if ((address & 255u) || address >= (UINT64_C(1) << 48) ||
       span > (UINT64_C(1) << 48) - address)
      return 0;
   return stride;
}

static bool
ps5_stage_color_surface(const struct pipe_surface *surface, bool to_staging)
{
   struct ps5_resource *resource;
   unsigned format_size;
   unsigned width;
   unsigned height;
   size_t tiled_layer_size;
   size_t x_offsets[256];
   uint8_t *staging;

   if (!surface || !surface->texture)
      return false;
   resource = (struct ps5_resource *)surface->texture;
   if (!resource->render_staging_size ||
       surface->level > resource->base.last_level ||
       surface->first_layer > surface->last_layer ||
       surface->last_layer >= ps5_surface_layer_count(surface) ||
       !(format_size = ps5_texture_format_size(surface->format)))
      return false;
   width = ps5_surface_width(surface);
   height = ps5_surface_height(surface);
   tiled_layer_size = ps5_tiled_color_surface_size(
      surface->format, width, height);
   staging = resource->data + resource->render_staging_offset;
   if (!tiled_layer_size ||
       surface->last_layer - surface->first_layer + 1u >
          resource->render_staging_size / tiled_layer_size ||
       resource->render_staging_offset >= resource->allocation_size ||
       resource->render_staging_size >
          resource->allocation_size - resource->render_staging_offset)
      return false;

   /* All supported color equations repeat their X XOR within 256 pixels.
    * Reuse those terms; add tile bases, XOR only the local 64 KiB address.
    * Ownership, copies, bounds and cache maintenance remain unchanged. */
   for (unsigned x = 0; x < MIN2(width, ARRAY_SIZE(x_offsets)); ++x) {
      x_offsets[x] = ps5_tiled_color_offset(surface->format, x, 0, width, 0);
      if (x_offsets[x] == SIZE_MAX)
         return false;
   }
   if (!to_staging)
      ps5_flush_gpu_data(staging, resource->render_staging_size);
   for (unsigned layer = surface->first_layer;
        layer <= surface->last_layer; ++layer) {
      size_t linear_base = (size_t)layer * resource->layer_stride +
                           resource->level_offset[surface->level];
      size_t tiled_base =
         (size_t)(layer - surface->first_layer) * tiled_layer_size;

      for (unsigned y = 0; y < height; ++y) {
         for (unsigned first_x = 0; first_x < width;
              first_x += ARRAY_SIZE(x_offsets)) {
            size_t row = tiled_base +
               ps5_tiled_color_offset(surface->format, first_x, y, width,
                                      layer - surface->first_layer);
            unsigned count = MIN2(width - first_x, ARRAY_SIZE(x_offsets));
            for (unsigned x = 0; x < count;) {
               /* Four 32-bit pixels are contiguous: X bits 0/1 map to
                * address bits 2/3, and every Y term leaves those bits clear.
                * Each group starts on a four-pixel boundary; keep the tail
                * scalar. memcpy also supports unaligned linear storage. */
               unsigned pixels = format_size == 4 && count - x >= 4 ? 4 : 1;
               unsigned bytes = pixels * format_size;
               size_t linear = linear_base +
                  (size_t)y * resource->level_stride[surface->level] +
                  (size_t)(first_x + x) * format_size;
               size_t tiled = (row & ~(size_t)0xffff) +
                  (x_offsets[x] & ~(size_t)0xffff) +
                  ((row ^ x_offsets[x]) & 0xffff);

               if (linear > resource->size ||
                   resource->size - linear < bytes ||
                   tiled > resource->render_staging_size ||
                   resource->render_staging_size - tiled < bytes)
                  return false;
               uint8_t *dst = to_staging ? staging + tiled
                                         : resource->data + linear;
               const uint8_t *src = to_staging ? resource->data + linear
                                               : staging + tiled;
               /* ponytail: only group the measured 32-bit hot path. */
               if (pixels == 4)
                  memcpy(dst, src, 16);
               else if (format_size == 4)
                  memcpy(dst, src, 4);
               else
                  memcpy(dst, src, format_size);
               x += pixels;
            }
         }
      }
   }
   if (to_staging)
      ps5_flush_gpu_data(staging, resource->render_staging_size);
   else
      ps5_flush_gpu_data(resource->data, resource->size);
   return true;
}

static bool
ps5_stage_depth_surface(const struct pipe_surface *surface, bool to_staging)
{
   struct ps5_resource *resource;
   bool packed;
   unsigned width;
   unsigned height;
   unsigned format_size;
   size_t tiled_layer_size;
   size_t stencil_layer_size;
   uint8_t *staging;

   if (!surface || !surface->texture)
      return false;
   resource = (struct ps5_resource *)surface->texture;
   packed = resource->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
   if (!resource->depth_staging_size ||
       (resource->base.format != PIPE_FORMAT_Z32_FLOAT && !packed) ||
       surface->level > resource->base.last_level ||
       surface->first_layer > surface->last_layer ||
       surface->last_layer >= ps5_surface_layer_count(surface) ||
       resource->depth_staging_offset >= resource->allocation_size ||
       resource->depth_staging_size >
          resource->allocation_size - resource->depth_staging_offset)
      return false;

   width = ps5_surface_width(surface);
   height = ps5_surface_height(surface);
   format_size = ps5_texture_format_size(resource->base.format);
   tiled_layer_size = ps5_tiled_depth_surface_size(width, height, 1);
   stencil_layer_size = ps5_tiled_stencil_surface_size(width, height);
   staging = resource->data + resource->depth_staging_offset;
   if (!format_size || !tiled_layer_size ||
       tiled_layer_size > resource->depth_staging_size ||
       surface->last_layer >
          resource->depth_staging_size / tiled_layer_size - 1u ||
       (packed &&
        (!resource->stencil_data || !stencil_layer_size ||
         stencil_layer_size > resource->stencil_allocation_size ||
         surface->last_layer >
            resource->stencil_allocation_size / stencil_layer_size - 1u)))
      return false;

   if (!to_staging) {
      ps5_flush_gpu_data(staging, resource->depth_staging_size);
      if (packed)
         ps5_flush_gpu_data(resource->stencil_data,
                            resource->stencil_allocation_size);
   }
   for (unsigned layer = surface->first_layer;
        layer <= surface->last_layer; ++layer) {
      size_t linear_base = (size_t)layer * resource->layer_stride +
                           resource->level_offset[surface->level];
      size_t tiled_base = (size_t)layer * tiled_layer_size;
      size_t stencil_base = (size_t)layer * stencil_layer_size;

      for (unsigned y = 0; y < height; ++y) {
         for (unsigned x = 0; x < width; ++x) {
            size_t linear = linear_base +
                            (size_t)y * resource->level_stride[surface->level] +
                            (size_t)x * format_size;
            size_t tiled = tiled_base +
                           ps5_tiled_depth_offset(x, y, width, layer);
            size_t stencil = stencil_base +
                             ps5_tiled_stencil_offset(x, y, width, layer);

            if (linear > resource->size ||
                resource->size - linear < format_size ||
                tiled > resource->depth_staging_size ||
                resource->depth_staging_size - tiled < sizeof(float) ||
                (packed && stencil >= resource->stencil_allocation_size))
               return false;
            if (to_staging) {
               memcpy(staging + tiled, resource->data + linear,
                      sizeof(float));
               if (packed)
                  resource->stencil_data[stencil] =
                     resource->data[linear + sizeof(float)];
            } else {
               memcpy(resource->data + linear, staging + tiled,
                      sizeof(float));
               if (packed)
                  resource->data[linear + sizeof(float)] =
                     resource->stencil_data[stencil];
            }
         }
      }
   }
   if (to_staging) {
      ps5_flush_gpu_data(staging, resource->depth_staging_size);
      if (packed)
         ps5_flush_gpu_data(resource->stencil_data,
                            resource->stencil_allocation_size);
   } else {
      ps5_flush_gpu_data(resource->data, resource->size);
   }
   return true;
}

#ifdef PS5_NATIVE_TITLE_RUNTIME
/* CPU-only scratch becomes reusable at unmap; GPU resources never enter here. */
static simple_mtx_t ps5_staging_mutex = SIMPLE_MTX_INITIALIZER;
static struct { void *memory; size_t size; } ps5_staging_free[8];
static unsigned ps5_staging_count;
static size_t ps5_staging_bytes;
static uint64_t ps5_staging_hits, ps5_staging_misses;

static void *
ps5_staging_take(size_t size)
{
   void *memory = NULL;
   simple_mtx_lock(&ps5_staging_mutex);
   for (unsigned i = 0; i < ps5_staging_count; ++i) {
      if (ps5_staging_free[i].size != size)
         continue;
      memory = ps5_staging_free[i].memory;
      ps5_staging_free[i] = ps5_staging_free[--ps5_staging_count];
      ps5_staging_bytes -= size;
      break;
   }
   if (memory) ++ps5_staging_hits; else ++ps5_staging_misses;
   simple_mtx_unlock(&ps5_staging_mutex);
   return memory;
}

static bool
ps5_staging_put(void *memory, size_t size)
{
   const size_t limit = 128u * 1024u * 1024u;
   simple_mtx_lock(&ps5_staging_mutex);
   bool retained = memory && size && size <= limit &&
      ps5_staging_count < ARRAY_SIZE(ps5_staging_free) && ps5_staging_bytes <= limit - size;
   if (retained) {
      ps5_staging_free[ps5_staging_count].memory = memory;
      ps5_staging_free[ps5_staging_count++].size = size;
      ps5_staging_bytes += size;
   }
   simple_mtx_unlock(&ps5_staging_mutex);
   return retained;
}

static void
ps5_staging_clear(void)
{
   simple_mtx_lock(&ps5_staging_mutex);
   printf("[ps5-staging-reuse] hits=%" PRIu64 " misses=%" PRIu64 " retained=%zu\n",
          ps5_staging_hits, ps5_staging_misses, ps5_staging_bytes);
   while (ps5_staging_count) {
      unsigned i = ps5_staging_count - 1;
      if (munmap(ps5_staging_free[i].memory, ps5_staging_free[i].size)) {
         fprintf(stderr, "[ps5-gallium] cached staging munmap failed\n");
         break;
      }
      ps5_staging_bytes -= ps5_staging_free[i].size;
      --ps5_staging_count;
   }
   simple_mtx_unlock(&ps5_staging_mutex);
}
#endif

static bool
ps5_transfer_alloc_staging(struct ps5_transfer *transfer, size_t size)
{
   if (!size || size > SIZE_MAX - (PS5_DIRECT_ALIGNMENT - 1u))
      return false;

   /* Keep small transfers cheap; full-image scratch must not exhaust the
    * native libc heap. This is CPU-only memory, never a GPU resource. */
   if (size < 0x10000) {
      transfer->staging = malloc(size);
      return transfer->staging != NULL;
   }
   size = (size + PS5_DIRECT_ALIGNMENT - 1u) &
          ~(size_t)(PS5_DIRECT_ALIGNMENT - 1u);
#ifdef PS5_NATIVE_TITLE_RUNTIME
   transfer->staging = ps5_staging_take(size);
   if (transfer->staging) {
      transfer->staging_mapping_size = size;
      return true;
   }
#endif
   transfer->staging = mmap(NULL, size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (transfer->staging == MAP_FAILED) {
      transfer->staging = NULL;
      return false;
   }
   transfer->staging_mapping_size = size;
   return true;
}

static void
ps5_transfer_free_staging(struct ps5_transfer *transfer)
{
   if (transfer->staging_mapping_size) {
#ifdef PS5_NATIVE_TITLE_RUNTIME
      if (!ps5_staging_put(transfer->staging, transfer->staging_mapping_size))
#endif
      if (munmap(transfer->staging, transfer->staging_mapping_size))
         fprintf(stderr, "[ps5-gallium] transfer staging munmap failed\n");
   } else {
      free(transfer->staging);
   }
   transfer->staging = NULL;
   transfer->staging_mapping_size = 0;
}

static void *
ps5_transfer_map(struct pipe_context *context, struct pipe_resource *base,
                 unsigned level, unsigned usage, const struct pipe_box *box,
                 struct pipe_transfer **out_transfer)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   ps5_map_caller_record((uintptr_t)__builtin_return_address(0), usage,
                         base ? base->target : UINT_MAX);
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {6, ps5_prepare_clock()};
#endif

   struct ps5_resource *resource = (struct ps5_resource *)base;
   struct ps5_transfer *transfer;
   size_t offset;
   unsigned format_size;

   (void)context;
   if (!out_transfer)
      return NULL;
   *out_transfer = NULL;
   /* Mesa initializes incomplete multisample fallback textures through a
    * write map. Broadcast each supplied texel to all samples on unmap;
    * reads still require the existing explicit resolve path. */
   struct ps5_resource single_sample;
   const struct ps5_resource *bounds_resource = resource;
   if (base->nr_samples > 1) {
      if (base->nr_samples != 4 || base->nr_storage_samples != 4 ||
          !(usage & PIPE_MAP_WRITE) || (usage & PIPE_MAP_READ) ||
          !(base->bind & PIPE_BIND_RENDER_TARGET) ||
          !ps5_msaa4_color_format(base->format) ||
          (base->target != PIPE_TEXTURE_2D && base->target != PIPE_TEXTURE_2D_ARRAY))
         return NULL;
      single_sample = *resource;
      single_sample.base.nr_samples = single_sample.base.nr_storage_samples = 1;
      bounds_resource = &single_sample;
   }
   if (!ps5_map_bounds(bounds_resource, level, box, &offset))
      return NULL;
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   static unsigned trace_count;
   if (__atomic_load_n(&ps5_prepare_calls[0], __ATOMIC_RELAXED) >= 60000 &&
       __atomic_load_n(&trace_count, __ATOMIC_RELAXED) < 32 &&
       __atomic_fetch_add(&trace_count, 1, __ATOMIC_RELAXED) < 32) {
      printf("[ps5-map-layout] target=%u format=%u bind=%x usage=%x level=%u "
             "image=%ux%u layers=%u box=%d,%d,%d/%dx%dx%d bytes=%zu\n",
             base->target, base->format, base->bind, usage, level,
             base->width0, base->height0, base->array_size,
             box->x, box->y, box->z, box->width, box->height, box->depth,
             resource->allocation_size);
   }
#endif
   /* Mesa's streaming uploader uses disjoint ranges in an existing buffer.
    * Honor its explicit no-wait contract; reads and texture staging still wait. */
   if (base->target != PIPE_BUFFER || (usage & PIPE_MAP_READ) ||
       !(usage & PIPE_MAP_UNSYNCHRONIZED))
      ps5_draw_batch_drain_buffer(base);
   else
      ps5_invalidate_gpu_writes(resource);

   if (usage & PIPE_MAP_PERSISTENT)
      resource->external_cpu_access = true;

   transfer = calloc(1, sizeof(*transfer));
   if (!transfer)
      return NULL;

   transfer->base.resource = base;
   transfer->base.level = level;
   transfer->base.usage = usage;
   transfer->base.box = *box;
   transfer->base.stride = resource->level_stride[level];
   transfer->base.layer_stride = resource->layer_stride;
   transfer->base.offset = (unsigned)offset;
   *out_transfer = &transfer->base;

   if ((usage & (PIPE_MAP_READ | PIPE_MAP_WRITE)) &&
       (resource->base.bind & PIPE_BIND_DEPTH_STENCIL) &&
       !resource->depth_staging_size &&
       (resource->base.format == PIPE_FORMAT_Z32_FLOAT ||
        resource->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)) {
      const bool packed =
         resource->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      const size_t depth_layer_size = ps5_tiled_depth_surface_size(
         resource->base.width0, resource->base.height0, 1);
      const size_t stencil_layer_size = ps5_tiled_stencil_surface_size(
         resource->base.width0, resource->base.height0);
      size_t staging_stride;
      size_t staging_layer_stride;
      size_t staging_size;

      format_size = ps5_texture_format_size(resource->base.format);
      staging_stride = (size_t)(unsigned)box->width * format_size;
      if (!format_size || level || box->width <= 0 || box->height <= 0 ||
          staging_stride > SIZE_MAX / (unsigned)box->height) {
         free(transfer);
         *out_transfer = NULL;
         return NULL;
      }
      staging_layer_stride = staging_stride * (unsigned)box->height;
      if (staging_layer_stride > SIZE_MAX / (unsigned)box->depth) {
         free(transfer);
         *out_transfer = NULL;
         return NULL;
      }
      staging_size = staging_layer_stride * (unsigned)box->depth;
      if (!ps5_transfer_alloc_staging(transfer, staging_size) ||
          (packed && (!resource->stencil_data ||
                      resource->stencil_allocation_size <
                         stencil_layer_size *
                         ps5_texture_level_layers(&resource->base, 0)))) {
         ps5_transfer_free_staging(transfer);
         free(transfer);
         *out_transfer = NULL;
         return NULL;
      }
      if (usage & PIPE_MAP_READ) {
         ps5_flush_gpu_data(resource->data, resource->allocation_size);
         if (packed)
            ps5_flush_gpu_data(resource->stencil_data,
                               resource->stencil_allocation_size);
         for (unsigned z = 0; z < (unsigned)box->depth; ++z) {
            const size_t depth_layer =
               ((size_t)(unsigned)box->z + z) * depth_layer_size;
            const size_t stencil_layer =
               ((size_t)(unsigned)box->z + z) * stencil_layer_size;

            for (unsigned y = 0; y < (unsigned)box->height; ++y) {
               for (unsigned x = 0; x < (unsigned)box->width; ++x) {
                  const size_t depth_offset = depth_layer +
                     ps5_tiled_depth_offset((unsigned)box->x + x,
                                            (unsigned)box->y + y,
                                            resource->base.width0, (unsigned)box->z + z);
                  uint8_t *pixel = (uint8_t *)transfer->staging +
                     z * staging_layer_stride + y * staging_stride +
                     x * format_size;

                  if (depth_offset > resource->allocation_size ||
                      resource->allocation_size - depth_offset < 4) {
                     ps5_transfer_free_staging(transfer);
                     free(transfer);
                     *out_transfer = NULL;
                     return NULL;
                  }
                  memset(pixel, 0, format_size);
                  memcpy(pixel, resource->data + depth_offset, 4);
                  if (packed) {
                     const size_t stencil_offset = stencil_layer +
                        ps5_tiled_stencil_offset((unsigned)box->x + x,
                           (unsigned)box->y + y, resource->base.width0,
                           (unsigned)box->z + z);

                     if (stencil_offset >=
                           resource->stencil_allocation_size) {
                        ps5_transfer_free_staging(transfer);
                        free(transfer);
                        *out_transfer = NULL;
                        return NULL;
                     }
                     pixel[4] = resource->stencil_data[stencil_offset];
                  }
               }
            }
         }
      }
      transfer->base.stride = staging_stride;
      transfer->base.layer_stride = staging_layer_stride;
      transfer->base.offset = 0;
      return transfer->staging;
   }

   if ((usage & (PIPE_MAP_READ | PIPE_MAP_WRITE)) &&
       (resource->base.bind & PIPE_BIND_RENDER_TARGET) &&
       !ps5_linear_sampled_layout(&resource->base)) {
      size_t staging_stride;
      size_t staging_size;
      unsigned y;

      format_size = ps5_texture_format_size(resource->base.format);
      if (level || box->depth != 1 ||
          !ps5_render_target_format(resource->base.format) ||
          (!(PS5_ENABLE_PADDED_FBO_CANDIDATE ||
             PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) &&
           (resource->base.width0 != PS5_RENDER_WIDTH ||
            resource->base.height0 != PS5_RENDER_HEIGHT)) ||
          box->width <= 0 || box->height <= 0) {
         free(transfer);
         *out_transfer = NULL;
         return NULL;
      }
      staging_stride = (size_t)(unsigned)box->width * format_size;
      if (staging_stride > SIZE_MAX / (unsigned)box->height) {
         free(transfer);
         *out_transfer = NULL;
         return NULL;
      }
      staging_size = staging_stride * (unsigned)box->height;
      if (!ps5_transfer_alloc_staging(transfer, staging_size)) {
         free(transfer);
         *out_transfer = NULL;
         return NULL;
      }
      if (usage & PIPE_MAP_READ) {
         const size_t layer_base =
            (size_t)(unsigned)box->z * resource->layer_stride;

         ps5_flush_gpu_data(resource->data, resource->allocation_size);
         for (y = 0; y < (unsigned)box->height; ++y) {
            unsigned x;

            for (x = 0; x < (unsigned)box->width; ++x) {
               size_t tiled = layer_base + ps5_tiled_color_offset(
                  resource->base.format,
                  (unsigned)box->x + x, (unsigned)box->y + y,
                  ps5_tiled_rgba8_width(resource), (unsigned)box->z);

               if (tiled > resource->allocation_size ||
                   resource->allocation_size - tiled < format_size) {
                  ps5_transfer_free_staging(transfer);
                  free(transfer);
                  *out_transfer = NULL;
                  return NULL;
               }
               memcpy((uint8_t *)transfer->staging +
                         y * staging_stride + x * format_size,
                      resource->data + tiled, format_size);
            }
         }
      }
      transfer->base.stride = staging_stride;
      transfer->base.layer_stride = staging_size;
      transfer->base.offset = 0;
      return transfer->staging;
   }
   if ((usage & PIPE_MAP_READ) && resource->base.target == PIPE_BUFFER)
      ps5_flush_gpu_data(resource->data + offset, (size_t)box->width);
   return resource->data + offset;
}

static void
ps5_publish_transfer_write(struct pipe_transfer *transfer, const struct pipe_box *box)
{
   if (!transfer || transfer->resource->target != PIPE_BUFFER ||
       !(transfer->usage & PIPE_MAP_WRITE))
      return;
   struct ps5_resource *resource = (struct ps5_resource *)transfer->resource;
   const int x = box ? box->x : 0;
   const int width = box ? box->width : transfer->box.width;
   if (transfer->box.x < 0 || x < 0 || width <= 0 ||
       x > transfer->box.width || width > transfer->box.width - x)
      return;
   size_t offset = (size_t)transfer->box.x + (unsigned)x;
   if (offset <= resource->size && (size_t)width <= resource->size - offset)
      ps5_flush_gpu_data(resource->data + offset, (size_t)width);
}

static void
ps5_transfer_flush_region(struct pipe_context *context,
                          struct pipe_transfer *transfer,
                          const struct pipe_box *box)
{
   if (!transfer || transfer->resource->target != PIPE_BUFFER ||
       (transfer->usage & PIPE_MAP_READ) ||
       !(transfer->usage & PIPE_MAP_UNSYNCHRONIZED))
      ps5_draw_batch_drain_buffer(transfer ? transfer->resource : NULL);
   (void)context;
   (void)transfer;
   ps5_publish_transfer_write(transfer, box);
}

static void
ps5_transfer_unmap(struct pipe_context *context,
                   struct pipe_transfer *transfer)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {7, ps5_prepare_clock()};
#endif

   struct ps5_transfer *ps5 = (struct ps5_transfer *)transfer;
   struct ps5_resource *resource =
      (struct ps5_resource *)transfer->resource;

   (void)context;
   if (resource->base.target != PIPE_BUFFER || (transfer->usage & PIPE_MAP_READ) ||
       !(transfer->usage & PIPE_MAP_UNSYNCHRONIZED))
      ps5_draw_batch_drain_buffer(transfer->resource);
   if (ps5->staging && (transfer->usage & PIPE_MAP_WRITE) &&
       (transfer->resource->bind & PIPE_BIND_DEPTH_STENCIL) &&
       !resource->depth_staging_size &&
       (transfer->resource->format == PIPE_FORMAT_Z32_FLOAT ||
        transfer->resource->format ==
           PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)) {
      const bool packed = resource->base.format ==
                          PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      const size_t depth_layer_size = ps5_tiled_depth_surface_size(
         resource->base.width0, resource->base.height0, 1);
      const size_t stencil_layer_size = ps5_tiled_stencil_surface_size(
         resource->base.width0, resource->base.height0);
      const unsigned format_size =
         ps5_texture_format_size(resource->base.format);

      for (unsigned z = 0; z < (unsigned)transfer->box.depth; ++z) {
         const size_t depth_layer =
            ((size_t)(unsigned)transfer->box.z + z) * depth_layer_size;
         const size_t stencil_layer =
            ((size_t)(unsigned)transfer->box.z + z) * stencil_layer_size;

         for (unsigned y = 0; y < (unsigned)transfer->box.height; ++y) {
            for (unsigned x = 0; x < (unsigned)transfer->box.width; ++x) {
               const size_t depth_offset = depth_layer +
                  ps5_tiled_depth_offset((unsigned)transfer->box.x + x,
                     (unsigned)transfer->box.y + y,
                     resource->base.width0, (unsigned)transfer->box.z + z);
               const uint8_t *pixel = (const uint8_t *)ps5->staging +
                  z * transfer->layer_stride + y * transfer->stride +
                  x * format_size;

               if (depth_offset <= resource->allocation_size &&
                   resource->allocation_size - depth_offset >= 4)
                  memcpy(resource->data + depth_offset, pixel, 4);
               if (packed) {
                  const size_t stencil_offset = stencil_layer +
                     ps5_tiled_stencil_offset(
                        (unsigned)transfer->box.x + x,
                        (unsigned)transfer->box.y + y,
                        resource->base.width0, (unsigned)transfer->box.z + z);

                  if (stencil_offset < resource->stencil_allocation_size)
                     resource->stencil_data[stencil_offset] = pixel[4];
               }
            }
         }
      }
      ps5_flush_gpu_data(resource->data, resource->allocation_size);
      if (packed)
         ps5_flush_gpu_data(resource->stencil_data,
                            resource->stencil_allocation_size);
   } else if (ps5->staging && (transfer->usage & PIPE_MAP_WRITE)) {
      size_t layer_base =
         (size_t)(unsigned)transfer->box.z * resource->layer_stride;
      unsigned format_size = ps5_texture_format_size(resource->base.format);

      for (unsigned y = 0; y < (unsigned)transfer->box.height; ++y) {
         for (unsigned x = 0; x < (unsigned)transfer->box.width; ++x) {
            const unsigned samples = MAX2(resource->base.nr_samples, 1u);
            for (unsigned sample = 0; sample < samples; ++sample) {
               size_t tiled = layer_base + (samples == 4
                  ? ps5_tiled_color_msaa4_offset(resource->base.format,
                       (unsigned)transfer->box.x + x, (unsigned)transfer->box.y + y,
                       sample, resource->base.width0, (unsigned)transfer->box.z)
                  : ps5_tiled_color_offset(resource->base.format,
                       (unsigned)transfer->box.x + x, (unsigned)transfer->box.y + y,
                       ps5_tiled_rgba8_width(resource), (unsigned)transfer->box.z));

               if (tiled <= resource->allocation_size &&
                   resource->allocation_size - tiled >= format_size)
                  memcpy(resource->data + tiled,
                         (uint8_t *)ps5->staging +
                            y * transfer->stride + x * format_size,
                         format_size);
            }
         }
      }
      ps5_flush_gpu_data(resource->data, resource->allocation_size);
   }
   if (!(transfer->usage & PIPE_MAP_FLUSH_EXPLICIT))
      ps5_publish_transfer_write(transfer, NULL);
   ps5_transfer_free_staging(ps5);
   free(ps5);
}

static void
ps5_blit_scissor_bounds(const struct pipe_blit_info *info,
                        unsigned *min_x, unsigned *min_y,
                        unsigned *max_x, unsigned *max_y)
{
   if (info->scissor_enable) {
      *min_x = info->scissor.minx;
      *min_y = info->scissor.miny;
      *max_x = info->scissor.maxx;
      *max_y = info->scissor.maxy;
   } else {
      *min_x = 0;
      *min_y = 0;
      *max_x = UINT_MAX;
      *max_y = UINT_MAX;
   }
}

static bool
ps5_color_view_format_compatible(enum pipe_format storage,
                                 enum pipe_format view)
{
   return storage == view ||
          util_format_linear(storage) == util_format_linear(view) ||
          /* CopyImage uses canonical color views to preserve the stored bits,
           * not convert between the resources' original numeric formats. */
          (!util_format_is_depth_or_stencil(storage) &&
           !util_format_is_depth_or_stencil(view) &&
           util_format_get_blockwidth(storage) == 1 &&
           util_format_get_blockheight(storage) == 1 &&
           util_format_get_blockwidth(view) == 1 &&
           util_format_get_blockheight(view) == 1 &&
           util_format_get_blocksize(storage) != 0 &&
           util_format_get_blocksize(storage) == util_format_get_blocksize(view));
}

static void
ps5_resolve_color_msaa4(struct pipe_context *context,
                        const struct pipe_blit_info *info)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *source = (struct ps5_resource *)info->src.resource;
   const bool integer = util_format_is_pure_integer(info->src.format);
   const bool sample0_only = info->sample0_only || integer;
   struct pipe_transfer *dst_transfer = NULL;
   size_t source_layer_base;
   uint8_t *dst;
   unsigned min_x, min_y, max_x, max_y;
   unsigned src_width;
   unsigned src_height;
   unsigned src_pixel_size;
   unsigned dst_pixel_size;
   int64_t src_x;
   int64_t src_y;

   if (info->mask != PIPE_MASK_RGBA || info->src.level || info->dst.level ||
       !ps5_color_view_format_compatible(info->src.resource->format,
                                         info->src.format) ||
       !ps5_color_view_format_compatible(info->dst.resource->format,
                                         info->dst.format) ||
       !ps5_msaa4_color_format(info->src.format) ||
       !ps5_render_target_format(info->dst.format) ||
       info->src.resource->nr_samples != 4 ||
       info->src.resource->nr_storage_samples != 4 ||
       info->dst.resource->nr_samples > 1 || info->dst_sample ||
       info->num_window_rectangles ||
       info->alpha_blend ||
       (info->filter != PIPE_TEX_FILTER_NEAREST &&
        info->filter != PIPE_TEX_FILTER_LINEAR) ||
       !info->src.box.width || !info->src.box.height ||
       info->src.box.width == INT_MIN ||
       info->src.box.height == INT_MIN ||
       info->dst.box.x < 0 || info->dst.box.y < 0 ||
       info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       info->src.box.depth != 1 || info->dst.box.depth != 1 ||
       (unsigned)abs(info->src.box.width) !=
          (unsigned)info->dst.box.width ||
       (unsigned)abs(info->src.box.height) !=
          (unsigned)info->dst.box.height ||
       info->src.box.z < 0 ||
       (unsigned)info->src.box.z >= source->base.array_size ||
       (unsigned)info->dst.box.x + (unsigned)info->dst.box.width >
          info->dst.resource->width0 ||
       (unsigned)info->dst.box.y + (unsigned)info->dst.box.height >
          info->dst.resource->height0) {
      printf("[ps5-gallium] msaa4-resolve rejected mask=%x fmt=%u/%u res=%u/%u samples=%u/%u "
             "filter=%u src=%d,%d,%d %dx%d dst=%d,%d %dx%d dstsize=%ux%u scissor=%u alpha=%u\n",
             info->mask, info->src.format, info->dst.format, info->src.resource->format,
             info->dst.resource->format, info->src.resource->nr_samples,
             info->dst.resource->nr_samples, info->filter, info->src.box.x, info->src.box.y,
             info->src.box.z, info->src.box.width, info->src.box.height, info->dst.box.x,
             info->dst.box.y, info->dst.box.width, info->dst.box.height,
             info->dst.resource->width0, info->dst.resource->height0, info->scissor_enable,
             info->alpha_blend);
      return;
   }
   src_width = (unsigned)abs(info->src.box.width);
   src_height = (unsigned)abs(info->src.box.height);
   src_x = info->src.box.x;
   src_y = info->src.box.y;
   if (info->src.box.width < 0)
      src_x += info->src.box.width;
   if (info->src.box.height < 0)
      src_y += info->src.box.height;
   if (src_x < 0 || src_y < 0 ||
       (uint64_t)src_x + src_width > source->base.width0 ||
       (uint64_t)src_y + src_height > source->base.height0) {
      printf("[ps5-gallium] msaa4-resolve rejected\n");
      return;
   }
   if (info->render_condition_enable &&
       !ps5_render_condition_passes(ps5))
      return;

   dst = ps5_transfer_map(context, info->dst.resource, info->dst.level,
                          PIPE_MAP_WRITE |
                             (info->scissor_enable ? PIPE_MAP_READ : 0),
                          &info->dst.box, &dst_transfer);
   if (!dst)
      return;
   src_pixel_size = ps5_texture_format_size(info->src.format);
   dst_pixel_size = ps5_texture_format_size(info->dst.format);
   if (!src_pixel_size || !dst_pixel_size) {
      ps5_transfer_unmap(context, dst_transfer);
      return;
   }
   source_layer_base = (size_t)(unsigned)info->src.box.z *
                       source->layer_stride;
   ps5_blit_scissor_bounds(info, &min_x, &min_y, &max_x, &max_y);
   ps5_flush_gpu_data(source->data, source->allocation_size);
   for (unsigned y = 0; y < (unsigned)info->dst.box.height; ++y) {
      for (unsigned x = 0; x < (unsigned)info->dst.box.width; ++x) {
         union pipe_color_union resolved = {{0}};
         unsigned dst_x = (unsigned)info->dst.box.x + x;
         unsigned dst_y = (unsigned)info->dst.box.y + y;

         if (dst_x < min_x || dst_x >= max_x ||
             dst_y < min_y || dst_y >= max_y)
            continue;

         for (unsigned sample = 0; sample < (sample0_only ? 1u : 4u);
              ++sample) {
            union pipe_color_union value;
            unsigned source_x = (unsigned)src_x +
               (info->src.box.width < 0 ? src_width - 1u - x : x);
            unsigned source_y = (unsigned)src_y +
               (info->src.box.height < 0 ? src_height - 1u - y : y);
            size_t offset = source_layer_base +
               ps5_tiled_color_msaa4_offset(
                  info->src.format, source_x, source_y, sample,
                  source->base.width0, (unsigned)info->src.box.z);

            if (offset > source->allocation_size ||
               source->allocation_size - offset < src_pixel_size) {
               ps5_transfer_unmap(context, dst_transfer);
               return;
            }
            for (unsigned channel = 0; channel < 4; ++channel)
               value.ui[channel] = 0;
            util_format_unpack_rgba(info->src.format, value.ui,
                                    source->data + offset, 1);
            if (sample0_only) {
               resolved = value;
            } else {
               for (unsigned channel = 0; channel < 4; ++channel)
                  resolved.f[channel] += value.f[channel] * 0.25f;
            }
         }
         if (info->swizzle_enable) {
            union pipe_color_union swizzled;

            util_format_apply_color_swizzle(&swizzled, &resolved,
                                            info->swizzle, integer);
            resolved = swizzled;
         }
         util_format_pack_rgba(info->dst.format,
                               dst + (size_t)y * dst_transfer->stride +
                                  x * dst_pixel_size,
                               resolved.ui, 1);
      }
   }
   ps5_transfer_unmap(context, dst_transfer);
   printf("[ps5-gallium] msaa4-resolve color=%dx%d\n",
          info->dst.box.width, info->dst.box.height);
}

/* Restrict a local resource copy to one subresource. Mip-chain depth images
 * keep canonical linear pixels: completed draws copy tiled scratch back there.
 * Do not reuse scratch here: two levels of the same texture share that storage.
 * Tiled offsets still need the original layer index for the slice XOR. */
static bool
ps5_depth_blit_layer(struct ps5_resource *resource, unsigned level, int layer,
                     unsigned mask)
{
   const unsigned samples = resource->base.nr_samples;
   size_t depth_size, stencil_size;

   if (level > resource->base.last_level ||
       level >= ARRAY_SIZE(resource->level_offset) || level >= 32 || layer < 0 ||
       !resource->base.width0 || !resource->base.height0 ||
       !ps5_depth_render_target(resource->base.target) ||
       resource->base.target == PIPE_TEXTURE_3D ||
       (unsigned)layer >= resource->base.array_size ||
       ((resource->base.target == PIPE_TEXTURE_1D ||
         resource->base.target == PIPE_TEXTURE_2D) && layer) ||
       (samples > 1 && samples != 4))
      return false;
   if (resource->depth_staging_size) {
      const bool packed = resource->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      const unsigned pixel_size = packed ? 8 : 4;
      const unsigned width = MAX2(resource->base.width0 >> level, 1u);
      const unsigned height = MAX2(resource->base.height0 >> level, 1u);
      const size_t stride = resource->level_stride[level];
      size_t offset, span;

      if (samples > 1 || !resource->data ||
          (resource->base.format != PIPE_FORMAT_Z32_FLOAT && !packed) ||
          ((mask & PIPE_MASK_S) && !packed) ||
          !resource->layer_stride || resource->layer_stride > resource->size ||
          stride < (uint64_t)width * pixel_size ||
          stride > SIZE_MAX / height ||
          (size_t)(unsigned)layer > resource->size / resource->layer_stride)
         return false;
      offset = (size_t)(unsigned)layer * resource->layer_stride;
      span = (height - 1u) * stride + (size_t)width * pixel_size;
      if (resource->level_offset[level] > resource->layer_stride ||
          span > resource->layer_stride - resource->level_offset[level] ||
          resource->level_offset[level] > resource->size - offset)
         return false;
      offset += resource->level_offset[level];
      if (span > resource->size - offset || offset > resource->allocation_size ||
          span > resource->allocation_size - offset)
         return false;
      resource->data += offset;
      resource->allocation_size = span;
      resource->stencil_data = packed ? resource->data + sizeof(float) : NULL;
      resource->stencil_allocation_size = packed ? span - sizeof(float) : 0;
      resource->base.width0 = width;
      resource->base.height0 = height;
      resource->level_stride[0] = stride;
      return true;
   }
   if (level || resource->base.last_level)
      return false;
   depth_size = ps5_tiled_depth_surface_size(
      resource->base.width0, resource->base.height0, samples);
   stencil_size = ps5_tiled_stencil_surface_size_samples(
      resource->base.width0, resource->base.height0, samples);
   if (!depth_size || !stencil_size ||
       ((mask & PIPE_MASK_Z) &&
        (!resource->data ||
         (size_t)(unsigned)layer >= resource->allocation_size / depth_size)) ||
       ((mask & PIPE_MASK_S) &&
        (!resource->stencil_data ||
         (size_t)(unsigned)layer >=
            resource->stencil_allocation_size / stencil_size)))
      return false;
   if (mask & PIPE_MASK_Z) {
      resource->data += (size_t)(unsigned)layer * depth_size;
      resource->allocation_size = depth_size;
   }
   if (mask & PIPE_MASK_S) {
      resource->stencil_data += (size_t)(unsigned)layer * stencil_size;
      resource->stencil_allocation_size = stencil_size;
   }
   return true;
}

static size_t
ps5_depth_blit_offset(const struct ps5_resource *resource, unsigned x, unsigned y,
                       unsigned sample, unsigned layer, bool stencil)
{
   if (resource->depth_staging_size)
      return (size_t)y * resource->level_stride[0] + (size_t)x *
         (resource->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ? 8u : 4u);
   if (resource->base.nr_samples == 4)
      return stencil ? ps5_tiled_stencil_msaa4_offset(x, y, sample, resource->base.width0, layer)
                     : ps5_tiled_depth_msaa4_offset(x, y, sample, resource->base.width0, layer);
   return stencil ? ps5_tiled_stencil_offset(x, y, resource->base.width0, layer)
                  : ps5_tiled_depth_offset(x, y, resource->base.width0, layer);
}

static void
ps5_resolve_depth_stencil_msaa4(struct pipe_context *context,
                                const struct pipe_blit_info *info)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *source = (struct ps5_resource *)info->src.resource;
   struct ps5_resource *destination =
      (struct ps5_resource *)info->dst.resource;
   const bool packed = source &&
      source->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
   const bool copy_multisample = destination &&
      destination->base.nr_samples == 4 &&
      destination->base.nr_storage_samples == 4;
   unsigned min_x, min_y, max_x, max_y;
   size_t source_depth_size;
   size_t destination_depth_size;
   size_t source_stencil_size;
   size_t destination_stencil_size;
   unsigned src_width;
   unsigned src_height;
   int64_t src_x;
   int64_t src_y;

   struct ps5_resource source_layer, destination_layer;
   if (source && destination) {
      source_layer = *source;
      destination_layer = *destination;
      source = &source_layer;
      destination = &destination_layer;
   }
   if (!source || !destination || !(info->mask & PIPE_MASK_ZS) ||
       (info->mask & ~PIPE_MASK_ZS) ||
       !ps5_depth_blit_layer(source, info->src.level, info->src.box.z, info->mask) ||
       !ps5_depth_blit_layer(destination, info->dst.level, info->dst.box.z, info->mask) ||
       info->src.box.depth != 1 || info->dst.box.depth != 1 ||
       !info->src.box.width || !info->src.box.height ||
       info->src.box.width == INT_MIN || info->src.box.height == INT_MIN ||
       info->dst.box.x < 0 || info->dst.box.y < 0 ||
       info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       (unsigned)abs(info->src.box.width) !=
          (unsigned)info->dst.box.width ||
       (unsigned)abs(info->src.box.height) !=
          (unsigned)info->dst.box.height ||
       info->src.format != info->dst.format ||
       info->src.format != source->base.format ||
       info->dst.format != destination->base.format ||
       (source->base.format != PIPE_FORMAT_Z32_FLOAT && !packed) ||
       ((info->mask & PIPE_MASK_S) && !packed) ||
       source->base.nr_samples != 4 ||
       source->base.nr_storage_samples != 4 ||
       ((destination->base.nr_samples > 1 ||
         destination->base.nr_storage_samples > 1) && !copy_multisample) ||
       info->dst_sample ||
       info->sample0_only || info->swizzle_enable ||
       info->num_window_rectangles || info->alpha_blend ||
       info->filter != PIPE_TEX_FILTER_NEAREST ||
       (uint64_t)(unsigned)info->dst.box.x +
          (unsigned)info->dst.box.width > destination->base.width0 ||
       (uint64_t)(unsigned)info->dst.box.y +
          (unsigned)info->dst.box.height > destination->base.height0) {
      printf("[ps5-gallium] msaa4-resolve depth-stencil rejected\n");
      return;
   }
   src_width = (unsigned)abs(info->src.box.width);
   src_height = (unsigned)abs(info->src.box.height);
   src_x = info->src.box.x;
   src_y = info->src.box.y;
   if (info->src.box.width < 0)
      src_x += info->src.box.width;
   if (info->src.box.height < 0)
      src_y += info->src.box.height;
   if (src_x < 0 || src_y < 0 ||
       (uint64_t)src_x + src_width > source->base.width0 ||
       (uint64_t)src_y + src_height > source->base.height0) {
      printf("[ps5-gallium] msaa4-resolve depth-stencil rejected\n");
      return;
   }
   if (info->render_condition_enable && !ps5_render_condition_passes(ps5))
      return;

   source_depth_size = source->allocation_size;
   destination_depth_size = destination->allocation_size;
   source_stencil_size = source->stencil_allocation_size;
   destination_stencil_size = destination->stencil_allocation_size;

   ps5_blit_scissor_bounds(info, &min_x, &min_y, &max_x, &max_y);
   if (info->mask & PIPE_MASK_Z)
      ps5_flush_gpu_data(source->data, source_depth_size);
   if (info->mask & PIPE_MASK_S)
      ps5_flush_gpu_data(source->stencil_data, source_stencil_size);
   for (unsigned y = 0; y < (unsigned)info->dst.box.height; ++y) {
      for (unsigned x = 0; x < (unsigned)info->dst.box.width; ++x) {
         unsigned source_x = (unsigned)src_x +
            (info->src.box.width < 0 ? src_width - 1u - x : x);
         unsigned source_y = (unsigned)src_y +
            (info->src.box.height < 0 ? src_height - 1u - y : y);
         unsigned dst_x = (unsigned)info->dst.box.x + x;
         unsigned dst_y = (unsigned)info->dst.box.y + y;

         if (dst_x < min_x || dst_x >= max_x ||
             dst_y < min_y || dst_y >= max_y)
            continue;
         for (unsigned sample = 0; sample < (copy_multisample ? 4u : 1u);
              ++sample) {
            if (info->mask & PIPE_MASK_Z) {
               size_t src_offset = ps5_depth_blit_offset(
                  source, source_x, source_y, sample, info->src.box.z, false);
               size_t dst_offset = ps5_depth_blit_offset(
                  destination, dst_x, dst_y,
                  copy_multisample ? sample : 0, info->dst.box.z, false);

               if (src_offset > source_depth_size ||
                   source_depth_size - src_offset < sizeof(float) ||
                   dst_offset > destination_depth_size ||
                   destination_depth_size - dst_offset < sizeof(float))
                  return;
               memcpy(destination->data + dst_offset,
                      source->data + src_offset, sizeof(float));
            }
            if (info->mask & PIPE_MASK_S) {
               size_t src_offset = ps5_depth_blit_offset(
                  source, source_x, source_y, sample, info->src.box.z, true);
               size_t dst_offset = ps5_depth_blit_offset(
                  destination, dst_x, dst_y,
                  copy_multisample ? sample : 0, info->dst.box.z, true);

               if (src_offset >= source_stencil_size ||
                   dst_offset >= destination_stencil_size)
                  return;
               destination->stencil_data[dst_offset] =
                  source->stencil_data[src_offset];
            }
         }
      }
   }
   if (info->mask & PIPE_MASK_Z)
      ps5_flush_gpu_data(destination->data, destination_depth_size);
   if (info->mask & PIPE_MASK_S)
      ps5_flush_gpu_data(destination->stencil_data,
                         destination_stencil_size);
   printf("[ps5-gallium] msaa4-%s depth-stencil=%dx%d mask=%x\n",
          copy_multisample ? "copy" : "resolve",
          info->dst.box.width, info->dst.box.height, info->mask);
}

static void
ps5_replicate_depth_stencil_msaa4(struct pipe_context *context,
                                  const struct pipe_blit_info *info)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *source = (struct ps5_resource *)info->src.resource;
   struct ps5_resource *destination =
      (struct ps5_resource *)info->dst.resource;
   const bool packed = source &&
      source->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
   unsigned min_x, min_y, max_x, max_y;
   size_t source_depth_size;
   size_t destination_depth_size;
   size_t source_stencil_size;
   size_t destination_stencil_size;

   struct ps5_resource source_layer, destination_layer;
   if (source && destination) {
      source_layer = *source;
      destination_layer = *destination;
      source = &source_layer;
      destination = &destination_layer;
   }
   if (!source || !destination || !(info->mask & PIPE_MASK_ZS) ||
       (info->mask & ~PIPE_MASK_ZS) ||
       !ps5_depth_blit_layer(source, info->src.level, info->src.box.z, info->mask) ||
       !ps5_depth_blit_layer(destination, info->dst.level, info->dst.box.z, info->mask) ||
       info->src.box.depth != 1 || info->dst.box.depth != 1 ||
       info->src.box.width <= 0 || info->src.box.height <= 0 ||
       info->src.box.width != info->dst.box.width ||
       info->src.box.height != info->dst.box.height ||
       info->src.box.x < 0 || info->src.box.y < 0 ||
       info->dst.box.x < 0 || info->dst.box.y < 0 ||
       info->src.format != info->dst.format ||
       info->src.format != source->base.format ||
       info->dst.format != destination->base.format ||
       (source->base.format != PIPE_FORMAT_Z32_FLOAT && !packed) ||
       ((info->mask & PIPE_MASK_S) && !packed) ||
       source->base.nr_samples > 1 ||
       destination->base.nr_samples != 4 ||
       destination->base.nr_storage_samples != 4 || info->dst_sample ||
       info->sample0_only || info->swizzle_enable ||
       info->num_window_rectangles || info->alpha_blend ||
       info->filter != PIPE_TEX_FILTER_NEAREST ||
       (uint64_t)(unsigned)info->src.box.x +
          (unsigned)info->src.box.width > source->base.width0 ||
       (uint64_t)(unsigned)info->src.box.y +
          (unsigned)info->src.box.height > source->base.height0 ||
       (uint64_t)(unsigned)info->dst.box.x +
          (unsigned)info->dst.box.width > destination->base.width0 ||
       (uint64_t)(unsigned)info->dst.box.y +
          (unsigned)info->dst.box.height > destination->base.height0) {
      printf("[ps5-gallium] msaa4-replicate depth-stencil rejected\n");
      return;
   }
   if (info->render_condition_enable && !ps5_render_condition_passes(ps5))
      return;

   source_depth_size = source->allocation_size;
   destination_depth_size = destination->allocation_size;
   source_stencil_size = source->stencil_allocation_size;
   destination_stencil_size = destination->stencil_allocation_size;

   ps5_blit_scissor_bounds(info, &min_x, &min_y, &max_x, &max_y);
   if (info->mask & PIPE_MASK_Z)
      ps5_flush_gpu_data(source->data, source_depth_size);
   if (info->mask & PIPE_MASK_S)
      ps5_flush_gpu_data(source->stencil_data, source_stencil_size);
   for (unsigned y = 0; y < (unsigned)info->dst.box.height; ++y) {
      for (unsigned x = 0; x < (unsigned)info->dst.box.width; ++x) {
         unsigned src_x = (unsigned)info->src.box.x + x;
         unsigned src_y = (unsigned)info->src.box.y + y;
         unsigned dst_x = (unsigned)info->dst.box.x + x;
         unsigned dst_y = (unsigned)info->dst.box.y + y;
         size_t src_depth_offset = 0;
         size_t src_stencil_offset = 0;

         if (dst_x < min_x || dst_x >= max_x ||
             dst_y < min_y || dst_y >= max_y)
            continue;
         if (info->mask & PIPE_MASK_Z) {
            src_depth_offset = ps5_depth_blit_offset(
               source, src_x, src_y, 0, info->src.box.z, false);
            if (src_depth_offset > source_depth_size ||
                source_depth_size - src_depth_offset < sizeof(float))
               return;
         }
         if (info->mask & PIPE_MASK_S) {
            src_stencil_offset = ps5_depth_blit_offset(
               source, src_x, src_y, 0, info->src.box.z, true);
            if (src_stencil_offset >= source_stencil_size)
               return;
         }
         for (unsigned sample = 0; sample < 4; ++sample) {
            if (info->mask & PIPE_MASK_Z) {
               size_t dst_offset = ps5_depth_blit_offset(
                  destination, dst_x, dst_y, sample, info->dst.box.z, false);

               if (dst_offset > destination_depth_size ||
                   destination_depth_size - dst_offset < sizeof(float))
                  return;
               memcpy(destination->data + dst_offset,
                      source->data + src_depth_offset, sizeof(float));
            }
            if (info->mask & PIPE_MASK_S) {
               size_t dst_offset = ps5_depth_blit_offset(
                  destination, dst_x, dst_y, sample, info->dst.box.z, true);

               if (dst_offset >= destination_stencil_size)
                  return;
               destination->stencil_data[dst_offset] =
                  source->stencil_data[src_stencil_offset];
            }
         }
      }
   }
   if (info->mask & PIPE_MASK_Z)
      ps5_flush_gpu_data(destination->data, destination_depth_size);
   if (info->mask & PIPE_MASK_S)
      ps5_flush_gpu_data(destination->stencil_data,
                         destination_stencil_size);
   printf("[ps5-gallium] msaa4-replicate depth-stencil=%dx%d mask=%x\n",
          info->dst.box.width, info->dst.box.height, info->mask);
}

static bool
ps5_blit_gpu_color(struct ps5_context *context, const struct pipe_blit_info *info)
{
   /* ponytail: plain-color copies using existing sampler layouts and
    * directly renderable views. Keep the measured floor and other fallbacks. */
   if (!PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE || !PS5_ENABLE_MRT_CANDIDATE ||
       !PS5_ENABLE_UBO_CANDIDATE || !context || !info ||
       !info->src.resource || !info->dst.resource ||
       info->mask != PIPE_MASK_RGBA ||
       (info->filter != PIPE_TEX_FILTER_NEAREST && info->filter != PIPE_TEX_FILTER_LINEAR) ||
       info->num_window_rectangles || info->alpha_blend || info->swizzle_enable ||
       info->sample0_only || info->dst_sample || context->render_condition_query ||
       context->stream_output_target_count ||
       !info->src.box.width || !info->src.box.height ||
       info->src.box.width == INT_MIN || info->src.box.height == INT_MIN ||
       info->dst.box.width <= 0 || info->dst.box.height <= 0)
      return false;

   int64_t left = info->dst.box.x, bottom = info->dst.box.y;
   int64_t right = left + info->dst.box.width, top = bottom + info->dst.box.height;
   if (info->scissor_enable) {
      left = MAX2(left, info->scissor.minx);
      bottom = MAX2(bottom, info->scissor.miny);
      right = MIN2(right, info->scissor.maxx);
      top = MIN2(top, info->scissor.maxy);
   }
   /* The CPU fallback maps the destination box, which must lie inside the
    * surface: a box clipped by the surface edge always takes the GPU path. */
   const int64_t dst_width = MAX2(info->dst.resource->width0 >> info->dst.level, 1u);
   const int64_t dst_height = MAX2(info->dst.resource->height0 >> info->dst.level, 1u);
   const bool dst_clipped = info->dst.box.x < 0 || info->dst.box.y < 0 ||
      (int64_t)info->dst.box.x + info->dst.box.width > dst_width ||
      (int64_t)info->dst.box.y + info->dst.box.height > dst_height;
   if (right <= left || top <= bottom ||
       (!dst_clipped &&
        (uint64_t)(right - left) * (top - bottom) < PS5_GPU_BLIT_MIN_PIXELS))
      return false;

   const bool resolve = info->src.resource->nr_samples == 4 &&
                        info->src.resource->nr_storage_samples == 4;
   /* Equal-sized color resolves accept either filter; no scaling occurs. */
   if (resolve && (!PS5_ENABLE_MSAA4_CANDIDATE ||
                   abs(info->src.box.width) != info->dst.box.width ||
                   abs(info->src.box.height) != info->dst.box.height))
      return false;

   const struct ps5_resource *resources[] = {
      (const struct ps5_resource *)info->src.resource,
      (const struct ps5_resource *)info->dst.resource,
   };
   const struct pipe_box *boxes[] = {&info->src.box, &info->dst.box};
   const unsigned levels[] = {info->src.level, info->dst.level};
   uintptr_t addresses[2];
   size_t spans[2];
   if (info->src.format != info->dst.format ||
       (info->src.format != PIPE_FORMAT_R8G8B8A8_UNORM &&
        info->src.format != PIPE_FORMAT_R8_UNORM &&
        info->src.format != PIPE_FORMAT_R8G8_UNORM &&
        info->src.format != PIPE_FORMAT_R16G16B16A16_FLOAT) ||
       (resolve && info->src.format != PIPE_FORMAT_R8G8B8A8_UNORM))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      const struct ps5_resource *r = resources[i];
      const struct pipe_box *b = boxes[i];
      if ((r->base.target != PIPE_TEXTURE_2D && r->base.target != PIPE_TEXTURE_2D_ARRAY) ||
          r->base.depth0 != 1 || !r->base.array_size ||
          (r->base.target == PIPE_TEXTURE_2D && r->base.array_size != 1) ||
          levels[i] > r->base.last_level || r->base.last_level >= ARRAY_SIZE(r->level_stride) ||
          r->base.last_level >= 16 ||
          ((!resolve || i) &&
           (r->base.nr_samples > 1 || r->base.nr_storage_samples > 1)) ||
          r->base.format != info->src.format ||
          !(r->base.bind & PIPE_BIND_RENDER_TARGET) ||
          (r->base.bind & PIPE_BIND_DISPLAY_TARGET) ||
          r->depth_staging_size ||
          !r->data || !r->allocation_size ||
          r->allocation_size > UINTPTR_MAX - (uintptr_t)r->data ||
          !r->base.width0 || !r->base.height0 ||
          r->base.width0 > PS5_MAX_COLOR_WIDTH || r->base.height0 > PS5_MAX_COLOR_HEIGHT ||
          (i == 0 && (b->x < 0 || b->y < 0)) || b->z < 0 ||
          (unsigned)b->z >= r->base.array_size || b->depth != 1)
         return false;
      unsigned width = MAX2(r->base.width0 >> levels[i], 1u);
      unsigned height = MAX2(r->base.height0 >> levels[i], 1u);
      /* The destination may extend past the surface (GL clips it with a
       * scissor); the blitter's quad is clipped to the scissor below. */
      if (i == 0 &&
          ((unsigned)b->x > width || (unsigned)b->y > height ||
           (int64_t)b->x + b->width < 0 || (int64_t)b->y + b->height < 0 ||
           (int64_t)b->x + b->width > width || (int64_t)b->y + b->height > height))
         return false;
      if (i == 1) {
         left = MAX2(left, 0);
         bottom = MAX2(bottom, 0);
         right = MIN2(right, (int64_t)width);
         top = MIN2(top, (int64_t)height);
         if (right <= left || top <= bottom)
            return false;
      }
      struct pipe_surface selected = {
         .texture = (struct pipe_resource *)&r->base, .format = r->base.format,
         .level = levels[i], .first_layer = b->z, .last_layer = b->z,
      };
      unsigned pitch = ps5_linear_color_pitch(&selected);
      size_t offset = 0;
      if (pitch) {
         offset = (size_t)b->z * r->layer_stride + r->level_offset[levels[i]];
         spans[i] = (size_t)(height - 1u) * pitch +
                    (size_t)width * ps5_texture_format_size(r->base.format);
      } else {
         if ((r->base.target != PIPE_TEXTURE_2D && !(resolve && !i)) ||
             r->base.last_level || (!resolve && b->z) ||
             ps5_linear_sampled_layout(&r->base) || r->render_staging_size)
            return false;
         offset = (size_t)b->z * r->layer_stride;
         spans[i] = resolve && !i ? ps5_tiled_color_msaa4_surface_size(
             r->base.format, r->base.width0, r->base.height0) :
             ps5_tiled_color_surface_size(r->base.format, r->base.width0,
                                           r->base.height0);
      }
      if (!spans[i] || offset > r->allocation_size || spans[i] > r->allocation_size - offset)
         return false;
      addresses[i] = (uintptr_t)r->data + offset;
   }
   /* Different mips/layers can share an allocation, never the same bytes. */
   if (addresses[0] < addresses[1] + spans[1] &&
       addresses[1] < addresses[0] + spans[0])
      return false;

   if (!context->blitter)
      context->blitter = util_blitter_create(&context->base);
   struct blitter_context *blitter = context->blitter;
   if (!blitter || blitter->running || !util_blitter_is_blit_supported(blitter, info))
      return false;
   struct pipe_surface surface;
   struct pipe_sampler_view templ;
   util_blitter_default_dst_texture(&surface, info->dst.resource, info->dst.level, info->dst.box.z);
   util_blitter_default_src_texture(blitter, &templ, info->src.resource, info->src.level);
   struct pipe_sampler_view *view = context->base.create_sampler_view(
      &context->base, info->src.resource, &templ);
   if (!view)
      return false;

   uint16_t viewport_valid = context->viewport_valid;
   uint16_t scissor_valid = context->scissor_valid;
   bool framebuffer_valid = context->framebuffer_valid, queries_enabled = context->queries_enabled;
   unsigned draws_before = context->draw_calls;
   util_blitter_save_vertex_buffers(blitter, context->vertex_buffers, context->vertex_buffer_count);
   util_blitter_save_vertex_elements(blitter, context->vertex_elements);
   util_blitter_save_vertex_shader(blitter, context->vs);
   util_blitter_save_tessctrl_shader(blitter, context->tcs);
   util_blitter_save_tesseval_shader(blitter, context->tes);
   util_blitter_save_geometry_shader(blitter, context->gs);
   util_blitter_save_so_targets(blitter, 0, NULL, context->stream_output_primitive);
   util_blitter_save_rasterizer(blitter, context->rasterizer);
   util_blitter_save_fragment_shader(blitter, context->fs);
   util_blitter_save_depth_stencil_alpha(blitter, context->depth_stencil_alpha);
   util_blitter_save_blend(blitter, context->blend);
   util_blitter_save_stencil_ref(blitter, &context->stencil_ref);
   util_blitter_save_viewport(blitter, &context->viewport[0]);
   util_blitter_save_scissor(blitter, &context->scissor[0]);
   util_blitter_save_sample_mask(blitter, context->sample_mask, 1);
   util_blitter_save_framebuffer(blitter, &context->framebuffer);
   util_blitter_save_fragment_sampler_states(blitter, PS5_MAX_TEXTURE_UNITS, context->samplers[1]);
   util_blitter_save_fragment_sampler_views(blitter, PS5_MAX_TEXTURE_UNITS, context->sampler_views[1]);
   /* Framebuffer changes submit prior producers. The deferred draw retains
    * both textures and uploader storage, and its normal GPU barriers order
    * sampling after those producers. CPU fallbacks still drain below. */
   context->deferred_blitter_draw = true;
   const struct pipe_scissor_state clip = {
      .minx = (uint16_t)left, .miny = (uint16_t)bottom,
      .maxx = (uint16_t)right, .maxy = (uint16_t)top,
   };
   util_blitter_blit_generic(blitter, &surface, &info->dst.box, view, &info->src.box,
                             info->src.resource->width0, info->src.resource->height0,
                             info->mask, info->filter, &clip,
                             false, false, 0, NULL);
   context->deferred_blitter_draw = false;
   pipe_sampler_view_reference(&view, NULL);
   context->viewport_valid = viewport_valid;
   context->scissor_valid = scissor_valid;
   context->framebuffer_valid = framebuffer_valid;
   context->queries_enabled = queries_enabled;
   if (context->draw_calls == draws_before)
      context->last_draw_status = -30;
   if (context->last_draw_status != 0 || draws_before < 3)
      printf("[ps5-gallium] blit-gpu-color status=%d draws=%u size=%dx%d\n",
             context->last_draw_status, context->draw_calls - draws_before,
             info->dst.box.width, info->dst.box.height);
   return true;
}

static void
ps5_blit(struct pipe_context *context, const struct pipe_blit_info *info)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *dst_resource;
   struct pipe_transfer *src_transfer = NULL;
   struct pipe_transfer *dst_transfer = NULL;
   struct pipe_box src_box;
   uint8_t *src;
   uint8_t *dst;
   unsigned src_width;
   unsigned src_height;
   unsigned src_pixel_size;
   unsigned dst_pixel_size;
   int64_t src_x;
   int64_t src_y;
   bool direct_tiled_dst;
   unsigned min_x, min_y, max_x, max_y;

   /* u_blitter suspends query accounting; deferred slots retain that state. */
   if (ps5_blit_gpu_color(ps5, info))
      return;
   ps5_draw_batch_drain_buffer(info ? info->src.resource : NULL);
   ps5_draw_batch_drain_buffer(info ? info->dst.resource : NULL);
   if (PS5_ENABLE_MSAA4_CANDIDATE && info && info->src.resource &&
       info->src.resource->nr_samples == 4) {
      if (info->mask & PIPE_MASK_ZS)
         ps5_resolve_depth_stencil_msaa4(context, info);
      else
         ps5_resolve_color_msaa4(context, info);
      return;
   }
   if (PS5_ENABLE_MSAA4_CANDIDATE && info && info->dst.resource &&
       info->dst.resource->nr_samples == 4 &&
       (info->mask & PIPE_MASK_ZS)) {
      ps5_replicate_depth_stencil_msaa4(context, info);
      return;
   }

   if (info && (info->mask & PIPE_MASK_ZS)) {
      struct ps5_resource *source =
         (struct ps5_resource *)info->src.resource;
      struct ps5_resource *destination =
         (struct ps5_resource *)info->dst.resource;
      unsigned source_width, source_height, destination_width, destination_height;
      size_t source_depth_size, destination_depth_size;
      size_t source_stencil_size, destination_stencil_size;
      const bool packed = source &&
         source->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      unsigned source_box_width = 0;
      unsigned source_box_height = 0;
      int64_t source_x = 0;
      int64_t source_y = 0;

      struct ps5_resource source_layer, destination_layer;
      if (source && destination) {
         source_layer = *source;
         destination_layer = *destination;
         source = &source_layer;
         destination = &destination_layer;
      }
      if (!source || !destination ||
          !(info->mask & PIPE_MASK_ZS) ||
          (info->mask & ~PIPE_MASK_ZS) ||
          !ps5_depth_blit_layer(source, info->src.level, info->src.box.z, info->mask) ||
          !ps5_depth_blit_layer(destination, info->dst.level, info->dst.box.z, info->mask) ||
          info->src.box.depth != 1 || info->dst.box.depth != 1 ||
          !info->src.box.width || !info->src.box.height ||
          info->src.box.width == INT_MIN ||
          info->src.box.height == INT_MIN ||
          info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
          info->src.format != info->dst.format ||
          info->src.format != source->base.format ||
          info->dst.format != destination->base.format ||
          (source->base.format != PIPE_FORMAT_Z32_FLOAT && !packed) ||
          ((info->mask & PIPE_MASK_S) && !packed) ||
          source->base.nr_samples > 1 ||
          destination->base.nr_samples > 1 || info->dst_sample ||
          info->sample0_only ||
          info->swizzle_enable || info->num_window_rectangles ||
          info->alpha_blend || info->filter != PIPE_TEX_FILTER_NEAREST) {
         printf("[ps5-gallium] software-blit depth-stencil rejected\n");
         return;
      }
      source_width = source->base.width0;
      source_height = source->base.height0;
      destination_width = destination->base.width0;
      destination_height = destination->base.height0;
      source_depth_size = source->allocation_size;
      destination_depth_size = destination->allocation_size;
      source_stencil_size = source->stencil_allocation_size;
      destination_stencil_size = destination->stencil_allocation_size;
      source_box_width = info->src.box.width < 0
                            ? (unsigned)-info->src.box.width
                            : (unsigned)info->src.box.width;
      source_box_height = info->src.box.height < 0
                             ? (unsigned)-info->src.box.height
                             : (unsigned)info->src.box.height;
      source_x = info->src.box.x;
      source_y = info->src.box.y;
      if (info->src.box.width < 0)
         source_x += info->src.box.width;
      if (info->src.box.height < 0)
         source_y += info->src.box.height;
      if (source_x < 0 || source_y < 0 || info->dst.box.x < 0 ||
          info->dst.box.y < 0 ||
          (uint64_t)source_x + source_box_width > source_width ||
          (uint64_t)source_y + source_box_height > source_height ||
          (uint64_t)(unsigned)info->dst.box.x +
             (unsigned)info->dst.box.width > destination_width ||
          (uint64_t)(unsigned)info->dst.box.y +
             (unsigned)info->dst.box.height > destination_height) {
         printf("[ps5-gallium] software-blit depth-stencil rejected\n");
         return;
      }
      if (info->render_condition_enable &&
          !ps5_render_condition_passes(ps5))
         return;
      ps5_blit_scissor_bounds(info, &min_x, &min_y, &max_x, &max_y);
      if (info->mask & PIPE_MASK_Z)
         ps5_flush_gpu_data(source->data, source_depth_size);
      if (info->mask & PIPE_MASK_S)
         ps5_flush_gpu_data(source->stencil_data, source_stencil_size);
      /* Full, identically tiled slices need no per-pixel address conversion.
       * Equal layer indices also preserve the hardware's layer XOR. */
      const bool copy_planes = !source->depth_staging_size &&
         !destination->depth_staging_size &&
         info->src.box.z == info->dst.box.z &&
         !info->src.box.x && !info->src.box.y &&
         !info->dst.box.x && !info->dst.box.y &&
         info->src.box.width == (int)source_width &&
         info->src.box.height == (int)source_height &&
         info->dst.box.width == (int)source_width &&
         info->dst.box.height == (int)source_height &&
         source_width == destination_width && source_height == destination_height &&
         !min_x && !min_y && max_x >= source_width && max_y >= source_height &&
         source_depth_size == destination_depth_size &&
         source_stencil_size == destination_stencil_size;
      if (copy_planes) {
         if (info->mask & PIPE_MASK_Z)
            memmove(destination->data, source->data, source_depth_size);
         if (info->mask & PIPE_MASK_S)
            memmove(destination->stencil_data, source->stencil_data, source_stencil_size);
      } else
      for (unsigned y = 0; y < (unsigned)info->dst.box.height; ++y) {
         unsigned sy = (unsigned)(((UINT64_C(2) * y + 1u) *
                                   source_box_height) /
                                  (UINT64_C(2) *
                                   (unsigned)info->dst.box.height));

         if (info->src.box.height < 0)
            sy = source_box_height - 1u - sy;
         for (unsigned x = 0; x < (unsigned)info->dst.box.width; ++x) {
            unsigned sx = (unsigned)(((UINT64_C(2) * x + 1u) *
                                      source_box_width) /
                                     (UINT64_C(2) *
                                      (unsigned)info->dst.box.width));
            unsigned src_px;
            unsigned src_py = (unsigned)source_y + sy;
            unsigned dst_px = (unsigned)info->dst.box.x + x;
            unsigned dst_py = (unsigned)info->dst.box.y + y;

            if (dst_px < min_x || dst_px >= max_x ||
                dst_py < min_y || dst_py >= max_y)
               continue;
            if (info->src.box.width < 0)
               sx = source_box_width - 1u - sx;
            src_px = (unsigned)source_x + sx;
            if (info->mask & PIPE_MASK_Z) {
               size_t src_offset = ps5_depth_blit_offset(
                  source, src_px, src_py, 0, info->src.box.z, false);
               size_t dst_offset = ps5_depth_blit_offset(
                  destination, dst_px, dst_py, 0, info->dst.box.z, false);

               if (src_offset > source_depth_size ||
                   source_depth_size - src_offset < sizeof(float) ||
                   dst_offset > destination_depth_size ||
                   destination_depth_size - dst_offset < sizeof(float)) {
                  printf("[ps5-gallium] software-blit depth-stencil rejected\n");
                  return;
               }
               memmove(destination->data + dst_offset,
                       source->data + src_offset, sizeof(float));
            }
            if (info->mask & PIPE_MASK_S) {
               size_t src_offset = ps5_depth_blit_offset(
                  source, src_px, src_py, 0, info->src.box.z, true);
               size_t dst_offset = ps5_depth_blit_offset(
                  destination, dst_px, dst_py, 0, info->dst.box.z, true);

               if (src_offset >= source_stencil_size ||
                   dst_offset >= destination_stencil_size) {
                  printf("[ps5-gallium] software-blit depth-stencil rejected\n");
                  return;
               }
               destination->stencil_data[dst_offset] =
                  source->stencil_data[src_offset];
            }
         }
      }
      if (info->mask & PIPE_MASK_Z)
         ps5_flush_gpu_data(destination->data, destination_depth_size);
      if (info->mask & PIPE_MASK_S)
         ps5_flush_gpu_data(destination->stencil_data,
                            destination_stencil_size);
      printf("[ps5-gallium] software-blit depth-stencil=%ux%u->%dx%d mask=%x bulk=%u\n",
             source_box_width, source_box_height, info->dst.box.width,
             info->dst.box.height, info->mask, copy_planes);
      return;
   }

   if (!info || !info->src.resource || !info->dst.resource ||
       info->mask != PIPE_MASK_RGBA ||
       info->src.box.depth != 1 || info->dst.box.depth != 1 ||
       info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       !info->src.box.width || !info->src.box.height ||
       info->src.box.width == INT_MIN || info->src.box.height == INT_MIN ||
       !ps5_color_view_format_compatible(info->src.resource->format,
                                         info->src.format) ||
       !ps5_color_view_format_compatible(info->dst.resource->format,
                                         info->dst.format) ||
       info->src.resource->nr_samples > 1 ||
       info->dst.resource->nr_samples > 1 || info->dst_sample ||
       info->sample0_only ||
       info->num_window_rectangles || info->alpha_blend ||
       (info->filter != PIPE_TEX_FILTER_NEAREST &&
        info->filter != PIPE_TEX_FILTER_LINEAR) ||
       !ps5_render_target_format(info->src.format) ||
       !ps5_render_target_format(info->dst.format) ||
       util_format_is_pure_uint(info->src.format) !=
          util_format_is_pure_uint(info->dst.format) ||
       util_format_is_pure_sint(info->src.format) !=
          util_format_is_pure_sint(info->dst.format)) {
      printf("[ps5-gallium] software-blit rejected\n");
      return;
   }
   if (info->render_condition_enable &&
       !ps5_render_condition_passes(ps5))
      return;
   ps5_blit_scissor_bounds(info, &min_x, &min_y, &max_x, &max_y);

   src_width = info->src.box.width < 0
                  ? (unsigned)-info->src.box.width
                  : (unsigned)info->src.box.width;
   src_height = info->src.box.height < 0
                   ? (unsigned)-info->src.box.height
                   : (unsigned)info->src.box.height;
   if (info->filter == PIPE_TEX_FILTER_LINEAR &&
       (util_format_is_pure_integer(info->src.format) ||
        util_format_is_pure_integer(info->dst.format))) {
      printf("[ps5-gallium] software-blit rejected linear-format=%u\n",
             info->src.format);
      return;
   }
   src_x = info->src.box.x;
   src_y = info->src.box.y;
   if (info->src.box.width < 0)
      src_x += info->src.box.width;
   if (info->src.box.height < 0)
      src_y += info->src.box.height;
   if (src_x < INT_MIN || src_x > INT_MAX ||
       src_y < INT_MIN || src_y > INT_MAX)
      return;
   src_box = info->src.box;
   src_box.x = (int)src_x;
   src_box.y = (int)src_y;
   src_box.width = (int)src_width;
   src_box.height = (int)src_height;

   unsigned linear_x = 0, linear_y = 0;
   if (info->filter == PIPE_TEX_FILTER_LINEAR) {
      /* Filtering clamps at the source IMAGE boundary, not the blit rectangle.
       * Map a one-texel halo so inset/upscaled edges can read their neighbors. */
      if (info->src.level > info->src.resource->last_level || info->src.level >= 32)
         return;
      int64_t width = MAX2(info->src.resource->width0 >> info->src.level, 1u);
      int64_t height = MAX2(info->src.resource->height0 >> info->src.level, 1u);
      if (src_x < 0 || src_y < 0 || src_x + src_width > width || src_y + src_height > height)
         return;
      int64_t left = MAX2(src_x - 1, 0), bottom = MAX2(src_y - 1, 0);
      int64_t right = MIN2(src_x + src_width + 1, width);
      int64_t top = MIN2(src_y + src_height + 1, height);
      if (right - left > INT_MAX || top - bottom > INT_MAX)
         return;
      linear_x = (unsigned)(src_x - left);
      linear_y = (unsigned)(src_y - bottom);
      src_box.x = (int)left; src_box.y = (int)bottom;
      src_box.width = (int)(right - left); src_box.height = (int)(top - bottom);
   }

   src_pixel_size = ps5_texture_format_size(info->src.format);
   dst_pixel_size = ps5_texture_format_size(info->dst.format);
   if (!src_pixel_size || !dst_pixel_size)
      return;
   src = ps5_transfer_map(context, info->src.resource, info->src.level,
                          PIPE_MAP_READ, &src_box, &src_transfer);
   if (!src)
      return;
   dst_resource = (struct ps5_resource *)info->dst.resource;
   direct_tiled_dst =
      (dst_resource->base.bind & PIPE_BIND_RENDER_TARGET) &&
      !ps5_linear_sampled_layout(&dst_resource->base);
   if (direct_tiled_dst) {
      size_t ignored_offset;

      /* Linear transfers address mip levels; native tile offsets only level 0. */
      if (info->dst.level ||
          !ps5_map_bounds(dst_resource, info->dst.level, &info->dst.box,
                          &ignored_offset)) {
         ps5_transfer_unmap(context, src_transfer);
         return;
      }
      dst = dst_resource->data;
   } else {
      dst = ps5_transfer_map(context, info->dst.resource, info->dst.level,
                             PIPE_MAP_WRITE |
                                (info->scissor_enable ? PIPE_MAP_READ : 0),
                             &info->dst.box, &dst_transfer);
      if (!dst) {
         ps5_transfer_unmap(context, src_transfer);
         return;
      }
   }

   for (unsigned y = 0; y < (unsigned)info->dst.box.height; ++y) {
      unsigned sy = (unsigned)(((UINT64_C(2) * y + 1u) * src_height) /
                               (UINT64_C(2) *
                                (unsigned)info->dst.box.height));
      if (info->src.box.height < 0)
         sy = src_height - 1u - sy;
      for (unsigned x = 0; x < (unsigned)info->dst.box.width; ++x) {
         unsigned sx = (unsigned)(((UINT64_C(2) * x + 1u) * src_width) /
                                  (UINT64_C(2) *
                                   (unsigned)info->dst.box.width));
         uint8_t *dst_pixel;
         union pipe_color_union result;
         unsigned dst_x = (unsigned)info->dst.box.x + x;
         unsigned dst_y = (unsigned)info->dst.box.y + y;

         if (info->src.box.width < 0)
            sx = src_width - 1u - sx;
         if (dst_x < min_x || dst_x >= max_x ||
             dst_y < min_y || dst_y >= max_y)
            continue;
         if (direct_tiled_dst) {
            size_t tiled =
               (size_t)(unsigned)info->dst.box.z *
                  dst_resource->layer_stride +
               ps5_tiled_color_offset(
               dst_resource->base.format,
               (unsigned)info->dst.box.x + x,
               (unsigned)info->dst.box.y + y,
               ps5_tiled_rgba8_width(dst_resource), (unsigned)info->dst.box.z);

            if (tiled > dst_resource->allocation_size ||
                dst_resource->allocation_size - tiled < dst_pixel_size) {
               ps5_transfer_unmap(context, src_transfer);
               return;
            }
            dst_pixel = dst_resource->data + tiled;
         } else {
            dst_pixel = dst + (size_t)y * dst_transfer->stride +
                        (size_t)x * dst_pixel_size;
         }
         if (info->filter == PIPE_TEX_FILTER_LINEAR) {
            int64_t fx = (int64_t)(((UINT64_C(2) * x + 1u) * src_width
                                     << 16) /
                                    (UINT64_C(2) *
                                     (unsigned)info->dst.box.width)) -
                         INT64_C(0x8000);
            int64_t fy = (int64_t)(((UINT64_C(2) * y + 1u) * src_height
                                     << 16) /
                                    (UINT64_C(2) *
                                     (unsigned)info->dst.box.height)) -
                         INT64_C(0x8000);
            int64_t x0;
            int64_t y0;
            uint64_t wx1;
            uint64_t wy1;
            unsigned ix0;
            unsigned ix1;
            unsigned iy0;
            unsigned iy1;

            if (info->src.box.width < 0)
               fx = (int64_t)(src_width - 1u) * INT64_C(0x10000) - fx;
            if (info->src.box.height < 0)
               fy = (int64_t)(src_height - 1u) * INT64_C(0x10000) - fy;
            fx += (int64_t)linear_x * INT64_C(0x10000);
            fy += (int64_t)linear_y * INT64_C(0x10000);
            x0 = fx >= 0 ? fx / INT64_C(0x10000)
                         : -((-fx + INT64_C(0xffff)) / INT64_C(0x10000));
            y0 = fy >= 0 ? fy / INT64_C(0x10000)
                         : -((-fy + INT64_C(0xffff)) / INT64_C(0x10000));
            wx1 = (uint64_t)(fx - x0 * INT64_C(0x10000));
            wy1 = (uint64_t)(fy - y0 * INT64_C(0x10000));
            ix0 = (unsigned)CLAMP(x0, 0, (int64_t)src_box.width - 1);
            ix1 = (unsigned)CLAMP(x0 + 1, 0, (int64_t)src_box.width - 1);
            iy0 = (unsigned)CLAMP(y0, 0, (int64_t)src_box.height - 1);
            iy1 = (unsigned)CLAMP(y0 + 1, 0, (int64_t)src_box.height - 1);
            union pipe_color_union p00, p10, p01, p11;
            float tx = (float)wx1 / 65536.0f;
            float ty = (float)wy1 / 65536.0f;

            util_format_unpack_rgba(
               info->src.format, p00.ui,
               src + (size_t)iy0 * src_transfer->stride +
                  (size_t)ix0 * src_pixel_size, 1);
            util_format_unpack_rgba(
               info->src.format, p10.ui,
               src + (size_t)iy0 * src_transfer->stride +
                  (size_t)ix1 * src_pixel_size, 1);
            util_format_unpack_rgba(
               info->src.format, p01.ui,
               src + (size_t)iy1 * src_transfer->stride +
                  (size_t)ix0 * src_pixel_size, 1);
            util_format_unpack_rgba(
               info->src.format, p11.ui,
               src + (size_t)iy1 * src_transfer->stride +
                  (size_t)ix1 * src_pixel_size, 1);
            for (unsigned channel = 0; channel < 4; ++channel) {
               float top = p00.f[channel] +
                           (p10.f[channel] - p00.f[channel]) * tx;
               float bottom = p01.f[channel] +
                              (p11.f[channel] - p01.f[channel]) * tx;

               result.f[channel] = top + (bottom - top) * ty;
            }
         } else {
            const uint8_t *src_pixel =
               src + (size_t)sy * src_transfer->stride +
                     (size_t)sx * src_pixel_size;

            if (info->src.format == info->dst.format && !info->swizzle_enable) {
               memcpy(dst_pixel, src_pixel, src_pixel_size);
               continue;
            }
            util_format_unpack_rgba(info->src.format, result.ui, src_pixel, 1);
         }
         if (info->swizzle_enable) {
            union pipe_color_union swizzled;

            util_format_apply_color_swizzle(&swizzled, &result, info->swizzle,
                                            util_format_is_pure_integer(info->src.format));
            result = swizzled;
         }
         util_format_pack_rgba(info->dst.format, dst_pixel, result.ui, 1);
      }
   }
   if (direct_tiled_dst)
      ps5_flush_gpu_data(dst_resource->data, dst_resource->allocation_size);
   else
      ps5_transfer_unmap(context, dst_transfer);
   ps5_transfer_unmap(context, src_transfer);
   printf("[ps5-gallium] software-blit color=%ux%u->%dx%d filter=%u\n",
          src_width, src_height, info->dst.box.width, info->dst.box.height,
          info->filter);
}

static void
ps5_texture_subdata(struct pipe_context *context,
                    struct pipe_resource *resource, unsigned level,
                    unsigned usage, const struct pipe_box *box,
                    const void *data, unsigned stride,
                    uintptr_t layer_stride)
{
   struct pipe_transfer *transfer = NULL;
   const uint8_t *source = data;
   uint8_t *destination;
   unsigned format_size;
   unsigned z;

   if (!resource || !box || !data || box->depth < 1)
      return;
   format_size = ps5_texture_format_size(resource->format);
   if (!format_size)
      return;
   destination = context->texture_map(
      context, resource, level, usage | PIPE_MAP_WRITE, box, &transfer);
   if (!destination || !transfer)
      return;

   for (z = 0; z < (unsigned)box->depth; ++z) {
      unsigned y;

      for (y = 0; y < (unsigned)box->height; ++y)
         memcpy(destination + z * transfer->layer_stride +
                   y * transfer->stride,
                source + z * layer_stride + y * stride,
                (size_t)box->width * format_size);
   }
   context->texture_unmap(context, transfer);
}

static bool
ps5_generate_mipmap(struct pipe_context *context,
                    struct pipe_resource *base,
                    enum pipe_format format,
                    unsigned base_level, unsigned last_level,
                    unsigned first_layer, unsigned last_layer)
{
   struct ps5_resource *resource = (struct ps5_resource *)base;
   unsigned format_size;
   bool depth;
   unsigned components;

   ps5_draw_batch_drain_buffer(base);
   if (!PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE || !resource ||
       format != resource->base.format ||
       base_level >= last_level || last_level > resource->base.last_level ||
       last_level >= ARRAY_SIZE(resource->level_stride) || last_level >= 32 ||
       !ps5_linear_sampled_layout(&resource->base) ||
       !ps5_sampled_texture_target(resource->base.target) ||
       !(format_size = ps5_texture_format_size(format)) ||
       util_format_is_compressed(format) ||
       util_format_is_pure_integer(format))
      return false;

   /* Depth formats have no RGBA conversion callbacks. Preserve the stencil
    * plane when filtering packed depth/stencil mip levels. */
   depth = util_format_has_depth(util_format_description(format));
   components = depth ? 1 : 4;

   if (resource->base.target == PIPE_TEXTURE_3D) {
      unsigned base_depth = MAX2(resource->base.depth0 >> base_level, 1u);

      if (first_layer || last_layer + 1u < base_depth)
         return false;
   } else if (first_layer > last_layer ||
              last_layer >= resource->base.array_size) {
      return false;
   }

   ps5_flush_gpu_data(resource->data, resource->size);
   for (unsigned level = base_level + 1; level <= last_level; ++level) {
      unsigned src_width = MAX2(resource->base.width0 >> (level - 1), 1u);
      unsigned src_height = MAX2(resource->base.height0 >> (level - 1), 1u);
      unsigned dst_width = MAX2(resource->base.width0 >> level, 1u);
      unsigned dst_height = MAX2(resource->base.height0 >> level, 1u);
      unsigned dst_first = first_layer;
      unsigned dst_last = last_layer;
      unsigned src_depth = 1;
      unsigned dst_depth = 1;

      if (resource->base.target == PIPE_TEXTURE_3D) {
         src_depth = MAX2(resource->base.depth0 >> (level - 1), 1u);
         dst_depth = MAX2(resource->base.depth0 >> level, 1u);
         dst_first = 0;
         dst_last = dst_depth - 1;
      }

      for (unsigned layer = dst_first; layer <= dst_last; ++layer) {
         /* A linear 2x2 filter is the existing box average for exact halvings.
          * Reuse checked blits for large levels; CPU handles NPOT and the tail.
          * No recursive draw while inside a staging/queue lock. */
         if (!depth && (base->target == PIPE_TEXTURE_2D || base->target == PIPE_TEXTURE_2D_ARRAY) &&
             (src_width == 2u * dst_width || src_width == 1) &&
             (src_height == 2u * dst_height || src_height == 1) &&
             (uint64_t)dst_width * dst_height >= PS5_GPU_BLIT_MIN_PIXELS) {
            struct pipe_blit_info blit = {0};
            blit.src.resource = blit.dst.resource = base;
            blit.src.format = blit.dst.format = format;
            blit.src.level = level - 1;
            blit.dst.level = level;
            blit.src.box = (struct pipe_box){.x = 0, .y = 0, .z = layer,
               .width = src_width, .height = src_height, .depth = 1};
            blit.dst.box = (struct pipe_box){.x = 0, .y = 0, .z = layer,
               .width = dst_width, .height = dst_height, .depth = 1};
            blit.mask = PIPE_MASK_RGBA;
            blit.filter = PIPE_TEX_FILTER_LINEAR;
            if (ps5_blit_gpu_color((struct ps5_context *)context, &blit)) {
               if (((struct ps5_context *)context)->last_draw_status)
                  return true; /* Attempted GPU failure must not replay another path. */
               /* A later mip can fall back to CPU filtering of this output. */
               ps5_draw_batch_drain_buffer(base);
               ps5_flush_gpu_data(resource->data, resource->size);
               continue;
            }
         }
         unsigned src_z0 = resource->base.target == PIPE_TEXTURE_3D
                              ? (unsigned)((uint64_t)layer * src_depth /
                                           dst_depth)
                              : layer;
         unsigned src_z1 = resource->base.target == PIPE_TEXTURE_3D
                              ? (unsigned)((uint64_t)(layer + 1u) * src_depth /
                                           dst_depth)
                              : layer + 1u;

         if (src_z1 <= src_z0)
            src_z1 = src_z0 + 1u;
         for (unsigned y = 0; y < dst_height; ++y) {
            unsigned src_y0 = (unsigned)((uint64_t)y * src_height /
                                         dst_height);
            unsigned src_y1 = (unsigned)((uint64_t)(y + 1u) * src_height /
                                         dst_height);

            if (src_y1 <= src_y0)
               src_y1 = src_y0 + 1u;
            for (unsigned x = 0; x < dst_width; ++x) {
               unsigned src_x0 = (unsigned)((uint64_t)x * src_width /
                                            dst_width);
               unsigned src_x1 = (unsigned)((uint64_t)(x + 1u) * src_width /
                                            dst_width);
               float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
               unsigned samples = 0;

               if (src_x1 <= src_x0)
                  src_x1 = src_x0 + 1u;
               for (unsigned z = src_z0; z < src_z1; ++z) {
                  for (unsigned sy = src_y0; sy < src_y1; ++sy) {
                     for (unsigned sx = src_x0; sx < src_x1; ++sx) {
                        const uint8_t *source =
                           resource->data +
                           (size_t)z * resource->layer_stride +
                           resource->level_offset[level - 1] +
                           (size_t)sy * resource->level_stride[level - 1] +
                           (size_t)sx * format_size;
                        float sample[4];

                        if (depth)
                           util_format_unpack_z_float(format, sample, source, 1);
                        else
                           util_format_unpack_rgba(format, sample, source, 1);
                        for (unsigned component = 0; component < components;
                             ++component)
                           sum[component] += sample[component];
                        samples++;
                     }
                  }
               }
               for (unsigned component = 0; component < components; ++component)
                  sum[component] /= samples;
               uint8_t *destination =
                  resource->data +
                     (size_t)layer * resource->layer_stride +
                     resource->level_offset[level] +
                     (size_t)y * resource->level_stride[level] +
                     (size_t)x * format_size;
               if (depth)
                  util_format_pack_z_float(format, destination, sum, 1);
               else
                  util_format_pack_rgba(format, destination, sum, 1);
            }
         }
      }
   }
   ps5_flush_gpu_data(resource->data, resource->size);
   printf("[ps5-gallium] generate-mipmap format=%u levels=%u-%u layers=%u-%u\n",
          format, base_level, last_level, first_layer, last_layer);
   return true;
}

static void
ps5_fence_reference(struct pipe_screen *screen,
                    struct pipe_fence_handle **destination,
                    struct pipe_fence_handle *source)
{
   struct ps5_fence *old = (struct ps5_fence *)*destination;
   struct ps5_fence *next = (struct ps5_fence *)source;

   (void)screen;
   if (old == next)
      return;
   if (next)
      p_atomic_inc(&next->references);
   if (old && p_atomic_dec_zero(&old->references))
      free(old);
   *destination = source;
}

static bool
ps5_fence_finish(struct pipe_screen *screen, struct pipe_context *context,
                 struct pipe_fence_handle *fence, uint64_t timeout)
{
   (void)screen;
   (void)context;
#ifdef PS5_DEFERRED_DRAW_BATCH
   return !fence || ps5_draw_batch_fence_finish(((struct ps5_fence *)fence)->sequence, timeout);
#else
   (void)fence;
   (void)timeout;
   return true;
#endif
}

static uint64_t
ps5_get_timestamp(struct pipe_screen *screen)
{
   (void)screen;
   /* GPU completion does not modify CPU texture storage. */
   ps5_screen_submit_lock(NULL);
   ps5_screen_submit_unlock(NULL);
   return os_time_get_nano();
}

static struct pipe_query *
ps5_create_query(struct pipe_context *base, unsigned type, unsigned index)
{
   struct ps5_query *query;
   bool timer_query;
   bool occlusion_query;
   bool primitive_query;
   bool overflow_query;

   timer_query = PS5_ENABLE_TIMER_QUERY_CANDIDATE &&
      (type == PIPE_QUERY_TIMESTAMP ||
       type == PIPE_QUERY_TIMESTAMP_DISJOINT ||
       type == PIPE_QUERY_TIME_ELAPSED ||
       type == PIPE_QUERY_GPU_FINISHED);
   occlusion_query = PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE &&
      (type == PIPE_QUERY_OCCLUSION_COUNTER ||
       type == PIPE_QUERY_OCCLUSION_PREDICATE ||
       type == PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE);
   primitive_query = PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE &&
      (type == PIPE_QUERY_PRIMITIVES_GENERATED ||
       type == PIPE_QUERY_PRIMITIVES_EMITTED);
   overflow_query = PS5_ENABLE_GLSL_460_CANDIDATE &&
      (type == PIPE_QUERY_SO_OVERFLOW_PREDICATE ||
       type == PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE);
   if ((primitive_query && index >= PIPE_MAX_VERTEX_STREAMS) ||
       (!primitive_query && !overflow_query && index) ||
       (type == PIPE_QUERY_SO_OVERFLOW_PREDICATE &&
        index >= PIPE_MAX_VERTEX_STREAMS) ||
       (type == PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE && index) ||
       (!timer_query && !occlusion_query && !primitive_query &&
        !overflow_query))
      return NULL;
   query = calloc(1, sizeof(*query));
   if (!query)
      return NULL;
   query->type = type;
   query->index = index;
   if (occlusion_query) {
      struct pipe_resource templ;

      memset(&templ, 0, sizeof(templ));
      templ.target = PIPE_BUFFER;
      templ.format = PIPE_FORMAT_R8_UNORM;
      templ.width0 = PS5_OCCLUSION_QUERY_BYTES;
      templ.height0 = 1;
      templ.depth0 = 1;
      templ.array_size = 1;
      query->buffer = base->screen->resource_create(base->screen, &templ);
      if (!query->buffer) {
         free(query);
         return NULL;
      }
   }
   return (struct pipe_query *)query;
}

static void
ps5_destroy_query(struct pipe_context *base, struct pipe_query *pipe_query)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_query *query = (struct ps5_query *)pipe_query;

   if (!query)
      return;
   ps5_draw_batch_drain_query(query);
   if (context->active_occlusion_query == query)
      context->active_occlusion_query = NULL;
   for (unsigned i = 0; i < PIPE_MAX_VERTEX_STREAMS; ++i) {
      if (context->active_primitives_generated_query[i] == query)
         context->active_primitives_generated_query[i] = NULL;
      if (context->active_primitives_emitted_query[i] == query)
         context->active_primitives_emitted_query[i] = NULL;
   }
   for (unsigned i = 0; i <= PIPE_MAX_VERTEX_STREAMS; ++i)
      if (context->active_streamout_overflow_query[i] == query)
         context->active_streamout_overflow_query[i] = NULL;
   if (context->render_condition_query == query)
      context->render_condition_query = NULL;
   pipe_resource_reference(&query->buffer, NULL);
   free(query);
}

static bool
ps5_begin_query(struct pipe_context *base, struct pipe_query *pipe_query)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_query *query = (struct ps5_query *)pipe_query;
   struct ps5_query **primitive_query;
   struct ps5_query **overflow_query;

   if (!query || query->active)
      return false;
   /* A new counter does not depend on earlier draws. Reusing an occlusion
    * object must first collect any slots that still point at its old value. */
   ps5_draw_batch_drain_query(query);
   primitive_query = ps5_active_primitive_query(
      context, query->type, query->index);
   if (primitive_query) {
      if (!PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE || *primitive_query)
         return false;
      query->value = 0;
      query->active = true;
      query->ready = false;
      *primitive_query = query;
      return true;
   }
   overflow_query = ps5_active_streamout_overflow_query(
      context, query->type, query->index);
   if (overflow_query) {
      if (!PS5_ENABLE_GLSL_460_CANDIDATE || *overflow_query)
         return false;
      query->value = 0;
      query->active = true;
      query->ready = false;
      *overflow_query = query;
      return true;
   }
   if (query->type == PIPE_QUERY_OCCLUSION_COUNTER ||
       query->type == PIPE_QUERY_OCCLUSION_PREDICATE ||
       query->type == PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE) {
      if (!PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE || !query->buffer ||
          context->active_occlusion_query)
         return false;
      query->value = 0;
      query->active = true;
      query->ready = false;
      context->active_occlusion_query = query;
      return true;
   }
   if (query->type != PIPE_QUERY_TIME_ELAPSED)
      return false;
   ps5_screen_submit_lock(NULL);
   ps5_screen_submit_unlock(NULL);
   query->start = os_time_get_nano();
   query->end = 0;
   query->active = true;
   query->ready = false;
   return true;
}

static bool
ps5_end_query(struct pipe_context *base, struct pipe_query *pipe_query)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_query *query = (struct ps5_query *)pipe_query;
   struct ps5_query **primitive_query;
   struct ps5_query **overflow_query;

   if (!query)
      return false;
   primitive_query = ps5_active_primitive_query(
      context, query->type, query->index);
   if (primitive_query) {
      if (!query->active || *primitive_query != query)
         return false;
      *primitive_query = NULL;
      query->active = false;
      query->ready = true;
      return true;
   }
   overflow_query = ps5_active_streamout_overflow_query(
      context, query->type, query->index);
   if (overflow_query) {
      if (!query->active || *overflow_query != query)
         return false;
      *overflow_query = NULL;
      query->active = false;
      query->ready = true;
      return true;
   }
   if (query->type == PIPE_QUERY_OCCLUSION_COUNTER ||
       query->type == PIPE_QUERY_OCCLUSION_PREDICATE ||
       query->type == PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE) {
      if (!query->active || context->active_occlusion_query != query)
         return false;
      context->active_occlusion_query = NULL;
      query->active = false;
      query->ready = true;
      return true;
   }
   if (query->type == PIPE_QUERY_TIME_ELAPSED) {
      if (!query->active)
         return false;
   } else if (query->active) {
      return false;
   }
   ps5_screen_submit_lock(NULL);
   ps5_screen_submit_unlock(NULL);
   query->end = os_time_get_nano();
   query->active = false;
   query->ready = true;
   return true;
}

static bool
ps5_get_query_result(struct pipe_context *base,
                     struct pipe_query *pipe_query, bool wait,
                     union pipe_query_result *result)
{
   const struct ps5_query *query = (const struct ps5_query *)pipe_query;

   (void)base;
   (void)wait;
   if (!query || !query->ready || !result ||
       !ps5_draw_batch_query_ready(query, wait))
      return false;
   if (query->type == PIPE_QUERY_TIMESTAMP_DISJOINT) {
      result->timestamp_disjoint.frequency = UINT64_C(1000000000);
      result->timestamp_disjoint.disjoint = false;
   } else if (query->type == PIPE_QUERY_GPU_FINISHED) {
      result->b = true;
   } else if (query->type == PIPE_QUERY_OCCLUSION_PREDICATE ||
              query->type == PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE) {
      result->b = query->value != 0;
   } else if (query->type == PIPE_QUERY_OCCLUSION_COUNTER) {
      result->u64 = query->value;
   } else if (query->type == PIPE_QUERY_PRIMITIVES_GENERATED ||
              query->type == PIPE_QUERY_PRIMITIVES_EMITTED) {
      result->u64 = query->value;
   } else if (query->type == PIPE_QUERY_SO_OVERFLOW_PREDICATE ||
              query->type == PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE) {
      result->b = query->value != 0;
   } else {
      result->u64 = query->end - query->start;
   }
   return true;
}

static void
ps5_get_query_result_resource(struct pipe_context *base,
                              struct pipe_query *pipe_query,
                              enum pipe_query_flags flags,
                              enum pipe_query_value_type result_type,
                              int index, struct pipe_resource *resource,
                              unsigned offset)
{
   union pipe_query_result result = {0};
   uint64_t value;

   if (index < 0) {
      const struct ps5_query *query = (const struct ps5_query *)pipe_query;
      value = query && query->ready && ps5_draw_batch_query_ready(query, false);
   } else {
      if (!ps5_get_query_result(base, pipe_query,
                                flags & PIPE_QUERY_WAIT, &result))
         return;
      value = result.u64;
   }

   if (result_type == PIPE_QUERY_TYPE_I32) {
      int32_t out = value > INT32_MAX ? INT32_MAX : (int32_t)value;
      pipe_buffer_write(base, resource, offset, sizeof(out), &out);
   } else if (result_type == PIPE_QUERY_TYPE_U32) {
      uint32_t out = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
      pipe_buffer_write(base, resource, offset, sizeof(out), &out);
   } else {
      pipe_buffer_write(base, resource, offset, sizeof(value), &value);
   }
}

static void
ps5_set_active_query_state(struct pipe_context *base, bool enable)
{
   struct ps5_context *context = (struct ps5_context *)base;
   /* Slots capture occlusion ownership at draw time; primitive counts are
    * accounted when each draw is built. Changing scope needs no GPU wait. */
   context->queries_enabled = enable;
}

static void
ps5_render_condition(struct pipe_context *base, struct pipe_query *pipe_query,
                     bool condition, enum pipe_render_cond_flag mode)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_query *query = (struct ps5_query *)pipe_query;

   if (query)
      ps5_draw_batch_drain_query(query);
   if (query && query->type != PIPE_QUERY_OCCLUSION_COUNTER &&
       query->type != PIPE_QUERY_OCCLUSION_PREDICATE &&
       query->type != PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE &&
       query->type != PIPE_QUERY_SO_OVERFLOW_PREDICATE &&
       query->type != PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE)
      return;
   (void)mode;
   context->render_condition_query = query;
   context->render_condition_inverted = condition;
}

static bool
ps5_render_condition_passes(const struct ps5_context *context)
{
   const struct ps5_query *query = context->render_condition_query;

   if (!query || !query->ready)
      return true;
   return (query->value != 0) != context->render_condition_inverted;
}

static bool
ps5_prepare_occlusion_query(struct ps5_query *query)
{
   struct ps5_resource *resource;
   uint64_t *samples;

   if (!query || !query->buffer)
      return false;
   resource = (struct ps5_resource *)query->buffer;
   if (!resource->data || resource->size < PS5_OCCLUSION_QUERY_BYTES)
      return false;
   samples = (uint64_t *)resource->data;
   for (unsigned rb = 0; rb < PS5_OCCLUSION_MAX_RBS; ++rb) {
      samples[rb * 2u] = PS5_OCCLUSION_VALID_BIT;
      samples[rb * 2u + 1u] = PS5_OCCLUSION_VALID_BIT;
   }
   ps5_flush_gpu_data(samples, PS5_OCCLUSION_QUERY_BYTES);
   return true;
}

static bool
ps5_collect_occlusion_query_resource(struct ps5_query *query,
                                     struct pipe_resource *buffer)
{
   struct ps5_resource *resource;
   const uint64_t *samples;
   uint64_t value = 0;

   if (!query || !buffer)
      return false;
   resource = (struct ps5_resource *)buffer;
   if (!resource->data || resource->size < PS5_OCCLUSION_QUERY_BYTES)
      return false;
   ps5_flush_gpu_data(resource->data, PS5_OCCLUSION_QUERY_BYTES);
   samples = (const uint64_t *)resource->data;
   for (unsigned rb = 0; rb < PS5_OCCLUSION_MAX_RBS; ++rb) {
      uint64_t start = samples[rb * 2u];
      uint64_t end = samples[rb * 2u + 1u];

      if ((start & PS5_OCCLUSION_VALID_BIT) &&
          (end & PS5_OCCLUSION_VALID_BIT))
         value += end - start;
   }
   query->value += value;
   return true;
}

static bool
ps5_collect_occlusion_query(struct ps5_query *query)
{
   return query && ps5_collect_occlusion_query_resource(query, query->buffer);
}

static void
ps5_flush(struct pipe_context *context, struct pipe_fence_handle **out_fence,
          unsigned flags)
{
   struct ps5_fence *fence;

   (void)flags;
   if (!out_fence) {
      ps5_draw_batch_submit();
      return;
   }
   fence = calloc(1, sizeof(*fence));
   if (!fence) {
      /* Even allocation failure must submit preceding work. */
      ps5_draw_batch_submit();
      return;
   }
#ifdef PS5_DEFERRED_DRAW_BATCH
   fence->sequence = ps5_draw_batch_fence_submit();
#else
   ps5_draw_batch_drain();
#endif
   fence->references = 1;
   context->screen->fence_reference(context->screen, out_fence,
                                    (struct pipe_fence_handle *)fence);
   /* Drop the creator reference; out_fence retains the remaining reference. */
   fence->references--;
}

static void
ps5_memory_barrier(struct pipe_context *context, unsigned flags)
{
   /* Queued compute waits for earlier work and invalidates for later work in
    * its own commands. Deferred graphics retains its resources and emits cache
    * releases at submission boundaries; GPU consumers need that boundary, not a
    * CPU completion wait. */
   if (flags & ~PIPE_BARRIER_ALL)
      ps5_draw_batch_drain();
   /* Client access still requires its fence/finish; visibility does not require
    * retiring unrelated GPU work on this CPU thread. */
   else if (flags) {
#ifdef PS5_DEFERRED_DRAW_BATCH
      /* Only queued dispatches pending: they already order on the GPU. */
      if (ps5_draw_batch_compute_only())
         return;
#endif
      ps5_draw_batch_submit();
   }
   (void)context;
}

static void
ps5_texture_barrier(struct pipe_context *context, unsigned flags)
{
   /* Texture-barrier flags are a separate namespace: SAMPLER is bit zero,
    * which means MAPPED_BUFFER in memory_barrier. */
   if (flags & ~(PIPE_TEXTURE_BARRIER_SAMPLER | PIPE_TEXTURE_BARRIER_FRAMEBUFFER))
      ps5_draw_batch_drain();
   else if (flags)
      ps5_draw_batch_submit();
   (void)context;
}

static bool
ps5_draw_primitive(unsigned mode, unsigned count, uint32_t *primitive_type)
{
   if (!primitive_type)
      return false;

   switch (mode) {
   case MESA_PRIM_POINTS:
      *primitive_type = 1;
      return count >= 1;
   case MESA_PRIM_LINES:
      *primitive_type = 2;
      return count >= 2;
   case MESA_PRIM_LINE_STRIP:
      *primitive_type = 3;
      return count >= 2;
   case MESA_PRIM_TRIANGLES:
      *primitive_type = 4;
      return count >= 3;
   case MESA_PRIM_TRIANGLE_FAN:
      *primitive_type = 5;
      return count >= 3;
   case MESA_PRIM_TRIANGLE_STRIP:
      *primitive_type = 6;
      return count >= 3;
   case MESA_PRIM_LINES_ADJACENCY:
      *primitive_type = 10;
      return count >= 4;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      *primitive_type = 11;
      return count >= 4;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      *primitive_type = 12;
      return count >= 6;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      *primitive_type = 13;
      return count >= 6;
   case MESA_PRIM_PATCHES:
      *primitive_type = 9;
      return count >= 1;
   default:
      return false;
   }
}

static uint32_t
ps5_fragment_primitive_type(const struct ps5_context *context,
                            uint32_t draw_primitive_type)
{
   if (!context->gs) {
      if (!context->tcs || !context->tes)
         return draw_primitive_type;
      if (context->tes->nir->info.tess.point_mode)
         return 1;
      return context->tes->nir->info.tess._primitive_mode ==
                TESS_PRIMITIVE_ISOLINES ? 2 : 4;
   }

   switch (context->gs->nir->info.gs.output_primitive) {
   case MESA_PRIM_POINTS:
      return 1;
   case MESA_PRIM_LINES:
   case MESA_PRIM_LINE_STRIP:
      return 3;
   case MESA_PRIM_TRIANGLES:
   case MESA_PRIM_TRIANGLE_STRIP:
      return 6;
   default:
      return 0;
   }
}

static uint32_t
ps5_index_value(const void *indices, unsigned index_size, unsigned index)
{
   return index_size == 2 ? ((const uint16_t *)indices)[index] :
                            ((const uint32_t *)indices)[index];
}

/* Keep the original uint32 base-vertex wrap semantics. Only a range straddling
 * the positive-bias wrap point needs per-index effective-bound calculation. */
static bool
ps5_index_bounds(const void *indices, unsigned index_size, unsigned count, int bias,
                 unsigned *minimum, unsigned *maximum,
                 unsigned *effective_minimum, unsigned *effective_maximum)
{
   if (index_size == 4 && util_get_cpu_caps()->has_sse4_1) {
      _mesa_uint_array_min_max(indices, minimum, maximum, count);
   } else {
      *minimum = UINT32_MAX;
      *maximum = 0;
      for (unsigned i = 0; i < count; ++i) {
         unsigned value = ps5_index_value(indices, index_size, i);
         *minimum = MIN2(*minimum, value);
         *maximum = MAX2(*maximum, value);
      }
   }
   if (bias < 0 && *minimum < (uint64_t)-(int64_t)bias)
      return false;
   *effective_minimum = *minimum + (uint32_t)bias;
   *effective_maximum = *maximum + (uint32_t)bias;
   if (bias > 0 && *minimum <= UINT32_MAX - (uint32_t)bias &&
       *maximum > UINT32_MAX - (uint32_t)bias) {
      *effective_minimum = UINT32_MAX;
      *effective_maximum = 0;
      for (unsigned i = 0; i < count; ++i) {
         uint32_t value = ps5_index_value(indices, index_size, i) + (uint32_t)bias;
         *effective_minimum = MIN2(*effective_minimum, value);
         *effective_maximum = MAX2(*effective_maximum, value);
      }
   }
   return *effective_maximum != UINT32_MAX;
}

static void
ps5_log_indices(const void *indices, unsigned index_size, unsigned count,
                unsigned limit)
{
   unsigned logged = MIN2(count, limit);

   for (unsigned i = 0; i < logged; ++i)
      printf("%s%u", i ? "," : "",
             ps5_index_value(indices, index_size, i));
   printf("%s\n", logged < count ? ",..." : "");
}

static unsigned
ps5_streamout_vertices_per_primitive(enum mesa_prim primitive)
{
   switch (primitive) {
   case MESA_PRIM_POINTS: return 1;
   case MESA_PRIM_LINES: return 2;
   case MESA_PRIM_TRIANGLES: return 3;
   default: return 0;
   }
}

static uint64_t
ps5_draw_primitive_count(uint32_t primitive_type, unsigned vertex_count)
{
   switch (primitive_type) {
   case 1: return vertex_count;
   case 2: return vertex_count / 2u;
   case 3: return vertex_count > 1 ? vertex_count - 1u : 0;
   case 4: return vertex_count / 3u;
   case 5:
   case 6: return vertex_count > 2 ? vertex_count - 2u : 0;
   case 10: return vertex_count / 4u;
   case 11: return vertex_count > 3 ? vertex_count - 3u : 0;
   case 12: return vertex_count / 6u;
   case 13: return vertex_count > 4 ? (vertex_count - 4u) / 2u : 0;
   default: return 0;
   }
}

static bool
ps5_streamout_storage(struct ps5_context *context, struct pipe_resource **slot,
                      uint64_t bytes)
{
   if (!bytes || bytes > UINT32_MAX)
      return false;
   if (*slot && (*slot)->width0 >= bytes)
      return true;
   struct pipe_resource templ = {
      .target = PIPE_BUFFER, .format = PIPE_FORMAT_R8_UNORM,
      .width0 = bytes, .height0 = 1, .depth0 = 1, .array_size = 1,
   };
   struct pipe_resource *replacement =
      context->base.screen->resource_create(context->base.screen, &templ);
   if (!replacement)
      return false;
   pipe_resource_reference(slot, NULL);
   *slot = replacement;
   return true;
}

static int
ps5_streamout_record_compare(const void *a, const void *b)
{
   const struct ps5_streamout_record *x = a, *y = b;
   if (x->primitive != y->primitive)
      return x->primitive < y->primitive ? -1 : 1;
   return (x->invocation > y->invocation) - (x->invocation < y->invocation);
}

/* Canonical tessellation records: reserved[1] = first patch (TES) or GS
 * input primitive, invocation = GS instance, primitive = normalized ordinal.
 * The ordinal only orders subgroups that share one patch on one engine. */
static int
ps5_tess_streamout_record_compare(const void *a, const void *b)
{
   const struct ps5_streamout_record *x = a, *y = b;
   if (x->reserved[1] != y->reserved[1])
      return x->reserved[1] < y->reserved[1] ? -1 : 1;
   if (x->invocation != y->invocation)
      return x->invocation < y->invocation ? -1 : 1;
   return (x->primitive > y->primitive) - (x->primitive < y->primitive);
}

static bool
ps5_prepare_primitive_query(struct ps5_context *context,
                            const PsbcShaderMetadata *metadata,
                            uint32_t *user_data, unsigned user_data_count)
{
   if (!metadata->primitive_query_valid ||
       metadata->primitive_query_buffer_user_data_dword >= user_data_count ||
       metadata->primitive_query_state_user_data_dword >= user_data_count ||
       metadata->primitive_query_counter_offset > 48 ||
       (metadata->primitive_query_counter_offset & 3u) ||
       !ps5_streamout_storage(context, &context->primitive_query_storage, 64))
      return false;
   struct ps5_resource *storage = (void *)context->primitive_query_storage;
   if ((uintptr_t)storage->data >> 32 != metadata->address32_hi)
      return false;
   memset(storage->data, 0, 64);
   ps5_flush_gpu_data(storage->data, 64);
   user_data[metadata->primitive_query_buffer_user_data_dword] = (uint32_t)(uintptr_t)storage->data;
   user_data[metadata->primitive_query_state_user_data_dword] |= metadata->primitive_query_enable_mask;
   return true;
}

static bool
ps5_prepare_streamout(struct ps5_context *context,
                      const PsbcShaderMetadata *metadata,
                      struct ps5_resource *descriptor_resource,
                      uint32_t *user_data, unsigned user_data_count,
                      uint32_t primitive_type, unsigned draw_count,
                      unsigned instance_count, uint32_t *enabled_mask,
                      uint32_t size_dwords[4], uint32_t stride_dwords[4],
                      uint32_t offset_dwords[4], uint64_t *written_vertices)
{
   uint32_t *descriptors = (uint32_t *)(descriptor_resource->data +
                                        PS5_STREAMOUT_DESCRIPTOR_OFFSET);
   struct ps5_streamout_control *control =
      (struct ps5_streamout_control *)(descriptor_resource->data +
                                       PS5_STREAMOUT_CONTROL_OFFSET);
   const bool global_control = context->gs != NULL || context->tes != NULL;
   uintptr_t table_address = (uintptr_t)descriptors;
   unsigned vertices_per_primitive =
      ps5_streamout_vertices_per_primitive(context->stream_output_primitive);
   uint64_t primitives = global_control ? 0 :
      ps5_draw_primitive_count(primitive_type, draw_count);
   uint32_t mask;
   uint64_t record_capacity = 0;
   uint64_t staging_primitives = 0;

   if (!metadata->streamout_valid || !vertices_per_primitive ||
       instance_count != 1 ||
       metadata->streamout_buffer_table_user_data_dword >= user_data_count ||
       (uint32_t)(table_address >> 32) != metadata->address32_hi ||
       (global_control
          ? PS5_STREAMOUT_CONTROL_OFFSET + PS5_STREAMOUT_CONTROL_BYTES
          : PS5_STREAMOUT_DESCRIPTOR_OFFSET + 4u * 16u) >
          descriptor_resource->size ||
       primitives > UINT64_MAX / instance_count)
      return false;
   primitives *= instance_count;
   if (metadata->streamout_enabled_stream_buffers_mask >>
       (PIPE_MAX_VERTEX_STREAMS * PIPE_MAX_SO_BUFFERS))
      return false;
   mask = ps5_streamout_buffer_mask(
      metadata->streamout_enabled_stream_buffers_mask);
   if (!mask)
      return false;
   if (context->gs && !context->tes) {
      /* Worst-case staging preserves early primitives on overflow even when
       * later workgroups finish first. Never replay shaders with side effects.
       * ponytail: host compaction until a GPU ordered-prefix path is available. */
      record_capacity = ps5_draw_primitive_count(primitive_type, draw_count);
      record_capacity *= MAX2(context->gs->nir->info.gs.invocations, 1);
      staging_primitives = record_capacity * context->gs->nir->info.gs.vertices_out;
   } else if (context->tes) {
      if (!context->patch_vertices)
         return false;
      /* Maximum tessellation level is 64: quads produce at most 2*64*64
       * triangles, with fewer primitives for the other tessellation modes. */
      staging_primitives = (uint64_t)(draw_count / context->patch_vertices) * 8192u;
      if (context->gs)
         staging_primitives *= (uint64_t)MAX2(context->gs->nir->info.gs.invocations, 1) *
                               context->gs->nir->info.gs.vertices_out;
      if (staging_primitives > (UINT64_C(1) << 24))
         return false;
      /* ponytail: a full hardware ordinal cycle needs a wider ordering key. */
      record_capacity = MIN2(staging_primitives, PS5_TESS_STREAMOUT_ORDINAL_COUNT - 1u);
   }
   if (context->gs || context->tes) {
      if (!record_capacity || record_capacity > (UINT32_MAX - 64u) / 64u ||
          !ps5_streamout_storage(context, &context->streamout_records,
                                 64u + record_capacity * 64u))
         return false;
      struct ps5_resource *records = (struct ps5_resource *)context->streamout_records;
      control = (struct ps5_streamout_control *)records->data;
      memset(control, 0, 64u + record_capacity * 64u);
   }
   memset(descriptors, 0,
          (global_control ? PS5_STREAMOUT_CONTROL_DESCRIPTOR + 1u : 4u) *
             16u);
   if (global_control)
      memset(control, 0, sizeof(*control));
   if (global_control)
      control->reserved[1] = record_capacity;
   memset(size_dwords, 0, 4u * sizeof(size_dwords[0]));
   memset(stride_dwords, 0, 4u * sizeof(stride_dwords[0]));
   memset(offset_dwords, 0, 4u * sizeof(offset_dwords[0]));

   for (unsigned index = 0; index < PIPE_MAX_SO_BUFFERS; ++index) {
      struct pipe_stream_output_target *pipe_target;
      struct ps5_stream_output_target *target;
      struct ps5_resource *resource;
      uintptr_t address;
      uint64_t end;
      uint64_t begin;
      uint64_t capacity;

      if (!(mask & BITFIELD_BIT(index)))
         continue;
      if (index >= context->stream_output_target_count ||
          !(pipe_target = context->stream_output_targets[index]))
         return false;
      target = (struct ps5_stream_output_target *)pipe_target;
      resource = (struct ps5_resource *)pipe_target->buffer;
      end = (uint64_t)pipe_target->buffer_offset +
            pipe_target->buffer_size;
      begin = (uint64_t)pipe_target->buffer_offset + target->offset;
      if (!metadata->streamout_strides_dwords[index] ||
          resource->base.target != PIPE_BUFFER || end > resource->size ||
          begin > end || end > UINT32_MAX || (uintptr_t)resource->data >> 48)
         return false;

      if ((begin & 3u) || (end & 3u))
         return false;
      address = (uintptr_t)resource->data + begin;
      if (record_capacity) {
         uint64_t staging_size = staging_primitives * vertices_per_primitive;
         if (staging_size > UINT32_MAX / (4u * metadata->streamout_strides_dwords[index]) ||
             !ps5_streamout_storage(context, &context->streamout_staging[index],
                staging_size * 4u * metadata->streamout_strides_dwords[index]))
            return false;
         struct ps5_resource *staging = (struct ps5_resource *)context->streamout_staging[index];
         address = (uintptr_t)staging->data;
         end = staging_size * 4u * metadata->streamout_strides_dwords[index];
         begin = 0;
      }
      if (address >> 48)
         return false;
      descriptors[index * 4u] = (uint32_t)address;
      descriptors[index * 4u + 1u] = (uint32_t)(address >> 32);
      /* The NGG shader uses byte offsets and byte-sized buffer descriptors.
       * Point the descriptor at the current target position so PrimitiveID
       * starts every draw at offset zero without a global GDS counter. */
      descriptors[index * 4u + 2u] = (uint32_t)(end - begin);
      descriptors[index * 4u + 3u] = UINT32_C(0x31016fac);
      size_dwords[index] = (uint32_t)((end - begin) >> 2);
      stride_dwords[index] = metadata->streamout_strides_dwords[index];
      offset_dwords[index] = 0;
      if (!global_control) {
         capacity = (end - begin) /
                    (4u * vertices_per_primitive * stride_dwords[index]);
         primitives = MIN2(primitives, capacity);
      }
      ps5_flush_gpu_data(resource->data, resource->size);
   }
   if (global_control) {
      uintptr_t control_address = (uintptr_t)control;
      uint32_t *control_descriptor =
         descriptors + PS5_STREAMOUT_CONTROL_DESCRIPTOR * 4u;

      if (control_address >> 48 ||
          (uint32_t)(control_address >> 32) != metadata->address32_hi)
         return false;
      control_descriptor[0] = (uint32_t)control_address;
      control_descriptor[1] = (uint32_t)(control_address >> 32);
      control_descriptor[2] = record_capacity ? 64u + record_capacity * 64u : sizeof(*control);
      control_descriptor[3] = UINT32_C(0x31016fac);
      if (record_capacity)
         ps5_flush_gpu_data(control, control_descriptor[2]);
   }
   user_data[metadata->streamout_buffer_table_user_data_dword] =
      (uint32_t)table_address;
   ps5_flush_gpu_data(
      descriptors,
      global_control
         ? PS5_STREAMOUT_CONTROL_OFFSET + PS5_STREAMOUT_CONTROL_BYTES -
              PS5_STREAMOUT_DESCRIPTOR_OFFSET
         : 4u * 16u);
   *enabled_mask = mask;
   *written_vertices = primitives * vertices_per_primitive;
   return true;
}

static bool
ps5_collect_geometry_streamout(
   struct ps5_context *context, struct ps5_resource *descriptor_resource,
   const PsbcShaderMetadata *metadata,
   uint64_t written_vertices[PIPE_MAX_VERTEX_STREAMS],
   uint64_t generated_primitives[PIPE_MAX_VERTEX_STREAMS])
{
   struct ps5_streamout_control *control =
      (struct ps5_streamout_control *)(descriptor_resource->data +
                                       PS5_STREAMOUT_CONTROL_OFFSET);
   unsigned vertices_per_primitive =
      ps5_streamout_vertices_per_primitive(context->stream_output_primitive);

   if (context->gs || context->tes) {
      struct ps5_resource *storage = (struct ps5_resource *)context->streamout_records;
      if (!storage || storage->size < sizeof(*control))
         return false;
      control = (struct ps5_streamout_control *)storage->data;
      ps5_flush_gpu_data(storage->data, storage->size);
      const unsigned count = control->reserved[0];
      if (!vertices_per_primitive || control->reserved[2] ||
          count > control->reserved[1] ||
          count > (storage->size - sizeof(*control)) / sizeof(struct ps5_streamout_record)) {
         printf("[ps5-gallium] streamout-record-header count=%u capacity=%u error=%u\n",
                count, control->reserved[1], control->reserved[2]);
         return false;
      }
      struct ps5_streamout_record *records = (void *)(control + 1);
      if (context->tes) {
         /* Raw TES: (ordinal, patch); raw TES+GS: (GS primitive, instance, ordinal). */
         for (unsigned i = 0; i < count; ++i) {
            const uint32_t ordinal = context->gs ? records[i].reserved[0] : records[i].primitive;
            records[i].reserved[1] = context->gs ? records[i].primitive : records[i].invocation;
            if (!context->gs)
               records[i].invocation = 0;
            records[i].primitive = ordinal;
            records[i].reserved[0] = 0;
         }
      }
      qsort(records, count, sizeof(*records), ps5_streamout_record_compare);
      if (context->tes && count) {
         /* ponytail: fewer than one cycle identifies one complete ordinal
          * interval; larger draws need a non-wrapping ordering mechanism. */
         const unsigned modulus = PS5_TESS_STREAMOUT_ORDINAL_COUNT;
         if (count >= modulus)
            return false;
         unsigned origin = 0, boundaries = 0;
         for (unsigned i = 0; i < count; ++i) {
            unsigned next = (i + 1) % count;
            if (records[i].primitive >= modulus)
               return false;
            unsigned gap = next ? records[next].primitive - records[i].primitive :
               modulus + records[0].primitive - records[i].primitive;
            if (gap == modulus + 1u - count) {
               origin = records[next].primitive;
               ++boundaries;
            } else if (gap != 1u) {
               printf("[ps5-gallium] tess-streamout-gap count=%u index=%u key=%u next=%u gap=%u first=%u last=%u\n",
                      count, i, records[i].primitive, records[next].primitive,
                      gap, records[0].primitive, records[count - 1].primitive);
               return false;
            }
         }
         if (boundaries != 1)
            return false;
         for (unsigned i = 0; i < count; ++i)
            records[i].primitive = (records[i].primitive + modulus - origin) % modulus;
         qsort(records, count, sizeof(*records), ps5_tess_streamout_record_compare);
      }
      uint64_t capacity[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
      uint64_t generated[4] = {0}, emitted[4] = {0};
      const unsigned mask = ps5_streamout_buffer_mask(metadata->streamout_enabled_stream_buffers_mask);
      const struct pipe_stream_output_info *so = context->gs ?
         &context->gs->stream_output : &context->tes->stream_output;
      for (unsigned i = 0; i < so->num_outputs; ++i) {
         const struct pipe_stream_output *output = &so->output[i];
         if (output->output_buffer >= 4 ||
             output->dst_offset + output->num_components >
                metadata->streamout_strides_dwords[output->output_buffer])
            return false;
      }
      for (unsigned buffer = 0; buffer < 4; ++buffer) {
         if (!(mask & BITFIELD_BIT(buffer)))
            continue;
         struct ps5_stream_output_target *target =
            (void *)context->stream_output_targets[buffer];
         struct ps5_resource *staging = (void *)context->streamout_staging[buffer];
         unsigned stream = ps5_streamout_buffer_stream(
            metadata->streamout_enabled_stream_buffers_mask, buffer);
         uint64_t stride = (uint64_t)metadata->streamout_strides_dwords[buffer] *
                           4u * vertices_per_primitive;
         if (!target || !staging || !stride || stream >= 4 ||
             target->offset > target->base.buffer_size)
            return false;
         struct ps5_resource *destination = (void *)target->base.buffer;
         if (!destination || target->base.buffer_offset > destination->size ||
             target->base.buffer_size > destination->size - target->base.buffer_offset)
            return false;
         capacity[stream] = MIN2(capacity[stream],
            (target->base.buffer_size - target->offset) / stride);
         ps5_flush_gpu_data(staging->data, staging->size);
      }
      /* Validate the complete receipt before changing any application buffer. */
      for (unsigned i = 0; i < count; ++i) {
         /* TES ordinals were proven contiguous above; patch keys set the order. */
         if (context->tes && (count >= PS5_TESS_STREAMOUT_ORDINAL_COUNT ||
                              records[i].primitive >= count)) {
            printf("[ps5-gallium] tess-streamout-order-rejected count=%u index=%u key=%u invocation=%u first=%u last=%u\n",
                   count, i, records[i].primitive, records[i].invocation,
                   records[0].primitive, records[count - 1].primitive);
            return false;
         }
         if (i && !(context->tes ? ps5_tess_streamout_record_compare :
                                   ps5_streamout_record_compare)(&records[i - 1], &records[i]))
            return false;
         for (unsigned stream = 0; stream < 4; ++stream) {
            if (records[i].emitted[stream] != records[i].generated[stream])
               return false;
            generated[stream] += records[i].generated[stream];
         }
         for (unsigned buffer = 0; buffer < 4; ++buffer) {
            if (!(mask & BITFIELD_BIT(buffer)))
               continue;
            unsigned stream = ps5_streamout_buffer_stream(
               metadata->streamout_enabled_stream_buffers_mask, buffer);
            struct ps5_resource *staging = (void *)context->streamout_staging[buffer];
            uint64_t bytes = (uint64_t)records[i].generated[stream] *
               vertices_per_primitive * metadata->streamout_strides_dwords[buffer] * 4u;
            if (records[i].offsets[buffer] > staging->size ||
                bytes > staging->size - records[i].offsets[buffer])
               return false;
         }
      }
      for (unsigned stream = 0; stream < 4; ++stream)
         if (generated[stream] != control->generated_primitives[stream])
            return false;
      for (unsigned i = 0; i < count; ++i) {
         uint64_t take[4];
         for (unsigned stream = 0; stream < 4; ++stream)
            take[stream] = MIN2((uint64_t)records[i].generated[stream],
                                capacity[stream] - emitted[stream]);
         for (unsigned buffer = 0; buffer < 4; ++buffer) {
            if (!(mask & BITFIELD_BIT(buffer)))
               continue;
            unsigned stream = ps5_streamout_buffer_stream(
               metadata->streamout_enabled_stream_buffers_mask, buffer);
            struct ps5_stream_output_target *target = (void *)context->stream_output_targets[buffer];
            struct ps5_resource *dst = (void *)target->base.buffer;
            struct ps5_resource *src = (void *)context->streamout_staging[buffer];
            uint64_t stride = (uint64_t)vertices_per_primitive *
                              metadata->streamout_strides_dwords[buffer] * 4u;
            uint8_t *destination = dst->data + target->base.buffer_offset +
               target->offset + emitted[stream] * stride;
            const uint8_t *source = src->data + records[i].offsets[buffer];
            /* Preserve skipped components and stride padding in the app buffer. */
            for (unsigned o = 0; o < so->num_outputs; ++o) {
               const struct pipe_stream_output *output = &so->output[o];
               if (output->output_buffer != buffer)
                  continue;
               for (uint64_t v = 0; v < take[stream] * vertices_per_primitive; ++v) {
                  uint64_t offset = (v * metadata->streamout_strides_dwords[buffer] +
                                     output->dst_offset) * 4u;
                  memcpy(destination + offset, source + offset, output->num_components * 4u);
               }
            }
         }
         for (unsigned stream = 0; stream < 4; ++stream)
            emitted[stream] += take[stream];
      }
      for (unsigned buffer = 0; buffer < 4; ++buffer) {
         if (mask & BITFIELD_BIT(buffer)) {
            struct ps5_resource *dst = (void *)context->stream_output_targets[buffer]->buffer;
            ps5_flush_gpu_data(dst->data, dst->size);
         }
      }
      for (unsigned stream = 0; stream < 4; ++stream) {
         written_vertices[stream] = emitted[stream] * vertices_per_primitive;
         generated_primitives[stream] = generated[stream];
      }
      printf("[ps5-gallium] geometry-streamout ordered groups=%u primitives=%llu emitted=%llu\n",
             count, (unsigned long long)generated[0], (unsigned long long)emitted[0]);
      return true;
   }

   if (!vertices_per_primitive ||
       PS5_STREAMOUT_CONTROL_OFFSET + sizeof(*control) >
          descriptor_resource->size)
      return false;
   ps5_flush_gpu_data(control, sizeof(*control));
   for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS; ++stream) {
      if (control->emitted_primitives[stream] >
          control->generated_primitives[stream])
         return false;
      written_vertices[stream] =
         (uint64_t)control->emitted_primitives[stream] *
         vertices_per_primitive;
      generated_primitives[stream] = control->generated_primitives[stream];
   }
   printf("[ps5-gallium] geometry-streamout generated=%u/%u/%u/%u emitted=%u/%u/%u/%u offsets=%u/%u/%u/%u\n",
          control->generated_primitives[0], control->generated_primitives[1],
          control->generated_primitives[2], control->generated_primitives[3],
          control->emitted_primitives[0], control->emitted_primitives[1],
          control->emitted_primitives[2], control->emitted_primitives[3],
          control->buffer_offsets[0],
          control->buffer_offsets[1], control->buffer_offsets[2],
          control->buffer_offsets[3]);
   return true;
}

static bool
ps5_vertex_buffer_descriptor(uintptr_t address, size_t available,
                             unsigned stride, uint32_t records,
                             uint32_t descriptor[4])
{
   if (!available || (!stride && available > UINT32_MAX))
      return false;
   descriptor[0] = (uint32_t)address;
   descriptor[1] = (uint32_t)(address >> 32) | (stride << 16);
   /* GFX10 zero-stride inputs use byte bounds, not a one-vertex limit.
    * CPU per-attribute range checks still precede descriptor construction. */
   descriptor[2] = stride ? records : (uint32_t)available;
   descriptor[3] = UINT32_C(0x5204) |
      (stride ? 0 : S_008F0C_OOB_SELECT(V_008F0C_OOB_SELECT_RAW));
   return true;
}

/* End bounds are validated separately; publish only the fetched vertex range. */
static uint64_t
ps5_vertex_fetch_begin(unsigned offset, unsigned stride, unsigned divisor,
                       unsigned first_vertex, unsigned first_instance,
                       unsigned split_instance)
{
   const uint64_t first = divisor
      ? (uint64_t)first_instance + split_instance / divisor : first_vertex;
   return offset + first * stride;
}

static void
ps5_draw_vbo_locked(struct pipe_context *base,
                    const struct pipe_draw_info *info,
                    unsigned drawid_offset,
                    const struct pipe_draw_indirect_info *indirect,
                    const struct pipe_draw_start_count_bias *draws,
                    unsigned num_draws,
                    struct ps5_batch_flush_cache *flush_cache,
                    bool *submitted)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {0, ps5_prepare_clock()};
#endif

   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_draw_start_count_bias draw;
   const PsbcShaderOutput *vertex_output;
   const PsbcShaderMetadata *vertex_metadata;
   const PsbcShaderMetadata *input_metadata;
   const uint8_t *vertex_package;
   size_t vertex_package_size;
   struct ps5_resource *descriptor_resource;
   struct ps5_resource *pixel_descriptor_resource;
   uintptr_t descriptor_address;
   uint32_t *descriptor;
   uint32_t user_data[32] = {0};
   uint32_t hull_user_data[32] = {0};
   uint32_t *input_user_data;
   uint32_t pixel_user_data[32] = {0};
   size_t descriptor_bytes[2] = {0};
   unsigned point_coord_input = 0;
   struct ps5_vertex_layout vertex_layout;
   struct ps5_vertex_layout fragment_layout = {0};
   struct ps5_fragment_exports fragment_exports;
   struct ps5_native_graphics_state graphics;
   unsigned user_data_count;
   unsigned input_user_data_count;
   unsigned pixel_user_data_count;
   uint32_t primitive_type;
   uint32_t fragment_primitive_type;
   struct ps5_resource *index_resource = NULL;
   size_t index_offset = 0;
   uint32_t base_vertex;
   unsigned vertex_count, first_vertex;
   bool streamout_active = false;
   bool primitive_query_active = false;
   uint32_t streamout_mask = 0;
   uint32_t streamout_size[4] = {0};
   uint32_t streamout_stride[4] = {0};
   uint32_t streamout_offset[4] = {0};
   uint64_t streamout_written_vertices[PIPE_MAX_VERTEX_STREAMS] = {0};
   uint64_t generated_primitives[PIPE_MAX_VERTEX_STREAMS] = {0};
   struct ps5_query *occlusion_query = context->queries_enabled
                                         ? context->active_occlusion_query
                                         : NULL;
   bool color_to_texture_barrier = false;
   bool depth_to_texture_barrier = false;
   unsigned sample_count = 1;
   unsigned color_target_count;
   bool provoking_vtx_last;
   bool poly_line_smooth;
   bool tessellation_active;
   bool draw_id_used;
   uint32_t vs_out_control = 0;
   uint32_t vs_out_control_valid = 0;

   if (submitted)
      *submitted = false;

   if (!info || !draws || !num_draws) {
      context->last_draw_status = num_draws ? -2 : 0;
      return;
   }
   if (num_draws > 1) {
      util_draw_multi(base, info, drawid_offset, indirect, draws, num_draws);
      return;
   }
   draw = draws[0];
   if (!indirect && info->mode == MESA_PRIM_PATCHES) {
      if (!context->patch_vertices) {
         context->last_draw_status = -2;
         return;
      }
      draw.count -= draw.count % context->patch_vertices;
   } else if (!indirect && !u_trim_pipe_prim(info->mode, &draw.count)) {
      context->last_draw_status = 0;
      return;
   }
   if (!draw.count) {
      context->last_draw_status = 0;
      return;
   }
   draws = &draw;
   if (!indirect && context->stream_output_target_count &&
       info->instance_count > 1) {
      uint64_t last_instance = (uint64_t)info->start_instance +
                               info->instance_count - 1u;

      if (last_instance > UINT32_MAX) {
         context->last_draw_status = -2;
         return;
      }
      printf("[ps5-gallium] streamout-instanced-split count=%u start=%u\n",
             info->instance_count, info->start_instance);
      for (unsigned instance = 0; instance < info->instance_count; ++instance) {
         struct pipe_draw_info single = *info;
         const unsigned saved_instance = context->split_instance_id;

         single.instance_count = 1;
         if (!context->gs && !context->tcs && !context->tes)
            context->split_instance_id = instance;
         else
            single.start_instance = info->start_instance + instance;
         ps5_draw_vbo_locked(base, &single, drawid_offset, NULL, draws, 1,
                             NULL, submitted);
         context->split_instance_id = saved_instance;
         if (context->last_draw_status != 0)
            return;
      }
      return;
   }
   context->draw_calls++;
#ifdef PS5_DRAW_PROFILE
   if (context->draw_calls % 10000u == 0)
      ps5_context_profile_report(context);
#endif
   if (!ps5_render_condition_passes(context)) {
      context->last_draw_status = 0;
      return;
   }
   if (indirect || num_draws != 1 ||
       (info->index_size && info->index_size != 2 && info->index_size != 4) ||
       info->primitive_restart || info->has_user_indices ||
       !info->instance_count ||
       !ps5_draw_primitive(info->mode, draws[0].count, &primitive_type)) {
      context->last_draw_status = -2;
      return;
   }
   generated_primitives[0] = (primitive_type == 9
      ? draws[0].count / context->patch_vertices
      : ps5_draw_primitive_count(primitive_type, draws[0].count)) *
      info->instance_count;
   base_vertex = info->index_size ? (uint32_t)draws[0].index_bias
                                  : draws[0].start;
   if (!info->index_size) {
      uint64_t end = (uint64_t)draws[0].start + draws[0].count;

      if (end > UINT32_MAX) {
         context->last_draw_status = -12;
         return;
      }
      vertex_count = (unsigned)end;
      first_vertex = draws[0].start;
#ifdef AGC_RUNTIME_DIAGNOSTICS
      if (draws[0].start)
         printf("[ps5-gallium] first-vertex first=%u count=%u descriptor-count=%u\n",
                draws[0].start, draws[0].count, vertex_count);
#endif
   }
   if (info->index_size) {
      const void *indices;
      uint64_t offset = (uint64_t)draws[0].start * info->index_size;
      uint64_t bytes = (uint64_t)draws[0].count * info->index_size;
      unsigned min_index = UINT32_MAX;
      unsigned max_index = 0;
      unsigned min_effective = UINT32_MAX;
      unsigned max_effective = 0;

      index_resource = (struct ps5_resource *)info->index.resource;
      if (!index_resource || index_resource->base.target != PIPE_BUFFER ||
          offset > index_resource->size ||
          bytes > index_resource->size - offset) {
         context->last_draw_status = -12;
         return;
      }
      index_offset = (size_t)offset;
      indices = (const uint8_t *)index_resource->data + index_offset;
      if (info->was_line_loop) {
         printf("[ps5-gallium] lowered-line-loop mode=%u count=%u start=%u offset=%zu base=%p selected=%p indices=",
                info->mode, draws[0].count, draws[0].start, index_offset,
                index_resource->data, indices);
         ps5_log_indices(indices, info->index_size, draws[0].count, 8);
      }
      if (info->was_primitive_restart) {
         printf("[ps5-gallium] lowered-primitive-restart mode=%u count=%u start=%u offset=%zu base=%p selected=%p indices=",
                info->mode, draws[0].count, draws[0].start, index_offset,
                index_resource->data, indices);
         ps5_log_indices(indices, info->index_size, draws[0].count, 16);
      }
      if (info->was_index_ubyte) {
         printf("[ps5-gallium] lowered-index-ubyte mode=%u index-size=%u count=%u start=%u offset=%zu base=%p selected=%p indices=",
                info->mode, info->index_size, draws[0].count, draws[0].start,
                index_offset, index_resource->data, indices);
         ps5_log_indices(indices, info->index_size, draws[0].count, 16);
      }
#ifdef AGC_RUNTIME_DIAGNOSTICS
      if (info->index_size == 2 && !info->was_index_ubyte) {
         printf("[ps5-gallium] native-index-u16 mode=%u index-size=%u count=%u start=%u offset=%zu base=%p selected=%p indices=",
                info->mode, info->index_size, draws[0].count, draws[0].start,
                index_offset, index_resource->data, indices);
         ps5_log_indices(indices, info->index_size, draws[0].count, 16);
      }
      if (info->index_size == 4) {
         printf("[ps5-gallium] native-index-u32 mode=%u index-size=%u count=%u start=%u offset=%zu base=%p selected=%p indices=",
                info->mode, info->index_size, draws[0].count, draws[0].start,
                index_offset, index_resource->data, indices);
         ps5_log_indices(indices, info->index_size, draws[0].count, 16);
      }
#endif
      if (!ps5_index_bounds(indices, info->index_size, draws[0].count,
                            draws[0].index_bias, &min_index, &max_index,
                            &min_effective, &max_effective)) {
         context->last_draw_status = -12;
         return;
      }
      vertex_count = max_effective + 1u;
      first_vertex = min_effective;
#ifdef AGC_RUNTIME_DIAGNOSTICS
      if (draws[0].index_bias)
         printf("[ps5-gallium] base-vertex bias=%d raw-min=%u raw-max=%u effective-min=%u effective-max=%u\n",
                draws[0].index_bias, min_index, max_index,
                min_effective, max_effective);
#endif
   }
   tessellation_active = context->tcs || context->tes;
   if (!context->vs || !context->fs || !context->framebuffer_valid ||
       (tessellation_active && (!context->tcs || !context->tes ||
                                info->mode != MESA_PRIM_PATCHES)) ||
       (!tessellation_active && info->mode == MESA_PRIM_PATCHES)) {
      context->last_draw_status = -3;
      return;
   }
   for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i) {
      const struct pipe_resource *target =
         context->framebuffer.cbufs[i].texture;

      if (target) {
         sample_count = MAX2(target->nr_samples, 1);
         break;
      }
   }
   if (!context->framebuffer.nr_cbufs &&
       context->framebuffer.zsbuf.texture)
      sample_count = MAX2(context->framebuffer.zsbuf.texture->nr_samples, 1);
   /* AGC submission still needs a valid physical color target when the API
    * framebuffer is depth-only.  Keep writes disabled with target_mask=0. */
   color_target_count = MAX2(context->framebuffer.nr_cbufs, 1);
   if (!ps5_encode_graphics_state(context, &graphics)) {
      context->last_draw_status = -18;
      return;
   }
   descriptor_resource = (struct ps5_resource *)
      context->vertex_descriptor_table;
   if (!descriptor_resource ||
       !ps5_vertex_layout_from_state(context->vs,
                                     context->vertex_elements,
                                     &vertex_layout)) {
      context->last_draw_status = -14;
      return;
   }
   if (tessellation_active) {
      if (!ps5_select_tessellation_pipeline(
             context,
             (uint32_t)((uintptr_t)descriptor_resource->data >> 32),
             &vertex_layout)) {
         context->last_draw_status = -19;
         return;
      }
   } else if (!ps5_select_shader_variant(
                 context->vs,
                 (uint32_t)((uintptr_t)descriptor_resource->data >> 32),
                  &vertex_layout, primitive_type,
                  context->rasterizer &&
                     context->rasterizer->flatshade_first,
                 false, false,
                 !context->gs &&
                    !(context->fs->nir->info.inputs_read &
                      VARYING_BIT_PRIMITIVE_ID) &&
                    !BITSET_TEST(context->fs->nir->info.system_values_read,
                                 SYSTEM_VALUE_PRIMITIVE_ID),
                 false, -1, NULL) ||
              !ps5_select_geometry_pipeline(
                 context,
                 (uint32_t)((uintptr_t)descriptor_resource->data >> 32),
                 &vertex_layout, primitive_type)) {
      context->last_draw_status = -19;
      return;
   }
   pixel_descriptor_resource =
      (struct ps5_resource *)context->descriptor_storage[1];
   fragment_primitive_type =
      ps5_fragment_primitive_type(context, primitive_type);
   provoking_vtx_last = context->rasterizer &&
                         context->rasterizer->flatshade_first;
   poly_line_smooth = sample_count == 1 && context->rasterizer &&
      (((fragment_primitive_type == 2 || fragment_primitive_type == 3) &&
        context->rasterizer->line_smooth) ||
       ((fragment_primitive_type >= 4 && fragment_primitive_type <= 6) &&
        context->rasterizer->poly_smooth));
   fragment_exports = ps5_fragment_exports_for_framebuffer(&context->framebuffer);
   fragment_exports.ignore_sample_mask |= !context->rasterizer || !context->rasterizer->multisample;
   if (fragment_exports.ignore_sample_mask)
      fragment_exports.rasterization_samples = 0;
   /* glMinSampleShading: invocations per pixel, a power of two up to the
    * sample count. A shader reading sample inputs always runs per sample. */
   fragment_exports.ps_iter_samples = 1;
   if (fragment_exports.rasterization_samples > 1) {
      const unsigned samples = fragment_exports.rasterization_samples;
      fragment_exports.ps_iter_samples =
         context->fs->nir->info.fs.uses_sample_shading ? samples :
         MIN2(util_next_power_of_two(MAX2(context->min_samples, 1u)), samples);
   }
   if (!pixel_descriptor_resource || !fragment_primitive_type ||
       !ps5_select_shader_variant(
          context->fs,
          (uint32_t)((uintptr_t)pixel_descriptor_resource->data >> 32),
          &fragment_layout, fragment_primitive_type,
          provoking_vtx_last,
          sample_count > 1 && context->rasterizer &&
             context->rasterizer->multisample && graphics.alpha_to_one,
          /* Both explicit GS IDs and the compiler's implicit VS IDs use
           * per-vertex transport on PS5. */
          poly_line_smooth, false, false,
          context->legacy_primitive_conversion ?
             context->legacy_flat_input_vertex : -1,
          &fragment_exports)) {
      context->last_draw_status = -14;
      return;
   }
   streamout_active = PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE &&
                      context->stream_output_target_count != 0;
   if (tessellation_active) {
      vertex_output = &context->tessellation_output.tes;
      vertex_package = context->tessellation_tes_package;
      vertex_package_size = context->tessellation_tes_package_size;
   } else if (context->gs) {
      if (streamout_active) {
         if (!context->gs->stream_output.num_outputs ||
             !context->geometry_streamout_package) {
            context->last_draw_status = -21;
            return;
         }
         vertex_output = &context->geometry_streamout_output;
         vertex_package = context->geometry_streamout_package;
         vertex_package_size = context->geometry_streamout_package_size;
      } else {
         vertex_output = &context->geometry_output;
         vertex_package = context->geometry_package;
         vertex_package_size = context->geometry_package_size;
      }
   } else {
      if (streamout_active) {
         if (!context->vs->stream_output.num_outputs ||
             !context->vs->active->streamout_package) {
            context->last_draw_status = -21;
            return;
         }
         vertex_output = &context->vs->active->streamout_output;
         vertex_package = context->vs->active->streamout_package;
         vertex_package_size = context->vs->active->streamout_package_size;
      } else {
         vertex_output = &context->vs->active->output;
         vertex_package = context->vs->active->package;
         vertex_package_size = context->vs->active->package_size;
      }
   }
   user_data_count = vertex_output->metadata.user_sgpr_count;
   input_metadata = tessellation_active
      ? &context->tessellation_output.hs.metadata
      : &vertex_output->metadata;
   input_user_data = tessellation_active ? hull_user_data : user_data;
   input_user_data_count = input_metadata->user_sgpr_count;
   pixel_user_data_count =
      context->fs->active->output.metadata.user_sgpr_count;
   if (user_data_count > 32 || input_user_data_count > 32 ||
       pixel_user_data_count > 32) {
      context->last_draw_status = -11;
      return;
   }
   if (tessellation_active &&
       context->tessellation_output.runtime.final_offchip_layout_valid) {
      unsigned dword = context->tessellation_output.runtime
                          .final_offchip_layout_user_data_dword;
      if (dword >= user_data_count) {
         context->last_draw_status = -10;
         return;
      }
      user_data[dword] = context->tessellation_output.runtime
                            .final_offchip_layout;
   }
   vertex_metadata = &vertex_output->metadata;
   if (vertex_metadata->hardware_stage == PSBC_HW_STAGE_NGG) {
      /* RADV's NGG ABI places GS outputs after the ES input ring in LDS.
       * The compiler legitimately returns zero when the GS reads no ES inputs. */
      if (!vertex_metadata->ngg_lds_layout_valid ||
          vertex_metadata->ngg_lds_layout_user_data_dword >= user_data_count ||
          vertex_metadata->ngg_lds_layout > UINT16_MAX) {
         context->last_draw_status = -10;
         return;
      }
      user_data[vertex_metadata->ngg_lds_layout_user_data_dword] =
         vertex_metadata->ngg_lds_layout;
   }
   if (vertex_metadata->clip_distance_mask ||
       vertex_metadata->cull_distance_mask) {
      uint32_t base_control = 0;
      unsigned packed_clip = 0;
      unsigned packed_index = 0;
      unsigned packed_cull;
      bool found = false;

      for (uint32_t i = 0; i < vertex_metadata->context_register_count; ++i) {
         if (vertex_metadata->context_registers[i].offset == UINT16_C(0x0207)) {
            base_control = vertex_metadata->context_registers[i].value;
            found = true;
            break;
         }
      }
      if (!found) {
         context->last_draw_status = -18;
         return;
      }
      for (unsigned distance = 0; distance < 8; ++distance) {
         if (!(vertex_metadata->clip_distance_mask & (1u << distance)))
            continue;
         if (context->rasterizer &&
             (context->rasterizer->clip_plane_enable & (1u << distance)))
            packed_clip |= 1u << packed_index;
         ++packed_index;
      }
      packed_cull = ((1u <<
         util_bitcount(vertex_metadata->cull_distance_mask)) - 1u) <<
         packed_index;
      if (fragment_primitive_type == 1)
         packed_cull |= packed_clip;
      vs_out_control = (base_control & UINT32_C(0xffff0000)) |
                       packed_clip | (packed_cull << 8);
      vs_out_control_valid = 1;
   }
   if (!input_metadata->base_vertex_valid ||
       input_metadata->base_vertex_user_data_dword >= input_user_data_count) {
      context->last_draw_status = -10;
      return;
   }
   input_user_data[input_metadata->base_vertex_user_data_dword] = base_vertex;
   if (!input_metadata->is_indexed_draw_valid ||
       input_metadata->is_indexed_draw_user_data_dword >=
          input_user_data_count) {
      context->last_draw_status = -10;
      return;
   }
   input_user_data[input_metadata->is_indexed_draw_user_data_dword] =
      info->index_size ? UINT32_MAX : 0;
   draw_id_used =
      (context->vs && BITSET_TEST(context->vs->nir->info.system_values_read,
                                  SYSTEM_VALUE_DRAW_ID)) ||
      (context->tcs && BITSET_TEST(context->tcs->nir->info.system_values_read,
                                   SYSTEM_VALUE_DRAW_ID)) ||
      (context->tes && BITSET_TEST(context->tes->nir->info.system_values_read,
                                   SYSTEM_VALUE_DRAW_ID)) ||
      (context->gs && BITSET_TEST(context->gs->nir->info.system_values_read,
                                  SYSTEM_VALUE_DRAW_ID));
   if (draw_id_used) {
      unsigned draw_id_dword = input_metadata->base_vertex_user_data_dword + 1u;

      if (draw_id_dword >= input_user_data_count) {
         context->last_draw_status = -10;
         return;
      }
      input_user_data[draw_id_dword] = drawid_offset;
   }
   /* An unused base instance has no compiler argument, including when XFB
    * splits an instanced draw into individual submissions. */
   if (input_metadata->start_instance_valid) {
      if (input_metadata->start_instance_user_data_dword >=
          input_user_data_count) {
         context->last_draw_status = -10;
         return;
      }
      input_user_data[input_metadata->start_instance_user_data_dword] = info->start_instance;
   }
   if (input_metadata->instance_id_bias_valid) {
      if (input_metadata->instance_id_bias_user_data_dword >= input_user_data_count) {
         context->last_draw_status = -10;
         return;
      }
      input_user_data[input_metadata->instance_id_bias_user_data_dword] =
         context->split_instance_id;
   } else if (context->split_instance_id) {
      context->last_draw_status = -10;
      return;
   }
   if (input_metadata->vertex_buffer_table_valid) {
      uint32_t binding_mask = 0;
      uint32_t binding_records[PIPE_MAX_ATTRIBS] = {0};
      size_t binding_bytes[PIPE_MAX_ATTRIBS] = {0};
      size_t binding_begin[PIPE_MAX_ATTRIBS];
      memset(binding_begin, 0xff, sizeof(binding_begin));
      unsigned descriptor_index = 0;
      unsigned element_index;
      unsigned binding;

      if (!context->vertex_elements || !context->vertex_elements->count ||
          !context->vertex_buffer_count) {
         context->last_draw_status = -7;
         return;
      }
      for (element_index = 0;
           element_index < vertex_layout.count;
           ++element_index) {
         const struct pipe_vertex_element *element =
            &context->vertex_elements->elements[element_index];
         const struct pipe_vertex_buffer *vertex_buffer;
         struct ps5_resource *vertex_resource;
         PsbcVertexFormat ignored_format;
         unsigned format_size = ps5_vertex_format_size(element->src_format);
         uint64_t records = element->src_stride ? vertex_count : 1u;
         uint64_t descriptor_records = records;
         uint64_t required;

         if (element->vertex_buffer_index >= context->vertex_buffer_count ||
             !format_size ||
             !ps5_vertex_format(element->src_format, &ignored_format)) {
            context->last_draw_status = -8;
            return;
         }
         vertex_buffer =
            &context->vertex_buffers[element->vertex_buffer_index];
         if (vertex_buffer->is_user_buffer ||
             !vertex_buffer->buffer.resource) {
            context->last_draw_status = -8;
            return;
         }
         if (element->instance_divisor) {
            records = (uint64_t)info->start_instance +
                      ((uint64_t)context->split_instance_id + info->instance_count - 1u) /
                         element->instance_divisor +
                      1u;
            if (records > UINT32_MAX) {
               context->last_draw_status = -9;
               return;
            }
         }
         vertex_resource =
            (struct ps5_resource *)vertex_buffer->buffer.resource;
         required = (uint64_t)vertex_buffer->buffer_offset +
                    element->src_offset +
                    (records - 1u) * element->src_stride +
                    format_size;
         if (element->src_stride)
            descriptor_records =
               (element->src_offset + (records - 1u) * element->src_stride +
                format_size + element->src_stride - 1u) /
               element->src_stride;
         if (vertex_resource->base.target != PIPE_BUFFER ||
             required > vertex_resource->size ||
             descriptor_records > UINT32_MAX) {
            context->last_draw_status = -9;
            return;
         }
         binding_records[element->vertex_buffer_index] =
            MAX2(binding_records[element->vertex_buffer_index],
                 (uint32_t)descriptor_records);
         binding_bytes[element->vertex_buffer_index] =
            MAX2(binding_bytes[element->vertex_buffer_index],
                 (size_t)(required - vertex_buffer->buffer_offset));
         binding_begin[element->vertex_buffer_index] =
            MIN2(binding_begin[element->vertex_buffer_index],
                 ps5_vertex_fetch_begin(element->src_offset, element->src_stride,
                    element->instance_divisor, first_vertex, info->start_instance,
                    context->split_instance_id));
         binding_mask |= BITFIELD_BIT(element->vertex_buffer_index);
      }
      descriptor_address = (uintptr_t)descriptor_resource->data;
      if ((uint32_t)(descriptor_address >> 32) !=
             input_metadata->address32_hi ||
          input_metadata->vertex_buffer_table_user_data_dword >=
             input_user_data_count) {
         context->last_draw_status = -10;
         return;
      }
      if ((size_t)__builtin_popcount(binding_mask) * 16 >
          descriptor_resource->size) {
         context->last_draw_status = -9;
         return;
      }
      descriptor = (uint32_t *)descriptor_resource->data;
      for (binding = 0; binding < context->vertex_buffer_count; ++binding) {
         const struct pipe_vertex_element *element = NULL;
         const struct pipe_vertex_buffer *vertex_buffer;
         struct ps5_resource *vertex_resource;
         uintptr_t vertex_address;

         if (!(binding_mask & BITFIELD_BIT(binding)))
            continue;
         for (element_index = 0;
              element_index < vertex_layout.count;
              ++element_index) {
            if (context->vertex_elements->elements[element_index]
                   .vertex_buffer_index == binding) {
               element = &context->vertex_elements->elements[element_index];
               break;
            }
         }
         if (!element) {
            context->last_draw_status = -9;
            return;
         }
         vertex_buffer = &context->vertex_buffers[binding];
         vertex_resource =
            (struct ps5_resource *)vertex_buffer->buffer.resource;
         vertex_address = (uintptr_t)vertex_resource->data +
                          vertex_buffer->buffer_offset;
         if (!ps5_vertex_buffer_descriptor(
                vertex_address, vertex_resource->size - vertex_buffer->buffer_offset,
                element->src_stride, binding_records[binding],
                descriptor + descriptor_index * 4)) {
            context->last_draw_status = -9;
            return;
         }
         ps5_flush_batch_backing(
            vertex_resource->external_cpu_access ? NULL : flush_cache,
            2 + PS5_MAX_TEXTURE_UNITS + binding,
            (const void *)(vertex_address + binding_begin[binding]),
            binding_bytes[binding] - binding_begin[binding]);
         descriptor_index++;
      }
      input_user_data[input_metadata->vertex_buffer_table_user_data_dword] =
         (uint32_t)descriptor_address;
      ps5_flush_gpu_data(descriptor_resource->data,
                         descriptor_index * 16);
   }
   if (tessellation_active &&
       (!ps5_prepare_tessellation_buffers(context, input_metadata,
                                          input_user_data, input_user_data_count) ||
        !ps5_prepare_tessellation_buffers(context, vertex_metadata,
                                          user_data, user_data_count))) {
      printf("[ps5-gallium] resource-prepare reject=tessellation-buffers\n");
      context->last_draw_status = -15;
      return;
   }
   if (!tessellation_active && !ps5_prepare_constant(context, context->vs, 0, input_user_data,
                             input_user_data_count, input_metadata,
                             context->gs ? NULL : &descriptor_bytes[0])) {
      printf("[ps5-gallium] resource-prepare reject=vertex-constants\n");
      context->last_draw_status = -15;
      return;
   }
   if (!tessellation_active && !ps5_prepare_texture(context, context->vs, 0, input_user_data,
                            input_user_data_count, input_metadata, NULL,
                            context->gs ? NULL : &descriptor_bytes[0])) {
      printf("[ps5-gallium] resource-prepare reject=vertex-textures\n");
      context->last_draw_status = -15;
      return;
   }
   if (descriptor_bytes[0])
      ps5_flush_gpu_data(((struct ps5_resource *)context->descriptor_storage[0])->data,
                         descriptor_bytes[0]);
   if (!tessellation_active &&
       !ps5_prepare_vertex_storage(context, input_metadata, input_user_data,
                                   input_user_data_count)) {
      printf("[ps5-gallium] resource-prepare reject=vertex-storage\n");
      context->last_draw_status = -15;
      return;
   }
   if (!tessellation_active &&
       !ps5_prepare_preraster_images(context, context->vs, input_metadata,
                                     input_user_data, input_user_data_count,
                                     PS5_VERTEX_IMAGE_OFFSET)) {
      printf("[ps5-gallium] resource-prepare reject=vertex-images\n");
      context->last_draw_status = -15;
      return;
   }
   if (tessellation_active &&
       (!ps5_prepare_texture(context, context->vs, 0, input_user_data,
                             input_user_data_count, input_metadata, NULL, NULL) ||
        !ps5_prepare_texture(context, context->vs, 0, user_data,
                             user_data_count, vertex_metadata, NULL, NULL))) {
      printf("[ps5-gallium] resource-prepare reject=tessellation-textures\n");
      context->last_draw_status = -15;
      return;
   }
   if (context->gs && !tessellation_active &&
       !ps5_prepare_geometry_storage(context, vertex_metadata, user_data,
                                     user_data_count)) {
      printf("[ps5-gallium] resource-prepare reject=geometry-storage\n");
      context->last_draw_status = -15;
      return;
   }
   if (context->gs && !tessellation_active &&
       !ps5_prepare_preraster_images(context, context->gs, vertex_metadata,
                                     user_data, user_data_count,
                                     PS5_GEOMETRY_IMAGE_OFFSET)) {
      printf("[ps5-gallium] resource-prepare reject=geometry-images\n");
      context->last_draw_status = -15;
      return;
   }
   if (!ps5_prepare_fragment_storage(context, pixel_user_data, pixel_user_data_count)) {
      printf("[ps5-gallium] resource-prepare reject=fragment-storage buffers=%u images=%u invalid=%u/%u bindings=%u\n",
             ps5_shader_storage_count(context->fs), context->fs->nir->info.num_images,
             context->fragment_bindings_invalid, context->fragment_images_invalid,
             context->fs->active->output.metadata.descriptor_binding_count);
      context->last_draw_status = -15;
      return;
   }
   if (!ps5_prepare_constant(context, context->fs, 1, pixel_user_data,
                             pixel_user_data_count, NULL, &descriptor_bytes[1])) {
      printf("[ps5-gallium] resource-prepare reject=fragment-constants\n");
      context->last_draw_status = -15;
      return;
   }
   if (!ps5_prepare_texture(context, context->fs, 1, pixel_user_data,
                            pixel_user_data_count, NULL, flush_cache, &descriptor_bytes[1])) {
      printf("[ps5-gallium] resource-prepare reject=fragment-textures\n");
      context->last_draw_status = -15;
      return;
   }
   if (descriptor_bytes[1])
      ps5_flush_gpu_data(((struct ps5_resource *)context->descriptor_storage[1])->data,
                         descriptor_bytes[1]);
   if (streamout_active &&
       !ps5_prepare_streamout(
          context, vertex_metadata, descriptor_resource, user_data,
          user_data_count, primitive_type, draws[0].count,
          info->instance_count, &streamout_mask, streamout_size,
          streamout_stride, streamout_offset,
          &streamout_written_vertices[0])) {
      context->last_draw_status = -21;
      return;
   }
   primitive_query_active = !streamout_active && (context->gs || tessellation_active) &&
      context->queries_enabled && ps5_any_generated_query(context);
   if (primitive_query_active &&
       !ps5_prepare_primitive_query(context, vertex_metadata, user_data, user_data_count)) {
      context->last_draw_status = -21;
      return;
   }
   if (!ps5_agc_gate2_run || !ps5_agc_gate2_set_packages ||
       (!PS5_ENABLE_MRT_CANDIDATE &&
        !PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE &&
        !ps5_agc_gate2_set_framebuffer) ||
       ((PS5_ENABLE_MRT_CANDIDATE ||
         PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) &&
        (!ps5_agc_gate2_set_framebuffers ||
         (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE &&
          !ps5_agc_gate2_set_scanout) ||
         (PS5_ENABLE_MRT_CANDIDATE &&
          !ps5_agc_gate2_set_graphics_state_mrt))) ||
       !ps5_agc_gate2_set_vertex_user_data ||
       (tessellation_active && !ps5_agc_gate2_set_hull_user_data) ||
       !ps5_agc_gate2_set_index_buffer ||
       !ps5_agc_gate2_set_draw_state ||
       !ps5_agc_gate2_set_instance_count ||
       !ps5_agc_gate2_set_pixel_user_data ||
       !ps5_agc_gate2_set_depth_buffer ||
       !ps5_agc_gate2_set_clip_control ||
       !ps5_agc_gate2_set_vs_out_control ||
       (PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE &&
        !ps5_agc_gate2_set_color_to_texture_barrier) ||
       (PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE &&
        !ps5_agc_gate2_set_depth_to_texture_barrier) ||
       (PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE &&
        !ps5_agc_gate2_set_depth_target_extents) ||
       (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
        !ps5_agc_gate2_set_depth_target_view) ||
       ((PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE ||
         PS5_ENABLE_TEXTURE_RG_CANDIDATE ||
         PS5_ENABLE_PACKED_FLOAT_CANDIDATE ||
         PS5_ENABLE_TEXTURE_INTEGER_CANDIDATE ||
         PS5_ENABLE_RGB10_A2UI_CANDIDATE) &&
        !ps5_agc_gate2_set_color_target_info) ||
       (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE &&
        !ps5_agc_gate2_set_color_target_extents) ||
       (PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE &&
        !ps5_agc_gate2_set_streamout) ||
       (occlusion_query && !ps5_agc_gate2_set_occlusion_query) ||
       (PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE &&
        !ps5_agc_gate2_set_dual_source_blend) ||
       ((PS5_ENABLE_MSAA4_CANDIDATE ||
         PS5_ENABLE_SMOOTH_RASTER_CANDIDATE) &&
        !ps5_agc_gate2_set_multisample_state) ||
       (PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE &&
        !ps5_agc_gate2_set_point_line_state) ||
       (PS5_ENABLE_POINT_COORD_CANDIDATE &&
        (!ps5_agc_gate2_set_interp_control ||
         !ps5_agc_gate2_set_point_coord_input)) ||
       (PS5_ENABLE_VIEWPORT_ARRAY_CANDIDATE &&
        !ps5_agc_gate2_set_viewport_states) ||
       (PS5_ENABLE_BORDER_COLOR_CANDIDATE &&
        !ps5_agc_gate2_set_border_color_table) ||
       (tessellation_active && !ps5_agc_gate2_set_tessellation) ||
       ((context->gs || tessellation_active) &&
        !ps5_agc_gate2_set_ngg_control) ||
       (!PS5_ENABLE_MRT_CANDIDATE &&
        !ps5_agc_gate2_set_graphics_state)) {
      context->last_draw_status = -3;
      return;
   }

   if (PS5_ENABLE_POINT_COORD_CANDIDATE) {
      const PsbcShaderMetadata *pixel = &context->fs->active->output.metadata;
      for (unsigned i = 0; i < pixel->input_semantic_count; ++i)
         if ((pixel->input_semantics[i] & 0xffu) == PSBC_SEMANTIC_POINT_COORD)
            point_coord_input = i + 1u;
      if (!!point_coord_input != !!(context->fs->nir->info.inputs_read & VARYING_BIT_PNTC)) {
         context->last_draw_status = -10;
         return;
      }
   }

   if (PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE &&
       ps5_agc_gate2_set_streamout(
          streamout_active ? vertex_package : NULL,
          streamout_active ? vertex_package_size : 0,
          streamout_mask, streamout_size, streamout_stride,
          streamout_offset) != 0) {
      context->last_draw_status = -21;
      return;
   }

   if (((PS5_ENABLE_MSAA4_CANDIDATE ||
         PS5_ENABLE_SMOOTH_RASTER_CANDIDATE) &&
        ps5_agc_gate2_set_multisample_state(
           sample_count, context->sample_mask,
           context->rasterizer && context->rasterizer->multisample,
           graphics.alpha_to_coverage, poly_line_smooth,
           fragment_exports.ps_iter_samples > 1 ? fragment_exports.ps_iter_samples : 0) != 0) ||
       (PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE &&
        ps5_agc_gate2_set_dual_source_blend(
           graphics.dual_source_blend) != 0) ||
       (PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE &&
        ps5_agc_gate2_set_point_line_state(
           graphics.point_line, graphics.point_line_valid) != 0) ||
       (PS5_ENABLE_POINT_COORD_CANDIDATE &&
        ps5_agc_gate2_set_interp_control(
           graphics.interp_control, graphics.interp_control_valid) != 0) ||
       (PS5_ENABLE_POINT_COORD_CANDIDATE &&
        ps5_agc_gate2_set_point_coord_input(
           point_coord_input) != 0) ||
       (PS5_ENABLE_BORDER_COLOR_CANDIDATE &&
        ps5_agc_gate2_set_border_color_table(
           ((struct ps5_resource *)context->border_color_storage)->data,
           ((struct ps5_resource *)context->border_color_storage)
              ->allocation_size) != 0) ||
        ps5_agc_gate2_set_clip_control(graphics.clip_control,
                                       graphics.clip_control_valid) != 0 ||
        ps5_agc_gate2_set_vs_out_control(vs_out_control,
                                         vs_out_control_valid) != 0 ||
       (PS5_ENABLE_VIEWPORT_ARRAY_CANDIDATE &&
        ps5_agc_gate2_set_viewport_states(
           graphics.viewport, graphics.scissor, PS5_MAX_VIEWPORTS) != 0) ||
       (PS5_ENABLE_MRT_CANDIDATE
          ? ps5_agc_gate2_set_graphics_state_mrt(
               graphics.blend_control, color_target_count,
               graphics.target_mask, graphics.color_control,
               graphics.color_control_valid, graphics.blend_color,
               graphics.viewport[0], graphics.scissor[0],
               graphics.rasterizer_control, graphics.rasterizer_valid,
               graphics.polygon_offset, graphics.polygon_offset_valid)
          : ps5_agc_gate2_set_graphics_state(
               graphics.blend_control[0], graphics.target_mask,
               graphics.color_control, graphics.color_control_valid,
               graphics.blend_color, graphics.viewport[0], graphics.scissor[0],
               graphics.rasterizer_control, graphics.rasterizer_valid,
               graphics.polygon_offset, graphics.polygon_offset_valid)) != 0) {
      context->last_draw_status = -18;
      return;
   }

   if (ps5_agc_gate2_set_packages(vertex_package,
                                  vertex_package_size,
                                  context->fs->active->package,
                                  context->fs->active->package_size) != 0) {
      context->last_draw_status = -4;
      return;
   }
   if (ps5_agc_gate2_set_tessellation &&
       ps5_agc_gate2_set_tessellation(
          tessellation_active ? context->tessellation_hs_package : NULL,
          tessellation_active ? context->tessellation_hs_package_size : 0,
          tessellation_active
             ? context->tessellation_output.runtime.hs_rsrc2 : 0,
          tessellation_active
             ? context->tessellation_output.runtime.ls_hs_config : 0,
          tessellation_active
             ? context->tessellation_output.runtime.tf_param : 0,
          tessellation_active ? context->patch_vertices : 0) != 0) {
      context->last_draw_status = -19;
      return;
   }
   if (ps5_agc_gate2_set_ngg_control &&
       ps5_agc_gate2_set_ngg_control(
          context->gs != NULL || tessellation_active,
          context->gs || tessellation_active
             ? UINT32_C(0x000007fe) : 0) != 0) {
      context->last_draw_status = -19;
      return;
   }
   {
      void *targets[PS5_MAX_RENDER_TARGETS];
      size_t target_sizes[PS5_MAX_RENDER_TARGETS];
      uint32_t target_info[PS5_MAX_RENDER_TARGETS];
      uint32_t target_widths[PS5_MAX_RENDER_TARGETS];
      uint32_t target_heights[PS5_MAX_RENDER_TARGETS];
      uint32_t target_views[PS5_MAX_RENDER_TARGETS];
      uint32_t target_pitches[PS5_MAX_RENDER_TARGETS] = {0};
      bool any_linear = false;
      struct ps5_resource *fallback = NULL;
      struct ps5_screen *screen = (struct ps5_screen *)base->screen;

      for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i) {
         if (context->framebuffer.cbufs[i].texture) {
            fallback = (struct ps5_resource *)
               context->framebuffer.cbufs[i].texture;
            break;
         }
      }
      if (!fallback && screen->render_pool)
         fallback = (struct ps5_resource *)screen->render_pool;
      if (!fallback) {
         context->last_draw_status = -5;
         return;
      }

      for (unsigned i = 0; i < color_target_count; ++i) {
         const struct pipe_surface *surface = &context->framebuffer.cbufs[i];
         struct ps5_resource *target = surface->texture
            ? (struct ps5_resource *)surface->texture
            : fallback;
         size_t layer_offset = 0;
         const unsigned first_layer = ps5_color_surface_first_layer(surface);
         /* The native layout ABI supports mixed linear/tiled MRTs. */
         target_pitches[i] = ps5_linear_color_pitch(surface);
         any_linear |= target_pitches[i] != 0;

         if (surface->texture) {
            if (target->render_staging_size && !target_pitches[i]) {
               if (!ps5_stage_color_surface(surface, true)) {
                  context->last_draw_status = -5;
                  return;
               }
               layer_offset = target->render_staging_offset;
            } else {
               layer_offset =
                  (size_t)(surface->first_layer - first_layer) * target->layer_stride +
                  target->level_offset[surface->level];
            }
         }

         if (layer_offset >= target->allocation_size) {
            context->last_draw_status = -5;
            return;
         }
         targets[i] = target->data + layer_offset;
         target_sizes[i] = target_pitches[i] ? target->size - layer_offset :
                          surface->texture && target->render_staging_size
                              ? target->render_staging_size
                              : target->allocation_size - layer_offset;
         /* Missing color slots have writes disabled. Describe only one texel:
          * the single-sample display pool is not a full-size MSAA allocation.
          * Raster bounds and depth dimensions still use the real framebuffer. */
         target_widths[i] = surface->texture ? ps5_surface_width(surface) : 1;
         target_heights[i] = surface->texture ? ps5_surface_height(surface) : 1;
         target_views[i] = surface->texture
                              ? first_layer | ((surface->last_layer -
                                 surface->first_layer + first_layer) << 13)
                              : 0;
         target_info[i] = surface->texture
            ? ps5_color_target_info(surface->format)
            : ps5_color_target_info(PIPE_FORMAT_R8G8B8A8_UNORM);
         color_to_texture_barrier |=
            (target->base.bind & (PIPE_BIND_RENDER_TARGET |
                                  PIPE_BIND_SAMPLER_VIEW)) ==
            (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
         color_to_texture_barrier |= target->base.nr_samples == 4;
         color_to_texture_barrier |= context->fs && context->fs->fbfetch_unit;
      }
      if (PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE) {
         struct ps5_resource *scanout = screen->render_pool
            ? (struct ps5_resource *)screen->render_pool : NULL;

         if (!scanout ||
             ps5_agc_gate2_set_scanout(
                scanout->data, scanout->allocation_size) != 0) {
            context->last_draw_status = -5;
            return;
         }
      }
      if (any_linear) {
         if (ps5_agc_gate2_set_color_target_layouts(targets, target_sizes, target_info,
                target_widths, target_heights, target_pitches, color_target_count) != 0) {
            context->last_draw_status = -25;
            return;
         }
      } else if (((PS5_ENABLE_MRT_CANDIDATE ||
            PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE)
             ? ps5_agc_gate2_set_framebuffers(
                  targets, target_sizes, color_target_count)
             : ps5_agc_gate2_set_framebuffer(targets[0], target_sizes[0])) !=
          0) {
         context->last_draw_status = -5;
         return;
      }
      if (!any_linear && (PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE ||
           PS5_ENABLE_TEXTURE_RG_CANDIDATE) &&
          ps5_agc_gate2_set_color_target_info(
             target_info, color_target_count) != 0) {
         context->last_draw_status = -24;
         return;
      }
      if (!any_linear && PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE &&
          ps5_agc_gate2_set_color_target_extents(
             target_widths, target_heights,
             color_target_count) != 0) {
         context->last_draw_status = -25;
         return;
      }
      if (ps5_agc_gate2_set_color_target_views &&
          ps5_agc_gate2_set_color_target_views(
             target_views, color_target_count) != 0) {
         context->last_draw_status = -28;
         return;
      }
   }
   if (context->framebuffer.zsbuf.texture) {
      struct ps5_resource *depth = (struct ps5_resource *)
         context->framebuffer.zsbuf.texture;
      const struct pipe_surface *depth_surface =
         &context->framebuffer.zsbuf;
      const struct pipe_depth_stencil_alpha_state *dsa =
         context->depth_stencil_alpha;
      struct ps5_native_depth_stencil_state native;
      void *depth_data = depth->data;
      size_t depth_allocation = depth->allocation_size;
      unsigned depth_width = ps5_surface_width(depth_surface);
      unsigned depth_height = ps5_surface_height(depth_surface);
      bool packed =
         depth->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      size_t depth_size = ps5_tiled_depth_surface_size(
         depth_width, depth_height, depth->base.nr_samples);
      size_t stencil_size = ps5_tiled_stencil_surface_size_samples(
         depth_width, depth_height, depth->base.nr_samples);
      uint32_t depth_view = depth_surface->first_layer |
         (depth_surface->last_layer << 13);

      depth_to_texture_barrier =
         (depth->base.bind & (PIPE_BIND_DEPTH_STENCIL |
                              PIPE_BIND_SAMPLER_VIEW)) ==
         (PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW);

      if (!ps5_encode_depth_stencil_state(dsa, &context->stencil_ref,
                                          &native) ||
          ((dsa->stencil[0].enabled || dsa->stencil[1].enabled) &&
           !packed)) {
         context->last_draw_status = -17;
         return;
      }
      if (depth->depth_staging_size) {
         if (!ps5_stage_depth_surface(depth_surface, true)) {
            context->last_draw_status = -17;
            return;
         }
         depth_data = depth->data + depth->depth_staging_offset;
         depth_allocation = depth->depth_staging_size;
      }
#ifdef PS5_GPU_PRESENT_BATCH
      /* Disabled depth AND stencil cannot access either backing buffer.
       * Any enabled/unknown control retains both flushes before GPU use. */
      const bool flush_depth_stencil = native.depth_control != 0;
#else
      const bool flush_depth_stencil = true;
#endif
#ifdef AGC_RUNTIME_DIAGNOSTICS
      printf("[ps5-gallium] depth-state format=%u address=%p enabled=%u write=%u func=%u control=%08x\n",
             depth->base.format, depth_data, dsa->depth_enabled,
             dsa->depth_writemask, dsa->depth_func,
             native.depth_control);
#endif
      if (PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE &&
          ps5_agc_gate2_set_depth_target_extents(
             depth_width, depth_height) != 0) {
         context->last_draw_status = -27;
         return;
      }
      if (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
          ps5_agc_gate2_set_depth_target_view(depth_view) != 0) {
         context->last_draw_status = -29;
         return;
      }
      if (flush_depth_stencil) {
         if (depth_data == depth->data)
            ps5_flush_texture_backing(flush_cache, 0, depth, depth_allocation, false);
         else
            ps5_flush_batch_backing(flush_cache, 0, depth_data, depth_allocation);
      }
      if (packed) {
         if (!ps5_agc_gate2_set_depth_stencil_buffer ||
             !depth->stencil_data ||
             depth_allocation < depth_size ||
             depth->stencil_allocation_size < stencil_size) {
            context->last_draw_status = -17;
            return;
         }
         if (flush_depth_stencil)
            ps5_flush_texture_backing(flush_cache, 1, depth,
                                       depth->stencil_allocation_size, true);
#ifdef AGC_RUNTIME_DIAGNOSTICS
         printf("[ps5-gallium] stencil-state format=%u depth=%p/%zu stencil=%p/%zu control=%08x refmask=%08x refmask-bf=%08x\n",
                depth->base.format, depth_data, depth_allocation,
                depth->stencil_data, depth->stencil_allocation_size,
                native.stencil_control, native.stencil_refmask,
                native.stencil_refmask_bf);
#endif
         if (ps5_agc_gate2_set_depth_stencil_buffer(
                depth_data, depth_allocation, depth->stencil_data,
                depth->stencil_allocation_size, native.depth_control,
                native.stencil_control, native.stencil_refmask,
                native.stencil_refmask_bf) != 0) {
            context->last_draw_status = -17;
            return;
         }
      } else if (ps5_agc_gate2_set_depth_buffer(
                    depth_data, depth_allocation,
                    native.depth_control) != 0) {
         context->last_draw_status = -17;
         return;
      }
   } else {
      if ((PS5_ENABLE_DYNAMIC_DEPTH_TARGET_CANDIDATE &&
           ps5_agc_gate2_set_depth_target_extents(
              PS5_RENDER_WIDTH, PS5_RENDER_HEIGHT) != 0) ||
          (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
           ps5_agc_gate2_set_depth_target_view(0) != 0) ||
          ps5_agc_gate2_set_depth_buffer(NULL, 0, 0) != 0) {
         context->last_draw_status = -17;
         return;
      }
   }
   if (PS5_ENABLE_RENDER_TO_TEXTURE_CANDIDATE &&
       ps5_agc_gate2_set_color_to_texture_barrier(
          color_to_texture_barrier) != 0) {
      context->last_draw_status = -23;
      return;
   }
   if (PS5_ENABLE_DEPTH_TEXTURE_CANDIDATE &&
       ps5_agc_gate2_set_depth_to_texture_barrier(
          depth_to_texture_barrier) != 0) {
      context->last_draw_status = -26;
      return;
   }
   if (ps5_agc_gate2_set_vertex_user_data(user_data,
                                          user_data_count) != 0) {
      context->last_draw_status = -6;
      return;
   }
   if (tessellation_active &&
       ps5_agc_gate2_set_hull_user_data(hull_user_data,
                                        input_user_data_count) != 0) {
      context->last_draw_status = -6;
      return;
   }
   if (ps5_agc_gate2_set_pixel_user_data(
          pixel_user_data,
          context->fs->active->output.metadata.descriptor_set0_valid
             ? pixel_user_data_count : 0) != 0) {
      context->last_draw_status = -16;
      return;
   }
   if (ps5_agc_gate2_set_draw_state(primitive_type, draws[0].count) != 0) {
      context->last_draw_status = -19;
      return;
   }
   if (ps5_agc_gate2_set_instance_count(info->instance_count) != 0) {
      context->last_draw_status = -19;
      return;
   }
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
   if (info->instance_count > 1)
      printf("[ps5-gallium] instanced count=%u vertices=%u\n",
             info->instance_count, draws[0].count);
#endif
   if (info->index_size) {
      int rc;

      ps5_flush_batch_backing(
         index_resource->external_cpu_access ? NULL : flush_cache,
         2 + PS5_MAX_TEXTURE_UNITS + PIPE_MAX_ATTRIBS,
         index_resource->data + index_offset,
         (size_t)draws[0].count * info->index_size);
      if (ps5_agc_gate2_set_index_buffer_typed)
         rc = ps5_agc_gate2_set_index_buffer_typed(
            (uint8_t *)index_resource->data + index_offset,
            draws[0].count, info->index_size);
      else if (info->index_size == 2)
         rc = ps5_agc_gate2_set_index_buffer(
            (uint8_t *)index_resource->data + index_offset,
            draws[0].count);
      else
         rc = -1;
      if (rc != 0) {
         context->last_draw_status = -13;
         return;
      }
   } else if (ps5_agc_gate2_set_index_buffer(NULL, 0) != 0) {
      context->last_draw_status = -13;
      return;
   }
   if (occlusion_query &&
       (!ps5_prepare_occlusion_query(occlusion_query) ||
        ps5_agc_gate2_set_occlusion_query(
           ((struct ps5_resource *)occlusion_query->buffer)->data,
           PS5_OCCLUSION_QUERY_BYTES,
           occlusion_query->type !=
              PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE) != 0)) {
      context->last_draw_status = -22;
      return;
   }
   /* Temporary Gate 5 bring-up: the runner still owns the target and command
    * submission, but now consumes packages compiled by Gallium state. */
   if (submitted)
      *submitted = true;
   context->last_draw_status = ps5_agc_gate2_run();
   if (context->last_draw_status == 0) {
      for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i) {
         const struct pipe_surface *surface = &context->framebuffer.cbufs[i];
         const struct ps5_resource *target = surface->texture
            ? (const struct ps5_resource *)surface->texture : NULL;

         if (target && target->render_staging_size &&
             !ps5_linear_color_pitch(surface) &&
             !ps5_stage_color_surface(surface, false)) {
            context->last_draw_status = -29;
            break;
         }
         if (target && target->render_staging_size &&
             !ps5_linear_color_pitch(surface) &&
             context->samplers[1][0] &&
             ((const struct ps5_sampler_state *)
                 context->samplers[1][0])->base.compare_mode)
            printf("[ps5-gallium] shadow-output first=%02x%02x%02x%02x\n",
                   target->data[0], target->data[1], target->data[2],
                   target->data[3]);
      }
      if (context->last_draw_status == 0 &&
          context->framebuffer.zsbuf.texture) {
         const struct pipe_surface *surface = &context->framebuffer.zsbuf;
         const struct ps5_resource *depth =
            (const struct ps5_resource *)surface->texture;

         if (depth->depth_staging_size &&
             !ps5_stage_depth_surface(surface, false))
            context->last_draw_status = -29;
      }
   }
#if PS5_PUBLIC_TEXTURE_RG_TILE_TEST
   if (context->last_draw_status == 0 && context->draw_calls <= 3 &&
       !ps5_record_public_rg_tile(context))
      context->last_draw_status = -28;
#endif
   if (occlusion_query) {
      int disable_status =
         ps5_agc_gate2_set_occlusion_query(NULL, 0, false);

      if (context->last_draw_status == 0 &&
          (disable_status != 0 ||
           (!flush_cache && !ps5_collect_occlusion_query(occlusion_query))))
         context->last_draw_status = -22;
   }
   if (streamout_active && (context->gs || tessellation_active) &&
       context->last_draw_status == 0 &&
       !ps5_collect_geometry_streamout(
          context, descriptor_resource, vertex_metadata, streamout_written_vertices,
          generated_primitives))
      context->last_draw_status = -21;
   if (streamout_active && context->last_draw_status == 0) {
      unsigned vertices_per_primitive =
         ps5_streamout_vertices_per_primitive(
            context->stream_output_primitive);
      uint64_t emitted_primitives[PIPE_MAX_VERTEX_STREAMS] = {0};

      for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS; ++stream)
         emitted_primitives[stream] = vertices_per_primitive
            ? streamout_written_vertices[stream] / vertices_per_primitive : 0;

      for (unsigned index = 0;
           index < context->stream_output_target_count; ++index) {
         struct ps5_stream_output_target *target =
            (struct ps5_stream_output_target *)
               context->stream_output_targets[index];
         unsigned stream = ps5_streamout_buffer_stream(
            vertex_output->metadata.streamout_enabled_stream_buffers_mask,
            index);

         if (target && (streamout_mask & BITFIELD_BIT(index))) {
            target->offset += (unsigned)(streamout_written_vertices[stream] *
                                         streamout_stride[index] * 4u);
            target->vertex_count +=
               (unsigned)streamout_written_vertices[stream];
         }
      }
      if (context->queries_enabled && vertices_per_primitive)
         for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS;
              ++stream)
            if (context->active_primitives_emitted_query[stream])
               context->active_primitives_emitted_query[stream]->value +=
                  emitted_primitives[stream];
      if (context->queries_enabled) {
         bool any_overflow = false;

         for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS;
              ++stream) {
            bool overflow = emitted_primitives[stream] <
                            generated_primitives[stream];

            any_overflow |= overflow;
            if (overflow &&
                context->active_streamout_overflow_query[stream])
               context->active_streamout_overflow_query[stream]->value = 1;
         }
         if (any_overflow &&
             context->active_streamout_overflow_query[
                PIPE_MAX_VERTEX_STREAMS])
            context->active_streamout_overflow_query[
               PIPE_MAX_VERTEX_STREAMS]->value = 1;
      }
   }
   if (primitive_query_active && context->last_draw_status == 0) {
      struct ps5_resource *storage = (void *)context->primitive_query_storage;
      ps5_flush_gpu_data(storage->data, 64);
      const uint32_t *counts = (void *)(storage->data + vertex_metadata->primitive_query_counter_offset);
      for (unsigned stream = 0; stream < 4; ++stream)
         generated_primitives[stream] = counts[stream];
      printf("[ps5-gallium] primitive-query generated=%llu\n", (unsigned long long)generated_primitives[0]);
   }
   if (context->queries_enabled && context->last_draw_status == 0 &&
       (!context->gs || streamout_active || primitive_query_active))
      for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS;
           ++stream)
         if (context->active_primitives_generated_query[stream])
            context->active_primitives_generated_query[stream]->value +=
               generated_primitives[stream];
#ifdef PS5_PUBLIC_STENCIL_TEST
   if (context->last_draw_status == 0 && context->draw_calls == 17) {
      struct ps5_resource *depth = context->framebuffer.zsbuf.texture
         ? (struct ps5_resource *)context->framebuffer.zsbuf.texture : NULL;

      if (!ps5_record_public_stencil(depth))
         context->last_draw_status = -20;
   }
#endif
}

static void
ps5_draw_vbo_without_adjacency(
   struct pipe_context *base, const struct pipe_draw_info *info,
   unsigned drawid_offset, const struct pipe_draw_start_count_bias *draw)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_draw_info new_info = *info;
   struct pipe_draw_start_count_bias new_draw = *draw;
   struct pipe_resource *release_buffer = NULL;
   const void *source = NULL;
   uint32_t *output;
   unsigned primitive_count;
   unsigned vertices_per_primitive;
   unsigned output_count;
   unsigned output_offset;
   unsigned min_index = UINT_MAX;
   unsigned max_index = 0;

   switch (info->mode) {
   case MESA_PRIM_LINES_ADJACENCY:
      new_info.mode = MESA_PRIM_LINES;
      primitive_count = draw->count / 4u;
      vertices_per_primitive = 2;
      break;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      new_info.mode = MESA_PRIM_LINES;
      primitive_count = draw->count >= 4 ? draw->count - 3u : 0;
      vertices_per_primitive = 2;
      break;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      new_info.mode = MESA_PRIM_TRIANGLES;
      primitive_count = draw->count / 6u;
      vertices_per_primitive = 3;
      break;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      new_info.mode = MESA_PRIM_TRIANGLES;
      primitive_count = draw->count >= 6 ? (draw->count - 4u) / 2u : 0;
      vertices_per_primitive = 3;
      break;
   default:
      return;
   }
   if (!primitive_count) {
      context->last_draw_status = 0;
      return;
   }
   if ((uint64_t)primitive_count * vertices_per_primitive > UINT_MAX / 4u) {
      context->last_draw_status = -2;
      return;
   }
   output_count = primitive_count * vertices_per_primitive;
   if (info->index_size) {
      struct ps5_resource *resource = (struct ps5_resource *)info->index.resource;
      uint64_t source_offset = (uint64_t)draw->start * info->index_size;
      uint64_t source_size = (uint64_t)draw->count * info->index_size;

      if (info->has_user_indices ||
          (info->index_size != 2 && info->index_size != 4) ||
          !resource || !resource->data || source_offset > resource->size ||
          source_size > resource->size - source_offset) {
         context->last_draw_status = -2;
         return;
      }
      source = (const uint8_t *)resource->data + source_offset;
   }
   u_upload_alloc(base->stream_uploader, 0, output_count * 4u, 4,
                  &output_offset, &new_info.index.resource,
                  &release_buffer, (void **)&output);
   if (!output) {
      context->last_draw_status = -2;
      return;
   }

   for (unsigned primitive = 0; primitive < primitive_count; ++primitive) {
      unsigned positions[3];

      switch (info->mode) {
      case MESA_PRIM_LINES_ADJACENCY:
         positions[0] = primitive * 4u + 1u;
         positions[1] = primitive * 4u + 2u;
         break;
      case MESA_PRIM_LINE_STRIP_ADJACENCY:
         positions[0] = primitive + 1u;
         positions[1] = primitive + 2u;
         break;
      case MESA_PRIM_TRIANGLES_ADJACENCY:
         positions[0] = primitive * 6u;
         positions[1] = primitive * 6u + 2u;
         positions[2] = primitive * 6u + 4u;
         break;
      case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY: {
         unsigned first = primitive * 2u;

         positions[0] = first + (primitive & 1u ? 2u : 0u);
         positions[1] = first + (primitive & 1u ? 0u : 2u);
         positions[2] = first + 4u;
         break;
      }
      default:
         __builtin_unreachable();
      }
      for (unsigned vertex = 0; vertex < vertices_per_primitive; ++vertex) {
         uint64_t value = info->index_size
                             ? ps5_index_value(source, info->index_size,
                                               positions[vertex])
                             : (uint64_t)draw->start + positions[vertex];

         if (value > UINT_MAX) {
            u_upload_unmap(base->stream_uploader);
            pipe_resource_release(base, release_buffer);
            context->last_draw_status = -2;
            return;
         }
         output[primitive * vertices_per_primitive + vertex] = (uint32_t)value;
         min_index = MIN2(min_index, (unsigned)value);
         max_index = MAX2(max_index, (unsigned)value);
      }
   }
   u_upload_unmap(base->stream_uploader);
   new_info.index_size = 4;
   new_info.has_user_indices = false;
   new_info.primitive_restart = false;
   new_info.index_bounds_valid = true;
   new_info.min_index = min_index;
   new_info.max_index = max_index;
   new_draw.start = output_offset / 4u;
   new_draw.count = output_count;
   new_draw.index_bias = info->index_size ? draw->index_bias : 0;
   base->draw_vbo(base, &new_info, drawid_offset, NULL, &new_draw, 1);
   pipe_resource_release(base, release_buffer);
}

#ifdef PS5_MULTIDRAW_BATCH
int ps5_agc_gate2_batch_begin(void) __attribute__((weak));
int ps5_agc_gate2_batch_begin_framebuffer(void) __attribute__((weak));
int ps5_agc_gate2_batch_end(void) __attribute__((weak));
int ps5_agc_gate2_batch_submit(void) __attribute__((weak));
int ps5_agc_gate2_batch_retire(int wait) __attribute__((weak));

enum ps5_batch_eligibility {
   PS5_BATCH_ELIGIBLE,
   PS5_BATCH_REJECT_INPUT,
   PS5_BATCH_REJECT_SHADER,
   PS5_BATCH_REJECT_QUERY,
   PS5_BATCH_REJECT_FRAMEBUFFER,
   PS5_BATCH_REJECT_DEPTH,
   PS5_BATCH_REJECT_VERTEX,
   PS5_BATCH_REJECT_TEXTURE,
};

#ifdef PS5_DRAW_PROFILE
static void
ps5_record_framebuffer_fallback(struct ps5_context *context, const unsigned key[16])
{
   /* Bounded diagnostic: count exact layouts; never allocate or print per draw. */
   for (unsigned i = 0; i < 32; ++i) {
      if (!context->framebuffer_fallbacks[i].count ||
          !memcmp(context->framebuffer_fallbacks[i].key, key, sizeof(unsigned) * 16)) {
         memcpy(context->framebuffer_fallbacks[i].key, key, sizeof(unsigned) * 16);
         ++context->framebuffer_fallbacks[i].count;
         return;
      }
   }
   ++context->framebuffer_fallback_overflow;
}
#endif

static bool
ps5_batch_eligibility(struct ps5_context *context, unsigned rejects)
{
   if (!rejects)
      context->batch_eligible++;
   else
      for (unsigned i = 0; i < PS5_BATCH_REJECT_TEXTURE; ++i)
         if (rejects & BITFIELD_BIT(i))
            context->batch_reject[i]++;
#ifdef PS5_DRAW_PROFILE
   if (++context->batch_checks % 100000 == 0)
      printf("[ps5-batch-checks] calls=%" PRIu64 " eligible=%" PRIu64
             " input=%" PRIu64 " shader=%" PRIu64 " query=%" PRIu64
             " framebuffer=%" PRIu64 " depth=%" PRIu64
             " vertex=%" PRIu64 " texture=%" PRIu64 "\n",
             context->batch_checks, context->batch_eligible,
             context->batch_reject[0], context->batch_reject[1],
             context->batch_reject[2], context->batch_reject[3],
             context->batch_reject[4], context->batch_reject[5],
             context->batch_reject[6]);
#endif
   return rejects == 0;
}

static bool
ps5_multidraw_eligible(struct ps5_context *context,
                       const struct pipe_draw_info *info,
                       const struct pipe_draw_indirect_info *indirect,
                       const struct pipe_draw_start_count_bias *draws,
                       unsigned num_draws)
{
   const struct ps5_resource *depth =
      (const struct ps5_resource *)context->framebuffer.zsbuf.texture;
   const struct pipe_depth_stencil_alpha_state *dsa = context->depth_stencil_alpha;
   unsigned rejects = 0;

   if (!info || !draws || !num_draws || indirect ||
       (info->mode != MESA_PRIM_TRIANGLES &&
        !(context->deferred_blitter_draw && context->blitter && context->blitter->running &&
          info->mode == MESA_PRIM_TRIANGLE_FAN && num_draws == 1 &&
          !info->index_size && info->instance_count == 1 && !info->start_instance &&
          draws[0].start == 0 && draws[0].count == 4)) || !info->instance_count ||
       info->primitive_restart || info->has_user_indices ||
       (info->index_size && info->index_size != 2 && info->index_size != 4) ||
       (info->index_size && !info->index.resource))
      rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_INPUT - 1);
   if (!context->vs || !context->fs || context->gs ||
       (context->fs && ps5_shader_uses_storage(context->fs)) ||
       (context->vs && ps5_shader_uses_storage(context->vs)))
      rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_SHADER - 1);
   if (context->stream_output_target_count || context->render_condition_query)
      rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_QUERY - 1);
   if (!context->framebuffer_valid ||
       context->framebuffer.nr_cbufs > PS5_MAX_RENDER_TARGETS) {
      rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_FRAMEBUFFER - 1);
#ifdef PS5_DRAW_PROFILE
      ps5_record_framebuffer_fallback(context, (unsigned[16]){
         context->framebuffer_valid ? 2 : 1, context->framebuffer.nr_cbufs});
#endif
   }
   else
      for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i) {
         const struct pipe_surface *surface = &context->framebuffer.cbufs[i];
         const struct ps5_resource *target =
            (const struct ps5_resource *)surface->texture;

         /* Reject only attachments that still require a CPU copy. */
         if (target && target->render_staging_size &&
             !ps5_linear_color_pitch(surface)) {
            rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_FRAMEBUFFER - 1);
#ifdef PS5_DRAW_PROFILE
            ps5_record_framebuffer_fallback(context, (unsigned[16]){
               3, context->framebuffer.nr_cbufs, i, surface->format,
               target->base.format, target->base.target,
               ps5_surface_width(surface), ps5_surface_height(surface),
               target->base.nr_samples, surface->level,
               surface->first_layer, surface->last_layer,
               surface->level < ARRAY_SIZE(target->level_stride)
                  ? target->level_stride[surface->level] : 0,
               target->base.bind, target->base.nr_storage_samples,
               (unsigned)((uintptr_t)target->data & 255u)});
#endif
            break;
         }
      }
   if (depth && (depth->depth_staging_size || !dsa ||
                  !ps5_depth_render_target(depth->base.target) ||
                  (depth->base.format != PIPE_FORMAT_Z32_FLOAT &&
                   depth->base.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) ||
                  context->framebuffer.zsbuf.format != depth->base.format ||
                  context->framebuffer.zsbuf.level > depth->base.last_level ||
                  context->framebuffer.zsbuf.first_layer >
                     context->framebuffer.zsbuf.last_layer ||
                  context->framebuffer.zsbuf.last_layer >=
                     ps5_texture_level_layers(
                        &depth->base, context->framebuffer.zsbuf.level)))
      rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_DEPTH - 1);
   if (context->vertex_buffer_count > PIPE_MAX_ATTRIBS)
      rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_VERTEX - 1);
   for (unsigned i = 0; i < MIN2(context->vertex_buffer_count,
                                  PIPE_MAX_ATTRIBS); ++i)
      if (context->vertex_buffers[i].is_user_buffer)
         rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_VERTEX - 1);
   const struct ps5_shader *texture_stages[] = {context->vs, context->fs};
   for (unsigned stage = 0; stage < ARRAY_SIZE(texture_stages); ++stage) {
      if (!texture_stages[stage])
         continue;
      for (unsigned unit = 0; unit < PS5_MAX_TEXTURE_UNITS; ++unit) {
         if (!ps5_texture_used(context, texture_stages[stage], NULL, unit))
            continue;
         const struct pipe_sampler_view *view = context->sampler_views[stage][unit];
         const struct ps5_resource *texture = view
            ? (const struct ps5_resource *)view->texture : NULL;
         const bool staged_stencil = texture &&
            texture->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
            view->format == PIPE_FORMAT_X32_S8X24_UINT;

         /* Descriptor preparation handles all valid targets, formats, levels,
          * layers and sample counts. Only per-draw staging and attachment alias
          * hazards prevent deferred execution. */
         if (!view)
            continue;
         if (!texture || !texture->data || !texture->size ||
             texture == depth || texture->depth_staging_size || staged_stencil) {
            rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_TEXTURE - 1);
            continue;
         }
         for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i)
            if (texture == (const struct ps5_resource *)
                              context->framebuffer.cbufs[i].texture) {
               rejects |= BITFIELD_BIT(PS5_BATCH_REJECT_TEXTURE - 1);
               break;
            }
      }
   }
   return ps5_batch_eligibility(context, rejects);
}

#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DEFERRED_DRAW_BATCH)
/* Only confirmed-retired, exclusively owned descriptor snapshots enter here.
 * The fixed 16KiB snapshot class bounds this cache to 12MiB per context. */
static struct pipe_resource *
ps5_descriptor_take(struct pipe_context *base, unsigned stage,
                     const struct pipe_resource *templ)
{
   struct ps5_context *context = (struct ps5_context *)base;
   for (unsigned i = 0; i < context->descriptor_cache_count[stage]; ++i) {
      struct pipe_resource *r = context->descriptor_cache[stage][i];
      if (r->width0 != templ->width0 || r->format != templ->format ||
          r->bind != templ->bind || r->usage != templ->usage || r->flags != templ->flags)
         continue;
      context->descriptor_cache[stage][i] =
         context->descriptor_cache[stage][--context->descriptor_cache_count[stage]];
      ++context->descriptor_cache_hits;
      return r;
   }
   ++context->descriptor_cache_misses;
   return base->screen->resource_create(base->screen, templ);
}

static void
ps5_descriptor_recycle(struct ps5_context *context, struct pipe_resource *storage[3])
{
   for (unsigned stage = 0; stage < 3; ++stage) {
      struct pipe_resource *r = storage[stage];
      if (!r || r->reference.count != 1 || r->target != PIPE_BUFFER ||
          r->width0 != PS5_DIRECT_ALIGNMENT ||
          context->descriptor_cache_count[stage] == PS5_MULTIDRAW_BATCH_CAPACITY)
         continue;
      context->descriptor_cache[stage][context->descriptor_cache_count[stage]++] = r;
      storage[stage] = NULL; /* Transfer the existing reference. */
   }
}

static void
ps5_descriptor_cache_clear(struct ps5_context *context)
{
   for (unsigned stage = 0; stage < 3; ++stage)
      while (context->descriptor_cache_count[stage])
         pipe_resource_reference(
            &context->descriptor_cache[stage][--context->descriptor_cache_count[stage]], NULL);
}
#endif

static bool
ps5_batch_copy_descriptors(struct pipe_context *base,
                           struct pipe_resource *const saved[3],
                           struct pipe_resource *storage[3])
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {1, ps5_prepare_clock()};
#endif

   for (unsigned stage = 0; stage < 3; ++stage) {
      const struct ps5_resource *source = (const struct ps5_resource *)saved[stage];
      if (!source || !source->data || source->base.target != PIPE_BUFFER)
         return false;
      struct pipe_resource templ = source->base;
      const size_t live_bytes = stage ? ps5_descriptor_snapshot_size(
         (const struct ps5_context *)base, stage - 1) : source->size;
      if (stage)
         templ.width0 = MAX2(PS5_DIRECT_ALIGNMENT, live_bytes);
      if (templ.width0 > source->size)
         return false;
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DEFERRED_DRAW_BATCH)
      storage[stage] = ps5_descriptor_take(base, stage, &templ);
#else
      storage[stage] = base->screen->resource_create(base->screen, &templ);
#endif
      struct ps5_resource *copy = (struct ps5_resource *)storage[stage];
      if (!copy || !copy->data || copy->size != templ.width0 ||
          (uintptr_t)copy->data >> 32 != (uintptr_t)source->data >> 32)
         return false;
      /* Ordinary draws regenerate every vertex descriptor they can fetch before
       * enqueue. Keep private storage and its lifetime, but not stale contents.
       * Merged stages and stream output retain the conservative snapshot. */
      const struct ps5_context *context = (const struct ps5_context *)base;
      if (stage || context->gs || context->tcs || context->tes ||
          context->stream_output_target_count) {
         size_t begin = 0;
         const struct ps5_shader *shader = stage == 1 ? context->vs : context->fs;
         /* Constant preparation clears this prefix before any consumer. Keep
          * inline constants above it and retain layouts that do not clear it. */
         if (stage && !context->gs && !context->tcs && !context->tes &&
             shader && shader->nir->info.num_ubos && !ps5_shader_uses_storage(shader))
            begin = PS5_CONSTANT_DATA_OFFSET;
         if (begin > live_bytes)
            return false;
         if (live_bytes > begin)
            memcpy(copy->data + begin, source->data + begin, live_bytes - begin);
      }
   }
   return true;
}

#define PS5_BATCH_RESOURCE_COUNT (PIPE_MAX_ATTRIBS + \
   2 * PS5_MAX_CONSTANT_BUFFERS + 2 * PS5_MAX_TEXTURE_UNITS + \
   PS5_MAX_RENDER_TARGETS + 4)

static unsigned
ps5_batch_retain_resources(const struct ps5_context *context,
                           const struct pipe_draw_info *info,
                           struct pipe_resource *retained[PS5_BATCH_RESOURCE_COUNT])
{
   const struct ps5_screen *screen = (const struct ps5_screen *)context->base.screen;
   unsigned retained_count = 0;
   for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i)
      pipe_resource_reference(&retained[retained_count++],
                              context->framebuffer.cbufs[i].texture);
   pipe_resource_reference(&retained[retained_count++], context->framebuffer.zsbuf.texture);
   pipe_resource_reference(&retained[retained_count++], screen->render_pool);
   /* Border entries are append-only for the lifetime of the context. */
   pipe_resource_reference(&retained[retained_count++], context->border_color_storage);
   if (info->index_size)
      pipe_resource_reference(&retained[retained_count++], info->index.resource);
   /* Retain only bindings this pipeline can read. Unused stale bindings must
    * not turn an unrelated CPU upload into a GPU completion dependency. */
   uint32_t vertex_bindings = 0;
   if (context->vertex_elements)
      for (unsigned i = 0; i < context->vertex_elements->count; ++i)
         if (context->vertex_elements->elements[i].vertex_buffer_index < PIPE_MAX_ATTRIBS)
            vertex_bindings |= BITFIELD_BIT(context->vertex_elements->elements[i].vertex_buffer_index);
   for (unsigned i = 0; i < context->vertex_buffer_count; ++i)
      if (vertex_bindings & BITFIELD_BIT(i))
         pipe_resource_reference(&retained[retained_count++], context->vertex_buffers[i].buffer.resource);
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct ps5_shader *shader = stage ? context->fs : context->vs;
      const bool all = context->gs || context->tcs || context->tes;
      const unsigned count = all ? PS5_MAX_CONSTANT_BUFFERS :
         shader ? MIN2(shader->nir->info.num_ubos, PS5_MAX_CONSTANT_BUFFERS) : 0;
      for (unsigned i = 0; i < count; ++i) {
         unsigned binding = all ? i : ps5_constant_state_binding(shader, i);
         if (binding < PS5_MAX_CONSTANT_BUFFERS &&
             context->constants[stage][binding].valid &&
             !context->constants[stage][binding].copied)
            pipe_resource_reference(&retained[retained_count++], context->constants[stage][binding].buffer);
      }
   }
   const struct ps5_shader *texture_stages[] = {context->vs, context->fs};
   for (unsigned stage = 0; stage < ARRAY_SIZE(texture_stages); ++stage) {
      if (!texture_stages[stage])
         continue;
      for (unsigned unit = 0; unit < PS5_MAX_TEXTURE_UNITS; ++unit)
         if (context->sampler_views[stage][unit] &&
             ps5_texture_used(context, texture_stages[stage], NULL, unit))
            pipe_resource_reference(&retained[retained_count++],
                                    context->sampler_views[stage][unit]->texture);
   }
   return retained_count;
}

static bool
ps5_try_multi_draw_batch(struct pipe_context *base,
                         const struct pipe_draw_info *info, unsigned drawid_offset,
                         const struct pipe_draw_indirect_info *indirect,
                         const struct pipe_draw_start_count_bias *draws,
                         unsigned num_draws)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_resource *saved[3] = {context->vertex_descriptor_table,
      context->descriptor_storage[0], context->descriptor_storage[1]};
   struct pipe_resource *storage[PS5_MULTIDRAW_BATCH_CAPACITY][3] = {{0}};
   struct pipe_resource *retained[PS5_BATCH_RESOURCE_COUNT] = {0};
   unsigned retained_count = 0;
   unsigned slots = MIN2(num_draws, PS5_MULTIDRAW_BATCH_CAPACITY);
   bool handled = false, retired = true;

   if (num_draws < 2 || context->active_occlusion_query ||
       ps5_any_primitive_query(context) ||
       !ps5_agc_gate2_batch_begin || !ps5_agc_gate2_batch_end ||
       !ps5_multidraw_eligible(context, info, indirect, draws, num_draws))
      return false;
   /* Preserve copied inline uniforms as well as descriptor storage. Each slot
    * is reused only after the previous chunk has completely retired. */
   for (unsigned slot = 0; slot < slots; ++slot)
      if (!ps5_batch_copy_descriptors(base, saved, storage[slot]))
         goto release;
   retained_count = ps5_batch_retain_resources(context, info, retained);

   ps5_screen_submit_lock(base->screen);
   handled = true;
   context->last_draw_status = 0;
   for (unsigned first = 0; first < num_draws;) {
      struct ps5_batch_flush_cache flush_cache = {0};
      if ((ps5_agc_gate2_batch_begin_framebuffer ? ps5_agc_gate2_batch_begin_framebuffer() :
           ps5_agc_gate2_batch_begin()) != 0) {
         context->last_draw_status = -30;
         break;
      }
      bool retry_unsubmitted = false;
      for (unsigned slot = 0; slot < slots && first < num_draws; ++slot, ++first) {
         bool submitted = false;
         context->vertex_descriptor_table = storage[slot][0];
         context->descriptor_storage[0] = storage[slot][1];
         context->descriptor_storage[1] = storage[slot][2];
         if (draws[first].count)
            ps5_draw_vbo_locked(base, info,
               drawid_offset + (info->increment_draw_id ? first : 0),
               NULL, &draws[first], 1, &flush_cache, &submitted);
         if (context->last_draw_status) {
            retry_unsubmitted = !submitted;
            break;
         }
      }
      context->vertex_descriptor_table = saved[0];
      context->descriptor_storage[0] = saved[1];
      context->descriptor_storage[1] = saved[2];
      if (ps5_agc_gate2_batch_end() != 0) {
         context->last_draw_status = -30;
         retired = false;
      }
      if (retired && retry_unsubmitted) {
         context->last_draw_status = 0;
         ps5_draw_vbo_locked(base, info,
            drawid_offset + (info->increment_draw_id ? first : 0),
            NULL, &draws[first], 1, NULL, NULL);
         if (!context->last_draw_status) {
            ++first;
            continue;
         }
      }
      if (context->last_draw_status)
         break;
   }
   ps5_screen_submit_unlock(base->screen);
   printf("[ps5-gallium] multi-draw-batched draws=%u result=%d\n",
          num_draws, context->last_draw_status);
release:
   /* ponytail: post-retirement cleanup failure pins bounded resources until
    * process teardown; unconfirmed retirement terminates before returning. */
   if (retired) {
      for (unsigned slot = 0; slot < slots; ++slot)
         for (unsigned stage = 0; stage < 3; ++stage)
            pipe_resource_reference(&storage[slot][stage], NULL);
      for (unsigned i = 0; i < retained_count; ++i)
         pipe_resource_reference(&retained[i], NULL);
   }
   return handled;
}
#endif

#ifdef PS5_DEFERRED_DRAW_BATCH
/* Each FIFO slot pins its descriptors and resources until matching native retirement. */
struct ps5_deferred_slot {
   struct pipe_resource *storage[3];
   struct pipe_resource *retained[PS5_BATCH_RESOURCE_COUNT];
   unsigned retained_count;
   struct ps5_query *occlusion_query;
   struct pipe_resource *occlusion_buffer;
   bool compute; /* a queued dispatch, ordered by its own GPU waits */
};

struct ps5_deferred_batch {
   struct ps5_context *owner;
   uint64_t sequence;
   unsigned count;
   struct ps5_batch_flush_cache flush_cache;
   struct ps5_deferred_slot slots[PS5_MULTIDRAW_BATCH_CAPACITY];
};
static struct ps5_deferred_batch ps5_deferred;
static struct ps5_deferred_batch ps5_inflight[PS5_INFLIGHT_BATCH_CAPACITY];
static unsigned ps5_inflight_head, ps5_inflight_count;
static uint64_t ps5_submitted_sequence, ps5_completed_sequence;

static void
ps5_deferred_slot_release(struct ps5_deferred_slot *slot)
{
   pipe_resource_reference(&slot->occlusion_buffer, NULL);
   for (unsigned stage = 0; stage < 3; ++stage)
      pipe_resource_reference(&slot->storage[stage], NULL);
   for (unsigned i = 0; i < slot->retained_count; ++i)
      pipe_resource_reference(&slot->retained[i], NULL);
   memset(slot, 0, sizeof(*slot));
}

void
ps5_context_queue_present(struct pipe_context *base, unsigned buffer_index)
{
#ifdef PS5_GPU_PRESENT_BATCH
   extern int ps5_agc_gate2_batch_present(unsigned);
   simple_mtx_lock(&ps5_deferred_mutex);
   if (ps5_deferred.owner == (struct ps5_context *)base && ps5_deferred.count &&
       ps5_agc_gate2_batch_present(buffer_index) != 0) {
      fputs("[ps5-gallium] queued presentation failed; terminating before resource release\n", stderr);
      _Exit(EXIT_FAILURE);
   }
   simple_mtx_unlock(&ps5_deferred_mutex);
#else
   (void)base;
   (void)buffer_index;
#endif
}

static bool
ps5_draw_batch_compute_only(void)
{
   simple_mtx_lock(&ps5_deferred_mutex);
   bool compute = true;
   for (unsigned slot = 0; slot < ps5_deferred.count; ++slot)
      compute &= ps5_deferred.slots[slot].compute;
   simple_mtx_unlock(&ps5_deferred_mutex);
   return compute;
}

static void
ps5_deferred_batch_release(struct ps5_deferred_batch *batch)
{
   if (!batch->owner)
      return;
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
   printf("[ps5-deferred-batch] draws=%u result=0\n", batch->count);
#endif
   for (unsigned slot = 0; slot < batch->count; ++slot) {
      if (batch->slots[slot].occlusion_query &&
          !ps5_collect_occlusion_query_resource(
             batch->slots[slot].occlusion_query,
             batch->slots[slot].occlusion_buffer)) {
         fputs("[ps5-gallium] deferred query collection failed; terminating before resource release\n", stderr);
         fflush(stderr);
         _Exit(EXIT_FAILURE);
      }
#ifdef PS5_NATIVE_TITLE_RUNTIME
      ps5_descriptor_recycle(batch->owner, batch->slots[slot].storage);
#endif
      ps5_deferred_slot_release(&batch->slots[slot]);
   }
   /* Released slots are already zero; unused capacity never owned resources. */
   memset(batch, 0, offsetof(struct ps5_deferred_batch, slots));
}

static bool
ps5_draw_batch_retire_one_locked(bool wait)
{
   if (!ps5_inflight_count)
      return false;
   int status = ps5_agc_gate2_batch_retire
      ? ps5_agc_gate2_batch_retire(wait)
      : -1;
   if (!status)
      return false;
   if (status < 0) {
      fputs("[ps5-gallium] deferred batch retirement failed; terminating before resource release\n", stderr);
      fflush(stderr);
      _Exit(EXIT_FAILURE);
   }
   struct ps5_deferred_batch *batch = &ps5_inflight[ps5_inflight_head];
   ps5_completed_sequence = batch->sequence;
   ps5_deferred_batch_release(batch);
   ps5_inflight_head = (ps5_inflight_head + 1) % PS5_INFLIGHT_BATCH_CAPACITY;
   --ps5_inflight_count;
   return true;
}

static void
ps5_draw_batch_retire_locked(bool wait)
{
   while (ps5_draw_batch_retire_one_locked(wait)) {}
}

static void
ps5_draw_batch_flush_locked(void)
{
   if (!ps5_deferred.owner)
      return;
   if (ps5_agc_gate2_batch_submit && ps5_agc_gate2_batch_retire) {
      ps5_draw_batch_retire_locked(false);
      if (ps5_inflight_count == PS5_INFLIGHT_BATCH_CAPACITY)
         ps5_draw_batch_retire_one_locked(true);
      if (ps5_agc_gate2_batch_submit() != 0) {
         fputs("[ps5-gallium] deferred batch submission failed; terminating before resource release\n", stderr);
         fflush(stderr);
         _Exit(EXIT_FAILURE);
      }
      ps5_deferred.sequence = ++ps5_submitted_sequence;
      const size_t live_bytes = offsetof(struct ps5_deferred_batch, slots) +
         ps5_deferred.count * sizeof(ps5_deferred.slots[0]);
      memcpy(&ps5_inflight[(ps5_inflight_head + ps5_inflight_count) % PS5_INFLIGHT_BATCH_CAPACITY],
             &ps5_deferred, live_bytes);
      ++ps5_inflight_count;
      memset(&ps5_deferred, 0, live_bytes);
      ps5_draw_batch_retire_locked(false);
      return;
   }
   if (ps5_agc_gate2_batch_end() != 0) {
      /* No late error can safely be reported as success by a later GL call. */
      fputs("[ps5-gallium] deferred batch cleanup failed; terminating before resource release\n", stderr);
      fflush(stderr);
      _Exit(EXIT_FAILURE);
   }
   ps5_deferred_batch_release(&ps5_deferred);
}

static uint64_t
ps5_draw_batch_fence_submit(void)
{
   simple_mtx_lock(&ps5_deferred_mutex);
   ps5_draw_batch_flush_locked();
   uint64_t sequence = ps5_submitted_sequence;
   simple_mtx_unlock(&ps5_deferred_mutex);
   return sequence;
}

static bool
ps5_draw_batch_fence_finish(uint64_t sequence, uint64_t timeout)
{
   const uint64_t start = timeout ? os_time_get_nano() : 0;
   for (;;) {
      simple_mtx_lock(&ps5_deferred_mutex);
      while (ps5_completed_sequence < sequence && ps5_inflight_count &&
             ps5_draw_batch_retire_one_locked(false)) {}
      bool complete = ps5_completed_sequence >= sequence;
      simple_mtx_unlock(&ps5_deferred_mutex);
      if (complete)
         return true;
      if (!timeout)
         return false;
      uint64_t elapsed = (uint64_t)os_time_get_nano() - start;
      if (elapsed >= timeout)
         return false;
      /* Zero-timeout probes return above; no driver lock is held here. */
      if (elapsed < 200000) {
         __builtin_ia32_pause();
         continue;
      }
      /* ponytail: sleep-poll after the spin; use a verified event when available. */
      uint64_t remaining = timeout - elapsed;
      os_time_sleep(MIN2(100u, remaining / 1000u));
   }
}

static bool
ps5_memory_overlaps(const void *a, size_t a_size, const void *b, size_t b_size)
{
   uintptr_t first = (uintptr_t)a, second = (uintptr_t)b;
   /* Unknown or invalid backing must not bypass synchronization. */
   if (!a || !b || !a_size || !b_size ||
       a_size > UINTPTR_MAX - first || b_size > UINTPTR_MAX - second)
      return true;
   return first <= second ? second - first < a_size : first - second < b_size;
}

static void
ps5_draw_batch_drain_query(struct ps5_query *query)
{
   simple_mtx_lock(&ps5_deferred_mutex);
   const struct ps5_deferred_batch *batches[PS5_INFLIGHT_BATCH_CAPACITY + 1] = {&ps5_deferred};
   for (unsigned i = 0; i < ps5_inflight_count; ++i)
      batches[i + 1] = &ps5_inflight[(ps5_inflight_head + i) % PS5_INFLIGHT_BATCH_CAPACITY];
   bool pending = false;
   for (unsigned i = 0; i <= ps5_inflight_count; ++i)
      for (unsigned slot = 0; slot < batches[i]->count; ++slot)
         pending |= batches[i]->slots[slot].occlusion_query == query;
   if (pending) {
      ps5_draw_batch_flush_locked();
      ps5_draw_batch_retire_locked(true);
   }
   simple_mtx_unlock(&ps5_deferred_mutex);
}

/* Complete only the batches contributing to this query. Availability probes
 * neither wait for retirement nor force submission when the FIFO is full. */
static bool
ps5_draw_batch_query_ready(const struct ps5_query *query, bool wait)
{
   simple_mtx_lock(&ps5_deferred_mutex);
   bool queued = false;
   for (unsigned slot = 0; slot < ps5_deferred.count; ++slot)
      queued |= ps5_deferred.slots[slot].occlusion_query == query;
   if (queued) {
      ps5_draw_batch_retire_locked(false);
      if (!wait && ps5_inflight_count == PS5_INFLIGHT_BATCH_CAPACITY) {
         simple_mtx_unlock(&ps5_deferred_mutex);
         return false;
      }
      ps5_draw_batch_flush_locked();
   }
   uint64_t sequence = 0;
   for (unsigned i = 0; i < ps5_inflight_count; ++i) {
      const struct ps5_deferred_batch *batch =
         &ps5_inflight[(ps5_inflight_head + i) % PS5_INFLIGHT_BATCH_CAPACITY];
      for (unsigned slot = 0; slot < batch->count; ++slot)
         if (batch->slots[slot].occlusion_query == query)
            sequence = batch->sequence;
   }
   while (ps5_completed_sequence < sequence && ps5_draw_batch_retire_one_locked(wait)) {}
   bool ready = ps5_completed_sequence >= sequence;
   simple_mtx_unlock(&ps5_deferred_mutex);
   return ready;
}

static bool
ps5_buffer_overlaps_resource(const struct ps5_resource *buffer,
                             const struct pipe_resource *base)
{
   const struct ps5_resource *resource = (const struct ps5_resource *)base;
   if (!resource)
      return false;
   /* The display pool also owns the allocator's arena. GPU scanout uses only
    * its two front slots; every accessed arena suballocation is retained
    * separately. Treating the whole parent as accessed would drain all uploads. */
   size_t bytes = resource->allocation_size;
   if (base->target == PIPE_TEXTURE_2D && (base->bind & PIPE_BIND_DISPLAY_TARGET))
      bytes = MIN2(bytes, PS5_RENDER_ARENA_OFFSET);
   if (&buffer->base == base ||
       ps5_memory_overlaps(buffer->data, buffer->allocation_size,
                            resource->data, bytes))
      return true;
   bool cpu_stencil = buffer->stencil_data || buffer->stencil_allocation_size;
   bool gpu_stencil = resource->stencil_data || resource->stencil_allocation_size;
   return (gpu_stencil && ps5_memory_overlaps(buffer->data, buffer->allocation_size,
                             resource->stencil_data, resource->stencil_allocation_size)) ||
       (cpu_stencil && ps5_memory_overlaps(buffer->stencil_data, buffer->stencil_allocation_size,
                          resource->data, bytes)) ||
       (cpu_stencil && gpu_stencil &&
        ps5_memory_overlaps(buffer->stencil_data, buffer->stencil_allocation_size,
                           resource->stencil_data, resource->stencil_allocation_size));
}

static bool
ps5_deferred_batch_overlaps(const struct ps5_deferred_batch *batch,
                            const struct ps5_resource *buffer)
{
   if (!batch->owner)
      return false;
   if (!buffer)
      return true;
   for (unsigned slot = 0; slot < batch->count; ++slot) {
      for (unsigned stage = 0; stage < 3; ++stage)
         if (ps5_buffer_overlaps_resource(buffer, batch->slots[slot].storage[stage]))
            return true;
      for (unsigned i = 0; i < batch->slots[slot].retained_count; ++i)
         if (ps5_buffer_overlaps_resource(buffer, batch->slots[slot].retained[i]))
            return true;
   }
   return false;
}

static void
ps5_draw_batch_drain_buffer(struct pipe_resource *base)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {3, ps5_prepare_clock()};
#endif

   struct ps5_resource *buffer = (struct ps5_resource *)base;
   simple_mtx_lock(&ps5_deferred_mutex);
   if (buffer) {
      buffer->texture_publication_epoch = 0;
      buffer->stencil_publication_epoch = 0;
   }
   /* Dedicated direct allocations and owned arena slots cannot overlap other
    * ordinary textures. Display aliases, exports and unknown backing retain
    * global invalidation. Views of an ordinary texture share this resource. */
   if (!buffer || (buffer->base.target != PIPE_BUFFER &&
       ((buffer->base.bind & PIPE_BIND_DISPLAY_TARGET) || buffer->external_cpu_access ||
        (buffer->direct_start < 0 && !buffer->render_arena_slot_count))))
      ++ps5_texture_publication_epoch;
   /* ponytail: whole allocations, bounded by the batch capacity. Range tracking only
    * if conservative alias/arena overlap becomes a measured bottleneck. */
   if (ps5_deferred_batch_overlaps(&ps5_deferred, buffer)) {
      ps5_draw_batch_flush_locked();
      ps5_draw_batch_retire_locked(true);
   } else {
      /* Find the youngest dependent batch. Newer unrelated work must stay in
       * flight; retiring the entire FIFO turns a small upload into a finish. */
      for (unsigned i = ps5_inflight_count; i-- > 0;) {
         if (ps5_deferred_batch_overlaps(
               &ps5_inflight[(ps5_inflight_head + i) % PS5_INFLIGHT_BATCH_CAPACITY], buffer)) {
            /* Do not submit unrelated queued work for a CPU access. */
            const uint64_t sequence =
               ps5_inflight[(ps5_inflight_head + i) % PS5_INFLIGHT_BATCH_CAPACITY].sequence;
            while (ps5_completed_sequence < sequence &&
                   ps5_draw_batch_retire_one_locked(true)) {}
            break;
         }
      }
   }
   if (base && (base->bind & PIPE_BIND_DISPLAY_TARGET) &&
       ps5_agc_gate2_wait_present && ps5_agc_gate2_wait_present() != 0) {
      fputs("[ps5-gallium] scanout reuse before confirmed flip completion\n", stderr);
      _Exit(EXIT_FAILURE);
   }
   ps5_invalidate_gpu_writes(buffer);
   simple_mtx_unlock(&ps5_deferred_mutex);
}

static bool
ps5_try_deferred_draw(struct pipe_context *base,
                      const struct pipe_draw_info *info, unsigned drawid_offset,
                      const struct pipe_draw_indirect_info *indirect,
                      const struct pipe_draw_start_count_bias *draws,
                      unsigned num_draws)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {2, ps5_prepare_clock()};
#endif

   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_resource *saved[3] = {context->vertex_descriptor_table,
      context->descriptor_storage[0], context->descriptor_storage[1]};
   struct pipe_resource *storage[3] = {0};
   struct ps5_query *occlusion_query = context->queries_enabled
      ? context->active_occlusion_query : NULL;
   struct pipe_resource *occlusion_buffer = NULL;
   bool handled = false;
   bool submitted = false;

   simple_mtx_lock(&ps5_deferred_mutex);
   if (ps5_deferred.owner && ps5_deferred.owner != context)
      ps5_draw_batch_flush_locked();
   if (!ps5_agc_gate2_batch_begin || !ps5_agc_gate2_batch_end || num_draws != 1 ||
       (context->blitter && context->blitter->running && !context->deferred_blitter_draw) ||
       !ps5_multidraw_eligible(context, info, indirect, draws, num_draws)) {
      ps5_draw_batch_flush_locked();
      goto out;
   }
   if (!draws[0].count) {
      context->last_draw_status = 0;
      handled = true;
      goto out;
   }
   if (!ps5_batch_copy_descriptors(base, saved, storage)) {
      ps5_draw_batch_flush_locked();
      goto out; /* Allocation failure falls back only before this draw stages. */
   }
   if (occlusion_query) {
      occlusion_buffer = base->screen->resource_create(
         base->screen, occlusion_query->buffer);
      if (!occlusion_buffer) {
         ps5_draw_batch_flush_locked();
         goto out;
      }
   }
   if (!ps5_deferred.owner) {
      if ((ps5_agc_gate2_batch_begin_framebuffer ? ps5_agc_gate2_batch_begin_framebuffer() :
           ps5_agc_gate2_batch_begin()) != 0) {
         context->last_draw_status = -30;
         handled = true;
         goto out;
      }
      ps5_deferred.owner = context;
   }
   unsigned slot = ps5_deferred.count++;
   for (unsigned stage = 0; stage < 3; ++stage) {
      ps5_deferred.slots[slot].storage[stage] = storage[stage];
      storage[stage] = NULL;
   }
   ps5_deferred.slots[slot].retained_count = ps5_batch_retain_resources(
      context, info, ps5_deferred.slots[slot].retained);
   ps5_deferred.slots[slot].occlusion_buffer = occlusion_buffer;
   occlusion_buffer = NULL;
   context->vertex_descriptor_table = ps5_deferred.slots[slot].storage[0];
   context->descriptor_storage[0] = ps5_deferred.slots[slot].storage[1];
   context->descriptor_storage[1] = ps5_deferred.slots[slot].storage[2];
   struct pipe_resource *saved_query_buffer = NULL;
   if (occlusion_query) {
      saved_query_buffer = occlusion_query->buffer;
      occlusion_query->buffer = ps5_deferred.slots[slot].occlusion_buffer;
   }
   context->last_draw_status = 0;
   ps5_draw_vbo_locked(base, info, drawid_offset, indirect, draws, num_draws,
                       &ps5_deferred.flush_cache, &submitted);
   context->vertex_descriptor_table = saved[0];
   context->descriptor_storage[0] = saved[1];
   context->descriptor_storage[1] = saved[2];
   if (occlusion_query) {
      occlusion_query->buffer = saved_query_buffer;
      if (!context->last_draw_status)
         ps5_deferred.slots[slot].occlusion_query = occlusion_query;
   }
   if (context->last_draw_status && !submitted) {
      --ps5_deferred.count;
      ps5_deferred_slot_release(&ps5_deferred.slots[slot]);
      ps5_draw_batch_flush_locked();
      context->last_draw_status = 0;
      goto out; /* The ordinary path may retry this provably unsubmitted draw. */
   }
   handled = true;
   if (context->last_draw_status || ps5_deferred.count == PS5_MULTIDRAW_BATCH_CAPACITY)
      ps5_draw_batch_flush_locked();
out:
   for (unsigned stage = 0; stage < 3; ++stage)
      pipe_resource_reference(&storage[stage], NULL);
   pipe_resource_reference(&occlusion_buffer, NULL);
   simple_mtx_unlock(&ps5_deferred_mutex);
   return handled;
}
#endif

static bool
ps5_lower_default_tess_levels(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const float *levels = data;
   unsigned first, count;
   if (intr->intrinsic == nir_intrinsic_load_tess_level_outer_default) {
      first = 0; count = 4;
   } else if (intr->intrinsic == nir_intrinsic_load_tess_level_inner_default) {
      first = 4; count = 2;
   } else {
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *values[4];
   for (unsigned i = 0; i < count; ++i)
      values[i] = nir_imm_float(b, levels[first + i]);
   nir_def_rewrite_uses(&intr->def, nir_vec(b, values, count));
   nir_instr_remove(&intr->instr);
   return true;
}

static nir_shader *
ps5_default_tcs_nir(uint64_t outputs, uint8_t vertices, const float levels[6])
{
   if (!vertices || vertices > 32)
      return NULL;
   unsigned locations[64], count = 0;
   u_foreach_bit64(slot, outputs)
      locations[count++] = slot;
   nir_shader *nir = nir_create_passthrough_tcs_impl(
      psbc_get_nir_options(PSBC_STAGE_TESS_CTRL), locations, count, vertices);
   if (!nir)
      return NULL;
   nir_lower_system_values(nir);
   nir_shader_intrinsics_pass(nir, ps5_lower_default_tess_levels,
                              nir_metadata_control_flow, (void *)levels);
   nir_lower_io_passes(nir, false);
   nir_remove_dead_variables(nir, nir_var_shader_in | nir_var_shader_out | nir_var_system_value, NULL);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return nir;
}

static bool
ps5_prepare_default_tcs(struct ps5_context *context)
{
   if (!context->vs || !context->base.create_tcs_state)
      return false;
   uint64_t outputs = context->vs->nir->info.outputs_written;
   if (context->default_tcs && context->default_tcs_outputs == outputs &&
       context->default_tcs_vertices == context->patch_vertices &&
       !memcmp(context->default_tcs_levels, context->tess_levels, sizeof(context->tess_levels)))
      return true;
   /* ponytail: cache baked default levels; use a UBO if levels change every draw. */
   struct pipe_shader_state state = {.type = PIPE_SHADER_IR_NIR};
   state.ir.nir = ps5_default_tcs_nir(outputs, context->patch_vertices, context->tess_levels);
   if (!state.ir.nir)
      return false;
   struct ps5_shader *shader = context->base.create_tcs_state(&context->base, &state);
   if (!shader)
      return false;
   context->base.delete_tcs_state(&context->base, context->default_tcs);
   context->default_tcs = shader;
   context->default_tcs_outputs = outputs;
   context->default_tcs_vertices = context->patch_vertices;
   memcpy(context->default_tcs_levels, context->tess_levels, sizeof(context->tess_levels));
   return true;
}

/* Tessellation followed by a geometry shader is drawn in two passes: the
 * tessellated primitives are captured through transform feedback and drawn
 * again as ordinary primitives for the geometry shader. Both halves are
 * pipelines the hardware runs correctly, while the linked pipeline does not
 * at every tessellation level. Measured at level 64 with quads: in each ring
 * whose side has a multiple of six segments, a geometry shader emitting 64
 * vertices per primitive loses the second half of every primitive's output,
 * with hardware instancing (4 x 16) and without (1 x 64) alike. An input
 * primitive whose output exceeds one 128-thread subgroup needs the
 * per-instance subgroup mode, which hangs the GPU behind the tessellator. The
 * linked pipeline remains the fallback for draws that cannot be captured. */
#define PS5_TESS_CAPTURE_MAX_SLOTS 16u
#define PS5_TESS_CAPTURE_MAX_BYTES (UINT64_C(256) << 20)

static uint64_t
ps5_next_shader_serial(void)
{
   static uint64_t serial;

   return __atomic_add_fetch(&serial, 1, __ATOMIC_RELAXED);
}

/* The varying slots a geometry shader reads, in ascending order. They are
 * captured and replayed as whole 32-bit vec4 slots. */
static bool
ps5_tess_capture_slots(const nir_shader *gs, uint8_t *slots, unsigned *count)
{
   uint64_t read = 0;

   *count = 0;
   if (!gs->info.io_lowered)
      return false;
   nir_foreach_function_impl(impl, (nir_shader *)gs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_per_vertex_input)
               continue;
            const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (intr->def.bit_size != 32 || sem.location + sem.num_slots > 64)
               return false;
            read |= BITFIELD64_RANGE(sem.location, sem.num_slots);
         }
      }
   }
   /* The second pass still needs one vertex record per captured vertex. */
   if (!read)
      read = VARYING_BIT_POS;
   u_foreach_bit64(slot, read) {
      if (*count == PS5_TESS_CAPTURE_MAX_SLOTS)
         return false;
      slots[(*count)++] = slot;
   }
   return true;
}

/* The evaluation shader with every slot in "slots" captured to buffer 0, one
 * vec4 per slot. NULL when a slot is not written by direct 32-bit stores. */
static nir_shader *
ps5_tess_capture_tes_nir(const nir_shader *tes, const uint8_t *slots,
                         unsigned count)
{
   nir_shader *nir;
   bool valid = true;
   bool captured = false;

   if (!tes->info.io_lowered || !(nir = nir_shader_clone(NULL, tes)))
      return NULL;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_has_io_xfb(intr))
               continue;
            const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            /* The application's own captures belong to the geometry shader. */
            nir_io_xfb xfb = {0};
            for (unsigned i = 0; i < count; ++i) {
               if (slots[i] < sem.location ||
                   slots[i] >= sem.location + sem.num_slots)
                  continue;
               /* st/mesa lowers evaluation outputs to direct 32-bit stores. */
               if (intr->intrinsic != nir_intrinsic_store_output ||
                   !nir_src_is_const(intr->src[1]) ||
                   intr->src[0].ssa->bit_size != 32 ||
                   nir_intrinsic_component(intr) +
                      intr->src[0].ssa->num_components > 4) {
                  valid = false;
                  continue;
               }
               if (slots[i] != sem.location + nir_src_as_uint(intr->src[1]))
                  continue;
               u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
                  const unsigned component = nir_intrinsic_component(intr) + c;
                  xfb.out[component].num_components = 1;
                  xfb.out[component].buffer = 0;
                  xfb.out[component].offset = i * 4u + component;
               }
               captured = true;
            }
            nir_intrinsic_set_io_xfb(intr, xfb);
         }
      }
   }
   memset(nir->info.xfb_stride, 0, sizeof(nir->info.xfb_stride));
   nir->info.xfb_stride[0] = count * 4u;
   nir->info.has_transform_feedback_varyings = true;
   ralloc_free(nir->xfb_info);
   nir->xfb_info = NULL;
   if (valid && captured)
      nir_gather_xfb_info_from_intrinsics(nir);
   if (!nir->xfb_info) {
      ralloc_free(nir);
      return NULL;
   }
   return nir;
}

/* A vertex shader that replays the captured slots: attribute i feeds slot i. */
static nir_shader *
ps5_tess_capture_vs_nir(const uint8_t *slots, unsigned count)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, psbc_get_nir_options(PSBC_STAGE_VERTEX),
      "ps5-tess-capture-vs");
   bool position = false;

   for (unsigned i = 0; i < count; ++i) {
      const bool scalar = slots[i] == VARYING_SLOT_PSIZ;
      nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
                                             glsl_vec4_type(), NULL);
      nir_variable *out = nir_variable_create(
         b.shader, nir_var_shader_out,
         scalar ? glsl_float_type() : glsl_vec4_type(), NULL);
      in->data.location = VERT_ATTRIB_GENERIC0 + i;
      in->data.driver_location = i;
      out->data.location = slots[i];
      nir_def *value = nir_load_var(&b, in);
      nir_store_var(&b, out, scalar ? nir_channel(&b, value, 0) : value,
                    scalar ? 1 : 15);
      position |= slots[i] == VARYING_SLOT_POS;
   }
   if (!position) {
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                             glsl_vec4_type(), NULL);
      out->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, out, nir_imm_vec4(&b, 0, 0, 0, 1), 15);
   }
   nir_lower_io_passes(b.shader, false);
   nir_remove_dead_variables(b.shader,
                             nir_var_shader_in | nir_var_shader_out, NULL);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

/* The capture pass rasterizes nothing; its fragment shader links to nothing. */
static nir_shader *
ps5_tess_capture_fs_nir(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, psbc_get_nir_options(PSBC_STAGE_FRAGMENT),
      "ps5-tess-capture-fs");

   b.shader->info.io_lowered = true;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

/* Whether a tessellation + geometry draw takes the two-pass path: always,
 * unless PS5_TESS_GS_PATH=direct asks for the linked pipeline (diagnosis). */
static bool
ps5_tess_gs_two_pass(void)
{
   static int direct = -1;

   if (direct < 0) {
      const char *path = getenv("PS5_TESS_GS_PATH");
      direct = path && !strcmp(path, "direct");
   }
   return !direct;
}

static void
ps5_release_tess_capture(struct ps5_context *context)
{
   struct ps5_shader *tes = context->tess_capture_tes;
   struct ps5_shader *vs = context->tess_capture_vs;

   context->tess_capture_tes_serial = 0;
   context->tess_capture_gs_serial = 0;
   context->tess_capture_tes = NULL;
   context->tess_capture_vs = NULL;
   /* One function deletes the shader state of every stage. */
   context->base.delete_vs_state(&context->base, tes);
   context->base.delete_vs_state(&context->base, vs);
   free(context->tess_capture_elements);
   context->tess_capture_elements = NULL;
}

static bool
ps5_prepare_tess_capture(struct ps5_context *context)
{
   struct pipe_context *base = &context->base;
   struct pipe_shader_state state = {.type = PIPE_SHADER_IR_NIR};
   uint8_t slots[PS5_TESS_CAPTURE_MAX_SLOTS];
   struct ps5_vertex_elements *elements;
   struct ps5_shader *tes;
   unsigned count;

   if (context->tess_capture_tes_serial == context->tes->serial &&
       context->tess_capture_gs_serial == context->gs->serial)
      return context->tess_capture_tes != NULL;
   ps5_release_tess_capture(context);
   context->tess_capture_tes_serial = context->tes->serial;
   context->tess_capture_gs_serial = context->gs->serial;
   if (!context->tess_capture_fs) {
      state.ir.nir = ps5_tess_capture_fs_nir();
      context->tess_capture_fs = base->create_fs_state(base, &state);
   }
   if (!context->tess_capture_fs ||
       !ps5_tess_capture_slots(context->gs->nir, slots, &count))
      return false;
   tes = calloc(1, sizeof(*tes));
   elements = calloc(1, sizeof(*elements));
   if (tes)
      tes->nir = ps5_tess_capture_tes_nir(context->tes->nir, slots, count);
   state.ir.nir = tes && tes->nir && elements
      ? ps5_tess_capture_vs_nir(slots, count) : NULL;
   context->tess_capture_vs = state.ir.nir
      ? base->create_vs_state(base, &state) : NULL;
   if (!context->tess_capture_vs) {
      printf("[ps5-gallium] tess-capture unavailable slots=%u\n", count);
      if (tes)
         ralloc_free(tes->nir);
      free(tes);
      free(elements);
      return false;
   }
   tes->stage = PSBC_STAGE_TESS_EVAL;
   tes->serial = ps5_next_shader_serial();
   tes->stream_output.num_outputs = count;
   tes->stream_output.stride[0] = count * 4u;
   elements->count = count;
   for (unsigned i = 0; i < count; ++i) {
      tes->stream_output.output[i] = (struct pipe_stream_output){
         .register_index = i, .num_components = 4, .dst_offset = i * 4u,
      };
      elements->elements[i] = (struct pipe_vertex_element){
         .src_offset = i * 16u, .src_stride = count * 16u,
         .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      };
   }
   context->tess_capture_tes = tes;
   context->tess_capture_elements = elements;
   printf("[ps5-gallium] tess-capture ready slots=%u\n", count);
   return true;
}

#ifndef PS5_KLOG
#define PS5_KLOG(...) do { \
      int sceKernelDebugOutText(int channel, const char *text) __attribute__((weak)); \
      static unsigned ps5_klog_reported; \
      char ps5_klog_message[256]; \
      if (sceKernelDebugOutText && ps5_klog_reported++ < 32) { \
         snprintf(ps5_klog_message, sizeof(ps5_klog_message), __VA_ARGS__); \
         sceKernelDebugOutText(0, ps5_klog_message); \
      } \
   } while (0)
#define PS5_COMPUTE_REJECT() do { \
      PS5_KLOG("[ps5-gallium] compute dispatch dropped at ps5_screen.c:%u\n", __LINE__); \
      return; \
   } while (0)
#endif


static void
ps5_draw_vbo(struct pipe_context *base, const struct pipe_draw_info *info,
             unsigned drawid_offset,
             const struct pipe_draw_indirect_info *indirect,
             const struct pipe_draw_start_count_bias *draws,
             unsigned num_draws);

/* False when the draw was not taken over. */
static bool
ps5_draw_tess_gs_two_pass(struct pipe_context *base,
                          const struct pipe_draw_info *info,
                          unsigned drawid_offset,
                          const struct pipe_draw_start_count_bias *draw)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_shader *const vs = context->vs, *const tcs = context->tcs;
   struct ps5_shader *const tes = context->tes, *const gs = context->gs;
   struct ps5_shader *const fs = context->fs;
   struct pipe_rasterizer_state *const rasterizer = context->rasterizer;
   const enum mesa_prim primitive = gs->nir->info.gs.input_primitive;
   const unsigned vertices_per_primitive =
      ps5_streamout_vertices_per_primitive(primitive);
   const unsigned patches = context->patch_vertices
      ? draw->count / context->patch_vertices : 0;
   struct pipe_stream_output_target *saved_targets[PIPE_MAX_SO_BUFFERS] = {0};
   const unsigned saved_target_count = context->stream_output_target_count;
   const enum mesa_prim saved_primitive = context->stream_output_primitive;
   const bool queries_enabled = context->queries_enabled;
   struct pipe_stream_output_target *target;
   struct pipe_rasterizer_state discard;
   unsigned offsets[PIPE_MAX_SO_BUFFERS] = {0};
   unsigned captured;
   int capture_status;
   uint64_t bytes;

   if (!vertices_per_primitive || !rasterizer || !fs ||
       !ps5_prepare_tess_capture(context))
      return false;
   if (info->instance_count > 1) {
      /* One capture per instance. The instance moves into the base instance,
       * which only vertex-buffer divisors may observe. */
      const struct ps5_shader *const stages[] = {vs, tcs, tes};
      for (unsigned stage = 0; stage < ARRAY_SIZE(stages); ++stage)
         if (BITSET_TEST(stages[stage]->nir->info.system_values_read,
                         SYSTEM_VALUE_INSTANCE_ID) ||
             BITSET_TEST(stages[stage]->nir->info.system_values_read,
                         SYSTEM_VALUE_BASE_INSTANCE))
            return false;
      if ((uint64_t)info->start_instance + info->instance_count - 1u > UINT32_MAX)
         return false;
      for (unsigned instance = 0; instance < info->instance_count; ++instance) {
         struct pipe_draw_info single = *info;

         single.instance_count = 1;
         single.start_instance = info->start_instance + instance;
         if (!ps5_draw_tess_gs_two_pass(base, &single, drawid_offset, draw))
            return instance != 0;
         if (context->last_draw_status != 0)
            break;
      }
      return true;
   }
   if (!patches) {
      context->last_draw_status = 0;
      return true;
   }
   /* A quad patch at the maximum level yields 2 * 64 * 64 triangles. */
   bytes = (uint64_t)patches * 8192u * vertices_per_primitive *
           context->tess_capture_elements->count * 16u;
   if (bytes > PS5_TESS_CAPTURE_MAX_BYTES ||
       !ps5_streamout_storage(context, &context->tess_capture_buffer, bytes))
      return false;
   target = base->create_stream_output_target(
      base, context->tess_capture_buffer, 0, (unsigned)bytes);
   if (!target)
      return false;

   for (unsigned i = 0; i < saved_target_count; ++i)
      pipe_so_target_reference(&saved_targets[i],
                               context->stream_output_targets[i]);
   base->set_stream_output_targets(base, 1, &target, offsets, primitive);
   discard = *rasterizer;
   discard.rasterizer_discard = 1;
   context->rasterizer = &discard;
   context->tes = context->tess_capture_tes;
   context->gs = NULL;
   context->fs = context->tess_capture_fs;
   /* The application's queries count the geometry shader's primitives. */
   context->queries_enabled = false;
   context->tess_capture_active = true;
   context->last_draw_status = 0;
   ps5_draw_vbo(base, info, drawid_offset, NULL, draw, 1);
   capture_status = context->last_draw_status;
   captured = ((struct ps5_stream_output_target *)target)->vertex_count;
   context->queries_enabled = queries_enabled;
   context->fs = fs;
   context->gs = gs;
   context->tes = tes;
   context->rasterizer = rasterizer;
   memset(offsets, 0xff, sizeof(offsets)); /* append: keep the positions */
   base->set_stream_output_targets(base, saved_target_count, saved_targets,
                                   offsets, saved_primitive);
   for (unsigned i = 0; i < saved_target_count; ++i)
      pipe_so_target_reference(&saved_targets[i], NULL);
   pipe_so_target_reference(&target, NULL);
   printf("[ps5-gallium] tess-capture status=%d patches=%u vertices=%u\n",
          capture_status, patches, captured);

   if (capture_status == 0 && captured) {
      struct pipe_vertex_buffer saved_buffers[PIPE_MAX_ATTRIBS];
      const unsigned saved_buffer_count = context->vertex_buffer_count;
      struct ps5_vertex_elements *const elements = context->vertex_elements;
      const struct pipe_vertex_buffer buffer = {
         .buffer.resource = context->tess_capture_buffer,
      };
      const struct pipe_draw_info replay = {
         .mode = primitive, .instance_count = 1,
      };
      const struct pipe_draw_start_count_bias replay_draw = {
         .count = captured,
      };

      for (unsigned i = 0; i < saved_buffer_count; ++i) {
         saved_buffers[i] = context->vertex_buffers[i];
         saved_buffers[i].buffer.resource = NULL;
         pipe_resource_reference(&saved_buffers[i].buffer.resource,
                                 context->vertex_buffers[i].buffer.resource);
      }
      base->set_vertex_buffers(base, 1, &buffer);
      context->vertex_elements = context->tess_capture_elements;
      context->vs = context->tess_capture_vs;
      context->tcs = NULL;
      context->tes = NULL;
      ps5_draw_vbo(base, &replay, drawid_offset, NULL, &replay_draw, 1);
      /* The next two-pass draw rewrites the capture buffer. */
      ps5_draw_batch_drain();
      context->tes = tes;
      context->tcs = tcs;
      context->vs = vs;
      context->vertex_elements = elements;
      base->set_vertex_buffers(base, saved_buffer_count, saved_buffers);
      for (unsigned i = 0; i < saved_buffer_count; ++i)
         pipe_resource_reference(&saved_buffers[i].buffer.resource, NULL);
   }
   context->tess_capture_active = false;
   return capture_status == 0;
}

/* A rejected draw renders nothing; report the status of the first ones. */
static void
ps5_draw_vbo_reported(struct pipe_context *base, const struct pipe_draw_info *info,
                      unsigned drawid_offset,
                      const struct pipe_draw_indirect_info *indirect,
                      const struct pipe_draw_start_count_bias *draws,
                      unsigned num_draws)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_sampler_view *fbfetch_saved = NULL;
   void *fbfetch_saved_sampler = NULL;
   const unsigned fbfetch_unit = context->fs ? context->fs->fbfetch_unit : 0;
   context->last_draw_status = 0;
   if (fbfetch_unit) {
      const struct pipe_surface *color = &context->framebuffer.cbufs[0];
      struct pipe_sampler_view *view = NULL;
      if (color->texture) {
         struct pipe_sampler_view templ;
         u_sampler_view_default_template(&templ, color->texture, color->format);
         templ.target = PIPE_TEXTURE_2D;
         templ.u.tex.first_level = templ.u.tex.last_level = color->level;
         templ.u.tex.first_layer = templ.u.tex.last_layer = color->first_layer;
         view = base->create_sampler_view(base, color->texture, &templ);
      }
      pipe_sampler_view_reference(&fbfetch_saved,
                                  context->sampler_views[1][fbfetch_unit - 1]);
      base->set_sampler_views(base, MESA_SHADER_FRAGMENT, fbfetch_unit - 1, 1, 0,
                              &view);
      pipe_sampler_view_reference(&view, NULL);
      /* Texture binding requires a sampler per unit, even for texel fetches. */
      if (!context->fbfetch_sampler) {
         const struct pipe_sampler_state nearest = {
            .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE, .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
            .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
            .min_img_filter = PIPE_TEX_FILTER_NEAREST, .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
            .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
         };
         context->fbfetch_sampler = base->create_sampler_state(base, &nearest);
      }
      fbfetch_saved_sampler = context->samplers[1][fbfetch_unit - 1];
      base->bind_sampler_states(base, MESA_SHADER_FRAGMENT, fbfetch_unit - 1, 1,
                                &context->fbfetch_sampler);
   }
   ps5_draw_vbo(base, info, drawid_offset, indirect, draws, num_draws);
   if (fbfetch_unit) {
      base->set_sampler_views(base, MESA_SHADER_FRAGMENT, fbfetch_unit - 1, 1, 0,
                              &fbfetch_saved);
      pipe_sampler_view_reference(&fbfetch_saved, NULL);
      base->bind_sampler_states(base, MESA_SHADER_FRAGMENT, fbfetch_unit - 1, 1,
                                &fbfetch_saved_sampler);
   }
   if (context->last_draw_status < 0)
      PS5_KLOG("[ps5-gallium] draw dropped status=%d mode=%u\n",
                context->last_draw_status, info ? (unsigned)info->mode : 0u);
}

static void
ps5_draw_vbo(struct pipe_context *base, const struct pipe_draw_info *info,
             unsigned drawid_offset,
             const struct pipe_draw_indirect_info *indirect,
             const struct pipe_draw_start_count_bias *draws,
             unsigned num_draws)
{
   struct ps5_screen *screen = (struct ps5_screen *)base->screen;
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_resource *uploaded_indices = NULL;
   struct pipe_draw_info uploaded_info;
   struct pipe_draw_start_count_bias uploaded_draw;
   unsigned uploaded_offset;

   if (info && info->mode == MESA_PRIM_PATCHES && context->tes && !context->tcs) {
      if (!ps5_prepare_default_tcs(context)) {
         context->last_draw_status = -19;
         return;
      }
      context->tcs = context->default_tcs;
      ps5_draw_vbo(base, info, drawid_offset, indirect, draws, num_draws);
      context->tcs = NULL;
      return;
   }

   if (PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE && info && !indirect && draws &&
       num_draws == 1 && info->mode == MESA_PRIM_PATCHES &&
       !info->primitive_restart && context->tes && context->gs &&
       !context->tess_capture_active && ps5_tess_gs_two_pass()) {
      if (ps5_draw_tess_gs_two_pass(base, info, drawid_offset, &draws[0]))
         return;
      /* Not captured: the linked pipeline draws what it can. */
      context->last_draw_status = 0;
   }

   if (info && (info->mode == MESA_PRIM_QUADS ||
                info->mode == MESA_PRIM_QUAD_STRIP ||
                info->mode == MESA_PRIM_POLYGON)) {
      struct primconvert_config cfg = {
         .primtypes_mask = base->screen->caps.supported_prim_modes &
            ~(BITFIELD_BIT(MESA_PRIM_QUADS) |
              BITFIELD_BIT(MESA_PRIM_QUAD_STRIP) |
              BITFIELD_BIT(MESA_PRIM_POLYGON)),
         .restart_primtypes_mask = 0,
         .split_legacy_triangles = true,
      };
      struct primconvert_context *converter =
         util_primconvert_create_config(base, &cfg);
      if (!converter) {
         context->last_draw_status = -2;
         return;
      }
      util_primconvert_save_flatshade_first(
         converter, context->rasterizer && context->rasterizer->flatshade_first);
      context->legacy_primitive_conversion = true;
      context->legacy_first_index = UINT_MAX;
      util_primconvert_draw_vbo(converter, info, drawid_offset, indirect,
                                draws, num_draws);
      context->legacy_primitive_conversion = false;
      context->legacy_flat_input_vertex = -1;
      util_primconvert_destroy(converter);
      return;
   }

   if (indirect) {
      if (!PS5_ENABLE_DRAW_INDIRECT_CANDIDATE || !info) {
         context->last_draw_status = -2;
         return;
      }
      if (indirect->count_from_stream_output) {
         struct ps5_stream_output_target *target =
            (struct ps5_stream_output_target *)
               indirect->count_from_stream_output;
         struct pipe_draw_start_count_bias draw = {
            .count = target->vertex_count,
         };

         if (target->base.context != base) {
            context->last_draw_status = -2;
            return;
         }
         ps5_draw_vbo(base, info, drawid_offset, NULL, &draw, 1);
         return;
      }
      util_draw_indirect(base, info, drawid_offset, indirect);
      return;
   }

   if (context->legacy_primitive_conversion && info && draws &&
       num_draws == 1 && info->mode == MESA_PRIM_TRIANGLES &&
       draws[0].count == 3) {
      if (context->legacy_first_index == UINT_MAX)
         context->legacy_first_index = draws[0].start;
      const unsigned triangle =
         (draws[0].start - context->legacy_first_index) / 3u;
      context->legacy_flat_input_vertex = (triangle & 1u) ? 1 : 2;
   }

#ifdef PS5_DEFERRED_DRAW_BATCH
   if (!context->legacy_primitive_conversion &&
       ps5_try_deferred_draw(base, info, drawid_offset, indirect, draws, num_draws))
      return;
#endif
#ifdef PS5_MULTIDRAW_BATCH
   if (num_draws > 1 && ps5_try_multi_draw_batch(
          base, info, drawid_offset, indirect, draws, num_draws))
      return;
#endif
   if (num_draws > 1) {
      util_draw_multi(base, info, drawid_offset, indirect, draws, num_draws);
      return;
   }
   if (info && info->primitive_restart) {
      /* One rewritten draw preserves PrimitiveID across restart boundaries.
       * Mesa converts each strip separately before concatenating its indices. */
      struct primconvert_config cfg = {
         .primtypes_mask = base->screen->caps.supported_prim_modes,
         .restart_primtypes_mask = 0,
      };
      struct primconvert_context *converter = util_primconvert_create_config(base, &cfg);
      if (!converter) {
         context->last_draw_status = -2;
         return;
      }
      util_primconvert_save_flatshade_first(converter,
         context->rasterizer && context->rasterizer->flatshade_first);
      context->last_draw_status = 0;
      util_primconvert_draw_vbo(converter, info, drawid_offset, indirect, draws, num_draws);
      util_primconvert_destroy(converter);
      return;
   }
   if (info && draws && !indirect && info->index_size &&
       info->has_user_indices) {
      if (!util_upload_index_buffer(base, info, &draws[0],
                                    &uploaded_indices, &uploaded_offset, 4)) {
         context->last_draw_status = -2;
         printf("[ps5-gallium] user-index-upload-failed\n");
         return;
      }
      uploaded_info = *info;
      uploaded_info.index.resource = uploaded_indices;
      uploaded_info.has_user_indices = false;
      uploaded_draw = draws[0];
      uploaded_draw.start += uploaded_offset / info->index_size;
      info = &uploaded_info;
      draws = &uploaded_draw;
   }
   if (info && draws && !indirect && !((struct ps5_context *)base)->gs &&
       (info->mode == MESA_PRIM_LINES_ADJACENCY ||
       info->mode == MESA_PRIM_LINE_STRIP_ADJACENCY ||
       info->mode == MESA_PRIM_TRIANGLES_ADJACENCY ||
       info->mode == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY)) {
      ps5_draw_vbo_without_adjacency(base, info, drawid_offset, draws);
      pipe_resource_reference(&uploaded_indices, NULL);
      return;
   }
   /* ponytail: one hardware queue lock; split per queue if parallel submit
    * becomes measurable and the runtime stops using process-global setters. */
   ps5_screen_submit_lock(&screen->base);
   ps5_draw_vbo_locked(base, info, drawid_offset, indirect, draws, num_draws,
                       NULL, NULL);
#ifndef PS5_RUNTIME_QUIET
   if (context->tcs && context->tes)
      printf("[ps5-gallium] tessellation-draw status=%d hs=%08x final=%08x\n",
             context->last_draw_status,
             ps5_hash32(context->tessellation_output.hs.machine_code,
                        context->tessellation_output.hs.machine_code_size),
             ps5_hash32(context->tessellation_output.tes.machine_code,
                        context->tessellation_output.tes.machine_code_size));
#endif
   /* ponytail: storage draws retire synchronously; batching needs retained
    * storage resources and explicit shader-write visibility first. */
   if (!context->last_draw_status)
      for (unsigned i = 0; i < MIN2(ps5_shader_storage_count(context->fs), PS5_COMPUTE_STORAGE_SLOTS); ++i) {
         const struct pipe_shader_buffer *bound = &context->fragment_buffers[i];
         const struct ps5_resource *resource = (const struct ps5_resource *)bound->buffer;
         if (resource)
            ps5_flush_gpu_data(resource->data + bound->buffer_offset, bound->buffer_size);
      }
   if (!context->last_draw_status)
      for (unsigned i = 0;
           i < MIN2(ps5_shader_storage_count(context->gs),
                    PS5_COMPUTE_STORAGE_SLOTS); ++i) {
         const struct pipe_shader_buffer *bound = &context->geometry_buffers[i];
         const struct ps5_resource *resource =
            (const struct ps5_resource *)bound->buffer;
         if (resource)
            ps5_flush_gpu_data(resource->data + bound->buffer_offset,
                               bound->buffer_size);
      }
   if (!context->last_draw_status && context->tcs && context->tes) {
      const struct ps5_shader *stages[] = {context->vs, context->tcs, context->tes};
      for (unsigned stage = 0; stage < 3; ++stage)
         for (unsigned i = 0;
              i < MIN2(ps5_shader_storage_count(stages[stage]), PS5_COMPUTE_STORAGE_SLOTS); ++i) {
            const struct pipe_shader_buffer *bound = &context->preraster_buffers[stage][i];
            const struct ps5_resource *resource = (const struct ps5_resource *)bound->buffer;
            if (resource)
               ps5_flush_gpu_data(resource->data + bound->buffer_offset, bound->buffer_size);
         }
   }
   if (!context->last_draw_status && context->fs)
      for (unsigned i = 0; i < MIN2(context->fs->nir->info.num_images, PS5_COMPUTE_IMAGE_SLOTS); ++i) {
         const struct ps5_resource *resource = (const struct ps5_resource *)context->fragment_images[i].resource;
         if (resource)
            ps5_flush_gpu_data(resource->data, resource->size);
      }
   ps5_screen_submit_unlock(&screen->base);
   pipe_resource_reference(&uploaded_indices, NULL);
   if (context->last_draw_status != 0)
      printf("[ps5-gallium] draw-rejected status=%d mode=%u count=%u index=%u streamout=%u geometry=%u\n",
             context->last_draw_status, info ? info->mode : UINT32_MAX,
             draws && num_draws ? draws[0].count : 0,
             info ? info->index_size : 0,
             context->stream_output_target_count, context->gs != NULL);
}

static void
ps5_set_framebuffer_state(struct pipe_context *base,
                          const struct pipe_framebuffer_state *framebuffer)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned max_targets = PS5_ENABLE_MRT_CANDIDATE
                             ? PS5_MAX_RENDER_TARGETS : 1;
   bool colors_valid = framebuffer && framebuffer->nr_cbufs <= max_targets;
   bool has_color = false;
   bool has_depth = framebuffer && framebuffer->zsbuf.texture;
   unsigned color_samples = 0;

   if (colors_valid) {
      for (unsigned i = 0; i < framebuffer->nr_cbufs; ++i) {
         const struct pipe_surface *surface = &framebuffer->cbufs[i];

         if (!surface->texture)
            continue;
         has_color = true;
         if (!color_samples)
            color_samples = MAX2(surface->texture->nr_samples, 1);
         else if (color_samples != MAX2(surface->texture->nr_samples, 1))
            colors_valid = false;
         bool single_layer = surface->first_layer == surface->last_layer;

         colors_valid = colors_valid &&
            surface->level <= surface->texture->last_level &&
            ((((PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
                surface->texture->target == PIPE_TEXTURE_1D) ||
               surface->texture->target == PIPE_TEXTURE_2D ||
               (PS5_ENABLE_TEXTURE_RECTANGLE_CANDIDATE &&
                surface->texture->target == PIPE_TEXTURE_RECT)) &&
              surface->first_layer == 0 && single_layer) ||
             (PS5_ENABLE_LAYERED_RENDER_TARGET_CANDIDATE &&
              ((PS5_ENABLE_TEXTURE_1D_CANDIDATE &&
                surface->texture->target == PIPE_TEXTURE_1D_ARRAY) ||
               surface->texture->target == PIPE_TEXTURE_2D_ARRAY ||
               ps5_cube_texture_target(surface->texture->target) ||
               surface->texture->target == PIPE_TEXTURE_3D) &&
              surface->first_layer <= surface->last_layer &&
              surface->last_layer < ps5_surface_layer_count(surface))) &&
            (!ps5_linear_sampled_layout(surface->texture) ||
             (((struct ps5_resource *)surface->texture)
                 ->render_staging_size)) &&
            ps5_render_target_format(surface->format) &&
            ((PS5_ENABLE_PADDED_FBO_CANDIDATE ||
              PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE)
                ? (ps5_surface_width(surface) >= framebuffer->width &&
                   ps5_surface_height(surface) >= framebuffer->height &&
                   ps5_surface_width(surface) <= PS5_MAX_COLOR_WIDTH &&
                   ps5_surface_height(surface) <= PS5_MAX_COLOR_HEIGHT)
                : (ps5_surface_width(surface) == PS5_RENDER_WIDTH &&
                   ps5_surface_height(surface) == PS5_RENDER_HEIGHT));
         if (!colors_valid)
            break;
      }
   }

   /* A fixed-framebuffer batch publishes color/depth at its group tail. Close
    * that producer before changing attachments, so later sampling stays ordered. */
   if (!framebuffer || !util_framebuffer_state_equal(&context->framebuffer, framebuffer))
      ps5_draw_batch_submit();
   util_copy_framebuffer_state(&context->framebuffer, framebuffer);
   context->framebuffer_valid = framebuffer && colors_valid &&
      (has_color || has_depth ||
       (PS5_ENABLE_GLSL_430_CANDIDATE && !framebuffer->nr_cbufs)) &&
      ((PS5_ENABLE_PADDED_FBO_CANDIDATE ||
        PS5_ENABLE_DYNAMIC_COLOR_TARGET_CANDIDATE)
          ? (framebuffer->width > 0 && framebuffer->height > 0 &&
             framebuffer->width <= PS5_MAX_COLOR_WIDTH &&
             framebuffer->height <= PS5_MAX_COLOR_HEIGHT)
          : (framebuffer->width == PS5_RENDER_WIDTH &&
             framebuffer->height == PS5_RENDER_HEIGHT)) &&
      (!framebuffer->zsbuf.texture ||
        ((!has_color || MAX2(framebuffer->zsbuf.texture->nr_samples, 1) ==
                         color_samples) &&
        (framebuffer->zsbuf.format == PIPE_FORMAT_Z32_FLOAT ||
         framebuffer->zsbuf.format ==
            PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) &&
        ps5_depth_render_target(framebuffer->zsbuf.texture->target) &&
        (PS5_ENABLE_PADDED_FBO_CANDIDATE
            ? (ps5_surface_width(&framebuffer->zsbuf) >=
                  framebuffer->width &&
               ps5_surface_height(&framebuffer->zsbuf) >=
                  framebuffer->height &&
               ps5_surface_width(&framebuffer->zsbuf) <=
                  PS5_MAX_DEPTH_WIDTH &&
               ps5_surface_height(&framebuffer->zsbuf) <=
                  PS5_MAX_DEPTH_HEIGHT)
            : (ps5_surface_width(&framebuffer->zsbuf) ==
                  PS5_RENDER_WIDTH &&
               ps5_surface_height(&framebuffer->zsbuf) ==
                  PS5_RENDER_HEIGHT)) &&
        framebuffer->zsbuf.level <=
           framebuffer->zsbuf.texture->last_level &&
        framebuffer->zsbuf.first_layer <=
           framebuffer->zsbuf.last_layer &&
        framebuffer->zsbuf.last_layer <
           ps5_surface_layer_count(&framebuffer->zsbuf)));
   if (PS5_PUBLIC_TEXTURE_RG_RENDER_TEST)
      printf("[ps5-gallium] rg-framebuffer size=%ux%u targets=%u format=%u "
             "colors-valid=%u has-color=%u valid=%u\n",
             framebuffer ? framebuffer->width : 0,
             framebuffer ? framebuffer->height : 0,
             framebuffer ? framebuffer->nr_cbufs : 0,
             framebuffer && framebuffer->nr_cbufs
                ? framebuffer->cbufs[0].format : 0,
             colors_valid, has_color, context->framebuffer_valid);
}

static void
ps5_clear_bounds(const struct pipe_scissor_state *scissor,
                 unsigned width, unsigned height,
                 unsigned *min_x, unsigned *min_y,
                 unsigned *max_x, unsigned *max_y)
{
   *min_x = scissor ? MIN2(scissor->minx, width) : 0;
   *min_y = scissor ? MIN2(scissor->miny, height) : 0;
   *max_x = scissor ? MIN2(scissor->maxx, width) : width;
   *max_y = scissor ? MIN2(scissor->maxy, height) : height;
}

static bool
ps5_clear_msaa4_color(struct ps5_context *context, unsigned buffers,
                      uint32_t color_clear_mask,
                      const struct pipe_scissor_state *scissor_state,
                      const union pipe_color_union *color)
{
   unsigned color_buffers = buffers & PIPE_CLEAR_COLOR;

   if (!color_buffers || !color)
      return false;
   for (unsigned target_index = 0;
        target_index < context->framebuffer.nr_cbufs; ++target_index) {
      unsigned bit = PIPE_CLEAR_COLOR0 << target_index;
      const struct pipe_surface *surface =
         &context->framebuffer.cbufs[target_index];

      if (!(color_buffers & bit))
         continue;
      color_buffers &= ~bit;
      if (!surface->texture || surface->texture->nr_samples != 4 ||
          surface->texture->nr_storage_samples != 4 ||
          !ps5_msaa4_color_format(surface->format))
         return false;
   }
   if (color_buffers)
      return false;

   for (unsigned target_index = 0;
        target_index < context->framebuffer.nr_cbufs; ++target_index) {
      unsigned bit = PIPE_CLEAR_COLOR0 << target_index;
      unsigned write_mask =
         (color_clear_mask >> (4u * target_index)) & UINT32_C(0xf);
      const struct pipe_surface *surface =
         &context->framebuffer.cbufs[target_index];
      struct ps5_resource *target;
      unsigned width;
      unsigned height;
      unsigned min_x, min_y, max_x, max_y;
      unsigned pixel_size;
      uint8_t packed[16];

      if (!(buffers & bit))
         continue;
      target = (struct ps5_resource *)surface->texture;
      width = ps5_surface_width(surface);
      height = ps5_surface_height(surface);
      pixel_size = ps5_texture_format_size(surface->format);
      if (!pixel_size || pixel_size > sizeof(packed))
         return false;
      ps5_clear_bounds(scissor_state, width, height,
                       &min_x, &min_y, &max_x, &max_y);
      util_format_pack_rgba(surface->format, packed, color->ui, 1);
      for (unsigned layer = surface->first_layer;
           layer <= surface->last_layer; ++layer) {
         size_t layer_base = (size_t)layer * target->layer_stride;

         for (unsigned y = min_y; y < max_y; ++y) {
            for (unsigned x = min_x; x < max_x; ++x) {
               for (unsigned sample = 0; sample < 4; ++sample) {
                  size_t offset = layer_base + ps5_tiled_color_msaa4_offset(
                     surface->format, x, y, sample, width, layer);

                  if (offset > target->allocation_size ||
                      target->allocation_size - offset < pixel_size)
                     return false;
                  if (write_mask == PIPE_MASK_RGBA) {
                     memcpy(target->data + offset, packed, pixel_size);
                  } else {
                     union pipe_color_union merged = {{0}};

                     util_format_unpack_rgba(surface->format, merged.ui,
                                             target->data + offset, 1);
                     for (unsigned channel = 0; channel < 4; ++channel) {
                        if (write_mask & BITFIELD_BIT(channel))
                           merged.ui[channel] = color->ui[channel];
                     }
                     util_format_pack_rgba(surface->format,
                                           target->data + offset,
                                           merged.ui, 1);
                  }
               }
            }
         }
      }
      ps5_flush_gpu_data(target->data, target->allocation_size);
   }
   printf("[ps5-gallium] clear-msaa4-color targets=%u mask=%08x scissor=%u\n",
          context->framebuffer.nr_cbufs, color_clear_mask,
          scissor_state != NULL);
   return true;
}

static bool
ps5_clear_gpu_color(struct ps5_context *context, unsigned buffers,
                    uint32_t color_clear_mask,
                    const struct pipe_scissor_state *scissor_state,
                    const union pipe_color_union *color)
{
   /* Clear each validated array layer with the existing single-layer blitter.
    * Masks and incompatible layouts keep their checked CPU path. */
   if (!PS5_ENABLE_MRT_CANDIDATE || !PS5_ENABLE_UBO_CANDIDATE || !context ||
       !context->framebuffer_valid || !color ||
       (buffers & PIPE_CLEAR_COLOR) != PIPE_CLEAR_COLOR0 ||
       (color_clear_mask & PIPE_MASK_RGBA) != PIPE_MASK_RGBA ||
       context->framebuffer.nr_cbufs != 1 || context->render_condition_query ||
       context->stream_output_target_count)
      return false;

   /* CPU fallback drains users and publishes the whole attachment, even for a
    * one-pixel scissor. Base the cutoff on that cost, not the rectangle area. */
   unsigned left, bottom, right, top;
   ps5_clear_bounds(scissor_state, context->framebuffer.width, context->framebuffer.height,
                     &left, &bottom, &right, &top);
   if (right <= left || top <= bottom ||
       (uint64_t)context->framebuffer.width * context->framebuffer.height <
       PS5_GPU_CLEAR_MIN_PIXELS)
      return false;

   const struct pipe_surface *surface = &context->framebuffer.cbufs[0];
   const struct ps5_resource *target = (const struct ps5_resource *)surface->texture;
   /* RGBA8 linear/sRGB views share the same native tiled byte layout. The
    * surface format still controls clear quantization and GPU color encoding. */
   const bool native_rgba8_view = target && !target->render_staging_size &&
      !ps5_linear_sampled_layout(&target->base) &&
      (surface->format == PIPE_FORMAT_R8G8B8A8_UNORM ||
       surface->format == PIPE_FORMAT_R8G8B8A8_SRGB) &&
      (target->base.format == PIPE_FORMAT_R8G8B8A8_UNORM ||
       target->base.format == PIPE_FORMAT_R8G8B8A8_SRGB);
   if (!target || (target->base.target != PIPE_TEXTURE_2D &&
                   target->base.target != PIPE_TEXTURE_2D_ARRAY) ||
       ((target->base.nr_samples > 1 || target->base.nr_storage_samples > 1) &&
        !(PS5_ENABLE_MSAA4_CANDIDATE && target->base.nr_samples == 4 &&
          target->base.nr_storage_samples == 4 && !target->render_staging_size &&
          ps5_msaa4_color_format(surface->format))) ||
       surface->level || surface->first_layer > surface->last_layer ||
       surface->last_layer >= ps5_surface_layer_count(surface) || surface->last_layer > 2047 ||
       (surface->format != PIPE_FORMAT_R8G8B8A8_UNORM &&
        surface->format != PIPE_FORMAT_R8G8B8A8_SNORM &&
        surface->format != PIPE_FORMAT_B8G8R8A8_UNORM && surface->format != PIPE_FORMAT_R8_UNORM &&
        surface->format != PIPE_FORMAT_R8G8_UNORM && surface->format != PIPE_FORMAT_R16G16B16A16_FLOAT &&
        surface->format != PIPE_FORMAT_R11G11B10_FLOAT && surface->format != PIPE_FORMAT_R8G8B8A8_SRGB &&
        surface->format != PIPE_FORMAT_R10G10B10A2_UNORM && surface->format != PIPE_FORMAT_R16_FLOAT &&
        surface->format != PIPE_FORMAT_R32_FLOAT && surface->format != PIPE_FORMAT_R16G16_FLOAT &&
        surface->format != PIPE_FORMAT_R32G32B32A32_UINT) ||
       context->framebuffer.width != ps5_surface_width(surface) ||
       context->framebuffer.height != ps5_surface_height(surface))
      return false;
   for (unsigned layer = surface->first_layer; layer <= surface->last_layer; ++layer) {
      struct pipe_surface selected = *surface;
      selected.first_layer = selected.last_layer = layer;
      if ((target->render_staging_size ||
           (target->base.format != surface->format && !native_rgba8_view)) &&
          !ps5_linear_color_pitch(&selected))
         return false;
   }

   /* Slot zero may be an inline uniform copy, not a resource. Preserve its
    * bytes before u_blitter temporarily replaces it with the clear color. */
   const struct ps5_constant_state *state = &context->constants[1][0];
   uint8_t *copied_constants = NULL;
   struct pipe_constant_buffer cb = {0};
   if (state->valid) {
      cb.buffer = state->buffer;
      cb.buffer_offset = state->offset;
      cb.buffer_size = state->size;
      if (state->copied) {
         const struct ps5_resource *storage =
            (const struct ps5_resource *)context->descriptor_storage[1];
         size_t offset = ps5_copied_constant_offset(1);
         if (!state->size || state->size > PS5_MAX_CONSTANT_BUFFER_SIZE ||
             !storage || offset > storage->size ||
             state->size > storage->size - offset)
            return false;
         copied_constants = malloc(state->size);
         if (!copied_constants)
            return false;
         memcpy(copied_constants, storage->data + offset, state->size);
         cb.user_buffer = copied_constants;
      }
   }
   if (!context->blitter)
      context->blitter = util_blitter_create(&context->base);
   struct blitter_context *blitter = context->blitter;
   if (!blitter || blitter->running) {
      free(copied_constants);
      return false;
   }

   const unsigned first_layer = surface->first_layer, last_layer = surface->last_layer;
   const bool selected_target = scissor_state || first_layer || last_layer;
   for (unsigned layer = first_layer; layer <= last_layer; ++layer) {
      uint16_t viewport_valid = context->viewport_valid;
      bool framebuffer_valid = context->framebuffer_valid;
      bool queries_enabled = context->queries_enabled;
      unsigned draws_before = context->draw_calls;
      util_blitter_save_vertex_buffers(blitter, context->vertex_buffers, context->vertex_buffer_count);
      util_blitter_save_vertex_elements(blitter, context->vertex_elements);
      util_blitter_save_vertex_shader(blitter, context->vs);
      util_blitter_save_tessctrl_shader(blitter, context->tcs);
      util_blitter_save_tesseval_shader(blitter, context->tes);
      util_blitter_save_geometry_shader(blitter, context->gs);
      util_blitter_save_so_targets(blitter, 0, NULL, context->stream_output_primitive);
      util_blitter_save_rasterizer(blitter, context->rasterizer);
      util_blitter_save_fragment_shader(blitter, context->fs);
      util_blitter_save_depth_stencil_alpha(blitter, context->depth_stencil_alpha);
      util_blitter_save_blend(blitter, context->blend);
      util_blitter_save_stencil_ref(blitter, &context->stencil_ref);
      util_blitter_save_viewport(blitter, &context->viewport[0]);
      util_blitter_save_sample_mask(blitter, context->sample_mask, 1);
      util_blitter_save_fragment_constant_buffer_slot(blitter, &cb);
      if (selected_target)
         util_blitter_save_framebuffer(blitter, &context->framebuffer);

      /* Match the CPU fallback's target-format quantization. */
      uint8_t packed[16];
      union pipe_color_union quantized;
      util_format_pack_rgba(surface->format, packed, color->ui, 1);
      util_format_unpack_rgba(surface->format, quantized.ui, packed, 1);
      /* u_blitter disables query accounting around its internal draws. Deferred slots keep
       * that disabled accounting after query state is restored.
       * Only this validated color operation may defer its internal fan. The
       * caller has already completed any CPU depth/stencil part of a mixed clear. */
      context->deferred_blitter_draw = buffers == PIPE_CLEAR_COLOR0;
      if (selected_target) {
         struct pipe_surface selected = *surface;
         selected.first_layer = selected.last_layer = layer;
         util_blitter_clear_render_target(blitter, &selected, &quantized,
                                           left, bottom, right - left, top - bottom);
      } else {
         util_blitter_clear(blitter, context->framebuffer.width, context->framebuffer.height,
                            1, PIPE_CLEAR_COLOR0, &quantized, 0, 0, false);
      }
      context->deferred_blitter_draw = false;
      context->viewport_valid = viewport_valid;
      context->framebuffer_valid = framebuffer_valid;
      context->queries_enabled = queries_enabled;
      if (context->draw_calls == draws_before)
         context->last_draw_status = -30; /* Blitter upload failed before drawing. */
      if (context->last_draw_status != 0 || draws_before < 3)
         printf("[ps5-gallium] clear-gpu-color status=%d draws=%u\n",
                context->last_draw_status, context->draw_calls - draws_before);
      if (context->last_draw_status != 0)
         break;
   }
   free(copied_constants);
   /* Never hide an attempted GPU failure by retrying it on the CPU. */
   return true;
}

static bool
ps5_clear_gpu_depth_stencil(struct ps5_context *context, unsigned buffers,
                            uint8_t stencil_clear_mask,
                            const struct pipe_scissor_state *scissor_state,
                            double depth, unsigned stencil)
{
   /* Large clears avoid CPU attachment writes and cache flushing. The blitter
    * suspends query counting; partial stencil masks retain the CPU merge. */
   if (!PS5_ENABLE_MRT_CANDIDATE || !context || !context->framebuffer_valid ||
       !(buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) ||
       (buffers & ~(PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) ||
       ((buffers & PIPE_CLEAR_STENCIL) && stencil_clear_mask != 0xff) ||
       ((buffers & PIPE_CLEAR_DEPTH) && !(depth >= 0.0 && depth <= 1.0)) ||
       context->render_condition_query || context->stream_output_target_count)
      return false;
   struct pipe_surface surface = context->framebuffer.zsbuf;
   const struct ps5_resource *target = (const struct ps5_resource *)surface.texture;
   if (!target || (target->base.target != PIPE_TEXTURE_2D &&
                   target->base.target != PIPE_TEXTURE_2D_ARRAY) ||
       target->base.nr_samples > 1 || target->base.nr_storage_samples > 1 ||
       target->base.last_level || !target->base.array_size ||
       target->base.depth0 != 1 || !context->framebuffer.width || !context->framebuffer.height ||
       target->base.width0 > PS5_MAX_DEPTH_WIDTH || target->base.height0 > PS5_MAX_DEPTH_HEIGHT ||
       context->framebuffer.width > target->base.width0 ||
       context->framebuffer.height > target->base.height0 ||
       target->depth_staging_size || surface.level || surface.first_layer > surface.last_layer ||
       surface.last_layer >= target->base.array_size || surface.last_layer > 2047 ||
       (surface.format != PIPE_FORMAT_Z32_FLOAT &&
        surface.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) ||
       target->base.format != surface.format || !target->data ||
       ps5_tiled_depth_surface_size(target->base.width0, target->base.height0, 1) >
          target->allocation_size / (surface.last_layer + 1u) ||
       ((buffers & PIPE_CLEAR_STENCIL) &&
        (surface.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT || !target->stencil_data ||
         ps5_tiled_stencil_surface_size(target->base.width0, target->base.height0) >
            target->stencil_allocation_size / (surface.last_layer + 1u))))
      return false;
   unsigned left, bottom, right, top;
   ps5_clear_bounds(scissor_state, context->framebuffer.width, context->framebuffer.height,
                     &left, &bottom, &right, &top);
   if (right <= left || top <= bottom ||
       (uint64_t)context->framebuffer.width * context->framebuffer.height < PS5_GPU_CLEAR_MIN_PIXELS)
      return false;

   if (!context->blitter)
      context->blitter = util_blitter_create(&context->base);
   struct blitter_context *blitter = context->blitter;
   if (!blitter || blitter->running)
      return false;
   const unsigned first_layer = surface.first_layer, last_layer = surface.last_layer;
   for (unsigned layer = first_layer; layer <= last_layer; ++layer) {
      surface.first_layer = surface.last_layer = layer;
      uint16_t viewport_valid = context->viewport_valid;
      bool framebuffer_valid = context->framebuffer_valid;
      bool queries_enabled = context->queries_enabled;
      unsigned draws_before = context->draw_calls;
      util_blitter_save_vertex_buffers(blitter, context->vertex_buffers, context->vertex_buffer_count);
      util_blitter_save_vertex_elements(blitter, context->vertex_elements);
      util_blitter_save_vertex_shader(blitter, context->vs);
      util_blitter_save_tessctrl_shader(blitter, context->tcs);
      util_blitter_save_tesseval_shader(blitter, context->tes);
      util_blitter_save_geometry_shader(blitter, context->gs);
      util_blitter_save_so_targets(blitter, 0, NULL, context->stream_output_primitive);
      util_blitter_save_rasterizer(blitter, context->rasterizer);
      util_blitter_save_fragment_shader(blitter, context->fs);
      util_blitter_save_depth_stencil_alpha(blitter, context->depth_stencil_alpha);
      util_blitter_save_blend(blitter, context->blend);
      util_blitter_save_stencil_ref(blitter, &context->stencil_ref);
      util_blitter_save_viewport(blitter, &context->viewport[0]);
      util_blitter_save_sample_mask(blitter, context->sample_mask, 1);
      util_blitter_save_framebuffer(blitter, &context->framebuffer);
      context->deferred_blitter_draw = true;
      util_blitter_clear_depth_stencil(blitter, &surface, buffers, depth, stencil,
                                        left, bottom, right - left, top - bottom);
      context->deferred_blitter_draw = false;
      context->viewport_valid = viewport_valid;
      context->framebuffer_valid = framebuffer_valid;
      context->queries_enabled = queries_enabled;
      if (context->draw_calls == draws_before)
         context->last_draw_status = -30;
      if (context->last_draw_status || draws_before < 3)
         printf("[ps5-gallium] clear-gpu-depth-stencil status=%d draws=%u\n",
                context->last_draw_status, context->draw_calls - draws_before);
      if (context->last_draw_status != 0)
         break;
   }
   return true; /* Handled even on failure: never replay an attempted GPU operation. */
}

/* Full aligned clears need not pull old attachment contents into CPU caches.
 * Callers still drain GPU users and publish the cleared allocation as before. */
static void
ps5_clear_words(void *data, uint32_t value, size_t count)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(__SSE2__)
   if (count >= 16384 && !(count & 15u) && !((uintptr_t)data & 63u)) {
      __m128i *out = data;
      const __m128i fill = _mm_set1_epi32((int)value);
      for (size_t i = 0; i < count / 4; i += 4) {
         _mm_stream_si128(out + i, fill);
         _mm_stream_si128(out + i + 1, fill);
         _mm_stream_si128(out + i + 2, fill);
         _mm_stream_si128(out + i + 3, fill);
      }
      _mm_sfence();
      return;
   }
#endif
   util_memset32(data, value, count);
}

static bool
ps5_clear_depth_stencil(struct ps5_context *context, unsigned buffers,
                         uint8_t stencil_clear_mask,
                         const struct pipe_scissor_state *scissor_state,
                         double depth, unsigned stencil)
{
#ifdef PS5_DRAW_PROFILE
   const int64_t clear_start = os_time_get_nano();
#endif
   struct ps5_resource *resource;
   uint32_t clear_bits;
   size_t index;
   size_t depth_layer_size = 0;
   size_t stencil_layer_size = 0;
   unsigned first_depth_layer = 0;
   unsigned last_depth_layer = 0;

   resource = context && context->framebuffer.zsbuf.texture
                 ? (struct ps5_resource *)context->framebuffer.zsbuf.texture
                 : NULL;
   if (resource) {
      depth_layer_size = ps5_tiled_depth_surface_size(
         resource->base.width0, resource->base.height0,
         resource->base.nr_samples);
      stencil_layer_size = ps5_tiled_stencil_surface_size_samples(
         resource->base.width0, resource->base.height0,
         resource->base.nr_samples);
      first_depth_layer = context->framebuffer.zsbuf.first_layer;
      last_depth_layer = context->framebuffer.zsbuf.last_layer;
   }
   if (resource && resource->depth_staging_size &&
       (buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL))) {
      const struct pipe_surface *surface = &context->framebuffer.zsbuf;
      const bool packed = resource->base.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
      const unsigned pixel_size = packed ? 8 : 4;
      unsigned width, height;
      unsigned min_x, min_y, max_x, max_y;
      size_t stride, span;

      if (!context->framebuffer_valid ||
          (buffers & ~(PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) ||
          (resource->base.format != PIPE_FORMAT_Z32_FLOAT && !packed) ||
          ((buffers & PIPE_CLEAR_STENCIL) && !packed) ||
          resource->base.nr_samples > 1 || !resource->data ||
          surface->level > resource->base.last_level ||
          surface->level >= ARRAY_SIZE(resource->level_stride) || surface->level >= 32 ||
          first_depth_layer > last_depth_layer ||
          last_depth_layer >= ps5_surface_layer_count(surface) ||
          !resource->layer_stride || resource->size > resource->allocation_size ||
          last_depth_layer >= resource->size / resource->layer_stride ||
          ((buffers & PIPE_CLEAR_DEPTH) && !(depth >= 0.0 && depth <= 1.0)))
         goto reject;
      width = ps5_surface_width(surface);
      height = ps5_surface_height(surface);
      stride = resource->level_stride[surface->level];
      if (stride < (uint64_t)width * pixel_size || stride > SIZE_MAX / height)
         goto reject;
      span = (height - 1u) * stride + (size_t)width * pixel_size;
      if (resource->level_offset[surface->level] > resource->layer_stride ||
          span > resource->layer_stride - resource->level_offset[surface->level])
         goto reject;

      ps5_clear_bounds(scissor_state, width, height,
                       &min_x, &min_y, &max_x, &max_y);
      clear_bits = ps5_float_bits((float)depth);
      for (unsigned layer = first_depth_layer;
           layer <= last_depth_layer; ++layer) {
         size_t layer_base = (size_t)layer * resource->layer_stride +
                             resource->level_offset[surface->level];

         for (unsigned y = min_y; y < max_y; ++y) {
            for (unsigned x = min_x; x < max_x; ++x) {
               size_t offset = layer_base +
                  (size_t)y * resource->level_stride[surface->level] +
                  (size_t)x * pixel_size;

               if (offset > resource->size ||
                   resource->size - offset < pixel_size)
                  goto reject;
               if (buffers & PIPE_CLEAR_DEPTH)
                  memcpy(resource->data + offset, &clear_bits, sizeof(clear_bits));
               if (buffers & PIPE_CLEAR_STENCIL) {
                  uint8_t *value = resource->data + offset + sizeof(clear_bits);
                  *value = (*value & ~stencil_clear_mask) |
                           ((uint8_t)stencil & stencil_clear_mask);
               }
            }
         }
      }
      ps5_flush_gpu_data(resource->data, resource->size);
      printf("[ps5-gallium] clear-depth-mip level=%u size=%ux%u layers=%u-%u depth=%.9g/%08x scissor=%u\n",
             surface->level, width, height, first_depth_layer,
             last_depth_layer, depth, clear_bits,
             scissor_state != NULL);
      return true;
   }
   if (!context || !context->framebuffer_valid ||
       !resource || !(buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) ||
       (buffers & ~(PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) ||
       (resource->base.format != PIPE_FORMAT_Z32_FLOAT &&
        resource->base.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) ||
       !depth_layer_size || !stencil_layer_size ||
       first_depth_layer > last_depth_layer ||
       last_depth_layer >=
          ps5_surface_layer_count(&context->framebuffer.zsbuf) ||
       ((buffers & PIPE_CLEAR_STENCIL) &&
        (resource->base.format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
         !resource->stencil_data ||
         last_depth_layer >=
            resource->stencil_allocation_size / stencil_layer_size)) ||
       ((buffers & PIPE_CLEAR_DEPTH) &&
        (last_depth_layer >= resource->allocation_size / depth_layer_size ||
         depth_layer_size % sizeof(uint32_t) ||
         !(depth >= 0.0 && depth <= 1.0)))) {
reject:
      printf("[ps5-gallium] reject-clear buffers=%08x scissor=%u format=%u depth=%.9g\n",
             buffers, scissor_state != NULL,
             resource ? resource->base.format : 0, depth);
      return false;
   }

   /* A full-surface scissor is the same uniform fill, including tiled padding.
    * Partial scissors must retain per-pixel addressing and stencil merging. */
   if (scissor_state && !scissor_state->minx && !scissor_state->miny &&
       scissor_state->maxx >= resource->base.width0 &&
       scissor_state->maxy >= resource->base.height0)
      scissor_state = NULL;
   clear_bits = ps5_float_bits((float)depth);
   if (buffers & PIPE_CLEAR_DEPTH) {
      for (unsigned layer = first_depth_layer;
           layer <= last_depth_layer; ++layer) {
         uint8_t *layer_data = resource->data + layer * depth_layer_size;

         if (!scissor_state) {
            ps5_clear_words(layer_data, clear_bits,
                          depth_layer_size / sizeof(clear_bits));
         } else {
            unsigned min_x, min_y, max_x, max_y;

            ps5_clear_bounds(scissor_state, resource->base.width0,
                             resource->base.height0,
                             &min_x, &min_y, &max_x, &max_y);
            for (unsigned y = min_y; y < max_y; ++y) {
               for (unsigned x = min_x; x < max_x; ++x) {
                  for (unsigned sample = 0;
                       sample < MAX2(resource->base.nr_samples, 1); ++sample) {
                     size_t offset = resource->base.nr_samples == 4
                        ? ps5_tiled_depth_msaa4_offset(
                             x, y, sample, resource->base.width0, layer)
                        : ps5_tiled_depth_offset(
                             x, y, resource->base.width0, layer);

                     if (offset > depth_layer_size ||
                         depth_layer_size - offset < sizeof(clear_bits))
                        goto reject;
                     memcpy(layer_data + offset, &clear_bits,
                            sizeof(clear_bits));
                  }
               }
            }
         }
         ps5_flush_gpu_data(layer_data, depth_layer_size);
      }
   }
   if (buffers & PIPE_CLEAR_STENCIL) {
      uint8_t value = (uint8_t)stencil;

      for (unsigned layer = first_depth_layer;
           layer <= last_depth_layer; ++layer) {
         uint8_t *layer_data = resource->stencil_data +
                               layer * stencil_layer_size;

         if (!scissor_state) {
            for (index = 0; index < stencil_layer_size; ++index)
               layer_data[index] =
                  (layer_data[index] & ~stencil_clear_mask) |
                  (value & stencil_clear_mask);
         } else {
            unsigned min_x, min_y, max_x, max_y;

            ps5_clear_bounds(scissor_state, resource->base.width0,
                             resource->base.height0,
                             &min_x, &min_y, &max_x, &max_y);
            for (unsigned y = min_y; y < max_y; ++y) {
               for (unsigned x = min_x; x < max_x; ++x) {
                  for (unsigned sample = 0;
                       sample < MAX2(resource->base.nr_samples, 1); ++sample) {
                     size_t offset = resource->base.nr_samples == 4
                        ? ps5_tiled_stencil_msaa4_offset(
                             x, y, sample, resource->base.width0, layer)
                        : ps5_tiled_stencil_offset(
                             x, y, resource->base.width0, layer);

                     if (offset >= stencil_layer_size)
                        goto reject;
                     layer_data[offset] =
                        (layer_data[offset] & ~stencil_clear_mask) |
                        (value & stencil_clear_mask);
                  }
               }
            }
         }
         ps5_flush_gpu_data(layer_data, stencil_layer_size);
      }
   }
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
   printf("[ps5-gallium] clear-depth-stencil buffers=%08x format=%u size=%ux%u layers=%u-%u samples=%u allocation=%zu/%zu depth=%.9g/%08x stencil=%02x/%02x scissor=%u\n",
          buffers, resource->base.format, resource->base.width0,
          resource->base.height0, first_depth_layer, last_depth_layer,
          resource->base.nr_samples,
          resource->allocation_size, resource->stencil_allocation_size,
          depth, clear_bits, stencil & 0xffu, stencil_clear_mask,
          scissor_state != NULL);
#endif
#ifdef PS5_DRAW_PROFILE
   printf("[ps5-clear-cpu] depth_ns=%" PRIi64 " target=%u mip=%u levels=%u layers=%u depth0=%u fb=%ux%u condition=%u streamout=%u\n",
          os_time_get_nano() - clear_start, resource->base.target,
          context->framebuffer.zsbuf.level, resource->base.last_level,
          resource->base.array_size, resource->base.depth0,
          context->framebuffer.width, context->framebuffer.height,
          context->render_condition_query != NULL, context->stream_output_target_count);
#endif
   return true;
}

static bool
ps5_clear_full_tiled_color(struct ps5_resource *target,
                           const struct pipe_surface *surface, unsigned write_mask,
                           unsigned min_x, unsigned min_y, unsigned max_x, unsigned max_y,
                           const uint8_t packed[16])
{
   /* A uniform four-byte pixel is identical in every tiled address. Include
    * padding only for an entire single-level, single-layer allocation. */
   if ((target->base.bind & PIPE_BIND_DISPLAY_TARGET) ||
       target->render_staging_size || target->base.last_level ||
       target->base.array_size != 1 || target->base.depth0 != 1 ||
       target->base.nr_samples > 1 || target->base.nr_storage_samples > 1 ||
       surface->level || surface->first_layer || surface->last_layer ||
       write_mask != PIPE_MASK_RGBA || min_x || min_y ||
       max_x != target->base.width0 || max_y != target->base.height0 ||
       util_format_get_blocksize(surface->format) != 4 ||
       util_format_get_blocksize(target->base.format) != 4 ||
       !target->data || !target->allocation_size || target->allocation_size % 4)
      return false;
   uint32_t pixel;
   memcpy(&pixel, packed, sizeof(pixel));
   ps5_clear_words(target->data, pixel, target->allocation_size / sizeof(pixel));
   ps5_flush_gpu_data(target->data, target->allocation_size);
   return true;
}

static void
ps5_clear(struct pipe_context *base, unsigned buffers,
          uint32_t color_clear_mask, uint8_t stencil_clear_mask,
          const struct pipe_scissor_state *scissor_state,
          const union pipe_color_union *color, double depth,
          unsigned stencil)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {4, ps5_prepare_clock()};
#endif

   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_resource *resource;

   if (context && context->render_condition_query)
      ps5_draw_batch_drain_query(context->render_condition_query);
   resource = context && context->framebuffer.zsbuf.texture
                 ? (struct ps5_resource *)context->framebuffer.zsbuf.texture
                 : NULL;
   if (context && !ps5_render_condition_passes(context))
      return;
   /* Finish CPU depth/stencil writes before a color clear can enter the GPU
    * queue. No later CPU attachment clear may drain that new color/draw batch. */
   unsigned depth_buffers = buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL);
   if (depth_buffers) {
      if (ps5_clear_gpu_depth_stencil(context, depth_buffers, stencil_clear_mask,
                                      scissor_state, depth, stencil)) {
         if (context->last_draw_status != 0)
            return;
      } else {
         ps5_draw_batch_drain_buffer(resource ? &resource->base : NULL);
         if (!ps5_clear_depth_stencil(context, depth_buffers, stencil_clear_mask,
                                      scissor_state, depth, stencil))
            return;
      }
      buffers &= ~depth_buffers;
   }
   if (!buffers)
      return;
   if (ps5_clear_gpu_color(context, buffers, color_clear_mask, scissor_state, color)) {
      buffers &= ~PIPE_CLEAR_COLOR;
      if (!buffers || context->last_draw_status != 0)
         return;
   }
   /* GPU-only color clears stay ordered in the queue. CPU fallback writes
    * must wait for any draw using the attachments they are about to modify. */
   if (context)
      for (unsigned i = 0; i < context->framebuffer.nr_cbufs; ++i)
         if (buffers & (PIPE_CLEAR_COLOR0 << i))
            ps5_draw_batch_drain_buffer(context->framebuffer.cbufs[i].texture);
   if (PS5_ENABLE_MSAA4_CANDIDATE && context &&
       context->framebuffer_valid &&
       ps5_clear_msaa4_color(context, buffers, color_clear_mask,
                             scissor_state, color)) {
      buffers &= ~PIPE_CLEAR_COLOR;
      if (!buffers)
         return;
   }
   if (PS5_ENABLE_MRT_CANDIDATE && context &&
       context->framebuffer_valid && (buffers & PIPE_CLEAR_COLOR) &&
       color) {
#ifdef PS5_DRAW_PROFILE
      const int64_t clear_start = os_time_get_nano();
#endif
      unsigned color_buffers = buffers & PIPE_CLEAR_COLOR;

      for (unsigned target_index = 0;
           target_index < context->framebuffer.nr_cbufs; ++target_index) {
         unsigned bit = PIPE_CLEAR_COLOR0 << target_index;
         struct pipe_surface *surface =
            &context->framebuffer.cbufs[target_index];

         if (!(color_buffers & bit))
            continue;
         color_buffers &= ~bit;
         if (!surface->texture ||
             !ps5_render_target_format(surface->format) ||
             surface->texture->nr_samples > 1)
            goto reject;
      }
      if (color_buffers)
         goto reject;
      for (unsigned target_index = 0;
           target_index < context->framebuffer.nr_cbufs; ++target_index) {
         unsigned bit = PIPE_CLEAR_COLOR0 << target_index;
         unsigned write_mask =
            (color_clear_mask >> (4u * target_index)) & UINT32_C(0xf);
         const struct pipe_surface *surface =
            &context->framebuffer.cbufs[target_index];
         struct ps5_resource *target;
         size_t layer_base;
         unsigned width;
         unsigned height;
         unsigned min_x, min_y, max_x, max_y;
         unsigned format_size;
         uint8_t packed[16];

         if (!(buffers & bit))
            continue;
         target = (struct ps5_resource *)
            context->framebuffer.cbufs[target_index].texture;
         width = ps5_surface_width(surface);
         height = ps5_surface_height(surface);
         format_size = util_format_get_blocksize(surface->format);
         if (!format_size || format_size > sizeof(packed))
            goto reject;
         ps5_clear_bounds(scissor_state, width, height,
                          &min_x, &min_y, &max_x, &max_y);
         util_format_pack_rgba(surface->format, packed, color->ui, 1);
         if (ps5_clear_full_tiled_color(target, surface, write_mask,
                                       min_x, min_y, max_x, max_y, packed))
            continue;
         for (unsigned layer = surface->first_layer;
              layer <= surface->last_layer; ++layer) {
            layer_base = (size_t)layer * target->layer_stride +
                         target->level_offset[surface->level];
            for (unsigned y = min_y; y < max_y; ++y) {
               if (target->render_staging_size && write_mask == PIPE_MASK_RGBA &&
                   format_size == 4 && min_x < max_x && !(layer_base & 3) &&
                   !(target->level_stride[surface->level] & 3)) {
                  size_t offset = layer_base + (size_t)y * target->level_stride[surface->level] +
                                  (size_t)min_x * 4;
                  size_t bytes = (size_t)(max_x - min_x) * 4;
                  if (offset > target->size || bytes > target->size - offset)
                     goto reject;
                  uint32_t pixel;
                  memcpy(&pixel, packed, sizeof(pixel));
                  ps5_clear_words(target->data + offset, pixel, bytes / 4);
                  continue;
               }
               for (unsigned x = min_x; x < max_x; ++x) {
                  size_t offset = target->render_staging_size
                     ? layer_base +
                          (size_t)y * target->level_stride[surface->level] +
                          (size_t)x * format_size
                     : layer_base + ps5_tiled_color_offset(
                          target->base.format, x, y, width, layer);
                  size_t limit = target->render_staging_size
                                    ? target->size : target->allocation_size;

                  if (offset > limit || limit - offset < format_size)
                     goto reject;
                  if (write_mask == PIPE_MASK_RGBA) {
                     memcpy(target->data + offset, packed, format_size);
                  } else {
                     union pipe_color_union merged;

                     util_format_unpack_rgba(surface->format, merged.ui,
                                             target->data + offset, 1);
                     for (unsigned channel = 0; channel < 4; ++channel) {
                        if (write_mask & BITFIELD_BIT(channel))
                           merged.ui[channel] = color->ui[channel];
                     }
                     util_format_pack_rgba(surface->format,
                                           target->data + offset,
                                           merged.ui, 1);
                  }
               }
            }
         }
         ps5_flush_gpu_data(target->data, target->render_staging_size
                                             ? target->size
                                             : target->allocation_size);
      }
      if (context && !context->reported_large_color_fallback &&
          context->framebuffer.nr_cbufs == 1 &&
          (buffers & PIPE_CLEAR_COLOR0) &&
          (uint64_t)context->framebuffer.width * context->framebuffer.height >= 1280u * 720u) {
         const struct pipe_surface *surface = &context->framebuffer.cbufs[0];
         const struct ps5_resource *target = (const struct ps5_resource *)surface->texture;
         if (target) {
            unsigned x0, y0, x1, y1;
            ps5_clear_bounds(scissor_state, context->framebuffer.width,
                             context->framebuffer.height, &x0, &y0, &x1, &y1);
            context->reported_large_color_fallback = true;
            printf("[ps5-clear-fallback] valid=%u buffers=%x mask=%x condition=%u streamout=%u "
                   "format=%u/%u target=%u samples=%u/%u staging=%zu pitch=%u "
                   "mip=%u layers=%u-%u extent=%ux%u/%ux%u constants=%u/%u/%u blitter=%u bounds=%u,%u-%u,%u\n",
                   context->framebuffer_valid, buffers, color_clear_mask,
                   context->render_condition_query != NULL, context->stream_output_target_count,
                   surface->format, target->base.format, target->base.target,
                   target->base.nr_samples, target->base.nr_storage_samples,
                   target->render_staging_size, ps5_linear_color_pitch(surface),
                   surface->level, surface->first_layer, surface->last_layer,
                   context->framebuffer.width, context->framebuffer.height,
                   ps5_surface_width(surface), ps5_surface_height(surface),
                   context->constants[1][0].valid, context->constants[1][0].copied,
                   context->constants[1][0].size,
                   context->blitter ? context->blitter->running : 0, x0, y0, x1, y1);
         }
      }
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
         printf("[ps5-gallium] clear-mrt-color targets=%u mask=%08x scissor=%u"
             " buffers=%x format=%u target=%u width=%u height=%u query=%u\n",
             context->framebuffer.nr_cbufs, color_clear_mask,
             scissor_state != NULL, buffers,
             context->framebuffer.nr_cbufs ? context->framebuffer.cbufs[0].format : 0,
             context->framebuffer.nr_cbufs && context->framebuffer.cbufs[0].texture
                ? context->framebuffer.cbufs[0].texture->target : 0,
             context->framebuffer.width, context->framebuffer.height,
             context->active_occlusion_query != NULL);
#endif
#ifdef PS5_DRAW_PROFILE
      const struct pipe_surface *first = &context->framebuffer.cbufs[0];
      printf("[ps5-clear-cpu] color_ns=%" PRIi64 " resource=%ux%u mip=%u layers=%u-%u condition=%u streamout=%u\n",
             os_time_get_nano() - clear_start, ps5_surface_width(first), ps5_surface_height(first),
             first->level, first->first_layer, first->last_layer,
             context->render_condition_query != NULL, context->stream_output_target_count);
#endif
      buffers &= ~PIPE_CLEAR_COLOR;
      if (!buffers)
         return;
   }
reject:
   printf("[ps5-gallium] reject-clear buffers=%08x scissor=%u format=%u depth=%.9g\n",
          buffers, scissor_state != NULL,
          resource ? resource->base.format : 0, depth);
}

static bool
ps5_remove_point_size(nir_builder *builder, nir_instr *instruction,
                      void *data)
{
   nir_intrinsic_instr *intrinsic;

   (void)builder;
   (void)data;
   if (instruction->type != nir_instr_type_intrinsic)
      return false;
   intrinsic = nir_instr_as_intrinsic(instruction);
   if (intrinsic->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intrinsic).location != VARYING_SLOT_PSIZ)
      return false;
   nir_instr_remove(instruction);
   return true;
}

static bool
ps5_rebase_meta_vertex_input(nir_builder *builder, nir_instr *instruction,
                             void *data)
{
   const uint64_t inputs = *(const uint64_t *)data;
   nir_intrinsic_instr *intrinsic;
   nir_io_semantics semantics;

   (void)builder;
   if (instruction->type != nir_instr_type_intrinsic)
      return false;
   intrinsic = nir_instr_as_intrinsic(instruction);
   if (intrinsic->intrinsic != nir_intrinsic_load_input)
      return false;
   semantics = nir_intrinsic_io_semantics(intrinsic);
   /* Gallium elements are in used-slot order. Rebase the complete input set
    * when a meta shader uses legacy slots such as VERT_ATTRIB_POS, because
    * RADV vertex lowering expects VERT_ATTRIB_GENERIC0-based locations. */
   semantics.location = VERT_ATTRIB_GENERIC0 +
      util_bitcount64(inputs & BITFIELD64_MASK(semantics.location));
   nir_intrinsic_set_io_semantics(intrinsic, semantics);
   return true;
}

static bool
ps5_lower_fragment_color(nir_builder *builder, nir_instr *instruction,
                         void *data)
{
   const unsigned color_mask = *(const unsigned *)data;
   nir_intrinsic_instr *intrinsic;
   nir_io_semantics semantics;

   if (instruction->type != nir_instr_type_intrinsic)
      return false;
   intrinsic = nir_instr_as_intrinsic(instruction);
   if (intrinsic->intrinsic != nir_intrinsic_store_output)
      return false;
   semantics = nir_intrinsic_io_semantics(intrinsic);
   if (semantics.location != FRAG_RESULT_COLOR)
      return false;
   /* Mesa's builtin clear writes one untyped color to every enabled target.
    * RADV expects explicit per-target outputs, including sparse draw slots. */
   builder->cursor = nir_before_instr(instruction);
   u_foreach_bit(slot, color_mask) {
      nir_intrinsic_instr *store = nir_instr_as_intrinsic(
         nir_instr_clone(builder->shader, instruction));
      semantics.location = FRAG_RESULT_DATA0 + slot;
      nir_intrinsic_set_io_semantics(store, semantics);
      nir_builder_instr_insert(builder, &store->instr);
   }
   nir_instr_remove(instruction);
   return true;
}

static bool
ps5_vertex_format(enum pipe_format format, PsbcVertexFormat *out)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      *out = PSBC_VERTEX_FORMAT_R8G8B8A8_SNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R16G16_UNORM:
      *out = PSBC_VERTEX_FORMAT_R16G16_UNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R16G16_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R16G16_FLOAT;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R16G16B16A16_FLOAT;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R16G16_SNORM:
      *out = PSBC_VERTEX_FORMAT_R16G16_SNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R8_UINT:
      *out = PSBC_VERTEX_FORMAT_R8_UINT;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R8G8_UINT:
      *out = PSBC_VERTEX_FORMAT_R8G8_UINT;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R8G8B8A8_UINT:
      *out = PSBC_VERTEX_FORMAT_R8G8B8A8_UINT;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R8G8_UNORM:
      *out = PSBC_VERTEX_FORMAT_R8G8_UNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R8G8_SNORM:
      *out = PSBC_VERTEX_FORMAT_R8G8_SNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R32_FLOAT;
      return true;
   case PIPE_FORMAT_R32G32_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R32G32_FLOAT;
      return true;
   case PIPE_FORMAT_R32G32B32_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R32G32B32_FLOAT;
      return true;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT;
      return true;
   case PIPE_FORMAT_R64_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R64_FLOAT;
      return PS5_ENABLE_FP64_CANDIDATE;
   case PIPE_FORMAT_R64G64_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R64G64_FLOAT;
      return PS5_ENABLE_FP64_CANDIDATE;
   case PIPE_FORMAT_R64G64B64_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R64G64B64_FLOAT;
      return PS5_ENABLE_FP64_CANDIDATE;
   case PIPE_FORMAT_R64G64B64A64_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R64G64B64A64_FLOAT;
      return PS5_ENABLE_FP64_CANDIDATE;
   case PIPE_FORMAT_R11G11B10_FLOAT:
      *out = PSBC_VERTEX_FORMAT_R11G11B10_FLOAT;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32_SINT:
      *out = PSBC_VERTEX_FORMAT_R32_SINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32G32_SINT:
      *out = PSBC_VERTEX_FORMAT_R32G32_SINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32G32B32_SINT:
      *out = PSBC_VERTEX_FORMAT_R32G32B32_SINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32G32B32A32_SINT:
      *out = PSBC_VERTEX_FORMAT_R32G32B32A32_SINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32_UINT:
      *out = PSBC_VERTEX_FORMAT_R32_UINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32G32_UINT:
      *out = PSBC_VERTEX_FORMAT_R32G32_UINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32G32B32_UINT:
      *out = PSBC_VERTEX_FORMAT_R32G32B32_UINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R32G32B32A32_UINT:
      *out = PSBC_VERTEX_FORMAT_R32G32B32A32_UINT;
      return PS5_ENABLE_INTEGER_VERTEX_CANDIDATE;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      *out = PSBC_VERTEX_FORMAT_B8G8R8A8_UNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
      *out = PSBC_VERTEX_FORMAT_R8G8B8A8_UNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      *out = PSBC_VERTEX_FORMAT_R10G10B10A2_UNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      *out = PSBC_VERTEX_FORMAT_B10G10R10A2_UNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R10G10B10A2_SNORM:
      *out = PSBC_VERTEX_FORMAT_R10G10B10A2_SNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_B10G10R10A2_SNORM:
      *out = PSBC_VERTEX_FORMAT_B10G10R10A2_SNORM;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R10G10B10A2_USCALED:
      *out = PSBC_VERTEX_FORMAT_R10G10B10A2_USCALED;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_B10G10R10A2_USCALED:
      *out = PSBC_VERTEX_FORMAT_B10G10R10A2_USCALED;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_R10G10B10A2_SSCALED:
      *out = PSBC_VERTEX_FORMAT_R10G10B10A2_SSCALED;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   case PIPE_FORMAT_B10G10R10A2_SSCALED:
      *out = PSBC_VERTEX_FORMAT_B10G10R10A2_SSCALED;
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE;
   default:
      return false;
   }
}

static unsigned
ps5_vertex_format_size(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32_FLOAT:
      return 4;
   case PIPE_FORMAT_R32G32_FLOAT:
      return 8;
   case PIPE_FORMAT_R32G32B32_FLOAT:
      return 12;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return 16;
   case PIPE_FORMAT_R64_FLOAT:
      return PS5_ENABLE_FP64_CANDIDATE ? 8 : 0;
   case PIPE_FORMAT_R64G64_FLOAT:
      return PS5_ENABLE_FP64_CANDIDATE ? 16 : 0;
   case PIPE_FORMAT_R64G64B64_FLOAT:
      return PS5_ENABLE_FP64_CANDIDATE ? 24 : 0;
   case PIPE_FORMAT_R64G64B64A64_FLOAT:
      return PS5_ENABLE_FP64_CANDIDATE ? 32 : 0;
   default:
      if (PS5_ENABLE_INTEGER_VERTEX_CANDIDATE &&
          ps5_integer_vertex_format(format))
         return util_format_get_blocksize(format);
      return PS5_ENABLE_PACKED_VERTEX_CANDIDATE &&
             ps5_packed_vertex_format(format) ? util_format_get_blocksize(format) : 0;
   }
}

static unsigned
ps5_vertex_format_alignment(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      return 1;
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_UNORM:
      return 2;
   case PIPE_FORMAT_R64_FLOAT:
   case PIPE_FORMAT_R64G64_FLOAT:
   case PIPE_FORMAT_R64G64B64_FLOAT:
   case PIPE_FORMAT_R64G64B64A64_FLOAT:
      return 8;
   default:
      return 4;
   }
}

static bool
ps5_default_vertex_layout(const nir_shader *nir,
                          struct ps5_vertex_layout *layout)
{
   uint64_t inputs = nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;
   unsigned location;

   memset(layout, 0, sizeof(*layout));
   for (location = 0; location < 16; ++location) {
      PsbcVertexAttribute *attribute;

      if (!(inputs & BITFIELD64_BIT(location)))
         continue;
      attribute = &layout->attributes[layout->count++];
      attribute->location = location;
      attribute->binding = layout->count - 1;
      attribute->format = PSBC_VERTEX_FORMAT_R32G32B32A32_FLOAT;
      attribute->stride = 16;
      attribute->alignment = 4;
   }
   return layout->count <= PSBC_MAX_VERTEX_ATTRIBUTES;
}

static bool
ps5_vertex_layout_from_state(const struct ps5_shader *shader,
                             const struct ps5_vertex_elements *elements,
                             struct ps5_vertex_layout *layout)
{
   uint64_t inputs = shader->nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;
   unsigned location;
   unsigned element_index = 0;

   memset(layout, 0, sizeof(*layout));
   for (location = 0; location < 16; ++location) {
      const struct pipe_vertex_element *element;
      PsbcVertexAttribute *attribute;

      if (!(inputs & BITFIELD64_BIT(location)))
         continue;
      if (!elements || element_index >= elements->count ||
          layout->count >= PSBC_MAX_VERTEX_ATTRIBUTES)
         return false;
      element = &elements->elements[element_index++];
      attribute = &layout->attributes[layout->count++];
      if (!ps5_vertex_format(element->src_format, &attribute->format))
         return false;
      attribute->location = location;
      attribute->binding = element->vertex_buffer_index;
      attribute->offset = element->src_offset;
      attribute->stride = element->src_stride;
      attribute->alignment = ps5_vertex_format_alignment(element->src_format);
      attribute->instance_divisor = element->instance_divisor;
   }
   /* A shared vertex-element state may contain unused trailing attributes
    * (Mesa's position-only depth-clear VS uses its two-element blit state).
    * Every consumed element was validated above; unused elements need no fetch. */
   return true;
}

static bool
ps5_append_ubo_descriptors(PsbcCompileOptions *options, unsigned first,
                           unsigned count)
{
   if (first + count > 2u * PS5_MAX_CONSTANT_BUFFERS ||
       PSBC_GALLIUM_UBO_BINDING_BASE + first + count >
          PSBC_MAX_DESCRIPTOR_BINDINGS ||
       options->descriptor_binding_count + count >
          PSBC_MAX_DESCRIPTOR_BINDINGS)
      return false;

   for (unsigned index = 0; index < count; ++index) {
      const unsigned binding = first + index;
      PsbcDescriptorBinding *descriptor =
         &options->descriptor_bindings[options->descriptor_binding_count++];

      descriptor->set = 0;
      descriptor->binding = PSBC_GALLIUM_UBO_BINDING_BASE + binding;
      descriptor->type = PSBC_DESCRIPTOR_UNIFORM_BUFFER;
      descriptor->array_size = 1;
      descriptor->offset = PS5_TEXTURE_DESCRIPTOR_BYTES + binding * 16u;
      descriptor->stride = 16;
   }
   return true;
}

static bool
ps5_append_texture_descriptor(PsbcCompileOptions *options, unsigned binding, unsigned base_offset)
{
   PsbcDescriptorBinding *descriptor;

   if (binding >= PS5_MERGED_TEXTURE_UNITS || base_offset > PS5_CONSTANT_DATA_OFFSET ||
       (binding + 1u) * PS5_TEXTURE_DESCRIPTOR_STRIDE > PS5_CONSTANT_DATA_OFFSET - base_offset)
      return false;
   for (unsigned index = 0; index < options->descriptor_binding_count;
        ++index) {
      descriptor = &options->descriptor_bindings[index];
      if (descriptor->set != 0 || descriptor->binding != binding)
         continue;
      return descriptor->type == PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER &&
             descriptor->array_size == 1 &&
             descriptor->offset == base_offset + binding * PS5_TEXTURE_DESCRIPTOR_STRIDE &&
             descriptor->stride == PS5_TEXTURE_DESCRIPTOR_STRIDE;
   }
   if (options->descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
      return false;

   descriptor =
      &options->descriptor_bindings[options->descriptor_binding_count++];
   descriptor->set = 0;
   descriptor->binding = binding;
   descriptor->type = PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER;
   descriptor->array_size = 1;
   descriptor->offset = base_offset + binding * PS5_TEXTURE_DESCRIPTOR_STRIDE;
   descriptor->stride = PS5_TEXTURE_DESCRIPTOR_STRIDE;
   return true;
}

static bool
ps5_shader_has_indirect_ubo(const struct ps5_shader *shader)
{
   nir_foreach_function_impl(impl, shader->nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_ubo &&
                !nir_src_is_const(intr->src[0]))
               return true;
         }
      }
   }
   return false;
}

/* Merged VS+GS pipelines address buffers through per-stage descriptor arrays
 * when either stage uses storage or indexes a UBO array dynamically; the
 * compiler requires constant UBO slots otherwise. */
static bool
ps5_geometry_uses_buffer_arrays(const struct ps5_context *context)
{
   return ps5_shader_uses_storage(context->vs) ||
          ps5_shader_uses_storage(context->gs) ||
          (context->vs && context->vs->nir->info.num_ubos &&
           ps5_shader_has_indirect_ubo(context->vs)) ||
          (context->gs && context->gs->nir->info.num_ubos &&
           ps5_shader_has_indirect_ubo(context->gs));
}

static bool
ps5_shader_compile_options(const struct ps5_shader *shader,
                           uint32_t address32_hi,
                           const struct ps5_vertex_layout *layout,
                           uint32_t primitive_type,
                           bool provoking_vtx_last,
                           PsbcCompileOptions *options)
{
   memset(options, 0, sizeof(*options));
   options->target = PSBC_TARGET_PS5;
   options->stage = shader->stage;
   options->entrypoint = "main";
   options->optimise = true;
   options->ngg = shader->stage == PSBC_STAGE_VERTEX;
   options->primitive_type = primitive_type;
   options->provoking_vtx_last = provoking_vtx_last;
   options->address32_hi = address32_hi;
   if (shader->stage == PSBC_STAGE_FRAGMENT && ps5_shader_uses_storage(shader)) {
      /* Internal mixed FS banks; keep the qualified 3.3 descriptor ABI. */
      if (shader->nir->info.num_ssbos > PS5_COMPUTE_STORAGE_SLOTS ||
          shader->nir->info.num_images > PS5_COMPUTE_IMAGE_SLOTS ||
          shader->nir->info.num_ubos > PS5_MAX_CONSTANT_BUFFERS)
         return false;
      options->gallium_buffer_arrays = true;
      options->descriptor_binding_count = 1;
      options->descriptor_bindings[0] = (PsbcDescriptorBinding){
         .binding = PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_FRAGMENT),
         .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
         .array_size = PS5_COMPUTE_STORAGE_SLOTS, .stride = 16,
      };
      if (shader->nir->info.num_images) {
         options->descriptor_binding_count = 2;
         options->descriptor_bindings[1] = (PsbcDescriptorBinding){
            .binding = PSBC_GALLIUM_IMAGE_ARRAY_BINDING(PSBC_STAGE_FRAGMENT),
            .type = PSBC_DESCRIPTOR_STORAGE_IMAGE, .array_size = PS5_COMPUTE_IMAGE_SLOTS,
            .offset = PS5_COMPUTE_STORAGE_SLOTS * 16, .stride = 32,
         };
      }
      if (shader->nir->info.num_ubos)
         options->descriptor_bindings[options->descriptor_binding_count++] = (PsbcDescriptorBinding){
            .binding = PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_FRAGMENT),
            .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER, .array_size = PS5_MAX_CONSTANT_BUFFERS,
            .offset = PS5_FRAGMENT_UBO_OFFSET, .stride = 16,
         };
      for (unsigned binding = 0; binding < PS5_MAX_TEXTURE_UNITS; ++binding)
         if (BITSET_TEST(shader->nir->info.textures_used, binding) &&
             !ps5_append_texture_descriptor(options, binding, PS5_FRAGMENT_TEXTURE_OFFSET))
            return false;
      return true;
   }
   if (shader->nir->info.num_images > PS5_COMPUTE_IMAGE_SLOTS ||
       shader->nir->info.num_ssbos > PS5_COMPUTE_STORAGE_SLOTS ||
       ((shader->nir->info.num_images || shader->nir->info.num_ssbos) &&
        shader->stage != PSBC_STAGE_VERTEX))
      return false;
   options->vertex_attribute_count = layout->count;
   memcpy(options->vertex_attributes, layout->attributes,
          layout->count * sizeof(layout->attributes[0]));
   for (unsigned binding = 0; binding < PS5_MAX_TEXTURE_UNITS; ++binding) {
      if (!BITSET_TEST(shader->nir->info.textures_used, binding))
         continue;
      if (!ps5_append_texture_descriptor(options, binding, 0))
         return false;
   }
   if (shader->nir->info.num_ssbos) {
      if (options->descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
         return false;
      options->gallium_buffer_arrays = true;
      options->descriptor_bindings[options->descriptor_binding_count++] =
         (PsbcDescriptorBinding){
            .binding = PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_VERTEX),
            .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
            .array_size = PS5_COMPUTE_STORAGE_SLOTS,
            .offset = PS5_VERTEX_STORAGE_OFFSET,
            .stride = 16,
         };
   }
   if (shader->nir->info.num_images) {
      if (options->descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
         return false;
      options->gallium_buffer_arrays = true;
      options->descriptor_bindings[options->descriptor_binding_count++] =
         (PsbcDescriptorBinding){
            .binding = PSBC_GALLIUM_IMAGE_ARRAY_BINDING(PSBC_STAGE_VERTEX),
            .type = PSBC_DESCRIPTOR_STORAGE_IMAGE,
            .array_size = PS5_COMPUTE_IMAGE_SLOTS,
            .offset = PS5_VERTEX_IMAGE_OFFSET,
            .stride = 32,
         };
   }
   if (shader->nir->info.num_ubos) {
      if (shader->nir->info.num_ubos >
             (PS5_ENABLE_UBO_CANDIDATE ? PS5_MAX_CONSTANT_BUFFERS : 1))
         return false;
      if (ps5_shader_uses_storage(shader) || ps5_shader_has_indirect_ubo(shader)) {
         if (options->descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
            return false;
         options->gallium_buffer_arrays = true;
         options->descriptor_bindings[options->descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PSBC_GALLIUM_UBO_ARRAY_BINDING(shader->stage),
               .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER,
               .array_size = shader->nir->info.num_ubos,
               .offset = PS5_TEXTURE_DESCRIPTOR_BYTES,
               .stride = 16,
            };
      } else if (!ps5_append_ubo_descriptors(options, 0,
                                              shader->nir->info.num_ubos)) {
         return false;
      }
   }
   return true;
}

struct ps5_ubo_offset_state {
   unsigned first;
   unsigned source_count;
   bool valid;
};

/* Gallium sampler slots are stage-local, even inside a linked GL program. */
struct ps5_texture_offset_state {
   unsigned first;
   bool valid;
};

static bool
ps5_offset_geometry_texture(nir_builder *builder, nir_instr *instruction,
                             void *data)
{
   struct ps5_texture_offset_state *state = data;
   nir_tex_instr *tex;

   (void)builder;
   if (instruction->type != nir_instr_type_tex)
      return false;
   tex = nir_instr_as_tex(instruction);
   if (state->first > PSBC_MAX_DESCRIPTOR_BINDINGS - PS5_MAX_TEXTURE_UNITS ||
       tex->texture_index >= PS5_MAX_TEXTURE_UNITS ||
       tex->sampler_index >= PS5_MAX_TEXTURE_UNITS) {
      state->valid = false;
      return false;
   }
   tex->texture_index += state->first;
   tex->sampler_index += state->first;
   return true;
}

static bool
ps5_offset_ubo_index(nir_builder *builder, nir_instr *instruction, void *data)
{
   struct ps5_ubo_offset_state *state = data;
   nir_intrinsic_instr *intrinsic;
   unsigned index;

   if (instruction->type != nir_instr_type_intrinsic)
      return false;
   intrinsic = nir_instr_as_intrinsic(instruction);
   if (intrinsic->intrinsic != nir_intrinsic_load_ubo)
      return false;
   builder->cursor = nir_before_instr(instruction);
   if (!nir_src_is_const(intrinsic->src[0])) {
      if (!state->first)
         return false;
      nir_src_rewrite(&intrinsic->src[0],
                      nir_iadd_imm(builder, intrinsic->src[0].ssa,
                                   state->first));
      return true;
   }
   index = nir_src_as_uint(intrinsic->src[0]);
   if (index >= state->source_count ||
       state->first + index >= 2u * PS5_MAX_CONSTANT_BUFFERS) {
      state->valid = false;
      return false;
   }

   nir_src_rewrite(&intrinsic->src[0],
                   nir_imm_int(builder, state->first + index));
   return true;
}

static bool
ps5_stream_output_info_valid(const struct pipe_stream_output_info *info,
                             PsbcStage stage)
{
   unsigned used_buffers = 0;

   if (!info->num_outputs)
      return true;
   if (!PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE ||
       (stage != PSBC_STAGE_VERTEX && stage != PSBC_STAGE_TESS_EVAL &&
        stage != PSBC_STAGE_GEOMETRY) ||
       info->num_outputs > PIPE_MAX_SO_OUTPUTS)
      return false;
   for (unsigned index = 0; index < info->num_outputs; ++index) {
      const struct pipe_stream_output *output = &info->output[index];

      if (output->output_buffer >= PIPE_MAX_SO_BUFFERS ||
          output->stream >= PIPE_MAX_VERTEX_STREAMS ||
          (stage != PSBC_STAGE_GEOMETRY && output->stream) ||
          !output->num_components || output->num_components > 4 ||
          output->start_component + output->num_components > 4 ||
          !info->stride[output->output_buffer] ||
          output->dst_offset + output->num_components >
             info->stride[output->output_buffer])
         return false;
      used_buffers |= BITFIELD_BIT(output->output_buffer);
   }
   /* A buffer may declare xfb_stride without capturing anything (GL 4.4
    * enhanced layouts); it is simply never written. */
   (void)used_buffers;
   return true;
}

static nir_shader *
ps5_stream_output_carrier_nir(const nir_shader *source)
{
   nir_shader *carrier = nir_shader_clone(NULL, source);

   if (!carrier)
      return NULL;
   ralloc_free(carrier->xfb_info);
   carrier->xfb_info = NULL;
   carrier->info.has_transform_feedback_varyings = false;
   memset(carrier->info.xfb_stride, 0, sizeof(carrier->info.xfb_stride));
   return carrier;
}

static bool
ps5_stream_output_lower_instance_id(nir_builder *builder,
                                    nir_intrinsic_instr *intrinsic,
                                    void *data)
{
   nir_def *base_instance;
   nir_def *instance_id;

   (void)data;
   if (intrinsic->intrinsic != nir_intrinsic_load_instance_id)
      return false;

   builder->cursor = nir_after_instr(&intrinsic->instr);
   base_instance = nir_load_base_instance(builder);
   instance_id = nir_iadd(builder, &intrinsic->def, base_instance);
   nir_def_rewrite_uses_after(&intrinsic->def, instance_id);
   BITSET_SET(builder->shader->info.system_values_read,
              SYSTEM_VALUE_BASE_INSTANCE);
   return true;
}

static nir_shader *
ps5_stream_output_split_nir(const nir_shader *source)
{
   nir_shader *split = nir_shader_clone(NULL, source);

   if (!split)
      return NULL;
   nir_shader_intrinsics_pass(split, ps5_stream_output_lower_instance_id,
                              nir_metadata_control_flow, NULL);
   return split;
}

static unsigned ps5_streamout_buffer_mask(unsigned stream_buffer_mask);

static bool
ps5_stream_output_metadata_matches(
   const struct pipe_stream_output_info *info,
   const PsbcShaderMetadata *metadata)
{
   unsigned mask = 0;

   if (!metadata->streamout_valid)
      return false;
   for (unsigned index = 0; index < info->num_outputs; ++index)
      mask |= BITFIELD_BIT(info->output[index].output_buffer +
                           info->output[index].stream * PIPE_MAX_SO_BUFFERS);
   if (metadata->streamout_enabled_stream_buffers_mask != mask)
      return false;
   /* Buffers that only declare xfb_stride capture nothing and are not written. */
   for (unsigned buffer = 0; buffer < PIPE_MAX_SO_BUFFERS; ++buffer)
      if ((ps5_streamout_buffer_mask(mask) & BITFIELD_BIT(buffer)) &&
          metadata->streamout_strides_dwords[buffer] != info->stride[buffer])
         return false;
   return true;
}

static unsigned
ps5_streamout_buffer_mask(unsigned stream_buffer_mask)
{
   unsigned mask = 0;

   for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS; ++stream)
      mask |= stream_buffer_mask >> (stream * PIPE_MAX_SO_BUFFERS);
   return mask & BITFIELD_MASK(PIPE_MAX_SO_BUFFERS);
}

static unsigned
ps5_streamout_buffer_stream(unsigned stream_buffer_mask, unsigned buffer)
{
   for (unsigned stream = 0; stream < PIPE_MAX_VERTEX_STREAMS; ++stream)
      if (stream_buffer_mask &
          BITFIELD_BIT(buffer + stream * PIPE_MAX_SO_BUFFERS))
         return stream;
   return 0;
}

static bool
ps5_lower_poly_line_smooth_enabled(nir_builder *builder,
                                   nir_intrinsic_instr *intrinsic,
                                   void *data)
{
   (void)data;
   if (intrinsic->intrinsic !=
       nir_intrinsic_load_poly_line_smooth_enabled)
      return false;

   builder->cursor = nir_before_instr(&intrinsic->instr);
   nir_def_replace(&intrinsic->def, nir_imm_true(builder));
   return true;
}

static bool
ps5_remove_sample_mask(nir_builder *builder, nir_intrinsic_instr *intrinsic, void *data)
{
   (void)builder;
   (void)data;
   if (intrinsic->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intrinsic).location != FRAG_RESULT_SAMPLE_MASK)
      return false;
   nir_instr_remove(&intrinsic->instr);
   return true;
}

static bool
ps5_select_shader_variant(struct ps5_shader *shader, uint32_t address32_hi,
                          const struct ps5_vertex_layout *layout,
                          uint32_t primitive_type,
                          bool provoking_vtx_last, bool alpha_to_one,
                          bool poly_line_smooth,
                          bool omit_implicit_primitive_id,
                          bool primitive_id_per_primitive,
                          int flat_input_vertex,
                          const struct ps5_fragment_exports *exports)
{
   const struct ps5_fragment_exports no_exports = {.color_mask = 1};
   const bool broadcast_color = shader->stage == PSBC_STAGE_FRAGMENT &&
      (shader->nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_COLOR));
   struct ps5_shader_variant *variant;
   PsbcCompileOptions options;
   PsbcResult result;
   nir_shader *carrier = NULL;
   nir_shader *variant_nir = NULL;
   nir_shader *streamout_nir = NULL;
   const nir_shader *package_nir;

   if (!exports)
      exports = &no_exports;
   const bool remove_sample_mask = shader->stage == PSBC_STAGE_FRAGMENT &&
      exports->ignore_sample_mask &&
      (shader->nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK));
   for (variant = shader->variants; variant; variant = variant->next) {
      if (variant->primitive_type == primitive_type &&
          variant->provoking_vtx_last == provoking_vtx_last &&
          variant->alpha_to_one == alpha_to_one &&
          variant->poly_line_smooth == poly_line_smooth &&
          variant->omit_implicit_primitive_id == omit_implicit_primitive_id &&
          variant->primitive_id_per_primitive == primitive_id_per_primitive &&
          variant->flat_input_vertex == flat_input_vertex &&
          variant->exports.formats == exports->formats &&
          variant->exports.int8_mask == exports->int8_mask &&
          variant->exports.int10_mask == exports->int10_mask &&
          variant->exports.color_mask == exports->color_mask &&
          variant->exports.ignore_sample_mask == exports->ignore_sample_mask &&
          variant->exports.rasterization_samples == exports->rasterization_samples &&
          variant->exports.ps_iter_samples == exports->ps_iter_samples &&
          !memcmp(&variant->layout, layout, sizeof(*layout))) {
         shader->active = variant;
         return true;
      }
   }

   variant = calloc(1, sizeof(*variant));
   if (!variant)
      return false;
   variant->layout = *layout;
   variant->primitive_type = primitive_type;
   variant->provoking_vtx_last = provoking_vtx_last;
   variant->alpha_to_one = alpha_to_one;
   variant->poly_line_smooth = poly_line_smooth;
   variant->omit_implicit_primitive_id = omit_implicit_primitive_id;
   variant->primitive_id_per_primitive = primitive_id_per_primitive;
   variant->flat_input_vertex = flat_input_vertex;
   variant->exports = *exports;
   if (!ps5_shader_compile_options(shader, address32_hi, layout,
                                   primitive_type, provoking_vtx_last,
                                   &options)) {
      free(variant);
      return false;
   }
   options.spi_shader_col_format = exports->formats;
   options.omit_implicit_primitive_id = omit_implicit_primitive_id;
   options.primitive_id_per_primitive = primitive_id_per_primitive;
   options.flat_input_vertex_valid = flat_input_vertex >= 0;
   options.flat_input_vertex = flat_input_vertex >= 0 ? flat_input_vertex : 0;
   options.color_is_int8 = exports->int8_mask;
   options.color_is_int10 = exports->int10_mask;
   if (shader->stage == PSBC_STAGE_FRAGMENT) {
      options.rasterization_samples = poly_line_smooth ? 4 : exports->rasterization_samples;
      options.ps_iter_samples = exports->ps_iter_samples;
   }
   package_nir = shader->nir;
   if (shader->stage == PSBC_STAGE_FRAGMENT || alpha_to_one || poly_line_smooth) {
      if (shader->stage != PSBC_STAGE_FRAGMENT ||
          !(variant_nir = nir_shader_clone(NULL, shader->nir))) {
         free(variant);
         return false;
      }
      nir_shader_intrinsics_pass(variant_nir, ps5_lower_sample_interpolation,
                                 nir_metadata_control_flow,
                                 &options.rasterization_samples);
      /* OpenGL ignores shader sample-mask writes without multisampling.
       * Remove user writes before optional line/polygon smoothing adds its own. */
      if (remove_sample_mask) {
         nir_shader_intrinsics_pass(variant_nir, ps5_remove_sample_mask,
                                    nir_metadata_control_flow, NULL);
         nir_shader_gather_info(variant_nir, nir_shader_get_entrypoint(variant_nir));
      }
      if (broadcast_color) {
         unsigned mask = exports->color_mask;
         nir_shader_instructions_pass(variant_nir, ps5_lower_fragment_color,
                                      nir_metadata_control_flow, &mask);
         nir_shader_gather_info(variant_nir,
                                nir_shader_get_entrypoint(variant_nir));
      }
      if (alpha_to_one)
         nir_lower_alpha_to_one(variant_nir);
      if (poly_line_smooth) {
         nir_lower_poly_line_smooth(variant_nir, 4);
         nir_shader_intrinsics_pass(
            variant_nir, ps5_lower_poly_line_smooth_enabled,
            nir_metadata_control_flow, NULL);
      }
      package_nir = variant_nir;
   }
   if (shader->stream_output.num_outputs) {
      carrier = ps5_stream_output_carrier_nir(package_nir);
      if (!carrier) {
         ralloc_free(variant_nir);
         free(variant);
         return false;
      }
      package_nir = carrier;
   }
   result = ps5_compile_cached_nir(package_nir, &options, &variant->output);
   ralloc_free(carrier);
   ralloc_free(variant_nir);
   printf("[ps5-gallium] compile-shader stage=%u primitive=%u provoking-last=%u smooth=%u variant-attrs=%u result=%d bytes=%zu hash=%08x user-sgprs=%u scratch=%u/%u table=%u\n",
          shader->stage, primitive_type, provoking_vtx_last,
          poly_line_smooth, layout->count,
          result,
          variant->output.machine_code_size,
          ps5_hash32(variant->output.machine_code,
                     variant->output.machine_code_size),
          variant->output.metadata.user_sgpr_count,
          variant->output.metadata.scratch_bytes_per_wave,
          variant->output.metadata.scratch_size_per_thread,
          variant->output.metadata.scratch_buffer_table_user_data_dword);
   if (variant->output.metadata.input_semantic_count)
      printf("[ps5-gallium] input-semantics count=%u first=%08x\n",
             variant->output.metadata.input_semantic_count,
             variant->output.metadata.input_semantics[0]);
   if (result != PSBC_RESULT_OK ||
       ps5_agc_package_build(&variant->output, 4, &variant->package,
                             &variant->package_size) != 0) {
      psbc_free_output(&variant->output);
      free(variant);
      return false;
   }
   if (shader->stream_output.num_outputs) {
      streamout_nir = shader->stage == PSBC_STAGE_VERTEX
         ? nir_shader_clone(NULL, shader->nir)
         : ps5_stream_output_split_nir(shader->nir);
      if (!streamout_nir) {
         free(variant->package);
         psbc_free_output(&variant->output);
         free(variant);
         return false;
      }
      options.split_vertex_instances = shader->stage == PSBC_STAGE_VERTEX;
      result = ps5_compile_cached_nir(streamout_nir, &options,
                                &variant->streamout_output);
      ralloc_free(streamout_nir);
      printf("[ps5-gallium] compile-streamout result=%d bytes=%zu hash=%08x user-sgprs=%u mask=%x instance-base=%u/%u\n",
             result, variant->streamout_output.machine_code_size,
             ps5_hash32(variant->streamout_output.machine_code,
                        variant->streamout_output.machine_code_size),
             variant->streamout_output.metadata.user_sgpr_count,
             variant->streamout_output.metadata
                .streamout_enabled_stream_buffers_mask,
             variant->streamout_output.metadata.start_instance_valid,
             variant->streamout_output.metadata
                .start_instance_user_data_dword);
      if (result != PSBC_RESULT_OK ||
          !ps5_stream_output_metadata_matches(
             &shader->stream_output,
             &variant->streamout_output.metadata) ||
          ps5_agc_package_build(&variant->streamout_output, 4,
                                &variant->streamout_package,
                                &variant->streamout_package_size) != 0) {
         free(variant->package);
         free(variant->streamout_package);
         psbc_free_output(&variant->streamout_output);
         psbc_free_output(&variant->output);
         free(variant);
         return false;
      }
   }
   variant->next = shader->variants;
   shader->variants = variant;
   shader->active = variant;
   return true;
}

static void
ps5_release_geometry_pipeline(struct ps5_context *context)
{
   free(context->geometry_package);
   context->geometry_package = NULL;
   context->geometry_package_size = 0;
   psbc_free_output(&context->geometry_output);
   free(context->geometry_streamout_package);
   context->geometry_streamout_package = NULL;
   context->geometry_streamout_package_size = 0;
   psbc_free_output(&context->geometry_streamout_output);
   free(context->geometry_layout);
   context->geometry_layout = NULL;
   context->geometry_primitive_type = 0;
   context->geometry_provoking_vtx_last = false;
   context->geometry_vs = NULL;
   context->geometry_gs = NULL;
}

static uint32_t
ps5_tessellation_gl_tf_param(uint32_t tf_param)
{
   /* PSBC uses RADV's upper-left domain; OpenGL uses lower-left, as in
    * radeonsi's tessellator topology setup. Points and lines are unchanged. */
   unsigned topology = G_028B6C_TOPOLOGY(tf_param);
   if (topology == V_028B6C_OUTPUT_TRIANGLE_CW ||
       topology == V_028B6C_OUTPUT_TRIANGLE_CCW)
      tf_param ^= S_028B6C_TOPOLOGY(1);
   return tf_param;
}

static void
ps5_release_tessellation_pipeline(struct ps5_context *context)
{
   free(context->tessellation_hs_package);
   free(context->tessellation_tes_package);
   free(context->tessellation_layout);
   psbc_free_tessellation_output(&context->tessellation_output);
   context->tessellation_hs_package = NULL;
   context->tessellation_hs_package_size = 0;
   context->tessellation_tes_package = NULL;
   context->tessellation_tes_package_size = 0;
   context->tessellation_layout = NULL;
   context->tessellation_vs = NULL;
   context->tessellation_tcs = NULL;
   context->tessellation_tes = NULL;
   context->tessellation_gs = NULL;
   context->tessellation_streamout = false;
   context->tessellation_patch_vertices = 0;
}

static bool
ps5_select_tessellation_pipeline(struct ps5_context *context,
                                 uint32_t address32_hi,
                                 const struct ps5_vertex_layout *layout)
{
   PsbcTessellationOutput output = {0};
   PsbcTessellationCompileOptions options = {
      .input_patch_vertices = context->patch_vertices,
      .offchip_workgroup_capacity_dwords = 8192,
      .address32_hi = address32_hi,
   };
   uint8_t *hs_package = NULL;
   uint8_t *tes_package = NULL;
   nir_shader *carriers[4] = {0};
   const struct ps5_shader *stages[] = {context->vs, context->tcs, context->tes, context->gs};
   const nir_shader *inputs[4] = {0};
   const unsigned final_stage = context->gs ? 3 : 2;
   const bool streamout = PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE &&
                          context->stream_output_target_count != 0;
   size_t hs_size = 0;
   size_t tes_size = 0;
   uint32_t ring_itemsize = 0;
   uint32_t geometry_primitive_type = 0;
   struct ps5_vertex_layout *saved_layout;
   PsbcResult result;

   if (!context->tcs && !context->tes)
      return true;
   /* One input primitive's output must fit one 128-thread subgroup: the
    * per-instance subgroup mode hangs the GPU behind the tessellator. */
   if (context->gs &&
       MAX2(context->gs->nir->info.gs.invocations, 1) *
          context->gs->nir->info.gs.vertices_out > 128) {
      printf("[ps5-gallium] tessellation-select reject=geometry-subgroup\n");
      return false;
   }
   if (!PS5_ENABLE_TESSELLATION_CANDIDATE || !context->tcs ||
       !context->tes || !context->patch_vertices ||
       context->patch_vertices > 32 ||
       (streamout && !stages[final_stage]->stream_output.num_outputs) ||
       (context->gs &&
        !ps5_draw_primitive(context->gs->nir->info.gs.input_primitive, 6,
                            &geometry_primitive_type)) ||
       !ps5_tessellation_buffer_layout(context, &options.vertex))
      return false;
   options.vertex.target = PSBC_TARGET_PS5;
   options.vertex.stage = PSBC_STAGE_VERTEX;
   options.vertex.entrypoint = "main";
   options.vertex.optimise = true;
   options.vertex.ngg = true;
   options.vertex.address32_hi = address32_hi;
   options.vertex.primitive_type = geometry_primitive_type ? geometry_primitive_type : 4;
   options.vertex.ps5_global_streamout = streamout;
   options.vertex.ps5_global_primitive_query = true;
   options.vertex.vertex_attribute_count = layout->count;
   memcpy(options.vertex.vertex_attributes, layout->attributes,
          layout->count * sizeof(layout->attributes[0]));
   if (context->tessellation_vs == context->vs &&
       context->tessellation_tcs == context->tcs &&
       context->tessellation_tes == context->tes &&
       context->tessellation_gs == context->gs &&
       context->tessellation_streamout == streamout &&
       context->tessellation_patch_vertices == context->patch_vertices &&
       context->tessellation_layout &&
       !memcmp(context->tessellation_layout, layout, sizeof(*layout)))
      return true;

   /* Separable programs may all declare XFB; only the last stage captures. */
   for (unsigned stage = 0; stage <= final_stage; ++stage) {
      inputs[stage] = stages[stage]->nir;
      if ((!streamout || stage != final_stage) &&
          (inputs[stage]->xfb_info || inputs[stage]->info.has_transform_feedback_varyings)) {
         carriers[stage] = ps5_stream_output_carrier_nir(inputs[stage]);
         if (!carriers[stage]) {
            for (unsigned i = 0; i < 4; ++i)
               ralloc_free(carriers[i]);
            return false;
         }
         inputs[stage] = carriers[stage];
      }
      if (ps5_shader_texture_count(stages[stage])) {
         if (!carriers[stage])
            carriers[stage] = nir_shader_clone(NULL, inputs[stage]);
         struct ps5_texture_offset_state remap = {
            .first = PS5_TESSELLATION_TEXTURE_BINDING + stage * PS5_MAX_TEXTURE_UNITS,
            .valid = true,
         };
         if (carriers[stage])
            nir_shader_instructions_pass(carriers[stage], ps5_offset_geometry_texture,
                                         nir_metadata_control_flow, &remap);
         if (!carriers[stage] || !remap.valid) {
            for (unsigned i = 0; i < 4; ++i)
               ralloc_free(carriers[i]);
            return false;
         }
         carriers[stage]->info.num_textures += remap.first;
         inputs[stage] = carriers[stage];
      }
   }

   result = psbc_compile_nir_tessellation_pipeline(
      inputs[0], inputs[1], inputs[2], inputs[3],
      &options, &output);
#ifndef PS5_RUNTIME_QUIET
   if (result == PSBC_RESULT_INVALID_ARGUMENT) {
      for (unsigned stage = 0; stage < 4; ++stage)
         if (inputs[stage]) {
            printf("[ps5-gallium] rejected-tessellation-input stage=%u\n", stage);
            nir_print_shader((nir_shader *)inputs[stage], stdout);
         }
   }
#endif
   for (unsigned stage = 0; stage < 4; ++stage)
      ralloc_free(carriers[stage]);
   printf("[ps5-gallium] compile-tessellation result=%d hs=%zu final=%zu gs=%u patches=%u\n",
          result, output.hs.machine_code_size, output.tes.machine_code_size,
          context->gs != NULL, output.runtime.num_patches);
   if (context->gs && result == PSBC_RESULT_OK &&
       !ps5_geometry_ring_itemsize(&output.tes.metadata, &ring_itemsize))
      result = PSBC_RESULT_INTERNAL_ERROR;
   if (result != PSBC_RESULT_OK || !output.runtime.valid ||
       (streamout && !ps5_stream_output_metadata_matches(
          &stages[final_stage]->stream_output, &output.tes.metadata)) ||
       ps5_agc_package_build(&output.hs, 0, &hs_package, &hs_size) ||
       ps5_agc_package_build(&output.tes, ring_itemsize,
                             &tes_package, &tes_size)) {
      free(hs_package);
      free(tes_package);
      psbc_free_tessellation_output(&output);
      return false;
   }
   saved_layout = malloc(sizeof(*saved_layout));
   if (!saved_layout) {
      free(hs_package);
      free(tes_package);
      psbc_free_tessellation_output(&output);
      return false;
   }
   *saved_layout = *layout;
   ps5_release_tessellation_pipeline(context);
   context->tessellation_vs = context->vs;
   context->tessellation_tcs = context->tcs;
   context->tessellation_tes = context->tes;
   context->tessellation_gs = context->gs;
   context->tessellation_streamout = streamout;
   context->tessellation_patch_vertices = context->patch_vertices;
   context->tessellation_layout = saved_layout;
   output.runtime.tf_param = ps5_tessellation_gl_tf_param(output.runtime.tf_param);
   context->tessellation_output = output;
   context->tessellation_hs_package = hs_package;
   context->tessellation_hs_package_size = hs_size;
   context->tessellation_tes_package = tes_package;
   context->tessellation_tes_package_size = tes_size;
   return true;
}

static bool
ps5_geometry_ring_itemsize(const PsbcShaderMetadata *metadata,
                           uint32_t *itemsize)
{
   for (uint32_t i = 0; i < metadata->context_register_count; ++i) {
      if (metadata->context_registers[i].offset == UINT16_C(0x2ab)) {
         /* RADV lowers the logical ESGS stride to an immediate on GFX9+.
          * Keep the hardware multiplier at one or GS vertex offsets are
          * scaled once by VGT and again by the generated shader. */
         *itemsize = 1u;
         return true;
      }
   }
   return false;
}

/* Say which check refused a geometry pipeline (the draw is dropped). */
static bool
ps5_geometry_reject(unsigned line)
{
   printf("[ps5-gallium] geometry-select reject at ps5_screen.c:%u\n", line);
   return false;
}

static bool
ps5_select_geometry_pipeline(struct ps5_context *context,
                             uint32_t address32_hi,
                             const struct ps5_vertex_layout *layout,
                             uint32_t primitive_type)
{
   PsbcCompileOptions options;
   PsbcShaderOutput output = {0};
   PsbcShaderOutput streamout_output = {0};
   nir_shader *geometry_nir = NULL;
   nir_shader *carrier_nir = NULL;
   const nir_shader *package_nir;
   struct ps5_ubo_offset_state ubo_offset = {0};
   uint8_t *package = NULL;
   uint8_t *streamout_package = NULL;
   size_t package_size = 0;
   size_t streamout_package_size = 0;
   unsigned merged_textures;
   unsigned vertex_ubos;
   unsigned geometry_ubos;
   unsigned vertex_ssbos;
   unsigned geometry_ssbos;
   unsigned vertex_images;
   unsigned geometry_images;
   unsigned vertex_ubo_bindings;
   bool merged_storage;
   struct ps5_texture_offset_state texture_offset = {.first = PS5_MAX_TEXTURE_UNITS, .valid = true};
   uint32_t ring_itemsize;
   uint32_t streamout_ring_itemsize;
   PsbcResult result;
   bool provoking_vtx_last = context->rasterizer &&
                              context->rasterizer->flatshade_first;

   if (!context->gs)
      return true;
   if (!PS5_ENABLE_GEOMETRY_CANDIDATE)
      return ps5_geometry_reject(__LINE__);
   if (context->geometry_vs == context->vs &&
       context->geometry_gs == context->gs && context->geometry_layout &&
       context->geometry_primitive_type == primitive_type &&
       context->geometry_provoking_vtx_last == provoking_vtx_last &&
       !memcmp(context->geometry_layout, layout, sizeof(*layout)))
      return true;
   if (!ps5_shader_compile_options(context->vs, address32_hi, layout,
                                   primitive_type,
                                   provoking_vtx_last, &options)) {
      printf("[ps5-gallium] geometry-select reject=vertex-options textures=%u ubos=%u\n",
             context->vs->nir->info.num_textures,
             context->vs->nir->info.num_ubos);
      return ps5_geometry_reject(__LINE__);
   }
   vertex_ubos = context->vs->nir->info.num_ubos;
   geometry_ubos = context->gs->nir->info.num_ubos;
   vertex_ssbos = context->vs->nir->info.num_ssbos;
   geometry_ssbos = context->gs->nir->info.num_ssbos;
   vertex_images = context->vs->nir->info.num_images;
   geometry_images = context->gs->nir->info.num_images;
   merged_storage = ps5_geometry_uses_buffer_arrays(context);
   vertex_ubo_bindings = vertex_ubos
      ? ((vertex_ssbos || vertex_images ||
          ps5_shader_has_indirect_ubo(context->vs))
            ? 1u : vertex_ubos)
      : 0u;
   if (geometry_images > PS5_COMPUTE_IMAGE_SLOTS ||
       geometry_ssbos > PS5_COMPUTE_STORAGE_SLOTS ||
       vertex_ubos + geometry_ubos > 2u * PS5_MAX_CONSTANT_BUFFERS)
      return ps5_geometry_reject(__LINE__);
   if (merged_storage) {
      if (options.descriptor_binding_count < vertex_ubo_bindings)
         return ps5_geometry_reject(__LINE__);
      options.descriptor_binding_count -= vertex_ubo_bindings;
      options.gallium_buffer_arrays = true;
      if (geometry_ssbos) {
         if (options.descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
            return ps5_geometry_reject(__LINE__);
         options.descriptor_bindings[options.descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding =
                  PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_GEOMETRY),
               .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
               .array_size = PS5_COMPUTE_STORAGE_SLOTS,
               .offset = PS5_GEOMETRY_STORAGE_OFFSET,
               .stride = 16,
            };
      }
      if (geometry_images) {
         if (options.descriptor_binding_count >= PSBC_MAX_DESCRIPTOR_BINDINGS)
            return ps5_geometry_reject(__LINE__);
         options.descriptor_bindings[options.descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding =
                  PSBC_GALLIUM_IMAGE_ARRAY_BINDING(PSBC_STAGE_GEOMETRY),
               .type = PSBC_DESCRIPTOR_STORAGE_IMAGE,
               .array_size = PS5_COMPUTE_IMAGE_SLOTS,
               .offset = PS5_GEOMETRY_IMAGE_OFFSET,
               .stride = 32,
            };
      }
   }
   merged_textures = ps5_shader_texture_count(context->vs);
   for (unsigned binding = 0; binding < PS5_MAX_TEXTURE_UNITS; ++binding) {
      if (!BITSET_TEST(context->gs->nir->info.textures_used, binding))
         continue;
      if (!ps5_append_texture_descriptor(&options,
                                         PS5_MAX_TEXTURE_UNITS + binding, 0)) {
         printf("[ps5-gallium] geometry-select reject=texture-descriptor binding=%u\n",
                binding);
         return ps5_geometry_reject(__LINE__);
      }
      merged_textures++;
   }
   if (merged_textures != ps5_texture_count(
                             context, context->vs,
                             &context->geometry_output.metadata) ||
       options.descriptor_binding_count !=
          merged_textures + (vertex_ssbos != 0) + (geometry_ssbos != 0) +
             (vertex_images != 0) + (geometry_images != 0) +
             (merged_storage ? 0 : vertex_ubo_bindings)) {
      printf("[ps5-gallium] geometry-select reject=texture-count vertex=%u geometry=%u merged=%u descriptors=%u\n",
             context->vs->nir->info.num_textures,
             context->gs->nir->info.num_textures, merged_textures,
             options.descriptor_binding_count);
      return ps5_geometry_reject(__LINE__);
   }
   if (merged_storage) {
      if (options.descriptor_binding_count + (vertex_ubos != 0) +
             (geometry_ubos != 0) > PSBC_MAX_DESCRIPTOR_BINDINGS)
         return ps5_geometry_reject(__LINE__);
      if (vertex_ubos)
         options.descriptor_bindings[options.descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_VERTEX),
               .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER,
               .array_size = vertex_ubos,
               .offset = PS5_TEXTURE_DESCRIPTOR_BYTES,
               .stride = 16,
            };
      if (geometry_ubos)
         options.descriptor_bindings[options.descriptor_binding_count++] =
            (PsbcDescriptorBinding){
               .binding = PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_GEOMETRY),
               .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER,
               .array_size = geometry_ubos,
               .offset = PS5_TEXTURE_DESCRIPTOR_BYTES + vertex_ubos * 16u,
               .stride = 16,
            };
   } else if (!ps5_append_ubo_descriptors(&options, vertex_ubos,
                                           geometry_ubos)) {
      printf("[ps5-gallium] geometry-select reject=descriptors vertex-ubos=%u geometry-ubos=%u descriptors=%u\n",
             vertex_ubos, geometry_ubos,
             options.descriptor_binding_count);
      return ps5_geometry_reject(__LINE__);
   }
   geometry_nir = nir_shader_clone(NULL, context->gs->nir);
   if (!geometry_nir) {
      printf("[ps5-gallium] geometry-select reject=clone\n");
      return ps5_geometry_reject(__LINE__);
   }
   nir_shader_instructions_pass(geometry_nir, ps5_offset_geometry_texture,
                                nir_metadata_control_flow,
                                &texture_offset);
   if (!texture_offset.valid) {
      printf("[ps5-gallium] geometry-select reject=texture-remap\n");
      ralloc_free(geometry_nir);
      return ps5_geometry_reject(__LINE__);
   }
   if (geometry_nir->info.num_textures)
      geometry_nir->info.num_textures += PS5_MAX_TEXTURE_UNITS;
   BITSET_SHL(geometry_nir->info.textures_used, PS5_MAX_TEXTURE_UNITS);
   BITSET_SHL(geometry_nir->info.textures_used_by_txf, PS5_MAX_TEXTURE_UNITS);
   BITSET_SHL(geometry_nir->info.texture_buffers, PS5_MAX_TEXTURE_UNITS);
   BITSET_SHL(geometry_nir->info.samplers_used, PS5_MAX_TEXTURE_UNITS);
   nir_foreach_uniform_variable(var, geometry_nir) {
      if (glsl_type_is_sampler(glsl_without_array(var->type)))
         var->data.binding += PS5_MAX_TEXTURE_UNITS;
   }
   ubo_offset.first = vertex_ubos;
   ubo_offset.source_count = geometry_ubos;
   ubo_offset.valid = true;
   if (geometry_ubos && !merged_storage) {
      nir_shader_instructions_pass(geometry_nir, ps5_offset_ubo_index,
                                   nir_metadata_control_flow, &ubo_offset);
      geometry_nir->info.num_ubos = vertex_ubos + geometry_ubos;
   }
   if (!ubo_offset.valid) {
      printf("[ps5-gallium] geometry-select reject=ubo-remap\n");
      ralloc_free(geometry_nir);
      return ps5_geometry_reject(__LINE__);
   }
   options.stage = PSBC_STAGE_GEOMETRY;
   options.ngg = true;
   options.ps5_global_primitive_query = true;
   package_nir = geometry_nir;
   if (context->gs->stream_output.num_outputs) {
      carrier_nir = ps5_stream_output_carrier_nir(geometry_nir);
      if (!carrier_nir) {
         ralloc_free(geometry_nir);
         return ps5_geometry_reject(__LINE__);
      }
      package_nir = carrier_nir;
   }
   const uint64_t compile_start = os_time_get_nano();
   result = psbc_compile_nir_geometry_pipeline(
      context->vs->nir, package_nir, &options, &output);
   const uint64_t compile_ns = os_time_get_nano() - compile_start;
   ralloc_free(carrier_nir);
   printf("[ps5-gallium] compile-geometry result=%d bytes=%zu hash=%08x user-sgprs=%u textures=%u/%u/%u ubos=%u/%u descriptors=%u set0=%u/%u address32=%08x compile_ns=%llu\n",
          result, output.machine_code_size,
          ps5_hash32(output.machine_code, output.machine_code_size),
          output.metadata.user_sgpr_count,
          context->vs->nir->info.num_textures,
          context->gs->nir->info.num_textures, merged_textures,
          vertex_ubos, geometry_ubos,
          output.metadata.descriptor_binding_count,
          output.metadata.descriptor_set0_valid,
          output.metadata.descriptor_set0_user_data_dword,
          output.metadata.address32_hi, (unsigned long long)compile_ns);
   if (result != PSBC_RESULT_OK ||
       !ps5_geometry_ring_itemsize(&output.metadata, &ring_itemsize) ||
       ps5_agc_package_build(&output, ring_itemsize, &package,
                             &package_size) != 0) {
      ralloc_free(geometry_nir);
      free(package);
      psbc_free_output(&output);
      return ps5_geometry_reject(__LINE__);
   }
   if (context->gs->stream_output.num_outputs) {
      options.ps5_global_streamout = true;
      result = psbc_compile_nir_geometry_pipeline(
         context->vs->nir, geometry_nir, &options, &streamout_output);
      printf("[ps5-gallium] compile-geometry-streamout result=%d bytes=%zu hash=%08x user-sgprs=%u mask=%x\n",
             result, streamout_output.machine_code_size,
             ps5_hash32(streamout_output.machine_code,
                        streamout_output.machine_code_size),
             streamout_output.metadata.user_sgpr_count,
             streamout_output.metadata
                .streamout_enabled_stream_buffers_mask);
      if (result != PSBC_RESULT_OK ||
          !ps5_stream_output_metadata_matches(
             &context->gs->stream_output, &streamout_output.metadata) ||
          !ps5_geometry_ring_itemsize(&streamout_output.metadata,
                                      &streamout_ring_itemsize) ||
          ps5_agc_package_build(&streamout_output,
                                streamout_ring_itemsize,
                                &streamout_package,
                                &streamout_package_size) != 0) {
         ralloc_free(geometry_nir);
         free(package);
         free(streamout_package);
         psbc_free_output(&output);
         psbc_free_output(&streamout_output);
         return ps5_geometry_reject(__LINE__);
      }
   }
   ralloc_free(geometry_nir);
   ps5_release_geometry_pipeline(context);
   context->geometry_layout = malloc(sizeof(*layout));
   if (!context->geometry_layout) {
      free(package);
      free(streamout_package);
      psbc_free_output(&output);
      psbc_free_output(&streamout_output);
      return ps5_geometry_reject(__LINE__);
   }
   *context->geometry_layout = *layout;
   context->geometry_vs = context->vs;
   context->geometry_gs = context->gs;
   context->geometry_primitive_type = primitive_type;
   context->geometry_provoking_vtx_last = provoking_vtx_last;
   context->geometry_output = output;
   context->geometry_package = package;
   context->geometry_package_size = package_size;
   context->geometry_streamout_output = streamout_output;
   context->geometry_streamout_package = streamout_package;
   context->geometry_streamout_package_size = streamout_package_size;
   return true;
}

#ifdef PS5_NATIVE_TITLE_RUNTIME
/* ponytail: provably bounded LODs only; unknown ranges remain gated.
 * expanded operations need their own sampler and mip/view qualification. */
static bool
ps5_compute_texture_usage(nir_shader *nir, unsigned *used, unsigned *buffers,
                          unsigned *filtered,
                          unsigned max_lod[PS5_COMPUTE_TEXTURE_SLOTS],
                          unsigned *arrays,
                          uint8_t binding_size[PS5_COMPUTE_TEXTURE_SLOTS])
{
   *used = 0;
   *buffers = 0;
   *filtered = 0;
   *arrays = 0;
   memset(max_lod, 0, PS5_COMPUTE_TEXTURE_SLOTS * sizeof(*max_lod));
   memset(binding_size, 0, PS5_COMPUTE_TEXTURE_SLOTS * sizeof(*binding_size));
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            const bool buffer = tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;
            if (buffer) {
               if ((tex->op != nir_texop_txf && tex->op != nir_texop_txs) ||
                   tex->is_array || tex->is_shadow ||
                   tex->texture_index >= PS5_COMPUTE_TEXTURE_SLOTS ||
                   tex->def.bit_size != 32)
                  return false;
               unsigned coords = 0, lods = 0, array_size = 1;
               for (unsigned i = 0; i < tex->num_srcs; ++i) {
                  nir_src source = tex->src[i].src;
                  if (tex->src[i].src_type == nir_tex_src_coord) {
                     if (tex->op != nir_texop_txf || tex->coord_components != 1 ||
                         source.ssa->num_components != 1 || source.ssa->bit_size != 32)
                        return false;
                     ++coords;
                  } else if (tex->src[i].src_type == nir_tex_src_lod) {
                     if (source.ssa->num_components != 1 || source.ssa->bit_size != 32 ||
                         !nir_src_is_const(source) || nir_src_as_uint(source))
                        return false;
                     ++lods;
                  } else if (tex->src[i].src_type == nir_tex_src_texture_offset) {
                     if (source.ssa->num_components != 1 || source.ssa->bit_size != 32)
                        return false;
                     unsigned ceiling;
                     if (nir_src_is_const(source)) {
                        ceiling = nir_src_as_uint(source);
                     } else {
                        struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
                        if (!ranges) return false;
                        ceiling = nir_unsigned_upper_bound(nir, ranges,
                           nir_get_scalar(source.ssa, 0));
                        _mesa_hash_table_destroy(ranges, NULL);
                     }
                     if (ceiling >= PS5_COMPUTE_TEXTURE_SLOTS - tex->texture_index)
                        return false;
                     array_size = MAX2(array_size, ceiling + 1);
                  } else if (tex->src[i].src_type == nir_tex_src_sampler_offset) {
                     if (source.ssa->num_components != 1 ||
                         source.ssa->bit_size != 32)
                        return false;
                  } else {
                     return false;
                  }
               }
               if (coords != (tex->op == nir_texop_txf) || lods != 1)
                  return false;
               const unsigned mask = BITFIELD_RANGE(tex->texture_index, array_size);
               if ((*used & mask) && binding_size[tex->texture_index] != array_size)
                  return false;
               *used |= mask;
               *buffers |= mask;
               binding_size[tex->texture_index] = array_size;
               continue;
            }
            if (tex->sampler_dim == GLSL_SAMPLER_DIM_MS) {
               const bool query = tex->op == nir_texop_texture_samples;
               if ((!query && tex->op != nir_texop_txf_ms) || tex->is_shadow ||
                   tex->texture_index >= PS5_COMPUTE_TEXTURE_SLOTS || tex->def.bit_size != 32 ||
                   tex->num_srcs != (query ? 0 : 2) ||
                   (query ? tex->def.num_components != 1 : tex->coord_components != 2 + tex->is_array))
                  return false;
               unsigned coords = 0, samples = 0;
               for (unsigned i = 0; i < tex->num_srcs; ++i) {
                  nir_src src = tex->src[i].src;
                  if (src.ssa->bit_size != 32) return false;
                  if (tex->src[i].src_type == nir_tex_src_coord) {
                     if (src.ssa->num_components != tex->coord_components) return false;
                     ++coords;
                  } else if (tex->src[i].src_type == nir_tex_src_ms_index) {
                     if (src.ssa->num_components != 1 || !nir_src_is_const(src) || nir_src_as_uint(src) > 3)
                        return false;
                     ++samples;
                  } else return false;
               }
               if (coords != !query || samples != !query) return false;
               const unsigned bit = 1u << tex->texture_index;
               if ((*used & bit) && (((*arrays & bit) != 0) != tex->is_array)) return false;
               *used |= bit;
               if (tex->is_array) *arrays |= bit;
               binding_size[tex->texture_index] = 1;
               continue;
            }
            const bool volume = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
            const bool gather = tex->op == nir_texop_tg4;
            const bool gradient = tex->op == nir_texop_txd;
            const bool shadow_compare = tex->is_shadow &&
                                        (gather || tex->op == nir_texop_txl);
            const unsigned dimensions = tex->sampler_dim == GLSL_SAMPLER_DIM_1D ? 1 :
               tex->sampler_dim == GLSL_SAMPLER_DIM_2D ? 2 :
               (volume || tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) ? 3 : 0;
            /* Sampler arrays indexed by a (dynamically) uniform value carry
             * texture/sampler offsets; bound them like texel-buffer arrays. */
            unsigned index_sources = 0, array_size = 1;
            for (unsigned i = 0; i < tex->num_srcs; ++i) {
               if (tex->src[i].src_type != nir_tex_src_texture_offset &&
                   tex->src[i].src_type != nir_tex_src_sampler_offset)
                  continue;
               nir_src source = tex->src[i].src;
               if (source.ssa->num_components != 1 || source.ssa->bit_size != 32)
                  return false;
               ++index_sources;
               unsigned ceiling;
               if (nir_src_is_const(source)) {
                  ceiling = nir_src_as_uint(source);
               } else {
                  struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
                  if (!ranges) return false;
                  ceiling = nir_unsigned_upper_bound(nir, ranges,
                     nir_get_scalar(source.ssa, 0));
                  _mesa_hash_table_destroy(ranges, NULL);
               }
               if (ceiling >= PS5_COMPUTE_TEXTURE_SLOTS - tex->texture_index)
                  return false;
               array_size = MAX2(array_size, ceiling + 1);
            }
            if ((tex->op != nir_texop_txf && tex->op != nir_texop_txs &&
                 tex->op != nir_texop_txl && !gather && !gradient) ||
                !dimensions || (volume && tex->is_array) ||
                (gather && tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
                           tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE) ||
                (tex->is_shadow && tex->op != nir_texop_txs &&
                 (!shadow_compare || tex->dest_type != nir_type_float32)) ||
                tex->texture_index >= PS5_COMPUTE_TEXTURE_SLOTS ||
                tex->def.bit_size != 32 ||
                tex->num_srcs != (tex->op == nir_texop_txs || gather ? 1 :
                                  gradient ? 3 : 2) +
                   shadow_compare + index_sources)
               return false;
            if ((tex->op == nir_texop_txl || gather || gradient) &&
                (tex->sampler_index != tex->texture_index ||
                (tex->dest_type != nir_type_float32 && tex->dest_type != nir_type_int32 &&
                 tex->dest_type != nir_type_uint32)))
               return false;
            unsigned coords = 0, lods = 0, gradients = 0, comparators = 0;
            for (unsigned i = 0; i < tex->num_srcs; ++i) {
               if (tex->src[i].src_type == nir_tex_src_coord) {
                  if (tex->op == nir_texop_txs || tex->coord_components != dimensions + tex->is_array ||
                      tex->src[i].src.ssa->num_components != tex->coord_components ||
                      tex->src[i].src.ssa->bit_size != 32)
                     return false;
                  ++coords;
               } else if (tex->src[i].src_type == nir_tex_src_comparator) {
                  if (!shadow_compare || tex->src[i].src.ssa->num_components != 1 ||
                      tex->src[i].src.ssa->bit_size != 32)
                     return false;
                  ++comparators;
               } else if (tex->src[i].src_type == nir_tex_src_lod) {
                  nir_src source = tex->src[i].src;
                  if (source.ssa->num_components != 1 || source.ssa->bit_size != 32)
                     return false;
                  unsigned ceiling;
                  if (nir_src_is_const(source)) {
                     float lod = tex->op == nir_texop_txl ? nir_src_as_float(source) :
                                                          (float)nir_src_as_uint(source);
                     if (!(lod >= 0 && lod <= 15)) return false;
                     ceiling = (unsigned)lod + (lod > (unsigned)lod);
                  } else {
                     struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
                     if (!ranges) return false;
                     ceiling = nir_unsigned_upper_bound(nir, ranges, nir_get_scalar(source.ssa, 0));
                     _mesa_hash_table_destroy(ranges, NULL);
                     if (tex->op == nir_texop_txl) {
                        /* Positive finite IEEE floats have ordered unsigned bits.
                         * A raw-bit upper bound <= bits(15.0) also excludes signs,
                         * infinities and NaNs; unknown ranges fail conservatively. */
                        if (ceiling == UINT_MAX) {
                           ceiling = UINT_MAX;
                        } else {
                           if (ceiling > UINT32_C(0x41700000)) return false;
                           float lod; memcpy(&lod, &ceiling, sizeof(lod));
                           ceiling = (unsigned)lod + (lod > (unsigned)lod);
                        }
                     } else if (ceiling > 15) return false;
                  }
                  max_lod[tex->texture_index] = MAX2(max_lod[tex->texture_index], ceiling);
                  ++lods;
               } else if (tex->src[i].src_type == nir_tex_src_ddx ||
                          tex->src[i].src_type == nir_tex_src_ddy) {
                  if (!gradient || tex->src[i].src.ssa->num_components != dimensions ||
                      tex->src[i].src.ssa->bit_size != 32)
                     return false;
                  ++gradients;
               } else if (tex->src[i].src_type != nir_tex_src_texture_offset &&
                          tex->src[i].src_type != nir_tex_src_sampler_offset) {
                  return false;
               }
            }
            if (comparators != shadow_compare ||
                lods != (!gather && !gradient) || gradients != 2u * gradient ||
                coords != (tex->op == nir_texop_txs ? 0u : 1u))
               return false;
            const unsigned mask = BITFIELD_RANGE(tex->texture_index, array_size);
            if ((*used & mask) &&
                ((binding_size[tex->texture_index] != array_size) ||
                 (((*arrays & mask) != 0) != tex->is_array)))
               return false;
            *used |= mask;
            binding_size[tex->texture_index] = array_size;
            if (tex->is_array) *arrays |= mask;
            if (gradient)
               max_lod[tex->texture_index] = UINT_MAX;
            for (unsigned element = 1; element < array_size; ++element)
               max_lod[tex->texture_index + element] = max_lod[tex->texture_index];
            if (tex->op == nir_texop_txl || gather || gradient)
               *filtered |= mask;
         }
      }
   }
   return true;
}

static bool
ps5_compute_lower_texture_offset(const nir_instr *instr, const void *data)
{
   (void)data;
   return instr->type == nir_instr_type_tex;
}

/* Internal compute bring-up only: fixed groups and buffer resources. Public compute
 * caps stay off until images, public barriers and limits
 * are complete. Reuse the synchronous native submission owner. */
static void *
ps5_create_compute_state(struct pipe_context *base,
                         const struct pipe_compute_state *templ)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_compute_shader *shader = NULL;
   if (!templ || templ->ir_type != PIPE_SHADER_IR_NIR || !templ->prog)
      return NULL;
   nir_shader *nir = (nir_shader *)templ->prog; /* Gallium transfers ownership. */
   unsigned textures = 0, texture_buffers = 0, filtered = 0;
   unsigned max_lod[PS5_COMPUTE_TEXTURE_SLOTS], arrays = 0;
   uint8_t texture_binding_size[PS5_COMPUTE_TEXTURE_SLOTS];
   if (nir->info.stage != MESA_SHADER_COMPUTE)
      goto cleanup;
   /* Ordinary compute texture() has no implicit derivatives. Normalize it
    * before the usage validator requires an explicit, bounded LOD. */
   const nir_lower_tex_options tex_options = {
      .lower_invalid_implicit_lod = true, .lower_txp = ~0u,
      .lower_txp_array = true, .lower_txf_offset = true,
      .lower_tg4_offsets = true,
      .lower_rect_offset = true, .lower_rect = true,
      .lower_offset_filter = ps5_compute_lower_texture_offset,
   };
   nir_lower_tex(nir, &tex_options);
   /* Mesa binds user blocks after CB0 even without default uniforms.
    * Normalize both forms before checking the reserved-slot capacity. */
   nir_lower_uniforms_to_ubo(nir, false, false);
   const nir_lower_compute_system_values_options cs_options = {
      .lower_local_invocation_index = true,
   };
   nir_lower_compute_system_values(nir, &cs_options);
   if (exec_list_length(&nir->functions) != 1) {
      nir_lower_variable_initializers(nir, nir_var_function_temp);
      nir_lower_returns(nir);
      nir_inline_functions(nir);
      nir_opt_deref(nir);
      nir_remove_non_entrypoints(nir);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   }
   const bool texture_usage_ok = ps5_compute_texture_usage(nir, &textures,
      &texture_buffers, &filtered, max_lod, &arrays, texture_binding_size);
   if (nir->info.num_ubos > PS5_COMPUTE_CONSTANT_SLOTS ||
       nir->info.num_images > PS5_COMPUTE_IMAGE_SLOTS ||
       nir->info.num_textures > PS5_COMPUTE_TEXTURE_SLOTS ||
       nir->info.num_ssbos > PS5_COMPUTE_STORAGE_SLOTS ||
       !texture_usage_ok) {
      printf("[ps5-gallium] compute compile gated ubos=%u images=%u textures=%u ssbos=%u texture_usage=%u\n",
             nir->info.num_ubos, nir->info.num_images, nir->info.num_textures,
             nir->info.num_ssbos, texture_usage_ok);
      goto cleanup;
   }
   if (!context->compute_descriptors) {
      struct pipe_resource desc = {
         .target = PIPE_BUFFER, .format = PIPE_FORMAT_R8_UNORM,
         .width0 = PS5_COMPUTE_DESCRIPTOR_BYTES,
         .height0 = 1, .depth0 = 1, .array_size = 1,
         .usage = PIPE_USAGE_DEFAULT, .bind = PIPE_BIND_SHADER_BUFFER,
      };
      context->compute_descriptors = base->screen->resource_create(base->screen, &desc);
      if (!context->compute_descriptors)
         goto cleanup;
   }
   const struct ps5_resource *descriptors = (struct ps5_resource *)context->compute_descriptors;
   PsbcCompileOptions options = {
      .target = PSBC_TARGET_PS5, .stage = PSBC_STAGE_COMPUTE, .optimise = true,
      .compute_private_buffer = true, .compute_buffer_spills = true,
      .address32_hi = (uintptr_t)descriptors->data >> 32,
      .gallium_buffer_arrays = true, .descriptor_binding_count = 3,
      .descriptor_bindings = {{
         .binding = PSBC_GALLIUM_SSBO_ARRAY_BINDING(PSBC_STAGE_COMPUTE),
         .type = PSBC_DESCRIPTOR_STORAGE_BUFFER,
         .array_size = PS5_COMPUTE_STORAGE_SLOTS, .stride = 16,
      }, {
         .binding = PSBC_GALLIUM_UBO_ARRAY_BINDING(PSBC_STAGE_COMPUTE),
         .type = PSBC_DESCRIPTOR_UNIFORM_BUFFER,
         .array_size = PS5_COMPUTE_CONSTANT_SLOTS, .stride = 16,
         .offset = PS5_COMPUTE_STORAGE_SLOTS * 16,
      }, {
         .binding = PSBC_GALLIUM_IMAGE_ARRAY_BINDING(PSBC_STAGE_COMPUTE),
         .type = PSBC_DESCRIPTOR_STORAGE_IMAGE,
         .array_size = PS5_COMPUTE_IMAGE_SLOTS, .stride = 32,
         .offset = PS5_COMPUTE_BUFFER_SLOTS * 16,
      }},
   };
   for (unsigned i = 0; i < PS5_COMPUTE_TEXTURE_SLOTS; ++i) {
      if (!(textures & (1u << i)))
         continue;
      const unsigned array_size = texture_binding_size[i];
      options.descriptor_bindings[options.descriptor_binding_count++] =
         (PsbcDescriptorBinding){.binding = i,
            .type = PSBC_DESCRIPTOR_COMBINED_IMAGE_SAMPLER,
            .array_size = array_size,
            .stride = 48, .offset = PS5_COMPUTE_TEXTURE_OFFSET + i * 48};
      i += array_size - 1;
   }
   shader = calloc(1, sizeof(*shader));
   if (shader) {
      shader->ssbos = nir->info.num_ssbos;
      shader->ubos = nir->info.num_ubos;
      shader->images = nir->info.num_images;
      shader->textures = textures;
      shader->buffer_textures = texture_buffers;
      shader->filtered_textures = filtered;
      shader->array_textures = arrays;
      memcpy(shader->texture_lod, max_lod, sizeof(max_lod));
      memcpy(shader->texture_binding_size, texture_binding_size,
             sizeof(texture_binding_size));
      uint8_t *package = NULL;
      size_t package_size = 0;
      const PsbcResult compile_result = psbc_compile_nir(nir, &options,
                                                        &shader->output);
      const int package_result = compile_result == PSBC_RESULT_OK ?
         ps5_agc_package_build(&shader->output, 0, &package, &package_size) : -1;
      if (compile_result != PSBC_RESULT_OK || package_result) {
         printf("[ps5-gallium] compute compile failed psbc=%d package=%d\n",
                compile_result, package_result);
         psbc_free_output(&shader->output);
         free(shader);
         shader = NULL;
      } else {
         printf("[ps5-gallium] compute compiled ubos=%u ssbos=%u images=%u textures=%x code=%zu user-sgprs=%u descriptors=%u grid=%u private=%u scratch=%u\n",
                shader->ubos, shader->ssbos, shader->images, shader->textures,
                shader->output.machine_code_size,
                shader->output.metadata.user_sgpr_count,
                shader->output.metadata.descriptor_set0_user_data_dword,
                shader->output.metadata.compute_grid_size_user_data_dword,
                shader->output.metadata.compute_private_stride,
                shader->output.metadata.scratch_bytes_per_wave);
      }
      free(package);
   }
cleanup:
   ralloc_free(nir);
   return shader;
}

static void
ps5_bind_compute_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->cs = state;
}

static void
ps5_delete_compute_state(struct pipe_context *base, void *state)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_compute_shader *shader = state;
   if (!shader)
      return;
   if (context->cs == shader)
      context->cs = NULL;
   psbc_free_output(&shader->output);
   free(shader);
}

static void
ps5_get_compute_state_info(struct pipe_context *base, void *state,
                           struct pipe_compute_state_object_info *info)
{
   (void)base;
   struct ps5_compute_shader *shader = state;
   memset(info, 0, sizeof(*info));
   if (shader) {
      info->max_threads = 1024;
      info->preferred_simd_size = shader->output.metadata.compute_wave_size;
      info->simd_sizes = info->preferred_simd_size;
      info->private_memory = shader->output.metadata.scratch_bytes_per_wave /
                             info->preferred_simd_size;
   }
}

static void
ps5_set_shader_buffers(struct pipe_context *base, mesa_shader_stage stage,
                        unsigned start, unsigned count,
                        const struct pipe_shader_buffer *buffers, unsigned writable)
{
   struct ps5_context *context = (struct ps5_context *)base;
   bool *invalid;
   struct pipe_shader_buffer *bound_buffers;

   (void)writable; /* Conservatively flush every retained buffer for now. */
   switch (stage) {
   case MESA_SHADER_COMPUTE:
      invalid = &context->compute_bindings_invalid;
      bound_buffers = context->compute_buffers;
      break;
   case MESA_SHADER_FRAGMENT:
      invalid = &context->fragment_bindings_invalid;
      bound_buffers = context->fragment_buffers;
      break;
   case MESA_SHADER_GEOMETRY:
      invalid = &context->geometry_bindings_invalid;
      bound_buffers = context->geometry_buffers;
      break;
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
      invalid = &context->preraster_bindings_invalid[stage];
      bound_buffers = context->preraster_buffers[stage];
      break;
   default:
      return;
   }
   *invalid = true;
   if (start > PS5_COMPUTE_STORAGE_SLOTS || count > PS5_COMPUTE_STORAGE_SLOTS - start)
      return;
   for (unsigned i = 0; buffers && i < count; ++i) {
      const struct pipe_shader_buffer *b = &buffers[i];
      if (!b->buffer)
         continue;
      if (b->buffer->screen != base->screen || b->buffer->target != PIPE_BUFFER ||
          !b->buffer_size || (b->buffer_offset & 15u) ||
          b->buffer_offset > b->buffer->width0 || b->buffer_size > b->buffer->width0 - b->buffer_offset)
         return;
   }
   /* Validate the entire update before releasing any previously owned buffer. */
   /* Storage-consuming draws/dispatches finish synchronously. Deferred
    * draws exclude storage shaders and retain their other resource bindings;
    * changing a storage binding cannot invalidate a queued descriptor. */
   for (unsigned i = 0; i < count; ++i) {
      struct pipe_shader_buffer *bound = &bound_buffers[start + i];
      const struct pipe_shader_buffer *b = buffers ? &buffers[i] : NULL;
      pipe_resource_reference(&bound->buffer, b ? b->buffer : NULL);
      bound->buffer_offset = b && b->buffer ? b->buffer_offset : 0;
      bound->buffer_size = b && b->buffer ? b->buffer_size : 0;
   }
   *invalid = false;
}

static void
ps5_set_compute_constant_buffer(struct pipe_context *base, unsigned index,
                                 const struct pipe_constant_buffer *buffer)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_resource *uploaded = NULL;
   struct pipe_shader_buffer next = {0};
   if (index >= PS5_COMPUTE_CONSTANT_SLOTS)
      return;
   context->compute_constants_invalid |= 1u << index;
   if (buffer && (buffer->buffer || buffer->user_buffer)) {
      if (!buffer->buffer_size || buffer->buffer_size > PS5_MAX_CONSTANT_BUFFER_SIZE)
         return;
      next.buffer_size = buffer->buffer_size;
      if (buffer->user_buffer) {
         if (buffer->buffer || buffer->buffer_offset || !base->const_uploader)
            return;
         u_upload_data_ref(base->const_uploader, 0, next.buffer_size, 16,
                           buffer->user_buffer, &next.buffer_offset, &uploaded);
         u_upload_unmap(base->const_uploader);
         if (!uploaded)
            return;
         next.buffer = uploaded;
      } else {
         struct pipe_resource *resource = buffer->buffer;
         if (resource->screen != base->screen || resource->target != PIPE_BUFFER ||
             (buffer->buffer_offset & 15u) || buffer->buffer_offset > resource->width0 ||
             buffer->buffer_size > resource->width0 - buffer->buffer_offset)
            return;
         next.buffer = resource;
         next.buffer_offset = buffer->buffer_offset;
      }
   }
   struct pipe_shader_buffer *bound = &context->compute_buffers[PS5_COMPUTE_STORAGE_SLOTS + index];
   pipe_resource_reference(&bound->buffer, next.buffer);
   bound->buffer_offset = next.buffer_offset;
   bound->buffer_size = next.buffer_size;
   pipe_resource_reference(&uploaded, NULL);
   context->compute_constants_invalid &= ~(1u << index);
}

static void
ps5_set_shader_images(struct pipe_context *base, mesa_shader_stage stage,
                       unsigned start, unsigned count, unsigned unbind,
                       const struct pipe_image_view *images)
{
   struct ps5_context *context = (struct ps5_context *)base;
   bool *invalid;
   struct pipe_image_view *bound_images;
   if (stage == MESA_SHADER_COMPUTE) {
      invalid = &context->compute_images_invalid;
      bound_images = context->compute_images;
   } else if (stage == MESA_SHADER_FRAGMENT) {
      invalid = &context->fragment_images_invalid;
      bound_images = context->fragment_images;
   } else if (stage <= MESA_SHADER_GEOMETRY) {
      invalid = &context->preraster_images_invalid[stage];
      bound_images = context->preraster_images[stage];
   } else {
      return;
   }
   *invalid = true;
   if (start > PS5_COMPUTE_IMAGE_SLOTS || count > PS5_COMPUTE_IMAGE_SLOTS - start ||
       unbind > PS5_COMPUTE_IMAGE_SLOTS - start - count)
      return;
   for (unsigned i = 0; images && i < count; ++i) {
      const struct pipe_image_view *v = &images[i];
      uint32_t descriptor[8];
      if (!v->resource)
         continue;
      if (v->resource->screen != base->screen ||
          !(v->access & PIPE_IMAGE_ACCESS_READ_WRITE) ||
          ((v->access | v->shader_access) & ~(PIPE_IMAGE_ACCESS_READ_WRITE |
              PIPE_IMAGE_ACCESS_COHERENT | PIPE_IMAGE_ACCESS_VOLATILE)))
         return;
      if (ps5_image_view_incomplete(v))
         continue;
      if (v->resource->target == PIPE_BUFFER) {
         if (!ps5_image_buffer_descriptor(v,
               (uintptr_t)((struct ps5_resource *)v->resource)->data >> 32,
               descriptor))
            return;
      } else if (ps5_storage_image_view_descriptor(v, descriptor))
         return;
   }
   /* Storage-consuming draws/dispatches finish synchronously. Deferred
    * draws exclude storage shaders and retain their other resource bindings;
    * changing a storage binding cannot invalidate a queued descriptor. */
   for (unsigned i = 0; i < count + unbind; ++i) {
      struct pipe_image_view next = images && i < count ? images[i] : (struct pipe_image_view){0};
      struct pipe_image_view *bound = &bound_images[start + i];
      pipe_resource_reference(&bound->resource, next.resource);
      next.resource = bound->resource;
      *bound = next;
   }
   *invalid = false;
}

static void
ps5_set_compute_sampler_views(struct pipe_context *base, unsigned start, unsigned count,
                              unsigned unbind_trailing, struct pipe_sampler_view **views)
{
   struct ps5_context *context = (struct ps5_context *)base;
   context->compute_views_invalid = true;
   if (start > PS5_COMPUTE_TEXTURE_SLOTS || count > PS5_COMPUTE_TEXTURE_SLOTS - start ||
       unbind_trailing > PS5_COMPUTE_TEXTURE_SLOTS - start - count || (count && !views))
      return;
   for (unsigned i = 0; i < count; ++i) {
      const struct pipe_sampler_view *v = views[i];
      uint32_t descriptor[8];
      if (v && v->texture && v->texture->target == PIPE_BUFFER) {
         if (!ps5_texel_buffer_descriptor(v,
               (uintptr_t)((struct ps5_resource *)v->texture)->data >> 32,
               descriptor))
            return;
         continue;
      }
      uint32_t selector;
      const bool swizzle_ok = !v ||
         (ps5_texture_descriptor_swizzle(v->swizzle_r, v->format, &selector) &&
          ps5_texture_descriptor_swizzle(v->swizzle_g, v->format, &selector) &&
          ps5_texture_descriptor_swizzle(v->swizzle_b, v->format, &selector) &&
          ps5_texture_descriptor_swizzle(v->swizzle_a, v->format, &selector));
      const bool depth_special = v && v->texture &&
         (v->texture->format == PIPE_FORMAT_Z32_FLOAT ||
          (v->texture->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
           (v->format == v->texture->format ||
            v->format == PIPE_FORMAT_X32_S8X24_UINT)));
      const int descriptor_result = !v ? 0 : depth_special ?
         ps5_packed_depth_sampled_descriptor(v->texture, v->format,
            v->u.tex.first_level, v->u.tex.last_level, descriptor) :
         ps5_resource_sampled_image_descriptor(v->texture, v->u.tex.first_level,
                                               v->u.tex.last_level, descriptor);
      if (v && (!v->texture || v->texture->screen != base->screen ||
          v->target != v->texture->target ||
          (!depth_special && v->format != v->texture->format) ||
          v->u.tex.first_layer ||
          v->u.tex.last_layer != v->texture->array_size - 1 || !swizzle_ok ||
          descriptor_result)) {
         printf("[ps5-gallium] compute view rejected slot=%u target=%u/%u format=%u/%u levels=%u:%u/%u layers=%u:%u/%u depth=%u samples=%u:%u bind=%x swizzle=%u descriptor=%d\n",
                start + i, v->target, v->texture ? v->texture->target : 0,
                v->format, v->texture ? v->texture->format : 0,
                v->u.tex.first_level, v->u.tex.last_level, v->texture ? v->texture->last_level : 0,
                v->u.tex.first_layer, v->u.tex.last_layer, v->texture ? v->texture->array_size : 0,
                v->texture ? v->texture->depth0 : 0,
                v->texture ? v->texture->nr_samples : 0, v->texture ? v->texture->nr_storage_samples : 0,
                v->texture ? v->texture->bind : 0, swizzle_ok,
                descriptor_result);
         return;
      }
   }
   for (unsigned i = 0; i < count + unbind_trailing; ++i)
      pipe_sampler_view_reference(&context->compute_views[start + i], i < count ? views[i] : NULL);
   context->compute_views_invalid = false;
}

static void
ps5_set_compute_sampler_states(struct pipe_context *base, unsigned start, unsigned count,
                               void **states)
{
   struct ps5_context *context = (struct ps5_context *)base;
   uint32_t descriptors[PS5_COMPUTE_TEXTURE_SLOTS][4] = {{0}};
   context->compute_samplers_invalid = true;
   if (start > PS5_COMPUTE_TEXTURE_SLOTS || count > PS5_COMPUTE_TEXTURE_SLOTS - start ||
       (count && !states))
      return;
   for (unsigned i = 0; i < count; ++i) {
      if (!states[i]) continue;
      const struct pipe_sampler_state *s = &((const struct ps5_sampler_state *)states[i])->base;
      uint32_t wrap[3], min_filter, mag_filter, mip_filter;
      uint32_t anisotropy = ps5_texture_descriptor_anisotropy(s->max_anisotropy);
      /* Rectangle coordinates are normalized in NIR, including projectors
       * and offsets; keep the hardware sampler normalized for every target. */
      if (s->compare_mode > PIPE_TEX_COMPARE_R_TO_TEXTURE ||
          (s->compare_mode && s->compare_func > PIPE_FUNC_ALWAYS) ||
          s->max_anisotropy > 16 || !ps5_float_is_finite(s->min_lod) ||
          !ps5_float_is_finite(s->max_lod) ||
          !(s->min_lod >= 0 && s->max_lod >= s->min_lod) ||
          s->lod_bias != 0 || !ps5_texture_descriptor_mip_filter(s->min_mip_filter, &mip_filter) ||
          !ps5_texture_descriptor_wrap(s->wrap_s, &wrap[0]) ||
          !ps5_texture_descriptor_wrap(s->wrap_t, &wrap[1]) ||
          !ps5_texture_descriptor_wrap(s->wrap_r, &wrap[2]) ||
          !ps5_texture_descriptor_filter(s->min_img_filter, s->max_anisotropy,
                                         &min_filter) ||
          !ps5_texture_descriptor_filter(s->mag_img_filter, s->max_anisotropy,
                                         &mag_filter))
         return;
      descriptors[i][0] = wrap[0] | (wrap[1] << 3) | (wrap[2] << 6) |
                          (anisotropy << 9) |
                          ((anisotropy >> 1) << 16) |
                          (anisotropy << 21) |
                          (s->compare_mode ? s->compare_func << 12 : 0);
      /* Mesa's default max LOD is 1000. Validate before the existing encoder
       * saturates finite bounds to the supported levels 0..15. */
      descriptors[i][1] = ps5_texture_descriptor_unsigned_lod(s->min_lod) |
                         (ps5_texture_descriptor_unsigned_lod(s->max_lod) << 12) |
                         ((anisotropy ? anisotropy + 6u : 0u) << 24);
      descriptors[i][2] = ps5_texture_descriptor_lod_bias(s->lod_bias) |
                         (mag_filter << 20) | (min_filter << 22) |
                         (mip_filter << 26) |
                         (anisotropy ? 1u << 29 : 0u);
   }
   for (unsigned i = 0; i < count; ++i) {
      memcpy(context->compute_samplers[start + i], descriptors[i], sizeof(descriptors[i]));
      context->compute_sampler_mask &= ~(1u << (start + i));
      if (states[i]) context->compute_sampler_mask |= 1u << (start + i);
   }
   context->compute_samplers_invalid = false;
}

#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DEFERRED_DRAW_BATCH)
/* Queue the dispatch behind earlier GPU work in the deferred batch. Its slot
 * pins a private descriptor-table copy and every bound resource until the
 * batch retires; CPU access to any of them drains, then invalidates writes.
 * Returns false with nothing queued when the synchronous path must run it. */
static bool
ps5_queue_compute(struct ps5_context *context, struct pipe_resource *const *buffers,
                  unsigned buffer_count, struct pipe_resource *const *written,
                  unsigned written_count, const uint32_t groups[3])
{
   struct pipe_context *base = &context->base;
   const struct ps5_resource *source = (const struct ps5_resource *)context->compute_descriptors;
   /* Mapped storage has no CPU callback to wait for queued writes. */
   for (unsigned i = 0; i < written_count; ++i)
      if (((const struct ps5_resource *)written[i])->external_cpu_access)
         return false;
   if (!ps5_agc_gate2_batch_begin || !ps5_agc_gate2_batch_end ||
       buffer_count > PS5_BATCH_RESOURCE_COUNT ||
       PS5_COMPUTE_DESCRIPTOR_BYTES > PS5_DIRECT_ALIGNMENT)
      return false;
   struct pipe_resource templ = source->base;
   templ.width0 = PS5_DIRECT_ALIGNMENT; /* the recycled snapshot class */
   bool queued = false;
   simple_mtx_lock(&ps5_deferred_mutex);
   if (ps5_deferred.owner && ps5_deferred.owner != context)
      ps5_draw_batch_flush_locked();
   struct pipe_resource *table = ps5_descriptor_take(base, 2, &templ);
   struct ps5_resource *copy = (struct ps5_resource *)table;
   if (!copy || !copy->data || copy->size < PS5_COMPUTE_DESCRIPTOR_BYTES ||
       (uintptr_t)copy->data >> 32 != (uintptr_t)source->data >> 32)
      goto out;
   memcpy(copy->data, source->data, PS5_COMPUTE_DESCRIPTOR_BYTES);
   ps5_flush_gpu_data(copy->data, PS5_COMPUTE_DESCRIPTOR_BYTES);
   if (!ps5_deferred.owner) {
      if ((ps5_agc_gate2_batch_begin_framebuffer ? ps5_agc_gate2_batch_begin_framebuffer() :
           ps5_agc_gate2_batch_begin()) != 0)
         goto out;
      ps5_deferred.owner = context;
   }
   const unsigned slot = ps5_deferred.count++;
   struct ps5_deferred_slot *pinned = &ps5_deferred.slots[slot];
   int status = ps5_agc_compute_queue(base->screen, &context->cs->output, table,
                                      buffers, buffer_count, groups);
   if (status) {
      --ps5_deferred.count;
      if (!ps5_deferred.count) {
         ps5_draw_batch_flush_locked(); /* closes the empty batch it opened */
      }
      goto out;
   }
   pinned->storage[2] = table;
   pinned->compute = true;
   table = NULL;
   for (unsigned i = 0; i < buffer_count; ++i)
      pipe_resource_reference(&pinned->retained[pinned->retained_count++], buffers[i]);
   for (unsigned i = 0; i < written_count; ++i)
      ((struct ps5_resource *)written[i])->gpu_written = true;
   queued = true;
   if (ps5_deferred.count == PS5_MULTIDRAW_BATCH_CAPACITY)
      ps5_draw_batch_flush_locked();
out:
   pipe_resource_reference(&table, NULL);
   simple_mtx_unlock(&ps5_deferred_mutex);
   return queued;
}
#endif

/* Publish only bytes the CPU may have written since the resource was last
 * flushed. Owned buffers publish each CPU write at the write (unmap,
 * flush_region, subdata) or drain first, so the epoch covers them as it covers
 * textures. Persistent and exported mappings have no write callback. */
static void
ps5_publish_compute_resource(struct pipe_resource *base, size_t offset, size_t bytes)
{
   struct ps5_resource *resource = (struct ps5_resource *)base;
   if (base->target != PIPE_BUFFER) {
      ps5_flush_texture_backing(NULL, 0, resource, resource->allocation_size, false);
      return;
   }
   if (resource->external_cpu_access) {
      ps5_flush_gpu_data(resource->data + offset, bytes);
      return;
   }
#ifdef PS5_DEFERRED_DRAW_BATCH
   if (resource->texture_publication_epoch == ps5_texture_publication_epoch &&
       resource->texture_published_bytes >= resource->allocation_size)
      return;
#endif
   ps5_flush_gpu_data(resource->data, resource->allocation_size);
   resource->texture_publication_epoch = ps5_texture_publication_epoch;
   resource->texture_published_bytes = resource->allocation_size;
}



#ifndef PS5_KLOG
#define PS5_KLOG(...) do { \
      int sceKernelDebugOutText(int channel, const char *text) __attribute__((weak)); \
      static unsigned ps5_klog_reported; \
      char ps5_klog_message[256]; \
      if (sceKernelDebugOutText && ps5_klog_reported++ < 32) { \
         snprintf(ps5_klog_message, sizeof(ps5_klog_message), __VA_ARGS__); \
         sceKernelDebugOutText(0, ps5_klog_message); \
      } \
   } while (0)
#define PS5_COMPUTE_REJECT() do { \
      PS5_KLOG("[ps5-gallium] compute dispatch dropped at ps5_screen.c:%u\n", __LINE__); \
      return; \
   } while (0)
#endif

static void
ps5_launch_grid(struct pipe_context *base, const struct pipe_grid_info *grid)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_resource *buffers[PS5_AGC_COMPUTE_MAX_RESOURCES];
   /* Storage buffers and images the dispatch may write. */
   struct pipe_resource *written[PS5_COMPUTE_STORAGE_SLOTS + PS5_COMPUTE_IMAGE_SLOTS];
   size_t offsets[PS5_AGC_COMPUTE_MAX_RESOURCES], sizes[PS5_AGC_COMPUTE_MAX_RESOURCES];
   uint32_t groups[3];
   unsigned buffer_count = 0, written_count = 0;
   context->last_compute_status = -1;
   if (!context->cs || !context->compute_descriptors ||
       (context->cs->ssbos && context->compute_bindings_invalid) ||
       (context->compute_constants_invalid & BITFIELD_MASK(context->cs->ubos)) ||
       (context->cs->images && context->compute_images_invalid) ||
       (context->cs->textures && context->compute_views_invalid) ||
       (context->cs->filtered_textures && context->compute_samplers_invalid) || !grid ||
       grid->work_dim > 3 || grid->variable_shared_mem || grid->num_globals ||
       grid->draw_count || grid->indirect_draw_count) {
      PS5_KLOG("[ps5-gallium] compute rejected cs=%u descriptors=%u buffers=%u constants=%x images=%u views=%u samplers=%u grid=%u\n",
             context->cs != NULL, context->compute_descriptors != NULL,
             context->compute_bindings_invalid, context->compute_constants_invalid,
             context->compute_images_invalid, context->compute_views_invalid,
             context->compute_samplers_invalid, grid != NULL);
      return;
   }
   for (unsigned i = 0; i < 3; ++i) {
      if (grid->block[i] != context->cs->output.metadata.compute_workgroup_size[i] ||
          grid->grid_base[i] || (grid->last_block[i] && grid->last_block[i] != grid->block[i]))
         PS5_COMPUTE_REJECT();
   }
   if (!ps5_render_condition_passes(context)) {
      context->last_compute_status = 0;
      return;
   }
   memcpy(groups, grid->grid, sizeof(groups));
   if (grid->indirect) {
      const struct pipe_resource *command = grid->indirect;
      void *address = NULL;
      size_t size = 0;
      if (command->screen != base->screen || command->target != PIPE_BUFFER ||
          (grid->indirect_offset & 3u) || grid->indirect_offset > command->width0 ||
          sizeof(groups) > command->width0 - grid->indirect_offset)
         PS5_COMPUTE_REJECT();
      /* ponytail: CPU argument readback waits for queued writers and
       * invalidates their results. Native indirect packets need a GPU grid-size ABI. */
      ps5_draw_batch_drain_buffer(grid->indirect);
      if (ps5_resource_gpu_info(grid->indirect, &address, &size, NULL) || !address ||
          grid->indirect_offset > size || sizeof(groups) > size - grid->indirect_offset)
         PS5_COMPUTE_REJECT();
      const uint8_t *arguments = (const uint8_t *)address + grid->indirect_offset;
      ps5_flush_gpu_data(arguments, sizeof(groups));
      memcpy(groups, arguments, sizeof(groups));
   }
   for (unsigned i = 0; i < 3; ++i)
      if (groups[i] > 65535)
         PS5_COMPUTE_REJECT();
   if (!groups[0] || !groups[1] || !groups[2]) {
      context->last_compute_status = 0; /* A zero-sized dispatch has no work. */
      return;
   }
   struct ps5_resource *table = (struct ps5_resource *)context->compute_descriptors;
   memset(table->data, 0, PS5_COMPUTE_DESCRIPTOR_BYTES);
   for (unsigned i = 0; i < PS5_COMPUTE_BUFFER_SLOTS; ++i) {
      const struct pipe_shader_buffer *bound = &context->compute_buffers[i];
      if (!bound->buffer)
         continue;
      const struct ps5_resource *resource = (struct ps5_resource *)bound->buffer;
      const uintptr_t address = (uintptr_t)resource->data + bound->buffer_offset;
      uint32_t *srd = (uint32_t *)(table->data + i * 16);
      srd[0] = address;
      srd[1] = address >> 32;
      srd[2] = bound->buffer_size;
      srd[3] = UINT32_C(0x31016fac);
      if (i < PS5_COMPUTE_STORAGE_SLOTS)
         written[written_count++] = bound->buffer;
      offsets[buffer_count] = bound->buffer_offset;
      sizes[buffer_count] = bound->buffer_size;
      buffers[buffer_count++] = bound->buffer;
   }
   for (unsigned i = 0; i < PS5_COMPUTE_IMAGE_SLOTS; ++i) {
      const struct pipe_image_view *view = &context->compute_images[i];
      struct pipe_resource *resource = view->resource;
      if (!resource)
         continue;
      uint32_t *descriptor = (uint32_t *)(table->data +
         PS5_COMPUTE_BUFFER_SLOTS * 16 + i * 32);
      if (ps5_image_view_incomplete(view))
         continue;
      if (resource->target == PIPE_BUFFER ?
          !ps5_image_buffer_descriptor(view, context->cs->output.metadata.address32_hi,
                                       descriptor) :
          ps5_storage_image_view_descriptor(view, descriptor))
         PS5_COMPUTE_REJECT();
      written[written_count++] = resource;
      offsets[buffer_count] = resource->target == PIPE_BUFFER ? view->u.buf.offset : 0;
      sizes[buffer_count] = resource->target == PIPE_BUFFER ? view->u.buf.size : 0;
      buffers[buffer_count++] = resource;
   }
   for (unsigned i = 0; i < PS5_COMPUTE_TEXTURE_SLOTS; ++i) {
      if (!(context->cs->textures & (1u << i)))
         continue;
      const struct pipe_sampler_view *view = context->compute_views[i];
      uint32_t *descriptor = (uint32_t *)(table->data +
                                         PS5_COMPUTE_TEXTURE_OFFSET + i * 48);
      /* A dynamically indexed buffer array may contain unused, unbound
       * elements. Preserve their zero descriptor instead of dropping the grid. */
      if (!view && (context->cs->buffer_textures & (1u << i)))
         continue;
      if (!view)
         PS5_COMPUTE_REJECT();
      if (context->cs->buffer_textures & (1u << i)) {
         if (!ps5_texel_buffer_descriptor(view, context->cs->output.metadata.address32_hi,
                                          descriptor))
            PS5_COMPUTE_REJECT();
         offsets[buffer_count] = view->u.buf.offset;
         sizes[buffer_count] = view->u.buf.size;
         buffers[buffer_count++] = view->texture;
         continue;
      }
      const bool depth_special = view->texture->format == PIPE_FORMAT_Z32_FLOAT ||
         (view->texture->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
          (view->format == view->texture->format ||
           view->format == PIPE_FORMAT_X32_S8X24_UINT));
      if (depth_special ?
            ps5_packed_depth_sampled_descriptor(view->texture, view->format,
               view->u.tex.first_level, view->u.tex.last_level, descriptor) :
            ps5_resource_sampled_image_descriptor(view->texture,
               view->u.tex.first_level, view->u.tex.last_level, descriptor))
         PS5_COMPUTE_REJECT();
      const unsigned swizzles[] = {view->swizzle_r, view->swizzle_g,
                                   view->swizzle_b, view->swizzle_a};
      for (unsigned channel = 0; channel < 4; ++channel) {
         uint32_t selector;
         if (!ps5_texture_descriptor_swizzle(swizzles[channel], view->format, &selector))
            PS5_COMPUTE_REJECT();
         descriptor[3] = (descriptor[3] & ~(7u << (channel * 3))) |
                         (selector << (channel * 3));
      }
      if (context->cs->texture_lod[i] == UINT_MAX) {
         unsigned full_last_level = 0;
         unsigned extent = MAX3(view->texture->width0, view->texture->height0,
                                view->texture->depth0);
         while (extent >>= 1)
            ++full_last_level;
         if (view->u.tex.first_level ||
             view->u.tex.last_level < full_last_level)
            PS5_COMPUTE_REJECT();
      } else if (context->cs->texture_lod[i] >
                 view->u.tex.last_level - view->u.tex.first_level) {
         PS5_COMPUTE_REJECT();
      }
      if ((view->target == PIPE_TEXTURE_1D_ARRAY ||
           view->target == PIPE_TEXTURE_2D_ARRAY ||
           view->target == PIPE_TEXTURE_CUBE_ARRAY) !=
          ((context->cs->array_textures & (1u << i)) != 0))
         PS5_COMPUTE_REJECT();
      if (context->cs->filtered_textures & (1u << i)) {
         if (!(context->compute_sampler_mask & (1u << i)) ||
             !ps5_texture_format_size(view->format) ||
             (util_format_is_pure_integer(view->format) &&
              (context->compute_samplers[i][2] &
               ((UINT32_C(3) << 20) | (UINT32_C(3) << 22)))))
            PS5_COMPUTE_REJECT();
         memcpy(table->data + PS5_COMPUTE_TEXTURE_OFFSET + i * 48 + 32,
                context->compute_samplers[i], 16);
      }
      offsets[buffer_count] = sizes[buffer_count] = 0;
      buffers[buffer_count++] = view->texture;
   }
#ifdef PS5_DEFERRED_DRAW_BATCH
   simple_mtx_lock(&ps5_deferred_mutex); /* publication epochs */
#endif
   for (unsigned i = 0; i < buffer_count; ++i)
      ps5_publish_compute_resource(buffers[i], offsets[i], sizes[i]);
   ps5_flush_gpu_data(table->data, PS5_COMPUTE_DESCRIPTOR_BYTES);
#ifdef PS5_DEFERRED_DRAW_BATCH
   simple_mtx_unlock(&ps5_deferred_mutex);
#endif
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DEFERRED_DRAW_BATCH)
   /* Diagnostics: PS5_COMPUTE_SYNC=1 runs every dispatch synchronously. */
   static int compute_sync = -1;
   if (compute_sync < 0)
      compute_sync = getenv("PS5_COMPUTE_SYNC") != NULL;
   if (!compute_sync &&
       ps5_queue_compute(context, buffers, buffer_count, written, written_count, groups)) {
      context->last_compute_status = 0;
      ++context->dispatches;
      return;
   }
#endif
   context->last_compute_status = ps5_agc_compute_dispatch(base->screen, &context->cs->output,
      context->compute_descriptors, buffers, buffer_count, groups);
   /* The dispatch has completed. Mapped storage without a CPU callback sees
    * results now; everything else invalidates at its next CPU access. */
#ifdef PS5_DEFERRED_DRAW_BATCH
   simple_mtx_lock(&ps5_deferred_mutex);
#endif
   for (unsigned i = 0; i < written_count; ++i) {
      struct ps5_resource *resource = (struct ps5_resource *)written[i];
      if (resource->external_cpu_access)
         ps5_flush_gpu_data(resource->data, resource->allocation_size);
      else
         resource->gpu_written = true;
   }
#ifdef PS5_DEFERRED_DRAW_BATCH
   simple_mtx_unlock(&ps5_deferred_mutex);
#endif
   if (context->last_compute_status)
      printf("[ps5-gallium] compute submit failed status=%d resources=%u\n",
             context->last_compute_status, buffer_count);
   if (!context->last_compute_status)
      ++context->dispatches;
}
#endif

static void
ps5_lower_default_uniforms(nir_shader *nir)
{
   if (nir->num_uniforms)
      nir_lower_uniforms_to_ubo(nir, false, false);
   /* Sampler lowering also leaves dead dereferences when there is no UBO. */
   nir_opt_dce(nir);
   nir_remove_dead_variables(nir, nir_var_uniform, NULL);
}

/* Framebuffer fetch (KHR_blend_equation_advanced, non-coherent): a fragment
 * output read becomes a texel fetch of the pixel from color buffer 0, which the
 * draw binds to a free texture unit behind a color-to-texture barrier. */
static bool
ps5_lower_fbfetch_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned unit = *(const unsigned *)data;
   if (intr->intrinsic != nir_intrinsic_load_output)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   /* Integer pixel position; the backend has no whole-vector load_frag_coord. */
   nir_def *coord = nir_u2u32(b, nir_load_pixel_coord(b));
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->texture_index = unit;
   tex->sampler_index = unit;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   nir_def *value = nir_trim_vector(b, &tex->def, intr->def.num_components);
   if (intr->def.bit_size != 32)
      value = nir_f2fN(b, value, intr->def.bit_size);
   nir_def_replace(&intr->def, value);
   return true;
}

static bool
ps5_lower_fbfetch(struct ps5_shader *shader)
{
   nir_shader *nir = shader->nir;
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return true;
   /* Shader info may predate the output reads; look at the instructions. */
   uint64_t fetched = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_output)
               fetched |= BITFIELD64_BIT(nir_intrinsic_io_semantics(intr).location);
         }
      }
   }
   if (!fetched)
      return true;
   /* Only color attachment 0 can be fetched (pipe caps.fbfetch = 1). */
   if (fetched & ~(BITFIELD64_BIT(FRAG_RESULT_DATA0) | BITFIELD64_BIT(FRAG_RESULT_COLOR)))
      return false;
   unsigned unit = 0;
   while (unit < PS5_MAX_TEXTURE_UNITS && BITSET_TEST(nir->info.textures_used, unit))
      ++unit;
   if (unit >= PS5_MAX_TEXTURE_UNITS)
      return false;
   nir_shader_intrinsics_pass(nir, ps5_lower_fbfetch_instr, nir_metadata_control_flow, &unit);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   BITSET_SET(nir->info.textures_used, unit);
   BITSET_SET(nir->info.textures_used_by_txf, unit);
   nir->info.num_textures = MAX2(nir->info.num_textures, unit + 1);
   shader->fbfetch_unit = unit + 1;
   return true;
}

static void *
ps5_create_shader_state(struct pipe_screen *screen,
                        const struct pipe_shader_state *templ, PsbcStage stage,
                        uint32_t address32_hi)
{
   struct ps5_shader *shader;
   struct ps5_vertex_layout layout;
   struct pipe_shader_state converted;

   /* u_blitter's simple VS/FS generators still emit TGSI. Reuse Mesa's
    * translator; the public shader path and PSBC backend remain NIR-only. */
   if (templ && templ->type == PIPE_SHADER_IR_TGSI && templ->tokens &&
       stage != PSBC_STAGE_GEOMETRY && !templ->stream_output.num_outputs) {
      converted = *templ;
      converted.type = PIPE_SHADER_IR_NIR;
      converted.ir.nir = tgsi_to_nir(templ->tokens, screen, false);
      if (converted.ir.nir)
         nir_lower_io_passes(converted.ir.nir, false);
      templ = &converted;
   }

   if (!templ || templ->type != PIPE_SHADER_IR_NIR || !templ->ir.nir ||
       !ps5_stream_output_info_valid(&templ->stream_output, stage) ||
       (templ->stream_output.num_outputs && !templ->ir.nir->xfb_info))
      return NULL;

   /* Mesa's default uniforms arrive as load_uniform intrinsics and CB0 data.
    * PSBC consumes descriptor-backed load_ubo, with the same vec4 layout. */
   ps5_lower_default_uniforms(templ->ir.nir);

   printf("[ps5-gallium] create-shader stage=%u nir-stage=%d io-lowered=%u ubos=%u ssbos=%u images=%u textures=%u default-ubo=%u uniforms=%u face=%u/%u\n",
          stage, templ->ir.nir->info.stage, templ->ir.nir->info.io_lowered,
          templ->ir.nir->info.num_ubos,
          templ->ir.nir->info.num_ssbos,
          templ->ir.nir->info.num_images,
          templ->ir.nir->info.num_textures,
          templ->ir.nir->info.first_ubo_is_default_ubo,
          templ->ir.nir->num_uniforms,
          !!(templ->ir.nir->info.inputs_read & VARYING_BIT_FACE),
          BITSET_TEST(templ->ir.nir->info.system_values_read,
                      SYSTEM_VALUE_FRONT_FACE));
   if (stage == PSBC_STAGE_VERTEX && templ->ir.nir->info.io_lowered &&
       !PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE)
      nir_shader_instructions_pass(templ->ir.nir, ps5_remove_point_size,
                                   nir_metadata_control_flow, NULL);
   if (stage == PSBC_STAGE_VERTEX && templ->ir.nir->info.io_lowered) {
      uint64_t inputs = templ->ir.nir->info.inputs_read;
      if (inputs & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) {
         if (util_bitcount64(inputs) > 16) {
            ralloc_free(templ->ir.nir);
            return NULL;
         }
         nir_shader_instructions_pass(templ->ir.nir,
                                      ps5_rebase_meta_vertex_input,
                                      nir_metadata_control_flow,
                                      &inputs);
      }
      nir_opt_algebraic(templ->ir.nir);
      nir_opt_constant_folding(templ->ir.nir);
      nir_opt_dce(templ->ir.nir);
      nir_remove_dead_variables(templ->ir.nir, nir_var_shader_out, NULL);
      nir_shader_gather_info(templ->ir.nir,
                             nir_shader_get_entrypoint(templ->ir.nir));
      templ->ir.nir->num_outputs = 1;
   }
   shader = calloc(1, sizeof(*shader));
   if (!shader) {
      ralloc_free(templ->ir.nir);
      return NULL;
   }
   shader->stage = stage;
   shader->nir = templ->ir.nir;
   shader->serial = ps5_next_shader_serial();
   shader->stream_output = templ->stream_output;
   if (!ps5_lower_fbfetch(shader)) {
      ralloc_free(shader->nir);
      free(shader);
      return NULL;
   }
   if (stage == PSBC_STAGE_VERTEX) {
      if (!ps5_default_vertex_layout(shader->nir, &layout)) {
         ralloc_free(shader->nir);
         free(shader);
         return NULL;
      }
   } else {
      memset(&layout, 0, sizeof(layout));
   }
   if (stage != PSBC_STAGE_GEOMETRY && stage != PSBC_STAGE_TESS_CTRL &&
       stage != PSBC_STAGE_TESS_EVAL &&
       !(stage == PSBC_STAGE_VERTEX && ps5_shader_uses_storage(shader)) &&
       !ps5_select_shader_variant(shader, address32_hi, &layout, 0, false,
                                  false, false, false, false, -1, NULL)) {
      ralloc_free(shader->nir);
      free(shader);
      return NULL;
   }
   return shader;
}

static void *
ps5_create_vs_state(struct pipe_context *context,
                    const struct pipe_shader_state *templ)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *descriptors = (struct ps5_resource *)
      ps5->vertex_descriptor_table;

   if (!descriptors)
      return NULL;
   return ps5_create_shader_state(
      context->screen, templ, PSBC_STAGE_VERTEX,
      (uint32_t)((uintptr_t)descriptors->data >> 32));
}

static void *
ps5_create_fs_state(struct pipe_context *context,
                    const struct pipe_shader_state *templ)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *storage =
      (struct ps5_resource *)ps5->descriptor_storage[1];

   if (!storage)
      return NULL;
   return ps5_create_shader_state(
      context->screen, templ, PSBC_STAGE_FRAGMENT,
      (uint32_t)((uintptr_t)storage->data >> 32));
}

static void *
ps5_create_gs_state(struct pipe_context *context,
                    const struct pipe_shader_state *templ)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *descriptors = (struct ps5_resource *)
      ps5->vertex_descriptor_table;
   void *state;

   if (!PS5_ENABLE_GEOMETRY_CANDIDATE || !descriptors) {
      printf("[ps5-gallium] create-geometry state=failed\n");
      return NULL;
   }
   state = ps5_create_shader_state(
      context->screen, templ, PSBC_STAGE_GEOMETRY,
      (uint32_t)((uintptr_t)descriptors->data >> 32));
   printf("[ps5-gallium] create-geometry state=%s\n",
          state ? "ready" : "failed");
   return state;
}

static void *
ps5_create_tcs_state(struct pipe_context *context,
                     const struct pipe_shader_state *templ)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *descriptors =
      (struct ps5_resource *)ps5->vertex_descriptor_table;

   return PS5_ENABLE_TESSELLATION_CANDIDATE && descriptors
      ? ps5_create_shader_state(context->screen, templ, PSBC_STAGE_TESS_CTRL,
            (uint32_t)((uintptr_t)descriptors->data >> 32))
      : NULL;
}

static void *
ps5_create_tes_state(struct pipe_context *context,
                     const struct pipe_shader_state *templ)
{
   struct ps5_context *ps5 = (struct ps5_context *)context;
   struct ps5_resource *descriptors =
      (struct ps5_resource *)ps5->vertex_descriptor_table;

   return PS5_ENABLE_TESSELLATION_CANDIDATE && descriptors
      ? ps5_create_shader_state(context->screen, templ, PSBC_STAGE_TESS_EVAL,
            (uint32_t)((uintptr_t)descriptors->data >> 32))
      : NULL;
}

static void
ps5_bind_vs_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->vs = state;
}

static void
ps5_bind_fs_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->fs = state;
}

static void
ps5_bind_gs_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->gs = state;
#if !defined(PS5_NATIVE_TITLE_RUNTIME) || defined(AGC_RUNTIME_DIAGNOSTICS)
   printf("[ps5-gallium] bind-geometry state=%s\n",
          state ? "ready" : "null");
#endif
}

static void
ps5_bind_tcs_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->tcs = state;
}

static void
ps5_bind_tes_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->tes = state;
}

static void
ps5_set_patch_vertices(struct pipe_context *base, uint8_t patch_vertices)
{
   ((struct ps5_context *)base)->patch_vertices = patch_vertices;
}

static void
ps5_set_tess_state(struct pipe_context *base, const float outer[4], const float inner[2])
{
   struct ps5_context *context = (void *)base;
   memcpy(context->tess_levels, outer, 4 * sizeof(float));
   memcpy(context->tess_levels + 4, inner, 2 * sizeof(float));
}

static void
ps5_bind_blend_state(struct pipe_context *base, void *state)
{
   struct ps5_context *context = (struct ps5_context *)base;

   context->blend = state;
   if (context->blend && context->blend->logicop_enable)
      context->logicop_used = true;
}

static void
ps5_bind_rasterizer_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->rasterizer = state;
}

static void
ps5_bind_depth_stencil_alpha_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->depth_stencil_alpha = state;
}

static void *
ps5_copy_state(const void *state, size_t size)
{
   void *copy = malloc(size);

   if (copy)
      memcpy(copy, state, size);
   return copy;
}

static void *
ps5_create_blend_state(struct pipe_context *base,
                       const struct pipe_blend_state *state)
{
   (void)base;
   return ps5_copy_state(state, sizeof(*state));
}

static void *
ps5_create_rasterizer_state(struct pipe_context *base,
                            const struct pipe_rasterizer_state *state)
{
   (void)base;
   return ps5_copy_state(state, sizeof(*state));
}

static void *
ps5_create_depth_stencil_alpha_state(
   struct pipe_context *base,
   const struct pipe_depth_stencil_alpha_state *state)
{
   (void)base;
   return ps5_copy_state(state, sizeof(*state));
}

static void *
ps5_create_vertex_elements_state(struct pipe_context *base,
                                 unsigned count,
                                 const struct pipe_vertex_element *elements)
{
   struct ps5_vertex_elements *state;

   (void)base;
   if (count > PIPE_MAX_ATTRIBS || (count && !elements))
      return NULL;
   state = calloc(1, sizeof(*state));
   if (!state)
      return NULL;
   for (unsigned i = 0; i < count; ++i) {
      struct pipe_vertex_element element = elements[i];
      /* CSO/u_vbuf split a dual-slot uint64 input into two uint32 elements.
       * PSBC consumes Gallium's compact semantic plus high_dvec2, so restore
       * one raw 64-bit format per input before building layouts/descriptors. */
      if (PS5_ENABLE_FP64_CANDIDATE && element.dual_slot &&
          element.src_format == PIPE_FORMAT_R32G32B32A32_UINT && i + 1 < count) {
         const struct pipe_vertex_element *high = &elements[i + 1];
         if (high->dual_slot &&
             (high->src_format == PIPE_FORMAT_R32G32_UINT ||
              high->src_format == PIPE_FORMAT_R32G32B32A32_UINT) &&
             (uint64_t)element.src_offset + 16 == high->src_offset &&
             element.src_stride == high->src_stride &&
             element.vertex_buffer_index == high->vertex_buffer_index &&
             element.instance_divisor == high->instance_divisor) {
            element.src_format = high->src_format == PIPE_FORMAT_R32G32_UINT
               ? PIPE_FORMAT_R64G64B64_FLOAT : PIPE_FORMAT_R64G64B64A64_FLOAT;
            element.dual_slot = false;
            ++i;
         }
      }
      state->elements[state->count++] = element;
   }
   return state;
}

static void
ps5_bind_vertex_elements_state(struct pipe_context *base, void *state)
{
   ((struct ps5_context *)base)->vertex_elements = state;
}

static void
ps5_delete_vertex_elements_state(struct pipe_context *base, void *state)
{
   struct ps5_context *context = (struct ps5_context *)base;

   if (context->vertex_elements == state)
      context->vertex_elements = NULL;
   free(state);
}

static void
ps5_delete_fixed_state(struct pipe_context *base, void *state)
{
   (void)base;
   free(state);
}

static void
ps5_set_blend_color(struct pipe_context *base,
                    const struct pipe_blend_color *color)
{
   if (color)
      ((struct ps5_context *)base)->blend_color = *color;
}

static void
ps5_set_scissor_states(struct pipe_context *base, unsigned start,
                       unsigned count,
                       const struct pipe_scissor_state *states)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned i;

   if (!states || start >= PS5_MAX_VIEWPORTS ||
       count > PS5_MAX_VIEWPORTS - start)
      return;
   for (i = 0; i < count; ++i) {
      context->scissor[start + i] = states[i];
      context->scissor_valid |= UINT16_C(1) << (start + i);
   }
}

static void
ps5_set_viewport_states(struct pipe_context *base, unsigned start,
                        unsigned count,
                        const struct pipe_viewport_state *states)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned i;

   if (!states || start >= PS5_MAX_VIEWPORTS ||
       count > PS5_MAX_VIEWPORTS - start)
      return;
   for (i = 0; i < count; ++i) {
      context->viewport[start + i] = states[i];
      context->viewport_valid |= UINT16_C(1) << (start + i);
   }
}

static void
ps5_set_polygon_stipple(struct pipe_context *base,
                        const struct pipe_poly_stipple *stipple)
{
   (void)base;
   (void)stipple;
}

static void
ps5_set_window_rectangles(struct pipe_context *base, bool include,
                          unsigned count,
                          const struct pipe_scissor_state *rectangles)
{
   (void)base;
   (void)include;
   (void)count;
   (void)rectangles;
}

static void
ps5_set_sampler_views(struct pipe_context *base, mesa_shader_stage shader,
                      unsigned start, unsigned count,
                      unsigned unbind_trailing,
                      struct pipe_sampler_view **views)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned index;
   unsigned slot;

#ifdef PS5_NATIVE_TITLE_RUNTIME
   if (shader == MESA_SHADER_COMPUTE) {
      ps5_set_compute_sampler_views(base, start, count, unbind_trailing, views);
      return;
   }
#endif
   if (shader == MESA_SHADER_VERTEX)
      slot = 0;
   else if (shader == MESA_SHADER_FRAGMENT)
      slot = 1;
   else if (shader == MESA_SHADER_GEOMETRY &&
            PS5_ENABLE_GEOMETRY_CANDIDATE)
      slot = PS5_GEOMETRY_TEXTURE_SLOT;
   else if (shader == MESA_SHADER_TESS_CTRL && PS5_ENABLE_TESSELLATION_CANDIDATE)
      slot = PS5_TESS_CTRL_TEXTURE_SLOT;
   else if (shader == MESA_SHADER_TESS_EVAL && PS5_ENABLE_TESSELLATION_CANDIDATE)
      slot = PS5_TESS_EVAL_TEXTURE_SLOT;
   else
      return;
   if (start > PS5_MAX_TEXTURE_UNITS ||
       count > PS5_MAX_TEXTURE_UNITS - start ||
       unbind_trailing > PS5_MAX_TEXTURE_UNITS - start - count ||
       (count && !views))
      return;
   for (index = 0; index < count; ++index)
      pipe_sampler_view_reference(
         &context->sampler_views[slot][start + index], views[index]);
   for (; index < count + unbind_trailing; ++index)
      pipe_sampler_view_reference(
         &context->sampler_views[slot][start + index], NULL);
}

static void
ps5_bind_sampler_states(struct pipe_context *base, mesa_shader_stage shader,
                        unsigned start, unsigned count, void **states)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned slot;

#ifdef PS5_NATIVE_TITLE_RUNTIME
   if (shader == MESA_SHADER_COMPUTE) {
      ps5_set_compute_sampler_states(base, start, count, states);
      return;
   }
#endif

   if (shader == MESA_SHADER_VERTEX)
      slot = 0;
   else if (shader == MESA_SHADER_FRAGMENT)
      slot = 1;
   else if (shader == MESA_SHADER_GEOMETRY &&
            PS5_ENABLE_GEOMETRY_CANDIDATE)
      slot = PS5_GEOMETRY_TEXTURE_SLOT;
   else if (shader == MESA_SHADER_TESS_CTRL && PS5_ENABLE_TESSELLATION_CANDIDATE)
      slot = PS5_TESS_CTRL_TEXTURE_SLOT;
   else if (shader == MESA_SHADER_TESS_EVAL && PS5_ENABLE_TESSELLATION_CANDIDATE)
      slot = PS5_TESS_EVAL_TEXTURE_SLOT;
   else
      return;
   if (start > PS5_MAX_TEXTURE_UNITS ||
       count > PS5_MAX_TEXTURE_UNITS - start ||
       (count && !states))
      return;
   for (unsigned index = 0; index < count; ++index)
      context->samplers[slot][start + index] = states[index];
}

static void *
ps5_create_sampler_state(struct pipe_context *base,
                         const struct pipe_sampler_state *state)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_sampler_state *copy;
   union pipe_color_union *table;
   unsigned index;
   bool linear_filter;
   bool uses_border;

   if (!state)
      return NULL;
   copy = calloc(1, sizeof(*copy));
   if (!copy)
      return NULL;
   copy->base = *state;
   copy->words_valid = ps5_encode_sampler_words(state, copy->words);
   if (!PS5_ENABLE_BORDER_COLOR_CANDIDATE)
      return copy;

   linear_filter = state->min_img_filter != PIPE_TEX_FILTER_NEAREST ||
                   state->mag_img_filter != PIPE_TEX_FILTER_NEAREST;
#define PS5_WRAP_USES_BORDER(wrap) \
   ((wrap) == PIPE_TEX_WRAP_CLAMP_TO_BORDER || \
    (wrap) == PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER || \
    (linear_filter && ((wrap) == PIPE_TEX_WRAP_CLAMP || \
                       (wrap) == PIPE_TEX_WRAP_MIRROR_CLAMP)))
   uses_border = PS5_WRAP_USES_BORDER(state->wrap_s) ||
                 PS5_WRAP_USES_BORDER(state->wrap_t) ||
                 PS5_WRAP_USES_BORDER(state->wrap_r);
#undef PS5_WRAP_USES_BORDER
   if (!uses_border)
      return copy;

#define PS5_SIMPLE_BORDER(elt) \
   do { \
      if (state->border_color.elt[0] == 0 && \
          state->border_color.elt[1] == 0 && \
          state->border_color.elt[2] == 0 && \
          state->border_color.elt[3] == 0) \
         return copy; \
      if (state->border_color.elt[0] == 0 && \
          state->border_color.elt[1] == 0 && \
          state->border_color.elt[2] == 0 && \
          state->border_color.elt[3] == 1) { \
         copy->border_color_type = 1; \
         return copy; \
      } \
      if (state->border_color.elt[0] == 1 && \
          state->border_color.elt[1] == 1 && \
          state->border_color.elt[2] == 1 && \
          state->border_color.elt[3] == 1) { \
         copy->border_color_type = 2; \
         return copy; \
      } \
   } while (0)
   if (state->border_color_is_integer)
      PS5_SIMPLE_BORDER(ui);
   else
      PS5_SIMPLE_BORDER(f);
#undef PS5_SIMPLE_BORDER

   if (!context->border_color_storage) {
      free(copy);
      return NULL;
   }
   table = (union pipe_color_union *)
      ((struct ps5_resource *)context->border_color_storage)->data;
   for (index = 0; index < context->border_color_count; ++index)
      if (memcmp(&table[index], &state->border_color,
                 sizeof(state->border_color)) == 0)
         break;
   if (index == context->border_color_count) {
      if (index == PS5_BORDER_COLOR_COUNT) {
         free(copy);
         return NULL;
      }
      table[index] = state->border_color;
      context->border_color_count++;
      ps5_flush_gpu_data(&table[index], sizeof(table[index]));
   }
   copy->border_color_ptr = (uint16_t)index;
   copy->border_color_type = 3;
   return copy;
}

static void
ps5_delete_sampler_state(struct pipe_context *base, void *state)
{
   struct ps5_context *context = (struct ps5_context *)base;

   for (unsigned slot = 0; slot < PS5_TEXTURE_STAGE_COUNT; ++slot)
      for (unsigned index = 0; index < PS5_MAX_TEXTURE_UNITS; ++index)
         if (context->samplers[slot][index] == state)
            context->samplers[slot][index] = NULL;
   free(state);
}

static struct pipe_sampler_view *
ps5_create_sampler_view(struct pipe_context *base,
                        struct pipe_resource *texture,
                        const struct pipe_sampler_view *templ)
{
   struct pipe_sampler_view *view;

   if (texture && templ && texture->target == PIPE_BUFFER) {
      if (!PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE ||
          templ->target != PIPE_BUFFER ||
          !ps5_texel_buffer_format(templ->format) ||
          templ->u.buf.offset > texture->width0 ||
          templ->u.buf.size > texture->width0 - templ->u.buf.offset)
         return NULL;
      view = calloc(1, sizeof(*view));
      if (!view)
         return NULL;
      *view = *templ;
      view->reference.count = 1;
      view->texture = NULL;
      pipe_resource_reference(&view->texture, texture);
      view->context = base;
      return view;
   }
   if (!texture || !templ ||
       !ps5_sampled_texture_target(texture->target) ||
       !ps5_sampled_texture_format(texture->format) ||
       !ps5_sampled_texture_target(templ->target) ||
       !ps5_texture_view_target_compatible(texture->target, templ->target) ||
       !ps5_texture_view_format_compatible(texture->format, templ->format) ||
       templ->u.tex.first_level > templ->u.tex.last_level ||
       templ->u.tex.last_level > texture->last_level ||
       (templ->u.tex.last_level && !PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE) ||
       templ->u.tex.first_layer > templ->u.tex.last_layer ||
       templ->u.tex.last_layer >=
          ps5_texture_level_layers(texture, templ->u.tex.first_level) ||
       ((templ->target == PIPE_TEXTURE_1D ||
         templ->target == PIPE_TEXTURE_2D ||
         templ->target == PIPE_TEXTURE_RECT) &&
        templ->u.tex.first_layer != templ->u.tex.last_layer) ||
       (templ->target == PIPE_TEXTURE_CUBE &&
        templ->u.tex.last_layer - templ->u.tex.first_layer + 1 != 6) ||
       (templ->target == PIPE_TEXTURE_CUBE_ARRAY &&
        (templ->u.tex.last_layer - templ->u.tex.first_layer + 1) % 6) ||
       (templ->target == PIPE_TEXTURE_3D &&
        (templ->u.tex.first_layer || templ->u.tex.last_layer))) {
      printf("[ps5-gallium] sampler-view reject target=%u/%u format=%u/%u levels=%u..%u/%u layers=%u..%u/%u\n",
             texture ? texture->target : 0, templ ? templ->target : 0,
             texture ? texture->format : 0, templ ? templ->format : 0,
             templ ? templ->u.tex.first_level : 0,
             templ ? templ->u.tex.last_level : 0,
             texture ? texture->last_level : 0,
             templ ? templ->u.tex.first_layer : 0,
             templ ? templ->u.tex.last_layer : 0,
             texture ? texture->array_size : 0);
      return NULL;
   }
   view = calloc(1, sizeof(*view));
   if (!view)
      return NULL;
   *view = *templ;
   view->reference.count = 1;
   view->texture = NULL;
   pipe_resource_reference(&view->texture, texture);
   view->context = base;
   return view;
}

static void
ps5_sampler_view_destroy(struct pipe_context *base,
                         struct pipe_sampler_view *view)
{
   (void)base;
   pipe_resource_reference(&view->texture, NULL);
   free(view);
}

static void
ps5_set_stencil_ref(struct pipe_context *base,
                    const struct pipe_stencil_ref ref)
{
   ((struct ps5_context *)base)->stencil_ref = ref;
}

static void
ps5_set_min_samples(struct pipe_context *base, unsigned min_samples)
{
   ((struct ps5_context *)base)->min_samples = min_samples;
}

static void
ps5_set_sample_mask(struct pipe_context *base, unsigned sample_mask)
{
   ((struct ps5_context *)base)->sample_mask = sample_mask;
}

static void
ps5_set_constant_buffer(struct pipe_context *base, mesa_shader_stage shader,
                        unsigned index,
                        const struct pipe_constant_buffer *buffer)
{
#ifdef PS5_NATIVE_TITLE_RUNTIME
   if (shader == MESA_SHADER_COMPUTE) {
      ps5_set_compute_constant_buffer(base, index, buffer);
      return;
   }
#endif
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_constant_state *state;
   struct ps5_resource *storage;
   unsigned slot;
   unsigned descriptor_slot;
   unsigned maximum_size;
   size_t destination_offset;
   size_t copied_size;


   if (index >= (PS5_ENABLE_UBO_CANDIDATE
                    ? PS5_MAX_CONSTANT_BUFFERS : 1))
      return;
   switch (shader) {
   case MESA_SHADER_VERTEX:
      slot = 0;
      descriptor_slot = 0;
      break;
   case MESA_SHADER_FRAGMENT:
      slot = 1;
      descriptor_slot = 1;
      break;
   case MESA_SHADER_GEOMETRY:
      if (!PS5_ENABLE_GEOMETRY_CANDIDATE)
         return;
      slot = PS5_GEOMETRY_CONSTANT_SLOT;
      descriptor_slot = 0;
      break;
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
      if (!PS5_ENABLE_TESSELLATION_CANDIDATE)
         return;
      slot = shader == MESA_SHADER_TESS_CTRL
         ? PS5_TESS_CTRL_CONSTANT_SLOT : PS5_TESS_EVAL_CONSTANT_SLOT;
      descriptor_slot = 0;
      break;
   default:
      return;
   }
   state = &context->constants[slot][index];
   pipe_resource_reference(&state->buffer, NULL);
   state->offset = 0;
   state->valid = false;
   state->copied = false;
   state->size = 0;
   if (!buffer)
      return;
   maximum_size = PS5_ENABLE_UBO_CANDIDATE
                     ? PS5_MAX_CONSTANT_BUFFER_SIZE
                     : PS5_DIRECT_ALIGNMENT - PS5_CONSTANT_DATA_OFFSET;
   if (!buffer->buffer_size ||
       buffer->buffer_size > maximum_size ||
       (index && (buffer->buffer_offset & 15u)))
      return;
   if (buffer->user_buffer) {
      if (index)
         return;
      storage =
         (struct ps5_resource *)context->descriptor_storage[descriptor_slot];
      destination_offset = ps5_copied_constant_offset(slot);
      copied_size = (buffer->buffer_size + 15u) & ~15u;
      if (!storage || destination_offset > storage->size ||
          copied_size > storage->size - destination_offset)
         return;
      memcpy(storage->data + destination_offset, buffer->user_buffer,
             buffer->buffer_size);
      memset(storage->data + destination_offset + buffer->buffer_size, 0,
             copied_size - buffer->buffer_size);
      state->copied = true;
   } else {
      struct ps5_resource *resource =
         (struct ps5_resource *)buffer->buffer;

      if (!resource || resource->base.target != PIPE_BUFFER ||
          buffer->buffer_offset > resource->size ||
          buffer->buffer_size > resource->size - buffer->buffer_offset)
         return;
      pipe_resource_reference(&state->buffer, buffer->buffer);
      state->offset = buffer->buffer_offset;
   }
   state->size = buffer->buffer_size;
   state->valid = true;
}

static void
ps5_set_vertex_buffers(struct pipe_context *base, unsigned count,
                       const struct pipe_vertex_buffer *buffers)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct pipe_vertex_buffer next[PIPE_MAX_ATTRIBS];
   bool valid = count && count <= PIPE_MAX_ATTRIBS && buffers;
   unsigned index;

   for (index = 0; valid && index < count; ++index)
      valid = !buffers[index].is_user_buffer;
   /* Take the new references before dropping the old ones: u_vbuf passes
    * back raw pointers it kept from an earlier call (it does not own the
    * references), so a buffer in this call can be one this context already
    * holds, and the context's reference may be the last. Dropping the old
    * references first destroys it, and referencing it afterwards fails
    * pipe_reference_described's "src had to be referenced" assertion (a
    * released resource) in builds with assertions on. */
   memset(next, 0, sizeof(next));
   for (index = 0; valid && index < count; ++index) {
      next[index] = buffers[index];
      next[index].buffer.resource = NULL;
      pipe_resource_reference(&next[index].buffer.resource,
                             buffers[index].buffer.resource);
   }
   for (index = 0; index < context->vertex_buffer_count; ++index)
      pipe_resource_reference(
         &context->vertex_buffers[index].buffer.resource, NULL);
   memcpy(context->vertex_buffers, next, sizeof(context->vertex_buffers));
   context->vertex_buffer_count = valid ? count : 0;
}

static struct pipe_stream_output_target *
ps5_create_stream_output_target(struct pipe_context *base,
                                struct pipe_resource *buffer,
                                unsigned buffer_offset,
                                unsigned buffer_size)
{
   struct ps5_stream_output_target *target;

   if (!PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE || !buffer ||
       buffer->target != PIPE_BUFFER || (buffer_offset & 3u) ||
       (buffer_size & 3u) || buffer_offset > buffer->width0 ||
       buffer_size > buffer->width0 - buffer_offset)
      return NULL;
   target = calloc(1, sizeof(*target));
   if (!target)
      return NULL;
   target->base.reference.count = 1;
   target->base.context = base;
   pipe_resource_reference(&target->base.buffer, buffer);
   target->base.buffer_offset = buffer_offset;
   target->base.buffer_size = buffer_size;
   return &target->base;
}

static void
ps5_stream_output_target_destroy(struct pipe_context *base,
                                 struct pipe_stream_output_target *pipe_target)
{
   struct ps5_stream_output_target *target =
      (struct ps5_stream_output_target *)pipe_target;

   (void)base;
   pipe_resource_reference(&target->base.buffer, NULL);
   free(target);
}

static uint32_t
ps5_stream_output_target_offset(struct pipe_stream_output_target *pipe_target)
{
   struct ps5_stream_output_target *target =
      (struct ps5_stream_output_target *)pipe_target;

   return target ? target->offset : 0;
}

static bool
ps5_stream_output_primitive_valid(enum mesa_prim primitive)
{
   return primitive == MESA_PRIM_POINTS || primitive == MESA_PRIM_LINES ||
          primitive == MESA_PRIM_TRIANGLES;
}

static void
ps5_set_stream_output_targets(struct pipe_context *base,
                              unsigned count,
                              struct pipe_stream_output_target **targets,
                              const unsigned *offsets,
                              enum mesa_prim output_primitive)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned append_mask = 0;
   unsigned index;

   if (!PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE ||
       count > PIPE_MAX_SO_BUFFERS ||
       (count && (!targets || !offsets ||
                  !ps5_stream_output_primitive_valid(output_primitive))))
      return;
   for (index = 0; index < count; ++index) {
      if (!targets[index])
         continue;
      if (targets[index]->context != base ||
          targets[index]->buffer->target != PIPE_BUFFER)
         return;
      if (offsets[index] == UINT32_MAX) {
         append_mask |= BITFIELD_BIT(index);
      } else if ((offsets[index] & 3u) ||
                 offsets[index] > targets[index]->buffer_size) {
         return;
      }
   }

   for (index = 0; index < count; ++index) {
      pipe_so_target_reference(&context->stream_output_targets[index],
                               targets[index]);
      if (targets[index] && !(append_mask & BITFIELD_BIT(index))) {
         struct ps5_stream_output_target *target =
            (struct ps5_stream_output_target *)targets[index];
         target->offset = offsets[index];
         target->vertex_count = 0;
      }
   }
   for (; index < context->stream_output_target_count; ++index)
      pipe_so_target_reference(&context->stream_output_targets[index], NULL);
   context->stream_output_target_count = count;
   context->stream_output_primitive = count ? output_primitive
                                             : MESA_PRIM_UNKNOWN;
}

static void
ps5_buffer_subdata(struct pipe_context *base, struct pipe_resource *resource,
                   unsigned usage, unsigned offset, unsigned size,
                   const void *data)
{
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {5, ps5_prepare_clock()};
#endif

   struct ps5_resource *buffer = (struct ps5_resource *)resource;

   (void)base;
   if (!buffer || buffer->base.target != PIPE_BUFFER || !data ||
       offset > buffer->size || size > buffer->size - offset)
      return;
   if (!(usage & PIPE_MAP_UNSYNCHRONIZED))
      ps5_draw_batch_drain_buffer(resource);
   else
      ps5_invalidate_gpu_writes(buffer);
   memcpy(buffer->data + offset, data, size);
   /* UNSYNCHRONIZED writes may fill previously unused holes inside an already
    * published bounding range. Publish at the write, even without a drain. */
   ps5_flush_gpu_data(buffer->data + offset, size);
}

/* Whole, identical single-level images need no detile/re-tile round trip. */
static bool
ps5_copy_identical_image(struct pipe_resource *dst_base, unsigned dst_level,
                         unsigned dst_x, unsigned dst_y, unsigned dst_z,
                         struct pipe_resource *src_base, unsigned src_level,
                         const struct pipe_box *box)
{
   struct ps5_resource *dst = (struct ps5_resource *)dst_base;
   struct ps5_resource *src = (struct ps5_resource *)src_base;
   if (!src || !dst || !box || src_level || dst_level || dst_x || dst_y || dst_z ||
       box->x || box->y || box->z || box->depth != 1 ||
       (src_base->target != PIPE_TEXTURE_2D && src_base->target != PIPE_TEXTURE_2D_ARRAY) ||
       dst_base->target != src_base->target ||
       src_base->last_level || dst_base->last_level ||
       src_base->array_size != 1 || dst_base->array_size != 1 ||
       src_base->depth0 != 1 || dst_base->depth0 != 1 ||
       src_base->nr_samples > 1 || dst_base->nr_samples > 1 ||
       src_base->nr_storage_samples > 1 || dst_base->nr_storage_samples > 1 ||
       src_base->format != dst_base->format || src_base->bind != dst_base->bind ||
       (src_base->bind & PIPE_BIND_DISPLAY_TARGET) ||
       box->width <= 0 || box->height <= 0 ||
       (unsigned)box->width != dst_base->width0 || (unsigned)box->height != dst_base->height0 ||
       src_base->width0 < dst_base->width0 || src_base->height0 < dst_base->height0 ||
       src->render_staging_size || dst->render_staging_size ||
       src->depth_staging_size || dst->depth_staging_size ||
       src->external_cpu_access || dst->external_cpu_access ||
       !src->data || !dst->data || !src->allocation_size ||
       src->allocation_size != dst->allocation_size ||
       src->stencil_allocation_size != dst->stencil_allocation_size ||
       (!!src->stencil_data != !!dst->stencil_data) ||
       (src->stencil_allocation_size && !src->stencil_data))
      return false;
   size_t bytes = src->allocation_size, stencil_bytes = src->stencil_allocation_size;
   const bool depth = (src_base->bind & PIPE_BIND_DEPTH_STENCIL) &&
      (src_base->format == PIPE_FORMAT_Z32_FLOAT ||
       src_base->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);
   if (depth) {
      if (src->level_offset[0] || dst->level_offset[0])
         return false;
      /* The destination is fully overwritten. Extra source columns/rows can
       * occupy only destination padding when both physical tile grids match. */
      bytes = ps5_tiled_depth_surface_size(src_base->width0, src_base->height0, 1);
      if (!bytes || bytes != ps5_tiled_depth_surface_size(dst_base->width0, dst_base->height0, 1) ||
          bytes > src->allocation_size)
         return false;
      stencil_bytes = src_base->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT
         ? ps5_tiled_stencil_surface_size(src_base->width0, src_base->height0) : 0;
      if (stencil_bytes && (!src->stencil_data ||
          stencil_bytes != ps5_tiled_stencil_surface_size(dst_base->width0, dst_base->height0) ||
          stencil_bytes > src->stencil_allocation_size))
         return false;
   } else if (src_base->target != PIPE_TEXTURE_2D ||
              src_base->width0 != dst_base->width0 || src_base->height0 != dst_base->height0 ||
              src->size != dst->size || src->stride != dst->stride ||
              src->layer_stride != dst->layer_stride ||
              src->level_offset[0] != dst->level_offset[0] || src->level_stride[0] != dst->level_stride[0]) {
      return false;
   }
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DRAW_PROFILE)
   struct ps5_prepare_scope scope __attribute__((cleanup(ps5_prepare_scope_end))) =
      {8, ps5_prepare_clock()};
#endif
   ps5_draw_batch_drain_buffer(src_base);
   ps5_draw_batch_drain_buffer(dst_base);
   ps5_flush_gpu_data(src->data, bytes);
   memmove(dst->data, src->data, bytes);
   ps5_flush_gpu_data(dst->data, bytes);
   if (stencil_bytes) {
      ps5_flush_gpu_data(src->stencil_data, stencil_bytes);
      memmove(dst->stencil_data, src->stencil_data, stencil_bytes);
      ps5_flush_gpu_data(dst->stencil_data, stencil_bytes);
   }
   return true;
}

static void
ps5_resource_copy_region(struct pipe_context *base, struct pipe_resource *dst,
                          unsigned dst_level, unsigned dst_x, unsigned dst_y, unsigned dst_z,
                          struct pipe_resource *src, unsigned src_level,
                          const struct pipe_box *src_box)
{
   if (ps5_copy_identical_image(dst, dst_level, dst_x, dst_y, dst_z, src, src_level, src_box))
      return;
   util_resource_copy_region(base, dst, dst_level, dst_x, dst_y, dst_z, src, src_level, src_box);
}

static void
ps5_sampler_view_release(struct pipe_context *base,
                         struct pipe_sampler_view *view)
{
   u_default_sampler_view_release(base, view);
}

static void
ps5_delete_shader_state(struct pipe_context *base, void *state)
{
   struct ps5_context *context = (struct ps5_context *)base;
   struct ps5_shader *shader = state;
   struct ps5_shader_variant *variant;

   if (!shader)
      return;
   /* Native workers still borrow variant packages until the batch drains. */
   ps5_draw_batch_drain();
   printf("[ps5-gallium] delete-shader stage=%u state=%p\n",
          shader->stage, (void *)shader);
   if (context->vs == shader)
      context->vs = NULL;
   if (context->tcs == shader)
      context->tcs = NULL;
   if (context->default_tcs == shader)
      context->default_tcs = NULL;
   if (context->tes == shader)
      context->tes = NULL;
   if (context->gs == shader)
      context->gs = NULL;
   if (context->fs == shader)
      context->fs = NULL;
   if (context->geometry_vs == shader || context->geometry_gs == shader)
      ps5_release_geometry_pipeline(context);
   if (context->tessellation_vs == shader ||
       context->tessellation_tcs == shader ||
       context->tessellation_tes == shader ||
       context->tessellation_gs == shader)
      ps5_release_tessellation_pipeline(context);
   while ((variant = shader->variants)) {
      shader->variants = variant->next;
      free(variant->package);
      free(variant->streamout_package);
      psbc_free_output(&variant->streamout_output);
      psbc_free_output(&variant->output);
      free(variant);
   }
   ralloc_free(shader->nir);
   free(shader);
}

int
ps5_shader_state_info(void *state, size_t *machine_code_size,
                      unsigned *hardware_stage, unsigned *unresolved_fields)
{
   struct ps5_shader *shader = state;

   if (!shader || !shader->active)
      return -1;
   if (machine_code_size)
      *machine_code_size = shader->active->output.machine_code_size;
   if (hardware_stage)
      *hardware_stage = shader->active->output.metadata.hardware_stage;
   if (unresolved_fields)
      *unresolved_fields = shader->active->output.metadata.unresolved_fields;
   return 0;
}

int
ps5_context_last_draw_status(struct pipe_context *base, unsigned *draw_calls)
{
   struct ps5_context *context = (struct ps5_context *)base;

   ps5_draw_batch_drain();
   if (!context)
      return -1;
   if (draw_calls)
      *draw_calls = context->draw_calls;
   return context->last_draw_status;
}

int
ps5_context_last_compute_status(struct pipe_context *base, unsigned *dispatches)
{
   struct ps5_context *context = (struct ps5_context *)base;
   if (!context)
      return -1;
   if (dispatches)
      *dispatches = context->dispatches;
   return context->last_compute_status;
}

static void
ps5_context_profile_report(struct ps5_context *context)
{
#ifdef PS5_DRAW_PROFILE
#ifdef PS5_NATIVE_TITLE_RUNTIME
   ps5_prepare_report();
#endif
   printf("[ps5-cpu-flush-summary] calls=%" PRIu64 " bytes=%" PRIu64 "\n",
          __atomic_load_n(&ps5_cpu_flush_calls, __ATOMIC_RELAXED),
          __atomic_load_n(&ps5_cpu_flush_bytes, __ATOMIC_RELAXED));
#endif
   /* Build/lifecycle receipt remains available without hot-path profiling. */
   fprintf(stderr, "[ps5-batch-summary] config gpu-present="
#ifdef PS5_GPU_PRESENT_BATCH
          "1"
#else
          "0"
#endif
          " multidraw="
#ifdef PS5_MULTIDRAW_BATCH
          "1"
#else
          "0"
#endif
          " deferred="
#ifdef PS5_DEFERRED_DRAW_BATCH
          "1"
#else
          "0"
#endif
          " eligible=%" PRIu64 " reject-input=%" PRIu64
          " reject-shader=%" PRIu64 " reject-query=%" PRIu64
          " reject-framebuffer=%" PRIu64 " reject-depth=%" PRIu64
          " reject-vertex=%" PRIu64 " reject-texture=%" PRIu64 " profile="
#ifdef PS5_DRAW_PROFILE
          "1\n",
#else
          "0\n",
#endif
          context->batch_eligible, context->batch_reject[0],
          context->batch_reject[1], context->batch_reject[2],
          context->batch_reject[3], context->batch_reject[4],
          context->batch_reject[5], context->batch_reject[6]);
#ifdef PS5_DRAW_PROFILE
   for (unsigned i = 0; i < 32 && context->framebuffer_fallbacks[i].count; ++i) {
      const unsigned *k = context->framebuffer_fallbacks[i].key;
      printf("[ps5-framebuffer-fallback] count=%" PRIu64
             " reason=%u targets=%u slot=%u format=%u resource_format=%u target=%u"
             " width=%u height=%u samples=%u mip=%u first_layer=%u last_layer=%u"
             " stride=%u bind=%u storage_samples=%u address_mod256=%u\n",
             context->framebuffer_fallbacks[i].count,
             k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
             k[8], k[9], k[10], k[11], k[12], k[13], k[14], k[15]);
   }
   if (context->framebuffer_fallback_overflow)
      printf("[ps5-framebuffer-fallback] overflow=%" PRIu64 "\n",
             context->framebuffer_fallback_overflow);
#endif
}

static void
ps5_context_destroy(struct pipe_context *base)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned index;

   ps5_draw_batch_drain();
#if defined(PS5_NATIVE_TITLE_RUNTIME) && defined(PS5_DEFERRED_DRAW_BATCH)
   printf("[ps5-descriptor-reuse] hits=%" PRIu64 " misses=%" PRIu64 " retained=%u\n",
          context->descriptor_cache_hits, context->descriptor_cache_misses,
          context->descriptor_cache_count[0] + context->descriptor_cache_count[1] +
          context->descriptor_cache_count[2]);
   ps5_descriptor_cache_clear(context);
#endif

   ps5_context_profile_report(context);
   for (index = 0; index < PS5_COMPUTE_BUFFER_SLOTS; ++index)
      pipe_resource_reference(&context->compute_buffers[index].buffer, NULL);
   for (index = 0; index < PS5_COMPUTE_STORAGE_SLOTS; ++index)
      pipe_resource_reference(&context->fragment_buffers[index].buffer, NULL);
   for (index = 0; index < PS5_COMPUTE_STORAGE_SLOTS; ++index)
      pipe_resource_reference(&context->geometry_buffers[index].buffer, NULL);
   for (unsigned stage = 0; stage < 3; ++stage)
      for (index = 0; index < PS5_COMPUTE_STORAGE_SLOTS; ++index)
         pipe_resource_reference(&context->preraster_buffers[stage][index].buffer, NULL);
   for (index = 0; index < PS5_COMPUTE_IMAGE_SLOTS; ++index)
      pipe_resource_reference(&context->compute_images[index].resource, NULL);
   for (index = 0; index < PS5_COMPUTE_IMAGE_SLOTS; ++index)
      pipe_resource_reference(&context->fragment_images[index].resource, NULL);
   for (unsigned stage = 0; stage < 4; ++stage)
      for (index = 0; index < PS5_COMPUTE_IMAGE_SLOTS; ++index)
         pipe_resource_reference(
            &context->preraster_images[stage][index].resource, NULL);
   pipe_resource_reference(&context->compute_descriptors, NULL);
   for (unsigned i = 0; i < PS5_COMPUTE_TEXTURE_SLOTS; ++i)
      pipe_sampler_view_reference(&context->compute_views[i], NULL);
   if (context->blitter)
      util_blitter_destroy(context->blitter);
   ps5_release_geometry_pipeline(context);
   ps5_release_tessellation_pipeline(context);
   util_unreference_framebuffer_state(&context->framebuffer);
   for (index = 0; index < context->vertex_buffer_count; ++index)
      pipe_resource_reference(
         &context->vertex_buffers[index].buffer.resource, NULL);
   for (index = 0; index < context->stream_output_target_count; ++index)
      pipe_so_target_reference(&context->stream_output_targets[index], NULL);
   pipe_resource_reference(&context->streamout_records, NULL);
   pipe_resource_reference(&context->primitive_query_storage, NULL);
   ps5_delete_shader_state(base, context->default_tcs);
   ps5_release_tess_capture(context);
   ps5_delete_shader_state(base, context->tess_capture_fs);
   pipe_resource_reference(&context->tess_capture_buffer, NULL);
   for (index = 0; index < PIPE_MAX_SO_BUFFERS; ++index)
      pipe_resource_reference(&context->streamout_staging[index], NULL);
   pipe_resource_reference(&context->vertex_descriptor_table, NULL);
   for (unsigned slot = 0; slot < PS5_TEXTURE_STAGE_COUNT; ++slot)
      for (index = 0; index < PS5_MAX_TEXTURE_UNITS; ++index)
         pipe_sampler_view_reference(&context->sampler_views[slot][index],
                                     NULL);
   for (index = 0; index < PS5_CONSTANT_STAGE_COUNT; ++index) {
      unsigned binding;

      for (binding = 0; binding < PS5_MAX_CONSTANT_BUFFERS; ++binding)
         pipe_resource_reference(&context->constants[index][binding].buffer,
                                 NULL);
   }
   for (index = 0; index < PS5_DESCRIPTOR_STAGE_COUNT; ++index)
      pipe_resource_reference(&context->descriptor_storage[index], NULL);
   pipe_resource_reference(&context->border_color_storage, NULL);
   if (base->stream_uploader)
      u_upload_destroy(base->stream_uploader);
   if (base->screen->num_contexts)
      base->screen->num_contexts--;
   free(context);
}

static struct pipe_context *
ps5_context_create(struct pipe_screen *screen, void *priv, unsigned flags)
{
   struct ps5_context *context;
   struct pipe_resource descriptor_template;

   /* Robust buffer access and reset notification need no context state here. */
   if (flags & ~(PIPE_CONTEXT_PREFER_THREADED | PIPE_CONTEXT_ROBUST_BUFFER_ACCESS |
                 PIPE_CONTEXT_LOSE_CONTEXT_ON_RESET))
      return NULL;

   context = calloc(1, sizeof(*context));
   if (!context)
      return NULL;

   memset(&descriptor_template, 0, sizeof(descriptor_template));
   descriptor_template.target = PIPE_BUFFER;
   descriptor_template.format = PIPE_FORMAT_R8_UNORM;
   descriptor_template.width0 = PS5_DIRECT_ALIGNMENT;
   descriptor_template.height0 = 1;
   descriptor_template.depth0 = 1;
   descriptor_template.array_size = 1;
   context->vertex_descriptor_table =
      screen->resource_create(screen, &descriptor_template);
   if (!context->vertex_descriptor_table) {
      free(context);
      return NULL;
   }
   descriptor_template.width0 = PS5_ENABLE_UBO_CANDIDATE
                                   ? PS5_DESCRIPTOR_STORAGE_BYTES
                                   : PS5_DIRECT_ALIGNMENT;
   for (unsigned index = 0; index < PS5_DESCRIPTOR_STAGE_COUNT; ++index) {
      context->descriptor_storage[index] =
         screen->resource_create(screen, &descriptor_template);
      if (!context->descriptor_storage[index]) {
         unsigned release;

         for (release = 0; release < index; ++release)
            pipe_resource_reference(
               &context->descriptor_storage[release], NULL);
         pipe_resource_reference(&context->vertex_descriptor_table, NULL);
         free(context);
         return NULL;
      }
   }
   if (PS5_ENABLE_BORDER_COLOR_CANDIDATE) {
      descriptor_template.width0 = PS5_BORDER_COLOR_BYTES;
      context->border_color_storage =
         screen->resource_create(screen, &descriptor_template);
      if (!context->border_color_storage) {
         for (unsigned index = 0; index < PS5_DESCRIPTOR_STAGE_COUNT;
              ++index)
            pipe_resource_reference(&context->descriptor_storage[index],
                                    NULL);
         pipe_resource_reference(&context->vertex_descriptor_table, NULL);
         free(context);
         return NULL;
      }
   }

   context->base.screen = screen;
   context->base.priv = priv;
   context->sample_mask = UINT32_C(0xffff);
   context->patch_vertices = 3;
   for (unsigned i = 0; i < 6; ++i)
      context->tess_levels[i] = 1.0f;
   context->queries_enabled = true;
   context->base.destroy = ps5_context_destroy;
#ifdef PS5_NATIVE_TITLE_RUNTIME
   context->base.create_compute_state = ps5_create_compute_state;
   context->base.bind_compute_state = ps5_bind_compute_state;
   context->base.delete_compute_state = ps5_delete_compute_state;
   context->base.get_compute_state_info = ps5_get_compute_state_info;
   context->base.set_shader_buffers = ps5_set_shader_buffers;
   context->base.set_shader_images = ps5_set_shader_images;
   context->base.launch_grid = ps5_launch_grid;
#endif
   context->base.draw_vbo = ps5_draw_vbo_reported;
   context->base.memory_barrier = ps5_memory_barrier;
   context->base.texture_barrier = ps5_texture_barrier;
   context->base.create_query = ps5_create_query;
   context->base.destroy_query = ps5_destroy_query;
   context->base.begin_query = ps5_begin_query;
   context->base.end_query = ps5_end_query;
   context->base.get_query_result = ps5_get_query_result;
   context->base.get_query_result_resource = ps5_get_query_result_resource;
   context->base.set_active_query_state = ps5_set_active_query_state;
   if (PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE)
      context->base.render_condition = ps5_render_condition;
   context->base.create_vs_state = ps5_create_vs_state;
   context->base.bind_vs_state = ps5_bind_vs_state;
   context->base.delete_vs_state = ps5_delete_shader_state;
   context->base.create_fs_state = ps5_create_fs_state;
   context->base.bind_fs_state = ps5_bind_fs_state;
   context->base.delete_fs_state = ps5_delete_shader_state;
   context->base.create_gs_state = ps5_create_gs_state;
   context->base.bind_gs_state = ps5_bind_gs_state;
   context->base.delete_gs_state = ps5_delete_shader_state;
   if (PS5_ENABLE_TESSELLATION_CANDIDATE) {
      context->base.create_tcs_state = ps5_create_tcs_state;
      context->base.bind_tcs_state = ps5_bind_tcs_state;
      context->base.delete_tcs_state = ps5_delete_shader_state;
      context->base.create_tes_state = ps5_create_tes_state;
      context->base.bind_tes_state = ps5_bind_tes_state;
      context->base.delete_tes_state = ps5_delete_shader_state;
      context->base.set_patch_vertices = ps5_set_patch_vertices;
      context->base.set_tess_state = ps5_set_tess_state;
   }
   context->base.create_blend_state = ps5_create_blend_state;
   context->base.bind_blend_state = ps5_bind_blend_state;
   context->base.delete_blend_state = ps5_delete_fixed_state;
   context->base.create_sampler_state = ps5_create_sampler_state;
   context->base.bind_sampler_states = ps5_bind_sampler_states;
   context->base.delete_sampler_state = ps5_delete_sampler_state;
   context->base.create_rasterizer_state = ps5_create_rasterizer_state;
   context->base.bind_rasterizer_state = ps5_bind_rasterizer_state;
   context->base.delete_rasterizer_state = ps5_delete_fixed_state;
   context->base.create_depth_stencil_alpha_state =
      ps5_create_depth_stencil_alpha_state;
   context->base.bind_depth_stencil_alpha_state =
      ps5_bind_depth_stencil_alpha_state;
   context->base.delete_depth_stencil_alpha_state = ps5_delete_fixed_state;
   context->base.create_vertex_elements_state =
      ps5_create_vertex_elements_state;
   context->base.bind_vertex_elements_state = ps5_bind_vertex_elements_state;
   context->base.delete_vertex_elements_state =
      ps5_delete_vertex_elements_state;
   context->base.set_blend_color = ps5_set_blend_color;
   context->base.set_stencil_ref = ps5_set_stencil_ref;
   context->base.set_sample_mask = ps5_set_sample_mask;
   context->base.set_min_samples = ps5_set_min_samples;
   context->base.get_sample_position = u_default_get_sample_position;
   context->base.set_scissor_states = ps5_set_scissor_states;
   context->base.set_viewport_states = ps5_set_viewport_states;
   context->base.set_polygon_stipple = ps5_set_polygon_stipple;
   context->base.set_window_rectangles = ps5_set_window_rectangles;
   context->base.set_sampler_views = ps5_set_sampler_views;
   context->base.create_sampler_view = ps5_create_sampler_view;
   context->base.sampler_view_destroy = ps5_sampler_view_destroy;
   context->base.set_constant_buffer = ps5_set_constant_buffer;
   context->base.set_vertex_buffers = ps5_set_vertex_buffers;
   if (PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE) {
      context->base.create_stream_output_target =
         ps5_create_stream_output_target;
      context->base.stream_output_target_destroy =
         ps5_stream_output_target_destroy;
      context->base.set_stream_output_targets =
         ps5_set_stream_output_targets;
      context->base.stream_output_target_offset =
         ps5_stream_output_target_offset;
   }
   context->base.buffer_subdata = ps5_buffer_subdata;
   context->base.texture_subdata = ps5_texture_subdata;
   context->base.generate_mipmap = ps5_generate_mipmap;
   context->base.sampler_view_release = ps5_sampler_view_release;
   context->base.set_framebuffer_state = ps5_set_framebuffer_state;
   context->base.clear = ps5_clear;
   context->base.flush = ps5_flush;
   context->base.buffer_map = ps5_transfer_map;
   context->base.texture_map = ps5_transfer_map;
   context->base.transfer_flush_region = ps5_transfer_flush_region;
   context->base.buffer_unmap = ps5_transfer_unmap;
   context->base.texture_unmap = ps5_transfer_unmap;
   context->base.resource_copy_region = ps5_resource_copy_region;
   if (PS5_ENABLE_SOFTWARE_BLIT_CANDIDATE || PS5_ENABLE_MSAA4_CANDIDATE)
      context->base.blit = ps5_blit;
   context->base.resource_release = u_default_resource_release;
   context->base.stream_uploader = u_upload_create_default(&context->base);
   if (!context->base.stream_uploader) {
      for (unsigned index = 0; index < PS5_DESCRIPTOR_STAGE_COUNT; ++index)
         pipe_resource_reference(&context->descriptor_storage[index], NULL);
      pipe_resource_reference(&context->border_color_storage, NULL);
      pipe_resource_reference(&context->vertex_descriptor_table, NULL);
      free(context);
      return NULL;
   }
   context->base.const_uploader = context->base.stream_uploader;
   screen->num_contexts++;
   return &context->base;
}

static void
ps5_screen_destroy(struct pipe_screen *base)
{
   struct ps5_screen *screen = (struct ps5_screen *)base;

   ps5_draw_batch_drain();
#ifdef PS5_NATIVE_TITLE_RUNTIME
   ps5_staging_clear();
#endif
   simple_mtx_destroy(&screen->submit_mutex);
   simple_mtx_destroy(&screen->resource_mutex);
   free(base);
}

struct pipe_screen *
ps5_screen_create(void)
{
   printf("[ps5-cache] clflushopt=%u cacheline=%u\n",
          util_get_cpu_caps()->has_clflushopt, util_get_cpu_caps()->cacheline);
   struct ps5_screen *screen = calloc(1, sizeof(*screen));
   struct nir_shader_compiler_options *vs_options;
   struct nir_shader_compiler_options *tcs_options;
   struct nir_shader_compiler_options *tes_options;
   struct nir_shader_compiler_options *gs_options;
   struct nir_shader_compiler_options *fs_options;
   struct nir_shader_compiler_options *cs_options;
   struct pipe_caps *caps;
   struct pipe_shader_caps *vs_caps;
   struct pipe_shader_caps *tcs_caps;
   struct pipe_shader_caps *tes_caps;
   struct pipe_shader_caps *gs_caps;
   struct pipe_shader_caps *fs_caps;

   if (!screen)
      return NULL;

   simple_mtx_init(&screen->resource_mutex, mtx_plain);
   simple_mtx_init(&screen->submit_mutex, mtx_plain);

   screen->base.destroy = ps5_screen_destroy;
   screen->base.get_name = ps5_get_name;
   screen->base.get_vendor = ps5_get_vendor;
   screen->base.get_device_vendor = ps5_get_device_vendor;
   screen->base.context_create = ps5_context_create;
   screen->base.is_format_supported = ps5_is_format_supported;
   screen->base.can_create_resource = ps5_can_create_resource;
   screen->base.resource_create = ps5_resource_create;
   screen->base.resource_destroy = ps5_resource_destroy;
   screen->base.fence_reference = ps5_fence_reference;
   screen->base.fence_finish = ps5_fence_finish;
   screen->base.get_timestamp = ps5_get_timestamp;

   caps = (struct pipe_caps *)&screen->base.caps;
   /* Buffer-only unsynchronized maps don't touch context state. Their callers
    * own disjoint ranges; explicit flush/unmap publishes the changed bytes. */
   const char *threaded = getenv("PS5_GLTHREAD");
   caps->map_unsynchronized_thread_safe = threaded && !strcmp(threaded, "1");
   caps->allow_mapped_buffers_during_execution = caps->map_unsynchronized_thread_safe;
   caps->max_label_length = 256; /* Mesa's software object-label storage. */
   caps->graphics = true;
   caps->accelerated = 1;
   caps->uma = true;
   caps->npot_textures = true;
   caps->texture_shadow_map = true;
   caps->native_fp32_depth = true;
   caps->clear_scissored = true;
   caps->mixed_framebuffer_sizes = PS5_ENABLE_PADDED_FBO_CANDIDATE;
   caps->mixed_colorbuffer_formats = PS5_ENABLE_MRT_CANDIDATE;
   caps->dest_surface_srgb_control =
      PS5_ENABLE_FRAMEBUFFER_SRGB_CANDIDATE;
   caps->blend_equation_separate = true;
   caps->doubles = PS5_ENABLE_FP64_CANDIDATE;
   caps->int64 = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->glsl_feature_level = PS5_ENABLE_GLSL_460_CANDIDATE ? 460 :
                              PS5_ENABLE_GLSL_450_CANDIDATE ? 450 :
                              PS5_ENABLE_GLSL_440_CANDIDATE ? 440 :
                              PS5_ENABLE_GLSL_430_CANDIDATE ? 430 :
                              PS5_ENABLE_GLSL_420_CANDIDATE ? 420 :
                              PS5_ENABLE_GLSL_410_CANDIDATE ? 410 :
                              PS5_ENABLE_GLSL_400_CANDIDATE ? 400 :
                              PS5_ENABLE_GLSL_330_CANDIDATE ? 330 :
                              PS5_ENABLE_GEOMETRY_CANDIDATE ? 150 : 140;
   caps->glsl_feature_level_compatibility =
      PS5_ENABLE_GLSL_460_CANDIDATE ? 460 :
      PS5_ENABLE_GLSL_450_CANDIDATE ? 450 :
      PS5_ENABLE_GLSL_440_CANDIDATE ? 440 :
      PS5_ENABLE_GLSL_430_CANDIDATE ? 430 :
      PS5_ENABLE_GLSL_420_CANDIDATE ? 420 :
      PS5_ENABLE_GLSL_410_CANDIDATE ? 410 :
      PS5_ENABLE_GLSL_400_CANDIDATE ? 400 :
      PS5_ENABLE_GLSL_330_CANDIDATE ? 330 :
      PS5_ENABLE_GEOMETRY_CANDIDATE ? 150 : 140;
   caps->max_render_targets = PS5_ENABLE_MRT_CANDIDATE
                                  ? PS5_MAX_RENDER_TARGETS : 1;
   caps->indep_blend_enable = PS5_ENABLE_INDEPENDENT_BLEND_CANDIDATE;
   caps->indep_blend_func = PS5_ENABLE_INDEPENDENT_BLEND_CANDIDATE;
   caps->max_dual_source_render_targets =
      PS5_ENABLE_DUAL_SOURCE_BLEND_CANDIDATE ? 1 : 0;
   caps->max_texture_2d_size = PS5_MAX_TEXTURE_2D_SIZE;
   caps->max_texture_cube_levels =
      PS5_ENABLE_TEXTURE_CUBE_CANDIDATE ? PS5_MAX_TEXTURE_CUBE_LEVELS : 0;
   caps->max_texture_array_layers =
      PS5_ENABLE_TEXTURE_ARRAY_CANDIDATE ? PS5_MAX_TEXTURE_ARRAY_LAYERS : 0;
   caps->max_texture_3d_levels =
      PS5_ENABLE_TEXTURE_3D_CANDIDATE ? PS5_MAX_TEXTURE_3D_LEVELS : 0;
   caps->rasterizer_subpixel_bits = 8;
   caps->min_texel_offset = -8;
   caps->max_texel_offset = 7;
   caps->min_texture_gather_offset = -8;
   caps->max_texture_gather_offset = 7;
   caps->max_texture_gather_components = 4;
   /* GFX10 rasterization provides upper-left window coordinates.  Mesa's
    * state tracker lowers OpenGL's lower-left convention through CB0. */
   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_half_integer = true;
   caps->fs_coord_pixel_center_integer = true;
   caps->fs_position_is_sysval = true;
   caps->fs_face_is_integer_sysval = true;
   caps->point_size_fixed = PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE
                               ? PIPE_POINT_SIZE_LOWER_NEVER
                               : PIPE_POINT_SIZE_LOWER_ALWAYS;
   caps->min_point_size = PS5_MIN_POINT_LINE_SIZE;
   caps->min_point_size_aa = PS5_MIN_POINT_LINE_SIZE;
   caps->max_point_size = PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE
                             ? PS5_MAX_POINT_LINE_SIZE : 1.0f;
   caps->max_point_size_aa = caps->max_point_size;
   caps->point_size_granularity = PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE
                                     ? 0.125f : 1.0f;
   caps->min_line_width = PS5_MIN_POINT_LINE_SIZE;
   caps->min_line_width_aa = PS5_MIN_POINT_LINE_SIZE;
   caps->max_line_width = PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE
                             ? PS5_MAX_POINT_LINE_SIZE : 1.0f;
   caps->max_line_width_aa = caps->max_line_width;
   caps->line_width_granularity = PS5_ENABLE_POINT_LINE_SIZE_CANDIDATE
                                     ? 0.125f : 1.0f;
   caps->max_viewports =
      PS5_ENABLE_VIEWPORT_ARRAY_CANDIDATE ? PS5_MAX_VIEWPORTS : 1;
   caps->max_varyings = 16;
   caps->max_shader_patch_varyings =
      PS5_ENABLE_TESSELLATION_CANDIDATE ? 30 : 0;
   caps->max_stream_output_buffers =
      PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE ? PIPE_MAX_SO_BUFFERS : 0;
   caps->max_stream_output_separate_components =
      PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE ? 4 : 0;
   caps->max_stream_output_interleaved_components =
      PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE ? 64 : 0;
   caps->max_vertex_streams =
      PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE ? PIPE_MAX_VERTEX_STREAMS : 0;
   caps->max_geometry_output_vertices =
      PS5_ENABLE_GEOMETRY_CANDIDATE ? 256 : 0;
   caps->max_geometry_total_output_components =
      PS5_ENABLE_GEOMETRY_CANDIDATE ? 1024 : 0;
   caps->max_gs_invocations = PS5_ENABLE_GEOMETRY_CANDIDATE ? 32 : 0;
   caps->max_vertex_buffers = 16;
   caps->max_vertex_attrib_stride = 2048;
   caps->max_vertex_element_src_offset = 2047;
   caps->supported_prim_modes = (1u << MESA_PRIM_POINTS) |
                                (1u << MESA_PRIM_LINES) |
                                (1u << MESA_PRIM_LINE_STRIP) |
                                (1u << MESA_PRIM_TRIANGLES) |
                                 (1u << MESA_PRIM_TRIANGLE_FAN) |
                                 (1u << MESA_PRIM_TRIANGLE_STRIP) |
                                 (1u << MESA_PRIM_QUADS) |
                                 (1u << MESA_PRIM_QUAD_STRIP) |
                                 (1u << MESA_PRIM_POLYGON) |
                                (1u << MESA_PRIM_LINES_ADJACENCY) |
                                (1u << MESA_PRIM_LINE_STRIP_ADJACENCY) |
                                (1u << MESA_PRIM_TRIANGLES_ADJACENCY) |
                                (1u << MESA_PRIM_TRIANGLE_STRIP_ADJACENCY);
   if (PS5_ENABLE_TESSELLATION_CANDIDATE)
      caps->supported_prim_modes |= 1u << MESA_PRIM_PATCHES;
   caps->primitive_restart = true;
   caps->supported_prim_modes_with_restart = caps->supported_prim_modes;
   caps->quads_follow_provoking_vertex_convention = true;
   caps->vertex_color_unclamped = true;
   caps->vs_instanceid = true;
   caps->start_instance = PS5_ENABLE_GLSL_420_CANDIDATE;
   caps->vertex_element_instance_divisor = true;
   /* The compiler lowers narrow fetches with their actual element alignment. */
   caps->vertex_input_alignment = PIPE_VERTEX_INPUT_ALIGNMENT_ELEMENT;
   caps->texture_swizzle = PS5_ENABLE_TEXTURE_SWIZZLE_CANDIDATE;
   caps->fragment_shader_texture_lod = PS5_ENABLE_SHADER_TEXTURE_LOD_CANDIDATE;
   caps->max_texture_lod_bias =
      PS5_ENABLE_TEXTURE_MIPMAP_CANDIDATE ? 16.0f : 0.0f;
   caps->generate_mipmap = true;
   caps->max_constant_buffer_size =
      PS5_ENABLE_UBO_CANDIDATE ? PS5_MAX_CONSTANT_BUFFER_SIZE : 0;
   caps->constant_buffer_offset_alignment =
      PS5_ENABLE_UBO_CANDIDATE ? 16 : 0;
   caps->query_time_elapsed = PS5_ENABLE_TIMER_QUERY_CANDIDATE;
   caps->query_timestamp = PS5_ENABLE_TIMER_QUERY_CANDIDATE;
   caps->query_timestamp_bits = PS5_ENABLE_TIMER_QUERY_CANDIDATE ? 64 : 0;
   caps->occlusion_query = PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE;
   caps->conditional_render = PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE;
   caps->conditional_render_inverted =
      PS5_ENABLE_OCCLUSION_QUERY_CANDIDATE;
   caps->draw_indirect = PS5_ENABLE_DRAW_INDIRECT_CANDIDATE;
   caps->texture_query_lod = PS5_ENABLE_SHADER_TEXTURE_LOD_CANDIDATE;
   caps->stream_output_pause_resume =
      PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE;
   caps->stream_output_interleave_buffers =
      PS5_ENABLE_TRANSFORM_FEEDBACK_CANDIDATE;
   caps->depth_clip_disable = PS5_ENABLE_DEPTH_CLAMP_CANDIDATE;
   caps->texture_multisample = PS5_ENABLE_MSAA4_CANDIDATE;
   caps->sample_shading = PS5_ENABLE_MSAA4_CANDIDATE;
   caps->texture_buffer_objects = PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE;
   caps->max_texel_buffer_elements = PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE
                                        ? PS5_MAX_TEXEL_BUFFER_ELEMENTS : 0;
   caps->texture_buffer_offset_alignment = PS5_ENABLE_TEXTURE_BUFFER_CANDIDATE
                                              ? 16 : 0;
   caps->fake_sw_msaa = !PS5_ENABLE_MSAA4_CANDIDATE &&
                        PS5_ENABLE_FAKE_SW_MSAA_CANDIDATE;
   caps->seamless_cube_map = PS5_ENABLE_SEAMLESS_CUBE_CANDIDATE;
   caps->seamless_cube_map_per_texture =
      PS5_ENABLE_SEAMLESS_CUBE_CANDIDATE;
   caps->cube_map_array = PS5_ENABLE_TEXTURE_CUBE_ARRAY_CANDIDATE;
   caps->copy_between_compressed_and_plain_formats =
      PS5_ENABLE_GLSL_430_CANDIDATE;
   caps->framebuffer_no_attachment = PS5_ENABLE_GLSL_430_CANDIDATE;
   caps->robust_buffer_access_behavior = PS5_ENABLE_GLSL_430_CANDIDATE;
   caps->sampler_view_target = PS5_ENABLE_GLSL_430_CANDIDATE;
   caps->buffer_map_persistent_coherent = PS5_ENABLE_GLSL_440_CANDIDATE;
   caps->query_buffer_object = PS5_ENABLE_GLSL_440_CANDIDATE;
   caps->texture_mirror_clamp_to_edge = PS5_ENABLE_GLSL_440_CANDIDATE;
   caps->shader_array_components = PS5_ENABLE_GLSL_440_CANDIDATE;
   caps->clip_halfz = PS5_ENABLE_GLSL_450_CANDIDATE;
   caps->cull_distance = PS5_ENABLE_GLSL_450_CANDIDATE;
   caps->fs_fine_derivative = PS5_ENABLE_GLSL_450_CANDIDATE;
   caps->texture_query_samples = PS5_ENABLE_GLSL_450_CANDIDATE;
   caps->texture_barrier = PS5_ENABLE_GLSL_450_CANDIDATE;
   /* Non-coherent fetch of color buffer 0: KHR_blend_equation_advanced. */
   caps->fbfetch = PS5_ENABLE_GLSL_450_CANDIDATE;
   caps->gl_spirv = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->multi_draw_indirect = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->multi_draw_indirect_params = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->polygon_offset_clamp = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->draw_parameters = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->shader_ballot = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->shader_group_vote = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->anisotropic_filter = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->max_texture_anisotropy = PS5_ENABLE_GLSL_460_CANDIDATE ? 16.0f : 0.0f;
   caps->query_so_overflow = PS5_ENABLE_GLSL_460_CANDIDATE;
   caps->gl_begin_end_buffer_size = 512 * 1024;
   caps->min_map_buffer_alignment = 64;
   vs_caps = (struct pipe_shader_caps *)
      &screen->base.shader_caps[MESA_SHADER_VERTEX];
   tcs_caps = (struct pipe_shader_caps *)
      &screen->base.shader_caps[MESA_SHADER_TESS_CTRL];
   tes_caps = (struct pipe_shader_caps *)
      &screen->base.shader_caps[MESA_SHADER_TESS_EVAL];
   gs_caps = (struct pipe_shader_caps *)
      &screen->base.shader_caps[MESA_SHADER_GEOMETRY];
   fs_caps = (struct pipe_shader_caps *)
      &screen->base.shader_caps[MESA_SHADER_FRAGMENT];
   vs_caps->supported_irs = 1u << PIPE_SHADER_IR_NIR;
   fs_caps->supported_irs = 1u << PIPE_SHADER_IR_NIR;

   /*
    * This is the first deliberately small programmable-pipeline envelope.
    * Both stages accept NIR and have been compiled to loadable AGC packages
    * on the console.  The sampler counts describe compiler/hardware limits;
    * texture formats remain unadvertised until Gallium descriptors are wired.
    */
   vs_caps->max_instructions = 16384;
   vs_caps->max_alu_instructions = 16384;
   vs_caps->max_tex_instructions = 16384;
   vs_caps->max_tex_indirections = 16;
   vs_caps->max_control_flow_depth = 32;
   vs_caps->max_inputs = 16;
   vs_caps->max_outputs = 32;
   vs_caps->max_const_buffer0_size =
      PS5_MAX_DEFAULT_CONSTANT_BUFFER_SIZE;
   vs_caps->max_const_buffers =
      PS5_ENABLE_UBO_CANDIDATE ? PS5_MAX_CONSTANT_BUFFERS : 1;
   vs_caps->max_temps = 256;
   vs_caps->max_texture_samplers = 16;
   vs_caps->max_sampler_views = 16;
   vs_caps->cont_supported = true;
   vs_caps->indirect_temp_addr = true;
   vs_caps->indirect_const_addr = true;
   vs_caps->integers = true;
   if (PS5_ENABLE_GLSL_420_CANDIDATE)
      vs_caps->max_shader_images = PS5_COMPUTE_IMAGE_SLOTS;
   if (PS5_ENABLE_GLSL_430_CANDIDATE)
      vs_caps->max_shader_buffers = PS5_COMPUTE_STORAGE_SLOTS;

   if (PS5_ENABLE_GEOMETRY_CANDIDATE) {
      *gs_caps = *vs_caps;
      gs_caps->max_inputs = 16;
      gs_caps->max_outputs = 32;
      gs_caps->max_texture_samplers = PS5_MAX_TEXTURE_UNITS;
      gs_caps->max_sampler_views = PS5_MAX_TEXTURE_UNITS;
      gs_caps->max_shader_buffers = PS5_COMPUTE_STORAGE_SLOTS;
   }
   if (PS5_ENABLE_TESSELLATION_CANDIDATE) {
      *tcs_caps = *vs_caps;
      *tes_caps = *vs_caps;
      tcs_caps->max_inputs = tcs_caps->max_outputs = 32;
      tes_caps->max_inputs = tes_caps->max_outputs = 32;
      tcs_caps->max_texture_samplers = tcs_caps->max_sampler_views = PS5_MAX_TEXTURE_UNITS;
      tes_caps->max_texture_samplers = tes_caps->max_sampler_views = PS5_MAX_TEXTURE_UNITS;
      /* Mesa's UBO capability is all-stage: fewer than 15 here would also
       * remove UBOs from the already-qualified VS/FS core profile. */
      tcs_caps->max_const_buffers = tes_caps->max_const_buffers =
         PS5_MAX_CONSTANT_BUFFERS;
      if (PS5_ENABLE_GLSL_430_CANDIDATE)
         tcs_caps->max_shader_buffers = tes_caps->max_shader_buffers =
            PS5_COMPUTE_STORAGE_SLOTS;
   }

   fs_caps->max_instructions = 16384;
   fs_caps->max_alu_instructions = 16384;
   fs_caps->max_tex_instructions = 16384;
   fs_caps->max_tex_indirections = 16;
   fs_caps->max_control_flow_depth = 32;
   fs_caps->max_inputs = 32;
   fs_caps->max_outputs = 8;
   fs_caps->max_const_buffer0_size =
      PS5_MAX_DEFAULT_CONSTANT_BUFFER_SIZE;
   fs_caps->max_const_buffers =
      PS5_ENABLE_UBO_CANDIDATE ? PS5_MAX_CONSTANT_BUFFERS : 1;
   fs_caps->max_temps = 256;
   fs_caps->max_texture_samplers = 16;
   fs_caps->max_sampler_views = 16;
   fs_caps->cont_supported = true;
   fs_caps->indirect_temp_addr = true;
   fs_caps->indirect_const_addr = true;
   fs_caps->integers = true;
   if (PS5_ENABLE_GLSL_420_CANDIDATE) {
      fs_caps->max_shader_buffers = PS5_COMPUTE_STORAGE_SLOTS;
      fs_caps->max_shader_images = PS5_COMPUTE_IMAGE_SLOTS;
      caps->image_store_formatted = true;
      caps->shader_buffer_offset_alignment = 16;
      caps->max_shader_buffer_size = 1u << 27;
   }
#if PS5_ENABLE_COMPUTE_API_TEST || PS5_ENABLE_GLSL_430_CANDIDATE
   /* Private GL integration fixture only. Initialize before Mesa/CSO caches
    * capabilities; this is not a release/conformance feature advertisement. */
   struct pipe_shader_caps *cs_caps = (struct pipe_shader_caps *)
      &screen->base.shader_caps[MESA_SHADER_COMPUTE];
   struct pipe_compute_caps *compute_caps = (struct pipe_compute_caps *)
      &screen->base.compute_caps;
   *cs_caps = *fs_caps;
   cs_caps->max_const_buffers = PS5_COMPUTE_CONSTANT_SLOTS;
   cs_caps->max_shader_buffers = fs_caps->max_shader_buffers = PS5_COMPUTE_STORAGE_SLOTS;
   cs_caps->max_shader_images = fs_caps->max_shader_images = PS5_COMPUTE_IMAGE_SLOTS;
   caps->compute = true;
   caps->image_store_formatted = true;
   caps->shader_buffer_offset_alignment = 16;
   caps->max_shader_buffer_size = 1u << 27;
   *compute_caps = (struct pipe_compute_caps){
      .max_threads_per_block = 1024, .max_local_size = 32768,
      .max_grid_size = {65535, 65535, 65535},
      .max_block_size = {1024, 1024, 64},
   };
#endif
   vs_options = (struct nir_shader_compiler_options *)(uintptr_t)
      psbc_get_nir_options(PSBC_STAGE_VERTEX);
   tcs_options = (struct nir_shader_compiler_options *)(uintptr_t)
      psbc_get_nir_options(PSBC_STAGE_TESS_CTRL);
   tes_options = (struct nir_shader_compiler_options *)(uintptr_t)
      psbc_get_nir_options(PSBC_STAGE_TESS_EVAL);
   fs_options = (struct nir_shader_compiler_options *)(uintptr_t)
      psbc_get_nir_options(PSBC_STAGE_FRAGMENT);
   gs_options = (struct nir_shader_compiler_options *)(uintptr_t)
      psbc_get_nir_options(PSBC_STAGE_GEOMETRY);
   cs_options = (struct nir_shader_compiler_options *)(uintptr_t)
      psbc_get_nir_options(PSBC_STAGE_COMPUTE);
   vs_options->io_options |= nir_io_has_intrinsics;
   tcs_options->io_options |= nir_io_has_intrinsics;
   tes_options->io_options |= nir_io_has_intrinsics;
   gs_options->io_options |= nir_io_has_intrinsics;
   fs_options->io_options |= nir_io_has_intrinsics;
   cs_options->io_options |= nir_io_has_intrinsics;
   screen->base.nir_options[MESA_SHADER_VERTEX] = vs_options;
   screen->base.nir_options[MESA_SHADER_TESS_CTRL] = tcs_options;
   screen->base.nir_options[MESA_SHADER_TESS_EVAL] = tes_options;
   screen->base.nir_options[MESA_SHADER_GEOMETRY] = gs_options;
   screen->base.nir_options[MESA_SHADER_FRAGMENT] = fs_options;
   screen->base.nir_options[MESA_SHADER_COMPUTE] = cs_options;
   return &screen->base;
}
