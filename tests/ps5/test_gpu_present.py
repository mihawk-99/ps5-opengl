#!/usr/bin/env python3
# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compile the actual queued-presentation tail and completion checks with mocks."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / "src/platform/ps5_agc_native_runtime.c").read_text()
assert source.count("runtime_video_framebuffer_size >= framebuffer_size") == 2
start = source.index("int ps5_agc_gate2_batch_present(")
body = source[start:source.index("\n#endif\n\n#if defined(PS5_DRAW_PROFILE)", start)]
start = source.index("int ps5_agc_gate2_present(unsigned buffer_index, unsigned swap_interval)")
present = source[start:source.index("\n#endif", source.index("    return result;", start))]
screen_source = (root / "src/gallium/ps5/ps5_screen.c").read_text()
a = screen_source.index("void\nps5_screen_present_lock(")
present_lock = screen_source[a:screen_source.index("\n}\n", a) + 3]
lock_support = r'''
#define PS5_DEFERRED_DRAW_BATCH 1
struct pipe_screen { int unused; };
static int ps5_deferred_mutex, locked;
static void simple_mtx_lock(int *m) { assert(m == &ps5_deferred_mutex && !locked); locked=1; }
static void ps5_draw_batch_flush_locked(void) {
    assert(locked);
    if (runtime_batch_active) { runtime_batch_active=runtime_batch_count=0; ++runtime_pending_batches; }
}
static void ps5_draw_batch_retire_locked(int wait) { assert(locked && !wait); }
'''
code = r'''
#include <assert.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#define PS5_GPU_PRESENT_BATCH 1
#define PS5_MULTIDRAW_BATCH 1
#define PS5_PROFILE_MARK(i) ((void)0)
#define COMMAND_BYTES 0x4000u
#define FRAMEBUFFER_POOL_BYTES 128u
static size_t runtime_video_framebuffer_size;
static int runtime_gpu_present_deferred, runtime_gpu_present_is_cpu;
int ps5_agc_gate2_wait_present(void);
typedef struct { uint32_t *bottom, *top, *up, *down; uintptr_t callback; } agc_command_buffer_t;
static uint8_t memory[COMMAND_BYTES + 64];
static uint64_t completion;
static struct runtime_batch_entry { struct { void *words; uint32_t word_count; } submit;
    void *memory; size_t bytes; uint64_t *marker; uint32_t expected;
} runtime_batch_entries[1];
static unsigned runtime_batch_count, runtime_batch_active, runtime_batch_faulted,
                runtime_pending_batches;
static int runtime_video_registered, runtime_video_handle, runtime_gpu_present_buffer;
static uint64_t runtime_gpu_present_marker;
static unsigned runtime_gpu_present_count, runtime_present_count, out_of_space;
static unsigned tails, releases, flushes, cpu_flips, waits, idle_calls;
static int idle_error, flip_error, status_error, pending_error, wait_error, tail_error;
static unsigned finish_after, pending_after, polls;
static int64_t runtime_next_render_marker(void) { return 101; }
static uint8_t command_out_of_space(agc_command_buffer_t *c, uint32_t n, void *p) {
    (void)c; (void)n; (void)p; out_of_space = 1; return 0;
}
static void flush_gpu_data(const void *p, size_t bytes) {
    assert(p == memory && bytes == 12 * sizeof(uint32_t)); ++flushes;
}
static uint32_t *set_flip(void *p, uint32_t handle, int index, uint32_t mode, int64_t marker) {
    agc_command_buffer_t *c = p;
    assert(handle == 7 && index == 1 && mode == 1 && marker == 101);
    assert(c->up == c->bottom + 4 && c->top - c->bottom == COMMAND_BYTES / 4);
    assert(completion == 17 && runtime_batch_entries[0].expected == 17);
    ++tails; c->up += 4;
    return tail_error == 1 ? NULL : c->up;
}
static uint32_t *release_mem(void *p, uint8_t event, int16_t cache, uint64_t gcr,
    int8_t dst, void *address, uint32_t selection, uint64_t marker,
    uint16_t x, uint16_t y, int8_t z, int32_t w) {
    agc_command_buffer_t *c = p;
    assert(tails == 1 && c->up == c->bottom + 8 && event == 40 && cache == 0x30c);
    assert(!gcr && !dst && address == &completion && selection == 1 && marker == 101);
    assert(!x && !y && !z && !w);
    ++releases; c->up += 4;
    if (tail_error == 3) out_of_space = 1;
    return tail_error == 2 ? NULL : c->up;
}
static struct {
    uint32_t *(*set_flip)(void *, uint32_t, int, uint32_t, int64_t);
    uint32_t *(*release_mem)(void *, uint8_t, int16_t, uint64_t, int8_t, void *, uint32_t,
        uint64_t, uint16_t, uint16_t, int8_t, int32_t);
} runtime_batch_api = {set_flip, release_mem};
/* Packet encoding/failure propagation is covered by test_submit_retirement.
 * This mock checks presentation routes its final ownership marker through it. */
static uint32_t *runtime_release_completion(void *api, agc_command_buffer_t *command,
                                            volatile uint64_t *marker, uint32_t value) {
    assert(api == &runtime_batch_api);
    return release_mem(command,40,0x30c,0,0,(void *)marker,1,value,0,0,0,0);
}

static int status(int handle, void *p) {
    assert(handle == 7); ((uint64_t *)p)[3] = waits >= finish_after ? 101 : 99;
    return status_error;
}
static int pending(int handle) { assert(handle == 7); return pending_error ? pending_error : (waits < finish_after || waits < pending_after); }
static int vblank(int handle) { assert(handle == 7); ++waits; return wait_error; }
static int sceKernelUsleep(uint32_t us) { assert(us == 1000); ++waits; ++polls; return wait_error; }
static int cpu_flip(int handle, int index, uint32_t mode, int64_t marker) {
    assert(handle == 7 && index == 1 && mode == 1 && marker == 101); ++cpu_flips; return flip_error;
}
static struct {
    int (*get_flip_status)(int, void *);
    int (*is_flip_pending)(int);
    int (*wait_vblank)(int);
    int (*submit_flip)(int, int, uint32_t, int64_t);
} runtime_video_api = {status, pending, vblank, cpu_flip};
static int runtime_video_wait_idle(void) { ++idle_calls; return idle_error; }
''' + body + present + lock_support + present_lock + r'''
static void reset(void) {
    runtime_video_registered = 1; runtime_video_handle = 7;
    runtime_pending_batches = 0;
    runtime_gpu_present_buffer = -1; runtime_gpu_present_marker = 0;
    runtime_batch_active = runtime_batch_count = 1; runtime_batch_faulted = 0;
    runtime_batch_entries[0].submit.words = memory;
    runtime_batch_entries[0].submit.word_count = 4;
    runtime_batch_entries[0].memory = memory;
    runtime_batch_entries[0].bytes = sizeof(memory);
    runtime_batch_entries[0].marker = &completion;
    runtime_batch_entries[0].expected = completion = 17;
    tails = releases = flushes = cpu_flips = waits = idle_calls = 0;
    runtime_present_count = runtime_gpu_present_count = 0;
    idle_error = flip_error = status_error = pending_error = wait_error = tail_error = 0;
    finish_after = pending_after = polls = 0;
    runtime_video_framebuffer_size = runtime_gpu_present_deferred = runtime_gpu_present_is_cpu = 0;
}
int main(void) {
    reset(); /* A second context queued after the first context flushed. */
    ps5_screen_present_lock(NULL);
    assert(locked && !runtime_batch_active && !runtime_batch_count && runtime_pending_batches==1);
    assert(ps5_agc_gate2_present(1,0)==0);
    locked=0;
    reset(); assert(ps5_agc_gate2_batch_present(1) == 0);
    runtime_batch_count = runtime_batch_active = 0; runtime_pending_batches = 1;
    assert(ps5_agc_gate2_present(1, 0) == 0 && !polls && !cpu_flips);
    assert(runtime_gpu_present_deferred && runtime_gpu_present_buffer == 1);
    reset(); runtime_batch_count = runtime_batch_active = 0; runtime_pending_batches = 1;
    assert(ps5_agc_gate2_present(1, 0) == 0 && cpu_flips == 1 && !waits);
    for (unsigned n = 0; n <= 2001; ++n) {
        reset(); assert(ps5_agc_gate2_batch_present(1) == 0);
        assert(tails == 1 && releases == 1 && flushes == 1);
        assert(runtime_batch_entries[0].expected == 101 && completion == 17);
        assert(ps5_agc_gate2_present(1, 1) != 0); /* Unretired command tail. */
        completion = 101; runtime_batch_count = runtime_batch_active = 0;
        finish_after = n;
        assert((ps5_agc_gate2_present(1, 1) == 0) == (n <= 2000));
        assert(!cpu_flips && waits == (n <= 2000 ? n : 2000));
        assert(polls == waits && idle_calls == 1);
        assert(runtime_present_count == (n <= 2000) && runtime_gpu_present_count == (n <= 2000));
        assert(runtime_gpu_present_buffer == (n <= 2000 ? -1 : 1));
    }
    /* Marker publication and queue idle can become visible separately. */
    reset(); assert(ps5_agc_gate2_batch_present(1) == 0);
    runtime_batch_count = runtime_batch_active = 0;
    finish_after = 2; pending_after = 7;
    assert(ps5_agc_gate2_present(1, 1) == 0 && polls == 7 && waits == 7);
    assert(idle_calls == 1 && runtime_gpu_present_buffer == -1);
    reset(); runtime_video_framebuffer_size = FRAMEBUFFER_POOL_BYTES;
    assert(ps5_agc_gate2_batch_present(1) == 0);
    runtime_batch_count = runtime_batch_active = 0; finish_after = 7;
    assert(ps5_agc_gate2_present(1, 1) == 0 && !polls);
    assert(runtime_gpu_present_deferred && runtime_gpu_present_buffer == 1);
    assert(!runtime_gpu_present_count && runtime_present_count == 1);
    assert(ps5_agc_gate2_wait_present() == 0 && polls == 7);
    assert(!runtime_gpu_present_deferred && runtime_gpu_present_buffer == -1);
    assert(runtime_gpu_present_count == 1 && ps5_agc_gate2_wait_present() == 0 && polls == 7);
    /* Empty next swaps retire the old flip before issuing a new CPU flip. */
    reset(); runtime_video_framebuffer_size = FRAMEBUFFER_POOL_BYTES;
    assert(ps5_agc_gate2_batch_present(1) == 0);
    runtime_batch_count = runtime_batch_active = 0; finish_after = 3;
    assert(ps5_agc_gate2_present(1, 1) == 0 && !polls);
    assert(ps5_agc_gate2_present(1, 1) == 0 && polls == 3 && cpu_flips == 1);
    assert(runtime_gpu_present_is_cpu && runtime_gpu_present_deferred);
    finish_after = waits + 3;
    assert(ps5_agc_gate2_wait_present() == 0 && polls == 6);
    assert(runtime_gpu_present_count == 1 && !runtime_gpu_present_is_cpu);
    /* CPU-submitted flips after GL flush use the same deferred ownership. */
    reset(); runtime_video_framebuffer_size = FRAMEBUFFER_POOL_BYTES;
    runtime_batch_count = runtime_batch_active = 0; finish_after = 5;
    assert(ps5_agc_gate2_present(1, 1) == 0 && cpu_flips == 1 && !waits);
    assert(runtime_gpu_present_deferred && runtime_gpu_present_is_cpu);
    assert(ps5_agc_gate2_wait_present() == 0 && polls == 5);
    assert(!runtime_gpu_present_count && !runtime_gpu_present_deferred);
    reset(); runtime_video_framebuffer_size = FRAMEBUFFER_POOL_BYTES;
    runtime_batch_count = runtime_batch_active = 0; flip_error = -8;
    assert(ps5_agc_gate2_present(1, 1) == -8 && !runtime_gpu_present_deferred && runtime_gpu_present_buffer == -1);
    /* A failed deferred wait retains ownership and cannot queue another flip. */
    reset(); runtime_video_framebuffer_size = FRAMEBUFFER_POOL_BYTES;
    assert(ps5_agc_gate2_batch_present(1) == 0);
    runtime_batch_count = runtime_batch_active = 0;
    assert(ps5_agc_gate2_present(1, 1) == 0); status_error = -21;
    assert(ps5_agc_gate2_wait_present() == -21 && runtime_gpu_present_deferred);
    assert(ps5_agc_gate2_present(1, 1) != 0 && !cpu_flips && runtime_gpu_present_buffer == 1);
    for (int error = 0; error < 10; ++error) {
        reset();
        if (error == 0) runtime_batch_entries[0].submit.word_count = COMMAND_BYTES / 4;
        if (error == 1) runtime_batch_entries[0].bytes = COMMAND_BYTES - 1;
        if (error == 2) runtime_gpu_present_buffer = 0;
        if (error == 3) idle_error = -1;
        if (error == 4) runtime_batch_faulted = 1;
        if (error == 5) runtime_video_registered = 0;
        if (error == 6) runtime_batch_count = 0;
        if (error >= 7) tail_error = error - 6;
        assert(ps5_agc_gate2_batch_present(1) != 0 && !flushes);
        assert(runtime_batch_entries[0].expected == 17);
    }
    for (int error = 0; error < 4; ++error) {
        reset(); assert(ps5_agc_gate2_batch_present(1) == 0);
        runtime_batch_count = runtime_batch_active = 0;
        if (error == 0) status_error = -21;
        if (error == 1) pending_error = -22;
        if (error == 2) { wait_error = -23; finish_after = 1; }
        assert(ps5_agc_gate2_present(error == 3 ? 0 : 1, 1) != 0);
        assert(runtime_gpu_present_buffer == 1 && !cpu_flips && !runtime_present_count);
    }
    reset(); runtime_batch_count = runtime_batch_active = 0; finish_after = 3;
    assert(ps5_agc_gate2_present(1, 0) == 0 && cpu_flips == 1 && !waits);
    assert(runtime_gpu_present_deferred && runtime_gpu_present_buffer == 1);
    assert(ps5_agc_gate2_wait_present() == 0 && polls == 3);
    reset(); runtime_batch_count = runtime_batch_active = 0;
    assert(ps5_agc_gate2_present(1, 2) == -1 && !cpu_flips && !waits);
    reset(); runtime_batch_count = runtime_batch_active = 0;
    assert(ps5_agc_gate2_present(1, 1) == 0 && cpu_flips == 1 && waits == 1);
    assert(!runtime_gpu_present_count); /* Empty/readback-drained batch uses original path. */
}
'''
with tempfile.TemporaryDirectory() as tmp:
    c, exe = Path(tmp) / "present.c", Path(tmp) / "present"
    c.write_text(code)
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(c), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    broken = code.replace("   ps5_draw_batch_flush_locked();", "   if (0) ps5_draw_batch_flush_locked();")
    assert broken != code
    c.write_text(broken)
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(c), "-o", str(exe)], check=True)
    assert subprocess.run([str(exe)], cwd=tmp, capture_output=True).returncode != 0
egl = (root / "src/egl/ps5_egl.c").read_text()
assert "&fence, surface->window ? ps5_before_swap_flush : NULL, surface)" in egl
assert "ps5_context_queue_present(ps5_current_context->st->pipe, surface->buffer_index)" in egl
assert egl.index("ps5_screen_prepare_present(ps5_display.screen)") < \
       egl.index("ps5_agc_gate2_present(surface->buffer_index,")
make = (root / "toolchain/ps5-opengl-core33.mk").read_text()
egl_rule = make[make.index("$(PS5_OPENGL_BUILD)/ps5_egl.o:"):make.index("$(PS5_OPENGL_BUILD)/ps5_screen.o:")]
assert "$(PS5_OPENGL_RUNTIME_DEFINES)" in egl_rule
assert "if (runtime_gpu_present_buffer >= 0)\n        return -1; /* Unconfirmed presentation" in source
print("PASS: bounded GPU-flip tail, post-tail marker, exact flip/idle completion, failures, CPU fallback")

# Compile the real first-swap registration gate. A clear-only frame must not
# depend on an application draw to open/register the native scanout pool.
start = source.index("int ps5_agc_gate2_prepare_present(")
prepare = source[start:source.index("\nstatic int runtime_video_wait_idle", start)]
a = screen_source.index("int\nps5_screen_prepare_present(")
prepare_screen = screen_source[a:screen_source.index("\n}\n", a) + 3]
code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define FRAMEBUFFER_BYTES 64u
#define FRAMEBUFFER_ALIGNMENT 16u
typedef struct { int value; } agc_api_t;
typedef struct { int value; } video_api_t;
static int runtime_video_registered;
static uint8_t *runtime_video_framebuffer;
static size_t runtime_video_framebuffer_size;
static int loads, flushes, acquires, load_result, acquire_result;
static int load_apis(void *a, void *b, void *c, agc_api_t *agc, video_api_t *video) {
    assert(!a && !b && !c && agc && video); ++loads; return load_result;
}
static void flush_gpu_data(const void *p, size_t n) {
    assert(p && n >= FRAMEBUFFER_BYTES); ++flushes;
}
static int runtime_video_acquire(const video_api_t *video, uint8_t *framebuffer,
                                 size_t size, int *attempts) {
    assert(video && framebuffer && size >= FRAMEBUFFER_BYTES && attempts);
    ++acquires; *attempts = 1; return acquire_result;
}
''' + prepare + r'''
#define PS5_SCANOUT_POOL_BYTES 128u
static int (*prepare_hook)(void *, size_t) = ps5_agc_gate2_prepare_present;
#define ps5_agc_gate2_prepare_present prepare_hook
struct pipe_screen { int unused; };
struct ps5_resource { void *data; size_t allocation_size; };
struct ps5_screen { struct pipe_screen base; struct ps5_resource *render_pool; };
''' + prepare_screen + r'''
int main(void) {
    _Alignas(FRAMEBUFFER_ALIGNMENT) uint8_t pool[128] = {0};
    assert(ps5_agc_gate2_prepare_present(NULL, sizeof(pool)) == -1);
    assert(ps5_agc_gate2_prepare_present(pool, FRAMEBUFFER_BYTES - 1) == -1);
    assert(ps5_agc_gate2_prepare_present(pool + 1, sizeof(pool) - 1) == -1);
    assert(!loads && !flushes && !acquires);

    runtime_video_registered = 1;
    runtime_video_framebuffer = pool;
    runtime_video_framebuffer_size = sizeof(pool);
    assert(ps5_agc_gate2_prepare_present(pool, sizeof(pool)) == 0);
    assert(ps5_agc_gate2_prepare_present(pool, FRAMEBUFFER_BYTES) == 0);
    assert(!loads && !flushes && !acquires);

    /* A large offscreen arena must not reopen an already registered scanout. */
    struct ps5_resource resource = {pool, sizeof(pool) + 4096};
    struct ps5_screen screen = {{0}, &resource};
    assert(ps5_screen_prepare_present(&screen.base) == 0);
    assert(!loads && !flushes && !acquires);
    resource.allocation_size = PS5_SCANOUT_POOL_BYTES - 1;
    assert(ps5_screen_prepare_present(&screen.base) == -1);
    assert(ps5_screen_prepare_present(NULL) == -1);
    assert(!loads && !flushes && !acquires);

    runtime_video_registered = 0;
    load_result = -7;
    assert(ps5_agc_gate2_prepare_present(pool, sizeof(pool)) == -1);
    assert(loads == 1 && !flushes && !acquires);

    load_result = 0;
    acquire_result = -9;
    assert(ps5_agc_gate2_prepare_present(pool, sizeof(pool)) == -9);
    assert(loads == 2 && flushes == 1 && acquires == 1);
    acquire_result = 0;
    assert(ps5_agc_gate2_prepare_present(pool, sizeof(pool)) == 0);
    assert(loads == 3 && flushes == 2 && acquires == 2);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    c, exe = Path(tmp) / "prepare-present.c", Path(tmp) / "prepare-present"
    c.write_text(code)
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                    str(c), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print("PASS: first clear-only swap registers scanout; reuse and failures are bounded")

# Compile the actual scanout-flush gate, including the unchanged default path.
start = source.rindex("    PS5_PROFILE_MARK(1);", 0, source.index("    /* The first queued draw"))
flush_gate = source[start:source.index("    PS5_PROFILE_MARK(2);", start)]
code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#define PS5_PROFILE_MARK(i) ((void)0)
static int flushes;
static int writes_scanout;
static int runtime_scanout_flush_needed = 1;
#ifdef TEST_SCANOUT_CLASSIFICATION
#define PS5_RUNTIME_WRITES_SCANOUT() writes_scanout
#endif
static void flush_gpu_data(const void *p, size_t n) { assert(p && n); ++flushes; }
static void run(int runtime_batch_active, unsigned runtime_batch_count,
                int runtime_video_registered, void *framebuffer, size_t framebuffer_pool_bytes,
                void *runtime_video_framebuffer, size_t runtime_video_framebuffer_size) {
    (void)runtime_batch_active; (void)runtime_batch_count; (void)runtime_video_registered;
    (void)runtime_video_framebuffer; (void)runtime_video_framebuffer_size;
''' + flush_gate + r'''
}
int main(void) {
    char pools[2];
    for (unsigned state = 0; state < 128; ++state) {
        writes_scanout = (state & 64) != 0;
        int active = state & 1, queued = state & 2, registered = state & 4;
        int same_pointer = state & 8, same_size = state & 16;
        int larger_registration = state & 32; /* The application registers its entire render arena. */
        flushes = 0;
        runtime_scanout_flush_needed = 1; /* Each state starts on a new pool. */
        run(active, queued ? 2 : 0, registered, pools, 64,
            pools + !same_pointer, larger_registration ? 128 : same_size ? 64 : 32);
#ifdef PS5_GPU_PRESENT_BATCH
        int reusable = active;
#ifdef TEST_SCANOUT_CLASSIFICATION
        reusable |= !writes_scanout;
#endif
        assert(flushes == !(reusable && registered && same_pointer &&
                           (same_size || larger_registration)));
#else
        assert(flushes == 1);
#endif
    }
    /* Queue splits do not dirty a GPU-written pool; CPU writers flush their own ranges. */
    flushes = 0;
    run(1, 0, 1, pools, 64, pools, 64);
    run(1, 1, 1, pools, 64, pools, 64);
    run(1, 2, 1, pools, 64, pools, 64);
    run(1, 0, 1, pools, 64, pools, 64);
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == 0);
#else
    assert(flushes == 4);
#endif
    flush_gpu_data(pools, 64); /* CPU writer owns publication. */
    int published = flushes;
    run(1, 0, 1, pools, 64, pools, 64);
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == published);
#else
    assert(flushes == published + 1);
#endif
    /* Unbatched GPU writes to a registered pool (each window glClear) flush it once,
     * until a new pool or registration; an unregistered pool is flushed every time. */
    writes_scanout = 1;
    runtime_scanout_flush_needed = 1;
    flushes = 0;
    run(0, 0, 1, pools, 64, pools, 64);
    run(0, 0, 1, pools, 64, pools, 64);
    run(0, 0, 1, pools, 64, pools, 64);
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == 1);
#else
    assert(flushes == 3);
#endif
    flushes = 0;
    runtime_scanout_flush_needed = 1;
    run(0, 0, 1, pools, 64, pools, 64);
    run(0, 0, 0, pools, 64, pools, 64);
    run(0, 0, 0, pools, 64, pools, 64);
    assert(flushes == 3);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    c, exe = Path(tmp) / "flush.c", Path(tmp) / "flush"
    c.write_text(code)
    for flags in ([], ["-DPS5_GPU_PRESENT_BATCH=1"],
                  ["-DPS5_GPU_PRESENT_BATCH=1", "-DTEST_SCANOUT_CLASSIFICATION=1"]):
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", *flags,
                        str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
# Every new pool and every registration or release of the scanout pool asks for its
# one whole-pool flush again.
assert len(re.findall(r"(?m)^\s+runtime_scanout_flush_needed = 1;", source)) == 3, \
    "pool changes must re-arm the flush"
print("PASS: registered GPU pool survives queue splits and repeated unbatched writes; "
      "CPU publication, replacement and default paths retained")

screen = (root / "src/gallium/ps5/ps5_screen.c").read_text()
start = screen.index("struct ps5_batch_flush_cache {")
flush_cache = screen[start:screen.index("\nstatic bool\nps5_stage_packed_depth_samples", start)]
start = screen.index("#ifdef PS5_GPU_PRESENT_BATCH\n      /* Disabled depth AND stencil")
policy = screen[start:screen.index("#endif", start) + len("#endif")] + "\n"
start = screen.index("      if (flush_depth_stencil)")
depth_flush = screen[start:screen.index("      if (packed)", start)]
start = screen.index("         if (flush_depth_stencil)")
stencil_flush = screen[start:screen.index(';', start) + 1] + "\n"
code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define PS5_MAX_TEXTURE_UNITS 16
#define PIPE_MAX_ATTRIBS 16
static unsigned flushes;
static void ps5_flush_gpu_data(const void *p, size_t n) { assert(p && n); ++flushes; }
#define PIPE_BUFFER 0
#define PIPE_BIND_DISPLAY_TARGET 8
static uint64_t ps5_texture_publication_epoch=1;
struct ps5_resource {
    struct { unsigned target,bind; } base;
    void *data,*stencil_data; size_t stencil_allocation_size,depth_staging_size,texture_published_bytes,stencil_published_bytes;
    uint64_t texture_publication_epoch,stencil_publication_epoch; bool external_cpu_access;
};
''' + flush_cache + r'''
static void run(uint32_t control, struct ps5_batch_flush_cache *flush_cache,
                char *backing, size_t bytes) {
    struct { uint32_t depth_control; } native = {control};
    (void)native;
    void *depth_data = backing;
    size_t depth_allocation = bytes;
    struct ps5_resource buffer = {.base={.target=2},.data=backing,.stencil_data=backing+1,.stencil_allocation_size=8}, *depth=&buffer;
''' + policy + depth_flush + "\n{\n" + stencil_flush + "}\n" + r'''
}
int main(void) {
    char backing[3] = {0};
    for (unsigned control = 0; control < 256; ++control) {
        flushes = 0; run(control, NULL, backing, 32);
#ifdef PS5_GPU_PRESENT_BATCH
        assert(flushes == (control ? 2u : 0u));
#else
        assert(flushes == 2);
#endif
    }
    /* Disable, CPU update, re-enable: active draw still flushes both buffers. */
    flushes = 0;
    run(0, NULL, backing, 32); run(2, NULL, backing, 32);
    run(1, NULL, backing, 32); run(3, NULL, backing, 32);
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == 6);
#else
    assert(flushes == 8);
#endif
    /* Disabled first draw must not warm the cache. Both planes remain distinct. */
    struct ps5_batch_flush_cache cache = {0};
    flushes = 0;
    run(0, &cache, backing, 32);
    run(2, &cache, backing, 32);
    run(2, &cache, backing, 32);
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == 2);
#else
    assert(flushes == 6);
#endif
    /* Changed size/pointer must flush again; never reuse it across a drain. */
    flushes = 0;
    run(2, &cache, backing, 64); /* Depth size only: stencil backing unchanged. */
    run(2, &cache, backing + 1, 64); /* Both pointers change. */
    cache = (struct ps5_batch_flush_cache){0};
    run(2, &cache, backing + 1, 64); /* New batch / CPU-write boundary. */
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == 5);
#else
    assert(flushes == 6);
#endif
    /* Nonbatched access cannot consume or populate a batch cache. */
    flushes = 0;
    run(2, NULL, backing + 1, 64);
    run(2, NULL, backing + 1, 64);
    assert(flushes == 4);
    flushes = 0;
    ps5_flush_batch_backing(&cache, 1, backing + 2, 16);
    ps5_flush_batch_backing(&cache, 1, backing + 2, 16);
#ifdef PS5_GPU_PRESENT_BATCH
    assert(flushes == 1); /* Stencil size changes independently of depth. */
#else
    assert(flushes == 2);
#endif
}
'''
with tempfile.TemporaryDirectory() as tmp:
    c, exe = Path(tmp) / "depth_flush.c", Path(tmp) / "depth_flush"
    c.write_text(code)
    for flags in ([], ["-DPS5_GPU_PRESENT_BATCH=1"]):
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", *flags,
                        str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
print("PASS: depth/stencil flushes at first enabled use, backing/size changes, CPU/drain boundaries; nonbatched/default unchanged")

# A registered pool can contain offscreen arena allocations after two scanouts.
backend = (root / "src/platform/ps5_agc_runtime_backend.c").read_text()
a = backend.index("static bool\nps5_agc_writes_scanout(")
predicate = backend[a:backend.index("#define PS5_RUNTIME_WRITES_SCANOUT", a)]
a = source.index("static int runtime_video_prepare_draw(void)")
prepare_draw = source[a:source.index("\nint ps5_agc_gate2_present", a)]
code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#define PS5_GPU_PRESENT_BATCH 1
#define PS5_AGC_FRAMEBUFFER_POOL_BYTES 128u
static void *ps5_agc_scanout_target;
static void *ps5_agc_mrt_targets[8];
static unsigned ps5_agc_mrt_mask, ps5_agc_mrt_count;
static int runtime_video_registered = 1, wait_error;
static unsigned waits;
static int ps5_agc_gate2_wait_present(void) { ++waits; return wait_error; }
''' + predicate + r'''
#define PS5_RUNTIME_WRITES_SCANOUT() ps5_agc_writes_scanout()
''' + prepare_draw + r'''
int main(void) {
 assert(ps5_agc_writes_scanout());
 ps5_agc_scanout_target = (void *)0x10000;
 ps5_agc_mrt_count = 1; ps5_agc_mrt_mask = 15;
 assert(ps5_agc_writes_scanout()); /* Unknown target is conservative. */
 for(unsigned i=0;i<8;++i) ps5_agc_mrt_targets[i]=(void *)0x20000;
 ps5_agc_mrt_count=8; ps5_agc_mrt_mask=UINT32_MAX;
 assert(!ps5_agc_writes_scanout());
 assert(runtime_video_prepare_draw()==0 && !waits);
 for(unsigned slot=0;slot<8;++slot) {
  for(unsigned offset=0;offset<128;offset+=16) {
   ps5_agc_mrt_targets[slot]=(void *)(uintptr_t)(0x10000+offset);
   assert(ps5_agc_writes_scanout());
  }
  ps5_agc_mrt_targets[slot]=(void *)0x10080;
  assert(!ps5_agc_writes_scanout());
  ps5_agc_mrt_targets[slot]=(void *)0x20000;
 }
 ps5_agc_mrt_targets[0]=(void *)0x10000;
 assert(runtime_video_prepare_draw()==0 && waits==1);
 wait_error=-9; assert(runtime_video_prepare_draw()==-1 && waits==2);
 ps5_agc_mrt_mask=0;
 assert(runtime_video_prepare_draw()==0 && waits==2);
 runtime_video_registered=0; assert(runtime_video_prepare_draw()==-1);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    exe = Path(tmp) / 'scanout-reuse'
    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-x','c','-','-o',str(exe)], input=code,text=True,check=True)
    subprocess.run([str(exe)],check=True)
print('PASS: offscreen overlap, both scanout aliases, every MRT slot, masks, unknown targets and wait failure')
