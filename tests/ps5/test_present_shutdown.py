#!/usr/bin/env python3
# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Inject presenter shutdown failures into the actual runtime/EGL destructors."""
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
runtime = (root / "src/platform/ps5_agc_native_runtime.c").read_text()
egl = (root / "src/egl/ps5_egl.c").read_text()
shutdown = runtime[runtime.index("int ps5_agc_gate2_shutdown_present(void)"):
                   runtime.index("static int runtime_video_acquire(")]
acquire_wait = (
    runtime[runtime.index("static int runtime_video_acquire("):
            runtime.index("int ps5_agc_gate2_prepare_present(")] +
    runtime[runtime.index("static int runtime_video_wait_idle(void)\n{"):
            runtime.index("static int runtime_video_prepare_draw(")]
)
video_api = runtime[runtime.index("typedef struct video_api {"):
                    runtime.index("} video_api_t;") + len("} video_api_t;")]
hfr = runtime[runtime.index("#if PS5_SCANOUT_FPS > 60\n#ifndef PS5_NATIVE_TITLE_RUNTIME"):
              runtime.index("int ps5_agc_gate2_shutdown_present(void)")]
terminate = egl[egl.index("EGLAPI EGLBoolean EGLAPIENTRY\neglTerminate("):
                egl.index("EGLAPI EGLint EGLAPIENTRY\neglGetError(")]
destroy = egl[egl.index("EGLAPI EGLBoolean EGLAPIENTRY\neglDestroySurface("):
              egl.index("EGLAPI EGLDisplay EGLAPIENTRY\neglGetCurrentDisplay(")]
code = r'''
#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define PS5_MULTIDRAW_BATCH 1
#define RENDER_MARKER 100
static int runtime_batch_faulted, runtime_batch_active;
static unsigned runtime_batch_count, runtime_pending_batches;
#ifdef PS5_NATIVE_TITLE_RUNTIME
static void runtime_require_retirement(int ready) { assert(ready); }
static int runtime_work_cache_clear(void) {
    assert(!runtime_batch_faulted && !runtime_batch_active && !runtime_batch_count && !runtime_pending_batches);
    return 0;
}
static int runtime_shader_pair_clear(void) {
    assert(!runtime_batch_faulted && !runtime_batch_active && !runtime_batch_count && !runtime_pending_batches);
    return 0;
}
static int runtime_compute_program_clear(void) {
    assert(!runtime_batch_faulted && !runtime_batch_active && !runtime_batch_count && !runtime_pending_batches);
    return 0;
}
#endif
#ifdef PS5_GPU_PRESENT_BATCH
static int runtime_gpu_present_buffer = -1;
static int deferred_flip, deferred_error;
static int ps5_agc_gate2_wait_present(void) {
    if (deferred_flip && deferred_error) return deferred_error;
    if (deferred_flip) { runtime_gpu_present_buffer = -1; deferred_flip = 0; }
    return 0;
}
static unsigned runtime_gpu_present_count;
#endif
typedef struct { uint64_t words[8]; } video_attribute_t;
typedef struct { void *a, *b, *c, *d; } video_buffer_t;
''' + video_api + r'''
#define FRAMEBUFFER_BYTES 64
#define FRAMEBUFFER_POOL_BYTES 128
#define DISPLAY_WIDTH 8
#define DISPLAY_HEIGHT 8
#define VIDEO_OUT_PIXEL_FORMAT 1
static video_api_t runtime_video_api;
static int runtime_video_handle = -1, runtime_video_registered;
static int runtime_scanout_flush_needed; /* Re-armed by every registration and release. */
static unsigned char *runtime_video_framebuffer;
static size_t runtime_video_framebuffer_size;
static unsigned runtime_present_count;
static uint64_t runtime_render_marker;
static int close_failure, closes, unregisters;
static int close_video(int handle) { assert(handle == 7); ++closes; return close_failure ? -1 : 0; }
static int unregister_video(int handle, int group) {
    assert(handle == 7 && group == 0); ++unregisters;
    return (int)UINT32_C(0x80290009); /* Active scanout may keep its buffer set busy. */
}
static unsigned opens, registrations, sleeps, pending_calls, waits, pending_until;
static int acquire_failure, pending_error, wait_error;
static unsigned sleep_error_at;
static int open_video(int32_t user, int32_t bus, int32_t index, const void *p) {
    assert(user == 0xff && !bus && !index && !p); ++opens;
    return acquire_failure == 1 ? -10 : 7;
}
static int flip_rate(int32_t handle, int32_t rate) {
    assert(handle == 7 && !rate); return acquire_failure == 2 ? -11 : 0;
}
static void attribute(video_attribute_t *a, uint64_t format, uint32_t tile,
                      uint32_t w, uint32_t h, uint64_t x, uint32_t y, uint64_t z) {
    assert(format == 1 && !tile && w == 8 && h == 8 && !x && !y && !z);
    a->words[0] = 1;
}
static int register_video(int32_t handle, int32_t group, int32_t start,
                          video_buffer_t *b, int32_t count, video_attribute_t *a,
                          int32_t flags, void *p) {
    assert(handle == 7 && !group && !start && count == 2 && !flags && !p);
    assert(b[0].a && b[1].a && a->words[0] == 1);
    ++registrations;
    return acquire_failure == 3 ? -12 : 0;
}
static int sceKernelUsleep(uint32_t us) {
    assert(us == 500000); ++sleeps;
    return sleep_error_at && sleeps == sleep_error_at ? -23 : 0;
}
#if PS5_SCANOUT_FPS > 60
int sceVideoOutIsOutputSupported(int32_t h, uint32_t mode, const void *a, const void *b, const void *c) {
    assert(h == 7 && mode == 15 && !a && !b && !c); return 1;
}
int sceVideoOutConfigureOutput(int32_t h, uint32_t mode, const void *a, const void *b, const void *c) {
    assert(h == 7 && (mode == 1 || mode == 15) && !a && !b && !c); return 0;
}
#endif
static int pending(int32_t handle) {
    assert(handle == 7); ++pending_calls;
    return pending_error ? pending_error : (waits < pending_until);
}
static int vblank(int32_t handle) { assert(handle == 7); ++waits; return wait_error; }
static int runtime_video_wait_idle(void);
''' + hfr + shutdown + acquire_wait + r'''
#define EGLAPI
#define EGLAPIENTRY
#define EGL_TRUE 1
#define EGL_FALSE 0
#define EGL_BAD_DISPLAY 0x3008
#define EGL_BAD_SURFACE 0x300d
#define EGL_BAD_ACCESS 0x3002
#define PS5_EGL_LOCK() ((void)0)
typedef void *EGLDisplay;
typedef void *EGLSurface;
typedef unsigned EGLBoolean;
struct pipe_resource { int unused; };
struct pipe_screen { void (*destroy)(struct pipe_screen *); };
struct ps5_egl_display {
    bool initialized; unsigned contexts, surfaces;
    struct pipe_screen *screen; int frontend;
    struct pipe_resource *scanout[2];
};
struct ps5_egl_surface {
    unsigned magic; bool window, current, used;
    int drawable; struct pipe_resource *depth_stencil, *targets[2];
    struct ps5_egl_surface *next;
};
static struct ps5_egl_display ps5_display;
static struct ps5_egl_surface surface, *ps5_surfaces, *ps5_window_surface;
static struct pipe_resource resource;
static unsigned char scanout[64];
static int releases, locked, egl_error;
static void ps5_set_error(int value) { egl_error = value; }
static int ps5_valid_display(EGLDisplay d, bool initialized) {
    return d == &ps5_display && (!initialized || ps5_display.initialized);
}
static struct ps5_egl_surface *ps5_surface(EGLSurface s) {
    return s == ps5_surfaces && surface.magic == 1 ? &surface : NULL;
}
static void ps5_screen_submit_lock(struct pipe_screen *s) { assert(s && !locked); ++locked; }
static void ps5_screen_submit_unlock(struct pipe_screen *s) { assert(s && locked == 1); --locked; }
static void pipe_resource_reference(struct pipe_resource **to, void *from) {
    assert(!from); releases += *to != NULL; *to = NULL;
}
static void st_api_destroy_drawable(int *drawable) { (void)drawable; ++releases; }
static void st_screen_destroy(int *frontend) { (void)frontend; ++releases; }
static void destroy_screen(struct pipe_screen *s) { assert(s); ++releases; }
static void release_surface(void *s) { assert(s == &surface); ++releases; }
#define free release_surface
''' + terminate + destroy + r'''
#undef free
static void setup(void) {
    static struct pipe_screen screen = {destroy_screen};
    runtime_video_api = (video_api_t){.unregister_buffers=unregister_video, .close=close_video,
        .open=open_video, .set_flip_rate=flip_rate, .set_attribute2=attribute,
        .register_buffers2=register_video, .is_flip_pending=pending, .wait_vblank=vblank};
    runtime_video_handle = 7; runtime_video_registered = 1;
    runtime_video_framebuffer = scanout; runtime_video_framebuffer_size = sizeof(scanout);
    runtime_present_count = 6; runtime_render_marker = 200;
    runtime_batch_faulted = runtime_batch_active = runtime_batch_count = 0;
    runtime_pending_batches = 0;
#ifdef PS5_GPU_PRESENT_BATCH
    runtime_gpu_present_buffer = -1; runtime_gpu_present_count = 6;
    deferred_flip = deferred_error = 0;
#endif
    close_failure = closes = unregisters = releases = locked = egl_error = 0;
    opens = registrations = sleeps = pending_calls = waits = pending_until = 0;
    acquire_failure = pending_error = wait_error = 0;
    ps5_display = (struct ps5_egl_display){.initialized=true, .screen=&screen,
        .scanout={&resource, &resource}, .surfaces=1};
    surface = (struct ps5_egl_surface){.magic=1, .window=true, .used=true,
        .depth_stencil=&resource, .targets={&resource, &resource}};
    ps5_surfaces = ps5_window_surface = &surface;
}
static void inject(int failure) {
    close_failure = failure == 1;
    runtime_batch_active = failure == 2;
    runtime_batch_faulted = failure == 3;
    runtime_batch_count = failure == 4;
    pending_error = failure == 5 ? -21 : 0;
    wait_error = failure == 6 ? -22 : 0;
    pending_until = failure == 7 ? 121 : failure == 6 ? waits + 1 : 0;
#ifdef PS5_GPU_PRESENT_BATCH
    runtime_gpu_present_buffer = failure == 8 ? 0 : -1;
    runtime_pending_batches = failure == 9;
#else
    runtime_pending_batches = failure == 8;
#endif
}
int main(void) {
#if PS5_SCANOUT_FPS > 60
    setup();
    video_api_t api = runtime_video_api;
    runtime_video_handle = -1; runtime_video_registered = 0;
    runtime_video_framebuffer = NULL; runtime_video_framebuffer_size = 0;
    int attempts = -1;
    assert(!runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(opens == 1 && registrations == 1 && sleeps == 0 && attempts == 1);
    assert(!runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(opens == 1 && sleeps == 0 && attempts == 0); /* Live reuse, not reopen. */
    assert(!ps5_agc_gate2_shutdown_present() && closes == 1 && sleeps == 0);
    assert(!ps5_agc_gate2_shutdown_present() && closes == 1 && sleeps == 0);
    assert(!runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(sleeps == 10 && opens == 2 && registrations == 2 && attempts == 1);
    assert(!runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(sleeps == 10 && opens == 2 && attempts == 0);

    /* A close error keeps ownership; settling cannot authorize replacement. */
    unsigned char replacement[FRAMEBUFFER_POOL_BYTES];
    close_failure = 1;
    assert(runtime_video_acquire(&api, replacement, sizeof(replacement), &attempts));
    assert(runtime_video_handle == 7 && runtime_video_framebuffer == scanout);
    assert(opens == 2 && sleeps == 10 && attempts == 0);
    close_failure = 0;
    assert(!runtime_video_acquire(&api, replacement, sizeof(replacement), &attempts));
    assert(opens == 3 && sleeps == 20 && attempts == 1);

    /* An interrupted settle never opens/registers; a caller retry waits anew. */
    assert(!ps5_agc_gate2_shutdown_present());
    sleep_error_at = sleeps + 4;
    assert(runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts) == -23);
    assert(opens == 3 && registrations == 3 && sleeps == 24 && attempts == 0);
    assert(runtime_video_handle == -1 && !runtime_video_registered);
    assert(!ps5_agc_gate2_shutdown_present()); /* Must retain the reopen guard. */
    sleep_error_at = 0;
    assert(!runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(opens == 4 && sleeps == 34 && attempts == 1);

    /* Failed setup that successfully closes also needs guarded reacquisition. */
    assert(!ps5_agc_gate2_shutdown_present());
    acquire_failure = 2;
    assert(runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(opens == 5 && sleeps == 44 && runtime_video_handle == -1);
    acquire_failure = 0;
    assert(!runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts));
    assert(opens == 6 && sleeps == 54);
    assert(!ps5_agc_gate2_shutdown_present());
    puts("present-HFR-reopen: PASS first/live acquisition, five-second settle, close/setup errors and interrupted waits");
    return 0;
#endif
    for (unsigned n = 0; n <= 121; ++n) {
        setup(); pending_until = n;
        assert((ps5_agc_gate2_shutdown_present() == 0) == (n <= 120));
        assert(waits == (n > 120 ? 120 : n));
        assert(closes == (n <= 120) && unregisters == 0);
        if (n > 120)
            assert(runtime_video_handle == 7 && runtime_video_registered && runtime_video_framebuffer == scanout);
            assert(runtime_scanout_flush_needed == 1);
    }
    setup(); pending_error = -21;
    assert(runtime_video_wait_idle() != 0 && !waits && pending_calls == 1);
    for (int error = -22; error <= 22; error += 44) {
        setup(); pending_until = 3; wait_error = error;
        assert(runtime_video_wait_idle() != 0 && waits == 1 && pending_calls == 1);
    }
    for (unsigned n = 0; n <= 121; ++n) {
        setup(); pending_until = n;
        assert((runtime_video_wait_idle() == 0) == (n <= 120));
        assert(waits == (n > 120 ? 120 : n) && pending_calls == waits + 1);
    }
    for (int failure = 0; failure <= 3; ++failure) {
        for (int close_error = 0; close_error <= 1; ++close_error) {
            setup();
            video_api_t api = runtime_video_api;
            runtime_video_api = (video_api_t){0};
            runtime_video_handle = -1; runtime_video_registered = 0;
            runtime_video_framebuffer = NULL; runtime_video_framebuffer_size = 0;
            acquire_failure = failure; close_failure = close_error;
            int attempts = 0;
            int rc = runtime_video_acquire(&api, scanout, sizeof(scanout), &attempts);
            assert((rc == 0) == (failure == 0));
            assert(attempts == (failure == 1 ? 3 : 1));
            assert(opens == (unsigned)attempts && sleeps == (failure == 1 ? 2u : 0u));
            if (!failure || (failure >= 2 && close_error)) {
                assert(runtime_video_handle == 7 && runtime_video_api.close == close_video);
                assert(runtime_video_registered == !failure);
                if (close_error) {
                    assert(!eglDestroySurface(&ps5_display, &surface));
                    assert(egl_error == EGL_BAD_ACCESS && !releases && surface.magic == 1);
                }
                close_failure = 0;
                assert(eglDestroySurface(&ps5_display, &surface));
                assert(runtime_video_handle == -1 && !runtime_video_registered);
            } else {
                assert(runtime_video_handle < 0 && !runtime_video_registered);
            }
        }
    }
    for (int registered = 0; registered <= 1; ++registered) {
        setup(); close_failure = 1; runtime_video_registered = registered;
        assert(ps5_agc_gate2_shutdown_present() != 0);
        assert(runtime_video_handle == 7 && runtime_video_framebuffer == scanout);
        assert(runtime_video_framebuffer_size == sizeof(scanout) && runtime_present_count == 6);
        assert(runtime_render_marker == 200 && runtime_video_api.close == close_video);
        assert(runtime_video_registered == registered);
        close_failure = 0;
        assert(ps5_agc_gate2_shutdown_present() == 0);
        assert(runtime_video_handle == -1 && !runtime_video_framebuffer && !runtime_video_registered);
        assert(unregisters == 0 && closes == 2);
        assert(ps5_agc_gate2_shutdown_present() == 0 && closes == 2); /* Idempotent. */
    }
    /* A smaller same-base view reuses the registered pool. Replacing the
     * backing must not open/register while the old close fails. */
    for (int changed_pointer = 0; changed_pointer <= 1; ++changed_pointer) {
        unsigned char replacement[FRAMEBUFFER_POOL_BYTES];
        setup(); close_failure = 1;
        video_api_t api = runtime_video_api;
        unsigned char *next = changed_pointer ? replacement : scanout;
        size_t bytes = changed_pointer ? sizeof(replacement) : sizeof(scanout) / 2;
        int attempts = -1;
        if (!changed_pointer) {
            assert(!runtime_video_acquire(&api, next, bytes, &attempts));
            assert(!attempts && !opens && !registrations && !sleeps && !closes);
            close_failure = 0;
            assert(!ps5_agc_gate2_shutdown_present() && closes == 1);
            continue;
        }
        assert(runtime_video_acquire(&api, next, bytes, &attempts) != 0);
        assert(!attempts && !opens && !registrations && !sleeps && closes == 1 && !unregisters);
        assert(runtime_video_handle == 7 && runtime_video_registered);
        assert(runtime_video_framebuffer == scanout && runtime_video_framebuffer_size == sizeof(scanout));
        assert(runtime_video_api.close == close_video && runtime_present_count == 6 && runtime_render_marker == 200);
        assert(!releases && surface.magic == 1 && ps5_display.scanout[0] == &resource);
        close_failure = 0;
        assert(!runtime_video_acquire(&api, next, bytes, &attempts));
        assert(attempts == 1 && opens == 1 && registrations == 1 && closes == 2 && !sleeps && !unregisters);
        assert(runtime_video_registered && runtime_video_framebuffer == next && runtime_video_framebuffer_size == bytes);
        assert(!ps5_agc_gate2_shutdown_present() && closes == 3);
    }
#ifdef PS5_GPU_PRESENT_BATCH
    setup(); runtime_gpu_present_buffer = 1; deferred_flip = 1; deferred_error = -9;
    assert(!eglDestroySurface(&ps5_display, &surface));
    assert(!closes && !releases && deferred_flip && runtime_gpu_present_buffer == 1);
    deferred_error = 0;
    assert(eglDestroySurface(&ps5_display, &surface));
    assert(closes == 1 && releases == 5 && !deferred_flip && runtime_gpu_present_buffer == -1);
#endif
    int last_failure = 8;
#ifdef PS5_GPU_PRESENT_BATCH
    last_failure = 9;
#endif
    for (int failure = 1; failure <= last_failure; ++failure) {
        setup(); inject(failure);
        assert(!eglDestroySurface(&ps5_display, &surface));
        assert(egl_error == EGL_BAD_ACCESS && !releases && !locked);
        assert(surface.magic == 1 && ps5_surfaces == &surface && ps5_window_surface == &surface);
        assert(ps5_display.surfaces == 1 && surface.targets[0] == &resource);
#ifdef PS5_GPU_PRESENT_BATCH
        if (failure == 8) {
            assert(!pending_calls && !waits && !closes && !unregisters);
            assert(runtime_gpu_present_buffer == 0 && runtime_gpu_present_count == 6);
            assert(runtime_video_registered && runtime_video_handle == 7 && runtime_video_framebuffer == scanout);
        }
#endif
        inject(0); /* Simulated recovery only: no hardware error is injected. */
        assert(eglDestroySurface(&ps5_display, &surface));
        assert(!surface.magic && !ps5_surfaces && !ps5_window_surface && !ps5_display.surfaces);
        assert(releases == 5 && !locked);
#ifdef PS5_GPU_PRESENT_BATCH
        assert(runtime_gpu_present_buffer == -1 && !runtime_gpu_present_count && runtime_video_handle == -1);
#endif
        assert(eglTerminate(&ps5_display) && !ps5_display.initialized && !locked);

        setup(); ps5_display.surfaces = 0; inject(failure);
        assert(!eglTerminate(&ps5_display));
        assert(egl_error == EGL_BAD_ACCESS && !releases && !locked);
        assert(ps5_display.initialized && ps5_display.scanout[0] == &resource && ps5_display.screen);
#ifdef PS5_GPU_PRESENT_BATCH
        if (failure == 8) {
            assert(!pending_calls && !waits && !closes && !unregisters);
            assert(runtime_gpu_present_buffer == 0 && runtime_gpu_present_count == 6);
            assert(runtime_video_registered && runtime_video_handle == 7 && runtime_video_framebuffer == scanout);
        }
#endif
        inject(0);
        assert(eglTerminate(&ps5_display));
        assert(!ps5_display.initialized && !ps5_display.scanout[0] && releases == 4 && !locked);
#ifdef PS5_GPU_PRESENT_BATCH
        assert(runtime_gpu_present_buffer == -1 && !runtime_gpu_present_count && runtime_video_handle == -1);
#endif
    }
    puts("present-shutdown: PASS close errors/batch guards retain runtime, surface and display ownership");
    puts("present-acquire/wait: PASS failed acquisition retains close ownership; errors stop; 120 waits bounded");
    puts("present-drain: PASS bounded drains precede close-only teardown; failures retain EGL resources");
    puts("present-reacquire: PASS smaller same-base views reuse; failed close retains old backing");
#ifdef PS5_GPU_PRESENT_BATCH
    puts("present-queued-shutdown: PASS unconfirmed scanout retains runtime/surface/display before drain or close");
#endif
}
'''
with tempfile.TemporaryDirectory() as temporary:
    executable = str(Path(temporary) / "present-shutdown")
    for flags in ([], ["-DPS5_GPU_PRESENT_BATCH=1"],
                  ["-DPS5_SCANOUT_FPS=120", "-DPS5_NATIVE_TITLE_RUNTIME=1"],
                  ["-DPS5_SCANOUT_FPS=120", "-DPS5_NATIVE_TITLE_RUNTIME=1", "-DPS5_GPU_PRESENT_BATCH=1"]):
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wno-address",
                        *flags, "-x", "c", "-o", executable, "-"], input=code, text=True, check=True)
        subprocess.run([executable], cwd=temporary, check=True)
