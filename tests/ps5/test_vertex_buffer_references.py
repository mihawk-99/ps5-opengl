#!/usr/bin/env python3
# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the actual vertex-buffer setter with Mesa's reference-counting contract.

u_vbuf hands set_vertex_buffers raw pointers it kept from earlier calls, so a
buffer in one call can be one the context already holds, possibly with the
context's reference as the last. The setter must take the new references before
it drops the old ones. The stub below is pipe_reference_described from Mesa
26.2 (src/gallium/auxiliary/util/u_inlines.h), assertions included. The previous
order, run through the same stub, must fail the assertion."""
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = (root / 'src/gallium/ps5/ps5_screen.c').read_text()
start = source.rindex('static void\nps5_set_vertex_buffers(')
function = source[start:source.index('\n}\n', start) + 3]
assert function.count('pipe_resource_reference(') >= 2

PREVIOUS = r'''
static void
ps5_set_vertex_buffers_previous(struct pipe_context *base, unsigned count,
                                const struct pipe_vertex_buffer *buffers)
{
   struct ps5_context *context = (struct ps5_context *)base;
   unsigned index;

   for (index = 0; index < context->vertex_buffer_count; ++index)
      pipe_resource_reference(
         &context->vertex_buffers[index].buffer.resource, NULL);
   memset(context->vertex_buffers, 0, sizeof(context->vertex_buffers));
   context->vertex_buffer_count = 0;
   if (!count)
      return;
   if (count > PIPE_MAX_ATTRIBS || !buffers)
      return;
   for (index = 0; index < count; ++index) {
      if (buffers[index].is_user_buffer)
         return;
   }
   for (index = 0; index < count; ++index) {
      context->vertex_buffers[index] = buffers[index];
      context->vertex_buffers[index].buffer.resource = NULL;
      pipe_resource_reference(
         &context->vertex_buffers[index].buffer.resource,
         buffers[index].buffer.resource);
   }
   context->vertex_buffer_count = count;
}
'''

PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define PIPE_MAX_ATTRIBS 32
struct pipe_reference { int64_t count; };
struct pipe_screen;
struct pipe_resource { struct pipe_reference reference; struct pipe_screen *screen; unsigned id; };
struct pipe_screen { void (*resource_destroy)(struct pipe_screen *, struct pipe_resource *); };
struct pipe_vertex_buffer { bool is_user_buffer; unsigned buffer_offset; union { struct pipe_resource *resource; const void *user; } buffer; };
struct pipe_context { struct pipe_screen *screen; };
struct ps5_context { struct pipe_context base; struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS]; unsigned vertex_buffer_count; };

/* pipe_reference_described, Mesa 26.2 u_inlines.h, with the same assertions */
static bool pipe_reference_described(struct pipe_reference *dst, struct pipe_reference *src)
{
   if (dst != src) {
      if (src) {
         int64_t count = ++src->count;
         assert(count != 1); /* src had to be referenced */
         (void)count;
      }
      if (dst) {
         int64_t count = --dst->count;
         assert(count != -1); /* dst had to be referenced */
         if (!count)
            return true;
      }
   }
   return false;
}
static int destroyed;
static void pipe_resource_reference(struct pipe_resource **dst, struct pipe_resource *src)
{
   struct pipe_resource *old = *dst;
   if (pipe_reference_described(old ? &old->reference : NULL, src ? &src->reference : NULL))
      old->screen->resource_destroy(old->screen, old);
   *dst = src;
}
static void destroy(struct pipe_screen *screen, struct pipe_resource *resource)
{
   (void)screen;
   ++destroyed;
   resource->reference.count = 0;
}
static struct pipe_screen screen = { destroy };
static struct pipe_resource make(unsigned id) { return (struct pipe_resource){ {1}, &screen, id }; }
static struct pipe_vertex_buffer vb(struct pipe_resource *resource)
{
   struct pipe_vertex_buffer buffer = {0};
   buffer.buffer.resource = resource;
   return buffer;
}
'''

CASES = r'''
int main(void)
{
   struct ps5_context context;
   struct pipe_context *base = &context.base;
   /* a: held by the application (count 1); b: the same */
   struct pipe_resource a = make(1), b = make(2);
   struct pipe_vertex_buffer set[2];

   memset(&context, 0, sizeof(context));
   context.base.screen = &screen;

   /* bind a: the context takes its own reference */
   set[0] = vb(&a);
   SETTER(base, 1, set);
   assert(a.reference.count == 2 && context.vertex_buffer_count == 1);

   /* the application drops its reference: the context's is the last */
   a.reference.count = 1;
   /* the same buffer again, as u_vbuf re-sends it */
   SETTER(base, 1, set);
   assert(destroyed == 0 && a.reference.count == 1 && context.vertex_buffers[0].buffer.resource == &a);

   /* the same buffer in a different slot together with a new one */
   set[0] = vb(&b);
   set[1] = vb(&a);
   SETTER(base, 2, set);
   assert(destroyed == 0 && a.reference.count == 1 && b.reference.count == 2 && context.vertex_buffer_count == 2);

   /* replacing: the old one goes when it was the context's last reference */
   set[0] = vb(&b);
   SETTER(base, 1, set);
   assert(destroyed == 1 && a.reference.count == 0 && b.reference.count == 2 && context.vertex_buffer_count == 1);

   /* a user buffer anywhere unbinds everything and keeps no reference */
   set[0] = vb(&b);
   set[1].is_user_buffer = true;
   SETTER(base, 2, set);
   assert(context.vertex_buffer_count == 0 && b.reference.count == 1);

   /* unbinding */
   set[0] = vb(&b);
   SETTER(base, 1, set);
   assert(b.reference.count == 2);
   SETTER(base, 0, NULL);
   assert(b.reference.count == 1 && context.vertex_buffer_count == 0);
   SETTER(base, 1, set);
   SETTER(base, PIPE_MAX_ATTRIBS + 1, set);
   assert(b.reference.count == 1 && context.vertex_buffer_count == 0);
   puts("ok");
   return 0;
}
'''

def build(setter, body, tmp, name):
    code = PRELUDE + PREVIOUS + body + '\n#define SETTER ' + setter + '\n' + CASES
    binary = str(Path(tmp) / name)
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-O1',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-fno-pie', '-no-pie',
                    '-x', 'c', '-o', binary, '-'], input=code, text=True, check=True, timeout=60)
    return binary

with tempfile.TemporaryDirectory(prefix='ps5-vertex-buffer-refs-') as tmp:
    current = build('ps5_set_vertex_buffers', function, tmp, 'current')
    subprocess.run([current], check=True, timeout=30)
    # the previous order, run through the same stub, destroys the buffer and
    # then references it: the assertion must trip
    previous = build('ps5_set_vertex_buffers_previous', '', tmp, 'previous')
    result = subprocess.run([previous], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'count != 1' in result.stderr, result.stderr
print('PASS: vertex buffers re-sent while the context holds their last reference keep their references; '
      'the previous order trips the reference assertion')
