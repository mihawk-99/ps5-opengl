#!/usr/bin/env python3
# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every vertex format a GL vertex array can ask for must reach native fetch.

Mesa's state tracker turns glVertexAttribPointer (type, size, normalized or integer)
into one of the pipe formats in varray.c's vertex_formats table. A format the screen
does not admit for vertex buffers is not rejected: u_vbuf converts the whole range of
every draw on the CPU into a new stream buffer, which games that draw large vertex
buffers with 8- or 16-bit attributes (Sodium and Iris in Minecraft) turn into
gigabytes of direct memory. This compiles the driver's actual admission functions and
the compiler's actual mapping and checks them against that table; no GPU required.

The formats left to u_vbuf are exactly those with no hardware fetch format: 32-bit
normalized and scaled integers, fixed point and doubles (which become floats).
"""
import re
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
driver = (root / 'src/gallium/ps5/ps5_screen.c').read_text()
compiler = (root / 'third_party/opengnm-psbc/libpsbc/psbc_compile.c').read_text()
mesa = (root / 'third_party/mesa-26.2.0/src/mesa/main/varray.c').read_text()


def function(source, name):
    match = re.search(r'(?m)^(?:static enum pipe_format )?' + name + r'\([^;{}]*\)\s*\{', source)
    start = source.index(name, match.start())
    return ' ' + source[start:source.index('\n}', start) + 2]


# Every pipe format Mesa can hand the driver for a vertex attribute.
tables = mesa[mesa.index('static const uint8_t vertex_formats'):mesa.index('/**\n * Return a PIPE_FORMAT_x')]
gl_formats = sorted(set(re.findall(r'PIPE_FORMAT_\w+', tables)))
assert len(gl_formats) > 100, 'the varray.c vertex format tables were not found'

# No hardware fetch format exists for these (ac_get_vtx_format_info has no entry): u_vbuf converts them.
left_to_u_vbuf = re.compile(
    r'PIPE_FORMAT_R32(G32)?(B32)?(A32)?_(UNORM|SNORM|USCALED|SSCALED|FIXED)$|'
    r'PIPE_FORMAT_R64(G64)?(B64)?(A64)?_(FLOAT|UINT)$')
native = [name for name in gl_formats if not left_to_u_vbuf.match(name)]

functions = ('bool' + function(driver, 'ps5_packed_vertex_format') + '\n' +
             'bool' + function(driver, 'ps5_integer_vertex_format') + '\n' +
             'bool' + function(driver, 'ps5_vertex_format') + '\n' +
             'enum pipe_format' + function(compiler, 'psbc_vertex_pipe_format'))
pipe = sorted(set(re.findall(r'PIPE_FORMAT_\w+', functions)) | set(gl_formats))
psbc = sorted(set(re.findall(r'PSBC_VERTEX_FORMAT_\w+', functions)) | {'PSBC_VERTEX_FORMAT_NONE'})
flags = sorted(set(re.findall(r'PS5_ENABLE_\w+', functions)))
floats32 = ['PIPE_FORMAT_R32_FLOAT', 'PIPE_FORMAT_R32G32_FLOAT', 'PIPE_FORMAT_R32G32B32_FLOAT',
            'PIPE_FORMAT_R32G32B32A32_FLOAT']  # admitted by name in ps5_is_format_supported

code = ('#include <stdio.h>\n#include <stdbool.h>\n' +
        'enum pipe_format {' + ','.join(pipe) + '};\n' +
        'typedef enum {' + ','.join(psbc) + '} PsbcVertexFormat;\n' +
        '\n'.join('#define ' + flag + ' 1' for flag in flags) + '\n' + functions + '\n'
        'static int failures;\n'
        'static void check(const char *name, enum pipe_format format, int expect_admitted) {\n'
        '   PsbcVertexFormat out = PSBC_VERTEX_FORMAT_NONE;\n'
        '   bool admitted = ps5_vertex_format(format, &out);\n'
        '   bool screen = ps5_packed_vertex_format(format) || ps5_integer_vertex_format(format);\n'
        '   if (expect_admitted == 1 && !(admitted && screen && psbc_vertex_pipe_format(out) == format)) {\n'
        '      printf("FAIL: %s is not native (admitted=%d reported=%d)\\n", name, admitted, screen); failures++; }\n'
        '   if (expect_admitted == 2 && !admitted) { printf("FAIL: %s is not admitted\\n", name); failures++; }\n'
        '}\n'
        'int main(void) {\n')
for name in native:
    expect = 2 if name in floats32 else 1
    code += f'   check("{name}", {name}, {expect});\n'
code += '   printf("%d GL vertex formats checked, %d failures\\n", ' + str(len(native)) + ', failures);\n'
code += '   return failures != 0;\n}\n'

with tempfile.TemporaryDirectory() as tmp:
    exe = Path(tmp) / 'vertex-format-table'
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-x', 'c', '-',
                    '-o', str(exe)], input=code, text=True, check=True)
    result = subprocess.run([str(exe)], text=True, capture_output=True)
print(result.stdout, end='')
assert result.returncode == 0, 'some GL vertex formats are not admitted for native fetch'
print(f'PASS: all {len(native)} GL vertex formats with a hardware fetch format reach native fetch; '
      f'{len(gl_formats) - len(native)} are left to u_vbuf')
