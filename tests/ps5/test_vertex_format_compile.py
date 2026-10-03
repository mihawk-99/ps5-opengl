#!/usr/bin/env python3
# PS5 OpenGL - OpenGL implementation for PlayStation 5.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every vertex format the driver admits for native fetch must compile.

Builds one vertex shader per input class (float, signed integer, unsigned integer)
and runs the PSBC compiler on it for every format the driver's ps5_vertex_format
admits, with an attribute at offset 0 and a second one at an odd offset inside a
stride that is only aligned to the format's channel size, the way a packed game
vertex is laid out. A format the compiler cannot lower fails here instead of on a
console. Needs glslangValidator and the host compiler (toolchain/build-opengnm-psbc.sh).
"""
import re
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
compiler = root / 'third_party/opengnm-psbc/opengnm-psbc'
driver = (root / 'src/gallium/ps5/ps5_screen.c').read_text()
cli = (root / 'third_party/opengnm-psbc/cmd/psbc/main.c').read_text()
assert compiler.exists(), 'run toolchain/build-opengnm-psbc.sh first'

body = driver[driver.rindex('\nps5_vertex_format(enum pipe_format format'):]
body = body[:body.index('\n}\n')]
admitted = sorted(set(re.findall(r'PIPE_FORMAT_(\w+):\n\s+\*out', body)))
known = set(re.findall(r'strcmp\(name, "(\w+)"\)', cli))
names = [name for name in admitted if name.lower() in known]
assert names == admitted and len(names) > 70, 'every admitted format needs a compiler name in cmd/psbc/main.c'

shaders = {
    'float': 'vec4',
    'sint': 'ivec4',
    'uint': 'uvec4',
}


def element_bytes(name):
    channels = re.match(r'([A-Z0-9]+)_', name).group(1)
    if channels.startswith('R10') or channels.startswith('B10') or channels.startswith('R11'):
        return 4, 4
    sizes = [int(size) for size in re.findall(r'[RGBA](\d+)', channels)]
    return sum(sizes) // 8, sizes[0] // 8


def input_class(name):
    return 'uint' if name.endswith('_UINT') else 'sint' if name.endswith('_SINT') else 'float'


failures = []
with tempfile.TemporaryDirectory(prefix='ps5-vertex-compile-') as tmp:
    tmp = Path(tmp)
    spirv = {}
    for kind, glsl_type in shaders.items():
        source = tmp / f'{kind}.vert'
        source.write_text(f'#version 450\nlayout(location=0) in {glsl_type} a;\n'
                          f'layout(location=1) in {glsl_type} b;\n'
                          f'void main() {{ gl_Position = vec4(a) + vec4(b); }}\n')
        spirv[kind] = tmp / f'{kind}.spv'
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.2', '-S', 'vert', str(source),
                        '-o', str(spirv[kind])], check=True, capture_output=True)
    for name in names:
        size, channel = element_bytes(name)
        kind = input_class(name)
        # second attribute at offset 1 (or the channel size) in a stride of two elements plus one byte
        second = 1 if channel == 1 else channel
        stride = 2 * size + second
        format_name = name.lower()
        command = [str(compiler), '-g', '-s', 'vertex', '--raw', '--address32-hi', '2',
                   '--vertex-attribute', f'0:{format_name}:0:0:{stride}:{channel}',
                   '--vertex-attribute', f'1:{format_name}:0:{size + second}:{stride}:{channel}',
                   '-f', str(spirv[kind]), '-o', str(tmp / f'{name}.bin')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        output = tmp / f'{name}.bin'
        if result.returncode or not output.exists() or output.stat().st_size == 0:
            failures.append((name, (result.stderr or result.stdout).strip().splitlines()[-1:]))
for name, why in failures:
    print(f'FAIL: {name}: {why}')
assert not failures, f'{len(failures)} of {len(names)} admitted vertex formats do not compile'
print(f'PASS: all {len(names)} admitted vertex formats compile, aligned and at an odd offset')
