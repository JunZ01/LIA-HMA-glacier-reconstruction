"""Regenerate the scalar momentum stencils from the frozen PoLIM source.

Python changes: retain MATLAB's one-based indices with padded arrays; use
SciPy sparse assignment; restrict options in core.py. No runtime translation.
See LICENSE and NOTICE for upstream terms and attribution.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'reference/classic/solver_u_core.m').read_text()

def translate(block):
    block = re.sub(r'%[^\n]*', '', block)
    block = re.sub(r'\.\.\.\s*\n', '', block)
    lines, indent = [], 1
    arrays = 'LHS RHS L1 L2 L3 L4 L5 L6 Ls1 Ls2 LsW Lb1_LFL Lb2_LFL'.split()
    def expr(s):
        s = s.strip().rstrip(';').replace('^', '**')
        for a in arrays:
            pattern = re.compile(r'\b'+a+r'\(')
            while (match := pattern.search(s)):
                start, end, depth = match.end(), match.end(), 1
                while depth:
                    depth += (s[end]=='(')-(s[end]==')')
                    end += 1
                s=s[:match.start()]+a+'['+s[start:end-1]+']'+s[end:]
        return s
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == 'end':
            indent -= 1
            continue
        if line.startswith('elseif '):
            indent -= 1
            lines.append('    '*indent + 'elif '+expr(line[7:])+':')
            indent += 1
        elif line == 'else':
            indent -= 1
            lines.append('    '*indent + 'else:')
            indent += 1
        elif line.startswith('if '):
            lines.append('    '*indent+'if '+expr(line[3:])+':')
            indent += 1
        elif line.startswith('for '):
            var, bounds = line[4:].split('=')
            lo, hi = bounds.strip().split(':')
            lines.append('    '*indent+f'for {var.strip()} in range({lo}, {hi}+1):')
            indent += 1
        else:
            lines.append('    '*indent+expr(line))
    if indent != 1:
        raise ValueError('unbalanced extracted block')
    return '\n'.join(lines)

interior = source.split('%------------------------------------LHS---------------------------')[1].split('\n    end\nend\n%%')[0]
surface = source.split('    for i = 2:Ms-1\n        if i==2')[1].split("elseif strcmp(type_SBC, 'Dirichlet')")[0]
surface = 'for i = 2:Ms-1\n        if i==2' + surface
basal = source.split("elseif strcmpi(type_BBC, 'LFlaw')")[1].split("elseif strcmpi(type_BBC, 'LFlaw_simple')")[0]
basal = 'for i = 2:Ms-1' + basal.split('for i = 2:Ms-1')[1]
header = '"""Generated scalar stencils, derived from PoLIM classic; see LICENSE/NOTICE.\nRegenerate with python -m polim_cpu.build_stencils.\n"""\n\n'
out = header
out += 'def interior(LHS, L1, L2, L3, L4, L5, L6, i, j, k, N, Ms, dx, dzeta):\n'+translate(interior)+'\n\n'
out += 'def surface(LHS, L1, L2, L3, L4, L5, Ls1, Ls2, LsW, N, Ms, dx, dzeta):\n'+translate(surface)+'\n\n'
# RHS is set by the caller; all remaining statements are literal stencils.
basal = re.sub(r'RHS\([^;]*;', '', basal)
out += 'def basal_linear(LHS, L1, L2, L3, L4, L5, L6, Lb1_LFL, Lb2_LFL, N, Ms, dx, dzeta):\n'+translate(basal)+'\n'
(ROOT / '_stencils.py').write_text(out)
for name in ('LICENSE', 'NOTICE'):
    (ROOT / name).write_bytes((ROOT / 'reference/classic' / name).read_bytes())
print('Generated _stencils.py and preserved upstream notices.')
