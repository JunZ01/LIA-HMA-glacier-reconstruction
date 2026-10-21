"""Non-executing reader for the project's numeric centreline pickle format.

pickletools only tokenizes bytes. GLOBAL/REDUCE/BUILD are interpreted as
descriptions of float64 arrays, never as Python imports, constructors or calls.
Unsupported opcodes, types and object dtypes fail closed.
"""
import gzip
from pathlib import Path
import pickletools
import numpy as np


def read_network(path):
    raw=gzip.decompress(Path(path).read_bytes())
    if len(raw)>8_000_000: raise ValueError('Network file exceeds bounded format size')
    stack=[]; memo={}; marker=object()
    def marked():
        i=len(stack)-1
        while i>=0 and stack[i] is not marker: i-=1
        if i<0: raise ValueError('Missing mark')
        values=stack[i+1:]; del stack[i:]; return values
    for op,arg,pos in pickletools.genops(raw):
        name=op.name
        if name in ('PROTO','FRAME'): continue
        if name=='EMPTY_DICT': stack.append({})
        elif name=='EMPTY_LIST': stack.append([])
        elif name=='MARK': stack.append(marker)
        elif name=='MEMOIZE': memo[len(memo)]=stack[-1]
        elif name in ('BINGET','LONG_BINGET'): stack.append(memo[arg])
        elif name in ('SHORT_BINUNICODE','BINUNICODE','SHORT_BINBYTES','BINBYTES','BININT','BININT1','BININT2','BINFLOAT'): stack.append(arg)
        elif name=='NONE': stack.append(None)
        elif name=='NEWTRUE': stack.append(True)
        elif name=='NEWFALSE': stack.append(False)
        elif name=='TUPLE': stack.append(tuple(marked()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            n=int(name[-1]); values=tuple(stack[-n:]); del stack[-n:]; stack.append(values)
        elif name=='STACK_GLOBAL':
            symbol=stack.pop(); module=stack.pop()
            if (module,symbol) not in {('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')}:
                raise ValueError('Unsupported symbolic global')
            stack.append(('symbol',module,symbol))
        elif name=='REDUCE':
            arguments=stack.pop(); symbol=stack.pop()
            if not isinstance(symbol,tuple) or len(symbol)!=3 or symbol[0]!='symbol': raise ValueError('No callable evaluation allowed')
            if symbol[2]=='dtype' and arguments==('f8',False,True): stack.append({'__numeric_description__':'dtype','endian':'<'})
            elif symbol[2]=='_reconstruct' and arguments==(('symbol','numpy','ndarray'),(0,),b'b'):
                stack.append({'__numeric_description__':'array'})
            else: raise ValueError('Unsupported numeric description')
        elif name=='BUILD':
            state=stack.pop(); target=stack[-1]
            if target.get('__numeric_description__')=='dtype':
                if state[0]!=3 or state[1] not in ('<','>') or state[2:5]!=(None,None,None): raise ValueError('Unsupported dtype state')
                target['endian']=state[1]
            elif target.get('__numeric_description__')=='array':
                version,shape,description,fortran,values=state
                if version!=1 or len(shape)>2 or any(not isinstance(n,int) or n<0 or n>100000 for n in shape): raise ValueError('Invalid array shape')
                if description.get('__numeric_description__')!='dtype' or not isinstance(values,bytes): raise ValueError('Only raw float64 numeric arrays supported')
                size=int(np.prod(shape))
                if size>1_000_000 or len(values)!=size*8: raise ValueError('Invalid numeric payload size')
                # Fixed numeric frombuffer reads data; no pickled code is invoked.
                target['data']=np.frombuffer(values,dtype=description['endian']+'f8').copy().reshape(shape,order='F' if fortran else 'C')
            else: raise ValueError('No object build allowed')
        elif name=='APPENDS':
            values=marked(); stack[-1].extend(values)
        elif name=='SETITEMS':
            values=marked()
            if len(values)%2: raise ValueError('Invalid mapping')
            for k,v in zip(values[::2],values[1::2]): stack[-1][k]=v
        elif name=='STOP':
            if len(stack)!=1: raise ValueError('Invalid final stack')
            break
        else: raise ValueError('Unsupported opcode: '+name)
    def plain(v):
        if isinstance(v,dict):
            if v.get('__numeric_description__')=='array': return v['data']
            if '__numeric_description__' in v: raise ValueError('Unexpected dtype outside array')
            return {k:plain(a) for k,a in v.items()}
        if isinstance(v,list): return [plain(a) for a in v]
        if isinstance(v,tuple): return tuple(plain(a) for a in v)
        return v
    result=plain(stack[0])
    if result.get('format')!='lia-projected-centerline-network-v1': raise ValueError('Unknown network format')
    return result
