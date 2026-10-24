"""Experimental bounded shared storage for stateless prefill graph regions."""
import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group

class Regions:
    def __init__(self):
        self.pool=torch.xpu.graph_pool_handle();self.inputs={};self.outputs={};self.entries={};self.checks=[]
    def call(self,module,name,fn,args):
        m=args[0].shape[0]
        if m <= 32 or m > 256 or torch.xpu.is_current_stream_capturing():
            return fn(*args)
        bucket = 256
        key=(id(module),name,bucket,tuple(x is None for x in args))
        stat=[]
        for i,x in enumerate(args):
            if x is None:stat.append(None);continue
            signature=(name,i,tuple(x.shape[1:]),x.dtype,x.device)
            if signature not in self.inputs:self.inputs[signature]=torch.empty((256,*x.shape[1:]),dtype=x.dtype,device=x.device)
            y=self.inputs[signature][:bucket];y[:m].copy_(x)
            if m<bucket:y[m:].zero_()
            stat.append(y)
        entry=self.entries.get(key)
        if entry is None:
            expected=fn(*stat);single=isinstance(expected,torch.Tensor)
            values=(expected,) if single else expected
            expected=tuple(x.clone() if x is not None else None for x in values)
            outputs=[]
            for i,x in enumerate(values):
                if x is None:outputs.append(None);continue
                signature=(name,i,tuple(x.shape[1:]),x.dtype,x.device)
                if signature not in self.outputs:self.outputs[signature]=torch.empty((256,*x.shape[1:]),dtype=x.dtype,device=x.device)
                outputs.append(self.outputs[signature][:bucket])
            del values
            graph=torch.xpu.XPUGraph()
            with torch.xpu.graph(graph,pool=self.pool):
                value=fn(*stat);values=(value,) if single else value
                for dst,src in zip(outputs,values):
                    if dst is not None:dst.copy_(src)
            del value,values
            graph.replay();torch.xpu.synchronize()
            passed=all(torch.equal(a,b) if a is not None else b is None for a,b in zip(expected,outputs))
            verdict=torch.tensor(int(passed),dtype=torch.int32);dist.all_reduce(verdict,op=dist.ReduceOp.MIN,group=get_tp_group().cpu_group)
            self.checks.append(dict(name=name,bucket=bucket,passed=bool(verdict.item())))
            assert verdict.item(),self.checks[-1]
            entry=self.entries[key]=(graph,outputs,single)
        else:entry[0].replay()
        result=tuple(x[:m] if x is not None else None for x in entry[1])
        return result[0] if entry[2] else result


_regions = None

import os as _os

# B70-V32-08: B70_PREFILL_HC_GRAPHS=0 turns off the 33-256-row prefill HC graph regions (one captured graph
# per HC module plus their pool memory, sized before KV): KV memory over short-prefill speed.
_B70_PREFILL_HC_GRAPHS = _os.environ.get("B70_PREFILL_HC_GRAPHS", "1") == "1"


def prefill_hc(module, name, fn, args):
    global _regions
    if not _B70_PREFILL_HC_GRAPHS:
        return None
    rows = args[0].shape[0]
    if rows <= 32 or torch.xpu.is_current_stream_capturing():
        return None
    if _regions is None:
        _regions = Regions()
    if rows > 256:
        # Profile-time initialization accounts bounded persistent storage before
        # KV sizing. These operations are stateless and preserve the inputs.
        warmed = getattr(module, "_prefill_hc_warmed", set())
        if name not in warmed:
            for width in (256,):
                _regions.call(module, name, fn, tuple(torch.zeros_like(x[:width]) if x is not None else None for x in args))
            warmed.add(name)
            module._prefill_hc_warmed = warmed
        return None
    return _regions.call(module, name, fn, args)
