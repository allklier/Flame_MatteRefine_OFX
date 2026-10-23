# Loads a DiffMatte checkpoint with weights_only=True (allowlisting only the omegaconf / typing data
# classes its pickle references, checked with pickletools) and saves the model state dict alone.
import sys, collections, typing, torch
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
torch.serialization.add_safe_globals([ListConfig, ContainerMetadata, Metadata, AnyNode, typing.Any,
                                      collections.defaultdict, dict, list, int])
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("top-level keys:", list(d.keys()))
for k, v in d.items():
    if k != "model": print(" ", k, repr(v)[:300])
sd = d["model"]
print("tensors", len(sd), "params", sum(v.numel() for v in sd.values()))
print("first", list(sd)[:4]); print("last", list(sd)[-4:])
torch.save(collections.OrderedDict((k, v.clone()) for k, v in sd.items()), sys.argv[2])
