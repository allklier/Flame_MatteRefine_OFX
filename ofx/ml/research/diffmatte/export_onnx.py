# Exports DiffMatte as one ONNX graph: input "pixel_values" [1,4,T,T] (as ViTMatte in Matte Refine
# ML), output "alphas" [1,1,T,T]; the ViT encoder runs once, the DUNet decoder `steps` times
# (DDIM unrolled), the initial noise is a constant (seed 0) in the graph. Then checks the graph
# with ONNX Runtime (CPU) against the PyTorch wrapper and the authors' ddim_sample.
#   export_onnx.py <variant> <weights.pt> <steps> <tile> <out.onnx> [seed]
import sys, time, hashlib, os, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import dm_common as dc
import onnx, onnxruntime as ort
variant, wts, steps, T, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
m = dc.build(variant, wts)
tile = dc.Tile(m, steps, dc.make_noise(T, seed)).eval()
# A test input: random picture, a trimap with all three classes.
g = torch.Generator().manual_seed(1)
rgb = torch.rand(1, 3, T, T, generator=g)
yy, xx = torch.meshgrid(torch.arange(T), torch.arange(T), indexing="ij")
r = ((yy - T / 2) ** 2 + (xx - T / 2) ** 2).sqrt()
tri = torch.where(r < T * 0.25, 1.0, torch.where(r < T * 0.35, 0.5, 0.0))[None, None].float()
x = torch.cat([(rgb - 0.5) / 0.5, tri], 1)
with torch.no_grad():
    ref_t = tile(x)
t0 = time.time()
torch.onnx.export(tile, (x,), out, input_names=["pixel_values"], output_names=["alphas"],
                  opset_version=17, do_constant_folding=True, dynamo=False)
print(f"exported in {time.time() - t0:.1f}s")
mo = onnx.load(out)
onnx.checker.check_model(mo)
ops = sorted({n.op_type for n in mo.graph.node})
print("nodes", len(mo.graph.node), "ops", ops)
print("inputs", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in mo.graph.input])
print("outputs", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in mo.graph.output])
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
o = s.run(None, {"pixel_values": x.numpy()})[0]
print(f"ORT CPU vs PyTorch wrapper: max|d| {np.abs(o - ref_t.numpy()).max():.2e}, mean|d| {np.abs(o - ref_t.numpy()).mean():.2e}")
ref = dc.reference(m, steps, rgb, tri, dc.make_noise(T, seed))
print(f"ORT CPU vs authors' ddim_sample: max|d| {np.abs(o - ref.numpy()).max():.2e}")
t0 = time.time(); s.run(None, {"pixel_values": x.numpy()}); print(f"ORT CPU one tile {1000*(time.time()-t0):.0f} ms")
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"FILE {out} bytes {os.path.getsize(out)} sha256 {h}")
