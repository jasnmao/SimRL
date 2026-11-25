"""Thin, fast wrapper around the trained YOLO sphere detector.

The trained weights (``yolo_model/best.pt``) are a YOLOv8-style detect model with
two classes:

    0 -> yellow_sphere   (the pointing target)
    1 -> purple_sphere   (the projection of the hand pointing position)

Only the two bounding-box *centres* are needed, and exactly one instance of each
class is present in every frame, so we call the raw network (no ultralytics
letterbox / NMS / result-object overhead) and take a confidence-weighted mean of
the top-k anchor centres for each class.  That is both cheaper and less quantised
(DFL gives sub-cell precision) than taking a single arg-max box.

A forward pass at 640x640 takes ~12 ms of GPU time but ~40 ms of wall time,
because the network is a chain of ~200 tiny kernels and Windows launch latency
dominates.  The forward pass is therefore captured once per batch size in a CUDA
graph and replayed, which brings the call down to the GPU time.
"""

from __future__ import annotations

import numpy as np
import torch

CLASS_NAMES = ("yellow_sphere", "purple_sphere")
TOPK = 5


class SphereDetector:
    def __init__(
        self,
        weights: str = "yolo_model/best.pt",
        device: str = "cuda",
        conf: float = 0.25,
        half: bool = True,
        imgsz: int = 640,
        use_cuda_graph: bool = False,
    ):
        from ultralytics import YOLO

        yolo = YOLO(weights)
        self.net = yolo.model.eval()
        self.device = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
        self.net.to(self.device)
        self.half = bool(half) and self.device.type == "cuda"
        if self.half:
            self.net.half()
        self.conf = conf
        self.imgsz = imgsz
        self.nc = len(CLASS_NAMES)
        self.dtype = torch.float16 if self.half else torch.float32
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self._graphs: dict[int, tuple] = {}
        self.graph_calls = 0

    # ------------------------------------------------------------------
    def _run(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out["one2many"] if "one2many" in out else next(iter(out.values()))
        return out

    def _graph_for(self, b: int):
        """Capture (once) a CUDA graph for a fixed batch size."""
        if b in self._graphs:
            return self._graphs[b]
        in_u8 = torch.zeros((b, self.imgsz, self.imgsz, 3), dtype=torch.uint8, device=self.device)
        in_f = torch.zeros((b, 3, self.imgsz, self.imgsz), dtype=self.dtype, device=self.device)
        # warm-up on a side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self._run(in_f)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                out = self._run(in_f)
        graph.replay()
        torch.cuda.synchronize()
        self._graphs[b] = (graph, in_u8, in_f, out)
        return self._graphs[b]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_raw(self, batch_u8: np.ndarray) -> torch.Tensor:
        """batch_u8: (B,H,W,3) RGB uint8 -> (B, 4+nc, A) float tensor."""
        b = batch_u8.shape[0]
        if self.use_cuda_graph:
            graph, in_u8, in_f, out = self._graph_for(b)
            in_u8.copy_(torch.from_numpy(np.ascontiguousarray(batch_u8)))
            in_f.copy_(in_u8.permute(0, 3, 1, 2).to(self.dtype).div_(255.0))
            graph.replay()
            self.graph_calls += 1
            return out
        x = torch.from_numpy(np.ascontiguousarray(batch_u8)).to(self.device)
        x = x.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        return self._run(x)

    @torch.no_grad()
    def detect(self, batch_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect both spheres in a batch of RGB images.

        Returns
        -------
        centers : (B, 2, 2) float32  -- [:, 0] = yellow (x,y), [:, 1] = purple
        conf    : (B, 2)   float32
        """
        pred = self.forward_raw(batch_u8)
        if pred.dim() != 3:
            raise RuntimeError(f"unexpected detector output shape {tuple(pred.shape)}")
        boxes = pred[:, :4, :].float()                    # cx, cy, w, h (input px)
        scores = pred[:, 4 : 4 + self.nc, :].float()      # already sigmoided by the head
        k = min(TOPK, scores.shape[-1])
        top = scores.topk(k, dim=-1)                      # (B,nc,k)
        w = top.values
        w = torch.where(w > self.conf * 0.4, w, torch.zeros_like(w))
        wn = w / w.sum(-1, keepdim=True).clamp_min(1e-9)  # (B,nc,k)
        idx = top.indices                                 # (B,nc,k)
        B = boxes.shape[0]
        cx = boxes[:, 0, :].unsqueeze(1).expand(B, self.nc, -1).gather(-1, idx)
        cy = boxes[:, 1, :].unsqueeze(1).expand(B, self.nc, -1).gather(-1, idx)
        centers = torch.stack([(cx * wn).sum(-1), (cy * wn).sum(-1)], dim=-1)  # (B,nc,2)
        conf = top.values[..., 0]                         # (B,nc)
        # apparent size of the best anchor box: lets the caller sanity-check the
        # regression branch (see VecPointingEnv.step)
        s0 = boxes[:, 2:4, :].unsqueeze(1).expand(B, self.nc, 2, -1)      # (B,nc,2,A)
        s0 = s0.gather(-1, idx.unsqueeze(2).expand(B, self.nc, 2, -1))    # (B,nc,2,k)
        size = (s0 * wn.unsqueeze(2)).sum(-1).abs().mean(-1)              # (B,nc)
        return (centers.cpu().numpy().astype(np.float32),
                conf.cpu().numpy().astype(np.float32),
                size.cpu().numpy().astype(np.float32))

    def warmup(self, batch: int = 8):
        """Capture the CUDA graph for ``batch`` images (and its warm-up forward)."""
        img = np.zeros((int(batch), self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(img)
