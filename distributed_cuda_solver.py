"""Four-rank MPI + CUDA reaction-diffusion stencil laboratory.

Run in a Google Colab NVIDIA T4 runtime:
  mpirun --allow-run-as-root --oversubscribe -np 4 \
      python3 distributed_cuda_solver.py <YOUR_REAL_GITHUB_USERNAME>

Every process owns only a 258 x 258 padded subdomain; field arrays are never
collected to a root process. Requires mpi4py, NumPy, and numba.cuda.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
UP, DOWN, LEFT, RIGHT = range(4)
SEND_TAGS = (120, 121, 122, 123)
OPPOSITE = (DOWN, UP, RIGHT, LEFT)
DIRECTIONS = ("up", "down", "left", "right")


def get_gpu_uuid() -> str:
    """Read the *actual* active GPU hardware identifier, never a placeholder."""
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
        timeout=20,
    ).strip().splitlines()[0].strip()
    if not value or not value.startswith("GPU-"):
        raise RuntimeError(f"Unexpected nvidia-smi GPU UUID: {value!r}")
    return value


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Match the SHA-256 username::GPU-UUID seed rule in the PDF starter."""
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


@dataclass
class DistributedContext:
    cart: object
    rank: int
    size: int
    coords: tuple[int, int]
    neighbors: tuple[int, int, int, int]
    seed: int
    gpu_uuid: str
    github_user: str
    ny: int
    nx: int


def init_distributed_context(github_user: str) -> DistributedContext:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly four MPI ranks (2 x 2); got {size}")
    if not github_user.strip() or github_user.startswith("REPLACE_"):
        raise ValueError("Provide your actual GitHub handle as argument 1")
    cuda.select_device(0)
    cart = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    rank = cart.Get_rank()
    coords = tuple(int(v) for v in cart.Get_coords(rank))
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    uuid = get_gpu_uuid() if rank == 0 else None
    uuid = cart.bcast(uuid, root=0)
    seed = derive_student_seed(github_user, uuid)
    return DistributedContext(
        cart, rank, size, coords, (up, down, left, right),
        seed, uuid, github_user, GLOBAL_NY // 2, GLOBAL_NX // 2,
    )


def initialize_subdomains(ctx: DistributedContext):
    """Construct seeded physics using rank-local, including ghost, coordinates."""
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    row_rank, col_rank = ctx.coords
    # Local padded index 0 maps to the preceding global boundary/neighbour.
    # Owned index 1 maps to the rank's first global interior cell.
    x = (col_rank * ctx.nx + np.arange(ctx.nx + 2)) * dx
    y = (row_rank * ctx.ny + np.arange(ctx.ny + 2)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    kappa = 0.08 + 0.16 * (
        np.sin(2 * np.pi * xx) ** 2 * np.cos(2 * np.pi * yy) ** 2
    )
    rng = np.random.default_rng(ctx.seed)
    centers = rng.uniform(0.28, 0.72, size=(2, 2))
    amplitudes = rng.uniform(0.7, 1.1, size=2)
    widths = (0.075, 0.09)

    u = np.zeros((ctx.ny + 2, ctx.nx + 2), dtype=np.float64)
    for (xc, yc), amp, width in zip(centers, amplitudes, widths):
        u[1:-1, 1:-1] += amp * np.exp(
            -((xx[1:-1, 1:-1] - xc) ** 2 +
              (yy[1:-1, 1:-1] - yc) ** 2) / (2.0 * width**2)
        )

    # Positive, compact forcing; cubic damping dominates the global integral
    # over the short 400-step experiment. Exact energy behavior is still
    # measured and reported rather than assumed.
    source = np.zeros_like(u)
    source[1:-1, 1:-1] = 0.01 * np.exp(
        -((xx[1:-1, 1:-1] - 0.5) ** 2 +
          (yy[1:-1, 1:-1] - 0.5) ** 2) / (2.0 * 0.035**2)
    )
    return u, np.ascontiguousarray(kappa), source, dx, dy


# TASK 3: each CUDA lane packs four independent boundary-vector elements.
@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right,
                      ny, nx):
    t = cuda.grid(1)
    if t < nx:
        send_up[t] = u[1, t + 1]
        send_down[t] = u[ny, t + 1]
    if t < ny:
        send_left[t] = u[t + 1, 1]
        send_right[t] = u[t + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right,
                        ny, nx, has_up, has_down, has_left, has_right):
    t = cuda.grid(1)
    if t < nx:
        u[0, t + 1] = recv_up[t] if has_up else 0.0
        u[ny + 1, t + 1] = recv_down[t] if has_down else 0.0
    if t < ny:
        u[t + 1, 0] = recv_left[t] if has_left else 0.0
        u[t + 1, nx + 1] = recv_right[t] if has_right else 0.0


# TASK 4: 16x16 CUDA threads, precisely 18x18 shared-memory tile.
@cuda.jit
def tiled_stencil_kernel(u, kappa, source, u_next, dt_step, inv_dx2,
                         inv_dy2, r_start, r_end, c_start, c_end):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    r = r_start + cuda.blockIdx.y * TILE_DIM + ty
    c = c_start + cuda.blockIdx.x * TILE_DIM + tx
    last_r = u.shape[0] - 1
    last_c = u.shape[1] - 1

    rr = r
    cc = c
    if rr > last_r:
        rr = last_r
    if cc > last_c:
        cc = last_c

    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    smem[ty + 1, tx + 1] = u[rr, cc]
    if tx == 0:
        cleft = c - 1
        if cleft < 0:
            cleft = 0
        smem[ty + 1, 0] = u[rr, cleft]
    if tx == TILE_DIM - 1:
        cright = c + 1
        if cright > last_c:
            cright = last_c
        smem[ty + 1, SHARED_DIM - 1] = u[rr, cright]
    if ty == 0:
        rtop = r - 1
        if rtop < 0:
            rtop = 0
        smem[0, tx + 1] = u[rtop, cc]
    if ty == TILE_DIM - 1:
        rbottom = r + 1
        if rbottom > last_r:
            rbottom = last_r
        smem[SHARED_DIM - 1, tx + 1] = u[rbottom, cc]
    cuda.syncthreads()  # all threads, even the inactive edge threads

    if r < r_end and c < c_end:
        center = smem[ty + 1, tx + 1]
        north = smem[ty, tx + 1]
        south = smem[ty + 2, tx + 1]
        west = smem[ty + 1, tx]
        east = smem[ty + 1, tx + 2]
        kc = kappa[r, c]
        kn = 0.5 * (kc + kappa[r - 1, c])
        ks = 0.5 * (kc + kappa[r + 1, c])
        kw = 0.5 * (kc + kappa[r, c - 1])
        ke = 0.5 * (kc + kappa[r, c + 1])
        flux_x = (ke * (east - center) - kw * (center - west)) * inv_dx2
        flux_y = (ks * (south - center) - kn * (center - north)) * inv_dy2
        u_next[r, c] = center + dt_step * (
            flux_x + flux_y - LAMBDA_DAMP * center**3 + source[r, c]
        )


# Device-only blockwise reduction, no global-array gathering onto rank zero.
@cuda.jit
def diagnostics_kernel(u, prev, part_energy, part_residual, ny, nx):
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    scratch_energy = cuda.shared.array(shape=256, dtype=float64)
    scratch_residual = cuda.shared.array(shape=256, dtype=float64)
    energy = 0.0
    residual_sq = 0.0
    if idx < ny * nx:
        r = idx // nx + 1
        c = idx % nx + 1
        energy = u[r, c]
        difference = u[r, c] - prev[r, c]
        residual_sq = difference * difference
    scratch_energy[tid] = energy
    scratch_residual[tid] = residual_sq
    cuda.syncthreads()
    stride = 128
    while stride >= 1:
        if tid < stride:
            scratch_energy[tid] += scratch_energy[tid + stride]
            scratch_residual[tid] += scratch_residual[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        part_energy[cuda.blockIdx.x] = scratch_energy[0]
        part_residual[cuda.blockIdx.x] = scratch_residual[0]


class HaloBuffers:
    def __init__(self, ny: int, nx: int):
        self.send_device = []
        self.recv_device = []
        self.send_host = []
        self.recv_host = []
        for n in (nx, nx, ny, ny):
            self.send_device.append(cuda.device_array(n, dtype=np.float64))
            self.recv_device.append(cuda.to_device(np.zeros(n, dtype=np.float64)))
            self.send_host.append(cuda.pinned_array(n, dtype=np.float64))
            self.recv_host.append(cuda.pinned_array(n, dtype=np.float64))


def exchange_halos_async(ctx: DistributedContext, buffers: HaloBuffers):
    """Receive opposite-direction tags and send the local-direction tags."""
    requests = []
    for d, nbr in enumerate(ctx.neighbors):
        if nbr != MPI.PROC_NULL:
            requests.append(ctx.cart.Irecv(
                buffers.recv_host[d], source=nbr, tag=SEND_TAGS[OPPOSITE[d]]
            ))
    for d, nbr in enumerate(ctx.neighbors):
        if nbr != MPI.PROC_NULL:
            requests.append(ctx.cart.Isend(
                buffers.send_host[d], dest=nbr, tag=SEND_TAGS[d]
            ))
    return requests


def launch_box(u, kappa, source, u_next, dt_step, dx, dy,
               r0, r1, c0, c1, stream):
    if r0 >= r1 or c0 >= c1:
        return
    grid = (
        ((c1 - c0 + TILE_DIM - 1) // TILE_DIM),
        ((r1 - r0 + TILE_DIM - 1) // TILE_DIM),
    )
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        u, kappa, source, u_next, dt_step,
        1.0 / (dx * dx), 1.0 / (dy * dy), r0, r1, c0, c1,
    )


def step_overlapped(ctx, u, u_next, kappa, source, dt_step, dx, dy,
                    halo, stream_boundary, stream_interior,
                    overlap: bool, sample_events: bool = False):
    """A(pack/D2H) -> B(MPI+interior) -> C(H2D/unpack+boundary).

    Serialized baseline deliberately waits for MPI completion before
    launching the same interior tiled kernel. Mathematics/order are identical.
    """
    n = max(ctx.nx, ctx.ny)
    pack_halos_kernel[(n + 255) // 256, 256, stream_boundary](
        u, *halo.send_device, ctx.ny, ctx.nx,
    )
    for d, nbr in enumerate(ctx.neighbors):
        if nbr != MPI.PROC_NULL:
            halo.send_device[d].copy_to_host(
                halo.send_host[d], stream=stream_boundary
            )
    stream_boundary.synchronize()  # MPI may not touch a D2H buffer until ready
    requests = exchange_halos_async(ctx, halo)

    events = None
    if sample_events:
        events = [cuda.event(timing=True) for _ in range(4)]
    if not overlap:
        MPI.Request.Waitall(requests)

    if events is not None:
        events[0].record(stream_interior)
    launch_box(u, kappa, source, u_next, dt_step, dx, dy,
               2, ctx.ny, 2, ctx.nx, stream_interior)
    if events is not None:
        events[1].record(stream_interior)
    if overlap:
        MPI.Request.Waitall(requests)
    else:
        # The baseline does not overlap GPU interior work with H2D/ring work.
        stream_interior.synchronize()

    # Interprocess writes are now finished. Stage received buffers on the GPU.
    for d, nbr in enumerate(ctx.neighbors):
        if nbr != MPI.PROC_NULL:
            halo.recv_device[d].copy_to_device(
                halo.recv_host[d], stream=stream_boundary
            )
    flags = tuple(int(nbr != MPI.PROC_NULL) for nbr in ctx.neighbors)
    unpack_halos_kernel[(n + 255) // 256, 256, stream_boundary](
        u, *halo.recv_device, ctx.ny, ctx.nx, *flags,
    )

    if events is not None:
        events[2].record(stream_boundary)
    # Four disjoint pieces of the one-cell perimeter ring.
    launch_box(u, kappa, source, u_next, dt_step, dx, dy,
               1, 2, 1, ctx.nx + 1, stream_boundary)
    launch_box(u, kappa, source, u_next, dt_step, dx, dy,
               ctx.ny, ctx.ny + 1, 1, ctx.nx + 1, stream_boundary)
    launch_box(u, kappa, source, u_next, dt_step, dx, dy,
               2, ctx.ny, 1, 2, stream_boundary)
    launch_box(u, kappa, source, u_next, dt_step, dx, dy,
               2, ctx.ny, ctx.nx, ctx.nx + 1, stream_boundary)
    if events is not None:
        events[3].record(stream_boundary)

    stream_interior.synchronize()
    stream_boundary.synchronize()
    if events is None:
        return None
    return {
        "interior_kernel_ms": float(cuda.event_elapsed_time(events[0], events[1])),
        "boundary_ring_ms": float(cuda.event_elapsed_time(events[2], events[3])),
    }


def distributed_diagnostics(ctx, u, prev, d_e, d_r, dx, dy):
    total_cells = ctx.nx * ctx.ny
    blocks = (total_cells + 255) // 256
    diagnostics_kernel[blocks, 256](u, prev, d_e, d_r, ctx.ny, ctx.nx)
    cuda.synchronize()
    local = np.asarray(
        [np.sum(d_e.copy_to_host()), np.sum(d_r.copy_to_host())],
        dtype=np.float64,
    ) * (dx * dy)
    global_stats = np.empty_like(local)
    ctx.cart.Allreduce(local, global_stats, op=MPI.SUM)
    return float(global_stats[0]), math.sqrt(max(0.0, float(global_stats[1])))


def audit_ptx():
    signatures = tiled_stencil_kernel.signatures
    if not signatures:
        raise RuntimeError("Stencil kernel has not been compiled on a real GPU")
    asm = tiled_stencil_kernel.inspect_asm(signatures[0])
    if isinstance(asm, dict):
        asm = next(iter(asm.values()))
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", asm)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", asm)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", asm)),
        "st.shared.any": len(re.findall(r"\bst\.shared\.", asm)),
        "ld.shared.any": len(re.findall(r"\bld\.shared\.", asm)),
    }
    counts["ptx_audit_passed"] = (
        counts["st.shared.any"] > 0 and
        counts["ld.shared.any"] > 0 and counts["bar.sync"] > 0
    )
    return counts


def run_mode(ctx, u0, kappa_dev, source_dev, dt_step, dx, dy,
             halo, stream_boundary, stream_interior, overlap):
    current = cuda.to_device(u0)
    scratch = cuda.device_array(u0.shape, dtype=np.float64)
    d_e = cuda.device_array((ctx.ny * ctx.nx + 255) // 256, dtype=np.float64)
    d_r = cuda.device_array_like(d_e)
    mode_name = "overlapped" if overlap else "serialized"
    energies = {}
    residuals = {}
    event_samples = []
    energy0, residual0 = distributed_diagnostics(
        ctx, current, current, d_e, d_r, dx, dy
    )
    energies["0"] = energy0
    residuals["0"] = residual0
    if ctx.rank == 0:
        print(f"{mode_name} step=0 energy={energy0:.16e} l2_residual={residual0:.16e}",
              flush=True)
    ctx.cart.Barrier()
    started = MPI.Wtime()
    previous = current
    for t in range(1, STEPS + 1):
        sample = t % 100 == 0
        gpu_events = step_overlapped(
            ctx, current, scratch, kappa_dev, source_dev, dt_step, dx, dy,
            halo, stream_boundary, stream_interior,
            overlap=overlap, sample_events=sample,
        )
        previous, current, scratch = current, scratch, current
        if sample:
            energy, residual = distributed_diagnostics(
                ctx, current, previous, d_e, d_r, dx, dy
            )
            energies[str(t)] = energy
            residuals[str(t)] = residual
            if gpu_events:
                event_samples.append(gpu_events)
            if ctx.rank == 0:
                print(f"{mode_name} step={t} energy={energy:.16e} "
                      f"l2_residual={residual:.16e}", flush=True)
    ctx.cart.Barrier()
    local_ms = 1000.0 * (MPI.Wtime() - started)
    send_time = np.asarray([local_ms], dtype=np.float64)
    global_max_time = np.empty_like(send_time)
    ctx.cart.Allreduce(send_time, global_max_time, op=MPI.MAX)

    # Event samples are per-context and can differ across ranks. Report max.
    if event_samples:
        local_event_means = np.array([
            np.mean([e[k] for e in event_samples])
            for k in ("interior_kernel_ms", "boundary_ring_ms")
        ], dtype=np.float64)
        global_event_means = np.empty_like(local_event_means)
        ctx.cart.Allreduce(local_event_means, global_event_means, op=MPI.MAX)
    else:
        global_event_means = [0.0, 0.0]
    return current, {
        "wall_ms": float(global_max_time[0]),
        "energy_trace": energies,
        "l2_residual_trace": residuals,
        "cuda_event_ms": {
            "interior_kernel": float(global_event_means[0]),
            "boundary_ring": float(global_event_means[1]),
            "sample_count": len(event_samples),
        },
    }


def run_solver(ctx: DistributedContext):
    u0, kappa, source, dx, dy = initialize_subdomains(ctx)
    kappa_local_max = np.array([np.max(kappa[1:-1, 1:-1])], dtype=np.float64)
    kappa_global_max = np.empty_like(kappa_local_max)
    ctx.cart.Allreduce(kappa_local_max, kappa_global_max, op=MPI.MAX)
    kappa_max = float(kappa_global_max[0])
    dt_max = dx * dx / (4.0 * kappa_max)
    dt_step = 0.8 * dt_max
    kappa_dev = cuda.to_device(kappa)
    source_dev = cuda.to_device(source)
    halo = HaloBuffers(ctx.ny, ctx.nx)
    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()
    if ctx.rank == 0:
        print(f"Identity={ctx.github_user} GPU_UUID={ctx.gpu_uuid} "
              f"INSTANCE_SEED={ctx.seed}", flush=True)
        print(f"mpi_ranks={ctx.size} dims=[2,2] local_shape="
              f"({ctx.ny + 2},{ctx.nx + 2}) "
              f"max_kappa={kappa_max:.16e} dx={dx:.16e} "
              f"dt_max={dt_max:.16e} dt={dt_step:.16e}", flush=True)
    print(f"rank={ctx.rank} coords={ctx.coords} neighbors={ctx.neighbors}",
          flush=True)

    # Compile all kernels and start the GPU contexts before comparing modes.
    warm_u = cuda.to_device(u0)
    warm_out = cuda.device_array_like(warm_u)
    step_overlapped(ctx, warm_u, warm_out, kappa_dev, source_dev,
                    dt_step, dx, dy, halo, stream_boundary, stream_interior,
                    overlap=True)
    warm_e = cuda.device_array((ctx.ny * ctx.nx + 255) // 256,
                               dtype=np.float64)
    warm_r = cuda.device_array_like(warm_e)
    distributed_diagnostics(ctx, warm_out, warm_u, warm_e, warm_r, dx, dy)
    ctx.cart.Barrier()

    final_serialized, report_serialized = run_mode(
        ctx, u0, kappa_dev, source_dev, dt_step, dx, dy, halo,
        stream_boundary, stream_interior, overlap=False,
    )
    final_overlap, report_overlap = run_mode(
        ctx, u0, kappa_dev, source_dev, dt_step, dx, dy, halo,
        stream_boundary, stream_interior, overlap=True,
    )

    # Each rank compares ONLY its own region. Allreduce returns one scalar.
    local_diff = float(np.max(np.abs(
        final_serialized.copy_to_host()[1:-1, 1:-1]
        - final_overlap.copy_to_host()[1:-1, 1:-1]
    )))
    local_diff_array = np.array([local_diff], dtype=np.float64)
    global_diff_array = np.empty_like(local_diff_array)
    ctx.cart.Allreduce(local_diff_array, global_diff_array, op=MPI.MAX)
    parity = float(global_diff_array[0])

    # CUDA PTX is architecture and compiler-version dependent. Audit the
    # *actual* JIT output from this run, never hardcode instruction counts.
    ptx = audit_ptx()
    local_ptx_ok = np.array([int(ptx["ptx_audit_passed"])], dtype=np.int32)
    all_ranks_ptx_ok = np.empty_like(local_ptx_ok)
    ctx.cart.Allreduce(local_ptx_ok, all_ranks_ptx_ok, op=MPI.MIN)
    global_ptx_passed = bool(all_ranks_ptx_ok[0])
    if ctx.rank == 0:
        trace = [report_overlap["energy_trace"][str(i)]
                 for i in (0, 100, 200, 300, 400)]
        energy_monotonic = all(
            trace[i + 1] <= trace[i] + 1e-15
            for i in range(len(trace) - 1)
        )
        manifest = {
            "github_username": ctx.github_user,
            "gpu_uuid": ctx.gpu_uuid,
            "instance_seed": ctx.seed,
            "mpi_ranks": 4,
            "cart_dims": [2, 2],
            "cart_periods": [False, False],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [ctx.ny + 2, ctx.nx + 2],
            "steps": STEPS,
            "lambda_damping": LAMBDA_DAMP,
            "max_diffusivity": kappa_max,
            "dx": dx,
            "dy": dy,
            "dt_max": dt_max,
            "dt": dt_step,
            "serialized_ms": report_serialized["wall_ms"],
            "overlapped_ms": report_overlap["wall_ms"],
            "overlap_speedup": report_serialized["wall_ms"] /
                               report_overlap["wall_ms"],
            "parity_max_diff": parity,
            "parity_passed": parity < 1e-12,
            "ptx_audit_passed": global_ptx_passed,
            "ptx_instruction_counts": ptx,
            "cuda_event_ms": {
                "serialized": report_serialized["cuda_event_ms"],
                "overlapped": report_overlap["cuda_event_ms"],
            },
            "energy_trace": {
                "serialized": report_serialized["energy_trace"],
                "overlapped": report_overlap["energy_trace"],
            },
            "l2_residual_trace": {
                "serialized": report_serialized["l2_residual_trace"],
                "overlapped": report_overlap["l2_residual_trace"],
            },
            "energy_monotonic_nonincreasing": energy_monotonic,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        Path("submission_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("PTX_AUDIT=" + json.dumps(ptx, sort_keys=True), flush=True)
        print(f"serialized_ms={report_serialized['wall_ms']:.6f} "
              f"overlapped_ms={report_overlap['wall_ms']:.6f} "
              f"speedup={manifest['overlap_speedup']:.6f}", flush=True)
        print(f"parity_max_diff={parity:.16e} "
              f"parity_passed={manifest['parity_passed']} "
              f"energy_monotonic={energy_monotonic}", flush=True)
        print("Wrote submission_manifest.json", flush=True)
    ctx.cart.Barrier()
    if not global_ptx_passed or parity >= 1e-12:
        raise RuntimeError("Parity or PTX audit did not pass. Inspect manifest.")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 distributed_cuda_solver.py GITHUB_USERNAME")
    ctx = init_distributed_context(sys.argv[1])
    run_solver(ctx)


if __name__ == "__main__":
    main()
