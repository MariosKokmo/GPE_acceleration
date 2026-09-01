r"""
GPU-friendly binary snapshot writer for BEC simulations.

This replaces per-point text I/O with bulk binary writes. Two backends are
provided so that no extra dependency is mandatory:

1. **PyTorch backend** (always available) — uses ``torch.save`` to write
   ``.pt`` files. One file per simulation run stores all snapshots.
2. **HDF5 backend** (requires ``h5py``) — writes a single ``.h5`` file with
   chunked, optionally compressed datasets.

Both backends support non-blocking GPU-to-CPU transfers (pinned memory with
``non_blocking=True``), appending snapshots incrementally during the simulation
loop, and storing scalar observables (RMS, energies) alongside the field data.

The dataset layout inside a run file is::

  /metadata            -- grid, frequencies, potential, etc.
  /time                -- 1-D array of snapshot timestamps          (shots,)
  /density/column      -- column density on the x-z plane          (shots, n1, n3)
  /density/full        -- full 3D density |psi|^2 (optional)       (shots, n1, n2, n3)
  /psi/real            -- Re(psi) (optional, for restarts)         (shots, n1, n2, n3)
  /psi/imag            -- Im(psi) (optional, for restarts)         (shots, n1, n2, n3)
  /observables/rms     -- RMS radius per snapshot                  (shots,)
  /observables/energy  -- energy breakdown per snapshot            (shots, 4)  [kin, pot, int, total]
  /cross_section       -- cross-section line density               (shots, n1)

Example:
    >>> writer = SnapshotWriter(run_name="my_sim", n1=512, n2=16, n3=512, shots=150)
    >>> # inside loop:
    >>> writer.append(psi, t, uext, rms, energy_dict, cross_line_row)
    >>> # after loop:
    >>> writer.finalise()
"""

import os
import torch
import numpy as np
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Pinned-memory staging helper
# ---------------------------------------------------------------------------

def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    r"""
    Move a GPU tensor to the CPU through pinned memory, for an async transfer.

    Args:
        tensor (torch.Tensor): Tensor to move; one already on the CPU is
            returned as-is.

    Returns:
        torch.Tensor: The tensor on the CPU. The copy is issued as
        non-blocking and the current CUDA stream is synchronised before
        returning, so the result is safe to read immediately.
    """
    if not tensor.is_cuda:
        return tensor
    cpu_tensor = torch.empty(
        tensor.shape, dtype=tensor.dtype,
        pin_memory=True
    )
    cpu_tensor.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return cpu_tensor


# ---------------------------------------------------------------------------
# Torch-based backend (zero extra dependencies)
# ---------------------------------------------------------------------------

class _TorchBuffer:
    r"""
    Accumulate snapshot data in CPU lists and write a single ``.pt`` file.

    Nothing reaches the disk until :meth:`finalise` is called, so the whole run
    is held in host memory. This is the fallback backend, used when ``h5py`` is
    not installed.

    Args:
        path (str): Path of the ``.pt`` file to write.
    """

    def __init__(self, path: str):
        self._path = path
        self._data: Dict[str, list] = {
            "time": [],
            "column_density": [],
            "rms": [],
            "energy": [],
            "cross_section": [],
        }
        self._metadata: Optional[dict] = None
        self._store_full_psi = False

    def set_metadata(self, meta: dict):
        r"""
        Store the run metadata to be embedded in the output file.

        Args:
            meta (dict): Simulation metadata, such as the grid parameters and
                the trap frequencies.
        """
        self._metadata = meta

    def enable_full_psi(self):
        r"""
        Store the full complex wavefunction for every snapshot, as separate
        real and imaginary arrays.
        """
        self._store_full_psi = True
        self._data["psi_real"] = []
        self._data["psi_imag"] = []

    def append(
        self,
        psi: torch.Tensor,
        t: float,
        column_density: torch.Tensor,
        rms: float,
        energy: Dict[str, float],
        cross_section: torch.Tensor,
    ):
        r"""
        Buffer one snapshot in host memory.

        Args:
            psi (torch.Tensor): Complex wavefunction of shape
                ``(n1, n2, n3)``; stored only when the full
                :math:`\psi` has been enabled.
            t (float): Current simulation time.
            column_density (torch.Tensor): Column-integrated density on the
                :math:`(x, z)` plane, shape ``(n1, n3)``.
            rms (float): RMS radius at this snapshot.
            energy (dict): Energy breakdown, with the keys ``e_kin``,
                ``e_pot``, ``e_int`` and ``E_total``; missing keys default to
                zero.
            cross_section (torch.Tensor): Cross-section line density, shape
                ``(n1,)``.
        """
        self._data["time"].append(t)
        self._data["column_density"].append(_to_cpu(column_density).clone())
        self._data["rms"].append(rms)
        self._data["energy"].append([
            energy.get("e_kin", 0.0),
            energy.get("e_pot", 0.0),
            energy.get("e_int", 0.0),
            energy.get("E_total", 0.0),
        ])
        self._data["cross_section"].append(_to_cpu(cross_section).clone())

        if self._store_full_psi:
            cpu_psi = _to_cpu(psi)
            self._data["psi_real"].append(cpu_psi.real.clone())
            self._data["psi_imag"].append(cpu_psi.imag.clone())

    def finalise(self):
        r"""
        Stack the buffered snapshots and write them to the ``.pt`` file.
        """
        payload = {
            "metadata": self._metadata,
            "time": torch.tensor(self._data["time"], dtype=torch.float64),
            "column_density": torch.stack(self._data["column_density"]),
            "rms": torch.tensor(self._data["rms"], dtype=torch.float64),
            "energy": torch.tensor(self._data["energy"], dtype=torch.float64),
            "cross_section": torch.stack(self._data["cross_section"]),
        }
        if self._store_full_psi and self._data.get("psi_real"):
            payload["psi_real"] = torch.stack(self._data["psi_real"])
            payload["psi_imag"] = torch.stack(self._data["psi_imag"])
        torch.save(payload, self._path)


# ---------------------------------------------------------------------------
# HDF5 backend (optional, activated when h5py is installed)
# ---------------------------------------------------------------------------

def _hdf5_available() -> bool:
    r"""
    Report whether the HDF5 backend can be used.

    Returns:
        bool: ``True`` when ``h5py`` is importable.
    """
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


class _HDF5Buffer:
    r"""
    Write snapshots directly into an HDF5 file with chunked datasets.

    Every dataset is pre-allocated to its final size at construction and filled
    in place, one snapshot per call to :meth:`append`, so host memory does not
    grow with the run length. The column density and the wavefunction datasets
    are chunked one snapshot at a time and gzip-compressed.

    Args:
        path (str): Path of the ``.h5`` file to write.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        shots (int): Number of snapshots that will be recorded.
    """

    def __init__(self, path: str, n1: int, n2: int, n3: int, shots: int):
        import h5py
        self._f = h5py.File(path, "w")
        self._idx = 0

        # Pre-allocate datasets
        self._f.create_dataset("time", shape=(shots,), dtype="f8")
        self._f.create_dataset(
            "density/column", shape=(shots, n1, n3), dtype="f8",
            chunks=(1, n1, n3), compression="gzip", compression_opts=1
        )
        self._f.create_dataset("observables/rms", shape=(shots,), dtype="f8")
        self._f.create_dataset("observables/energy", shape=(shots, 4), dtype="f8")
        self._f.create_dataset(
            "cross_section", shape=(shots, n1), dtype="f8",
            chunks=(1, n1)
        )
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._shots = shots
        self._psi_datasets_created = False

    def set_metadata(self, meta: dict):
        r"""
        Write the run metadata as attributes of the ``/metadata`` group.

        Values that HDF5 cannot store natively are saved as their string
        representation.

        Args:
            meta (dict): Simulation metadata, such as the grid parameters and
                the trap frequencies.
        """
        grp = self._f.require_group("metadata")
        for k, v in meta.items():
            try:
                grp.attrs[k] = v
            except TypeError:
                grp.attrs[k] = str(v)

    def enable_full_psi(self):
        r"""
        Create the ``psi/real`` and ``psi/imag`` datasets, so that the full
        complex wavefunction is stored for every snapshot.

        The call is idempotent: the datasets are created only once.
        """
        if not self._psi_datasets_created:
            shape = (self._shots, self._n1, self._n2, self._n3)
            chunks = (1, self._n1, self._n2, self._n3)
            self._f.create_dataset(
                "psi/real", shape=shape, dtype="f8",
                chunks=chunks, compression="gzip", compression_opts=1
            )
            self._f.create_dataset(
                "psi/imag", shape=shape, dtype="f8",
                chunks=chunks, compression="gzip", compression_opts=1
            )
            self._psi_datasets_created = True

    def append(
        self,
        psi: torch.Tensor,
        t: float,
        column_density: torch.Tensor,
        rms: float,
        energy: Dict[str, float],
        cross_section: torch.Tensor,
    ):
        r"""
        Write one snapshot into the pre-allocated datasets.

        Args:
            psi (torch.Tensor): Complex wavefunction of shape
                ``(n1, n2, n3)``; written only when the full :math:`\psi`
                datasets have been created.
            t (float): Current simulation time.
            column_density (torch.Tensor): Column-integrated density on the
                :math:`(x, z)` plane, shape ``(n1, n3)``.
            rms (float): RMS radius at this snapshot.
            energy (dict): Energy breakdown, with the keys ``e_kin``,
                ``e_pot``, ``e_int`` and ``E_total``; missing keys default to
                zero.
            cross_section (torch.Tensor): Cross-section line density, shape
                ``(n1,)``.
        """
        i = self._idx
        self._f["time"][i] = t
        self._f["density/column"][i] = _to_cpu(column_density).numpy()
        self._f["observables/rms"][i] = rms
        self._f["observables/energy"][i] = [
            energy.get("e_kin", 0.0),
            energy.get("e_pot", 0.0),
            energy.get("e_int", 0.0),
            energy.get("E_total", 0.0),
        ]
        self._f["cross_section"][i] = _to_cpu(cross_section).numpy()

        if self._psi_datasets_created:
            cpu_psi = _to_cpu(psi)
            self._f["psi/real"][i] = cpu_psi.real.numpy()
            self._f["psi/imag"][i] = cpu_psi.imag.numpy()

        self._idx += 1

    def finalise(self):
        r"""Flush the buffers and close the HDF5 file."""
        self._f.flush()
        self._f.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SnapshotWriter:
    r"""
    High-level writer that picks the best available backend.

    Args:
        run_name (str): Base name for the output file; the extension is added
            automatically.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        shots (int): Number of snapshots that will be recorded.
        backend (str): ``"auto"`` (default) selects HDF5 if it is available and
            falls back to torch; ``"torch"`` forces the PyTorch ``.pt``
            backend; ``"hdf5"`` forces HDF5.
        store_full_psi (bool): If ``True``, the full complex wavefunction is
            stored per snapshot. Defaults to ``False``, to save disk space.
        metadata (dict): Simulation metadata (grid parameters, frequencies and
            so on) to embed in the file.

    Raises:
        ImportError: If the HDF5 backend is requested but ``h5py`` is not
            installed.
    """

    def __init__(
        self,
        run_name: str,
        n1: int,
        n2: int,
        n3: int,
        shots: int,
        backend: str = "auto",
        store_full_psi: bool = False,
        metadata: Optional[dict] = None,
    ):
        backend = backend.strip().lower()
        if backend == "auto":
            backend = "hdf5" if _hdf5_available() else "torch"

        if backend == "hdf5":
            if not _hdf5_available():
                raise ImportError(
                    "h5py is required for the HDF5 backend. "
                    "Install it with: pip install h5py"
                )
            path = f"{run_name}.h5"
            self._buf = _HDF5Buffer(path, n1, n2, n3, shots)
        else:
            path = f"{run_name}.pt"
            self._buf = _TorchBuffer(path)

        if metadata is not None:
            self._buf.set_metadata(metadata)

        if store_full_psi:
            self._buf.enable_full_psi()

        self._path = path

    @property
    def filepath(self) -> str:
        r"""
        str: Path of the run file, with the extension chosen by the backend.
        """
        return self._path

    def append(
        self,
        psi: torch.Tensor,
        t: float,
        column_density: torch.Tensor,
        rms: float,
        energy: Dict[str, float],
        cross_section: torch.Tensor,
    ):
        r"""
        Record one snapshot.

        Args:
            psi (torch.Tensor): Current complex wavefunction of shape
                ``(n1, n2, n3)``; it may live on the GPU.
            t (float): Current simulation time.
            column_density (torch.Tensor): Column-integrated density on the
                :math:`(x, z)` plane, shape ``(n1, n3)``.
            rms (float): RMS radius at this snapshot.
            energy (dict): Energy breakdown, with the keys ``e_kin``,
                ``e_pot``, ``e_int`` and ``E_total``.
            cross_section (torch.Tensor): Cross-section line density, shape
                ``(n1,)``.
        """
        self._buf.append(psi, t, column_density, rms, energy, cross_section)

    def finalise(self):
        r"""Flush and close the output file."""
        self._buf.finalise()
