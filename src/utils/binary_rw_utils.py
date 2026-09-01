r"""
Binary read/write utilities backed by :class:`~src.utils.snapshot_writer.SnapshotWriter`.

This is a drop-in replacement for :mod:`~src.utils.read_write_utils` that
stores simulation data in a single binary file (PyTorch ``.pt`` or HDF5
``.h5``) instead of many small text files. All GPU-to-CPU transfers use pinned
memory for non-blocking copies.

The visualisation helpers operate directly on tensors or on the binary run
file, so no intermediate text is needed for phase figures, RMS plots or
cross-section plots.

Example:
    Recording, inside the simulation loop:

    >>> from src.utils import binary_rw_utils as brw
    >>>
    >>> # once, before the loop
    >>> recorder = brw.SimulationRecorder(
    ...     run_name="my_sim", n1=512, n2=16, n3=512,
    ...     shots=150, a_ho=1.2e-6,
    ...     x1=x1_tensor, x3=x3_tensor,
    ... )
    >>>
    >>> # each snapshot
    >>> recorder.record_snapshot(psi, t, rms, energy_dict, cross_line_row)
    >>>
    >>> # after the loop
    >>> recorder.finalise()

    Reading back:

    >>> reader = brw.SimulationReader("my_sim.pt")   # or .h5
    >>> times      = reader.times()
    >>> col_dens   = reader.column_density()          # (shots, n1, n3)
    >>> rms_arr    = reader.rms()
    >>> energies   = reader.energies()                # (shots, 4)
    >>> psi_real   = reader.psi_real()                # if stored

    Visualisation, standalone from the binary file:

    >>> brw.plot_rms_from_file("my_sim.pt")
    >>> brw.plot_cross_section_from_file("my_sim.pt")
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union

from src.utils.snapshot_writer import SnapshotWriter, _to_cpu


# ============================================================================
# Recording (replaces write_data / write_psi / write_rms / write_energy_terms)
# ============================================================================

class SimulationRecorder:
    r"""
    Accumulate per-snapshot data through
    :class:`~src.utils.snapshot_writer.SnapshotWriter`.

    It provides the same observables that :mod:`~src.utils.read_write_utils`
    wrote to text files, but keeps them in one binary run file.

    Args:
        run_name (str): Base file name; the extension is chosen automatically
            by the backend.
        n1 (int): Number of grid points along the first axis.
        n2 (int): Number of grid points along the second axis.
        n3 (int): Number of grid points along the third axis.
        shots (int): Expected number of snapshots.
        a_ho (float): Harmonic-oscillator length, used for the unit conversion
            when reading back.
        x1 (torch.Tensor): 1-D spatial axis along :math:`x`, in dimensionless
            units. Stored in the metadata so that readers can reconstruct the
            physical coordinates.
        x3 (torch.Tensor): 1-D spatial axis along :math:`z`, stored the same
            way.
        backend (str): ``"auto"``, ``"torch"`` or ``"hdf5"``; see
            :class:`~src.utils.snapshot_writer.SnapshotWriter`.
        store_full_psi (bool): If ``True``, the full complex wavefunction is
            saved every snapshot.
        metadata (dict): Extra metadata to embed, such as the grid parameters
            and the trap frequencies.
    """

    def __init__(
        self,
        run_name: str,
        n1: int,
        n2: int,
        n3: int,
        shots: int,
        a_ho: float = 1.0,
        x1: Optional[torch.Tensor] = None,
        x3: Optional[torch.Tensor] = None,
        backend: str = "auto",
        store_full_psi: bool = False,
        metadata: Optional[dict] = None,
    ):
        meta = dict(metadata) if metadata else {}
        meta["a_ho"] = a_ho
        if x1 is not None:
            meta["x1"] = _to_cpu(x1).tolist() if isinstance(x1, torch.Tensor) else list(x1)
        if x3 is not None:
            meta["x3"] = _to_cpu(x3).tolist() if isinstance(x3, torch.Tensor) else list(x3)
        meta["n1"] = n1
        meta["n2"] = n2
        meta["n3"] = n3

        self._writer = SnapshotWriter(
            run_name=run_name,
            n1=n1, n2=n2, n3=n3,
            shots=shots,
            backend=backend,
            store_full_psi=store_full_psi,
            metadata=meta,
        )
        self._n2 = n2

    @property
    def filepath(self) -> str:
        r"""str: Path of the run file being written."""
        return self._writer.filepath

    def record_snapshot(
        self,
        psi: torch.Tensor,
        t: float,
        rms: float,
        energy: Dict[str, float],
        cross_section: torch.Tensor,
    ):
        r"""
        Record a single snapshot.

        The column density
        :math:`n(x, z) = \sum_j \lvert \psi[i, j, k] \rvert^{2}` is computed
        here, so the caller does not have to.

        Args:
            psi (torch.Tensor): Complex wavefunction of shape
                ``(n1, n2, n3)``.
            t (float): Current simulation time.
            rms (float): RMS radius at this snapshot.
            energy (dict): Energy breakdown, with the keys ``e_kin``,
                ``e_pot``, ``e_int`` and ``E_total``.
            cross_section (torch.Tensor): Cross-section line density, shape
                ``(n1,)``.
        """
        column_density = torch.sum(torch.abs(psi) ** 2, dim=1)  # (n1, n3)
        self._writer.append(psi, t, column_density, rms, energy, cross_section)

    def finalise(self):
        r"""Flush and close the underlying run file."""
        self._writer.finalise()


# ============================================================================
# Reading back (replaces text-file parsing)
# ============================================================================

class SimulationReader:
    r"""
    Read a binary run file produced by :class:`SimulationRecorder`.

    Both ``.pt`` (torch) and ``.h5`` (HDF5) files are accepted; the backend is
    chosen from the extension. A torch file is loaded eagerly into memory,
    while an HDF5 file stays open until :meth:`close` is called.

    Args:
        path (str): Path of the run file to read.

    Raises:
        ValueError: If the file extension is neither ``.pt`` nor ``.h5`` /
            ``.hdf5``.
    """

    def __init__(self, path: str):
        self._path = path
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pt":
            self._backend = "torch"
            self._data = torch.load(path, map_location="cpu", weights_only=False)
        elif ext in (".h5", ".hdf5"):
            self._backend = "hdf5"
            import h5py
            self._f = h5py.File(path, "r")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def close(self):
        r"""Close the file, for the HDF5 backend; a no-op for torch files."""
        if self._backend == "hdf5":
            self._f.close()

    # --- field accessors ---------------------------------------------------

    def times(self) -> np.ndarray:
        r"""
        Return the snapshot timestamps.

        Returns:
            numpy.ndarray: Times of shape ``(shots,)``.
        """
        if self._backend == "torch":
            return self._data["time"].numpy()
        return self._f["time"][:]

    def column_density(self) -> np.ndarray:
        r"""
        Return the column density of every snapshot.

        Returns:
            numpy.ndarray: Column densities of shape ``(shots, n1, n3)``.
        """
        if self._backend == "torch":
            return self._data["column_density"].numpy()
        return self._f["density/column"][:]

    def rms(self) -> np.ndarray:
        r"""
        Return the RMS radius of every snapshot.

        Returns:
            numpy.ndarray: RMS radii of shape ``(shots,)``.
        """
        if self._backend == "torch":
            return self._data["rms"].numpy()
        return self._f["observables/rms"][:]

    def energies(self) -> np.ndarray:
        r"""
        Return the energy breakdown of every snapshot.

        Returns:
            numpy.ndarray: Energies of shape ``(shots, 4)``, the columns being
            ``[e_kin, e_pot, e_int, E_total]``.
        """
        if self._backend == "torch":
            return self._data["energy"].numpy()
        return self._f["observables/energy"][:]

    def cross_section(self) -> np.ndarray:
        r"""
        Return the cross-section line density of every snapshot.

        Returns:
            numpy.ndarray: Line densities of shape ``(shots, n1)``.
        """
        if self._backend == "torch":
            return self._data["cross_section"].numpy()
        return self._f["cross_section"][:]

    def psi_real(self) -> Optional[np.ndarray]:
        r"""
        Return the real part of the stored wavefunction.

        Returns:
            numpy.ndarray or None: :math:`\operatorname{Re} \psi` of shape
            ``(shots, n1, n2, n3)``, or ``None`` when the run did not store the
            full wavefunction.
        """
        if self._backend == "torch":
            t = self._data.get("psi_real")
            return t.numpy() if t is not None else None
        if "psi/real" in self._f:
            return self._f["psi/real"][:]
        return None

    def psi_imag(self) -> Optional[np.ndarray]:
        r"""
        Return the imaginary part of the stored wavefunction.

        Returns:
            numpy.ndarray or None: :math:`\operatorname{Im} \psi` of shape
            ``(shots, n1, n2, n3)``, or ``None`` when the run did not store the
            full wavefunction.
        """
        if self._backend == "torch":
            t = self._data.get("psi_imag")
            return t.numpy() if t is not None else None
        if "psi/imag" in self._f:
            return self._f["psi/imag"][:]
        return None

    def metadata(self) -> dict:
        r"""
        Return the metadata embedded in the run file.

        Returns:
            dict: The simulation metadata, or an empty dict when the file
            carries none.
        """
        if self._backend == "torch":
            return self._data.get("metadata", {})
        meta = {}
        if "metadata" in self._f:
            for k, v in self._f["metadata"].attrs.items():
                meta[k] = v
        return meta


# ============================================================================
# Visualisation helpers (work on tensors or binary files)
# ============================================================================

def save_figure_phase(phase: torch.Tensor, frame: int):
    r"""
    Save an image of the phase at the :math:`y` midplane.

    The behaviour matches
    :func:`~src.utils.read_write_utils.save_figure_phase`, except that the
    figure is closed afterwards.

    Args:
        phase (torch.Tensor): Phase array of shape ``(n1, n2, n3)``; a complex
            array is reduced to its real part.
        frame (int): Snapshot index used in the title and the file name.
    """
    n1, n2, n3 = phase.shape
    slice_data = phase[:, n2 // 2, :]
    if phase.dtype == torch.cdouble:
        plt.imshow(slice_data.cpu().real, cmap="jet")
    else:
        plt.imshow(slice_data.cpu(), cmap="jet")
    cb = plt.colorbar()
    plt.title(f"Phase t = {frame}")
    plt.savefig(f"phase_t_{frame}.png")
    cb.remove()
    plt.close()


def plot_rms(rms_values: np.ndarray, times: Optional[np.ndarray] = None,
             title: str = "RMS", save_path: str = "RMS.png"):
    r"""
    Plot the RMS radius from arrays, with no intermediate text file.

    Args:
        rms_values (numpy.ndarray): RMS radius per snapshot.
        times (numpy.ndarray): Snapshot times. Defaults to the snapshot index.
        title (str): Title of the figure.
        save_path (str): Path of the PNG to write.
    """
    plt.figure()
    x = times if times is not None else np.arange(len(rms_values))
    plt.plot(x, rms_values)
    plt.title(title)
    plt.ylabel("RMS")
    plt.xlabel("time")
    plt.savefig(save_path)
    plt.close()


def plot_rms_from_file(run_file: str, save_path: str = "RMS.png"):
    r"""
    Read a binary run file and plot the RMS radius.

    Args:
        run_file (str): Path of the ``.pt`` or ``.h5`` run file.
        save_path (str): Path of the PNG to write.
    """
    reader = SimulationReader(run_file)
    plot_rms(reader.rms(), reader.times(), save_path=save_path)
    reader.close()


def plot_cross_section(cross_line_data: np.ndarray,
                       save_path: str = "cross_section_line.png"):
    r"""
    Save a 3-D waterfall plot of the cross-section line density.

    Args:
        cross_line_data (numpy.ndarray): Line densities of shape
            ``(shots, dim)``, one row per snapshot.
        save_path (str): Path of the PNG to write.
    """
    shots, dim = cross_line_data.shape
    ax = plt.figure(figsize=(12, 16)).add_subplot(projection="3d")
    y = np.arange(dim)
    for t_idx in range(shots):
        x = np.ones(dim) * t_idx
        ax.plot(x, y, cross_line_data[t_idx, :])
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    ax.set_zlabel("density")
    plt.savefig(save_path)
    plt.close()


def plot_cross_section_from_file(run_file: str,
                                 save_path: str = "cross_section_line.png"):
    r"""
    Read a binary run file and plot the cross-section line density.

    Args:
        run_file (str): Path of the ``.pt`` or ``.h5`` run file.
        save_path (str): Path of the PNG to write.
    """
    reader = SimulationReader(run_file)
    plot_cross_section(reader.cross_section(), save_path=save_path)
    reader.close()


def plot_energies(energies: np.ndarray, times: Optional[np.ndarray] = None,
                  save_path: str = "energies.png"):
    r"""
    Plot the energy breakdown from a ``(shots, 4)`` array.

    One curve is drawn per column, labelled ``E_kin``, ``E_pot``, ``E_int`` and
    ``E_total``.

    Args:
        energies (numpy.ndarray): Energies of shape ``(shots, 4)``.
        times (numpy.ndarray): Snapshot times. Defaults to the snapshot index.
        save_path (str): Path of the PNG to write.
    """
    plt.figure()
    x = times if times is not None else np.arange(energies.shape[0])
    labels = ["E_kin", "E_pot", "E_int", "E_total"]
    for col, label in enumerate(labels):
        plt.plot(x, energies[:, col], label=label)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.savefig(save_path)
    plt.close()


def plot_energies_from_file(run_file: str, save_path: str = "energies.png"):
    r"""
    Read a binary run file and plot the energy breakdown.

    Args:
        run_file (str): Path of the ``.pt`` or ``.h5`` run file.
        save_path (str): Path of the PNG to write.
    """
    reader = SimulationReader(run_file)
    plot_energies(reader.energies(), reader.times(), save_path=save_path)
    reader.close()


# ============================================================================
# Legacy-compatible free functions
# ============================================================================
# These mirror the signatures in ``read_write_utils`` so they can be used as
# a drop-in when only swapping the import.  They delegate to the helpers above.

def write_rms(rms_meas: dict, simulation_name: str):
    r"""
    Write the RMS measurements to a text file (legacy-compatible).

    Args:
        rms_meas (dict): RMS radius per snapshot, keyed by time or snapshot
            index.
        simulation_name (str): Base name of the simulation, used for the file
            name.
    """
    with open(f"{simulation_name}_RMS_meas.txt", "w") as f:
        f.write("t\tr\n")
        for t, r in rms_meas.items():
            f.write(f"{t}\t{r}\n")


def save_rms_figure(title: str):
    r"""
    Read a tab-delimited RMS text file and save a plot (legacy-compatible).

    Args:
        title (str): Path of the file to read; its stem is reused for the title
            and the output file name.
    """
    data = np.loadtxt(title, skiprows=1, delimiter="\t")
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.title(title[:-4])
    plt.ylabel("RMS")
    plt.xlabel("time")
    plt.savefig(f"RMS_{title[:-3]}.png")
    plt.close()


def save_cross_section_line_figure(cross_line_data: torch.Tensor):
    r"""
    Save a 3-D cross-section plot from a tensor (legacy-compatible).

    Args:
        cross_line_data (torch.Tensor): Line densities of shape
            ``(shots, dim)``; it must live on the CPU.
    """
    plot_cross_section(cross_line_data.numpy())


def save_tensor_to_csv(tensor: torch.Tensor, filename: str):
    r"""
    Save a tensor to CSV, without an index or a header (legacy-compatible).

    Args:
        tensor (torch.Tensor): Tensor to save; it must live on the CPU.
        filename (str): Destination path.
    """
    import pandas as pd
    df = pd.DataFrame(tensor.numpy())
    df.to_csv(filename, index=False, header=None)


def write_energy_terms(energies: list, filename: str):
    r"""
    Write the energy terms to a text file (legacy-compatible).

    One row per snapshot, ``e_kin,e_pot,e_int,E_total``.

    Args:
        energies (list[dict]): One dictionary per snapshot, with the keys
            ``e_kin``, ``e_pot``, ``e_int`` and ``E_total``.
        filename (str): Destination path.
    """
    with open(filename, "w") as f:
        for energy in energies:
            f.write(
                f"{energy['e_kin']},{energy['e_pot']},"
                f"{energy['e_int']},{energy['E_total']}\n"
            )
