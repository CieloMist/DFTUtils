"""
dft_job_manager.py
==================

A SLURM job-submission manager for ASE-driven DFT calculations across two HPC
clusters. Every job is submitted as `python <your_ASE_script>.py`; the DFT code
runs *inside* that script (ASE's calculator launches VASP/QE). This manager only
writes a correct sbatch wrapper -- account, partition, resources, module loads,
conda activation -- and submits it via `slurmpy`.

    HPCs  : 'quest'    -- Northwestern University's Quest cluster
            'bridges2' -- PSC Bridges-2
    Codes : 'vasp' / 'qe' / None

`code` is optional: pass 'vasp' or 'qe' to load that code's module defaults and
get default per-job-type scripts, or leave it as None for jobs that just run
plain Python (no DFT code, no DFT modules) -- in that case use submit_script()
or submit(command=...).

Everything that varies by cluster (and, where needed, by code) lives in the
editable registries below:

    DEFAULT_ACCOUNTS    account to charge                         [hpc]
    DEFAULT_DIRECTIVES  always-on #SBATCH lines (mem, constraint) [hpc]
    CLUSTERS            partitions, walltime caps, cores/node, purge cmd [hpc]
    DEFAULT_MODULES     `module load` lines                       [hpc][code]
    ENV_SETUP           conda activation etc.                     [hpc]
    DEFAULT_PYTHON      interpreter for the ASE/python script     [hpc]
    DEFAULT_SCRIPTS     folder holding your scripts               [hpc]
    DEFAULT_RUN_DIR     working dir jobs cd into ("."=submit dir) [hpc]
    CODES[code].scripts default script filename                   [job_type]

Every submit method takes dry_run=True to render (and return) the sbatch script
without submitting -- and without needing slurmpy installed.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

try:
    from slurmpy import Slurm
except ImportError:  # pragma: no cover
    Slurm = None


def _load(*names: str) -> List[str]:
    """Convenience: turn module names into `module load <name>` lines."""
    return [f"module load {n}" for n in names]


# ===========================================================================
# Cluster configuration (per HPC)
# ===========================================================================
@dataclass(frozen=True)
class ClusterConfig:
    name: str
    cores_per_node: int
    partitions: Dict[str, float]          # partition -> max walltime (hours)
    auto_partitions: Sequence[str]
    default_partition: str
    gpu_partition: Optional[str]
    purge_command: str = "module purge"
    launcher: str = "srun"                # only used when mpi=True
    account_flag: str = "account"


CLUSTERS: Dict[str, ClusterConfig] = {
    "quest": ClusterConfig(
        name="quest",
        cores_per_node=48,                # matches the quest10-13 constraint nodes
        partitions={
            "short": 4.0, "normal": 48.0, "long": 168.0,
            "gengpu": 48.0, "genhimem": 48.0,
        },
        auto_partitions=("short", "normal", "long"),
        default_partition="normal",
        gpu_partition="gengpu",
        purge_command="module purge all",
    ),
    "bridges2": ClusterConfig(
        name="bridges2",
        cores_per_node=128,
        partitions={
            "RM": 48.0, "RM-shared": 48.0, "RM-512": 48.0,
            "EM": 120.0, "GPU": 48.0, "GPU-shared": 48.0,
        },
        auto_partitions=("RM",),
        default_partition="RM",
        gpu_partition="GPU",
        purge_command="module purge",
    ),
}


# ===========================================================================
# Default driver script per job type (filenames from your shared scripts).
# ===========================================================================
@dataclass(frozen=True)
class CodeConfig:
    name: str
    scripts: Dict[str, str]


CODES: Dict[str, CodeConfig] = {
    "vasp": CodeConfig(
        name="vasp",
        scripts={
            "relaxation":     "Relax_ASE.py",
            "singlepoint":    "ASE_SinglePoint.py",
            "postprocessing": "ASE_PostProcess.py",   # TODO confirm filename
        },
    ),
    "qe": CodeConfig(
        name="qe",
        scripts={
            "relaxation":     "Relax_QE.py",
            "singlepoint":    "QE_SinglePoint.py",
            "postprocessing": "QE_PostProcess.py",     # TODO confirm filename
        },
    ),
}

_CODE_ALIASES = {
    "vasp": "vasp",
    "qe": "qe", "espresso": "qe", "quantum-espresso": "qe",
    "quantumespresso": "qe", "quantum_espresso": "qe", "pwscf": "qe",
}


# ===========================================================================
# Per-HPC defaults you EDIT.
# ===========================================================================
DEFAULT_ACCOUNTS: Dict[str, str] = {
    "quest":    "p32212",        # from your scripts
    "bridges2": "mat260005p",    # PSC allocation
}

# #SBATCH directives applied to every job on the HPC (per-call extra_directives
# override these). These appear in all your Quest scripts.
DEFAULT_DIRECTIVES: Dict[str, Dict[str, Union[str, int]]] = {
    "quest": {
        "mem-per-cpu": "4G",
        "constraint": "[quest10|quest11|quest12|quest13]",
    },
    "bridges2": {},
}

# Module / setup lines, resolved per [hpc][code]. Values are full shell lines,
# so `module use ...` works alongside `module load ...` (use the _load() helper
# for the common case). Only consulted when a code is selected.
DEFAULT_MODULES: Dict[str, Dict[str, List[str]]] = {
    "quest": {
        "qe":   _load("quantum-espresso/7.5-openmpi-gcc-13.3.0"),
        "vasp": _load("vasp/6.5.0-openmpi-intel-mkl-2021.4.0"),
    },
    "bridges2": {
        "qe":   _load("QuantumEspresso/7.5-intel"),
        "vasp": _load("VASP/6.5.1-gcc+vtst"),
    },
}

# Shell lines run AFTER modules and BEFORE `cd` (conda activation, etc.).
# On Bridges-2, oneAPI is sourced LAST so its MKL/runtime libs win
# LD_LIBRARY_PATH at runtime; DEFAULT_PYTHON is pinned to an absolute conda
# path so that ordering doesn't steal `python` from the conda env.
ENV_SETUP: Dict[str, List[str]] = {
    "quest":    ["source ~/.bashrc", "source activate atomistic"],
    "bridges2": ["source ~/.bashrc",
                 "source activate atomistic",              # conda first...
                 "source /opt/intel/oneapi/setvars.sh"]    # ...MKL/oneAPI last so it wins at runtime
}

# Interpreter for the script. On Bridges-2 this is an ABSOLUTE conda path so it
# can't be shadowed by oneAPI's intelpython on PATH (see ENV_SETUP note).
DEFAULT_PYTHON: Dict[str, str] = {
    "quest":    "python",
    "bridges2": "/jet/home/dsmithc/.conda/envs/atomistic/bin/python",
}

# Folder holding your driver scripts, so you call them by bare filename.
DEFAULT_SCRIPTS: Dict[str, str] = {
    "quest":    "/projects/p32212/DefaultScripts/DFTUtils/Python_Scripts",                    # your scripts folder
    "bridges2": "/ocean/projects/mat260005p/dsmithc/DefaultScripts/DFTUtils/Python_Scripts",  # your scripts folder
}

# Working directory the job `cd`s into before running (where the script looks
# for its inputs/config and writes outputs). "." = the submission directory,
# i.e. wherever you launched the job from. Distinct from DEFAULT_SCRIPTS, which
# only locates the script file itself.
DEFAULT_RUN_DIR: Dict[str, str] = {
    "quest":    ".",
    "bridges2": ".",
}


# ===========================================================================
# Job manager
# ===========================================================================
class DFTJobManager:
    """
    Submit ASE-driven DFT (or plain Python) jobs to a chosen HPC cluster.

    Parameters
    ----------
    hpc : str                 'quest' or 'bridges2'.
    code : str, optional      'vasp', 'qe', or None. When 'vasp'/'qe', that
                              code's DEFAULT_MODULES are loaded and the typed
                              submit methods get a default script. When None,
                              no DFT modules are loaded and you must supply a
                              script=/command= (e.g. via submit_script()).
    account : str, optional   Defaults to DEFAULT_ACCOUNTS[hpc].
    default_nodes : int       Used when a submit call omits `nodes`.
    ntasks_per_node : int, optional
                              Default tasks/node (overridable per call).
    modules : list[str], optional
                              Replace module loads for ALL jobs of this instance
                              (module *names*, one `module load` each). Useful
                              with code=None to still load e.g. a python module.
    env_setup : list[str], optional      Replace ENV_SETUP[hpc].
    default_directives : dict, optional   Replace DEFAULT_DIRECTIVES[hpc].
    default_scripts_dir, python_exe : str, optional
                              Override DEFAULT_SCRIPTS[hpc] / DEFAULT_PYTHON[hpc].
    default_run_dir : str, optional
                              Working directory jobs `cd` into when `directory`
                              is not passed. Overrides DEFAULT_RUN_DIR[hpc]
                              (default "." = the submission directory).
    bash_setup : str          Strictness line inserted after env setup, before
                              `cd` (default "set -eo pipefail"; "" to omit).
    date_in_name : bool       Passed to slurmpy.
    scripts_dir, log_dir : str
                              Where slurmpy writes generated sbatch scripts/logs
                              (scripts_dir is distinct from default_scripts_dir).
    clear_old_logs : bool     If True, wipe previous *.out/*.err (and generated
                              *.sh) from log_dir/scripts_dir at construction.
    """

    def __init__(
        self,
        hpc: str,
        code: Optional[str] = None,
        account: Optional[str] = None,
        *,
        default_nodes: int = 1,
        ntasks_per_node: Optional[int] = None,
        modules: Optional[List[str]] = None,
        env_setup: Optional[List[str]] = None,
        default_directives: Optional[Dict[str, Union[str, int]]] = None,
        default_scripts_dir: Optional[str] = None,
        default_run_dir: Optional[str] = None,
        python_exe: Optional[str] = None,
        bash_setup: str = "set -eo pipefail",
        date_in_name: bool = False,
        scripts_dir: str = "slurm-scripts",
        log_dir: str = "logs",
        clear_old_logs: bool = False,
    ):
        hpc_key = hpc.lower()
        if hpc_key not in CLUSTERS:
            raise ValueError(f"Unknown HPC '{hpc}'. Choose: {sorted(CLUSTERS)}.")

        # code is optional: None -> no DFT code, no DFT modules.
        if code is None:
            code_key: Optional[str] = None
        else:
            code_key = _CODE_ALIASES.get(code.lower().replace(" ", "-"))
            if code_key is None:
                raise ValueError(
                    f"Unknown code '{code}'. Choose: 'vasp', 'qe', or None "
                    f"(aliases: {sorted(set(_CODE_ALIASES) - {'vasp', 'qe'})})."
                )

        self._hpc_key = hpc_key
        self._code_key = code_key
        self.cfg = CLUSTERS[hpc_key]
        self.code = CODES[code_key] if code_key is not None else None

        account = account or DEFAULT_ACCOUNTS.get(hpc_key)
        if not account:
            raise ValueError(
                f"No account given and none configured for '{hpc_key}'. Set "
                f"DEFAULT_ACCOUNTS['{hpc_key}'] or pass account=..."
            )
        self.account = account

        # DFT module defaults only when a code is selected; otherwise empty
        # (a code-less job can still load modules via modules= or extra_modules=).
        if code_key is None:
            self._modules = []
        else:
            try:
                self._modules = DEFAULT_MODULES[hpc_key][code_key]
            except KeyError:
                raise ValueError(
                    f"No modules configured for ({hpc_key}, {code_key}). Add them "
                    f"to DEFAULT_MODULES or pass modules=[...]."
                )
        self._modules_override = list(modules) if modules is not None else None

        self._env_setup = list(env_setup) if env_setup is not None else list(ENV_SETUP.get(hpc_key, []))
        self._default_directives = dict(default_directives) if default_directives is not None else dict(DEFAULT_DIRECTIVES.get(hpc_key, {}))
        self.default_scripts_dir = default_scripts_dir if default_scripts_dir is not None else DEFAULT_SCRIPTS.get(hpc_key)
        self.default_run_dir = default_run_dir if default_run_dir is not None else DEFAULT_RUN_DIR.get(hpc_key, ".")
        self._python = python_exe or DEFAULT_PYTHON.get(hpc_key, "python")
        self.bash_setup = bash_setup

        self.default_nodes = default_nodes
        self.ntasks_per_node = ntasks_per_node
        self.date_in_name = date_in_name
        self.scripts_dir = scripts_dir
        self.log_dir = log_dir
        if clear_old_logs:
            self.clear_logs()

    # ------------------------------------------------------------------ #
    def script_path(self, name: str) -> str:
        """Resolve a script name against default_scripts_dir (abs passes through)."""
        if os.path.isabs(name):
            return name
        if not self.default_scripts_dir:
            raise ValueError(
                f"No default_scripts_dir for '{self.cfg.name}'. Set "
                f"DEFAULT_SCRIPTS['{self.cfg.name}'], pass default_scripts_dir=, "
                f"or use an absolute path."
            )
        return os.path.join(self.default_scripts_dir, name)

    # ------------------------------------------------------------------ #
    # Log housekeeping
    # ------------------------------------------------------------------ #
    def clear_logs(self, include_scripts: bool = True) -> int:
        """Delete slurm log files from previous runs in this manager's log_dir.

        include_scripts : bool
            When True, also delete the generated `*.sh` files in scripts_dir.

        Removes only `*.out` / `*.err` in `self.log_dir` (the names slurmpy
        writes) and, optionally, `*.sh` in `self.scripts_dir`. Never touches any
        other files or directories. Returns the number of files removed; safe to
        call before the directories exist (returns 0).
        """
        patterns = [
            os.path.join(self.log_dir, "*.out"),
            os.path.join(self.log_dir, "*.err"),
        ]
        if include_scripts:
            patterns.append(os.path.join(self.scripts_dir, "*.sh"))
        removed = 0
        for pattern in patterns:
            for path in glob.glob(pattern):
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1
        return removed

    # ------------------------------------------------------------------ #
    # Public submit methods
    #
    # Each forwards to _submit_job via **kw. See _submit_job for the full list
    # of shared keyword arguments (time, nodes, ntasks_per_node, partition,
    # script, script_args, command, mpi, python, gpu, depends_on, extra_modules,
    # extra_setup, extra_directives, name, dry_run).
    # ------------------------------------------------------------------ #
    def submit_relaxation(self, directory=None, **kw):
        """Submit a geometry/cell relaxation.

        directory : str, optional
            Calculation directory the job `cd`s into before running. Defaults to
            the per-HPC run directory (DEFAULT_RUN_DIR[hpc], "." = submission dir).
        **kw :
            Shared submit keywords (see _submit_job). With a code selected the
            default script is CODES[code].scripts['relaxation']; with code=None
            you must pass script= or command=.
        """
        return self._submit_job("relaxation", directory, _name_prefix="relax", _default_time="04:00:00", **kw)

    def submit_singlepoint(self, directory=None, **kw):
        """Submit a single-point / SCF energy evaluation.

        directory : str, optional
            Calculation directory the job `cd`s into before running. Defaults to
            the per-HPC run directory (DEFAULT_RUN_DIR[hpc], "." = submission dir).
        **kw :
            Shared submit keywords (see _submit_job). Default script with a code
            selected is CODES[code].scripts['singlepoint'].
        """
        return self._submit_job("singlepoint", directory, _name_prefix="scf", _default_time="04:00:00", **kw)

    def submit_postprocessing(self, directory=None, **kw):
        """Submit a post-processing step (often chained after relax/SCF).

        directory : str, optional
            Calculation directory the job `cd`s into before running. Defaults to
            the per-HPC run directory (DEFAULT_RUN_DIR[hpc], "." = submission dir).
        **kw :
            Shared submit keywords (see _submit_job). Default script with a code
            selected is CODES[code].scripts['postprocessing'].
        """
        return self._submit_job("postprocessing", directory, _name_prefix="post", _default_time="04:00:00", **kw)

    def submit_script(self, script, directory=None, **kw):
        """Run one of your scripts by filename (no job-type assumptions).

        script : str
            Script filename, resolved against default_scripts_dir (an absolute
            path is used unchanged). Executed as `<python> <script> [args]`.
        directory : str, optional
            Working/calculation directory the job `cd`s into before running.
            Defaults to the per-HPC run directory (DEFAULT_RUN_DIR[hpc], "." = submission dir).
        **kw :
            Shared submit keywords (see _submit_job) -- commonly script_args,
            time, nodes, ntasks_per_node, partition, depends_on, extra_modules,
            extra_setup, extra_directives, name, dry_run. `name` defaults to the
            script stem (plus the dir tag when a directory is given explicitly).
        """
        base = os.path.splitext(os.path.basename(script))[0]
        if directory is not None:
            kw.setdefault("name", f"{base}_{_tag(directory)}")
        else:
            kw.setdefault("name", base)
        return self._submit_job(None, directory, script=script, _name_prefix=base, _default_time="04:00:00", **kw)

    def submit(self, directory=None, *, command, name, **kw):
        """Run an arbitrary shell command as the job body (escape hatch).

        directory : str, optional
            Working directory the job `cd`s into before running `command`.
            Defaults to the per-HPC run directory (DEFAULT_RUN_DIR[hpc], "." = submission dir).
        command : str
            Exact shell command to run -- no python/script wrapping is added.
            Use this for non-Python tools (e.g. `bader CHGCAR`).
        name : str
            SLURM job name (required, since there's no script to derive it from).
        **kw :
            Shared submit keywords (see _submit_job).
        """
        return self._submit_job(None, directory, command=command, name=name, _name_prefix="job", _default_time="04:00:00", **kw)

    def submit_workflow(self, steps: Sequence[Dict], *, dry_run: bool = False) -> List:
        """Submit a chained sequence; each step depends (afterok) on the prior.

        steps : sequence of dict
            Each dict has `kind` in {'relaxation','singlepoint','postprocessing',
            'script','custom'} plus that method's kwargs (e.g. directory, time,
            nodes, script). A `depends_on` is injected automatically linking each
            step to the previous job id unless you supply your own.
        dry_run : bool
            Render every step instead of submitting.

        Returns the list of job ids (or rendered scripts when dry_run=True).
        """
        dispatch = {
            "relaxation": self.submit_relaxation,
            "singlepoint": self.submit_singlepoint,
            "postprocessing": self.submit_postprocessing,
            "script": self.submit_script,
            "custom": self.submit,
        }
        results: List = []
        previous_id: Optional[int] = None
        for raw in steps:
            step = dict(raw)
            kind = step.pop("kind")
            if kind not in dispatch:
                raise ValueError(f"Unknown step kind '{kind}'.")
            if previous_id is not None and "depends_on" not in step:
                step["depends_on"] = [previous_id]
            result = dispatch[kind](dry_run=dry_run, **step)
            results.append(result)
            previous_id = result if not dry_run else None
        return results

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    def _submit_job(
        self,
        job_type: Optional[str],
        directory: str,
        *,
        _name_prefix: str,
        _default_time: str,
        name: Optional[str] = None,
        script: Optional[str] = None,
        script_args: Union[str, Sequence[str]] = "",
        command: Optional[str] = None,
        time: Optional[Union[str, float]] = None,
        nodes: Optional[int] = None,
        ntasks_per_node: Optional[int] = None,
        partition: Optional[str] = None,
        mpi: bool = False,
        python: Optional[str] = None,
        gpu: bool = False,
        depends_on: Optional[Sequence[int]] = None,
        extra_modules: Optional[List[str]] = None,
        extra_setup: Optional[List[str]] = None,
        extra_directives: Optional[Dict[str, Union[str, int]]] = None,
        dry_run: bool = False,
    ):
        """Build and submit (or render) one job. Central place for all keywords.

        Parameters
        ----------
        job_type : str | None
            Internal: 'relaxation'|'singlepoint'|'postprocessing' selects the
            default script from CODES[code].scripts; None means no default
            (submit_script/submit supply their own script/command).
        directory : str | None
            Calculation/working directory; the body does `cd {directory}` before
            running. Defaults to the per-HPC run directory (DEFAULT_RUN_DIR[hpc],
            "." = the submission directory) when None.
        _name_prefix : str
            Internal: prefix for the auto-generated job name "<prefix>_<dir-tag>".
        _default_time : str
            Internal: walltime used when `time` is not given.
        name : str, optional
            SLURM job name (-J). Defaults to "<_name_prefix>_<dir-tag>".
        script : str, optional
            Script filename to run (resolved via script_path). Overrides the
            job-type default. Ignored if `command` is given.
        script_args : str | sequence of str
            Extra CLI arguments appended after the script (a list is joined with
            spaces). E.g. "--encut 520" or ["--encut", "520"].
        command : str, optional
            Raw shell command to use as the run line instead of building
            `<python> <script>`. Highest precedence; skips script resolution.
        time : str | float, optional
            Walltime as "HH:MM:SS", "D-HH:MM:SS", or a number of hours. Validated
            against the partition's cap. Falls back to `_default_time`.
        nodes : int, optional
            Node count (-N). Defaults to the manager's default_nodes.
        ntasks_per_node : int, optional
            MPI tasks per node (--ntasks-per-node). Defaults to the per-instance
            value or the cluster's cores_per_node.
        partition : str, optional
            SLURM partition. If omitted, auto-selected as the smallest partition
            whose walltime cap fits `time` (or the GPU partition if gpu=True).
        mpi : bool
            If True, prepend the cluster launcher (e.g. `srun`) to the run line.
            Leave False for ASE/driver scripts that spawn mpirun internally.
        python : str, optional
            Interpreter for this job, overriding the manager's default
            (DEFAULT_PYTHON[hpc]).
        gpu : bool
            If True and no `partition` given, route to the cluster's GPU partition.
        depends_on : sequence of int, optional
            Job ids this job must wait for (submitted with SLURM `afterok`).
        extra_modules : list of str, optional
            Additional module *names* to `module load` for this job only.
        extra_setup : list of str, optional
            Additional shell lines inserted after env setup (e.g. `export ...`,
            or probe commands like `which python`). Run before strict mode/`cd`.
        extra_directives : dict, optional
            Extra `#SBATCH` directives ({key: value}); override DEFAULT_DIRECTIVES.
        dry_run : bool
            If True, render and return the sbatch script text without submitting
            (and without needing slurmpy installed).

        Returns
        -------
        int | str | None
            The slurm job id on submission, or the rendered script if dry_run.
        """
        cfg = self.cfg
        # Resolve the working directory: an explicit `directory` wins, otherwise
        # fall back to the per-HPC default run directory ("." = submission dir).
        explicit_dir = directory is not None
        run_dir = directory if explicit_dir else self.default_run_dir
        if not run_dir:
            raise ValueError(
                f"No directory given and no default run directory configured for "
                f"'{cfg.name}'. Pass directory=, set default_run_dir=, or set "
                f"DEFAULT_RUN_DIR['{cfg.name}']."
            )
        # When the directory is the per-HPC default, its basename adds nothing to
        # the job name, so use just the prefix; otherwise tag with the dir.
        job_name = name or (f"{_name_prefix}_{_tag(run_dir)}" if explicit_dir else _name_prefix)
        nodes = nodes or self.default_nodes
        ntpn = ntasks_per_node or self.ntasks_per_node or cfg.cores_per_node
        ntasks = nodes * ntpn

        walltime = _normalize_walltime(time if time is not None else _default_time)
        hours = _to_hours(walltime)
        if partition is None and gpu:
            partition = cfg.gpu_partition
        partition = self._resolve_partition(partition, hours)
        self._validate_walltime(partition, hours)

        # run line: raw command > named script > job-type default script
        if command is None:
            if script is None and self.code is not None:
                script = self.code.scripts.get(job_type)
            if script is None:
                if self.code is not None:
                    raise ValueError(
                        f"No default {self.code.name} script for job_type="
                        f"{job_type!r}; pass script= or command=."
                    )
                raise ValueError(
                    "No code was selected at init, so there is no default script; "
                    "pass script= (e.g. via submit_script()) or command=."
                )
            pieces: List[str] = []
            if mpi:
                pieces.append(
                    cfg.launcher.format(ntasks=ntasks)
                    if "{ntasks}" in cfg.launcher else cfg.launcher
                )
            pieces.append(python or self._python)
            pieces.append(self.script_path(script))
            args = _fmt_args(script_args)
            if args:
                pieces.append(args)
            command = " ".join(pieces)

        module_lines = self._resolve_modules()
        body = self._build_body(run_dir, command, module_lines, extra_modules, extra_setup)

        slurm_kwargs: Dict[str, Union[str, int]] = {
            cfg.account_flag: self.account,
            "partition": partition,
            "nodes": nodes,
            "ntasks-per-node": ntpn,
            "time": walltime,
        }
        for k, v in self._default_directives.items():
            slurm_kwargs.setdefault(k, v)
        if extra_directives:
            slurm_kwargs.update(extra_directives)

        if dry_run:
            preview = self._render_preview(job_name, slurm_kwargs, body)
            print(preview)
            return preview

        if Slurm is None:
            raise RuntimeError(
                "slurmpy is not installed; `pip install slurmpy` to submit "
                "(use dry_run=True to preview without it)."
            )
        s = Slurm(job_name, slurm_kwargs, date_in_name=self.date_in_name,
                  scripts_dir=self.scripts_dir, log_dir=self.log_dir,
                  bash_strict=False)
        return s.run(body, depends_on=list(depends_on) if depends_on else None)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_modules(self) -> List[str]:
        """Return the `module load`/setup lines for this job (override > code defaults)."""
        if self._modules_override is not None:
            return [f"module load {m}" for m in self._modules_override]
        return list(self._modules)

    def _build_body(self, directory, command, module_lines, extra_modules, extra_setup) -> str:
        """Assemble the script body: purge -> modules -> env setup -> strict -> cd -> run."""
        lines: List[str] = [self.cfg.purge_command]
        lines += list(module_lines)
        lines += [f"module load {m}" for m in (extra_modules or [])]
        lines += list(self._env_setup)
        lines += list(extra_setup or [])
        lines.append("")
        # Strict mode only AFTER environment setup, so sourcing ~/.bashrc and
        # conda activation aren't killed by nounset/errexit. Empty -> omit.
        if self.bash_setup:
            lines.append(self.bash_setup)
        lines.append(f"cd {directory}")
        lines.append(command)
        return "\n".join(lines)

    def _resolve_partition(self, partition, hours) -> str:
        """Validate an explicit partition, or auto-pick the smallest that fits `hours`."""
        cfg = self.cfg
        if partition is not None:
            if partition not in cfg.partitions:
                raise ValueError(
                    f"Unknown partition '{partition}' for {cfg.name}. "
                    f"Known: {sorted(cfg.partitions)}."
                )
            return partition
        for p in cfg.auto_partitions:
            if cfg.partitions[p] >= hours:
                return p
        raise ValueError(
            f"Walltime {hours:.2f} h exceeds the largest auto-select partition on "
            f"{cfg.name} ({cfg.auto_partitions}). Pass partition= or reduce time."
        )

    def _validate_walltime(self, partition, hours) -> None:
        """Raise if the requested walltime exceeds the partition's cap."""
        cap = self.cfg.partitions[partition]
        if hours > cap + 1e-9:
            raise ValueError(
                f"Requested {hours:.2f} h exceeds the {cap} h cap of partition "
                f"'{partition}' on {self.cfg.name}."
            )

    def _render_preview(self, job_name, slurm_kwargs, body) -> str:
        """Render the sbatch script as text (matches what slurmpy would submit)."""
        lines = [
            "#!/bin/bash", "",
            f"#SBATCH -J {job_name}",
            f"#SBATCH -o {self.log_dir}/{job_name}.%J.out",
            f"#SBATCH -e {self.log_dir}/{job_name}.%J.err",
        ]
        for k, v in slurm_kwargs.items():
            flag = f"--{k}={v}" if len(k) > 1 else f"-{k} {v}"
            lines.append(f"#SBATCH {flag}")
        lines += ["", body, ""]
        return "\n".join(lines)


# ===========================================================================
# Helpers
# ===========================================================================
def _tag(directory: str) -> str:
    """Short filename-safe label from a directory path (its basename)."""
    return os.path.basename(os.path.normpath(directory)) or "job"


def _fmt_args(args: Union[str, Sequence[str]]) -> str:
    """Normalize script_args (string or list) to a single arg string."""
    if not args:
        return ""
    if isinstance(args, (list, tuple)):
        return " ".join(str(a) for a in args)
    return str(args)


def _normalize_walltime(time: Union[str, float, int]) -> str:
    """Accept 'HH:MM:SS', 'D-HH:MM:SS', or a number of hours -> SLURM string."""
    if isinstance(time, (int, float)):
        total_minutes = int(round(time * 60))
        h, m = divmod(total_minutes, 60)
        return f"{h:02d}:{m:02d}:00"
    return str(time)


def _to_hours(time: Union[str, float, int]) -> float:
    """Convert a SLURM walltime string (or hours number) to float hours."""
    walltime = _normalize_walltime(time)
    days, rest = 0, walltime
    if "-" in rest:
        d, rest = rest.split("-", 1)
        days = int(d)
    bits = [int(b) for b in rest.split(":")]
    if len(bits) == 3:
        h, m, s = bits
    elif len(bits) == 2:
        h, m, s = bits[0], bits[1], 0
    elif len(bits) == 1:
        h, m, s = bits[0], 0, 0
    else:
        raise ValueError(f"Unrecognized walltime format: {walltime!r}")
    return days * 24 + h + m / 60 + s / 3600


# ===========================================================================
# Example usage (previews only)
# ===========================================================================
if __name__ == "__main__":
    def banner(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

    banner("QUEST + QE single-point  (no directory -> runs in DEFAULT_SCRIPTS[quest])")
    DFTJobManager("quest", "qe").submit_singlepoint(
        nodes=8, ntasks_per_node=8, partition="short", time="02:00:00", dry_run=True)

    banner("BRIDGES-2 + VASP relaxation  (explicit directory override)")
    DFTJobManager("bridges2", "vasp").submit_relaxation(
        "calcs/Fe2O3/relax", nodes=2, time="24:00:00", dry_run=True)

    banner("BRIDGES-2, NO code -- run a plain Python script in the default dir")
    DFTJobManager("bridges2").submit_script(
        "make_E0.py", time="00:30:00", dry_run=True)