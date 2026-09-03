"""
SSD-friendly write guard  --  skip disk writes whose result did not change.

The pipeline is re-run over the same data many times while parameters are being
tuned, and most outputs (heatmaps, traces, csv tables, pdf pages) come out
byte-for-byte identical every run.  Rewriting them burns SSD program/erase
cycles for nothing.

This module renders each output into RAM first, hashes it, compares that hash
with the file already on disk and only writes when the two differ.

    SSD_FRIENDLY_MODE = False      <- single switch, restores plain writes

Nothing else in the project has to change: install() monkey-patches the three
writers the project actually uses (matplotlib, tifffile, openpyxl), and the two
hand-rolled `open()` call sites use guarded_open() instead.

Design rules
------------
* Never break the pipeline.  Any unexpected problem in the buffered path falls
  back to the plain, unguarded write.
* Never buffer what cannot pay off.  If the target file does not exist yet
  there is nothing to compare against, so the write goes straight to disk with
  no RAM copy and no hashing.
* Never hold more than MAX_BUFFER_BYTES of an output in RAM; bigger targets are
  written straight through.
* Bytes on disk stay exactly what they were before the patch.  The guard only
  decides *whether* to write, never *what* to write.
"""

import functools
import inspect
import io
import os
import re
import sys
import zipfile

# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

# The master switch.  False  ->  every write happens unconditionally, exactly
# as it did before this module existed.
SSD_FRIENDLY_MODE = True

# Environment override, handy for a one-off run without editing the source:
#   DRS2P_SSD_FRIENDLY=0  disables,  =1  enables.
_ENV_OVERRIDE = os.environ.get("DRS2P_SSD_FRIENDLY")

# Outputs larger than this are written straight through instead of being
# buffered in RAM (judged by the size of the file already on disk).
MAX_BUFFER_BYTES = 2 * 1024**3  # 2 GiB

# Print one line per skipped/written file.
VERBOSE = False

# Fixed timestamp handed to libraries that would otherwise stamp "now" into
# their output (matplotlib pdf/svg/ps, zip containers on Python >= 3.12).
# Without this those formats differ on every run and could never be skipped.
# 946684800 == 2000-01-01T00:00:00Z, safely inside the 1980+ range DOS/zip
# timestamps require in every timezone.
SOURCE_DATE_EPOCH = "946684800"

# matplotlib salts the element ids it writes into SVG files with a fresh
# uuid4() per figure unless svg.hashsalt is set, so two identical figures never
# produce identical SVG bytes.  Only internal ids change, never the rendering.
SVG_HASHSALT = "drs2p"


def _enabled():
    if _ENV_OVERRIDE is not None:
        return _ENV_OVERRIDE not in ("0", "false", "False", "no", "")
    return SSD_FRIENDLY_MODE


# ---------------------------------------------------------------------------
# Hashing  --  fastest available, collisions are acceptable here
# ---------------------------------------------------------------------------

try:
    import xxhash  # optional, ~10x faster than anything in the stdlib

    HASH_NAME = "xxh3_64"

    def _new_hasher():
        return xxhash.xxh3_64()

    def _digest_bytes(data):
        return xxhash.xxh3_64_intdigest(data)

except ImportError:  # pragma: no cover - depends on the environment
    import zlib

    HASH_NAME = "crc32"

    class _Crc32:
        __slots__ = ("_v",)

        def __init__(self):
            self._v = 0

        def update(self, chunk):
            self._v = zlib.crc32(chunk, self._v)

        def intdigest(self):
            return self._v

    def _new_hasher():
        return _Crc32()

    def _digest_bytes(data):
        return zlib.crc32(data)


_READ_CHUNK = 4 * 1024 * 1024


def _digest_file(path):
    """Hash a file on disk in chunks. Returns None if it cannot be read."""
    try:
        hasher = _new_hasher()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_READ_CHUNK)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.intdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Comparison normalisers
# ---------------------------------------------------------------------------
# Some containers embed a creation timestamp, so two runs producing identical
# content still differ byte-wise.  A normaliser maps such a file to a canonical
# form used *only for the comparison* - the bytes actually written are always
# the untouched originals.

_XLSX_TIMESTAMPS = re.compile(rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:\1>")


def _normalize_zip(data):
    """Canonical form of a zip container: sorted member names plus contents,
    with member mtimes and document timestamps dropped."""
    out = bytearray()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(z.namelist()):
            body = _XLSX_TIMESTAMPS.sub(b"", z.read(name))
            out += name.encode("utf-8", "replace")
            out += b"\0" + str(len(body)).encode() + b"\0" + body + b"\0"
    return bytes(out)


_NORMALISERS = {
    ".xlsx": _normalize_zip,
    ".xlsm": _normalize_zip,
    ".zip": _normalize_zip,
}


def _normaliser_for(path):
    return _NORMALISERS.get(os.path.splitext(path)[1].lower())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

_stats = {"written": 0, "skipped": 0, "bytes_written": 0, "bytes_skipped": 0}


def reset_stats():
    for key in _stats:
        _stats[key] = 0


def report(prefix="SSD-friendly mode:"):
    """One-line summary of what the guard saved. Safe to call when disabled."""
    if not _enabled():
        return
    saved = _stats["bytes_skipped"]
    total = saved + _stats["bytes_written"]
    share = (100.0 * saved / total) if total else 0.0
    print(
        f"{prefix} {_stats['skipped']} file(s) unchanged and left alone "
        f"({saved / 1024**2:.1f} MiB not rewritten, {share:.0f}% of output), "
        f"{_stats['written']} file(s) written "
        f"({_stats['bytes_written'] / 1024**2:.1f} MiB), hash={HASH_NAME}"
    )


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------


def unchanged(path, data):
    """True when `path` already holds exactly `data` (per the fast hash)."""
    try:
        norm = _normaliser_for(path)
        if norm is None:
            # Cheap reject first: a size mismatch cannot be the same content.
            if os.path.getsize(path) != len(data):
                return False
            return _digest_file(path) == _digest_bytes(data)

        with open(path, "rb") as f:
            old = f.read()
        return _digest_bytes(norm(old)) == _digest_bytes(norm(data))
    except (OSError, ValueError, zipfile.BadZipFile):
        # Unreadable / not the container we expected -> treat as changed.
        return False


def write_if_changed(path, data):
    """Write `data` to `path` unless the file already holds it.

    Returns True when a write actually happened.
    """
    if _enabled() and os.path.exists(path) and unchanged(path, data):
        _stats["skipped"] += 1
        _stats["bytes_skipped"] += len(data)
        if VERBOSE:
            print(f"    [ssd] skipped: {path}")
        return False

    with open(path, "wb") as f:
        f.write(data)
    _stats["written"] += 1
    _stats["bytes_written"] += len(data)
    if VERBOSE:
        print(f"    [ssd] written: {path}")
    return True


# ---------------------------------------------------------------------------
# Helpers shared by the decorator and the library patches
# ---------------------------------------------------------------------------


def _is_path(obj):
    return isinstance(obj, (str, bytes, os.PathLike))


def _as_str_path(target):
    path = os.fspath(target)
    return path.decode("utf-8", "replace") if isinstance(path, bytes) else path


def _should_buffer(target):
    """Buffer only when there is an existing file worth comparing against."""
    if not _enabled() or not _is_path(target):
        return False
    try:
        return 0 < os.path.getsize(_as_str_path(target)) <= MAX_BUFFER_BYTES
    except OSError:
        return False  # missing file -> nothing to compare, write straight out


class _RetainingBytesIO(io.BytesIO):
    """BytesIO that keeps its payload readable after close().

    PdfPages closes the stream it was handed; without this the bytes would be
    gone before they could be compared.
    """

    def __init__(self):
        super().__init__()
        self.value = b""

    def close(self):
        if not self.closed:
            self.value = self.getvalue()
        super().close()

    def payload(self):
        return self.value if self.closed else self.getvalue()


# ---------------------------------------------------------------------------
# Public API 1:  the decorator
# ---------------------------------------------------------------------------


def guarded_writer(path_arg):
    """Decorate a function that writes its output to a path.

    The wrapped function must accept a binary file-like object wherever it
    accepts a path -- true for matplotlib, tifffile, openpyxl, csv, ruamel and
    every other writer used here.

    `path_arg` is the *name* of the path parameter, so it works for plain
    functions and for methods alike (`self` binds normally):

        @guarded_writer("output_path")
        def save_tiff(self, output_path, data): ...
    """

    def decorate(func):
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):  # C-implemented callable
            signature = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled() or signature is None:
                return func(*args, **kwargs)

            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)

            target = bound.arguments.get(path_arg)
            if not _should_buffer(target):
                return func(*args, **kwargs)

            path = _as_str_path(target)
            buffer = _RetainingBytesIO()
            bound.arguments[path_arg] = buffer

            try:
                result = func(*bound.args, **bound.kwargs)
                payload = buffer.payload()
            except Exception:
                # Buffering failed for a reason we did not anticipate; do the
                # plain write so the pipeline keeps its output.
                return func(*args, **kwargs)

            write_if_changed(path, payload)
            return result

        wrapper._ssd_guarded = True
        return wrapper

    return decorate


# ---------------------------------------------------------------------------
# Public API 2:  drop-in replacement for open() in write mode
# ---------------------------------------------------------------------------


class _GuardedFile:
    """Context manager that mimics `open(path, mode, ...)` for writing.

    Text mode goes through a TextIOWrapper over a BytesIO with the very same
    encoding/newline settings, so the buffered bytes are exactly the bytes
    `open()` would have produced (including the platform newline translation).
    """

    def __init__(self, path, mode, **kwargs):
        self.path = os.fspath(path)
        self.mode = mode
        self.kwargs = kwargs
        self.buffer = None
        self.stream = None

    def __enter__(self):
        if not _should_buffer(self.path):
            self.stream = open(self.path, self.mode, **self.kwargs)
            return self.stream

        self.buffer = io.BytesIO()
        if "b" in self.mode:
            return self.buffer

        self.stream = io.TextIOWrapper(
            self.buffer,
            encoding=self.kwargs.get("encoding"),
            errors=self.kwargs.get("errors"),
            newline=self.kwargs.get("newline"),
            write_through=True,
        )
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        if self.buffer is None:  # unguarded passthrough
            self.stream.close()
            return False

        if self.stream is not None:  # text mode: flush the wrapper first
            self.stream.flush()
            self.stream.detach()

        if exc_type is None:
            write_if_changed(self.path, self.buffer.getvalue())
        self.buffer.close()
        return False


def guarded_open(path, mode="w", **kwargs):
    """`with guarded_open(path, "w") as f:` -- same contract as open().

    Only write modes are intercepted; anything else is a plain open().
    """
    if "w" not in mode and "x" not in mode:
        return open(path, mode, **kwargs)
    return _GuardedFile(path, mode, **kwargs)


# ---------------------------------------------------------------------------
# Public API 3:  patch the third-party writers, so call sites stay untouched
# ---------------------------------------------------------------------------

_installed = False


def _rebind(old, new):
    """Re-point `from x import y` aliases that already grabbed the original.

    traces.py does `from matplotlib.backends.backend_pdf import PdfPages` and
    stabilization.py does `from tifffile import imwrite` at import time, i.e.
    long before install() runs, so patching the source module alone would miss
    them.  This sweeps every loaded module and repoints any module-level name
    still bound to the original object.
    """
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            names = vars(module)
        except TypeError:  # pragma: no cover - exotic module objects
            continue
        for name, value in list(names.items()):
            if value is old:
                try:
                    setattr(module, name, new)
                except Exception:  # pragma: no cover
                    pass


def _patch_matplotlib_savefig():
    import matplotlib as mpl
    from matplotlib.figure import Figure

    try:
        if mpl.rcParams["svg.hashsalt"] is None:
            mpl.rcParams["svg.hashsalt"] = SVG_HASHSALT
    except Exception:
        pass  # only an optimisation; never let it cost us the savefig guard

    original = Figure.savefig
    if getattr(original, "_ssd_guarded", False):
        return

    @functools.wraps(original)
    def savefig(self, fname, *args, **kwargs):
        if not _should_buffer(fname):
            return original(self, fname, *args, **kwargs)

        # A stream carries no filename, so matplotlib cannot infer the image
        # format from it and would quietly fall back to
        # rcParams["savefig.format"] (png) -- writing PNG bytes into a .pdf.
        # Take whatever the caller asked for (AutoStatLib always passes
        # format=), else the suffix; with neither, use the plain path.
        path = _as_str_path(fname)
        fmt = kwargs.get("format") or os.path.splitext(path)[1].lstrip(".").lower()
        if not fmt:
            return original(self, fname, *args, **kwargs)

        buffer = _RetainingBytesIO()
        try:
            result = original(self, buffer, *args, **{**kwargs, "format": fmt})
            payload = buffer.payload()
        except Exception:
            return original(self, fname, *args, **kwargs)

        write_if_changed(path, payload)
        return result

    savefig._ssd_guarded = True
    Figure.savefig = savefig
    # pyplot.savefig() and AutoStatLib's plot.save() both end up in
    # Figure.savefig, so patching it here covers every call site.


def _patch_matplotlib_pdfpages():
    from matplotlib.backends import backend_pdf

    original_pdfpages = backend_pdf.PdfPages
    if getattr(original_pdfpages, "_ssd_guarded", False):
        return

    class GuardedPdfPages(original_pdfpages):
        """PdfPages that accumulates the document in RAM and only lands it on
        disk at close(), if it differs from what is already there."""

        _ssd_guarded = True

        def __init__(self, filename, *args, **kwargs):
            self._ssd_path = None
            self._ssd_buffer = None
            if _should_buffer(filename):
                self._ssd_path = _as_str_path(filename)
                self._ssd_buffer = _RetainingBytesIO()
                filename = self._ssd_buffer
            super().__init__(filename, *args, **kwargs)

        def close(self):
            super().close()  # a genuine failure here must still propagate
            if self._ssd_path is None:
                return
            path, buffer = self._ssd_path, self._ssd_buffer
            self._ssd_path, self._ssd_buffer = None, None  # close() may repeat
            payload = buffer.payload()
            if payload:  # an empty document is not worth touching the disk for
                write_if_changed(path, payload)

    backend_pdf.PdfPages = GuardedPdfPages
    _rebind(original_pdfpages, GuardedPdfPages)


def _patch_tifffile():
    import tifffile

    for name in ("imwrite", "imsave"):
        original = getattr(tifffile, name, None)
        if original is None or getattr(original, "_ssd_guarded", False):
            continue
        guarded = guarded_writer("file")(original)
        setattr(tifffile, name, guarded)
        _rebind(original, guarded)


def _patch_openpyxl():
    from openpyxl.workbook import Workbook

    if getattr(Workbook.save, "_ssd_guarded", False):
        return
    Workbook.save = guarded_writer("filename")(Workbook.save)


def install():
    """Route matplotlib / tifffile / openpyxl writes through the guard.

    Idempotent, and a missing library is simply skipped.  Call once at start-up
    -- aliases created by earlier `from x import y` statements are re-pointed,
    so import order does not matter.
    """
    global _installed
    if _installed or not _enabled():
        return

    # Make timestamp-stamping formats reproducible, otherwise pdf/svg/xlsx
    # would differ on every run and could never be skipped.
    os.environ.setdefault("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)

    patches = (
        _patch_matplotlib_savefig,
        _patch_matplotlib_pdfpages,
        _patch_tifffile,
        _patch_openpyxl,
    )
    for patch in patches:
        try:
            patch()
        except ImportError:
            pass  # library not installed in this environment
        except Exception as exc:  # pragma: no cover - never block the pipeline
            print(f"    [ssd] could not patch {patch.__name__}: {exc!r}")

    _installed = True
