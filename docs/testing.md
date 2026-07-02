# Testing

Run the full suite with:

```sh
make test          # or: uv run pytest
```

Most tests are pure unit tests and need no hardware. A handful of **integration
tests** (the ones using the `demo_core` fixture in
`shrimpy/tests/conftest.py` — e.g. `test_mantis_integration.py` and the
`DynaTrack` integration tests) drive a real `CMMCorePlus` against
Micro-Manager's built-in **demo devices**. These require the Micro-Manager
device adapters to be installed and discoverable by `pymmcore-plus`.

- **Windows / macOS:** run `uv run mmcore install` once — it downloads a
  prebuilt Micro-Manager and the demo adapters. This is also what CI does.
- **Linux:** there is no prebuilt nightly (`mmcore install` only fetches test
  adapters, which do not satisfy the device-interface version `pymmcore`
  requires), so the demo adapters must be **built from source**. Until they
  are, the `demo_core` tests error at fixture setup with
  `FileNotFoundError: ... MMConfig_demo.cfg` / "Could not find a compatible
  Micro-Manager installation". The rest of the suite still runs.

## Building the demo device adapters on Linux

You do **not** need to build MMCore or the full Micro-Manager application — the
`pymmcore` wheel already bundles MMCore. Only the demo **device adapters**
(which MMCore `dlopen`s at runtime) are missing, so this builds just those two
small, dependency-free adapters and drops them where `pymmcore-plus` looks.

### 1. Match the device-interface version

The built adapters must match the device-interface (DI) version that the
installed `pymmcore` expects, or MMCore refuses to load them.

```sh
uv run python -c "import pymmcore; print(pymmcore.__version__)"
# e.g. 12.2.2.75.0  -> DI version is the '75'
```

Check the DI version of `mmCoreAndDevices` before building:

```sh
git clone --depth 1 https://github.com/micro-manager/mmCoreAndDevices.git
grep DEVICE_INTERFACE_VERSION mmCoreAndDevices/MMDevice/MMDevice.h
# #define DEVICE_INTERFACE_VERSION 75   <- must equal the number above
```

If `main` no longer matches, check out an older commit/tag whose
`MMDevice.h` has the right number.

### 2. Compile the adapters

`DemoCamera` and `Utilities` (the canonical demo config references both) have
no external dependencies — no boost, no SWIG — so a modern C++ compiler is all
you need. They link statically against `MMDevice` (its `*.cpp` compiled in):

```sh
cd mmCoreAndDevices
BUILD=~/mm-build           # or any writable dir
mkdir -p "$BUILD"

for ADAPTER in DemoCamera Utilities; do
  g++ -shared -fPIC -std=c++17 -O2 -I MMDevice \
    DeviceAdapters/$ADAPTER/*.cpp MMDevice/*.cpp \
    -o "$BUILD/libmmgr_dal_$ADAPTER.so.0"
done
```

The `.so.0` suffix matters — MMCore looks for `libmmgr_dal_<name>.so.0`.

### 3. Add the demo configuration

Fetch the canonical `MMConfig_demo.cfg` (the same file CI uses; it references
`White Light Shutter`, `LED`, etc.):

```sh
curl -fsSL -o "$BUILD/MMConfig_demo.cfg" \
  https://raw.githubusercontent.com/micro-manager/micro-manager/main/bindist/any-platform/MMConfig_demo.cfg
```

### 4. Make it discoverable

`pymmcore-plus` auto-discovers a `Micro-Manager*` folder in its user-data dir,
so install the build there — then no environment variable is needed and no
repo files change:

```sh
MM=~/.local/share/pymmcore-plus/mm/Micro-Manager-2.0
mkdir -p "$MM"
cp "$BUILD"/libmmgr_dal_*.so.0 "$BUILD"/MMConfig_demo.cfg "$MM/"
```

(Alternatively, point `MICROMANAGER_PATH` at `$BUILD` for a one-off run:
`MICROMANAGER_PATH=$BUILD uv run pytest`.)

### 5. Verify

```sh
uv run python -c "from pymmcore_plus import find_micromanager; print(find_micromanager())"
# -> ~/.local/share/pymmcore-plus/mm/Micro-Manager-2.0
uv run pytest                     # all tests, including demo_core, should pass
```

## On this HPC (`comp_micro`)

The system module `micro-manager` (`module load micro-manager`) is an
Apptainer container whose adapters are built against an **older** device
interface and cannot be loaded by the venv's `pymmcore` — do not rely on it
for the test suite. Build the adapters from source as above instead.

Notes specific to this cluster:

- The build toolchain (`git`, `g++` 8.5, `autoconf`, `make`, …) is present by
  default; `g++ 8.5` handles `-std=c++17`. Newer compilers are available via
  `module load gcc/13.3` if needed.
- There is **no `sudo` and no system boost** — irrelevant here, because modern
  `MMDevice`/`DemoCamera`/`Utilities` need neither, and the build installs into
  `$HOME`.
- `OMP_NUM_THREADS=1` is pinned in `conftest.py`: `torch` and `pymmcore` each
  bring an OpenMP runtime, and both active in one process segfaults the
  camera-sequence C++ during a demo acquisition. Keep this when running demo
  tests outside pytest.
