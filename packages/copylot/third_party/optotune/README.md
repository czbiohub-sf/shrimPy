# Optotune MR-E Python SDK (unmodified wheels)

`optomdc-1.3.5434` and `optokummenberg-1.0.5422`, copied unmodified out of
Optotune's **MR-E Python SDK 1.3.5434** (covers MR-E-2 and MR-E-3), from
https://www.optotune.com/software-download/:

- archive: `MR-E_PythonSDK_1.3.5434.zip`
  (https://145326430.fs1.hubspotusercontent-eu1.net/hubfs/145326430/2025%20Website%20Downloads/Download%20Hub/MR-E_PythonSDK_1.3.5434.zip)
- archive sha256: `ef5130d80976e8551fedc40f88a36246641a034bc66cc8c613cd137aad54117a`

They are committed here only because uv cannot install a wheel from inside a zip
and the SDK is not on any package index; copylot's `optotune` extra points at
them through `[tool.uv.sources]` (see `packages/copylot/pyproject.toml`). Nothing
in them is edited, and copylot does not maintain this code.

`License.rtf` is Optotune's license, shipped in the same archive (permissive:
use, copy, modify and distribute, provided the copyright and permission notice
are kept). "SDKs are provided as is without any additional support from
Optotune."

## Updating

Download the new SDK zip from the page above, replace both wheels and
`License.rtf`, update the version pins and paths in `packages/copylot/pyproject.toml`
and the hashes below, then `uv lock`.

## Wheel sha256

    f93206f9a75107c31f434ac335ae16a60b6b9444b48bfa3716415f4744c1c577 *optokummenberg-1.0.5422-py3-none-any.whl
    8778b9c10429d2161c74b93a969ef1c77d5e7efdbffaeb724447e419ab774c92 *optomdc-1.3.5434-py3-none-any.whl
