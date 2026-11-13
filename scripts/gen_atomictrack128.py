"""Turn atomictrack into `atomictrack128`: atomictrack with 128-bit atomics (through portable-atomic) by default.

    python scripts/gen_atomictrack128.py               write the crate to ../atomictrack128
    python scripts/gen_atomictrack128.py <output_dir>  write the crate to <output_dir>
    python scripts/gen_atomictrack128.py --in-place    rewrite this crate itself (review with `git diff`)

Changes made:

- adds a `larger-atomics` feature (u128 / `portable_atomic::AtomicU128`, on by default) to Cargo.toml and
  src/math.rs, and a `guard-cross-process` feature (also on by default) which refuses to compile it without native
  128-bit atomics (portable-atomic's lock-based fallback is per-process, so it would be unsound for tracks in shared
  memory)
- adds portable-atomic as a regular dependency, and since it's there anyway, uses its `fence` in src/waiting.rs (a
  locked `xchg` on x86/x86_64 instead of `mfence`); also pins the 128-bit `id_of` outputs there
- Cargo.toml: package name, description, docs link, docs.rs features and flags (so docs.rs is able to build it), and drops
  the `scripts/` exclude so this script is included with atomictrack128
- README.md: a banner saying the crate is generated
- `atomictrack::` paths in Rust code become `atomictrack128::`, and the C API prefix `A_T_` becomes `A_T_128_`, so
  both crates can be linked into one binary with `capi`

Every edit must match exactly once, or the script stops before writing anything. Writing to an output directory
replaces only the paths in MANAGED there; anything else (`.git`, `target`, `Cargo.lock`) is left alone.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent
NAME = "atomictrack128"
FEATURE = "larger-atomics"
GUARD_FEATURE = "guard-cross-process"
UPSTREAM_LINK = "https://crates.io/crates/atomictrack"
BANNER = (
    f"**This is an auto-generated crate from [`atomictrack`]({UPSTREAM_LINK}), using `portable-atomic` and using "
    "128-bit atomics by default**"
)

# Paths (relative to the crate root) that make up the generated crate.
MANAGED = ["Cargo.toml", "README.md", "LICENSE-APACHE", "LICENSE-MIT", ".gitignore", "src", "tests", "examples", "scripts"]
# Files that get the crate path and C API prefix renames.
RENAMED_SUFFIXES = {".rs", ".h"}


# --- src/math.rs ------------------------------------------------------------------------------------------------

MATH_64 = """\
#[cfg(not(feature = "smaller-atomics"))]
use core::sync::atomic::AtomicU64;
#[cfg(not(feature = "smaller-atomics"))]
pub type AtomicType = AtomicU64;
#[cfg(not(feature = "smaller-atomics"))]
pub type NumericType = u64;

#[cfg(all(not(feature = "smaller-atomics"), not(target_has_atomic = "64")))]
compile_error!(
    "target doesn't support 64-bit atomics, enable the `smaller-atomics` feature to use 32-bit atomics instead"
);
"""

MATH_64_AND_128 = """\
#[cfg(feature = "larger-atomics")]
use portable_atomic::AtomicU128;
#[cfg(feature = "larger-atomics")]
pub type AtomicType = AtomicU128;
#[cfg(feature = "larger-atomics")]
pub type NumericType = u128;

#[cfg(not(any(feature = "smaller-atomics", feature = "larger-atomics")))]
use core::sync::atomic::AtomicU64;
#[cfg(not(any(feature = "smaller-atomics", feature = "larger-atomics")))]
pub type AtomicType = AtomicU64;
#[cfg(not(any(feature = "smaller-atomics", feature = "larger-atomics")))]
pub type NumericType = u64;

#[cfg(all(feature = "smaller-atomics", feature = "larger-atomics"))]
compile_error!("`smaller-atomics` and `larger-atomics` are mutually exclusive");

#[cfg(all(not(any(feature = "smaller-atomics", feature = "larger-atomics")), not(target_has_atomic = "64")))]
compile_error!(
    "target doesn't support 64-bit atomics, enable the `smaller-atomics` feature to use 32-bit atomics instead"
);

// Tracks can live in memory shared between processes, where portable-atomic's lock-based fallback (process-local
// locks) would be unsound, so `guard-cross-process` (on by default) requires native 128-bit atomics. On x86_64 that
// means `cmpxchg16b`, which is on by default for Windows and macOS targets; elsewhere, enable it with
// `-C target-feature=+cmpxchg16b`. Without the guard the fallback is allowed, which is fine as long as no track is
// shared between processes.
#[cfg(all(feature = "larger-atomics", feature = "guard-cross-process", target_arch = "x86_64", not(target_feature = "cmpxchg16b")))]
compile_error!(
    "`larger-atomics` needs native 128-bit atomics: build with `-C target-feature=+cmpxchg16b`, or turn off the `guard-cross-process` feature if no track is shared between processes"
);
#[cfg(all(feature = "larger-atomics", feature = "guard-cross-process"))]
const _: () = assert!(
    AtomicU128::is_always_lock_free(),
    "`larger-atomics` needs native 128-bit atomics (for example on x86_64 `-C target-feature=+cmpxchg16b`), or turn off the `guard-cross-process` feature if no track is shared between processes"
);
"""

# --- src/waiting.rs ---------------------------------------------------------------------------------------------

WAITING_IMPORT = """\
use core::{alloc::Layout, hash::Hasher, ptr::{self, NonNull}, sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering, fence}};
"""

WAITING_IMPORT_PORTABLE_FENCE = """\
use core::{alloc::Layout, hash::Hasher, ptr::{self, NonNull}, sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering}};
// portable-atomic's SeqCst fence is a locked `xchg` on x86/x86_64 instead of `mfence`, which is faster.
use portable_atomic::fence;
"""

ID_OF_PINNED_64 = """\
        #[cfg(not(feature = "smaller-atomics"))]
        let expected = [
            ("", 0x0338_dc4b_e2ce_cdae),
            ("renderer", 0x1df5_c40b_81fb_24e4),
            ("audio", 0x5969_afff_8526_d1e1),
        ];
"""

ID_OF_PINNED_64_AND_128 = """\
        #[cfg(not(any(feature = "smaller-atomics", feature = "larger-atomics")))]
        let expected = [
            ("", 0x0338_dc4b_e2ce_cdae),
            ("renderer", 0x1df5_c40b_81fb_24e4),
            ("audio", 0x5969_afff_8526_d1e1),
        ];
        // The 64-bit hash, with bit 63 kept since it isn't the suspended bit here.
        #[cfg(feature = "larger-atomics")]
        let expected = [
            ("", 0x0338_dc4b_e2ce_cdae),
            ("renderer", 0x1df5_c40b_81fb_24e4),
            ("audio", 0xd969_afff_8526_d1e1),
        ];
"""


def replace_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"error: expected exactly 1 match for {what}, found {n}; bring the script up to date")
    return text.replace(old, new)


def sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n != 1:
        sys.exit(f"error: expected exactly 1 match for {what}, found {n}; bring the script up to date")
    return new


def add_features(list_body, *features):
    items = [item.strip() for item in list_body.split(",") if item.strip()]
    items.extend(f'"{feature}"' for feature in features)
    return ", ".join(items)


def transform_cargo_toml(text):
    text = sub_once(r'^name = "atomictrack"$', f'name = "{NAME}"', text, "package name")
    text = sub_once(
        r'^description = "(.*)"$', r'description = "\1 (128-bit atomics version)"', text,
        "description",
    )
    text = sub_once(
        r'^documentation = "https://docs\.rs/atomictrack"$', f'documentation = "https://docs.rs/{NAME}"', text,
        "documentation link",
    )
    text = sub_once(r'^exclude = \["scripts/"\]\n', "", text, "scripts/ exclude")
    text = sub_once(r"^default = \[(.*)\]$", lambda m: f"default = [{add_features(m.group(1), FEATURE, GUARD_FEATURE)}]", text, "default features")
    text = sub_once(
        r"^(smaller-atomics = .*)$",
        r"\1" + "\n" + f"{FEATURE} = []  # marker feature for using 128-bit atomics through portable-atomic; "
        f"requires native 128-bit atomics on the target unless {GUARD_FEATURE} is off\n"
        f"{GUARD_FEATURE} = []  # marker feature to raise an explicit compile error when trying to compile with {FEATURE} on a target lacking them natively; "
        "portable-atomic is capable of falling back to per-process locks there, but this is unable to sufficiently support cross-process sharing of tracks",
        text, "smaller-atomics feature line",
    )
    text = sub_once(
        r"(^\[package\.metadata\.docs\.rs\]\nfeatures = )\[(.*)\]$",
        # docs.rs builds on x86_64 Linux, where `cmpxchg16b` is off by default. rustdoc needs the flag as well as rustc.
        lambda m: f"{m.group(1)}[{add_features(m.group(2), FEATURE)}]\n"
        'rustc-args = ["-C", "target-feature=+cmpxchg16b"]\n'
        'rustdoc-args = ["-C", "target-feature=+cmpxchg16b"]',
        text, "docs.rs features",
    )
    # 1.14 is the first portable-atomic with the faster x86 SeqCst fence. In addition, the `fallback` feature provides `AtomicU128` on
    # targets without native 128-bit atomics, which is allowed once the guard feature is turned off.
    text = sub_once(
        r"^\[dependencies\]\n",
        '[dependencies]\nportable-atomic = { version = "1.14", default-features = false, features = ["fallback"] }\n',
        text, "[dependencies] table",
    )
    return text


def transform_rust(text):
    text = re.sub(r"\batomictrack::", f"{NAME}::", text)
    text = re.sub(r"\bextern crate atomictrack\b", f"extern crate {NAME}", text)
    return re.sub(r"\bA_T_", "A_T_128_", text)


TRANSFORMS = {
    "Cargo.toml": [transform_cargo_toml],
    "README.md": [lambda text: f"{BANNER}\n\n{text}"],
    "src/math.rs": [lambda text: replace_once(text, MATH_64, MATH_64_AND_128, "the 64-bit types in src/math.rs")],
    "src/waiting.rs": [
        lambda text: replace_once(text, WAITING_IMPORT, WAITING_IMPORT_PORTABLE_FENCE, "the fence import in src/waiting.rs"),
        lambda text: replace_once(text, ID_OF_PINNED_64, ID_OF_PINNED_64_AND_128, "the pinned id_of outputs in src/waiting.rs"),
    ],
}


def managed_files(root):
    for entry in MANAGED:
        path = root / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file())


def generate():
    """Returns {relative path: (original bytes, generated bytes)} for every managed file."""
    out = {}
    for path in managed_files(SOURCE):
        rel = path.relative_to(SOURCE).as_posix()
        original = path.read_bytes()
        transforms = list(TRANSFORMS.get(rel, []))
        if path.suffix in RENAMED_SUFFIXES:
            transforms.append(transform_rust)
        if not transforms:
            out[rel] = (original, original)
            continue
        raw = original.decode("utf-8")
        newline = "\r\n" if "\r\n" in raw else "\n"
        text = raw.replace("\r\n", "\n")
        for transform in transforms:
            text = transform(text)
        out[rel] = (original, text.replace("\n", newline).encode("utf-8"))
    missing = [rel for rel in TRANSFORMS if rel not in out]
    if missing:
        sys.exit(f"error: missing from the source crate: {', '.join(missing)}")
    return out


def main():
    print("Note: This script should be run with a freshly cloned atomictrack folder.", flush=True)
    parser = argparse.ArgumentParser(description="Generate atomictrack128 from atomictrack.")
    parser.add_argument("output_dir", nargs="?", help="where to write the crate (default: ../atomictrack128)")
    parser.add_argument("--in-place", action="store_true", help="rewrite this crate itself")
    args = parser.parse_args()
    if args.in_place and args.output_dir:
        parser.error("give either an output directory or --in-place, not both")

    # Build everything in memory first, so a failed edit leaves every file untouched.
    files = generate()

    if args.in_place:
        changed = [rel for rel, (before, after) in files.items() if before != after]
        for rel in changed:
            (SOURCE / rel).write_bytes(files[rel][1])
        print(f"rewrote {SOURCE} into {NAME} in place ({len(changed)} files changed)")
        return

    out = Path(args.output_dir).resolve() if args.output_dir else SOURCE.parent / NAME
    if out == SOURCE:
        sys.exit("error: the output directory is the source crate; use --in-place for that")
    # The paths in MANAGED get deleted below, so only write into an empty directory or an earlier output of this script.
    if out.is_dir() and any(out.iterdir()):
        manifest = out / "Cargo.toml"
        if not manifest.is_file() or f'name = "{NAME}"' not in manifest.read_text(encoding="utf-8"):
            sys.exit(f"error: {out} is not empty and is not an {NAME} crate; refusing to replace files in it")
    out.mkdir(parents=True, exist_ok=True)
    for entry in MANAGED:
        target = out / entry
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for rel, (_, data) in files.items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"generated {NAME} in {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
