"""Where the third-party scoring models live.

Std-lib only, so every scoring environment can import it.

Each oracle is a separate published model under its own licence and is not
redistributed with this repository -- see ../oracles/README.md for the paper
to obtain each from.  Nothing here has a default: a path stays None until you
set its variable, and require_path() raises a named error the first time an
oracle that actually needs it is called.  Generation and filtering never touch
any of this.

    source ../oracles/env.sh        # sets all of the below
"""

import os

_HERE = os.path.dirname(os.path.realpath(__file__))

# Parent of scorers/, placed on sys.path so `import scorers.*` resolves.
SCORERS_PARENT = _HERE


def _env_path(name, default=None):
    """$name if set, else `default`; realpath'd.  None when neither is given."""
    v = os.environ.get(name)
    if v:
        return os.path.realpath(os.path.expanduser(v))
    return None if default is None else os.path.realpath(default)


def require_path(value, env_name, what):
    """Fail late, with the variable name, rather than at import time."""
    if not value:
        raise SystemExit(
            f"${env_name} is not set, but it is needed to {what}.\n"
            f"These are third-party model weights that this repository does "
            f"not redistribute; see oracles/README.md for the publication to "
            f"obtain each one from.")
    if not os.path.exists(value):
        raise SystemExit(f"${env_name} points at a missing path: {value}")
    return value


# regLM -- yeast promoter activity.  REGLM_REPO goes on sys.path.
REGLM_REPO = _env_path("REGLM_REPO")
REGLM_YEAST_CKPT = _env_path(
    "REGLM_YEAST_CKPT",
    os.path.join(REGLM_REPO, "yeast_promoters/03_regression_separate/"
                             "yeast_regression_separate_defined.ckpt")
    if REGLM_REPO else None)

# GEMORNA -- human 5'UTR translational capacity.
GEMORNA_5UTR_CKPT = _env_path("GEMORNA_5UTR_CKPT")

# PARM -- human promoter activity.  The parent of the per-cell-type dirs.
PARM_MODEL_DIR = _env_path("PARM_MODEL_DIR")
PARM_CELL_DIR = (os.path.join(PARM_MODEL_DIR, "{cell_type}")
                 if PARM_MODEL_DIR else None)

# Malinois -- in-vitro MPRA enhancer activity.
MALINOIS_DIR = _env_path("MALINOIS_DIR")
MALINOIS_ARTIFACT = _env_path(
    "MALINOIS_ARTIFACT",
    os.path.join(MALINOIS_DIR,
                 "MODELS-malinois_artifacts__20211113_021200__287348.tar.gz")
    if MALINOIS_DIR else None)


def _main():
    rows = [("SCORERS_PARENT", SCORERS_PARENT),
            ("REGLM_REPO", REGLM_REPO),
            ("REGLM_YEAST_CKPT", REGLM_YEAST_CKPT),
            ("GEMORNA_5UTR_CKPT", GEMORNA_5UTR_CKPT),
            ("PARM_MODEL_DIR", PARM_MODEL_DIR),
            ("MALINOIS_DIR", MALINOIS_DIR),
            ("MALINOIS_ARTIFACT", MALINOIS_ARTIFACT)]
    for name, v in rows:
        if not v:
            print(f"  {name:<22} -- unset (${name})")
        else:
            print(f"  {name:<22} {v}"
                  + ("" if os.path.exists(v) else "   (missing)"))


if __name__ == "__main__":
    _main()
