"""Command line entry point of mloom (``mloom`` / ``uv run -m mloom``)."""

import argparse

import mloom


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the requested command.

    Parameters
    ----------
    argv : list[str] | None
        Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit code (see "CLI" in ``docs/spec.md``).
    """
    parser = argparse.ArgumentParser(prog="mloom")
    parser.add_argument("--version", action="version", version=mloom.__version__)
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
