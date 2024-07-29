#!/usr/bin/env python3  # noqa: D100
import os


def _init_version(): # pragma: no cover
	init = os.path.join(os.path.dirname(__file__), "../../pyproject.toml")
	with open(init) as fid:
		data = fid.readlines()
	version_line = next(line for line in data if line.startswith("version ="))
	version = version_line.strip().split(" = ")[1]
	version = version.replace('"', "").replace("'", "")
	return version


if __name__ == "__main__": # pragma: no cover
	print(_init_version())
