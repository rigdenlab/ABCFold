#!/usr/bin/env python

"""
Create a copy of the PAE viewer standalone HTML with the data provided
via the CLI and open it in the default browser.

The script requires a local webserver to be run with the repository root
as the webroot. You can achieve this by changing to the repository root
directory (`pae-viewer`) and using the built-in Python HTTP module:

```bash
cd /your/download/path/pae-viewer
python3 -m http.server 8000
```

This will start an HTTP server handling requests via port 8000, which
is the default for this script. However, a custom port can be set via
the `--port` option.
"""

import argparse
import json
import os
# import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def resolved_path(value: str) -> Path:
    path = Path(value).resolve()

    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")

    return path


def load_pae_viewer(
    structure_path: Path,
    chain_labels: str,
    template_file: Path,
    scores_path: Optional[Path] = None,
    crosslinks_path: Optional[Path] = None,
    port: int = 8000,
    output_file: Optional[Path] = None,
):
    data = get_session_data(structure_path, chain_labels, scores_path, crosslinks_path)
    # session_path = create_session_file(
    #     structure_path.stem, data, output_file, template_file
    # )
    create_session_file(
        structure_path.stem, data, output_file, template_file
    )

    # webbrowser.open(
    #     f"localhost:{port}/{Path(*session_path.parts[-2:])}", new=0, autoraise=True
    # )


def get_session_data(
    structure_path: Path,
    chain_labels: str,
    scores_path: Optional[Path] = None,
    crosslinks_path: Optional[Path] = None,
) -> dict[str, Any]:
    data = {
        "structureFile": {
            "name": structure_path.name,
            "content": structure_path.read_text(),
        },
        "chainLabels": chain_labels,
    }

    if scores_path:
        data["scoresFile"] = {
            "name": scores_path.name,
            "content": scores_path.read_text(),
        }

    if crosslinks_path:
        data["crosslinksFile"] = {
            "name": crosslinks_path.name,
            "content": crosslinks_path.read_text(),
        }

    return data


def create_session_json_element(value: Any) -> str:
    return (
        '    <script type="application/json" id="session-data">\n'
        f"        {json.dumps(value)}\n"
        "    </script>"
    )


def create_session_file(
    handle: str, data: dict[str, str], output_file: Optional[Path], template_file: Path
) -> Path:
    output_file = Path(output_file) if output_file else None
    current_dir = Path(__file__).parent.resolve()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_id = os.urandom(4).hex()

    if output_file is not None:
        if not output_file.suffix == ".html":
            output_file = output_file.with_suffix(".html")
        session_path = output_file
    else:
        session_path = current_dir / f"{timestamp}_{random_id}_{handle}.html"
    session_path.touch()

    pae_template = str(template_file)
    with (
        open(pae_template, "r") as template_file_obj,
        open(session_path, "w") as session_file,
    ):
        template = template_file_obj.read()
        template = template.replace(
            "</head>", f"\n{create_session_json_element(data)}\n</head>"
        )

        session_file.write(template)

    return session_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-s",
        "--structure",
        help="path to structure file",
        type=resolved_path,
        required=True,
    )

    parser.add_argument(
        "-r",
        "--scores",
        help="path to JSON file containing scores",
        type=resolved_path,
        required=True,
    )

    parser.add_argument(
        "-l",
        "--labels",
        help="semicolon-separated list of chain labels",
        type=str,
    )

    parser.add_argument(
        "-c",
        "--crosslinks",
        help="path to CSV containing crosslinks",
        type=resolved_path,
    )

    parser.add_argument(
        "-p",
        "--port",
        help="port of the local file server",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "-o",
        "--output_file",
        help="output file",
    )
    parser.add_argument(
        "-t",
        "--template_file",
        help="template file",
        type=resolved_path,
        default="pae-viewer.html",
    )

    args = parser.parse_args()

    load_pae_viewer(
        args.structure,
        args.labels,
        args.template_file,
        args.scores,
        args.crosslinks,
        args.port,
        args.output_file,
    )
