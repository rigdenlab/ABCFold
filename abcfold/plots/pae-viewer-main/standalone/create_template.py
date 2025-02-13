import os
from pathlib import Path

from jinja2 import Template

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>{{ title }}</title>
    <meta http-equiv="content-type" content="application/xhtml+xml; charset=utf-8"/>
    <meta name="viewport" content="width=device-width, minimum-scale=1.0, maximum-scale=1.0"/>
    <meta name="theme-color" content="#ffffff">

    <link rel="icon" type="image/svg+xml" href="{{ favicon_svg }}">
    <link rel="icon" sizes="any" href="{{ favicon_ico }}" >

    <link rel="stylesheet" type="text/css" href="{{ bootstrap_min_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ bootstrap_css }}"/>

    <link rel="stylesheet" type="text/css" href="{{ pae_viewer_standalone_layout_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ common_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ structure_carousel_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ structure_viewer_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ structure_panel_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ pae_viewer_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ complex_sequence_viewer_css }}"/>
    <link rel="stylesheet" type="text/css" href="{{ pae_viewer_standalone_overrides_css }}"/>

    <script type="text/javascript" src="{{ jquery_min_js }}"></script>
    <script type="text/javascript" src="{{ jquery_ui_min_js }}"></script>
    <script type="text/javascript" src="{{ bootstrap_bundle_min_js }}"></script>
    <script type="text/javascript" src="{{ ngl_js }}"></script>
    <script type="text/javascript" src="{{ chroma_min_js }}"></script>
    <script type="text/javascript" src="{{ file_saver_min_js }}"></script>

    <script defer src="{{ setup_js }}"></script>

    <script defer type="module" src="{{ structure_carousel_js }}"></script>
    <script defer type="module" src="{{ pae_viewer_js }}"></script>
    <script defer type="module" src="{{ complex_sequence_viewer_js }}"></script>
</head>
<body>

<input type="hidden" id="relativePath" value="{{ relative_path }}">

<noscript>
    <div class="alert alert-danger text-center">
        <strong>Error:</strong> JavaScript is disabled or not supported
        by your browser, but this page requires it to work.
    </div>
</noscript>
<div id="container">
    <div class="header" style="background-color: white">
        <img class="logo" src="../.feature_viewer/static/ABCfold-logo.svg" alt="ABCFold">
        <a href="https://github.com/rigdenlab/ABCFold.git" class="github-corner" aria-label="View source on Github"><svg width="80" height="80" viewBox="0 0 250 250" style="fill:#343635; color:#fff; float: right; border: 0;" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a><style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>
    </div>
    <main>
        {{ pae_viewer_demo_tpl}}
    </main>
    <footer class="footer">
    </footer>
</div>
</body>
</html>
"""  # noqa


def render_html(args, output_file):

    template = Template(TEMPLATE)
    # create a relative link between the current directory and the template

    relative_path = Path(os.path.relpath(args.src_path, Path(output_file).parent))

    rendered_html = template.render(
        title=args.title,
        favicon_svg=str(relative_path.joinpath("standalone", "favicon.svg")),
        favicon_ico=str(relative_path.joinpath("standalone", "favicon.ico")),
        bootstrap_min_css=str(
            relative_path.joinpath("standalone", "libs", "bootstrap.min.css")
        ),
        bootstrap_css=str(
            relative_path.joinpath("standalone", "libs", "bootstrap@5.1.3.css")
        ),
        pae_viewer_standalone_layout_css=str(
            relative_path.joinpath("standalone", "css", args.standalonecss)
        ),
        common_css=str(relative_path.joinpath("standalone", "css", "common.css")),
        structure_carousel_css=str(
            relative_path.joinpath("src", "css", "structure-carousel.css")
        ),
        structure_viewer_css=str(
            relative_path.joinpath("src", "css", "structure-viewer.css")
        ),
        structure_panel_css=str(
            relative_path.joinpath("src", "css", "structure-panel.css")
        ),
        pae_viewer_css=str(relative_path.joinpath("src", "css", "pae-viewer.css")),
        complex_sequence_viewer_css=str(
            relative_path.joinpath("src", "css", "complex-sequence-viewer.css")
        ),
        pae_viewer_standalone_overrides_css=str(
            relative_path.joinpath(
                "standalone", "css", "paeViewerStandaloneOverrides.css"
            )
        ),
        jquery_min_js=str(
            relative_path.joinpath("standalone", "libs", "jquery.min.js")
        ),
        jquery_ui_min_js=str(
            relative_path.joinpath("standalone", "libs", "jquery-ui.min.js")
        ),
        bootstrap_bundle_min_js=str(
            relative_path.joinpath("standalone", "libs", "bootstrap.bundle.min.js")
        ),
        ngl_js=str(relative_path.joinpath("standalone", "libs", "ngl.js")),
        chroma_min_js=str(
            relative_path.joinpath("standalone", "libs", "chroma.min.js")
        ),
        file_saver_min_js=str(
            relative_path.joinpath("standalone", "libs", "FileSaver.min.js")
        ),
        setup_js=str(relative_path.joinpath("standalone", "js", "setup.js")),
        structure_carousel_js=str(
            relative_path.joinpath("src", "modules", "structure-carousel.js")
        ),
        pae_viewer_js=str(relative_path.joinpath("src", "modules", "pae-viewer.js")),
        complex_sequence_viewer_js=str(
            relative_path.joinpath("src", "modules", "complex-sequence-viewer.js")
        ),
        header_title=args.title,
        header_subtitle="Interactive display of the Predicted Aligned Error",
        relative_path=relative_path,
        pae_viewer_demo_tpl="{{paeViewerDemo.tpl}}",
    )

    return rendered_html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--output_file",
        help="Output file",
        type=str,
        default="pae-viewer.html",
    )

    parser.add_argument(
        "--title",
        help="Title of the HTML page",
        type=str,
        default="PAE Viewer",
    )
    parser.add_argument(
        "--standalonecss",
        type=str,
        default="paeViewerStandaloneLayout.css",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    rendered_html = render_html(args, output_file=args.output_file)

    # write html file
    with open(args.output_file, "w") as f:
        f.write(rendered_html)
