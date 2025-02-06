# import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo

from abcfold.processoutput.file_handlers import CifFile

logger = logging.getLogger("logger")


def plot_plddt(
    cif_models_dict: Dict[str, List[CifFile]],
    output_name: Union[str, Path],
    line_width: float = 1.6,
    dash: str = "dot",
    chain_line_occupancy: float = 0.8,
    show: bool = False,
    include_plotlyjs: bool = True,
) -> None:
    """
    Plots the pLDDT distribution of the models in the dictionary of cif models. Outputs
    and html file with the plot.

    Args:
        cif_models_dict: Dictionary of cif models to plot. The keys are the source of
            the models and the values are lists of CifFile objects.
            e.g. {"Alphafold3": [CifFile, CifFile, ...], "Boltz-1": [CifFile, ...],
            "Chai-1": [CifFile, ...]}

        output_name: Path to the output html file.
        line_width: Width of the lines in the plot.
        dash: Dash style of the lines in the plot.
        chain_line_occupancy: Opacity of the chain lines in the plot.
        show: If True, the plot will be displayed in the browser.

    Returns:
        None

    Outputs:
        An html file with the plot.
    """

    fig = go.Figure()
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    colours = list(px.colors.qualitative.T10)
    colour_index = 0
    added_lines = []

    cif_models = [
        cif_file for cif_files in cif_models_dict.values() for cif_file in cif_files
    ]

    for i, (key, cif_models) in enumerate(cif_models_dict.items()):
        for cif_model in cif_models:
            plddt = cif_model.residue_plddts
            chain_ranges = {
                chain: range(len(plddt))
                for chain, plddt in cif_model.residue_plddt_per_chain.items()
            }
            counter = 0
            for chain, chain_range in chain_ranges.items():
                counter += chain_range[-1]
                chain_name = f"Chain {chain}"

                if chain_name not in added_lines:
                    fig.add_vline(
                        x=counter,
                        line=dict(
                            color=colours[colour_index % len(colours)], dash="dash"
                        ),
                        opacity=chain_line_occupancy,
                        annotation_text=Bold(chain_name),
                        annotation_font_size=15,
                        annotation_position="top left",
                        annotation_textangle=-90,
                    )

                    colour_index += 1
                    added_lines.append(chain_name)

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(plddt))),
                    y=plddt,
                    mode="lines",
                    legendgroup=key,
                    legendgrouptitle_text=Bold(key),
                    name=int(cif_model.name.split("_")[-1]) + 1,
                    line=dict(dash=dash, width=line_width),
                )
            )
    models_no = len(cif_models)
    sources = i + 1

    steps = []
    steps.append(
        dict(
            method="restyle",
            args=[
                {"visible": [True] * models_no * (i + 1)},
                {"title": "All Models"},
            ],
            label="All Models",
        )
    )
    for i in range(sources - 1):
        step = dict(
            method="restyle",
            args=[
                {"visible": [False] * models_no * sources},
                {"title": f"Model {i+1}"},
            ],
            label=f"Model {i+1}",
        )
        for j in range(i, models_no * sources, models_no):
            step_list = step["args"]
            if isinstance(step_list, list):
                step_list[0]["visible"][j] = True
            else:
                logger.error(
                    "Error, MyPy made me change this for some reason and if \
this if failing that means the computer wins."
                )
                raise ValueError()
            # step["args"][0]["visible"][j] = True
        steps.append(step)

    fig.update_layout(
        xaxis_title=Bold("Residue Number"),
        yaxis_title=Bold("pLDDT Score"),
        title=Bold("pLDDT Distribution"),
        legend_title=Bold("Models"),
        plot_bgcolor="white",
        sliders=[dict(steps=steps, currentvalue={"prefix": Bold("Selection: ")})],
    )

    fig.update_layout(
        showlegend=True,
    )

    if show:
        fig.show()

    output_name = Path(output_name)

    if output_name.suffix == "":
        output_name = output_name.with_suffix(".html")

    if include_plotlyjs:
        fig.write_html(str(output_name))
    else:
        div = pyo.plot(fig, include_plotlyjs=False, output_type="div")
        output_name = output_name.with_suffix(".div.html")
        with open(output_name, "w") as f:
            f.write(div)


def Bold(string):
    return f"<b>{string}</b>"
