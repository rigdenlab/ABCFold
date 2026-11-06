from abcfold.output.cofolding_common import SeedSampleOutput


class IntellifoldOutput(SeedSampleOutput):
    """
    Object to process the output of an IntelliFold run.

    IntelliFold v2 drives a vendored AlphaFold 3 engine, so it writes the
    AlphaFold 3 layout: one ``seed-<seed>_sample-<n>`` directory per model
    holding ``*_model.cif``, ``*_summary_confidences.json`` and
    ``*_confidences.json``, with pLDDT already on a 0-100 scale.
    """

    results_prefix = "intellifold_results"
    label = "intellifold"
    sample_token = "_sample-"
    plddt_is_fraction = False

    def seed_for(self, output_dir, pathway):
        # Models live in <output>/<job>/seed-<seed>_sample-<n>/, so the seed is
        # read from the model directory rather than the run directory.
        for part in pathway.parts[::-1]:
            if part.startswith("seed-"):
                return part.split("_")[0].replace("seed-", "")
        return super().seed_for(output_dir, pathway)
