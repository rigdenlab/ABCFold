from abcfold.output.cofolding_common import SeedSampleOutput


class OpenDDEOutput(SeedSampleOutput):
    """
    Object to process the output of an OpenDDE run.

    OpenDDE writes ``<out>/<job>/seed_<seed>/predictions/`` holding
    ``<job>_sample_<rank>.cif``,
    ``<job>_summary_confidence_sample_<rank>.json`` and
    ``<job>_full_data_sample_<rank>.json``. Samples are already ordered by
    ranking score, and per-atom pLDDT is written into the B-factor column on a
    0-100 scale.
    """

    results_prefix = "opendde_results"
    label = "opendde"
    sample_token = "_sample_"
    plddt_is_fraction = False

    def seed_for(self, output_dir, pathway):
        for part in pathway.parts[::-1]:
            if part.startswith("seed_"):
                return part.replace("seed_", "")
        return super().seed_for(output_dir, pathway)

    def pae_data(self, scores: dict):
        # OpenDDE names the matrix token_pair_pae in its full data file.
        return scores.get("token_pair_pae", scores.get("pae"))
