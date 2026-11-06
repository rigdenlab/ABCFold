from abcfold.output.cofolding_common import SeedSampleOutput


class Esmfold2Output(SeedSampleOutput):
    """
    Object to process the output of an ESMFold2 run.

    ESMFold2 returns results in memory, so abcfold's runner writes them as
    ``esmfold2_results_seed-<seed>/<job>_seed-<seed>_sample-<n>_model.cif``
    alongside ``*_summary_confidences.json`` and ``*_confidences.json``.
    Per-token pLDDT is stored in the B-factor column on a 0-1 scale.
    """

    results_prefix = "esmfold2_results"
    label = "esmfold2"
    sample_token = "_sample-"
    plddt_is_fraction = True
