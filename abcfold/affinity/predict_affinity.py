# This code is adapted from Boltzina (https://github.com/ohuelab/boltzina)
# Licensed under the MIT License
# Original source: github.com/ohuelab/boltzina (affinity/predict_affinity.py)
# Modifications: [brief description of changes, if any]

from dataclasses import asdict
from pathlib import Path

from boltz.data.types import Manifest
from boltz.main import (Boltz2DiffusionParams, MSAModuleArgs, PairformerArgsV2,
                        get_cache_path)
from boltzina.data.module.inferencev2 import Boltz2InferenceDataModule
from boltzina.data.write.writer import BoltzAffinityWriter
from boltzina.model.models.boltz2 import Boltz2
from pytorch_lightning import Trainer, seed_everything


def load_boltz2_model(affinity_checkpoint=None,
                      sampling_steps_affinity=200,
                      diffusion_samples_affinity=5,
                      subsample_msa=True,
                      num_subsampled_msa=1024,
                      model="boltz2",
                      step_scale=None,
                      affinity_mw_correction=False,
                      skip_run_structure=True,
                      confidence_prediction=False,
                      use_kernels=False,
                      run_trunk_and_structure=True,
                      predict_affinity_args=None,
                      pairformer_args=None,
                      msa_args=None,
                      steering_args=None,
                      diffusion_process_args=None):
    """Load and return a Boltz2 model for affinity prediction.

    Args:
        affinity_checkpoint: Path to the affinity checkpoint file
        sampling_steps_affinity: Number of sampling steps
        diffusion_samples_affinity: Number of diffusion samples
        subsample_msa: Whether to subsample MSA
        num_subsampled_msa: Number of MSA sequences to subsample
        model: Model type ("boltz2")
        step_scale: Step scale for diffusion
        affinity_mw_correction: Whether to apply molecular weight correction

    Returns:
        Loaded Boltz2 model instance ready for inference
    """
    cache_dir = get_cache_path()
    cache_dir = Path(cache_dir)

    if affinity_checkpoint is None:
        affinity_checkpoint = cache_dir / "boltz2_aff.ckpt"

    if not affinity_checkpoint.exists():
        raise FileNotFoundError(
            f"Affinity checkpoint not found at {affinity_checkpoint}"
        )

    predict_affinity_args = {
        "recycling_steps": 5,
        "sampling_steps": sampling_steps_affinity,
        "diffusion_samples": diffusion_samples_affinity,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    } if predict_affinity_args is None else predict_affinity_args

    diffusion_params = (
        asdict(Boltz2DiffusionParams())
        if diffusion_process_args is None
        else diffusion_process_args
    )
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params["step_scale"] = step_scale

    pairformer_args = (
        asdict(PairformerArgsV2())
        if pairformer_args is None
        else pairformer_args
    )

    msa_args = asdict(MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )) if msa_args is None else msa_args

    model_module = Boltz2.load_from_checkpoint(
        affinity_checkpoint,
        strict=True,
        predict_args=predict_affinity_args,
        map_location="cpu",
        diffusion_process_args=diffusion_params,
        ema=False,
        pairformer_args=pairformer_args,
        msa_args=msa_args,
        steering_args={
            "fk_steering": False,
            "physical_guidance_update": False,
            "contact_guidance_update": False,
            "guidance_update": False
            } if steering_args is None else steering_args,
        affinity_mw_correction=affinity_mw_correction,
        skip_run_structure=skip_run_structure,
        run_trunk_and_structure=run_trunk_and_structure,
        confidence_prediction=True,
        use_kernels=use_kernels,
    )
    model_module.confidence_prediction = confidence_prediction
    model_module.eval()

    return model_module


def predict_affinity(out_dir,
                     model_module=None,
                     output_dir=None,
                     structures_dir=None,
                     msa_dir=None,
                     constraints_dir=None,
                     template_dir=None,
                     extra_mols_dir=None,
                     manifest_path=None,
                     affinity_checkpoint=None,
                     sampling_steps_affinity=200,
                     diffusion_samples_affinity=5,
                     subsample_msa=True,
                     num_subsampled_msa=1024,
                     model="boltz2",
                     step_scale=None,
                     override=False,
                     num_workers=1,
                     strategy="auto",
                     accelerator="gpu",
                     devices=1,
                     affinity_mw_correction=False,
                     seed=None,
                     batch_size=1):

    out_dir = Path(out_dir)

    cache_dir = get_cache_path()
    cache_dir = Path(cache_dir)
    mol_dir = cache_dir/'mols'

    if seed is not None:
        seed_everything(seed)

    if model_module is None:
        model_module = load_boltz2_model(
            affinity_checkpoint=affinity_checkpoint,
            sampling_steps_affinity=sampling_steps_affinity,
            diffusion_samples_affinity=diffusion_samples_affinity,
            subsample_msa=subsample_msa,
            num_subsampled_msa=num_subsampled_msa,
            model=model,
            step_scale=step_scale,
            affinity_mw_correction=affinity_mw_correction
        )
    structures_dir = (
        out_dir / "processed" / "structures"
        if structures_dir is None
        else Path(structures_dir)
    )
    msa_dir = out_dir / "processed" / "msa" if msa_dir is None else Path(msa_dir)
    constraints_dir = (
        out_dir / "processed" / "constraints"
        if constraints_dir is None
        else Path(constraints_dir)
    )
    template_dir = (
        out_dir / "processed" / "templates"
        if template_dir is None
        else Path(template_dir)
    )

    manifest = Manifest.load(
        out_dir / "processed" / "manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )

    output_dir = (
        out_dir / "predictions" if output_dir is None else Path(output_dir)
    )
    extra_mols_dir = (
        output_dir / "mols" if extra_mols_dir is None else Path(extra_mols_dir)
    )

    pred_writer = BoltzAffinityWriter(
        data_dir=structures_dir,
        output_dir=output_dir,
    )

    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=output_dir,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        num_workers=num_workers,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
        override_method="other",
        affinity=True,
        batch_size=batch_size,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )
    return trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=True,
    )
