import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import os
from copy import deepcopy
import torch
import torch.distributed
from dataclasses import field, dataclass
from typing import Optional
from torchinfo import summary

from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from torch.distributed.elastic.multiprocessing.errors import record
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from transformers import HfArgumentParser, TrainingArguments

from modules.metrics import classif_metrics, regression_metrics
from modules import utils
from modules.netFoundDataCollator import DataCollatorForFlowClassification
from modules.netFoundModels import netFoundFinetuningModel
from modules.netFoundTrainer import netFoundTrainer
from modules.netFoundTokenizer import netFoundTokenizer

random.seed(42)


@dataclass
class FineTuningDataTrainingArguments(utils.CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    num_labels: int = field(metadata={"help": "number of classes in the datasets"}, default=None)
    problem_type: Optional[str] = field(
        default=None,
        metadata={"help": "Override regression or classification task"},
    )
    label_maps: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to label_maps.json produced by the pipeline preprocessing step. "
                "When provided, the LabelEncoder is fitted on the full sorted label space "
                "defined there (same as osfing/predict.py), guaranteeing consistent "
                "label→integer mapping regardless of which classes appear in train vs dev."
            )
        },
    )
    p: float = field(
        default=None,
        metadata={
            "help": "noise rate"
        },
    )
    prediction_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "When set together with --do_predict, run inference on test_dir and write "
                "a TSV here with two aligned columns ('label'=predicted, 'true'=ground truth), "
                "both in the pipeline's label2id integer space (from --label_maps). This reuses "
                "the exact eval machinery, replacing the separate osfing/predict.py."
            )
        },
    )
    diagnose_saveload: bool = field(
        default=False,
        metadata={
            "help": (
                "Diagnostic mode: after --do_train + save_model(), reload the model from disk "
                "and compare it against the in-memory model (eval metrics, state_dict diff, raw "
                "safetensors keys, from_pretrained loading_info, forward input-dependence, and "
                "checkpoint-NNN vs top-level). Writes saveload_diagnostic.json to output_dir. "
                "No-op for normal runs."
            )
        },
    )

def get_label_encoder(problem_type: str, dataset = None, batch_size = 1):
    """
    Labels are strings by default because they are stored in arrow format with predefined str column datatype.
    This function returns mapping function that converts them to class numbers (classification) or float (regression)
    which later is mapped over the finetuning dataset.
    """


    if problem_type == "regression":
        encoder = lambda x: float(x)
        encoder.transform = lambda _b: [float(x) for x in _b]
    else:
        if dataset is None:
            raise ValueError("Dataset should be provided for iteration and getting class names")
        encoder = LabelEncoder()
        dataset = dataset.select_columns("labels")
        labels = set()
        for batch in tqdm(dataset.iter(batch_size=batch_size)):
            labels.update(batch["labels"])
        labels = list(sorted(labels))
        encoder.fit(labels)

    def mapping_function(_batch):
        _batch["labels"] = encoder.transform(_batch["labels"])
        return _batch

    return encoder, mapping_function


def load_finetuning_model(config, model_path, logger):
    """Build a netFoundFinetuningModel and load its weights from disk reliably.

    Why not ``from_pretrained``: under transformers 5.8.1, ``from_pretrained``
    SILENTLY FAILS to materialize this custom nested PreTrainedModel — it reports
    a clean load (0 missing/unexpected/mismatched) yet leaves every parameter at
    its random ``__init__`` value, so a reloaded fine-tuned model performs at
    random (eval_loss = ln(num_classes)). Disabling fast-init
    (``low_cpu_mem_usage=False``) does NOT help. Diagnosed with --diagnose_saveload:
    the on-disk safetensors is byte-correct (== the trained weights), and a plain
    ``load_state_dict`` reproduces the in-memory model exactly.

    We therefore build the model and load the state_dict directly. Shape-mismatched
    tensors are skipped (mirrors ``ignore_mismatched_sizes=True``, needed for the
    pretrained checkpoint whose protoEmbedding/MLP shapes differ from the
    fine-tuning head's config); missing/unexpected keys are tolerated (strict=False).
    """
    import glob
    model = netFoundFinetuningModel(config)
    src = {}
    st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            src.update(load_file(f))
    else:
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            src = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model weights (*.safetensors or pytorch_model.bin) found in {model_path}"
            )
    own = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        elif k in own:
            skipped_shape.append((k, list(v.shape), list(own[k].shape)))
    load_res = model.load_state_dict(filtered, strict=False)
    logger.warning(
        "load_finetuning_model(%s): loaded %d/%d tensors (skipped %d shape-mismatched, "
        "%d missing, %d unexpected)",
        model_path, len(filtered), len(src), len(skipped_shape),
        len(load_res.missing_keys), len(load_res.unexpected_keys),
    )
    if skipped_shape:
        logger.warning("  shape-mismatched (kept at init): %s",
                       [s[0] for s in skipped_shape][:20])
    return model


def _cpu_clone_state_dict(model):
    # Strip the torch.compile "_orig_mod." prefix so keys line up with a non-compiled reload.
    def _norm(k):
        return k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
    return {_norm(k): v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _safetensors_keys_shapes(model_dir):
    """Keys + shapes actually stored on disk (handles single or sharded safetensors)."""
    import glob as _glob
    out = {}
    files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover
        return {"__error__": f"safetensors unavailable: {exc}"}, [os.path.basename(f) for f in files]
    for f in files:
        try:
            with safe_open(f, framework="pt") as st:
                for k in st.keys():
                    out[k] = list(st.get_slice(k).get_shape())
        except Exception as exc:
            out[f"__error__{os.path.basename(f)}"] = str(exc)
    return out, [os.path.basename(f) for f in files]


def _state_dict_diff(sd_mem, sd_reload):
    import torch as _torch
    keys_mem, keys_rl = set(sd_mem), set(sd_reload)
    only_in_memory = sorted(keys_mem - keys_rl)
    only_in_reload = sorted(keys_rl - keys_mem)
    differing = []
    for k in sorted(keys_mem & keys_rl):
        a, b = sd_mem[k], sd_reload[k]
        if list(a.shape) != list(b.shape):
            differing.append({"key": k, "reason": "shape", "mem": list(a.shape), "reload": list(b.shape)})
            continue
        a32, b32 = a.float(), b.float()
        if not _torch.allclose(a32, b32, atol=1e-5, rtol=1e-4):
            differing.append({"key": k, "reason": "value", "max_abs_diff": float((a32 - b32).abs().max())})
    return only_in_memory, only_in_reload, differing


def _input_dependence(predictions):
    """Is the model output input-dependent, or collapsed/constant?"""
    import numpy as _np
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    logits = _np.asarray(logits, dtype=_np.float64)
    if logits.ndim != 2:
        logits = logits.reshape(logits.shape[0], -1)
    argmax = logits.argmax(axis=-1)
    uniq, counts = _np.unique(argmax, return_counts=True)
    return {
        "n_samples": int(logits.shape[0]),
        "n_unique_predictions": int(len(uniq)),
        "prediction_class_counts": {int(u): int(c) for u, c in zip(uniq, counts)},
        # ~0 => logits barely change across samples => input-independent (collapsed)
        "mean_logit_std_across_samples": float(logits.std(axis=0).mean()),
    }


def _eval_with_fresh_trainer(model, training_args, test_dataset, testing_tokenizer,
                             compute_metrics, data_collator):
    """Evaluate + predict a (reloaded) model through the SAME machinery as training-time eval."""
    model = model.to(training_args.device)
    model.eval()
    diag_trainer = netFoundTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        processing_class=testing_tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    metrics = diag_trainer.evaluate(eval_dataset=test_dataset)
    pred = diag_trainer.predict(test_dataset)
    return metrics, pred


def run_saveload_diagnostic(logger, in_memory_trainer, config, training_args, test_dataset,
                            data_collator, testing_tokenizer, compute_metrics):
    """
    Localize the netFound save/load bug (reloaded model → random inference).

    Compares the in-memory fine-tuned model against the same model reloaded from disk:
    eval metrics, state_dict tensor-by-tensor diff, raw safetensors keys on disk,
    from_pretrained loading_info (authoritative missing/unexpected/mismatched keys),
    forward input-dependence, and top-level model.safetensors vs latest checkpoint-NNN.

    Writes <output_dir>/saveload_diagnostic.json. Never raises (best-effort diagnostic).
    """
    import math
    import json as _json
    from transformers.trainer_utils import get_last_checkpoint

    out_dir = training_args.output_dir
    report = {"output_dir": out_dir}
    logger.warning("*** SAVE/LOAD DIAGNOSTIC (--diagnose_saveload) ***")

    try:
        num_labels = int(getattr(config, "num_labels", 0) or 0)
        report["num_labels"] = num_labels
        report["expected_random_eval_loss_ln_num_labels"] = math.log(num_labels) if num_labels > 0 else None

        # 1. In-memory baseline (the model that scores ~0.77) + its predictions + state_dict.
        m_inmem = in_memory_trainer.evaluate(eval_dataset=test_dataset)
        pred_inmem = in_memory_trainer.predict(test_dataset)
        sd_mem = _cpu_clone_state_dict(in_memory_trainer.model)
        report["in_memory"] = {
            "eval": {k: float(v) for k, v in m_inmem.items() if isinstance(v, (int, float))},
            "input_dependence": _input_dependence(pred_inmem.predictions),
        }

        # 4/5. Raw safetensors on disk (what save_model actually wrote).
        disk_keys, st_files = _safetensors_keys_shapes(out_dir)
        sd_mem_keys = set(sd_mem.keys())
        report["safetensors"] = {
            "files": st_files,
            "n_keys_on_disk": len([k for k in disk_keys if not k.startswith("__error__")]),
            "n_keys_in_memory_state_dict": len(sd_mem_keys),
            "keys_in_memory_but_not_on_disk": sorted(sd_mem_keys - set(disk_keys)),
            "keys_on_disk_but_not_in_memory": sorted(set(disk_keys) - sd_mem_keys - {
                k for k in disk_keys if k.startswith("__error__")}),
            "errors": {k: v for k, v in disk_keys.items() if k.startswith("__error__")},
        }

        # 2. Reload top-level model with authoritative loading_info.
        reloaded, loading_info = netFoundFinetuningModel.from_pretrained(
            out_dir, config=config, ignore_mismatched_sizes=True, output_loading_info=True
        )
        report["from_pretrained_loading_info"] = {
            "missing_keys": list(loading_info.get("missing_keys", [])),
            "unexpected_keys": list(loading_info.get("unexpected_keys", [])),
            "mismatched_keys": list(loading_info.get("mismatched_keys", [])),
            "error_msgs": list(loading_info.get("error_msgs", [])),
        }

        # 3. State-dict tensor-by-tensor diff (in-memory vs reloaded).
        only_mem, only_rl, differing = _state_dict_diff(sd_mem, reloaded.state_dict())
        report["state_dict_diff"] = {
            "n_keys_differing": len(differing),
            "keys_only_in_memory": only_mem,
            "keys_only_in_reload": only_rl,
            "differing": differing[:200],  # cap for readability
        }

        # 6 + 7. Reloaded eval + input-dependence (same eval machinery).
        m_reload, pred_reload = _eval_with_fresh_trainer(
            reloaded, training_args, test_dataset, testing_tokenizer, compute_metrics, data_collator
        )
        report["reloaded_top_level"] = {
            "eval": {k: float(v) for k, v in m_reload.items() if isinstance(v, (int, float))},
            "input_dependence": _input_dependence(pred_reload.predictions),
        }
        del reloaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 8. Latest checkpoint-NNN vs top-level (TODO diag #1).
        last_ckpt = get_last_checkpoint(out_dir)
        report["last_checkpoint"] = last_ckpt
        if last_ckpt:
            ckpt_disk_keys, ckpt_files = _safetensors_keys_shapes(last_ckpt)
            ckpt_model, ckpt_loading_info = netFoundFinetuningModel.from_pretrained(
                last_ckpt, config=config, ignore_mismatched_sizes=True, output_loading_info=True
            )
            _, _, ckpt_differing = _state_dict_diff(sd_mem, ckpt_model.state_dict())
            m_ckpt, pred_ckpt = _eval_with_fresh_trainer(
                ckpt_model, training_args, test_dataset, testing_tokenizer, compute_metrics, data_collator
            )
            report["reloaded_checkpoint"] = {
                "safetensors_files": ckpt_files,
                "n_keys_on_disk": len([k for k in ckpt_disk_keys if not k.startswith("__error__")]),
                "missing_keys": list(ckpt_loading_info.get("missing_keys", [])),
                "mismatched_keys": list(ckpt_loading_info.get("mismatched_keys", [])),
                "n_state_dict_keys_differing_vs_memory": len(ckpt_differing),
                "eval": {k: float(v) for k, v in m_ckpt.items() if isinstance(v, (int, float))},
                "input_dependence": _input_dependence(pred_ckpt.predictions),
            }
            del ckpt_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 9. Save-vs-load determination + candidate-fix bake-off (all in one job).
        #    Decides whether the on-disk file is correct (=> load bug) or corrupt
        #    (=> save bug), and tests reload strategies that bypass the fragile
        #    fast-init/meta-device path. The winner is the one whose eval matches
        #    the in-memory model (~the in_memory eval_loss) with ~0 differing keys.
        st_path = os.path.join(out_dir, "model.safetensors")
        file_sd = None
        try:
            from safetensors.torch import load_file as _load_file
            file_sd = _load_file(st_path)
            _, _, file_diff = _state_dict_diff(sd_mem, file_sd)
            report["file_vs_memory"] = {
                "n_keys_differing": len(file_diff),
                "interpretation": (
                    "SAVE bug: on-disk file != trained weights"
                    if len(file_diff) else
                    "file == trained weights => LOAD bug (from_pretrained not materializing)"
                ),
                "sample": file_diff[:10],
            }
        except Exception as exc:
            report["file_vs_memory"] = {"error": f"{type(exc).__name__}: {exc}"}

        fixes = {}
        # Candidate A: from_pretrained with fast-init disabled.
        try:
            mA, _ = netFoundFinetuningModel.from_pretrained(
                out_dir, config=config, ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False, output_loading_info=True,
            )
            _, _, dA = _state_dict_diff(sd_mem, mA.state_dict())
            eA, _ = _eval_with_fresh_trainer(
                mA, training_args, test_dataset, testing_tokenizer, compute_metrics, data_collator)
            fixes["from_pretrained_low_cpu_mem_usage_false"] = {
                "n_state_dict_keys_differing_vs_memory": len(dA),
                "eval_loss": float(eA.get("eval_loss")) if eA.get("eval_loss") is not None else None,
                "eval_accuracy": float(eA.get("eval_accuracy")) if eA.get("eval_accuracy") is not None else None,
            }
            del mA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            import traceback as _tb
            fixes["from_pretrained_low_cpu_mem_usage_false"] = {"error": f"{type(exc).__name__}: {exc}", "tb": _tb.format_exc()[-800:]}

        # Candidate B: fresh model() + manual load_state_dict from the safetensors file.
        if file_sd is not None:
            try:
                mB = netFoundFinetuningModel(config)
                load_res = mB.load_state_dict(file_sd, strict=False)
                _, _, dB = _state_dict_diff(sd_mem, mB.state_dict())
                eB, _ = _eval_with_fresh_trainer(
                    mB, training_args, test_dataset, testing_tokenizer, compute_metrics, data_collator)
                fixes["fresh_model_load_state_dict"] = {
                    "n_missing": len(load_res.missing_keys),
                    "n_unexpected": len(load_res.unexpected_keys),
                    "n_state_dict_keys_differing_vs_memory": len(dB),
                    "eval_loss": float(eB.get("eval_loss")) if eB.get("eval_loss") is not None else None,
                    "eval_accuracy": float(eB.get("eval_accuracy")) if eB.get("eval_accuracy") is not None else None,
                }
                del mB
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback as _tb
                fixes["fresh_model_load_state_dict"] = {"error": f"{type(exc).__name__}: {exc}", "tb": _tb.format_exc()[-800:]}
        report["candidate_fixes"] = fixes

        # Headline verdict.
        inmem_loss = report["in_memory"]["eval"].get("eval_loss")
        reload_loss = report["reloaded_top_level"]["eval"].get("eval_loss")
        report["verdict"] = {
            "in_memory_eval_loss": inmem_loss,
            "reloaded_eval_loss": reload_loss,
            "reload_collapsed": (
                reload_loss is not None
                and report["expected_random_eval_loss_ln_num_labels"] is not None
                and abs(reload_loss - report["expected_random_eval_loss_ln_num_labels"]) < 0.15
            ),
        }
    except Exception as exc:  # never crash the run on a diagnostic failure
        import traceback
        report["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        report["diagnostic_traceback"] = traceback.format_exc()
        logger.warning("save/load diagnostic raised: %s", exc)

    out_path = os.path.join(out_dir, "saveload_diagnostic.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            _json.dump(report, fh, indent=2, default=str)
        logger.warning("Wrote save/load diagnostic to %s", out_path)
    except Exception as exc:
        logger.warning("Could not write %s: %s", out_path, exc)
    logger.warning("DIAGNOSTIC SUMMARY: %s", _json.dumps(report.get("verdict", {}), default=str))
    logger.warning("DIAGNOSTIC state_dict differing keys: %s", report.get("state_dict_diff", {}).get("n_keys_differing"))
    logger.warning("DIAGNOSTIC from_pretrained missing/mismatched: %s / %s",
                   report.get("from_pretrained_loading_info", {}).get("missing_keys"),
                   report.get("from_pretrained_loading_info", {}).get("mismatched_keys"))
    logger.warning("DIAGNOSTIC file_vs_memory: %s", _json.dumps(report.get("file_vs_memory", {}).get("interpretation")
                                                                or report.get("file_vs_memory", {}), default=str))
    logger.warning("DIAGNOSTIC candidate_fixes: %s", _json.dumps(report.get("candidate_fixes", {}), default=str))
    return report


@record
def main():
    parser = HfArgumentParser(
        (utils.ModelArguments, FineTuningDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.streaming:
        raise NotImplementedError("Streaming mode for fine-tuning is not implemented.")

    utils.LOGGING_LEVEL = training_args.get_process_log_level()
    logger = utils.get_logger(name=__name__)

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    # Data preparation
    train_dataset, test_dataset = utils.load_train_test_datasets(logger, data_args)

    # Build the LabelEncoder from the canonical label space in label_maps.json when
    # available (same approach as osfing/predict.py).  This guarantees a consistent
    # label→integer mapping regardless of which classes happen to appear in train vs
    # dev, and keeps training and inference encoders in perfect sync.
    # Fallback: fit on the union of train + dev so we never crash on unseen labels.
    if data_args.label_maps and os.path.exists(data_args.label_maps):
        logger.info("Fitting LabelEncoder from label_maps.json: %s", data_args.label_maps)
        with open(data_args.label_maps, encoding="utf-8") as _fh:
            _label_maps = json.load(_fh)
        _all_label_names = sorted(_label_maps["label2id"].keys())
        label_encoder = LabelEncoder()
        label_encoder.fit(_all_label_names)

        def le_mapping_function(_batch):
            _batch["labels"] = label_encoder.transform(_batch["labels"]).tolist()
            return _batch
    else:
        logger.warning(
            "label_maps not provided or not found; fitting LabelEncoder on train+dev union. "
            "Pass --label_maps for guaranteed consistency with inference."
        )
        _all_splits = concatenate_datasets([train_dataset, test_dataset])
        label_encoder, le_mapping_function = get_label_encoder(
            data_args.problem_type, _all_splits, batch_size=1024
        )
    train_dataset = train_dataset.map(function=le_mapping_function, batched=True)
    test_dataset = test_dataset.map(function=le_mapping_function, batched=True)

    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    if "WORLD_SIZE" in os.environ:
        train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]),
                                              world_size=int(os.environ["WORLD_SIZE"]))
        test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]),
                                             world_size=int(os.environ["WORLD_SIZE"]))

    logger.warning("Tokenizing datasets")
    config = utils.update_config(model_args, data_args, training_args, config=None)
    training_tokenizer = netFoundTokenizer(config=config)

    test_config = deepcopy(config)
    test_config.p = 0
    testing_tokenizer = netFoundTokenizer(config=test_config)

    if "WORLD_SIZE" in os.environ and training_args.local_rank > 0:
        logger.warning("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    params = {
        "batched": True,
        "num_proc": data_args.preprocessing_num_workers or utils.get_90_percent_cpu_count(),
    }
    train_dataset = train_dataset.map(function=training_tokenizer, **params)
    test_dataset = test_dataset.map(function=testing_tokenizer, **params)

    if "WORLD_SIZE" in os.environ and training_args.local_rank == 0:
        logger.warning("Loading results from main process")
        torch.distributed.barrier()

    # Model initialization
    labels_dtype = torch.float32 if data_args.problem_type == "regression" else torch.long
    data_collator = DataCollatorForFlowClassification(training_tokenizer.pad_token_id, labels_dtype)
    if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
    ):
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        # NOTE: do NOT use netFoundFinetuningModel.from_pretrained here — under
        # transformers 5.8.1 it silently fails to materialize this custom nested
        # model (reports a clean load but leaves all weights at random init), so a
        # reloaded fine-tuned model performs at random. load_finetuning_model builds
        # the model and loads the state_dict directly (verified to reproduce the
        # in-memory model exactly). See its docstring + TODO.md "Bug bloqueante".
        model = load_finetuning_model(config, model_args.model_name_or_path, logger)
    else:
        model = netFoundFinetuningModel(config=config)
    model = utils.possibly_freeze(model, model_args)
    if os.environ.get("RANK", "0") == "0":
        summary(model)
    if config.compile:
        model = torch.compile(model, mode="max-autotune")

    if data_args.problem_type == "regression":
        compute_metrics = regression_metrics
    else:
        compute_metrics = lambda p: classif_metrics(p, label_encoder)

    training_args.accelerator_config.dispatch_batches = False
    callbacks: list = []
    if data_args.profile:
        callbacks.append(utils.TorchTBProfilerCallback())
    trainer = netFoundTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        processing_class=testing_tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    utils.init_tbwriter(training_args.output_dir)
    trainer.add_callback(utils.StepSyncCallback())
    trainer.add_callback(utils.LearningRateLogCallback(utils.TB_WRITER))
    trainer.add_callback(utils.ThroughputTimingCallback(utils.TB_WRITER))
    utils.start_gpu_logging(training_args.output_dir)
    utils.start_cpu_logging(training_args.output_dir)
    utils.start_ram_logging(training_args.output_dir)

    utils.verify_checkpoint(logger, training_args)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if getattr(data_args, "diagnose_saveload", False):
            run_saveload_diagnostic(
                logger=logger,
                in_memory_trainer=trainer,
                config=config,
                training_args=training_args,
                test_dataset=test_dataset,
                data_collator=data_collator,
                testing_tokenizer=testing_tokenizer,
                compute_metrics=compute_metrics,
            )

    if training_args.do_eval:
        logger.warning("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and data_args.prediction_path and data_args.problem_type != "regression":
        # Inference path for the NTFM-OSfing pipeline. Runs the SAME trainer / model /
        # tokenizer / LabelEncoder used by --do_eval (which is the machinery that yields
        # the correct accuracy), then writes predictions.tsv with two aligned columns
        # ('label'=predicted, 'true'=ground truth) in the pipeline's label2id space.
        # Replaces the divergent standalone osfing/predict.py.
        import json as _json
        import numpy as _np
        logger.warning("*** Predict → %s ***", data_args.prediction_path)
        pred_out = trainer.predict(test_dataset)
        preds_arr = pred_out.predictions[0] if isinstance(pred_out.predictions, tuple) else pred_out.predictions
        pred_le = _np.asarray(preds_arr).argmax(axis=-1).astype(int)
        true_le = _np.asarray(pred_out.label_ids).astype(int)
        pred_str = label_encoder.inverse_transform(pred_le)
        true_str = label_encoder.inverse_transform(true_le)
        with open(data_args.label_maps, encoding="utf-8") as _fh:
            _l2id = _json.load(_fh)["label2id"]
        out_path = data_args.prediction_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as _out:
            _out.write("label\ttrue\n")
            for _p, _t in zip(pred_str, true_str):
                _out.write(f"{_l2id[str(_p)]}\t{_l2id[str(_t)]}\n")
        logger.warning("Wrote %d predictions (label<TAB>true) to %s", len(pred_le), out_path)


if __name__ == "__main__":
    main()
