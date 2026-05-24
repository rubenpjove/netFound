"""
osfing/predict.py — Classification inference with a fine-tuned netFound model.

Loads a fine-tuned ``netFoundFinetuningModel`` checkpoint, runs inference on a
test Arrow dataset, and writes a ``predictions.tsv`` file compatible with the
NTFM-OSfing pipeline's ``metric_evaluation.py`` step.

Output format (identical to ET-BERT ``predictions.tsv``)::

    label
    0
    2
    1
    ...

Label mapping
-------------
The Arrow dataset produced by ``osfing/preprocess.py`` stores string class
names (e.g. ``"Windows 10"``) in the ``labels`` column.  During fine-tuning,
``netFoundFinetuning.py`` uses ``sklearn.LabelEncoder`` fitted on the sorted
unique labels to map them to integers.  We reconstruct the same encoder here
from ``label_maps.json`` (sorted keys of ``label2id``) so that the predicted
integer indices are consistent with the training run.  The final ``predictions.tsv``
uses our pipeline's integer IDs (from ``label2id``), not the LabelEncoder's indices.

Usage::

    python osfing/predict.py \\
        --model_dir       /path/to/finetuned_model \\
        --test_arrow      /path/to/test/arrow_dir \\
        --label_maps      /path/to/label_maps.json \\
        --prediction_path /path/to/predictions.tsv \\
        --size            base \\
        [--batch_size 32] \\
        [--num_workers 4] \\
        [--no_cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True,
                   help="Directory with the fine-tuned HuggingFace model (config.json + weights).")
    p.add_argument("--test_arrow", required=True,
                   help="Directory containing test *.arrow files.")
    p.add_argument("--label_maps", required=True,
                   help="Path to label_maps.json produced by the pipeline preprocessing step.")
    p.add_argument("--prediction_path", required=True,
                   help="Output path for predictions.tsv.")
    p.add_argument("--size", default="base", choices=["small", "base", "large"],
                   help="Model size — must match the fine-tuned checkpoint. Default: base.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Inference batch size per device.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes.")
    p.add_argument("--no_cuda", action="store_true", default=False,
                   help="Force CPU inference (not recommended for large models).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 — complexity acceptable for a standalone script
    args = _make_arg_parser().parse_args()

    # ── Heavy imports after PYTHONPATH is set (called as subprocess from pipeline) ──
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.preprocessing import LabelEncoder
    from transformers import TrainingArguments

    from modules import utils
    from modules.netFoundDataCollator import DataCollatorForFlowClassification
    from modules.netFoundModels import netFoundFinetuningModel
    from modules.netFoundTokenizer import netFoundTokenizer
    from modules.netFoundTrainer import netFoundTrainer

    # ── 1. Label mapping ───────────────────────────────────────────────────────
    logger.info("Loading label maps from %s", args.label_maps)
    with open(args.label_maps, encoding="utf-8") as fh:
        label_maps: dict = json.load(fh)
    label2id: dict[str, int] = label_maps["label2id"]

    # Reconstruct the LabelEncoder fitted during training (same sorted order).
    all_label_names = sorted(label2id.keys())
    label_encoder = LabelEncoder()
    label_encoder.fit(all_label_names)

    # ── 2. Config & model ──────────────────────────────────────────────────────
    logger.info("Loading fine-tuned model from %s (size=%s)", args.model_dir, args.size)

    # Build a minimal config, then load pretrained weights.
    # utils.update_config merges dataclass fields into netFoundConfig;
    # using SimpleNamespace avoids having to import the exact dataclasses.
    import types

    model_ns = types.SimpleNamespace(
        model_name_or_path=args.model_dir,
        size=args.size,
        # All other ModelArguments fields default to None (skipped by update_config).
        metaFeatures=None, num_hidden_layers=None, num_attention_heads=None,
        hidden_size=None, no_ptm=None, freeze_flow_encoder=None,
        freeze_burst_encoder=None, freeze_embeddings=None, freeze_base=None,
        use_flash_attn=None, roformer=None, compile=None, strip_payload=None,
    )
    data_ns = types.SimpleNamespace(
        train_dir=args.test_arrow, test_dir=None,
        no_meta=None, flat=None, limit_bursts=None,
        validation_dir=None, validation_split_percentage=None,
        data_cache_dir=None, overwrite_cache=None,
        max_bursts=None, max_seq_length=None,
        preprocessing_num_workers=args.num_workers,
        max_train_samples=None, max_eval_samples=None,
        streaming=False, tcpoptions=None, profile=None,
    )

    # TrainingArguments needs output_dir; use prediction file's parent dir.
    output_dir = str(Path(args.prediction_path).parent)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        use_cpu=args.no_cuda,       # transformers 5.x renamed no_cuda → use_cpu
        do_train=False,
        do_eval=False,
        report_to="none",          # disable MLflow / WandB during inference
    )

    config = utils.update_config(model_ns, data_ns, training_args, config=None)
    config.p = 0  # disable augmentation

    model = netFoundFinetuningModel.from_pretrained(args.model_dir, config=config)
    # Re-initialize position_ids buffer (defensive — same fix as netFoundFinetuning.py).
    # If the saved checkpoint accidentally persisted position_ids with Roberta-style offsets,
    # this guarantees correct [0..max_position_embeddings-1] values to avoid CUDA OOB.
    model.base_transformer.embeddings.register_buffer(
        "position_ids",
        torch.arange(config.max_position_embeddings).expand((1, -1)),
        persistent=False,
    )
    model.eval()

    # ── 3. Dataset ─────────────────────────────────────────────────────────────
    logger.info("Loading test Arrow dataset from %s", args.test_arrow)
    raw_dataset = load_dataset("arrow", data_dir=args.test_arrow)
    # load_dataset returns DatasetDict; take the first (and only) split.
    if hasattr(raw_dataset, "keys"):
        split_key = list(raw_dataset.keys())[0]
        dataset = raw_dataset[split_key]
    else:
        dataset = raw_dataset

    # total_bursts column is required by the tokenizer/collator; the tokenizer will
    # overwrite it with real per-flow burst counts during the map() below. Only add
    # the placeholder if the raw Arrow dataset does not already have this column.
    if "total_bursts" not in dataset.column_names:
        dataset = dataset.add_column("total_bursts", [0] * len(dataset))

    # Encode string labels → LabelEncoder int (same mapping as during training).
    def _encode_labels(batch: dict) -> dict:
        batch["labels"] = label_encoder.transform(batch["labels"]).tolist()
        return batch

    dataset = dataset.map(_encode_labels, batched=True, desc="Encoding labels")

    # Tokenise (no augmentation: p=0 already set on config).
    tokenizer = netFoundTokenizer(config=config)
    tokenizer.pretraining = False
    dataset = dataset.map(
        function=tokenizer,
        batched=True,
        num_proc=args.num_workers,
        desc="Tokenising",
    )

    # Drop intermediate columns that the collator does not expect.
    for col in ("burst_tokens", "directions", "counts"):
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    # ── 4. Inference ───────────────────────────────────────────────────────────
    data_collator = DataCollatorForFlowClassification(
        pad_token_id=tokenizer.pad_token_id,
        labels_dtype=torch.long,
    )

    trainer = netFoundTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Running inference on %d samples…", len(dataset))
    prediction_output = trainer.predict(dataset)
    # prediction_output.predictions: ndarray of shape [N, num_labels] (logits)
    # prediction_output.label_ids: ndarray [N] of the TRUE labels in the *same order*
    # as predictions — this is the only order-safe source of ground truth, since a
    # separate load_dataset() of the Arrow shards may concatenate them in a different
    # order than the Trainer's prediction dataloader.
    pred_le_ids: np.ndarray = np.argmax(prediction_output.predictions, axis=-1)
    true_le_ids = prediction_output.label_ids

    # ── 5. Decode predictions + ground truth → pipeline integer IDs ─────────────
    pred_str_labels = label_encoder.inverse_transform(pred_le_ids)
    pred_our_ids = [label2id[lbl] for lbl in pred_str_labels]
    if true_le_ids is not None:
        true_str_labels = label_encoder.inverse_transform(np.asarray(true_le_ids).astype(int))
        true_our_ids = [label2id[lbl] for lbl in true_str_labels]
    else:
        true_our_ids = [-1] * len(pred_our_ids)

    # ── 6. Write predictions.tsv ───────────────────────────────────────────────
    # Two columns: predicted label and ground-truth label, perfectly aligned (both
    # come from the same Trainer.predict() output). Downstream metric_evaluation and
    # plots read the ground truth from column 2 for netFound, avoiding any Arrow
    # re-read ordering mismatch. Column 1 ("label") keeps ET-BERT compatibility.
    out_path = Path(args.prediction_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("label\ttrue\n")
        for pid, tid in zip(pred_our_ids, true_our_ids):
            fh.write(f"{pid}\t{tid}\n")

    logger.info("Wrote %d predictions (with aligned ground truth) to %s", len(pred_our_ids), out_path)


if __name__ == "__main__":
    main()
