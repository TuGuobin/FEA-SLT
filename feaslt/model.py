"""FEA-SLT with frozen visual features, expression-aware fusion and LoRA."""

import os
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from feaslt.tconv import TemporalConv
from utils.helpers import create_mask
from utils.annotations import load_annotations
from utils.text import context_key
from feaslt.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from feaslt.clip_loss import clip_loss
from feaslt.asb import AbstractSLT
from feaslt.constants import LANG_MAP

from feaslt.fusion_modules import FEAM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


class FEA_SLT(AbstractSLT):
    LANGUAGE_MAPPING = LANG_MAP

    def __init__(
        self,
        tuning_type: str = "lora",
        model_name: Optional[str] = None,
        frame_sample_rate: int = 1,
        expression_sample_rate: int = 8,
        expression_extraction_method: str = "single",
        prompt: str = "",
        input_size: int = 1024,
        expression_size: int = 768,
        fusion_mode: str = "joint",
        inter_hidden: int = 1024,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "./models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        context_anno_path: Optional[str] = None,
        context_seed: int = 0,
        model_dtype: str = "bfloat16",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_expression_aware_fusion: bool = False,
        logit_scale: Optional[float] = None,
        weight_decay: float = 0.01,
        use_label_smoothing: bool = False,
        label_smoothing: float = 0.1,
        use_hard_code: bool = False,
        use_nucleus_sampling: bool = False,
        top_p: float = 0.9,
        temperature: float = 0.8,
        spatial_global_only: bool = False,
        use_temporal: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_temporal = use_temporal
        self.input_size = input_size
        self.spatial_global_only = spatial_global_only
        self.expression_size = expression_size
        self.prompt = prompt
        self.model_name = model_name
        self.is_mbart = "mbart" in model_name.lower() if model_name else False
        self.frame_sample_rate = frame_sample_rate
        self.expression_sample_rate = expression_sample_rate
        self.expression_extraction_method = expression_extraction_method
        self.fusion_modes = [mode.strip() for mode in fusion_mode.split(",")]
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.context_anno_path = context_anno_path
        if model_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("model_dtype must be float32, float16 or bfloat16.")
        self.model_dtype = getattr(torch, model_dtype)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_expression_aware_fusion = use_expression_aware_fusion
        self.logit_scale = logit_scale
        self.weight_decay = weight_decay

        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing
        self.use_hard_code = use_hard_code

        self.use_nucleus_sampling = use_nucleus_sampling
        self.top_p = top_p
        self.temperature = temperature
        self.length_penalty = 1

        self.spatial = "spatial" in self.fusion_modes or "joint" in self.fusion_modes
        self.spatiotemporal = (
            "spatiotemporal" in self.fusion_modes or "joint" in self.fusion_modes
        )
        self.expression = "expression" in self.fusion_modes or "joint" in self.fusion_modes

        if not model_name:
            raise ValueError("Set model_name to a T5, mT5 or mBART checkpoint.")
        if tuning_type not in ("freeze", "lora", "full"):
            raise ValueError("tuning_type must be freeze, lora or full.")
        if not set(self.fusion_modes).issubset({"joint", "spatial", "spatiotemporal", "expression"}):
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if "joint" in self.fusion_modes and len(self.fusion_modes) != 1:
            raise ValueError("Use joint alone, or a comma-separated list of streams.")
        if use_expression_aware_fusion and self.fusion_modes != ["joint"]:
            raise ValueError("Expression-aware fusion is implemented only for fusion_mode: joint.")
        if min(frame_sample_rate, expression_sample_rate, max_frame_len, max_txt_len) < 1:
            raise ValueError("Sampling rates and maximum lengths must be positive.")
        if expression_extraction_method not in ("single", "avgPool", "maxPool"):
            raise ValueError(f"Unknown expression extraction method: {expression_extraction_method}")
        if use_hard_code:
            raise ValueError("use_hard_code is unsupported; provide training context_anno_path instead.")
        if use_resampler:
            raise ValueError("use_resampler is not implemented in this release.")
        self.context_examples = []
        self.fixed_context = ""
        if use_in_context:
            if num_in_context < 1 or not context_anno_path:
                raise ValueError("In-context prompts require num_in_context > 0 and context_anno_path.")
            if os.path.basename(context_anno_path) != "train_ml.npy":
                raise ValueError("context_anno_path must point to train_ml.npy, never dev/test annotations.")
            for record in load_annotations(context_anno_path, "train").values():
                pairs = self._context_pairs(record)
                if len(pairs) >= num_in_context:
                    self.context_examples.append((record["fileid"], context_key(record["text"]), pairs))
            if not self.context_examples:
                raise ValueError(f"Training context requires at least {num_in_context} translations "
                                 "per example; prepare them locally or use configs/no_context.yaml.")
            fixed_example = random.Random(context_seed).choice(self.context_examples)
            self.fixed_context = " ".join(fixed_example[2][:num_in_context])

        self.prepare_models(model_name)

        if tuning_type == "freeze":
            self._freeze_model()
        elif tuning_type == "lora":
            self._apply_lora()

        self.set_container()

        self.save_hyperparameters()

        self.best_bleu1 = 0.0
        self.best_bleu2 = 0.0
        self.best_bleu3 = 0.0
        self.best_bleu4 = 0.0
        self.best_rougeL_f1 = 0.0

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.on_load_checkpoint(checkpoint)
        print(f"Checkpoint is loaded from {checkpoint_path}.")

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5/mBART model."""
        if self.is_mbart:
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q", "v"]

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        print(f"LoRA adapter applied to {self.model_name} model.")

    def _freeze_model(self) -> None:
        """Freeze the T5/mBART model parameters."""
        self.llm.eval()
        for params in self.llm.parameters():
            params.requires_grad = False
        print(f"{self.model_name} model frozen.")

    def train(self, mode=True):
        super().train(mode)
        if self.tuning_type == "freeze":
            self.llm.eval()
        return self

    def on_save_checkpoint(self, checkpoint):
        if self.use_in_context:
            checkpoint["fixed_context"] = self.fixed_context

    def on_load_checkpoint(self, checkpoint):
        if self.use_in_context and "fixed_context" in checkpoint:
            self.fixed_context = checkpoint["fixed_context"]

    def set_container(self) -> None:
        self.generated = []
        self.references = []
        self.result_ids = []
        self.result_languages = []

    def on_validation_epoch_start(self) -> None:
        self.set_container()

    def on_test_epoch_start(self) -> None:
        self.set_container()

    def prepare_models(self, model_name: str) -> None:
        """
        Prepare the textual and visual models.

        Args:
            model_name: Name or path of the T5/mBART model to use
        """

        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=self.cache_dir, torch_dtype=self.model_dtype,
        )
        self.is_mbart = self.llm.config.model_type == "mbart"
        print(f"Loaded {self.llm.config.model_type} model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        if self.spatial:
            # For s^2 ablation: use half input size when only global features are used
            spatial_input_size = (
                self.input_size // 2 if self.spatial_global_only else self.input_size
            )
            self.spatio_proj = build_vision_projector(
                "linear", spatial_input_size, self.inter_hidden
            )
        if self.spatiotemporal:
            self.spatiotemp_proj = build_vision_projector(
                "linear", 1024, self.inter_hidden
            )
        if self.expression:
            self.expression_proj = build_vision_projector(
                "linear", self.expression_size, self.inter_hidden
            )
        self.fusion_proj = build_vision_projector(
            "mlp2x_gelu", self.inter_hidden, self.llm.config.hidden_size
        )

        if self.use_temporal:
            self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        if self.use_expression_aware_fusion:
            self.expression_aware_fusion = FEAM(self.inter_hidden)

        if self.logit_scale is None:
            self.logit_scale = 2.6592
        self.logit_scale = nn.Parameter(torch.tensor(float(self.logit_scale)))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Use the shared Hugging Face/PEFT embedding interface."""
        return self.llm.get_input_embeddings()(input_ids)

    def get_language_code(self, lang: str) -> str:
        """
        Get language code for mBART models.

        Args:
            lang: Language name (e.g., 'English', 'Chinese', 'French')

        Returns:
            Language code for mBART (e.g., 'en_XX', 'zh_CN', 'fr_XX')
        """
        lang_lower = lang.lower()
        if lang_lower not in self.LANGUAGE_MAPPING:
            raise ValueError(f"Unsupported mBART target language: {lang}")
        code = self.LANGUAGE_MAPPING[lang_lower]
        if hasattr(self.tokenizer, "lang_code_to_id") and code not in self.tokenizer.lang_code_to_id:
            raise ValueError(f"Target language {code} is not supported by {self.model_name}.")
        return code

    def prepare_inputs(
        self,
        visual_outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        samples: Dict,
        split: str,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Prepare combined inputs for the T5 model.

        Args:
            visual_outputs: Visual features
            visual_mask: Mask for visual features
            samples: Input samples
            split: Current split (train, val, test)
            batch_idx: Current batch index

        Returns:
            Tuple of (joint_outputs, joint_mask, output_tokens, targets)
        """
        bs = visual_outputs.shape[0]

        prompts = [f"{self.prompt}"] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples["lang"])]

        if self.is_mbart:
            lang_codes = [self.get_language_code(lang) for lang in samples["lang"]]
            if len(set(lang_codes)) != 1:
                raise ValueError("mBART batches must have one target language.")
            self.tokenizer.tgt_lang = lang_codes[0]
            prompts = [
                f"{lang_code} {prompt}"
                for lang_code, prompt in zip(lang_codes, prompts)
            ]

        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples["ex_lang_trans"])]

        input_tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths

        input_embeds = self.get_input_embeddings(input_tokens.input_ids)

        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, : visual_lengths[i], :]
            prompt_embeds = input_embeds[i, : prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)

        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)

        target_kwargs = dict(padding="longest", return_tensors="pt")
        if self.is_mbart:
            output_tokens = self.tokenizer(text_target=samples["text"], **target_kwargs).to(self.device)
        else:
            output_tokens = self.tokenizer(samples["text"], **target_kwargs).to(self.device)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project enabled streams, fuse them, then apply the shared temporal connector."""
        streams, lengths, masks = [], [], []
        for enabled, values, length_key, projector in (
            (self.spatial, "pixel_values", "num_frames", "spatio_proj"),
            (self.spatiotemporal, "glor_values", "glor_lengths", "spatiotemp_proj"),
            (self.expression, "exp_values", "exp_lengths", "expression_proj"),
        ):
            if enabled:
                streams.append(getattr(self, projector)(pad_sequence(samples[values], batch_first=True)))
                lengths.append(samples[length_key])
                masks.append(create_mask(samples[length_key], device=self.device))
        if self.use_expression_aware_fusion:
            visual_outputs, visual_mask = self.expression_aware_fusion(*streams, *masks)
            total_lengths = visual_mask.sum(1)
        else:
            total_lengths = [sum(stream_lengths[index] for stream_lengths in lengths)
                             for index in range(len(samples["text"]))]
            visual_outputs = pad_sequence([
                torch.cat([stream[index, :stream_lengths[index]]
                           for stream, stream_lengths in zip(streams, lengths)], dim=0)
                for index in range(len(samples["text"]))
            ], batch_first=True)
            visual_mask = create_mask(total_lengths, device=self.device)
        if self.use_temporal:
            temporal = self.temporal_encoder(visual_outputs.transpose(1, 2), total_lengths)
            visual_outputs = temporal["visual_feat"].transpose(0, 1)
            visual_mask = create_mask(temporal["feat_len"].tolist(), device=self.device)
        return visual_outputs, visual_mask

    _context_key = staticmethod(context_key)

    @staticmethod
    def _context_pairs(record):
        return [f"{record[f'{lang}_text']}={record['text'].lower()}"
                for lang in ("en", "es", "fr", "de")
                if record.get(f"{lang}_text", "").strip()]

    def _batch_contexts(self, batch):
        if not self.use_in_context:
            return [""] * len(batch)
        if not self.training:
            return [self.fixed_context] * len(batch)
        donors = [
            (sample["id"], context_key(sample["text"]),
             self._context_pairs(sample.get("original_info", sample)))
            for sample in batch
        ]
        candidates = []
        for sample in batch:
            eligible = [index for index, (fileid, text, pairs) in enumerate(donors)
                        if fileid != sample["id"] and text != context_key(sample["text"])
                        and len(pairs) >= self.num_in_context]
            random.shuffle(eligible)
            candidates.append(eligible)
        # A shuffled, one-to-one assignment excludes self and duplicate target sentences.
        owners = {}
        assignments = {}

        def assign(target, visited):
            for donor in candidates[target]:
                if donor in visited:
                    continue
                visited.add(donor)
                if donor not in owners or assign(owners[donor], visited):
                    owners[donor] = target
                    assignments[target] = donor
                    return True
            return False

        if all(assign(target, set()) for target in range(len(batch))):
            return [" ".join(donors[assignments[index]][2][:self.num_in_context])
                    for index in range(len(batch))]
        # Singleton/duplicate-only batches cannot be permuted; use eligible training records.
        contexts = []
        for sample in batch:
            eligible = [pairs for fileid, text, pairs in self.context_examples
                        if fileid != sample["id"] and text != context_key(sample["text"])]
            pairs = random.choice(eligible) if eligible else []
            contexts.append(" ".join(pairs[:self.num_in_context]))
        return contexts

    def _sample_expression(self, feature, source_rate, raw_length, start, frame_length):
        if source_rate < 1 or len(feature) != (raw_length + source_rate - 1) // source_rate:
            raise ValueError("Expression length does not match its extraction interval and frame count.")
        if self.expression_extraction_method != "single":
            if source_rate != 1:
                raise ValueError("Pooling ablations require dense expression features (sample_rate=1).")
            cropped = feature[::self.frame_sample_rate][start:start + frame_length]
            segments = cropped.split(self.expression_sample_rate)
            return torch.stack([segment.mean(0) if self.expression_extraction_method == "avgPool"
                                else segment.max(0).values for segment in segments])
        # s_e is measured on the original video timeline, independently of spatial sampling.
        stride = self.expression_sample_rate
        if stride % source_rate:
            raise ValueError("Model expression interval must be a multiple of the extraction interval.")
        indices = torch.arange(0, len(feature), stride // source_rate, device=feature.device)
        positions = indices * source_rate
        first = start * self.frame_sample_rate
        last = min(raw_length, (start + frame_length) * self.frame_sample_rate)
        indices = indices[(positions >= first) & (positions < last)]
        if not len(indices):
            # A very short crop may fall between two extracted facial frames.
            positions = torch.arange(len(feature), device=feature.device) * source_rate
            indices = ((positions.float() - (first + last - 1) / 2).abs().argmin()).reshape(1)
        return feature[indices]

    def get_inputs(self, batch: List) -> Dict:
        """Keep every sample and only read enabled streams."""
        if not batch:
            raise ValueError("Cannot process an empty batch.")
        inputs = {key: [] for key in (
            "pixel_values", "glor_values", "exp_values", "ids", "text", "gloss",
            "lang", "num_frames", "glor_lengths", "exp_lengths", "ex_lang_trans",
        )}
        inputs["bool_mask_pos"] = []
        inputs["ex_lang_trans"] = self._batch_contexts(batch)
        for sample in batch:
            for key, source in (("ids", "id"), ("text", "text"), ("gloss", "gloss"), ("lang", "lang")):
                value = sample[source]
                inputs[key].append(value.lower() if key == "text" else value)
            features = {}
            for enabled, source, expected in (
                (self.spatial, "pixel_value", self.input_size // 2 if self.spatial_global_only else self.input_size),
                (self.expression, "exp_value", self.expression_size),
            ):
                if not enabled:
                    continue
                feature = sample[source]
                if feature is None or feature.ndim != 2 or not len(feature) or feature.shape[1] != expected:
                    raise ValueError(f"{sample['id']}: {source} must have shape [T, {expected}], T > 0.")
                features[source] = feature
            source_rate = sample.get("expression_sample_rate", 1)
            raw_length = (len(features["pixel_value"]) if self.spatial else
                          sample.get("num_frames", 0) if self.expression else 0)
            if self.expression and raw_length < 1:
                raw_length = (len(features["exp_value"]) - 1) * source_rate + 1
            frame_length = (raw_length + self.frame_sample_rate - 1) // self.frame_sample_rate
            start = 0
            if frame_length > self.max_frame_len:
                excess = frame_length - self.max_frame_len
                start = random.randint(0, excess)
                frame_length = self.max_frame_len
            inputs["num_frames"].append(frame_length)
            if self.spatial:
                inputs["pixel_values"].append(
                    features["pixel_value"][::self.frame_sample_rate][start:start + frame_length])
            if self.expression:
                inputs["exp_values"].append(self._sample_expression(
                    features["exp_value"], source_rate, raw_length, start, frame_length))
            if self.spatiotemporal:
                feature = sample["glor_value"]
                if isinstance(feature, list):
                    feature = torch.cat(feature, dim=0) if feature else None
                if feature is None or feature.ndim != 2 or not len(feature) or feature.shape[1] != 1024:
                    raise ValueError(f"{sample['id']}: motion features must have shape [T, 1024], T > 0.")
                inputs["glor_values"].append(feature)
                inputs["glor_lengths"].append(len(feature))
        inputs["exp_lengths"] = [len(feature) for feature in inputs["exp_values"]]
        return inputs

    def visual_textual_align(
        self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict
    ) -> torch.Tensor:
        """
        Calculate visual-textual alignment loss.

        Args:
            visual_outputs: Visual features
            visual_masks: Mask for visual features
            samples: Input samples

        Returns:
            Contrastive loss
        """
        output_tokens = self.tokenizer(
            samples["text"],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        text_embeds = self.get_input_embeddings(output_tokens.input_ids)

        image_mask = visual_masks.unsqueeze(-1).to(visual_outputs.dtype)
        text_mask = output_tokens.attention_mask.unsqueeze(-1).to(text_embeds.dtype)
        image_embeds = (visual_outputs * image_mask).sum(1) / image_mask.sum(1).clamp_min(1)
        text_embeds = (text_embeds * text_mask).sum(1) / text_mask.sum(1).clamp_min(1)

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        loss = clip_loss(logits_per_text)

        return loss

    def shared_step(
        self, inputs: Dict, split: str, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Shared logic for training, validation and testing steps.

        Args:
            inputs: Input dictionary
            split: Current split (train, val, test)
            batch_idx: Current batch index

        Returns:
            Tuple of (loss, log_dict)
        """
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
            visual_outputs, visual_masks, inputs, split, batch_idx
        )
        log_dict = {}
        alignment_only = self.cross_modal_align and (
            (self.warm_up_steps is None and not self.combined_loss)
            or (split == "train" and self.warm_up_steps is not None
                and self.global_step < self.warm_up_steps)
        )
        if not alignment_only:
            # Teacher forcing shifts the decoder input right, including its valid-token mask.
            target_mask = output_tokens.attention_mask
            decoder_mask = torch.cat((torch.ones_like(target_mask[:, :1]), target_mask[:, :-1]), dim=1)
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=decoder_mask,
                labels=targets,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            if self.use_label_smoothing:
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    label_smoothing=self.label_smoothing,
                    ignore_index=-100,
                )
            else:
                loss = outputs.loss
            log_dict[f"{split}/loss"] = loss
        if self.cross_modal_align:
            cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
            log_dict[f"{split}/contra_loss"] = cont_loss
            if alignment_only:
                loss = cont_loss
            else:
                loss = loss + self.alpha * cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        if split != "train":
            generation_kwargs = {
                "inputs_embeds": input_embeds,
                "attention_mask": input_masks,
                "max_length": self.max_txt_len,
                "early_stopping": True,
            }

            if self.use_nucleus_sampling:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "top_p": self.top_p,
                        "temperature": self.temperature,
                        "num_return_sequences": 1,
                    }
                )
            else:
                generation_kwargs.update({"num_beams": self.beam_size})

            if self.is_mbart:
                target_lang_codes = [
                    self.get_language_code(lang) for lang in inputs["lang"]
                ]
                if target_lang_codes:
                    if hasattr(self.tokenizer, "lang_code_to_id"):
                        forced_bos_token_id = self.tokenizer.lang_code_to_id.get(
                            target_lang_codes[0], self.tokenizer.bos_token_id
                        )
                        generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

            generated = self.llm.generate(**generation_kwargs)

            generated_strings = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            generated_strings = [gen.lower() for gen in generated_strings]

            self.generated.extend(generated_strings)
            self.references.extend(inputs["text"])
            self.result_ids.extend(inputs["ids"])
            self.result_languages.extend(inputs["lang"])

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.set_container()
            return
        if not self.generated or not self.references:
            print("\n===== No validation results to display =====")
            return

        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)

        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        results_dir = os.path.join(self.logdir)

        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split="val",
            save_results=True,
            output_dir=results_dir,
            filename_prefix=f"epoch{current_epoch:03d}_step{self.global_step:05d}",
            ids=self.result_ids,
            languages=self.result_languages,
        )

        bleu1_score = eval_res.get("val/bleu1", 0)
        bleu2_score = eval_res.get("val/bleu2", 0)
        bleu3_score = eval_res.get("val/bleu3", 0)
        bleu4_score = eval_res.get("val/bleu4", 0)
        rougeL_f1 = eval_res.get("val/rougeL_f1", 0)

        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        print(f"\n===== Validation Results (Epoch {current_epoch}) =====")

        bleu1_improvement = bleu1_score - self.best_bleu1
        bleu2_improvement = bleu2_score - self.best_bleu2
        bleu3_improvement = bleu3_score - self.best_bleu3
        bleu4_improvement = bleu4_score - self.best_bleu4
        rougeL_improvement = rougeL_f1 - self.best_rougeL_f1

        print(f"BLEU-1 Score: {bleu1_score:.4f} ({bleu1_improvement:+.4f} from best)")
        print(f"BLEU-2 Score: {bleu2_score:.4f} ({bleu2_improvement:+.4f} from best)")
        print(f"BLEU-3 Score: {bleu3_score:.4f} ({bleu3_improvement:+.4f} from best)")
        print(f"BLEU-4 Score: {bleu4_score:.4f} ({bleu4_improvement:+.4f} from best)")
        print(f"ROUGE-L F1: {rougeL_f1:.4f} ({rougeL_improvement:+.4f} from best)")
        print("=" * 50)
        print("")

        if bleu1_score > self.best_bleu1:
            self.best_bleu1 = bleu1_score
        if bleu2_score > self.best_bleu2:
            self.best_bleu2 = bleu2_score
        if bleu3_score > self.best_bleu3:
            self.best_bleu3 = bleu3_score
        if bleu4_score > self.best_bleu4:
            self.best_bleu4 = bleu4_score
        if rougeL_f1 > self.best_rougeL_f1:
            self.best_rougeL_f1 = rougeL_f1

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        if not self.generated or not self.references:
            print("\n===== No test results to display =====")
            return

        print("\n===== Test Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)

        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        results_dir = os.path.join(self.logdir)

        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split="test",
            save_results=True,
            output_dir=results_dir,
            filename_prefix=f"epoch{current_epoch:03d}_step{self.global_step:05d}",
            ids=self.result_ids,
            languages=self.result_languages,
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=1e-8,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )

        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, "dataloader"):
                train_dataloader = train_dataloader.dataloader

            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs

            if hasattr(self.trainer, "accumulate_grad_batches"):
                total_steps = total_steps // self.trainer.accumulate_grad_batches

        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
