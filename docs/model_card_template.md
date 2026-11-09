---
license: cc-by-4.0
base_model: {{BASE_MODEL}}
pipeline_tag: translation
language:
  - am
  - en
datasets:
  - google/fleurs
tags:
  - amharic
  - english
  - translation
  - peft
  - low-resource-language
---

# Ethiopian Language Translation AI (Amharic–English)

A small bidirectional Amharic–English LoRA adapter for the MIT-licensed M2M100 multilingual translation model. This is an experimental portfolio artifact, not a production or human-certified translator.

## Evaluation

Scores are corpus SacreBLEU / chrF++ (0–100; higher is better), using one reference per sentence. The full held-out FLEURS test set contains {{TEST_N}} aligned pairs. The LoRA epoch was selected on a separate validation split; test data were not used for training or epoch selection.

| Approach | English → Amharic BLEU / chrF++ | Amharic → English BLEU / chrF++ |
|---|---:|---:|
| M2M100 zero-shot | {{EN_BASE}} | {{AM_BASE}} |
| M2M100 LoRA adapter | {{EN_ADAPTER}} | {{AM_ADAPTER}} |

Mean held-out chrF++ gain over zero-shot: {{GAIN}} points; paired sentence-ID bootstrap 95% CI: [{{CI_LOW}}, {{CI_HIGH}}]. Direction-specific paired 95% CIs for chrF++ change: English→Amharic [{{CI_EN_LOW}}, {{CI_EN_HIGH}}] (includes zero); Amharic→English [{{CI_AM_LOW}}, {{CI_AM_HIGH}}]. The macro release gate requires validation improvement, a positive lower bootstrap bound, and no more than a 2-point decline in either direction. The English→Amharic absolute score is very low, and its interval does not establish a gain. These automatic overlap scores are not human judgments of meaning, fluency, or safety.

## Training

The adapter was trained with PEFT LoRA (rank 8, alpha 16) on both translation directions using {{TRAIN_N}} aligned train pairs. The best available saved checkpoint is epoch {{BEST_EPOCH}} (validation n={{VALIDATION_N}}). Although two epochs were requested, the CPU run did not leave a clean final report, so no epoch-2 checkpoint is claimed. The base checkpoint is [`{{BASE_MODEL}}`](https://huggingface.co/{{BASE_MODEL}}) at commit `{{BASE_REVISION}}`. Training used a fixed random seed and CPU-only PyTorch. Evaluation and inference use two-beam decoding with a 128-token generation cap.

## Data and attribution

Training and evaluation use text transcripts from Google [FLEURS](https://huggingface.co/datasets/google/fleurs), Amharic `am_et` and English `en_us`, revision [`{{DATASET_REVISION}}`](https://huggingface.co/datasets/google/fleurs/tree/{{DATASET_REVISION}}). The FLEURS dataset card declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is described by Conneau et al. (2022), [*FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*](https://arxiv.org/abs/2205.12446). Attribution: Google Research and the FLEURS authors. Repeated sentence IDs were deduplicated; conflicting IDs were excluded. No audio or source transcripts are included in this repository.

The adapter release is marked CC BY 4.0 to preserve clear attribution to its training data. The upstream M2M100 checkpoint declares MIT; the original project source code is separately MIT-licensed. Its model card does not fully enumerate the rights status of every original training corpus, which remains an upstream provenance limitation. The base checkpoint is not bundled; see the upstream model and dataset cards for their distinct terms.

## Inference

```python
from peft import PeftModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

base = "{{BASE_MODEL}}"
model_id = "{{REPO_ID}}"
revision = "{{BASE_REVISION}}"
tokenizer = M2M100Tokenizer.from_pretrained(base, revision=revision)
model = M2M100ForConditionalGeneration.from_pretrained(base, revision=revision)
model = PeftModel.from_pretrained(model, model_id)
tokenizer.src_lang = "en"  # use "am" for Amharic input
inputs = tokenizer("Good morning.", return_tensors="pt")
translated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.get_lang_id("am"),  # use "en" for English output
    num_beams=2,
    max_new_tokens=128,
)
print(tokenizer.batch_decode(translated, skip_special_tokens=True)[0])
```

## Intended use and limitations

Use for research, experimentation, and low-stakes translation exploration with human review. Do not use as the sole source for medical, legal, financial, emergency, or other consequential decisions. The data is small, read-speech-derived, and based on short benchmark sentences; it does not represent spontaneous conversation, dialect variation, code-switching, current events, or specialist terminology. The test set has {{TEST_N}} pairs and one reference per source. No human translation quality review or independent external evaluation has been performed.
