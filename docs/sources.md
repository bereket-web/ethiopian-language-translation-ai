# Data and model sources

## FLEURS transcripts

The parallel text is extracted from Google’s [FLEURS dataset](https://huggingface.co/datasets/google/fleurs), at immutable revision [`70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd). It uses the official `am_et` Amharic and `en_us` English TSV transcripts and joins them by the source sentence ID. The dataset card describes FLEURS as the speech version of the FLORES benchmark and declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It records about 2,009 n-way parallel sentences in 102 languages; the Amharic/English training split available at this revision is smaller and contains repeated recordings, so the project deduplicates by sentence ID and excludes conflicts.

The dataset is for speech research; this project uses only text transcripts. Audio is never downloaded by the preparation script. Raw TSVs and processed text remain local, are excluded from Git, and are not redistributed. Attribution: Conneau et al. (2022), [*FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*](https://arxiv.org/abs/2205.12446), plus the Google FLEURS dataset card. The parent FLEURS card notes these samples are read-speech-domain and based on FLoRes sentences.

## Transformer checkpoint

The transformer is [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M), pinned to commit `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`. Its model card declares [MIT](https://opensource.org/license/mit) and lists Amharic (`am`) and English (`en`). This project uses the upstream checkpoint as a zero-shot baseline and, if validation supports it, trains a small LoRA adapter. The base model weights are not copied into the project repository. The upstream checkpoint’s broader training-data documentation does not fully enumerate each original dataset’s rights; this remains a limitation even though the checkpoint is MIT-licensed.

Any project-trained adapter proposed for redistribution will carry explicit FLEURS attribution and use CC BY 4.0 terms, which permit research and commercial reuse subject to attribution. Original project code remains MIT; the upstream M2M100 base checkpoint remains MIT and is not copied into the adapter release.

The NLLB-200 model was excluded from this release candidate because its model card declares CC BY-NC 4.0, which is a less permissive fit for a portfolio artifact that may later be used in commercial contexts. OPUS-100 was not used because its Hub card lists the dataset license as unknown. FLORES+ was not used for training: its card states it is for evaluation only and requires accepting gated terms.

## Project code

The original source code is MIT-licensed in the root `LICENSE`. That license does not replace or relicense FLEURS transcripts or the M2M100 base checkpoint.
