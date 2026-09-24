# ForeSteer

Code for preparing ForeSteer training data, training the risk predictor, and evaluating on CHAIR, POPE, and MME.

## Setup

Use Linux with an NVIDIA GPU. Run all commands from the repository root.

```bash
conda env create -f environment.yml
conda activate mllm
```

For CHAIR, download the NLTK data (not included in the Conda environment):

```bash
python -m nltk.downloader punkt punkt_tab wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

## Models and data

Set the local model and dataset paths:

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/llava-v1.5-7b
export COCO_ROOT=/path/to/coco

# Evaluation on COCO val2014 (CHAIR / POPE)
export IMAGE_FOLDER=/path/to/coco/val2014
export COCO_ANNOTATION_PATH=/path/to/coco/annotations

# MME
export BASE_DIR=/path/to/MME_Benchmark
export REFERENCE_DIR=/path/to/mme/reference
```

- **Training:** Captions and risk labels are built from COCO **train2014** images under `$COCO_ROOT/train2014/`; the risk predictor is trained on these trajectories.
- **CHAIR / POPE testing:** Inference uses COCO **val2014** images from `IMAGE_FOLDER`. POPE's three question splits are included in `data/pope/coco/`.
- **Annotations:** Provide the train2014 and val2014 instance/caption JSON files in `$COCO_ROOT/annotations/`. The evaluator reads both annotation splits, but scores captions generated from val2014 images.
- **MME:** images under `<BASE_DIR>/<task>/` or `<BASE_DIR>/<task>/images/`, and `<REFERENCE_DIR>/<task>.txt` with tab-separated image filename, question, and ground-truth answer.

## Build data and train

The following command runs the complete pipeline for LLaVA: greedy generation on up to 5,000 COCO train2014 images, CHAIR token annotation, hidden-state extraction, and risk-predictor training.

```bash
bash scripts/run_experiments.sh llava prepare
```

To run the stages separately:

```bash
bash scripts/run_experiments.sh llava data
bash scripts/run_experiments.sh llava train
```

Training data is saved in `outputs/llava_coco_train2014/`; the predictor checkpoint is saved in `outputs/risk/llava/`. The default configuration is layer 16, discount `GAMMA=0.9`, horizon `RISK_HORIZON=16`, and seed 42. Set `MAX_SAMPLES` to change the number of training images and `MAX_NEW_TOKENS` to change the generation limit. Set `CUDA_VISIBLE_DEVICES=0,1,...` to distribute generation and extraction across GPUs.

If hidden states have already been extracted, use `SKIP_EXTRACT=1 bash scripts/run_experiments.sh llava train` to train another predictor without extracting them again. Use `RISK_OUTPUT` to select a different feature and checkpoint directory.

## Evaluation

ForeSteer uses the checkpoint produced above (or one trained with the same base model) and requires a positive `ALPHA`:

```bash
export ALPHA=20
export RUN_NAME=foresteer_alpha20

bash scripts/run_experiments.sh llava chair
bash scripts/run_experiments.sh llava pope
bash scripts/run_experiments.sh llava mme
```

Use `bash scripts/run_experiments.sh llava eval all` to run all three benchmarks.

For Qwen-VL-Chat, replace `llava` with `qwen_vl` and use its model weights. Each model must be trained on its own generated data. Base-model weights are not included. The checkpoint loader requires `MODEL_PATH` to match the resolved model path stored in the checkpoint. Set `RISK_CHECKPOINT=/path/to/checkpoint.pt` if it is stored outside the default `RISK_OUTPUT` directory.

Evaluation uses greedy decoding, seed 42, and batch size 1. CHAIR samples 500 images with at most 64 new tokens; POPE/MME use 3 new tokens. Override these with `SEED`, `BATCH_SIZE`, `MAX_SAMPLES` (CHAIR), and `MAX_LENGTH`. POPE runs all three splits. Set `ADAPTIVE_ALPHA=1` to skip steering when predicted risk is negative.

Predictions and metrics are saved to `outputs/eval/<model>/<run>/` (override with `OUT_DIR`). Reusing an output directory overwrites its results.
