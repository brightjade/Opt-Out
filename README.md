# Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport

Source code for our *ACL 2025* paper [Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](https://arxiv.org/abs/2406.12329).

This codebase implements various unlearning methods to make language models "forget" specific entities while preserving their general capabilities.

## 🔧 Installation

To install requirements:
```bash
conda create -n optout python=3.12.9
conda activate optout
pip install -r requirements.txt
```

## 🤗 Data

We provide the **ELUDe (Entity-Level Unlearning Dataset)** on Hugging Face: [https://huggingface.co/datasets/6rightjade/ELUDe](https://huggingface.co/datasets/6rightjade/ELUDe)

**ELUDe** is a comprehensive machine unlearning dataset focused on the removal of entire entities from large language models (LLMs). The dataset includes:

- **20 real-world target entities** (the entities listed below)
- **144 unique neighboring entities** from Wikipedia

## 📊 Available Entities

The codebase supports unlearning for **20 different entities**:

1. **Donald_Trump**
2. **Elizabeth_II** 
3. **Barack_Obama**
4. **Cristiano_Ronaldo**
5. **Michael_Jackson**
6. **Elon_Musk**
7. **Lady_Gaga**
8. **Adolf_Hitler**
9. **Eminem**
10. **Lionel_Messi**
11. **Justin_Bieber**
12. **Freddie_Mercury**
13. **Kim_Kardashian**
14. **Johnny_Depp**
15. **Steve_Jobs**
16. **Dwayne_Johnson**
17. **Michael_Jordan**
18. **Taylor_Swift**
19. **Stephen_Hawking**
20. **Kanye_West**

## 🧠 Unlearning Methods

### Core Methods
- **`original`** - The original performance of the model
- **`icu`** - **In-Context Unlearning**: Prompting baseline (Guardrail)
- **`ga`** - **Gradient Ascent**: Uses gradient ascent for unlearning
- **`dpo`** - **Direct Preference Optimization**: Uses DPO for unlearning
- **`npo`** - **Negative Preference Optimization**: Uses NPO for unlearning
- **`idk`** - **I Don't Know**: Makes the model respond with "I don't know"

### Data Augmentation Options
You can combine core methods with the following modifiers (except `original` and `icu`):

- **`+rt`** - **Retain Data**: Includes neighboring entity data to preserve nearby knowledge
- **`+wd`** - **World Data**: Uses Alpaca GPT-4 data for maintaining general knowledge (we use Alpaca GPT-4 data from [here](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM))
- **`+ot`** - **Optimal Transport**: Adds Wasserstein regularization for better unlearning

### Example Method Combinations
- `npo+rt+wd+ot` - NPO with retain data, world data, and optimal transport (Opt-Out)
- `dpo+rt+wd` - DPO with retain and world data
- `ga+rt` - Gradient ascent with retain data only
- `idk+wd` - IDK method with world data only

## 🚀 Usage

### Training

Use the training script to fine-tune models for entity unlearning:

```bash
bash scripts/train.sh
```

### Evaluation

Run evaluation on trained models:

```bash
bash scripts/eval.sh
```

## 📁 Directory Structure

```
Opt-Out/
├── run.py              # Main training/evaluation script
├── trainer.py          # Custom trainer implementation
├── model.py            # Model loading utilities  
├── dataset.py          # Data loading and processing
├── evaluator.py        # Evaluation logic
├── scripts/            # Execution scripts
│   ├── train.sh        # Training script
│   └── eval.sh         # Evaluation script
├── data/               # External data
```

## 📚 Citation

If you use this codebase, please cite our paper:

```bibtex
@article{choi2025optout,
  title={Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport},
  author={Choi, Minseok and Rim, Daniel and Lee, Dohyun and Choo, Jaegul},
  journal={arXiv preprint arXiv:2406.12329},
  year={2025}
}
```
