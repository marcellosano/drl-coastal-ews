# DRL-EWS: Deep Reinforcement Learning for Coastal Multi-Hazard Emergency Response

A simulation-based deep reinforcement learning framework that trains an agent to make protective action decisions during coastal storm events — evacuation orders, sandbag deployment, shelter advisories — across four storm types and a 72-hour event window.

Accompanying code for:

> Sano, M., Ferrario, D.M., Torresan, S., Critto, A. (2026). *A Deep Reinforcement Learning Framework for Operational Coastal Multi-Hazard Emergency Response* *(under review)*

---

## Authors

**Marcello Sano**<sup>1,2,3</sup>\*, **Davide Mauro Ferrario**<sup>1,2</sup>, **Silvia Torresan**<sup>1,2,4</sup>, **Andrea Critto**<sup>1,2</sup>

<sup>1</sup> Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University of Venice, Italy  
<sup>2</sup> CMCC Foundation, Euro-Mediterranean Centre on Climate Change, Italy  
<sup>3</sup> Griffith University, Gold Coast, Australia  
<sup>4</sup> National Biodiversity Future Center (NBFC), Palermo, Italy  

\* Corresponding author

---

## What the model does

The framework models coastal multi-hazard emergency management as a Markov Decision Process over 24 timesteps (3-hour intervals across a 72-hour event window). Key components:

| Component | Description |
|---|---|
| **Environment** | 20×20 synthetic coastal grid, 45 houses, 200 residents, 2 evacuation centres, 2 rivers |
| **Storm system** | Four types: SURGE, RAIN, WIND, MIXED — each with realistic approach/peak/retreat lifecycle |
| **State space** | 131-dimensional vector: hazard intensities, DBSCAN multi-hazard cluster features, asset/resource status, temporal context |
| **Action space** | 10 discrete emergency responses (monitor → advisory → targeted protection → zone evacuations → full evacuation) |
| **Agent** | PPO with storm-conditioned policy heads (one per storm type), action masking, curriculum learning |
| **Training** | 4,000 episodes, 24 curriculum levels (peak intensity 0.40 → 0.90), 99.9% life-safety performance |

The agent learns hazard-appropriate strategies rather than universal responses: coastal sandbagging for surge, riverside protection for rainfall, shelter-in-place for wind, full evacuation for compound events.

---

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU recommended for full training (CPU works but is slow)

```
numpy>=1.24
torch>=2.0
gymnasium>=0.29
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
tensorboard>=2.14
```

---

## Installation (local)

```bash
git clone https://github.com/marcellosano/drl-coastal-ews.git
cd drl-coastal-ews
pip install -r requirements.txt
```

---

## How to run

### Local

Open `drl_ews_v4_35_04022026.py` and set the runtime flags near the top of the file:

```python
SEED = 42
USE_GOOGLE_DRIVE = False   # ← set False for local runs
EVAL_ONLY = False
LOAD_CHECKPOINT = None     # or path to a .pt file to resume
EPISODES = 4000
CHECKPOINT_INTERVAL = 50
```

Then run:

```bash
python drl_ews_v4_35_04022026.py
```

Outputs are written to `./COASTAL_DRL_FINAL/`:

| Directory | Contents |
|---|---|
| `checkpoints/` | Model snapshots every `CHECKPOINT_INTERVAL` episodes + best/final models |
| `logs/` | TensorBoard logs (`tensorboard --logdir ./COASTAL_DRL_FINAL/logs`) |
| `metrics/` | Training metrics as JSON |
| `plots/` | Training curves, JSD evolution, storm sample plots (PNG) |
| `evaluations/` | Held-out evaluation results |
| `configs/` | Config dump (JSON) for reproducibility |

### Google Colab

1. Clone the repo inside a Colab cell:
   ```python
   !git clone https://github.com/marcellosano/drl-coastal-ews.git
   %cd drl-coastal-ews
   ```

2. Install packages not pre-installed on Colab:
   ```python
   !pip install gymnasium seaborn
   ```
   (`numpy`, `torch`, `matplotlib`, `pandas`, `scikit-learn`, `scipy`, `tensorboard` are pre-installed.)

3. Set `USE_GOOGLE_DRIVE = True` to save outputs to your Drive (recommended for long runs), or `False` to keep everything in `/content/`.

4. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU** — strongly recommended for training.

5. Run:
   ```python
   exec(open("drl_ews_v4_35_04022026.py").read())
   ```

**Runtime estimate:** `EPISODES = 4000` takes approximately 2–3 hours on a T4 GPU. For a quick smoke-test, set `EPISODES = 200` — the agent will train through a few curriculum levels and produce sample outputs without completing full convergence.

---

## Minimal reproducible example

To verify the environment runs correctly on your machine:

```python
# At the top of the script, change:
EPISODES = 50
USE_GOOGLE_DRIVE = False
```

Expected console output (first few lines):
```
ℹ Running locally (not in Colab)
✓ Base path: ./COASTAL_DRL_FINAL
Episode   1 | Reward:   1842 | Lives: 99.5% | Curriculum: 0
Episode   2 | Reward:   2104 | Lives: 100.0% | Curriculum: 0
...
```
Outputs land in `./COASTAL_DRL_FINAL/plots/training_curves.png` after training completes.

---

## Citation

If you use this code, please cite:

```bibtex
@software{sano2026drlews,
  author    = {Sano, Marcello and Ferrario, Davide Mauro and Torresan, Silvia and Critto, Andrea},
  title     = {Deep Reinforcement Learning for Coastal Multi-Hazard Emergency Response (DRL-EWS v1.0.0)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/marcellosano/drl-coastal-ews}
}
```

*(DOI will be updated after Zenodo archival.)*

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

This work was supported by the European Union through the Marie Skłodowska-Curie Fellowship EXPEDITE, "Exploring Opportunities for Developing a Risk and Resilience Climate Service Based on Big Data and Machine Learning" (Grant Agreement No. 101067784). The authors also acknowledge the Horizon 2020 MYRIAD-EU project (Grant Agreement No. 101003276) for its contribution to the broader development of multi-hazard and multi-risk research relevant to this study.
