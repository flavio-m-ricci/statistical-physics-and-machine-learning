# Statistical Physics and Machine Learning — Course Exercises

**Flavio Massimiliano Ricci**  
MSc in Physics, *Statistical Physics and Complexity*, Sapienza University of Rome

This repository collects my solutions and computational exercises from the **Statistical Physics and Machine Learning** course. It is intended as a transparent record of coursework and independent learning, **not as a collection of original research publications**.

## Contents

Exercises will be listed here as they are completed and reviewed.

| Notebook | Topic | Main methods |
| --- | --- | --- |
| *To be added* | *To be added* | *To be added* |

## Repository structure

- `notebooks/` — one self-contained Jupyter notebook per exercise, named `01_short_topic.ipynb`, `02_short_topic.ipynb`, etc.
- `templates/exercise_template.ipynb` — a blank template with sections for the problem, theory, implementation, results, and discussion.
- `figures/` — figures worth keeping outside notebooks.
- `requirements.txt` — initial Python dependencies (update when exercises are added).

## Running the exercises

Activate your preferred Python environment, then run:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open a notebook under `notebooks/`. The notebooks should use relative paths and fixed random seeds when appropriate, so that plots and results can be reproduced. If you use the shared virtual environment from your existing development workspace, you do not need to create another environment for this repository.

## How exercises are documented

Each notebook should contain:

1. **Problem and objective** — concise statement in my own words.
2. **Theory** — the relevant equations, assumptions, and derivations.
3. **Implementation** — readable Python code and comments.
4. **Numerical experiments** — plots, comparisons, and checks.
5. **Discussion** — interpretation, limitations, and possible extensions.

## Attribution and academic integrity

The exercises are based on course material from Sapienza University of Rome. I will identify any external papers, starter code, datasets, or other sources in the relevant notebooks. Any modifications or extensions will be distinguished from the original material. Instructor slides, problem sheets, restricted datasets, and third-party source code will not be redistributed without the appropriate permission or licence.

## Status

Work in progress. This repository will be expanded incrementally as exercises are completed.
