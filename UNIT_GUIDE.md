# DSB Unit 02 — Data Science Learning Materials

This educational repository contains lesson notebooks, exercises, practice datasets,
labs, and guided code-alongs for learning exploratory data analysis with Python.
Topics include pandas, joining data, visualisation, hypothesis testing, missing
data, preparing data for modeling, feature engineering, and working with APIs.

Each learning unit should contain its slide deck as a **PDF document in that
unit's folder**, alongside the notebooks and exercises. This README tracks which
decks still need to be added manually.

## Slide-deck completeness

**Inventory checked: 21 October 2026. All eight main units (01–08) have PDF
slide decks.** `99 - Stepped Code-Along` and `Labs` are still missing decks.

All eight unit decks have been checked for PDF file signatures.
`scripts/Command-Line-and-Git.pdf` is a supporting deck kept in `scripts`
because there is no corresponding unit folder.

The table lists every top-level content folder. File counts include nested
folders; Git metadata, `.DS_Store`, and Python cache files are excluded.
`data` and `scripts` provide supporting resources rather than lesson units, so
their slide-deck status is not applicable.

| Unit / folder | Current contents | PDF slide deck | Deck file / completion tracker |
| --- | --- | --- | --- |
| [01 - Intro](01%20-%20Intro/) | 3 notebooks: introduction to pandas, exploratory data analysis (EDA), and an EDA exercise. | Present | [EDA-Introduction to-Exploratory-Data-Analysis-and-pandas.pdf](01%20-%20Intro/EDA-Introduction%20to-Exploratory-Data-Analysis-and-pandas.pdf) |
| [02 - Exploring and Joining](02%20-%20Exploring%20and%20Joining/) | 3 lesson/exercise notebooks on distributions, descriptive statistics, and joins; 4 practice notebooks; 12 supporting data files (9 CSV, 1 TSV, 1 compressed CSV, and 1 Excel data dictionary). | Present | [EDA-Exploring-and-Joining-Data-with-pandas.pdf](02%20-%20Exploring%20and%20Joining/EDA-Exploring-and-Joining-Data-with-pandas.pdf) |
| [03 - Data Visualisation](03%20-%20Data%20Visualisation/) | 3 lesson notebooks on relationships, trends, matplotlib, and seaborn; 8 practice notebooks covering plotting and data analysis; 11 practice CSV datasets. | Present | [EDA-Data-Visualization-for-Exploration.pdf](03%20-%20Data%20Visualisation/EDA-Data-Visualization-for-Exploration.pdf) |
| [04 - Experimentation](04%20-%20Experimentation/) | 3 hypothesis-testing notebooks: base, live-classroom, and full-analysis versions. | Present | [EDA-Experimental-Design.pdf](04%20-%20Experimentation/EDA-Experimental-Design.pdf) |
| [05 - Missing Data](05%20-%20Missing%20Data/) | 1 notebook on missing data. | Present | [EDA-Missing-Data.pdf](05%20-%20Missing%20Data/EDA-Missing-Data.pdf) |
| [06 - Modeling](06%20-%20Modeling/) | 1 notebook on EDA for modeling. | Present | [EDA-for-Modeling.pdf](06%20-%20Modeling/EDA-for-Modeling.pdf) |
| [07 - Feature Engineering](07%20-%20Feature%20Engineering/) | 1 feature-engineering notebook. | Present | [EDA-Feature-Engineering.pdf](07%20-%20Feature%20Engineering/EDA-Feature-Engineering.pdf) |
| [08 - APIs](08%20-%20APIs/) | 3 notebooks: marketing strategy, financial markets, and a crimes API mini-lab. | Present | [EDA-APIs.pdf](08%20-%20APIs/EDA-APIs.pdf) |
| [99 - Stepped Code-Along](99%20-%20Stepped%20Code-Along/) | 5 guided notebooks (Modules 2.1–2.5): Python collections, NumPy and pandas; cleaning and validation; filtering, joining and summarising; visual EDA; advanced pandas transformations. | Missing | To add |
| [Labs](Labs/) | Titanic EDA lab: 1 student-submission notebook, a training CSV, a lab README, and site configuration/layout files. | Missing | To add |
| [data](data/) | Shared datasets: 5 CSV files (housing, LEGO, sodas, and Titanic) and `cpi.json`. | Not applicable — supporting data | — |
| [scripts](scripts/) | Git/SSH setup helper, its Python tests, setup instructions in a separate README, and a Command Line and Git PDF deck. | Not applicable — tooling | [Command Line and Git](scripts/Command-Line-and-Git.pdf) — supporting deck |

## Using the materials

Open the `.ipynb` files in Jupyter Notebook, JupyterLab, or a notebook-capable
editor such as VS Code. Follow the numbered units in order, using the practice
folders and Labs for exercises and `99 - Stepped Code-Along` for guided practice.
Keep the folder structure intact so notebooks can access their datasets.
See the [Titanic lab instructions](Labs/lab-eda-titanic/README.md) for that
assignment and the [scripts README](scripts/README.md) for the setup helper.

## Updating the tracker

1. Add the unit's actual slide deck as a `.pdf` file directly inside its folder.
   Use a descriptive filename, for example `01-intro-slides.pdf`.
2. Download the actual PDF file from the source, rather than saving its GitHub
   web page. Renaming an HTML file to `.pdf` does not convert it to PDF.
   Open the PDF to confirm that it contains the correct unit's slides. An
   assignment handout or another PDF does not count as the slide deck.
3. Change that row's status from **Missing** or **Needs replacement** to **Present**, and replace
   **To add** with a relative link to the PDF. If a unit has multiple decks,
   link each one and mark it **Partial** until all required decks are present.
4. Update the inventory date and completeness summary above to reflect the
   latest check. This table is maintained manually; adding a PDF alone does
   not update it automatically.
