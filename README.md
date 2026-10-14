# Tenancy Rent Caps under High Inflation

Analysis code for *Tenancy Rent Caps under High Inflation: Divergence between Sitting and New Tenants in Türkiye*, by Muhammet Rıdvan İnce (Kocaeli University).

The study examines changes in rents, housing burdens and consumption among sitting and new tenants in Türkiye. It uses repeated cross-sections from the Survey of Income and Living Conditions (SILC/GYKA) and the Household Budget Survey (HBS/HHBA). The estimates describe changes in conditional gaps between the two groups. They do not separately identify the effect of the rent cap from inflation and delayed rent adjustment.

The [English manuscript](paper/manuscript.docx) contains the tables, figures, references and author declarations. This is a manuscript version, not a published journal article.

## Data access

The TurkStat microdata cannot be redistributed under the author's access conditions. No raw records, household-level analysis files or source-data manifests are included in this repository. Reproducing the estimates requires independently authorised access to the relevant TurkStat files. See [TurkStat microdata access](https://www.tuik.gov.tr/Kurumsal/Mikro_Veri).

| Survey | Required years | Files |
| --- | --- | --- |
| SILC/GYKA cross-sections | 2014–2025 | Household (`h`) and individual (`fk`) files |
| HBS/HHBA | 2016–2019, 2022, 2024 | Household (`hane`), individual (`fert`) and expenditure (`tuketim`) files |

The main SILC estimation sample covers 2016–2025. The 2014–2015 observations are used for the longer descriptive and mobility series. The HBS analysis uses the six years listed above.

Keep the authorised files outside this repository. Set `TUIK_MICRODATA_ROOT` to the directory containing the year folders. The file resolver expects the Turkish CSV files in this structure; it also searches subdirectories within each `CSV` folder:

```text
<microdata root>/
  2014/GYKA_Kesit/CSV/gyk14_h.csv
  2014/GYKA_Kesit/CSV/gyk14_fk.csv
  ...
  2025/GYKA_Kesit/CSV/gyk25_h.csv
  2025/GYKA_Kesit/CSV/gyk25_fk.csv
  2016/HHBA_Hanehalki_Butce/CSV/hane_cd_2016.csv
  2016/HHBA_Hanehalki_Butce/CSV/fert_cd_2016.csv
  2016/HHBA_Hanehalki_Butce/CSV/tuketim_cd_2016.csv
  ...
```

The reader detects comma or semicolon separators and uses ISO-8859-9 encoding. Original variable names and file names must be retained.

## Running the analysis

The analysis environment uses Python 3.14.3. Package versions are recorded in [`requirements.txt`](analysis/codes/requirements.txt). From the repository directory, create a virtual environment and install these packages:

```sh
python -m venv .venv
```

On Windows (PowerShell):

```powershell
.venv\Scripts\python.exe -m pip install -r analysis/codes/requirements.txt
$env:TUIK_MICRODATA_ROOT = "D:\restricted_microdata\datas"
.venv\Scripts\python.exe -X utf8 analysis/codes/00_run_all.py
```

On macOS or Linux:

```sh
.venv/bin/python -m pip install -r analysis/codes/requirements.txt
export TUIK_MICRODATA_ROOT="/path/to/restricted_microdata/datas"
.venv/bin/python -X utf8 analysis/codes/00_run_all.py
```

The path in these examples must be replaced with the location of your authorised files. The runner stops at the first failed stage. It writes logs to `analysis/outputs/validation/` and a run summary to `last_run.json` in that directory. The 9,999 bootstrap replications make estimation substantially slower than the descriptive stages.

| Script | Purpose |
| --- | --- |
| `01_gyka_hazirlik.py` | Harmonise SILC household and individual files |
| `02_hhba_hazirlik.py` | Build the HBS tenant sample and consumption measures |
| `03_tanimlayici.py` | Descriptive statistics and measurement checks |
| `04_ana_model.py` | SILC event studies, period models and bootstrap inference |
| `05_doz_heterojenlik.py` | Tenancy duration, income and metropolitan-area comparisons |
| `06_hareketlilik.py` | New-tenant shares and mobility comparisons |
| `07_hhba_model.py` | HBS rent and consumption models |
| `08_saglamlik.py` | Alternative samples and specifications |
| `09_grafikler.py` | Figures with Turkish labels |
| `10_dogrulama.py` | Validate generated outputs and the model registry |
| `11_makale_tablolari.py` | Format the detailed analysis tables |
| `13_figures_en.py` | Figures with English labels |
| `common.py` | File resolution, weighting, estimation and inference helpers |

Script numbers are retained from the working analysis. Stage 12 was an editorial check of an earlier Turkish draft; it is not part of this replication run. Literature-search utilities and internal review notes are also omitted. The table formatter produces the detailed analysis tables; their numbering differs from the condensed journal manuscript. In the journal manuscript, Figure 1 corresponds to `sekil2_olay_calismasi_kira_EN` and Figure 2 to `sekil5_yeni_kiraci_payi_EN`.

## Estimation and outputs

SILC models use household weights and region-by-year fixed effects. Standard errors are clustered by the 26 NUTS-2 regions. The main period coefficients and specified contrasts use restricted wild cluster bootstrap-t inference with Webb weights and 9,999 replications. The random seed is `20261010`. HBS models use household weights and HC1 standard errors.

Running the scripts creates household-level files in `datas/processed/`, local source manifests in `datas/metadata/`, and tables, figures and logs in `analysis/outputs/`. These directories are excluded from version control. The `.gitignore` file permits only the reviewed manuscript, source code and repository documentation; new files must be added deliberately.

The estimation routines and analysis specifications are unchanged from the working project. This distribution adds the `TUIK_MICRODATA_ROOT` setting and records source paths relative to that directory. It omits the editorial stage from the runner. Microdata are neither downloaded nor supplied by the scripts.

## Contact

Muhammet Rıdvan İnce · Kocaeli University
[ORCID](https://orcid.org/0000-0003-2050-2545) · muhammet.ince@kocaeli.edu.tr
