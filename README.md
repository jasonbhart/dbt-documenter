# DBT Documenter tool #

This tool simplifies dbt YAML file maintenance by:

1. Creates model .yml files for any .sql files missing them
2. Parses .yml files and compares them to db materialized models and then adds/removes column differences
3. Fills in missing .yml file configuration sections
4. Uses AI to create descriptions for the model and related columns where they don't exist (preserves existing)
5. (FUTURE) Uses AI to create dbt tests for models and related columns where they don't exist (preserves existing)

*This tool requires an OpenAI API key to function*

---

## Setup

To use this tool you will clone this repository and then run the python script from the cloned repo location.

1. Clone the repo.
2. Create a .env file is the cwd (current working directory) and update the values to match your needs (set the BQ_DATASET_NAME to your dbt sandbox dataset). Similar to how below is shown
```GCP_PROJECT_ID=sandbox-219106
BQ_DATASET_NAME=  change this to your dbt dev env dataset name to use with pre-production dbt models Ex. dbt_model_KJ
OPENAI_API_KEY=  Add the OpenAI API key here
GCP_CREDENTIALS_FILE=.\keys.json path to service account keys.json file or user credentials file (i.e. ~/.config/credentials.json)
PROMPTLAYER_API_KEY=  Add the PL API key here (optional)
```
3. Make sure you have Python 3 installed (miniconda3 is suggested)
4. Create a Python virtual environment (`conda create -n dbt-documenter python=3.11`)
5. Activate the virtual environment (`conda activate dbt-documenter`)
6. Install needed Python modules (`pip install -r requirements.txt`)

---

## Usage

To run the tool you will specify the base directory in which you want the tool to start scanning and enriching dbt .yml files.  From this folder it will traverse any sub-directories which are encountered under this base directory. It is suggested you run this script on dbt models which are already version controlled so you can easily see the changes made and can manually review changes made.

Example:
`python run.py C:\dbt-repo-dir\models`

### Suggested workflow when creating new models

1. Create the .sql file and finalize field names and types before running this script
2. `dbt run` the new .sql file to materialize the model in your dbt sandbox dataset of the data warehouse
3. Run this script by specifying the folder/sub-folder which contains your new model (.sql file)
4. Script will create a .yml file for the new model and populate it
5. Review the results in the .yml file and make any changes necessary.

---

## Bugs and Limitations

* This script assumes the ideal outcome is to have one .yml file per .sql file, but will handle cases where multiple models are encased in a single .yml file.
* Processed .yml files are missing blank lines inserted between YML top-level sections.
* Empty .yml files with spaces or tabs might cause unusual results.  Suggest not creating empty .yml files as placeholders.
* Sometimes existing configuration items (i.e. tests) will be erroneously removed from the resulting .yml file; be sure to double check results (diffs) before commiting back to dbt repo.
