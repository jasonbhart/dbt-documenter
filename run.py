import os
import io
import sys
from pathlib import Path
import re
from typing import List, Dict, Union
import json
from json import JSONDecodeError
from dotenv import load_dotenv
from ruamel import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import tiktoken
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import PromptLayerChatOpenAI
from langchain import LLMChain
from langchain.cache import InMemoryCache
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

load_dotenv()

# Google Cloud project id
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
# BigQuery dataset name
BQ_DATASET_NAME = os.getenv('BQ_DATASET_NAME')
# Google Cloud service account key file
GCP_CREDENTIALS_FILE = os.getenv('GCP_CREDENTIALS_FILE')
# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

langchain.llm_cache = InMemoryCache()


def count_num_tokens(string: str) -> int:
    """Returns the number of tokens"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Create an instance of the LLM model. Use PromptLayer if in development mode.
if os.getenv('PROMPTLAYER_API_KEY'):
    llm = PromptLayerChatOpenAI(
        cache=False,
        verbose=True,
        temperature=0,
        # model="gpt-3.5-turbo-16k",
        model="gpt-4",
        openai_api_key=OPENAI_API_KEY,
        pl_tags=["langchain", "dbt-documenter"]
    )
else:
    llm = ChatOpenAI(
        cache=True,
        verbose=False,
        temperature=0,
        model="gpt-4",
        openai_api_key=OPENAI_API_KEY
    )

# Create a connection to BigQuery
engine = create_engine(
    f'bigquery://{GCP_PROJECT_ID}', credentials_path=GCP_CREDENTIALS_FILE
)


def replace_tab_with_spaces(input_stream):
    """Replace tabs with spaces in an input stream and modify it in place"""

    # Create an in-memory buffer to hold the modified content
    output_buffer = io.StringIO()

    for line in input_stream:
        # Replace tabs with spaces and write the modified line to the buffer
        output_buffer.write(line.replace("\t", "  "))

    # Get the modified content from the buffer
    modified_content = output_buffer.getvalue()

    # Create a new stream from the modified content
    return io.StringIO(modified_content)


class Answer(BaseModel):
    """Answer from the LLMChain"""
    name: str = Field(description="name of the dbt model/table")
    description: str = Field(description="description of the dbt model/table")
    columns: List[Dict] = Field(
        description="list of columns in the dbt model/table")
    tests: List[Union[Dict, str, None]] = Field(
        description="list of tests for the dbt model/table")
    docs: Dict = Field(
        description="configuration settings of docs for the dbt model/table")


EXAMPLE_OUTPUT = """
{
  "name": "contractor_billing_model",
  "description": "Details what/how much contractors (primarily Mentors and Graders) are billing Springboard each month.",
  "columns": [
    {
      "name": "contractor_billing_model_id",
      "description": "Unique ID of the billing model used within the Springboard system (PK)",
      "tests": [
        "not_null",
        "unique"
      ]
    },
    {
      "name": "contractor_id",
      "description": "Contractor ID associated with this billing model (FK to contractor table)",
      "tests": [
        "not_null",
      ]
    },
    {
      "name": "contractor_user_id",
      "description": "User ID of the Springboard user associated with the contractor (FK to user table)",
      "tests": [
        "not_null"
      ]
    },
    {
      "name": "contractor_name",
      "description": "Full name of the contractor",
      "tests": []
    },
    {
      "name": "is_active",
      "description": "Flag indicating whether the contractor is currently active",
      "tests": [
        {
          "accepted_values": {
            "values": [
              true,
              false
            ]
          }
        }
      ]
    },
    {
      "name": "role_family",
      "description": "The role family of the contractor",
      "tests": []
    },
    {
      "name": "invoice_item_type",
      "description": "Type of the invoice item",
      "tests": []
    },
    {
      "name": "invoice_item_quantity",
      "description": "Quantity of the invoice item",
      "tests": []
    },
    {
      "name": "invoice_item_payment_rate",
      "description": "Payment rate for the invoice item",
      "tests": []
    },
    {
      "name": "total_payment_amount",
      "description": "Total payment amount for the invoice item",
      "tests": []
    },
    {
      "name": "is_payment_amount_valid",
      "description": "Flag indicating whether the payment amount is valid",
      "tests": [
        {
          "accepted_values": {
            "values": [
              true,
              false
            ]
          }
        }
      ]
    },
    {
      "name": "invoice_date",
      "description": "Date of the invoice",
      "tests": []
    },
    {
      "name": "invoice_comments",
      "description": "Comments on the invoice",
      "tests": []
    },
    {
      "name": "unit",
      "description": "Unit of measurement for the invoice item",
      "tests": []
    },
  ],
  "tests": [],
  "docs": {
    "show": true
  }
}
"""

SYSTEM_TEMPLATE = (
    "Assume the role of a analytics engineer tasked with enhancing dbt config "
    "files by providing additional context where needed. Your responsibility "
    "involves appending descriptions for both the model and its respective "
    "columns. Be mindful to retain any pre-existing values and only supplement "
    "details where data is absent or incomplete. Do not add or remove tests.\n"
    "Your input will be the current dbt config file contents in a JSON blob "
    "that requires evaluation and augmentation.\n"
    "Your output will be the dbt config with added context in a JSON blob.\n\n"
    "Example output:"
    "{example_output}\n"
    "Results of SHOW CREATE TABLE command resulting from model's materialization:\n"
    "{create_statement}\n\n"
    "3 sample database table rows resulting from this model's materialization:\n"
    "{samples}\n\n"
)
system_message_prompt = SystemMessagePromptTemplate.from_template(
    SYSTEM_TEMPLATE)
HUMAN_TEMPLATE = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

output_parser = PydanticOutputParser(pydantic_object=Answer)

chat_prompt = ChatPromptTemplate(
    messages=[system_message_prompt, human_message_prompt],
    input_variables=["create_statement", "samples", "text"],
    partial_variables={"example_output": EXAMPLE_OUTPUT}
)
llm_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)


def parse_llm_response(result):
    """Parse the LLM response to get the file contents"""

    # Attempt parsing
    config_dict = result['text'].dict()
    if config_dict['name'] == "" \
            or config_dict['description'] == "" \
            or config_dict['columns'] == [] \
            or config_dict['docs'] == {}:
        raise OutputParserException("Unable to parse the LLM response.")

    return config_dict


def parse_columns_from_create_statement(create_statement):
    """Parse the CREATE statement to get a list of column names"""

    # Remove backticks, split the string by newline character and trim whitespaces
    lines = create_statement.replace("`", "").strip().split("\n")

    column_names = []

    # Iterate over lines
    for line in lines:
        # Stop after the closing parenthesis
        if line.startswith(")") or line.startswith("OPTIONS("):
            break

        # Skip the lines not representing column definitions
        if line.startswith("CREATE TABLE") or line.startswith("("):
            continue

        # Extract column name using regular expression
        match = re.match(r'\s*([^ ]+)', line)

        # If match is found, add to the list of column names
        if match:
            column_names.append(match.group(1))

    return column_names


def process_sql_files(directory):
    """Create a YML file for each SQL file that doesn't already have one"""
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    for dirpath, _, filenames in os.walk(directory):
        yml_files = [os.path.join(dirpath, f)
                     for f in filenames if f.endswith('.yml')]
        sql_files = [os.path.join(dirpath, f)
                     for f in filenames if f.endswith('.sql')]

        for sql_file in sql_files:
            model_name = os.path.splitext(os.path.basename(sql_file))[0]
            if not any(model_name in yml_file for yml_file in yml_files):
                new_yml_file = os.path.join(dirpath, f"{model_name}.yml")
                with open(new_yml_file, 'w', encoding="utf-8") as f:
                    data = {
                        'version': 2,
                        'models': [
                            {
                                'name': model_name,
                                'description': "",
                                'columns': [],
                                'tests': [],
                                'docs': {'show': True}
                            }
                        ]
                    }
                    yml = yaml.YAML()
                    yml.width = 80
                    yml.indent(mapping=2, sequence=4, offset=2)
                    # yml.yaml_set_comment_before_after_key(0, before='\n')
                    yml.dump(data, f)
                yml_files.append(new_yml_file)


def process_yml_files(directory):
    """Process each YML file in the directory"""

    sample_data = {}
    create_statement = {}
    for dirpath, _, filenames in os.walk(directory):
        yml_files = [os.path.join(dirpath, f)
                     for f in filenames if f.endswith('.yml')]

        for yml_file in yml_files:
            with open(yml_file, 'r', encoding="utf-8") as f:
                # print(f"Processing {yml_file}")
                try:
                    modified_file_stream = replace_tab_with_spaces(f)
                    data = yaml.safe_load(modified_file_stream)
                    if data and 'models' in data:
                        for model in data['models']:
                            model_name = model['name']
                            try:
                                with engine.connect() as connection:
                                    try:
                                        # Query sample data from BigQuery
                                        result = connection.execute(text(
                                            f'SELECT * FROM `{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{model_name}` TABLESAMPLE SYSTEM '
                                            '(1 PERCENT) QUALIFY ROW_NUMBER() OVER (ORDER BY RAND()) <= 3'
                                        ))
                                    except SQLAlchemyError as e:
                                        print(
                                            f"Skipping {model_name} due to materialization not found in Data Warehouse")
                                        continue
                                    sample_data[model_name] = '\n'.join(
                                        str(row._mapping) for row in result)

                                    # Query the CREATE statement for the model
                                    try:
                                        result = connection.execute(text(
                                            f'SELECT ddl FROM `{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.INFORMATION_SCHEMA.TABLES` WHERE table_name = "{model_name}"'
                                        ))
                                    except SQLAlchemyError as e:
                                        print(
                                            f"Skipping {model_name} due to materialization not found in Data Warehouse")
                                        continue
                                    row = result.fetchone()
                                    ddl = str(row._mapping['ddl'])
                                    create_statement[model_name] = ddl
                            except SQLAlchemyError as e:
                                print(
                                    f"Skipping {model_name} due to Data Warehouse connection error: {e}")
                                exit(1)
                            # Parse the CREATE statement to get a list of column names
                            db_columns = parse_columns_from_create_statement(
                                create_statement[model_name]
                            )

                            # Get a list of column names from the YML file
                            if 'columns' in model:
                                yml_columns = [column['name']
                                               for column in model['columns']]

                                # Find the columns that are in the YML file but not in the database
                                extra_columns = [
                                    column for column in yml_columns if column not in db_columns]

                                # Remove the extra columns from the YML file
                                model['columns'] = [
                                    column for column in model['columns'] if column['name'] not in extra_columns]
                            else:
                                yml_columns = []
                                model['columns'] = []

                            # Add the columns that are in the database but not in the YML file
                            for column in db_columns:
                                if column not in yml_columns:
                                    model['columns'].append({'name': column})

                            # Ensure each column has a 'description' and 'tests' child key
                            for column in model['columns']:
                                if 'description' not in column:
                                    column['description'] = ""
                                if 'tests' not in column:
                                    column['tests'] = []

                            # Ensure each model has a 'description', 'tests', and 'docs' child key
                            if 'description' not in model:
                                model['description'] = ""
                            if 'tests' not in model:
                                model['tests'] = []
                            if 'docs' not in model:
                                model['docs'] = {'show': True}
                            elif 'show' not in model['docs']:
                                model['docs']['show'] = True
                    else:
                        print(
                            f"Skipping {yml_file} due to missing 'models' section")

                except yaml.YAMLError as exc:
                    print("Unable to read file: " + str(exc))

            with open(yml_file, 'w', encoding="utf-8") as f:
                try:
                    yml = yaml.YAML()
                    yml.width = 80
                    yml.indent(mapping=2, sequence=4, offset=2)
                    # yml.yaml_set_comment_before_after_key(0, before='\n')
                    yml.dump(data, f)
                except yaml.YAMLError as exc:
                    print("Unable to write file: " + str(exc))

            with open(yml_file, 'r', encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f)
                    if data and 'models' in data:
                        for idx, model in enumerate(data['models']):
                            model_name = model['name']
                            model_dump = json.dumps(model)
                            if (model_name in sample_data) and (model_name in create_statement):
                                prompt_payload = chat_prompt.format(
                                    text=model_dump,
                                    samples=sample_data[model_name],
                                    create_statement=create_statement[model_name]
                                )
                                if count_num_tokens(prompt_payload) > 12000:
                                    print(
                                        f"Skipping LLM enrichment for {model_name} due to payload size")
                                    continue
                                try:
                                    chain_results = llm_chain({
                                        'text': model_dump,
                                        'samples': sample_data[model_name],
                                        'create_statement': create_statement[model_name]
                                    })
                                except OutputParserException:
                                    try:
                                        chain_results = llm_chain({
                                            'text': model_dump,
                                            'samples': sample_data[model_name],
                                            'create_statement': create_statement[model_name]
                                        })
                                    except OutputParserException:
                                        print(
                                            f"Skipping LLM enrichment for {model_name} due to OutputParserException")
                                        continue
                            else:
                                continue

                            try:
                                enriched_contents = parse_llm_response(
                                    result=chain_results
                                )
                            except OutputParserException:
                                print(
                                    f"Skipping LLM enrichment for {model_name} due to invalid response")
                                continue

                            print(f"Processed {model_name}")
                            data['models'][idx] = enriched_contents
                    else:
                        print(
                            f"Skipping {yml_file} due to missing 'models' section")

                except yaml.YAMLError as exc:
                    print("Unable to read file: " + str(exc))

            with open(yml_file, 'w', encoding="utf-8") as f:
                try:
                    yml = yaml.YAML()
                    yml.width = 80
                    yml.indent(mapping=2, sequence=4, offset=2)
                    # yml.yaml_set_comment_before_after_key(0, before='\n')
                    yml.dump(data, f)
                except yaml.YAMLError as exc:
                    print("Unable to write file: " + str(exc))


if len(sys.argv) != 2:
    print("Error: Please provide the directory to process.")
    print("Example: python run.py /path/to/directory")
    sys.exit(1)

path = Path(sys.argv[1]).resolve()
if not path.is_dir():
    print(f"Error: {sys.argv[1]} is not a valid directory.")
    sys.exit(1)

process_sql_files(path)
process_yml_files(path)
