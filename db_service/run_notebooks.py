import sys
import os
import base64
from os.path import isfile, join
from os import listdir
import argparse
import time

# Stetting up log
parentPath = os.path.abspath(".")
print('parent path', parentPath)
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from db_workspace_api import DatabricksWorkspace

parser = argparse.ArgumentParser('Run job in DB')
parser.add_argument('-job_id', '--job_id')
parser.add_argument('-dbx_api_token', '--dbx_token')
parser.add_argument('-dbx_ws_url', '--dbx_org_url')

args = parser.parse_args()

job_id = args.job_id
dbx_token = args.dbx_token
dbx_ws_url = args.dbx_org_url

# Initialize databricks workspace class
dbw = DatabricksWorkspace(dbxtoken=dbx_token, dbx_org_url=dbx_ws_url)

notebook_params = {
    "train_type": "CT_pipeline"
}

# Initiate job
r_run_job, error = dbw.run_job(job_id, notebook_params)
if error:
    raise Exception(f"Error initiating job")
run_id = r_run_job.json()["run_id"]

# Verify run completed
r_run_status, error = dbw.get_run_status(run_id)
if error:
    raise Exception(f"Error getting run status")

while r_run_status != "TERMINATED":
    r_run_status, error = dbw.get_run_status(run_id)
    if error:
        raise Exception(f"Error getting run status")
    time.sleep(5)

print("Run completed/terminated")