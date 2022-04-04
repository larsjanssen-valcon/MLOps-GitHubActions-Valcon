import sys
import os
import base64
from os.path import isfile, join
from os import listdir
import argparse

# Stetting up log
parentPath = os.path.abspath(".")
print('parent path', parentPath)
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from db_workspace_api import DatabricksWorkspace

parser = argparse.ArgumentParser('My program')
parser.add_argument('-source_path', '--path_to_source')
parser.add_argument('-target_path', '--path_to_target')
parser.add_argument('-dbx_api_token', '--dbx_token')
parser.add_argument('-dbx_ws_url', '--dbx_org_url')

args = parser.parse_args()

source_path = args.path_to_source
target_path = args.path_to_target
dbx_token = args.dbx_token
dbx_ws_url = args.dbx_org_url

# Initialize databricks workspace class
dbw = DatabricksWorkspace(dbxtoken=dbx_token, dbx_org_url=dbx_ws_url)

path = source_path # notebooks location in repo
dbx_dirs = []
file_loc = []
for (dir, dirnames, filenames) in os.walk(path):
    dbx_dirs.append(dir.replace(path, "").replace("\\", "/"))
    for file in filenames:
        file_loc.append(dir + '/' + file)

dev_path = target_path #'/Development/notebooks'

# Delete notebooks before upload
r_clearws, error = dbw.delete_object(target_path)
if error:
    raise Exception(f"Deleting directory in the databricks workspace failed with error code {str(error)}")

# Create directories in dbx workspace. If the dir does exists, api call will do nothing
for dir in dbx_dirs:
    new_dir = target_path + "/" + dir
    r_mkdir, error = dbw.make_directory(new_dir)
    if error:
        raise Exception(f"Creating directory in the databricks workspace failed with error code {str(error)}")

for file in file_loc:
    dbx_file_loc = file.replace(path, "").replace("\\", "/")
    dbx_dir = "/".join(dbx_file_loc.split("/")[:-1])
    dbx_file = dbx_file_loc.split("/")[-1]
    # print(dbx_file)
    data = open(file, "rb").read()
    encoded_data = base64.b64encode(data).decode()
    if ".py" in dbx_file:
        file_name = dbx_file.split(".py")[0]
        dbx_path =  target_path + "/" + dbx_dir + "/" + file_name
        print(f"Databricks workspace path {dbx_path}")

        # Delete notebooks before upload
        r_clearws, error = dbw.delete_object(dbx_path)
        if error:
            raise Exception(f"Creating directory in the databricks workspace failed with error code {str(error)}")

        # Upload notebooks to databricks workspace
        r_upload, error = dbw.upload_notebooks_dbx(encoded_data, dbx_path)
        if error:
            raise Exception(f"Creating directory in the databricks workspace failed with error code {str(error)}")

        if r_upload.status_code == 200:
            print(r_upload.json)
        else:
            raise Exception(f"Error with code {str(r_upload.status_code)} and text {r_upload.text}" )