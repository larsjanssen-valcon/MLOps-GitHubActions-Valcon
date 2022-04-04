import requests
from os.path import isfile, join
from os import listdir, mkdir


class DatabricksWorkspace():

    def __init__(self, dbxtoken, dbx_org_url):
        self.token = dbxtoken
        self.header = {"Authorization": "Bearer " + self.token}
        self.dbrks_org_url = dbx_org_url

    def make_directory(self, path_name: str) -> dict:
        '''
        Create a directory in the Databricks workspace      
        Parameters:
            path_name: databricks directory path
            e.g. '/Development/Notebooks'        
        Returns:
            r: response of the api call
            e: exception call (if exception is found during execution)
        '''
        mkdir_endpoint = "api/2.0/workspace/mkdirs"
        dbrks_mkdir_rest_url = f"{self.dbrks_org_url}/{mkdir_endpoint}"

        body = {
        "path": f'{path_name}'
        }
        try:
            r = requests.post(dbrks_mkdir_rest_url, headers = self.header, json = body)
            return r, None
        except Exception as e:
            return None, e

    def upload_notebooks_dbx(self, content: bytes, dbx_path: str) -> dict:
        """ Upload notebooks to the Databricks workspace

            Parameters:
                path_name: databricks directory path
                e.g. '/Development/Notebooks'        
                Returns:
                r: response of the api call
                e: exception call (if exception is found during execution)    
        """
        import_endpoint = "api/2.0/workspace/import"
        dbrks_import_rest_url = f"{self.dbrks_org_url}/{import_endpoint}"

        body = {
            "content": content,
            "path": dbx_path,
            "language": "PYTHON",
            "overwrite": True,
            "format": "SOURCE"
            }

        try:
            r = requests.post(dbrks_import_rest_url, headers = self.header, json = body)
            return r, None
        except Exception as e:
            return None, e

    def delete_object(self, path_name: str):  
        '''
        Delete a directory or notebook (deletes also all content if it is a directory)
        Parameters:
            path_name: Databricks workspace path of the folder or notebook to be deleted
                    example: /Development/FOLDER/FOLDER_2/....
        Returns:
            r: response of the api call
            e: exception call (if exception is found during execution)
        '''
        delete_endpoint = "api/2.0/workspace/delete"
        dbrks_clearws_rest_url = f"{self.dbrks_org_url}/{delete_endpoint}"
        
        body = {
        "path": f'{path_name}',
        "recursive": "true"
        }

        try:
            r = requests.post(dbrks_clearws_rest_url, headers = self.header, json = body)
            return r, None
        except Exception as e:
            return None, e

    def run_job(self, job_id: str, notebook_params: dict):
        """
        Method for running a set job on DB workspace
        """
        run_endpoint = f"{self.dbrks_org_url}/api/2.0/jobs/run-now"

        body = {
            "job_id": job_id,
            "notebook_params": notebook_params
        }

        try:
            r = requests.post(run_endpoint, headers=self.header, json=body)
            return r, None
        except Exception as e:
            return None, e

    def get_run_status(self, run_id: str):
        """
        Method for getting run status of job
        """
        run_endpoint = f"{self.dbrks_org_url}/api/2.0/jobs/runs/get"

        body = {
            "run_id": run_id
        }

        try:
            r = requests.get(run_endpoint, headers=self.header, json=body)
            status = r.json()['state']['life_cycle_state']
            return status, None
        except Exception as e:
            return None, e