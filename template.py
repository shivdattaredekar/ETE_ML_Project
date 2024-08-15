import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s]: %(message)s:')

project_name = 'Mlproject'


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py"


]


for filepath in list_of_files:
    filepath = Path(filepath) #This will detect the OS and path to handle the issues with working in different OS's
    filedir, filename = os.path.split(filepath) #this is use to separate the foldername and filenames
    # but what if filedir is empty then?
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory {filedir} for file {filename}')
    
    # but what if filepath is not there? This is to make sure if there is some content in any file we will not 
        #overwrite it. insted we will create another file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file {filepath}')
    else:
        logging.info(f'{filepath} is already exists')
    

















