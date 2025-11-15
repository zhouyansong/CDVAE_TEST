import os
import pathlib
import pandas as pd
import shutil

path = os.getcwd()
rootdir = pathlib.Path(path)/'c2db-direct'
df = pd.read_csv('c2db_with_cif.csv')

test_dir = pathlib.Path(path) / 'test'
test_dir.mkdir(exist_ok=True)

for material_id in df['material_id']:
    material_file = rootdir / f"{material_id}.vasp"  
    shutil.copy(material_file, test_dir / material_file.name)
