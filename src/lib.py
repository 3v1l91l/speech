from pathlib import Path
import pandas as pd

def get_path_label_df(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df
