import os
import pickle

def save_data(DATA, file_path='robot-S-track.p'):
    """
    Function that saves images and labels to picke file.
    Load picke firstly to ensure existing is not overwritten
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as _file:
            data = pickle.load(_file)
    else:
        data = {}
    DATA.update(data)

    with open(file_path, 'wb') as _file:
        pickle.dump(DATA, _file)