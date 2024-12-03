import os

def get_root_dir():
    dockerizable_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    return dockerizable_dir
print(get_root_dir())
def get_data_dir():
    return os.path.join(get_root_dir(), '', 'data')
print(get_data_dir())

def get_fcts_dir():
    return os.path.join(get_root_dir(), '', 'fcts')
print(get_fcts_dir())