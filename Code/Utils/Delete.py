from os import remove, mkdir, walk
from os.path import dirname, abspath, exists, isdir

def delete_file(file: str)-> bool:
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False