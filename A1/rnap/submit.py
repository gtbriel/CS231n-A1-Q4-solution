import os
import zipfile


_A1_FILES = [
  'pytorch101.py',
  'pytorch101.ipynb',
  'rede_duas_camadas.py',
  'rede_duas_camadas.ipynb',
]


def make_a1_submission(assignment_path, uniquename=None, umid=None):
  _make_submission(assignment_path, _A1_FILES, 'A1', uniquename, umid)


def _make_submission(
    assignment_path,
    file_list,
    assignment_no,
    uniquename=None,
    umid=None):
  if uniquename is None or umid is None:
    uniquename, umid = _get_user_info()
  zip_path = '{}_{}_{}.zip'.format(uniquename, umid, assignment_no)
  zip_path = os.path.join(assignment_path, zip_path)
  print('Gravando arquivo zip em: ', zip_path)
  with zipfile.ZipFile(zip_path, 'w') as zf:
    for filename in file_list:
      in_path = os.path.join(assignment_path, filename)
      if not os.path.isfile(in_path):
        raise ValueError('Não foi possível encontrar o arquivo "%s"' % filename)
      zf.write(in_path, filename)


def _get_user_info():
  if uniquename is None:
    uniquename = input('Digite seu primeiro nome (por exemplo, jurandy): ')
  if umid is None:
    umid = input('Digite seu RA (por exemplo, 12345678):')
  return uniquename, umid
