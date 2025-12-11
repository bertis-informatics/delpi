import xml.etree.ElementTree as ET

from .. import DATA_DIR


PREFIX = r'{http://www.unimod.org/xmlns/schema/unimod_2}'    
UNIMOD_XML_PATH = DATA_DIR / 'unimod.xml'


class UniModDatabase(object):

    def __init__(self) -> None:
        super().__init__()
        self.xml_path = UNIMOD_XML_PATH

    def __new__(cls):
        # make this class singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(UniModDatabase, cls).__new__(cls)
        return cls.instance

    def get_atoms(self):
        tree = ET.parse(self.xml_path)
        
        root = tree.getroot()
        brick_root = root.find(f'{PREFIX}mod_bricks')
        
        for child in brick_root:
            title = child.get('title') 
            full_name = child.get('full_name') 
            mono_mass = child.get('mono_mass') 
            avg_mass = child.get('avge_mass') 
            atom_elems = child.findall(f'{PREFIX}element')
            if len(atom_elems) == 1:
                yield {
                    'code': title, 
                    'name': full_name,
                    'avg_mass': avg_mass,
                    'mono_mass': mono_mass,
                }


    def get_modifications(self):

        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        mod_root = root.find(f'{PREFIX}modifications')

        for child in mod_root:
            mod_title = child.get('title') # PSI-MS name 
            mod_name = child.get('full_name') # description
            record_id = int(child.get('record_id'))
            delta_elem = child.find(f'{PREFIX}delta')
            atoms = [
                f'{atom_elem.get("symbol")}({atom_elem.get("number")})'
                    for atom_elem in delta_elem.findall(f'{PREFIX}element')
            ]
            composition_str = ' '.join(atoms)
            #composition_str = delta_elem.get('composition')
            yield {
                'accession_num': record_id, 
                'name': mod_title, 
                'description': mod_name, 
                'composition': composition_str
            }


