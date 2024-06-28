import json
from pathlib import Path


def parse_json(json_file):
    """
    Parse a json file and return a dictionary object
    """
    with open(json_file, 'r') as f:
        return json.load(f)



def main_pipeline_manifest(json_file):
    """
    Parse the pipeline manifest json file and verify that the required fields are present
    """
    manifest = parse_json(json_file)
    required_fields = ['base_path', 'mouse_name']
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Required field {field} not found in pipeline manifest")
    
    # ToDo, add more checks here








if __name__ == "__main__":
    json_file = Path('../examples/demo.json')
    main_pipeline_manifest(json_file)