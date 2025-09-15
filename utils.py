# call_agent_vexu-main/utils.py:
import os
from dotenv import load_dotenv
import json
import collections.abc

def get_env_variable(var_name, var_type=str):
    load_dotenv() 
    """
    Retrieves an environment variable and raises an error if it's not set.
    Optionally converts the variable to a specified type.
    Attempts to strip surrounding quotes from string values before conversion.
    """
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Error: Environment variable '{var_name}' is not set.")
    
    # Attempt to strip common surrounding quotes if value is a string
    if isinstance(value, str):
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

    if var_type == float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Error: Environment variable '{var_name}' with value '{value}' cannot be converted to float.")
    elif var_type == int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Error: Environment variable '{var_name}' with value '{value}' cannot be converted to int.")
    elif var_type == bool:
        # For boolean, we'll consider 'true', '1', 'yes' (case-insensitive) as True
        # and 'false', '0', 'no' (case-insensitive) as False.
        # Any other value for a boolean will raise an error.
        val_lower = value.lower()
        if val_lower in ['true', '1', 'yes']:
            return True
        elif val_lower in ['false', '0', 'no']:
            return False
        else:
            raise ValueError(f"Error: Environment variable '{var_name}' with value '{value}' cannot be converted to bool. Use 'true'/'false', '1'/'0', or 'yes'/'no'.")
    return value # Defaults to string if no specific type conversion needed or if var_type is str

def _truncate_strings_recursive(obj, max_len):
    """
    Recursively traverses an object, truncating strings longer than max_len.
    Creates new lists/dicts to avoid modifying the original object.
    """
    if isinstance(obj, str):
        if len(obj) > max_len:
            # Truncate and add an indicator
            return obj[:max_len] + '...'
        else:
            return obj
    elif isinstance(obj, collections.abc.Mapping): # Handles dictionaries
        # Create a new dict
        new_dict = {}
        for key, value_item in obj.items(): # Renamed 'value' to 'value_item' to avoid conflict
            # Recursively process the key (in case keys are strings, though unusual for JSON)
            # and the value
            processed_key = _truncate_strings_recursive(key, max_len)
            processed_value = _truncate_strings_recursive(value_item, max_len)
            new_dict[processed_key] = processed_value
        return new_dict
    elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes)):
        # Handles lists, tuples but not strings/bytes which are also sequences
        # Create a new list
        new_list = []
        for item in obj:
            new_list.append(_truncate_strings_recursive(item, max_len))
        # Return a list, as JSON uses arrays for both lists and tuples
        return new_list
    else:
        # For other types (int, float, bool, None), return as is
        return obj

def truncated_json_dumps(obj, max_string_len=1000, **kwargs):
    """
    Serializes an object to a JSON formatted string, like json.dumps,
    but truncates any string value exceeding max_string_len.

    Args:
        obj: The Python object to serialize.
        max_string_len (int): The maximum allowed length for string values.
                              Strings longer than this will be truncated and '...' appended.
                              Defaults to 1000.
        **kwargs: Additional keyword arguments to pass directly to json.dumps
                  (e.g., indent, sort_keys, separators).

    Returns:
        str: A JSON formatted string with long strings truncated.

    Raises:
        TypeError: If the processed object cannot be serialized by json.dumps.
    """
    if not isinstance(max_string_len, int) or max_string_len < 0:
        raise ValueError("max_string_len must be a non-negative integer")

    # Process the object recursively to truncate strings
    # This creates copies of structures containing strings, avoiding modification
    # of the original object.
    processed_obj = _truncate_strings_recursive(obj, max_string_len)

    # Use the standard json.dumps on the processed object
    return json.dumps(processed_obj, **kwargs)
