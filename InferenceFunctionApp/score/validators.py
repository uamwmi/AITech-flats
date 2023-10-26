# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

from jsonschema import validate, Draft202012Validator

score_function_input_schema = {
    "type": "object",
    "properties": {
        "image": {
            "type": "object",
            "properties": {
                "mime": {"type": "string"},
                "data": {"type": "string"},
            },
            "required": ["data"],
        }
    },
}
schema_validator = Draft202012Validator(score_function_input_schema)


def validate_score_function_input(json_body):
    """Validates the input JSON body against a schema.

    Args:
        json_body (str): JSON body

    Raises:
        ValidationError: If the input JSON body is not valid.
    """
    schema_validator.is_valid(json_body)
    validate(instance=json_body, schema=score_function_input_schema)
