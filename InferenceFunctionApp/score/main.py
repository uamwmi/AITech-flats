# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

"""Ignore import error for pylint."""
# pylint: disable=import-error
import json
import logging
import time

import azure.functions as func
from jsonschema.exceptions import ValidationError

from score import validators
from score.predict import predict_classes_from_base64_image


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main input function invoked on HTTP request.

    Args:
        req (func.HttpRequest): HTTP request with JSON body.

    Returns:
        func.HttpResponse: HTTP response with JSON body.
    """
    logging.info("Python HTTP trigger function processed a request.")

    try:
        json_body = json.loads(req.get_body().decode("utf-8"))
        validators.validate_score_function_input(json_body)
        start = time.time()
        image_base64 = json_body["image"]["data"]
        logging.info("Starting prediction for image")
        results = predict_classes_from_base64_image(image_base64)

        headers = {
            "Content-type": "application/json",
            "Access-Control-Allow-Origin": "*",
        }
        result_json = str(json.dumps(results))
        end = time.time()

        time_seconds = round(end - start, 5)
        logging.info(
            "Prediction took %s seconds (%s milliseconds)",
            time_seconds,
            time_seconds * 1000,
        )
        return func.HttpResponse(result_json, headers=headers)

    except ValueError as exception:
        logging.error(exception)
        return func.HttpResponse(
            "Bad request. Check if input is in json format.", status_code=400
        )
    except ValidationError as exception:
        logging.error(exception)
        return func.HttpResponse(
            "Bad request. Check json schema structure and values.",
            status_code=400)
    except Exception as exception:
        logging.error(exception)
        return func.HttpResponse("Error", status_code=500)
