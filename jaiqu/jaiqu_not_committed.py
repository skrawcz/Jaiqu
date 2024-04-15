import pprint
from typing import List, Optional, Tuple

import burr.core
from burr.core import Action, Application, ApplicationBuilder, State, default, expr, when
from burr.core.action import action
from burr.lifecycle import LifecycleAdapter, PostRunStepHook, PreRunStepHook

import jq
import json
from jsonschema import validate
from tqdm.auto import tqdm  # Use the auto submodule for notebook-friendly output if necessary
from helpers import identify_key, create_jq_string, repair_query, dict_to_jq_filter


@action(reads=["output_schema", "input_json", "key_hints"], writes=["valid_input_schema", "schema_properties"])
def validate_input_schema(state: State) -> tuple[dict, State]:
    output_schema = state["output_schema"]
    input_json = state["input_json"]
    key_hints = state.get("key_hints")
    results = {}
    valid = True
    with tqdm(total=len(output_schema['properties']), desc="Validating schema") as pbar:
        for key, value in output_schema['properties'].items():
            pbar.set_postfix_str(f"Key: {key}", refresh=True)
            response_key, response_reasoning = identify_key(key, value, input_json, None, key_hints)

            if response_key is not None:
                results[key] = {"identified": True, "key": response_key,
                                "message": response_reasoning,
                                **value}
            else:
                results[key] = {"identified": False, "key": response_key,
                                "message": response_reasoning,
                                **value}
            if key in output_schema['required']:
                results[key]['required'] = True
                if results[key]['identified'] == False:
                    valid = False
            else:
                results[key]['required'] = False
            pbar.update(1)

        return results, state.update(valid_input_schema=valid, schema_properties=results)

@action(reads=["valid_input_schema", "schema_properties", "reason"], writes=[])
def error_state(state: State) -> tuple[dict, State]:
    if state["valid_input_schema"] is False:
        schema_properties = state["schema_properties"]
        message = (f"The input JSON does not contain the required data to satisfy the output schema: \n\n{json.dumps(schema_properties, indent=2)}")
    elif state["status"] == "Failed n times":
        message = state["reason"]
    else:
        message = "Unknown error occurred."
    return {"error_message": message}, state.update(error_message=message)

@action(reads=["input_json", "schema_properties"], writes=["jq_strings_to_process"])
def create_jq_filter_query(state: State) -> tuple[dict, State]:
    schema_properties = state["schema_properties"]
    input_json = state["input_json"]
    filtered_schema = {k: v for k, v in schema_properties.items() if v['identified'] == True}

    jq_strings_to_process = []
    for key, value in filtered_schema.items():
        jq_string = create_jq_string(input_json, key, value, None)

        if jq_string == "None":  # If the response is empty, skip the key
            continue
        jq_strings_to_process.append((key, jq_string))
    state = state.update(jq_strings_to_process=jq_strings_to_process)
    return {"to_process": jq_strings_to_process}, state

@action(reads=["jq_strings_to_process", "current_filter", "tries"],
        writes=["current_filter", "status", "tries", "filter_query", "error"])
def compile_and_test_jq_filter(state: State) -> tuple[dict, State]:
    jq_strings_to_process = state["jq_strings_to_process"]
    if not jq_strings_to_process:
        return {"status": "Success"}, state.update(status="Success")

    current_filter = state.get("current_filter")
    if current_filter is None:
        key, jq_string = jq_strings_to_process.pop()
        tries = 0
        state = state.update(
            current_filter=(key, jq_string),
            tries=tries,
            jq_strings_to_process=jq_strings_to_process
        )
    else:
        tries = state["tries"] + 1
        key, jq_string = current_filter
    try:
        jq.compile(jq_string).input(state["input_json"]).all()
    except Exception as e:
        error = str(e)
        return {"status": "Fail", "error": error}, state.update(tries=tries, status="Fail", error=error)
    else:
        filter_query = state.get("filter_query", {})
        filter_query[key] = jq_string

    state = state.update(status="Continue", tries=0, current_filter=None, filter_query=filter_query)
    return {"status": "Continue"}, state

@action(reads=["current_filter", "error", "input_json"], writes=[])
def retry_create_jq_filter(state: State) -> tuple[dict, State]:
    key, jq_string = state.get("current_filter")
    tries = state["tries"]
    if tries >= state["max_retries"]:
        reason = f"Failed to create a valid jq filter for key '{key}' after {state['max_retries']} retries."
        return {"status": "Failed n times", "reason":reason}, state.update(status="Failed n times", reason=reason)

    error = state["error"]
    jq_string = repair_query(jq_string, error, state["input_json"], None)
    state = state.update(current_filter=(key, jq_string))
    return {"status": "Success"}, state

@action(reads=["filter_query", "output_schema", "input_json"], writes=[])
def validate_json(state: State) -> tuple[dict, State]:
    input_json = state["input_json"]
    output_schema = state["output_schema"]
    filter_query = state.get("filter_query", {})
    complete_filter = dict_to_jq_filter(filter_query)
    try:
        result = jq.compile(complete_filter).input(input_json).all()[0]
        validate(instance=result, schema=output_schema)
    except Exception as e:
        error = str(e)
        return {"status": "Fail", "error": error}, state.update(status="Fail", error=error)
    return {"status": "Success", "final_filter": complete_filter}, state.update(status="Success", final_filter=complete_filter)

@action(reads=[], writes=[])
def retry_validate_json(state: State) -> tuple[dict, State]:
    tries = state.get("tries", 0)
    if tries >= max_retries:
        raise RuntimeError(f"Failed to validate the jq filter after {max_retries} retries.")
    complete_filter = repair_query(complete_filter, str(e), input_json, openai_api_key)

    return {}, state

@action(reads=[], writes=[])
def final_jq_query_string(state: State) -> tuple[dict, State]:
    return {}, state

if __name__ == '__main__':
    app = (
        ApplicationBuilder()
        .with_state(
            **{
                "input_json": "",
                "output_schema": "",
            }
        )
        .with_actions(
            # bind the vector store to the AI conversational step
            validate_input_schema=validate_input_schema,
            error_state=error_state,
            create_jq_filter_query=create_jq_filter_query,
            compile_and_test_jq_filter=compile_and_test_jq_filter,
            retry_create_jq_filter=retry_create_jq_filter,
            validate_json=validate_json,
            retry_validate_json=retry_validate_json,
            final_jq_query_string=final_jq_query_string,
        )
        .with_transitions(
            ("validate_input_schema", "error_state", when(valid_input_schema=False)),
            ("validate_input_schema", "create_jq_filter_query", when(valid_input_schema=True)),
            ("create_jq_filter_query", "compile_and_test_jq_filter", default),
            ("compile_and_test_jq_filter", "retry_create_jq_filter", when(status="Fail")),
            ("compile_and_test_jq_filter", "compile_and_test_jq_filter", when(status="Continue")),
            ("retry_create_jq_filter", "compile_and_test_jq_filter", when(status="Success")),
            ("retry_create_jq_filter", "error_state", when(status="Failed n times")),
            ("compile_and_test_jq_filter", "validate_json", when(status="Success")),
            ("validate_json", "final_jq_query_string", when(status="Success")),
            ("validate_json", "retry_validate_json", when(status="Fail")),
            ("retry_validate_json", "final_jq_query_string", when(status="Success")),
            ("retry_validate_json", "error_state", when(status="Failed n times")),
        )
        .with_entrypoint("validate_input_schema")
        .with_tracker(project="example:jaiqu")
        .with_identifiers(partition_key="dagworks")
        .build()
    )
    app.visualize(
        output_file_path="jaiqu_burr", include_conditions=True, view=True, format="png"
    )