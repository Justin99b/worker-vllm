import re
import json
from typing import Sequence, Any, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager
)


@ToolParserManager.register_module(["commandr"])
class CommandRToolParser(ToolParser):
    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # No changes to the request in this minimal example.
        return request

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # Streaming extraction is not implemented in this minimal example.
        return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Finds 'Action: ```json [...]```' in the model output,
        parses the JSON, and transforms it into vLLM-compatible ToolCall objects.
        """
        pattern = r'Action:\s*```json\s*(\[[\s\S]*?\])\s*```'
        match = re.search(pattern, model_output, re.DOTALL)

        if match:
            json_str = match.group(1)
            try:
                # raw_tool_calls could be list or single dict
                raw_tool_calls = json.loads(json_str)
                if not isinstance(raw_tool_calls, list):
                    raw_tool_calls = [raw_tool_calls]

                tool_calls_list = []
                for call in raw_tool_calls:
                    # The model might have "tool_name", "parameters" keys
                    tool_name = call.get("tool_name", "")
                    raw_parameters = call.get("parameters", {})

                    # Convert raw_parameters into a JSON string (for FunctionCall.arguments)
                    arguments_str = ""
                    if isinstance(raw_parameters, dict):
                        # straightforward dict => dump to JSON
                        arguments_str = json.dumps(raw_parameters)
                    elif isinstance(raw_parameters, str):
                        # might already be JSON-encoded; try to parse & re-encode
                        try:
                            parsed = json.loads(raw_parameters)
                            arguments_str = json.dumps(parsed)
                        except json.JSONDecodeError:
                            # not valid JSON; keep it as is
                            arguments_str = raw_parameters
                    else:
                        # fallback for e.g. numbers, lists, etc. => just dump
                        arguments_str = json.dumps(raw_parameters)

                    # Build the ToolCall
                    tool_calls_list.append(
                        ToolCall(
                            function=FunctionCall(
                                name=tool_name,
                                arguments=arguments_str
                            )
                        )
                    )

                # Extract content before the "Action:" portion
                content = model_output[:match.start()].strip()

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls_list,
                    content=content
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, treat everything as regular text
                pass

        # If no match or JSON parse failed:
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output.strip()
        )
