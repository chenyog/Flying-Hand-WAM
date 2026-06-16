from typing import List, Type, Optional
from pydantic import BaseModel, Field
import json
import sys
from pathlib import Path
from openai import OpenAI

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "code_gen"))
from gpt_agent import qwen_api  # noqa: E402

model_name = "qwen3.7-plus"
client = OpenAI(
    api_key=qwen_api,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def generate(messages: List[dict], custom_format: Type[BaseModel]) -> Optional[BaseModel]:
    strformat = custom_format.schema_json()
    messages.append({
        "role": "system",
        "content": "you shall output a json object with the following format: " + strformat,
    })
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        temperature=0.8,
        top_p=1.0,
        response_format={"type": "json_object"},
    )
    json_content = response.choices[0].message.content

    if json_content:
        parsed_json = json.loads(json_content)
        return (custom_format.parse_obj(parsed_json)
                if hasattr(custom_format, "parse_obj") else custom_format.model_validate(parsed_json))

    return None


if __name__ == "__main__":
    pass
