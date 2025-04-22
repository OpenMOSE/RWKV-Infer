from jinja2 import Template
import json

# 修正済みテンプレート
template_str = """
{% if messages[0]['role'] == 'system' %}
    {% set merged_content = messages[0]['content'] + ' ' + messages[1]['content'] %}
    {% set merged_messages = [{'role': messages[1]['role'], 'content': merged_content}] + messages[2:] %}
{% else %}
    {% set merged_messages = messages %}
{% endif %}
{% for message in merged_messages %}
    {{ ('human' if message['role'] == 'user' else message['role']) + ': ' + (message['content'].split('<reasoning>')|first + message['content'].split('</reasoning>')|last if message['role'] == 'assistant' and '</reasoning>' in message['content'] else message['content']) }}
    {% if (loop.last and add_generation_prompt and merged_messages[-1]['role'] != 'assistant') or not loop.last %}
        {{ ' <sep> ' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt and merged_messages[-1]['role'] != 'assistant' %}
    {{ 'assistant:' }}
{% else %}
    {{ eos_token }}
{% endif %}
"""

# メッセージ例
messages = [
    {"role": "user", "content": "こんにちは！"},
    {"role": "assistant", "content": "こんにちは、何をお手伝いできますか？<reasoning>これは推論です</reasoning>"}
]

# テンプレートをレンダリング
template = Template(template_str)
prompt = template.render(messages=messages, add_generation_prompt=False, eos_token="</s>")