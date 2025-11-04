import os
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic, AsyncClient
from transformers import AutoTokenizer

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


client = AsyncClient(
    # api_key=os.getenv('ANTHROPIC_API_KEY')
    base_url=os.getenv('ANTHROPIC_BASE_URL'),
    auth_token=os.getenv('ANTHROPIC_AUTH_TOKEN'),

)
MODEL = 'claude-haiku-4-5'
# MODEL = 'claude-haiku-4-5'
# MODEL = 'claude-opus-4-1'
# MODEL = 'claude-sonnet-4-5'

async def send_message(content: str):
    message = await client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    return message

async def main():
    PROMPT = 'Write a super short software joke in Polish.'
    tokens = tokenizer.encode(PROMPT)
    print(tokens)
    response = await send_message(PROMPT)
    text = response.content[0].text
    print(text)
    tokens = tokenizer.encode(text)
    print(tokens)

if __name__ == '__main__':
    asyncio.run(main())
