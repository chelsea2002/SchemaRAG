from openai import OpenAI
from arg import main_args
from retrying import retry

client = OpenAI()
args = main_args()

@retry(stop_max_attempt_number=3, wait_fixed=3000)
def LLM_generation(instruct, prompt, n=args.n):  
    response = client.chat.completions.create(
        model=args.gpt,
        messages=[
            {"role": "system", "content": instruct},
            {"role": "user", "content": prompt}
        ],
        n=n,
    )
    results = [choice.message.content for choice in response.choices]
    return results


