from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

completion = client.chat.completions.create(
  model="coqcopilot7b",
  messages=[
    {"role": "user", "content": "Solve This Proof State:\n\nHypotheses:\np: nat\nHp: not (@eq nat p O)\n\nGoal:\nZp_invertible Zp_one\n\n"}
  ]
)

print(completion.choices[0].message)