---
title: AutoLearn API
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Self-hosted, OpenAI-compatible AI API platform
---

# AutoLearn API: live demo

An OpenAI-compatible AI platform: chat, function calling, guaranteed JSON, embeddings, answers from your own
documents, batch jobs and per-customer custom models, all on its own inference engine.

Click **Get a free API key** on the page, then try the models in the console, or call the API with the
official OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="https://likhith003-autolearn.hf.space/v1", api_key="<your demo key>")
print(client.chat.completions.create(model="autolearn-lite",
      messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
```

This demo runs on a free 2-CPU machine, so answers take a while. Use `autolearn-lite` for faster replies.
Data resets when the Space restarts.

Source code: https://github.com/likhithy99/autolearn-api
