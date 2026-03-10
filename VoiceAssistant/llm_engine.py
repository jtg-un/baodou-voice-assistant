import sys

from openai import OpenAI

import config


class LLMEngine:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.BASE_URL,
        )
        self.history = [
            {"role": "system", "content": "你是一个正在面试java岗位的应届毕业大学生。回复请保持在50字以内，不要使用Markdown格式。"}
        ]

    def get_reply(self, text):
        self.history.append({"role": "user", "content": text})

        try:
            # DeepSeek 的流式请求
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=self.history,
                stream=True
            )

            print("🤖 助手: ", end="", flush=True)
            full_response = ""

            # 核心优化：Token 到达即打印
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()  # 核心：强制操作系统刷新缓冲区
                        full_response += content

            sys.stdout.write("\n")
            sys.stdout.flush()

            self.history.append({"role": "assistant", "content": full_response})
            # 保持短对话记忆，减少 API 运算量
            if len(self.history) > 5:
                self.history = [self.history[0]] + self.history[-4:]

            return full_response

        except Exception as e:
            print(f"\n❌ DeepSeek 连接失败: {e}")
            return str(e)