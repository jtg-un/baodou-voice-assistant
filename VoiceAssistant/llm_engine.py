import sys
from openai import OpenAI


class LLMEngine:
    def __init__(self):
        # 默认值（可以从 config.py 读取初始值）
        self.api_key = ""
        self.base_url = "https://api.deepseek.com"
        self.model_name = "deepseek-chat"
        self.client = None
        self.system_prompt = "你是一个正在面试Java岗位的应届毕业生。回复请保持在50字以内。"
        self.history = [{"role": "system", "content": self.system_prompt}]

    def update_config(self, api_key, base_url, model_name):
        """由 UI 线程调用，实时更新连接参数"""
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # 更新配置后重置对话历史，确保人设生效
        self.history = [{"role": "system", "content": self.system_prompt}]

    def get_reply(self, text):
        if not self.client:
            return "❌ 请先配置并保存 API 信息"

        self.history.append({"role": "user", "content": text})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                stream=True
            )
            full_response = ""
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        full_response += content

            self.history.append({"role": "assistant", "content": full_response})
            return full_response
        except Exception as e:
            return f"⚠️ 连接失败: {str(e)}"