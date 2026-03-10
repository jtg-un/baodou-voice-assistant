import sys
from openai import OpenAI


class LLMEngine:
    def __init__(self):
        # 初始默认值
        self.api_key = ""
        self.base_url = "https://api.deepseek.com"
        self.model_name = "deepseek-chat"
        self.client = None

        # 默认人设
        self.system_prompt = "你是一个正在面试Java岗位的应届毕业生。回复请保持在50字以内。"
        self.history = [{"role": "system", "content": self.system_prompt}]

    def update_config(self, api_key, base_url, model_name, system_prompt):
        """
        由 UI 线程调用，实时更新连接参数和系统提示词
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = system_prompt

        # 重新实例化客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 关键：更新配置后必须重置对话历史，否则旧的人设会干扰新的回复
        self.history = [{"role": "system", "content": self.system_prompt}]
        print(f"⚙️ 后端配置已同步：模型 {self.model_name}, 人设已重置")

    def get_reply(self, text):
        if not self.client or not self.api_key:
            return "❌ 请先在配置面板填写 API Key 并点击保存！"

        # 添加用户输入到历史
        self.history.append({"role": "user", "content": text})

        try:
            # 发起流式请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                stream=True
            )

            print("🤖 助手: ", end="", flush=True)
            full_response = ""

            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        # 在控制台实时打印，方便调试
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        full_response += content

            sys.stdout.write("\n")

            # 将 AI 回复存入历史
            self.history.append({"role": "assistant", "content": full_response})

            # --- 对话管理：保持最近 6 轮对话，防止上下文过长 ---
            if len(self.history) > 7:  # 1个system + 6个对话
                self.history = [self.history[0]] + self.history[-6:]

            return full_response

        except Exception as e:
            error_msg = f"⚠️ API 调用失败: {str(e)}"
            print(f"\n{error_msg}")
            return error_msg