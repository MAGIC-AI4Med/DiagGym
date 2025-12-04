"""
DiagAgent Gradio Web Interface
基于 README 中的 🤖DiagAgent 部分自动生成的可运行前端界面

功能说明:
- 支持多轮交互式诊断
- 输入患者信息(年龄、性别、主诉、病史等)
- 智能推荐检查项目或给出最终诊断
- 可查看完整对话历史
"""

import gradio as gr
from typing import List, Dict, Tuple
import os

# ============================================================================
# 系统提示词 (来自 README 示例)
# ============================================================================
DIAGNOSE_INSTRUCTION = """You are a medical AI assistant. Help the doctor with diagnosis by analyzing patient information, suggesting relevant tests, and providing a final diagnosis when sufficient information is available.

RESPONSE FORMAT:

If more information is needed:
```
Current diagnosis: [your diagnosis according to the information provided]
Based on the patient's initial presentation, the following investigation(s) should be performed: [one additional test]
Reason: [reason for the test]
```

If sufficient information exists for diagnosis:
```
The available information is sufficient to make a diagnosis.

Diagnosis: [Diagnosis result]
Reason: [Diagnosis reason]
```"""


# ============================================================================
# DiagAgent 模型类 (基于 README 中的示例代码)
# ============================================================================
class TransformersLocalDiagAgent:
    """
    基于 Transformers 库的本地 DiagAgent 模型
    注意: 需要先下载模型到本地或从 HuggingFace 加载
    """
    def __init__(self, model_name_or_path: str = "Henrychur/DiagAgent-14B",
                 max_tokens: int = 8192, temperature: float = 0.0) -> None:
        """
        初始化 DiagAgent 模型

        Args:
            model_name_or_path: 模型路径或 HuggingFace 模型名称
            max_tokens: 最大生成 token 数
            temperature: 采样温度 (0.0 为贪婪解码)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"正在加载模型: {model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                # 如果没有安装 flash_attention_2,可以注释掉下面这行
                # attn_implementation="flash_attention_2"
            )
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.device = self.model.device
            print(f"模型加载成功! 设备: {self.device}")

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用模拟模式运行...")
            self.model = None
            self.tokenizer = None

    def diagnose(self, messages: List[Dict[str, str]]) -> str:
        """
        执行多轮诊断

        Args:
            messages: 对话消息列表,格式为 [{"role": "system/user/assistant", "content": "..."}]

        Returns:
            生成的诊断响应
        """
        # 如果模型未加载成功,返回模拟响应
        if self.model is None or self.tokenizer is None:
            return self._mock_diagnose(messages)

        try:
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # 生成响应
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 移除提示词部分
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip().replace("```", "")

        except Exception as e:
            return f"诊断过程出错: {str(e)}"

    def _mock_diagnose(self, messages: List[Dict[str, str]]) -> str:
        """
        模拟诊断响应 (当模型未加载时使用)
        """
        user_content = messages[-1]["content"] if messages else ""

        # 简单的模拟逻辑
        if "检查结果" in user_content or "exam result" in user_content.lower():
            return """The available information is sufficient to make a diagnosis.

Diagnosis: 根据患者的症状、病史和检查结果,初步诊断为相关疾病。
Reason: 综合患者的临床表现、既往病史以及检查结果,符合该诊断的典型特征。建议进一步观察和治疗。

(注意: 这是模拟响应,实际使用时请加载真实模型)"""
        else:
            return """Current diagnosis: 基于现有信息的初步诊断假设
Based on the patient's initial presentation, the following investigation(s) should be performed: 血常规检查 (Complete Blood Count)
Reason: 需要评估患者的基础血液状况,排除感染、贫血等可能性。

(注意: 这是模拟响应,实际使用时请加载真实模型)"""


# ============================================================================
# 全局变量
# ============================================================================
# 初始化模型 (可以通过环境变量或界面选择模型)
MODEL_NAME = os.getenv("DIAGAGENT_MODEL", "Henrychur/DiagAgent-14B")
diagagent = None  # 延迟加载


# ============================================================================
# 核心功能函数
# ============================================================================
def initialize_model(model_choice: str, use_mock: bool = False) -> str:
    """初始化或切换模型"""
    global diagagent

    if use_mock:
        diagagent = TransformersLocalDiagAgent(model_choice)
        diagagent.model = None  # 强制使用模拟模式
        return "✅ 使用模拟模式 (未加载真实模型)"

    try:
        diagagent = TransformersLocalDiagAgent(model_choice)
        if diagagent.model is not None:
            return f"✅ 模型加载成功: {model_choice}"
        else:
            return "⚠️ 模型加载失败,已切换到模拟模式"
    except Exception as e:
        return f"❌ 模型初始化失败: {str(e)}"


def format_patient_info(age: str, gender: str, chief_complaint: str,
                       history_present: str, past_medical: str,
                       family_history: str, allergy_history: str,
                       personal_history: str) -> str:
    """
    格式化患者信息为标准输入格式
    """
    info_parts = []

    if age and gender:
        info_parts.append(f"- Patient Information: {age} y/o {gender}")

    if chief_complaint:
        info_parts.append(f"- Chief Complaint: {chief_complaint}")

    if history_present:
        info_parts.append(f"- History of Present Illness: {history_present}")

    if past_medical:
        info_parts.append(f"- Past Medical History: {past_medical}")

    if personal_history:
        info_parts.append(f"- Personal History: {personal_history}")

    if family_history:
        info_parts.append(f"- Family History: {family_history}")

    if allergy_history:
        info_parts.append(f"- Allergy History: {allergy_history}")

    return "\n".join(info_parts)


def start_diagnosis(age: str, gender: str, chief_complaint: str,
                   history_present: str, past_medical: str,
                   family_history: str, allergy_history: str,
                   personal_history: str) -> Tuple[List, str, List]:
    """
    开始新的诊断会话

    Returns:
        (chat_history, diagnosis_result, message_history)
    """
    global diagagent

    # 确保模型已初始化
    if diagagent is None:
        diagagent = TransformersLocalDiagAgent(MODEL_NAME)

    # 格式化患者信息
    patient_info = format_patient_info(
        age, gender, chief_complaint, history_present,
        past_medical, family_history, allergy_history, personal_history
    )

    if not patient_info.strip():
        return [], "❌ 请至少填写患者基本信息、主诉或现病史", []

    # 初始化消息历史
    messages = [
        {"role": "system", "content": DIAGNOSE_INSTRUCTION},
        {"role": "user", "content": patient_info}
    ]

    # 调用模型
    try:
        response = diagagent.diagnose(messages)
        messages.append({"role": "assistant", "content": response})

        # 构建聊天历史显示 (使用新版 Gradio 的消息格式)
        chat_history = [
            {"role": "user", "content": "**[患者信息]**\n" + patient_info},
            {"role": "assistant", "content": response}
        ]

        return chat_history, f"✅ 诊断已开始\n\n{response}", messages

    except Exception as e:
        return [], f"❌ 诊断失败: {str(e)}", []


def continue_diagnosis(exam_result: str, chat_history: List,
                      message_history: List) -> Tuple[List, str, List]:
    """
    继续诊断 - 提供检查结果后继续

    Args:
        exam_result: 检查结果
        chat_history: Gradio 聊天历史
        message_history: 模型消息历史

    Returns:
        (updated_chat_history, diagnosis_result, updated_message_history)
    """
    global diagagent

    if not message_history:
        return chat_history, "❌ 请先开始诊断", message_history

    if not exam_result.strip():
        return chat_history, "❌ 请输入检查结果", message_history

    # 添加用户提供的检查结果
    user_message = f"检查结果:\n{exam_result}"
    message_history.append({"role": "user", "content": user_message})

    # 调用模型
    try:
        response = diagagent.diagnose(message_history)
        message_history.append({"role": "assistant", "content": response})

        # 更新聊天历史 (使用新版 Gradio 的消息格式)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})

        return chat_history, f"✅ 诊断已更新\n\n{response}", message_history

    except Exception as e:
        return chat_history, f"❌ 诊断失败: {str(e)}", message_history


def reset_session() -> Tuple[List, str, List, str, str, str, str, str, str, str, str]:
    """重置会话"""
    return (
        [],  # chat_history
        "",  # diagnosis_result
        [],  # message_history
        "",  # age
        "M",  # gender
        "",  # chief_complaint
        "",  # history_present
        "",  # past_medical
        "",  # family_history
        "",  # allergy_history
        ""   # personal_history
    )


# ============================================================================
# Gradio 界面构建
# ============================================================================
def build_interface():
    """构建 Gradio 界面"""

    with gr.Blocks(title="DiagAgent - 智能诊断助手") as demo:

        # 标题栏
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🤖 DiagAgent - RL训练的智能诊断助手</h1>
            <p>基于强化学习训练的多轮交互式医疗诊断AI系统</p>
        </div>
        """)

        # 状态变量
        message_history = gr.State([])  # 存储消息历史

        with gr.Row():
            # 左侧: 输入区域
            with gr.Column(scale=1):
                gr.Markdown("## 📋 患者信息输入")

                with gr.Group():
                    gr.Markdown("### 基本信息")
                    with gr.Row():
                        age_input = gr.Textbox(
                            label="年龄",
                            placeholder="例如: 45",
                            scale=1
                        )
                        gender_input = gr.Radio(
                            choices=["M", "F"],
                            label="性别",
                            value="M",
                            scale=1
                        )

                with gr.Group():
                    gr.Markdown("### 临床信息")
                    chief_complaint = gr.Textbox(
                        label="主诉 (Chief Complaint)",
                        placeholder="例如: 右下腹疼痛",
                        lines=2
                    )

                    history_present = gr.Textbox(
                        label="现病史 (History of Present Illness)",
                        placeholder="详细描述患者当前症状的发展过程...",
                        lines=5
                    )

                    past_medical = gr.Textbox(
                        label="既往史 (Past Medical History)",
                        placeholder="既往疾病、手术史等...",
                        lines=3
                    )

                with gr.Accordion("更多信息 (可选)", open=False):
                    personal_history = gr.Textbox(
                        label="个人史 (Personal History)",
                        placeholder="吸烟、饮酒、职业暴露等...",
                        lines=2
                    )

                    family_history = gr.Textbox(
                        label="家族史 (Family History)",
                        placeholder="家族遗传病史...",
                        lines=2
                    )

                    allergy_history = gr.Textbox(
                        label="过敏史 (Allergy History)",
                        placeholder="药物过敏、食物过敏等...",
                        lines=2
                    )

                with gr.Row():
                    start_btn = gr.Button("🚀 开始诊断", variant="primary", scale=2)
                    reset_btn = gr.Button("🔄 重置", scale=1)

                gr.Markdown("---")

                # 继续诊断区域
                gr.Markdown("## 🔬 继续诊断")
                exam_result_input = gr.Textbox(
                    label="检查结果",
                    placeholder="输入AI建议的检查项目的结果...\n例如: 血常规: WBC 12.5, RBC 4.2...",
                    lines=4
                )
                continue_btn = gr.Button("➡️ 提交检查结果并继续", variant="secondary")

            # 右侧: 输出区域
            with gr.Column(scale=1):
                gr.Markdown("## 💬 诊断过程")

                chat_display = gr.Chatbot(
                    label="对话历史",
                    height=400
                )

                diagnosis_output = gr.Markdown(
                    label="当前诊断结果",
                    value="等待开始诊断..."
                )

                with gr.Accordion("📊 系统信息", open=False):
                    gr.Markdown(f"""
                    <div style="background-color: #f0f4f8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <b>模型信息:</b> {MODEL_NAME}<br>
                    <b>工作模式:</b> 多轮交互式诊断<br>
                    <b>功能:</b>
                    <ul>
                        <li>✅ 智能推荐检查项目</li>
                        <li>✅ 动态更新诊断假设</li>
                        <li>✅ 自主决定诊断时机</li>
                    </ul>
                    <b>注意:</b> 首次运行会自动下载模型(约28GB),请确保网络畅通和磁盘空间充足。<br>
                    如果没有GPU或不想下载模型,系统会自动使用模拟模式。
                    </div>
                    """)

        # 按钮事件绑定
        start_btn.click(
            fn=start_diagnosis,
            inputs=[
                age_input, gender_input, chief_complaint,
                history_present, past_medical,
                family_history, allergy_history, personal_history
            ],
            outputs=[chat_display, diagnosis_output, message_history]
        )

        continue_btn.click(
            fn=continue_diagnosis,
            inputs=[exam_result_input, chat_display, message_history],
            outputs=[chat_display, diagnosis_output, message_history]
        ).then(
            fn=lambda: "",  # 清空检查结果输入框
            outputs=[exam_result_input]
        )

        reset_btn.click(
            fn=reset_session,
            outputs=[
                chat_display, diagnosis_output, message_history,
                age_input, gender_input, chief_complaint,
                history_present, past_medical, family_history,
                allergy_history, personal_history
            ]
        )

        # 示例
        gr.Examples(
            examples=[
                [
                    "65", "F", "腹痛、体重下降",
                    "患者报告1个月来体重下降10磅,早饱感,疲劳和缺乏精力。她描述胃部有一种\"空\"的感觉,与恶心或疼痛不同。",
                    "哮喘、高脂血症、高血压、骨关节炎、风湿性多肌痛、冠心病(NSTEMI伴LAD夹层)、骨质疏松症",
                    "父亲有冠心病;母亲有哮喘。无早期心肌梗死、心律失常、心肌病或心源性猝死家族史。",
                    "赖诺普利",
                    "未提供"
                ],
                [
                    "28", "F", "右下腹疼痛",
                    "患者G3P2,末次月经___,有子宫内膜异位症病史,因右下腹疼痛到急诊就诊。患者报告疼痛始于昨天下午晚些时候,最初为右下腹隐痛,后转为锐痛,未太困扰她,能继续活动,后自行缓解。今晨疼痛复发,频率和强度较昨天增加,锐痛/刺痛,间歇性,持续5-20分钟。",
                    "抑郁、焦虑;子宫内膜异位症腹腔镜手术x2",
                    "无明显异常",
                    "青霉素、阿莫西林、乳胶",
                    "过去___年未有性生活"
                ]
            ],
            inputs=[
                age_input, gender_input, chief_complaint,
                history_present, past_medical, family_history,
                allergy_history, personal_history
            ],
            label="📝 示例病例 (点击加载)"
        )

        # 页脚信息
        gr.Markdown("""
        ---
        ### 📚 使用说明
        1. **填写患者信息**: 至少填写基本信息、主诉和现病史
        2. **开始诊断**: 点击"开始诊断"按钮,AI会分析并给出初步诊断或建议检查
        3. **提供检查结果**: 如果AI建议做检查,在"检查结果"框输入结果并点击"提交"
        4. **重复步骤3**: 直到AI给出最终诊断

        ### ⚠️ 免责声明
        本系统仅用于研究和教育目的,不应替代专业医疗建议、诊断或治疗。任何医疗决策都应咨询合格的医疗专业人员。

        ### 📖 参考资料
        - 论文: [Evolving Diagnostic Agents in a Virtual Clinical Environment](http://arxiv.org/abs/2510.24654)
        - 模型: [HuggingFace - DiagAgent](https://huggingface.co/Henrychur/DiagAgent-14B)
        - GitHub: [DiagGym](https://github.com/...)
        """)

    return demo


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiagAgent Gradio Web Interface")
    parser.add_argument(
        "--model",
        type=str,
        default="/input0/DiagAgent-14B",
        help="模型名称或路径 (默认: Henrychur/DiagAgent-14B)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟模式(不加载真实模型)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公开分享链接"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="服务端口 (默认: 8080)"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="服务器地址 (默认: 0.0.0.0)"
    )

    args = parser.parse_args()

    # 更新全局模型名称
    MODEL_NAME = args.model

    # 预初始化模型
    print("="*60)
    print("DiagAgent Gradio Web Interface")
    print("="*60)

    if args.mock:
        print("⚠️  使用模拟模式运行 (未加载真实模型)")
        diagagent = TransformersLocalDiagAgent(MODEL_NAME)
        diagagent.model = None
    else:
        print(f"正在初始化模型: {MODEL_NAME}")
        print("首次运行可能需要下载模型,请耐心等待...")
        diagagent = TransformersLocalDiagAgent(MODEL_NAME)

    print("="*60)

    # 构建并启动界面
    demo = build_interface()

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        show_error=True
    )
