"""
DiagAgent Gradio Web Interface (English Version)
Automatically generated runnable frontend interface based on the 🤖DiagAgent section in README

Features:
- Multi-turn interactive diagnosis
- Input patient information (age, gender, chief complaint, medical history, etc.)
- Intelligent examination recommendations or final diagnosis
- View complete conversation history
"""

import gradio as gr
from typing import List, Dict, Tuple
import os

# ============================================================================
# System Instruction (from README example)
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
# DiagAgent Model Class (based on README example code)
# ============================================================================
class TransformersLocalDiagAgent:
    """
    Local DiagAgent model based on Transformers library
    Note: Model needs to be downloaded locally or loaded from HuggingFace
    """
    def __init__(self, model_name_or_path: str = "Henrychur/DiagAgent-14B",
                 max_tokens: int = 8192, temperature: float = 0.0) -> None:
        """
        Initialize DiagAgent model

        Args:
            model_name_or_path: Model path or HuggingFace model name
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy decoding)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading model: {model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                # Uncomment the line below if flash_attention_2 is not installed
                # attn_implementation="flash_attention_2"
            )
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.device = self.model.device
            print(f"Model loaded successfully! Device: {self.device}")

        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Running in mock mode...")
            self.model = None
            self.tokenizer = None

    def diagnose(self, messages: List[Dict[str, str]]) -> str:
        """
        Execute multi-turn diagnosis

        Args:
            messages: List of conversation messages in format [{"role": "system/user/assistant", "content": "..."}]

        Returns:
            Generated diagnostic response
        """
        # If model is not loaded, return mock response
        if self.model is None or self.tokenizer is None:
            return self._mock_diagnose(messages)

        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Remove prompt tokens
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip().replace("```", "")

        except Exception as e:
            return f"Diagnosis error: {str(e)}"

    def _mock_diagnose(self, messages: List[Dict[str, str]]) -> str:
        """
        Mock diagnosis response (used when model is not loaded)
        """
        user_content = messages[-1]["content"] if messages else ""

        # Simple mock logic
        if "exam result" in user_content.lower() or "examination result" in user_content.lower():
            return """The available information is sufficient to make a diagnosis.

Diagnosis: Based on the patient's symptoms, medical history, and examination results, the preliminary diagnosis indicates the relevant condition.
Reason: Comprehensive analysis of the patient's clinical presentation, past medical history, and examination findings are consistent with typical features of this diagnosis. Further observation and treatment are recommended.

(Note: This is a mock response. Please load the actual model for real use.)"""
        else:
            return """Current diagnosis: Preliminary diagnostic hypothesis based on available information
Based on the patient's initial presentation, the following investigation(s) should be performed: Complete Blood Count (CBC)
Reason: Need to assess the patient's baseline hematological status and rule out infection, anemia, and other possibilities.

(Note: This is a mock response. Please load the actual model for real use.)"""


# ============================================================================
# Global Variables
# ============================================================================
# Initialize model (can be set via environment variable or interface)
MODEL_NAME = os.getenv("DIAGAGENT_MODEL", "Henrychur/DiagAgent-14B")
diagagent = None  # Lazy loading


# ============================================================================
# Core Functions
# ============================================================================
def initialize_model(model_choice: str, use_mock: bool = False) -> str:
    """Initialize or switch model"""
    global diagagent

    if use_mock:
        diagagent = TransformersLocalDiagAgent(model_choice)
        diagagent.model = None  # Force mock mode
        return "✅ Using mock mode (real model not loaded)"

    try:
        diagagent = TransformersLocalDiagAgent(model_choice)
        if diagagent.model is not None:
            return f"✅ Model loaded successfully: {model_choice}"
        else:
            return "⚠️ Model loading failed, switched to mock mode"
    except Exception as e:
        return f"❌ Model initialization failed: {str(e)}"


def format_patient_info(age: str, gender: str, chief_complaint: str,
                       history_present: str, past_medical: str,
                       family_history: str, allergy_history: str,
                       personal_history: str) -> str:
    """
    Format patient information into standard input format
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
    Start a new diagnosis session

    Returns:
        (chat_history, diagnosis_result, message_history)
    """
    global diagagent

    # Ensure model is initialized
    if diagagent is None:
        diagagent = TransformersLocalDiagAgent(MODEL_NAME)

    # Format patient information
    patient_info = format_patient_info(
        age, gender, chief_complaint, history_present,
        past_medical, family_history, allergy_history, personal_history
    )

    if not patient_info.strip():
        return [], "❌ Please fill in at least basic patient information, chief complaint, or history of present illness", []

    # Initialize message history
    messages = [
        {"role": "system", "content": DIAGNOSE_INSTRUCTION},
        {"role": "user", "content": patient_info}
    ]

    # Call model
    try:
        response = diagagent.diagnose(messages)
        messages.append({"role": "assistant", "content": response})

        # Build chat history display (using new Gradio message format)
        chat_history = [
            {"role": "user", "content": "**[Patient Information]**\n" + patient_info},
            {"role": "assistant", "content": response}
        ]

        return chat_history, f"✅ Diagnosis started\n\n{response}", messages

    except Exception as e:
        return [], f"❌ Diagnosis failed: {str(e)}", []


def continue_diagnosis(exam_result: str, chat_history: List,
                      message_history: List) -> Tuple[List, str, List]:
    """
    Continue diagnosis - proceed after providing examination results

    Args:
        exam_result: Examination results
        chat_history: Gradio chat history
        message_history: Model message history

    Returns:
        (updated_chat_history, diagnosis_result, updated_message_history)
    """
    global diagagent

    if not message_history:
        return chat_history, "❌ Please start diagnosis first", message_history

    if not exam_result.strip():
        return chat_history, "❌ Please enter examination results", message_history

    # Add user-provided examination results
    user_message = f"Examination Results:\n{exam_result}"
    message_history.append({"role": "user", "content": user_message})

    # Call model
    try:
        response = diagagent.diagnose(message_history)
        message_history.append({"role": "assistant", "content": response})

        # Update chat history (using new Gradio message format)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})

        return chat_history, f"✅ Diagnosis updated\n\n{response}", message_history

    except Exception as e:
        return chat_history, f"❌ Diagnosis failed: {str(e)}", message_history


def reset_session() -> Tuple[List, str, List, str, str, str, str, str, str, str, str]:
    """Reset session"""
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
# Gradio Interface Builder
# ============================================================================
def build_interface():
    """Build Gradio interface"""

    with gr.Blocks(title="DiagAgent - Intelligent Diagnostic Assistant") as demo:

        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🤖 DiagAgent - RL-Trained Intelligent Diagnostic Assistant</h1>
            <p>Reinforcement Learning-Based Multi-Turn Interactive Medical Diagnosis AI System</p>
        </div>
        """)

        # State variables
        message_history = gr.State([])  # Store message history

        with gr.Row():
            # Left side: Input area
            with gr.Column(scale=1):
                gr.Markdown("## 📋 Patient Information Input")

                with gr.Group():
                    gr.Markdown("### Basic Information")
                    with gr.Row():
                        age_input = gr.Textbox(
                            label="Age",
                            placeholder="e.g., 45",
                            scale=1
                        )
                        gender_input = gr.Radio(
                            choices=["M", "F"],
                            label="Gender",
                            value="M",
                            scale=1
                        )

                with gr.Group():
                    gr.Markdown("### Clinical Information")
                    chief_complaint = gr.Textbox(
                        label="Chief Complaint",
                        placeholder="e.g., Right lower quadrant pain",
                        lines=2
                    )

                    history_present = gr.Textbox(
                        label="History of Present Illness",
                        placeholder="Detailed description of the patient's current symptom development...",
                        lines=5
                    )

                    past_medical = gr.Textbox(
                        label="Past Medical History",
                        placeholder="Previous illnesses, surgical history, etc...",
                        lines=3
                    )

                with gr.Accordion("Additional Information (Optional)", open=False):
                    personal_history = gr.Textbox(
                        label="Personal History",
                        placeholder="Smoking, alcohol, occupational exposure, etc...",
                        lines=2
                    )

                    family_history = gr.Textbox(
                        label="Family History",
                        placeholder="Family hereditary disease history...",
                        lines=2
                    )

                    allergy_history = gr.Textbox(
                        label="Allergy History",
                        placeholder="Drug allergies, food allergies, etc...",
                        lines=2
                    )

                with gr.Row():
                    start_btn = gr.Button("🚀 Start Diagnosis", variant="primary", scale=2)
                    reset_btn = gr.Button("🔄 Reset", scale=1)

                gr.Markdown("---")

                # Continue diagnosis area
                gr.Markdown("## 🔬 Continue Diagnosis")
                exam_result_input = gr.Textbox(
                    label="Examination Results",
                    placeholder="Enter the results of the examination recommended by AI...\ne.g., Complete Blood Count: WBC 12.5, RBC 4.2...",
                    lines=4
                )
                continue_btn = gr.Button("➡️ Submit Examination Results and Continue", variant="secondary")

            # Right side: Output area
            with gr.Column(scale=1):
                gr.Markdown("## 💬 Diagnosis Process")

                chat_display = gr.Chatbot(
                    label="Conversation History",
                    height=400
                )

                diagnosis_output = gr.Markdown(
                    label="Current Diagnosis Results",
                    value="Waiting to start diagnosis..."
                )

                with gr.Accordion("📊 System Information", open=False):
                    gr.Markdown(f"""
                    <div style="background-color: #f0f4f8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <b>Model Information:</b> {MODEL_NAME}<br>
                    <b>Working Mode:</b> Multi-turn interactive diagnosis<br>
                    <b>Features:</b>
                    <ul>
                        <li>✅ Intelligent examination recommendations</li>
                        <li>✅ Dynamic diagnosis updates</li>
                        <li>✅ Autonomous decision timing</li>
                    </ul>
                    <b>Note:</b> First run will automatically download the model (~28GB). Please ensure stable network and sufficient disk space.<br>
                    If you don't have a GPU or don't want to download the model, the system will automatically use mock mode.
                    </div>
                    """)

        # Button event bindings
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
            fn=lambda: "",  # Clear examination result input box
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

        # Examples
        gr.Examples(
            examples=[
                [
                    "65", "F", "Abdominal pain, weight loss",
                    "The patient reports a 1-month history of weight loss (10 lbs), early satiety, fatigue, and lack of energy. She describes an \"empty\" feeling in her stomach, different from nausea or pain.",
                    "Asthma, hyperlipidemia, hypertension, osteoarthritis, polymyalgia rheumatica, CAD (NSTEMI with LAD dissection), osteoporosis",
                    "Father had CAD; mother had asthma. No family history of early MI, arrhythmia, cardiomyopathies, or sudden cardiac death.",
                    "Lisinopril",
                    "Not provided"
                ],
                [
                    "28", "F", "Right lower quadrant pain",
                    "G3P2 LMP ___, hx of endometriosis, presents to the ED with RLQ pain. Pain started yesterday afternoon, initially as nagging pain in RLQ, then turned sharp. Pain returned this morning with increased frequency and intensity, sharp/stabbing, intermittent, lasting 5-20 mins.",
                    "Depression, anxiety; laparoscopic surgery for endometriosis x2",
                    "Non-contributory",
                    "Penicillin, amoxicillin, latex",
                    "Not sexually active for the past ___ years"
                ]
            ],
            inputs=[
                age_input, gender_input, chief_complaint,
                history_present, past_medical, family_history,
                allergy_history, personal_history
            ],
            label="📝 Example Cases (Click to Load)"
        )

        # Footer
        gr.Markdown("""
        ---
        ### 📚 User Guide
        1. **Fill in Patient Information**: Fill in at least basic information, chief complaint, and history of present illness
        2. **Start Diagnosis**: Click "Start Diagnosis" button, AI will analyze and provide preliminary diagnosis or recommend examinations
        3. **Provide Examination Results**: If AI recommends examinations, enter results in "Examination Results" box and click "Submit"
        4. **Repeat Step 3**: Until AI provides final diagnosis

        ### ⚠️ Disclaimer
        This system is for research and educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Any medical decisions should consult qualified healthcare professionals.

        ### 📖 References
        - Paper: [Evolving Diagnostic Agents in a Virtual Clinical Environment](http://arxiv.org/abs/2510.24654)
        - Model: [HuggingFace - DiagAgent](https://huggingface.co/Henrychur/DiagAgent-14B)
        - GitHub: [DiagGym](https://github.com/...)
        """)

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiagAgent Gradio Web Interface (English)")
    parser.add_argument(
        "--model",
        type=str,
        default="Henrychur/DiagAgent-14B",
        help="Model name or path (default: Henrychur/DiagAgent-14B)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode (do not load real model)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Service port (default: 7860)"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server address (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    # Update global model name
    MODEL_NAME = args.model

    # Pre-initialize model
    print("="*60)
    print("DiagAgent Gradio Web Interface (English)")
    print("="*60)

    if args.mock:
        print("⚠️  Running in mock mode (real model not loaded)")
        diagagent = TransformersLocalDiagAgent(MODEL_NAME)
        diagagent.model = None
    else:
        print(f"Initializing model: {MODEL_NAME}")
        print("First run may require model download, please be patient...")
        diagagent = TransformersLocalDiagAgent(MODEL_NAME)

    print("="*60)

    # Build and launch interface
    demo = build_interface()

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        show_error=True
    )
