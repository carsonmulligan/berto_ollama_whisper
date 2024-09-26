import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sounddevice as sd
import numpy as np
import whisper
import requests
import asyncio
import edge_tts
import simpleaudio as sa
import os
import json
from colorama import init, Fore
from pydub import AudioSegment
from scipy.io.wavfile import write
import sys

# Initialize colorama for colored text in terminal
init(autoreset=True)

# Load the Whisper model
print(Fore.CYAN + "加载 Whisper 模型...")
model = whisper.load_model('medium')  # Use 'large' for even better accuracy

# Set the desired model (replace with your model)
selected_model = 'llama2-uncensored:latest'  # Replace with the exact model name from Ollama

# Conversation history
conversation = []

def record_audio(fs=16000):
    """Record audio from the microphone until silence is detected."""
    print(Fore.GREEN + "Guan Guan 正在聆听...（按 Ctrl+C 停止）")
    try:
        duration = 0.5  # Chunk duration in seconds
        silence_threshold = 0.01  # Silence threshold
        max_silence_duration = 1.5  # Max duration of silence before stopping

        recording = []
        silence_duration = 0

        while True:
            audio_chunk = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            rms = np.sqrt(np.mean(audio_chunk**2))
            recording.append(audio_chunk)
            if rms < silence_threshold:
                silence_duration += duration
            else:
                silence_duration = 0
            if silence_duration > max_silence_duration:
                break
        full_recording = np.concatenate(recording, axis=0)
        return np.squeeze(full_recording)
    except Exception as e:
        print(Fore.RED + f"录音错误：{e}")
        return None

def transcribe_audio(audio_data):
    """Transcribe the recorded audio using Whisper."""
    print(Fore.CYAN + "正在转录...")
    try:
        # Save audio_data to a temporary WAV file
        write("temp.wav", 16000, audio_data)
        # Transcribe using Whisper
        options = dict(language='zh', fp16=False)  # Disable FP16
        result = model.transcribe("temp.wav", **options)
        text = result['text'].strip()
        print(Fore.YELLOW + f"你: {text}")
        os.remove("temp.wav")  # Clean up the temp file
        return text
    except Exception as e:
        print(Fore.RED + f"转录错误：{e}")
        return ""

def get_ai_response(user_input, selected_model):
    """Get a response from Guan Guan using Ollama."""
    # Check for 'guan guan stop' command
    if '关关停止' in user_input.lower():
        print(Fore.CYAN + "关关已停止对话，正在等待...")
        conversation.clear()
        return ""

    # Append the latest interaction
    conversation.append(f"用户: {user_input}")

    # Limit the conversation history
    recent_conversation = conversation[-4:]  # Adjust as needed

    # Exclude previous refusals
    filtered_conversation = [
        line for line in recent_conversation
        if not any(
            phrase in line.lower()
            for phrase in ["抱歉", "无法协助", "无法生成", "无法互动"]
        )
    ]

    conversation_text = "\n".join(filtered_conversation)
    prompt = f"{system_prompt}\n{conversation_text}\n关关:"

    # print(Fore.BLUE + f"Prompt sent to the model:\n{prompt}\n")

    url = 'http://localhost:11434/api/generate'
    data = {
        'model': selected_model,  # Use the selected model
        'prompt': prompt,
        'temperature': 0.5,
        'max_tokens': 200,
        'stop': ['用户:']
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        response_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_data = json.loads(line)
                if 'response' in json_data:
                    response_text += json_data['response']
        response_text = response_text.strip()

        # Check for refusal
        if "抱歉" in response_text.lower() and "无法" in response_text.lower():
            print(Fore.RED + "模型生成了拒绝回答的回复。")
            # Optionally, modify the prompt or notify the user
        else:
            conversation.append(f"关关: {response_text}")

        # print(Fore.BLUE + f"Model response:\n{response_text}\n")
        print(Fore.CYAN + f"关关: {response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"与 Ollama 通信时出错：{e}")
        return "抱歉，我目前无法回答。"
    except json.JSONDecodeError as e:
        print(Fore.RED + f"解析 JSON 响应时出错：{e}")
        return "抱歉，我目前无法回答。"

async def save_audio(text, filename, voice):
    """Convert text to speech using Edge TTS."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(filename)
    print(Fore.GREEN + f"成功保存音频到 {filename}")

def speak_response(response_text, voice='zh-CN-XiaoxiaoNeural'):
    """Play the spoken response."""
    if not response_text:
        return
    try:
        filename_mp3 = "response.mp3"
        filename_wav = "response.wav"
        asyncio.run(save_audio(response_text, filename_mp3, voice))

        if os.path.exists(filename_mp3) and os.path.getsize(filename_mp3) > 0:
            sound = AudioSegment.from_mp3(filename_mp3)
            sound.export(filename_wav, format="wav")

            if os.path.exists(filename_wav) and os.path.getsize(filename_wav) > 0:
                wave_obj = sa.WaveObject.from_wave_file(filename_wav)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                os.remove(filename_mp3)
                os.remove(filename_wav)
            else:
                print(Fore.RED + "错误：WAV 文件为空或不存在")
        else:
            print(Fore.RED + "错误：MP3 文件为空或不存在")
    except Exception as e:
        print(Fore.RED + f"播放音频时出错：{e}")

def generate_question(topic, selected_model):
    """Generate a question about the given topic using Ollama."""
    system_prompt = f"""
你是一个助理，会用中文生成关于{topic}的有趣问题。
请用中文提供一个关于{topic}的简短而清晰的问题。
"""
    data = {
        'model': selected_model,  # Use the selected model
        'prompt': system_prompt,
        'temperature': 0.7,
        'max_tokens': 50,
        'stop': ['\n']
    }
    headers = {'Content-Type': 'application/json'}
    url = 'http://localhost:11434/api/generate'

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        response_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_data = json.loads(line)
                if 'response' in json_data:
                    response_text += json_data['response']
        question = response_text.strip()
        return question
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"使用 Ollama 生成问题时出错：{e}")
        return None
    except json.JSONDecodeError as e:
        print(Fore.RED + f"解析 JSON 响应时出错：{e}")
        return None

def generate_follow_up_question(previous_response, selected_model, topic=None):
    """Generate a follow-up question about the given response using Ollama."""
    if topic:
        system_prompt = f"""
你是一个助理，会根据之前关于{topic}的回答生成有趣的后续问题。
请用中文针对以下回答提供一个简短而清晰的后续问题：“{previous_response}”
"""
    else:
        system_prompt = f"""
你是一个助理，会根据之前的回答生成有趣的后续问题。
请用中文针对以下回答提供一个简短而清晰的后续问题：“{previous_response}”
"""

    data = {
        'model': selected_model,  # Use the selected model
        'prompt': system_prompt,
        'temperature': 0.7,
        'max_tokens': 50,
        'stop': ['\n']
    }
    headers = {'Content-Type': 'application/json'}
    url = 'http://localhost:11434/api/generate'

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        response_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_data = json.loads(line)
                if 'response' in json_data:
                    response_text += json_data['response']
        follow_up_question = response_text.strip()
        return follow_up_question
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"使用 Ollama 生成后续问题时出错：{e}")
        return None
    except json.JSONDecodeError as e:
        print(Fore.RED + f"解析 JSON 响应时出错：{e}")
        return None

def present_cli_options(selected_model):
    """Present CLI options to the user."""
    options = {
        '1': "用语音回复关关。",
        '2': "输入文字回复关关。",
        '3': "针对关关的上一个回答提出一个后续问题。",
        '4': "听一个关于科学的问题。",
        '5': "听一个关于中国历史或政治的问题。",
        '6': "从三个科学问题中选择一个。",
        '7': "从三个中国历史或政治问题中选择一个。"
    }

    print(Fore.MAGENTA + "\n请选择一个选项：")
    for key, value in options.items():
        print(f"{key}. {value}")

    choice = input(Fore.MAGENTA + "输入你的选择编号：").strip()

    if choice not in options:
        print(Fore.RED + "无效的选项。请重试。")
        return None

    if choice == '1':
        # Option 1: Let the user respond by speaking
        print(Fore.GREEN + "请用语音回复关关。")
        audio_data = record_audio()
        if audio_data is None or len(audio_data) == 0:
            return None
        user_text = transcribe_audio(audio_data)
        return user_text

    elif choice == '2':
        # Option 2: Write a response directly for Guan Guan
        user_input = input(Fore.GREEN + "请输入你要回复关关的文字：").strip()
        if user_input:
            print(Fore.YELLOW + f"你（文字）：{user_input}")
            return user_input
        else:
            print(Fore.RED + "你没有输入任何回复。")
            return None

    elif choice == '3':
        # Option 3: Ask a follow-up question about Guan Guan's previous response
        if conversation:
            last_response = conversation[-1]
            print(Fore.GREEN + f"根据关关的上一个回答生成一个后续问题：{last_response}")
            follow_up_question = generate_follow_up_question(last_response, selected_model)
            if follow_up_question:
                print(Fore.YELLOW + f"你（后续问题）：{follow_up_question}")
                speak_response(follow_up_question, voice='zh-CN-XiaoxiaoNeural')  # Different voice for the user
                return follow_up_question
            else:
                print(Fore.RED + "无法生成后续问题。")
                return None
        else:
            print(Fore.RED + "没有关关的上一个回答可供参考。")
            return None

    elif choice == '4' or choice == '5':
        # Option 4 or 5: Generate a question about science or Chinese history/politics
        if choice == '4':
            topic = '科学'
        elif choice == '5':
            topic = '中国历史或政治'

        generated_question = generate_question(topic, selected_model)
        if generated_question:
            print(Fore.YELLOW + f"你（选项 {choice}）：{generated_question}")
            speak_response(generated_question, voice='zh-CN-XiaoxiaoNeural')  # Different voice for the user
            return generated_question
        else:
            print(Fore.RED + "无法生成问题。")
            return None

    elif choice == '6':
        # Option 6: Generate three questions about science for the user to choose
        topic = '科学'
        questions = []
        print(Fore.GREEN + "正在生成三个关于科学的问题...")
        for _ in range(3):
            question = generate_question(topic, selected_model)
            if question:
                questions.append(question)
            else:
                print(Fore.RED + "无法生成问题。")
                return None
        # Present the questions to the user
        print(Fore.YELLOW + "请选择以下问题之一：")
        for idx, question in enumerate(questions, start=1):
            print(f"{idx}. {question}")
        selection = input(Fore.MAGENTA + "输入你的选择编号：").strip()
        if selection in ['1', '2', '3']:
            selected_question = questions[int(selection)-1]
            print(Fore.YELLOW + f"你（选项 {selection}）：{selected_question}")
            speak_response(selected_question, voice='zh-CN-XiaoxiaoNeural')  # Different voice for the user
            return selected_question
        else:
            print(Fore.RED + "无效的选项。请重试。")
            return None

    elif choice == '7':
        # Option 7: Generate three questions about Chinese history or politics for the user to choose
        topic = '中国历史或政治'
        questions = []
        print(Fore.GREEN + "正在生成三个关于中国历史或政治的问题...")
        for _ in range(3):
            question = generate_question(topic, selected_model)
            if question:
                questions.append(question)
            else:
                print(Fore.RED + "无法生成问题。")
                return None
        # Present the questions to the user
        print(Fore.YELLOW + "请选择以下问题之一：")
        for idx, question in enumerate(questions, start=1):
            print(f"{idx}. {question}")
        selection = input(Fore.MAGENTA + "输入你的选择编号：").strip()
        if selection in ['1', '2', '3']:
            selected_question = questions[int(selection)-1]
            print(Fore.YELLOW + f"你（选项 {selection}）：{selected_question}")
            speak_response(selected_question, voice='zh-CN-XiaoxiaoNeural')  # Different voice for the user
            return selected_question
        else:
            print(Fore.RED + "无效的选项。请重试。")
            return None

def test_ollama_connection(selected_model):
    """Test the connection to Ollama."""
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': selected_model,
        'prompt': '用中文说：“你好，我正在运行”。',
        'temperature': 0.7,
        'max_tokens': 25
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print(Fore.GREEN + f"使用模型 '{selected_model}' 测试 Ollama 成功。")
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"连接 Ollama 时出错，模型 '{selected_model}'：{e}")
        sys.exit(1)

def main():
    """Main loop for the CLI interaction."""
    print(Fore.CYAN + "正在与关关开始对话！")

    # Test Ollama connection
    test_ollama_connection(selected_model)

    # Define the system prompt globally
    global system_prompt
    system_prompt = """
你是关关，一位友好的导师，精通科学和中国历史。请提供简短而清晰的回答。
"""

    initial_prompt = (
        "你好！我是关关，我只会说中文，欢迎你！"
    )
    print(Fore.CYAN + f"关关: {initial_prompt}")
    conversation.append(f"关关: {initial_prompt}")
    speak_response(initial_prompt)
    
    while True:
        try:
            # Present CLI options
            user_text = present_cli_options(selected_model)
            if user_text:
                print(Fore.GREEN + f"你: {user_text}")
                guan_response = get_ai_response(user_text, selected_model)
                speak_response(guan_response)
            else:
                continue

        except KeyboardInterrupt:
            print(Fore.RED + "\n对话已结束。")
            break
        except Exception as e:
            print(Fore.RED + f"发生错误：{e}")
            break

if __name__ == "__main__":
    main()
