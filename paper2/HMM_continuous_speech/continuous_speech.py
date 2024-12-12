'''

pip install SpeechRecognition
SpeechRecognition 是一个 Python 的高级库，作为接口提供对多种语音识别 API 和服务的支持。
SpeechRecognition 本身并没有独立的语音识别核心算法，它是一个高层封装库，依赖于后端语音识别服务或引擎。
后端的技术可能基于 深度学习（如 Google 和 AWS） 或 隐马尔可夫模型（如 CMU Sphinx） 等。它的优势在于提供了统一接口，方便开发者快速集成多种语音识别服务。

pip install pocketsphinx
基于隐马尔可夫模型(HMM)的连续音频流识别器,可将语音输入转换为文本输出。
PocketSphinx 是一个由 CMU（卡内基梅隆大学）开发的开源语音识别引擎的轻量级版本。它是 CMU Sphinx 项目的一部分。

PocketSphinx 是一个完整的离线语音识别引擎，可以直接用于设备上进行语音预测，但能力有限。
SpeechRecognition 是一个接口库，更多的是提供对其他语音识别引擎（包括 PocketSphinx）的支持，适合快速集成多种语音识别服务。

PortAudio 是一个跨平台音频 I/O 库，PyAudio 依赖于它。在 macOS 上可以通过 Homebrew 安装
brew install portaudio
PyAudio 是 SpeechRecognition 使用的一个依赖库，用于从麦克风捕获音频。
pip install pyaudio

要现在本地安装 PortAudio，再通过 pip 安装py对应的依赖包 PyAudio，本地没有会失败
'''

import speech_recognition as sr


def speech_recognition_EN():
    # 从麦克风获取音频
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # 收听 1 秒以校准环境噪声水平的能量阈值
        r.adjust_for_ambient_noise(source)
        print("请说话...")
        audio = r.listen(source)

        # 使用 sphinx 进行语音识别
        try:
            print("Sphinx 识别结果为：" + r.recognize_sphinx(audio))
        except sr.UnknownValueError:
            print("识别失败")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))


def speech_recognition_CN():
    # 从麦克风获取音频
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # 收听 1 秒以校准环境噪声水平的能量阈值
        r.adjust_for_ambient_noise(source)
        print("请说话...")
        audio = r.listen(source)

        # 使用 sphinx 进行语音识别
        try:
            print("Sphinx 识别结果为：" + r.recognize_sphinx(audio, language='zh-CN'))
        except sr.UnknownValueError:
            print("识别失败")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))


if __name__ == '__main__':
    n = input("英文识别输入E，中文识别输入C:")
    if n == "E" or n == "e":
        speech_recognition_EN()
    elif n == "C" or n == "c":
        speech_recognition_CN()
    else:
        print("输入正确的语言字母！")