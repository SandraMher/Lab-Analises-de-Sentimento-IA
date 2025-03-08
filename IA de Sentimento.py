# Analise de sentimento em texto

from transformers import pipeline

# Criar um pipeline de análise de sentimentos
analise = pipeline("sentiment-analysis")

# Testar com um texto
texto = "Eu adorei esse produto! É maravilhoso!"
resultado = analise(texto)

# Exibir o resultado
print(resultado)
[{'label': 'POSITIVE', 'score': 0.9998}]
textos = ["Esse filme foi incrível!", "O atendimento foi péssimo."]
resultado = analise(textos)
print(resultado)
analise = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Capturar audio e converter em texto

import speech_recognition as sr

# Inicializar o reconhecedor de fala
recognizer = sr.Recognizer()

# Usar o microfone do computador para captar o áudio
with sr.Microphone() as source:
    print("Por favor, fale algo...")
    audio = recognizer.listen(source)

# Converter áudio em texto
texto = recognizer.recognize_google(audio, language="pt-BR")
print(f"Texto reconhecido: {texto}")
from transformers import pipeline

# Criar o pipeline de análise de sentimentos
analise = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Analisar o texto transcrito
resultado = analise(texto)

# Exibir o resultado
print(f"Sentimento: {resultado}")
