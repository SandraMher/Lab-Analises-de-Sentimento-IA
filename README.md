from transformers import pipeline

analise = pipeline("sentiment-analysis")
resultado = analise("Este produto é ótimo!")
print(resultado)  # Retorna sentimento e confiança
saida esperada
[{'label': 'POSITIVE', 'score': 0.9998}]

 Converter Fala em Texto (Speech-to-Text)
Usando o SpeechRecognition para capturar o áudio e transcrever para texto
Fluxo Completo
Você fala no microfone.
O áudio é convertido em texto usando a transcrição (speech-to-text).
O texto é passado para o modelo de análise de sentimentos.
