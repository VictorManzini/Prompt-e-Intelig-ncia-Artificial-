# ============================================================
# EXERCÍCIO 3 — Análise de Sentimento com suas próprias frases
# ============================================================
# Aqui vamos usar IA para detectar se uma frase é
# POSITIVA (feliz/boa) ou NEGATIVA (triste/ruim).

# Importar o pipeline da Hugging Face
# "Importar" = trazer uma ferramenta para poder usar
from transformers import pipeline

# Criar o pipeline de análise de sentimento
# Isso baixa um modelo de IA treinado (~260MB, pode demorar 1-2 min na primeira vez)
print("⏳ Carregando o modelo de IA... (pode demorar na primeira vez)")
analisador = pipeline("sentiment-analysis")
print("✅ Modelo carregado!\n")

# ✍️ TROQUE as frases abaixo por frases SUAS!
# Dica: use frases em inglês para melhor resultado
# (o modelo padrão foi treinado em inglês)
minhas_frases = [
    "I love studying at FIAP, the classes are amazing!",      # ✍️ Troque!
    "This traffic in São Paulo is terrible.",                   # ✍️ Troque!
    "Python is a very interesting programming language.",       # ✍️ Troque!
    "I am worried about my grades this semester.",              # ✍️ Troque!
    "The pizza I had yesterday was the best ever!",             # ✍️ Troque!
]

# Analisar cada frase
print("🔍 RESULTADO DA ANÁLISE DE SENTIMENTO\n")
print(f"{'Frase':<55} {'Sentimento':<12} {'Certeza'}")
print("─" * 80)

for frase in minhas_frases:
    # O pipeline retorna uma lista com um dicionário dentro
    # Por isso usamos [0] para pegar o primeiro (e único) resultado
    resultado = analisador(frase)[0]

    # resultado tem dois campos:
    # - 'label': POSITIVE ou NEGATIVE
    # - 'score': de 0 a 1 (quanto mais perto de 1, mais certeza)

    label = resultado['label']       # POSITIVE ou NEGATIVE
    score = resultado['score']       # número entre 0 e 1
    emoji = "😊" if label == "POSITIVE" else "😟"

    # Mostrar resultado formatado
    print(f"  {emoji} {frase[:53]:<53} {label:<12} {score:.1%}")

print("\n💡 Observou algo interessante?")
print("   • Frases com palavras como 'love', 'best', 'amazing' tendem a ser POSITIVE")
print("   • Frases com 'terrible', 'worried', 'bad' tendem a ser NEGATIVE")
print("   • A IA acerta na maioria, mas pode errar em frases com sarcasmo ou ironia!")
