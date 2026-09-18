import pandas as pd
import matplotlib.pyplot as plt
import prince
import seaborn as sns
from scipy.stats import chi2_contingency
from pathlib import Path


# =====================================================
# CONFIGURAÇÕES INICIAIS
# =====================================================

ARQUIVO = 'dados.xlsx'
SHEET = 'Form Responses 1'

# Pastas de saída
Path('graficos').mkdir(exist_ok=True)
Path('resultados').mkdir(exist_ok=True)

# Estilo visual
plt.style.use('ggplot')
sns.set_theme(style='whitegrid', rc={'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# =====================================================
# LEITURA DOS DADOS
# =====================================================

print('Lendo arquivo...')

df = pd.read_excel(ARQUIVO, sheet_name=SHEET)

print(df.head())
print('\nTotal de respostas:', len(df))

# =====================================================
# RENOMEAR COLUNAS
# =====================================================

renomear = {
    'Q1. Área principal de atuação': 'area',
    'Q2. Nível de experiência profissional': 'experiencia',
    'Q3. Frequência de uso de LLMs (ex: ChatGPT, Copilot)': 'freq_uso',
    'Q4. Como você descreve seu nível de domínio em engenharia de prompt?': 'dominio_prompt',
    'Q5. Como você normalmente constrói seus prompts?': 'construcao_prompt',
    'Q6. Com que frequência você utiliza técnicas formais de engenharia de prompt?': 'uso_tecnicas',
    'Q7. Como você avalia a qualidade geral das respostas das LLMs?': 'qualidade',
    'Q8. As respostas atendem corretamente ao seu objetivo?': 'objetivo',
    'Q9. Como você classificaria a precisão técnica das respostas?': 'precisao',
    'Q10. Com que frequência você precisa reformular um prompt para obter uma resposta satisfatória?': 'reformulacao',
    'Q11. Quantas interações são necessárias para chegar a uma resposta ideal?': 'interacoes',
    'Q12. Como você avalia a facilidade de uso das LLMs no seu trabalho?': 'facilidade',
    'Q13. O uso de LLMs melhora sua produtividade?': 'produtividade',
    'Q14. O esforço necessário para obter boas respostas é:': 'esforco'
}

df = df.rename(columns=renomear)

# Nomes que serão usados nas legendas e títulos
mapa_nomes_reversos = {
    'area': 'Área de atuação',
    'experiencia': 'Experiência profissional',
    'freq_uso': 'Frequência do uso de LLMs',
    'dominio_prompt': 'Nível em eng. de prompt',
    'construcao_prompt': 'Como constrói seus prompts',
    'uso_tecnicas': 'Frequência do uso de técnicas formais',
    'qualidade': 'Qualidade respostas das LLMs',
    'objetivo': 'Atendem ao objetivo',
    'precisao': 'Precisão técnica das respostas',
    'reformulacao': 'Reformular para obter resposta satisfatória',
    'interacoes': 'Interações necessárias para resposta',
    'facilidade': 'Facilidade de uso das LLMs',
    'produtividade': 'LLMs melhora produtividade',
    'esforco': 'Esforço necessário para boas respostas'
}

# =====================================================
# MAPEAMENTO DE VALORES NUMÉRICOS PARA TEXTO (AP9)
# =====================================================

mapeamento_likert = {
    'uso_tecnicas': {
        1: '1 - Nunca', 2: '2 - Raramente', 3: '3 - Ocasionalmente', 
        4: '4 - Frequentemente', 5: '5 - Sempre'
    },
    'qualidade': {
        1: '1 - Muito baixa', 2: '2 - Baixa', 3: '3 - Moderada', 
        4: '4 - Alta', 5: '5 - Muito alta'
    },
    'objetivo': {
        1: '1 - Nunca', 2: '2 - Raramente', 3: '3 - Às vezes', 
        4: '4 - Frequentemente', 5: '5 - Sempre'
    },
    'reformulacao': {
        1: '1 - Sempre', 2: '2 - Frequentemente', 3: '3 - Às vezes', 
        4: '4 - Raramente', 5: '5 - Nunca'
    },
    'facilidade': {
        1: '1 - Muito difícil', 2: '2 - Difícil', 3: '3 - Moderada', 
        4: '4 - Fácil', 5: '5 - Muito fácil'
    }
}

for col, mapa in mapeamento_likert.items():
    if col in df.columns:
        df[col] = df[col].replace(mapa)

# =====================================================
# SELEÇÃO DAS VARIÁVEIS IMPORTANTES
# =====================================================

variaveis = [
    'dominio_prompt',
    'uso_tecnicas',
    'qualidade',
    'objetivo',
    'precisao',
    'reformulacao',
    'facilidade',
    'produtividade',
    'esforco'
]

titulos = [
    'Domínio do Prompt',
    'Uso das Técnicas',
    'Qualidade',
    'Atendem ao objetivo',
    'Precisão',
    'Reformulação',
    'Facilidade',
    'Produtividade',
    'Esforço'   
]

# =====================================================
# FUNÇÃO PARA GERAR GRÁFICOS DE FREQUÊNCIA
# =====================================================

def grafico_barras(coluna, name):
    plt.figure(figsize=(10, 5), facecolor='white')

    contagem = df[coluna].value_counts()

    sns.barplot(
        x=contagem.index,
        y=contagem.values,
        palette='Blues_d'
    )

    plt.title(f'Distribuição - {name}', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    plt.ylabel('Frequência')
    plt.tight_layout()

    caminho = f'graficos/barra_{coluna}.png'
    plt.savefig(caminho, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f'Gráfico salvo: {caminho}')

# =====================================================
# GERAR GRÁFICOS DESCRITIVOS
# =====================================================

for coluna, nome in zip(variaveis, titulos):
    grafico_barras(coluna, nome)

# =====================================================
# HEATMAP DE CORRELAÇÕES CATEGÓRICAS
# =====================================================

# Converter categorias para códigos numéricos

df_corr = pd.DataFrame()

for coluna in variaveis:
    df_corr[coluna] = pd.Categorical(df[coluna]).codes

corr = df_corr.corr(method='spearman')

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    xticklabels=titulos,  # Adicionado rótulos no eixo X
    yticklabels=titulos   # Adicionado rótulos no eixo Y
)

plt.title('Mapa de Correlação Entre Variáveis')
plt.tight_layout()

plt.savefig('graficos/heatmap_correlacao.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.close()

print('Heatmap salvo.')

# =====================================================
# FUNÇÃO PARA EXECUTAR ANACOR
# =====================================================


def executar_anacor(var1, var2):

    print(f'\nExecutando ANACOR: {var1} x {var2}')

    # -----------------------------
    # TABELA DE CONTINGÊNCIA
    # -----------------------------

    tabela = pd.crosstab(df[var1], df[var2])

    print('\nTabela de contingência:')
    print(tabela)

    # Exportar tabela
    tabela.to_excel(f'resultados/tabela_{var1}_{var2}.xlsx')

    # -----------------------------
    # TESTE QUI-QUADRADO
    # -----------------------------

    chi2, p, dof, expected = chi2_contingency(tabela)

    print(f'Qui-quadrado: {chi2:.4f}')
    print(f'p-value: {p:.4f}')

    # -----------------------------
    # ANÁLISE DE CORRESPONDÊNCIA
    # -----------------------------

    ca = prince.CA(
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )

    ca = ca.fit(tabela)

    # Coordenadas
    row_coords = ca.row_coordinates(tabela)
    col_coords = ca.column_coordinates(tabela)

    # -----------------------------
    # GRÁFICO ANACOR
    # -----------------------------

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Calcula um deslocamento proporcional à escala do gráfico
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    offset_x = (xlim[1] - xlim[0]) * 0.015
    offset_y = (ylim[1] - ylim[0]) * 0.015

    # Para as LINHAS: desloca ligeiramente para CIMA e para a DIREITA
    for label, (x, y) in row_coords.iterrows():
        ax.scatter(x, y, color='darkblue', marker='o', s=60)
        ax.annotate(
            str(label), 
            (x, y), 
            xytext=(x + offset_x, y + offset_y),
            ha='left', 
            va='bottom',
            fontsize=9, 
            color='darkblue',
            fontweight='bold'
        )

    # Para as COLUNAS: desloca ligeiramente para BAIXO e para a ESQUERDA
    for label, (x, y) in col_coords.iterrows():
        ax.scatter(x, y, color='darkred', marker='^', s=60)
        ax.annotate(
            str(label), 
            (x, y), 
            xytext=(x - offset_x, y - offset_y),
            ha='right', 
            va='top',
            fontsize=9, 
            color='darkred',
            fontweight='bold'
        )

    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

    plt.title(f'ANACOR - {mapa_nomes_reversos.get(var1, var1)} x {mapa_nomes_reversos.get(var2, var2)}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    caminho = f'graficos/anacor_{var1}_{var2}.png'
    plt.savefig(caminho, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f'Gráfico salvo: {caminho}')

    # -----------------------------
    # INÉRCIA EXPLICADA
    # -----------------------------

    eigenvalues = ca.eigenvalues_

    print('\nAutovalores:')
    print(eigenvalues)

    return {
        'variaveis': (var1, var2),
        'chi2': chi2,
        'p_value': p,
        'autovalores': eigenvalues
    }

# =====================================================
# PARES DE ANÁLISE
# =====================================================

pares = [
    ('dominio_prompt', 'qualidade'),
    ('dominio_prompt', 'precisao'),
    ('dominio_prompt', 'produtividade'),
    ('dominio_prompt', 'reformulacao'),
    ('uso_tecnicas', 'qualidade'),
    ('uso_tecnicas', 'precisao'),
    ('uso_tecnicas', 'produtividade'),
    ('uso_tecnicas', 'reformulacao'),
    ('facilidade', 'produtividade')
]

# =====================================================
# EXECUTAR TODAS AS ANACOR
# =====================================================

resultados = []

for var1, var2 in pares:

    resultado = executar_anacor(var1, var2)

    if resultado is not None:
        resultados.append(resultado)

    else:
        print(f'ANACOR ignorada para: {var1} x {var2}')

# =====================================================
# TABELA FINAL DE RESULTADOS
# =====================================================

if len(resultados) > 0:

    resultado_final = pd.DataFrame(resultados)

    resultado_final.to_excel(
        'resultados/resumo_anacor.xlsx',
        index=False
    )

    print('Resumo salvo com sucesso.')

else:
    print('Nenhuma ANACOR válida foi gerada.')