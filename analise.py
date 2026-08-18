import pandas as pd
import matplotlib.pyplot as plt
import prince
import seaborn as sns
from scipy.stats import chi2_contingency
from pathlib import Path
from typing import cast


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
sns.set_theme(style='whitegrid')

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
    plt.figure(figsize=(10, 5))

    contagem = df[coluna].value_counts()

    sns.barplot(
        x=contagem.index,
        y=contagem.values
    )

    plt.title(f'Distribuição - {name}')
    plt.xticks(rotation=25)
    plt.ylabel('Frequência')
    plt.tight_layout()

    caminho = f'graficos/barra_{coluna}.png'
    plt.savefig(caminho, dpi=300)
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

plt.savefig('graficos/heatmap_correlacao.png', dpi=300)
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

    plt.figure(figsize=(10, 8))

    # Garantir que existam pelo menos 2 dimensões
    if row_coords.shape[1] < 2 or col_coords.shape[1] < 2:
        print('ANACOR não possui duas dimensões suficientes.')
        return None

    # Alterado o parâmetro label para refletir o nome real das variáveis analisadas
    plt.scatter(
        row_coords.iloc[:, 0],
        row_coords.iloc[:, 1],
        label=mapa_nomes_reversos.get(var1, var1)
    )

    plt.scatter(
        col_coords.iloc[:, 0],
        col_coords.iloc[:, 1],
        label=mapa_nomes_reversos.get(var2, var2)
    )

    # Rótulos linhas
    for i, txt in enumerate(row_coords.index):
        x = row_coords.iloc[i, 0]
        y = row_coords.iloc[i, 1]

        plt.annotate(txt, cast(tuple[float, float], (x, y)))

    # Rótulos colunas
    for i, txt in enumerate(col_coords.index):
        x = col_coords.iloc[i, 0]
        y = col_coords.iloc[i, 1]

        plt.annotate(txt, cast(tuple[float, float], (x, y)))

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')

    plt.title(f'ANACOR - {mapa_nomes_reversos.get(var1, var1)} x {mapa_nomes_reversos.get(var2, var2)}')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    plt.legend()
    plt.tight_layout()

    caminho = f'graficos/anacor_{var1}_{var2}.png'

    plt.savefig(caminho, dpi=300)
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