# Projeto 2 - Reinforcement Learning e Ambientes Não Determinísticos

Projeto da disciplina de Inteligência Artificial e Robótica utilizando o ambiente **Taxi-v4** do Gymnasium.

O objetivo é implementar e comparar dois algoritmos tabulares de reinforcement learning:

- Q-learning
- SARSA

Os agentes são treinados e avaliados em ambientes determinístico e não determinístico.

## Integrante

Marcus Vinicius Aguiar Maia

## Ambiente

Foi utilizado o **Taxi-v4** do Gymnasium 1.3.0.

O ambiente possui:

- 500 estados
- 6 ações
- reward -1 por passo
- reward +20 pela entrega correta
- reward -10 para pickup/dropoff ilegal
- limite de 200 passos por episódio

Para o cenário não determinístico:

```python
gym.make("Taxi-v4", is_rainy=True, rainy_probability=0.6)
```

## Hiperparâmetros

Os mesmos valores foram utilizados para Q-learning e SARSA:

- episódios: 10.000
- alpha: 0.1
- gamma: 0.95
- epsilon inicial: 1.0
- epsilon mínimo: 0.01
- epsilon decay: 0.9995
- máximo de passos: 200
- 5 treinamentos por configuração

## Experimentos

O notebook executa:

1. cinco treinamentos de Q-learning no ambiente determinístico
2. cinco treinamentos de SARSA no ambiente determinístico
3. avaliação das políticas determinísticas em 100 episódios
4. execução das políticas determinísticas no ambiente chuvoso
5. cinco novos treinamentos de cada algoritmo no ambiente chuvoso
6. avaliação das novas políticas em 100 episódios
7. comparação das curvas de aprendizado, timesteps, desvio padrão, reward e taxa de sucesso

Cada avaliação utiliza as cinco Q-tables obtidas no treinamento, com 20 episódios por tabela, totalizando 100 execuções por algoritmo e condição.

## Estrutura

```
.
├── projeto2.ipynb
├── rl_taxi.py
├── requirements.txt
├── .gitignore
└── resultados/
```

- **rl_taxi.py**: implementação de Q-learning, SARSA, política epsilon-greedy e avaliação
- **projeto2.ipynb**: execução dos experimentos, gráficos, tabelas e discussão
- **resultados/**: arquivos CSV e imagens gerados pelo notebook

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois abra:

```bash
jupyter notebook projeto2.ipynb
```

Execute todas as células em ordem.

## Resultados

> **Observação sobre os resultados atuais:** o código de treinamento foi corrigido para tratar `terminated` e `truncated` de forma adequada no bootstrap das atualizações. Portanto, os números e gráficos abaixo correspondem à execução anterior e devem ser regenerados executando novamente o notebook do início ao fim.


### Ambiente determinístico

| Algoritmo | Timesteps médios | Desvio padrão | Reward médio | Sucesso |
|---|---:|---:|---:|---:|
| Q-learning | 13.23 | 2.49 | 7.77 | 100% |
| SARSA | 13.39 | 2.62 | 7.61 | 100% |

Os dois agentes apresentaram desempenho final muito próximo. O Q-learning teve pequena vantagem nos timesteps e no reward médio. Na curva de treinamento, a média móvel do Q-learning atingiu a referência de reward 8 por volta do episódio 9.616, enquanto o SARSA não atingiu essa referência durante os 10.000 episódios.

### Política determinística no ambiente chuvoso

| Algoritmo | Timesteps médios | Desvio padrão | Reward médio | Sucesso |
|---|---:|---:|---:|---:|
| Q-learning | 24.75 | 9.30 | -3.75 | 100% |
| SARSA | 25.02 | 9.14 | -4.02 | 100% |

As políticas continuaram concluindo a tarefa, porém precisaram de mais passos e apresentaram maior variabilidade. Isso mostra o impacto das transições não determinísticas sobre uma política aprendida em um ambiente determinístico.

### Treinamento e avaliação no ambiente chuvoso

| Algoritmo | Timesteps médios* | Desvio padrão* | Reward médio | Sucesso |
|---|---:|---:|---:|---:|
| Q-learning | 24.87 | 8.99 | -9.75 | 97% |
| SARSA | 29.16 | 17.10 | -13.92 | 97% |

\* Timesteps calculados apenas nos episódios concluídos com sucesso.

Com os mesmos hiperparâmetros e 10.000 episódios, o treinamento no ambiente chuvoso apresentou maior dificuldade. O Q-learning obteve melhor reward médio e menor número médio de timesteps que o SARSA. Os dois tiveram 97% de sucesso, indicando que o aprendizado ainda não ficou tão estável quanto no ambiente determinístico.

## Gráficos

Os gráficos gerados pelo notebook estão disponíveis na pasta `resultados/`:

- `qlearning_deterministico.png`
- `sarsa_deterministico.png`
- `comparacao_deterministico.png`
- `comparacao_chuvoso.png`

## Conclusão

No ambiente determinístico, Q-learning e SARSA aprenderam políticas eficientes e alcançaram 100% de sucesso na avaliação, com pequena vantagem do Q-learning.

Ao executar essas políticas no ambiente chuvoso, o número de passos e a variabilidade aumentaram, embora as tarefas ainda tenham sido concluídas em todas as 100 avaliações. Quando os agentes foram treinados diretamente no ambiente não determinístico com os mesmos hiperparâmetros, ambos apresentaram 97% de sucesso e resultados mais instáveis, com vantagem do Q-learning nos experimentos realizados.

Os resultados completos de cada execução estão em `resultados/avaliacao.csv` e nos arquivos de rewards da pasta `resultados/`.
