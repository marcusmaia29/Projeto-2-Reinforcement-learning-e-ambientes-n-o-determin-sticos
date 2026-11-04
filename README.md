# Projeto 2 - Reinforcement Learning e Ambientes Não Determinísticos

Projeto da disciplina de Inteligência Artificial e Robótica utilizando o ambiente **Taxi-v4** do Gymnasium.

O objetivo é implementar e comparar dois algoritmos tabulares de reinforcement learning:

- Q-learning
- SARSA

Os agentes são treinados e avaliados em ambientes determinístico e não determinístico.

## Ambiente

Foi utilizado o **Taxi-v4** do Gymnasium 1.3.0.

O ambiente possui:

- 500 estados;
- 6 ações;
- reward -1 por passo;
- reward +20 pela entrega correta;
- reward -10 para pickup/dropoff ilegal;
- limite de 200 passos por episódio.

Para o cenário não determinístico é utilizado:

```python
gym.make("Taxi-v4", is_rainy=True, rainy_probability=0.6)
```

## Hiperparâmetros

Os mesmos valores são utilizados para Q-learning e SARSA:

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

1. cinco treinamentos de Q-learning no ambiente determinístico;
2. cinco treinamentos de SARSA no ambiente determinístico;
3. avaliação das políticas determinísticas em 100 episódios;
4. execução das políticas determinísticas no ambiente chuvoso;
5. cinco novos treinamentos de cada algoritmo no ambiente chuvoso;
6. avaliação das novas políticas em 100 episódios;
7. comparação das curvas de aprendizado, timesteps, desvio padrão, reward e taxa de sucesso.

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

- **rl_taxi.py**: implementação de Q-learning, SARSA, política epsilon-greedy e avaliação.
- **projeto2.ipynb**: execução dos experimentos, gráficos, tabelas e discussão.
- **resultados/**: arquivos CSV e imagens gerados pelo notebook.

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

Os resultados numéricos não são fixados manualmente no repositório antes da execução. O notebook calcula os valores reais e salva os dados e gráficos na pasta `resultados/`.

Os principais resultados apresentados são:

- reward acumulado durante o treinamento;
- curvas médias das cinco execuções;
- timesteps médios para completar a tarefa;
- desvio padrão;
- reward médio;
- taxa de sucesso;
- comparação Q-learning x SARSA;
- impacto do não determinismo;
- comparação entre política transferida do ambiente determinístico e política treinada diretamente no ambiente chuvoso.
