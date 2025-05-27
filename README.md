README

Descrição do problema:

Classificacao de imagens : Gatos e cachorros,utilizando um modelo de rede neural convolucional treinado no dataset CIFAR-10,foi aplicado técnicas de pré-processamento de imagens e validação;


Justificativa das técnicas utilizadas:

Dataset CIFAR-10: usado para treinar com maior precisao e qualidade.

CNN: essa rede convulacional e otima em processamento de imagens, tambem ja foi ultilizada em sala de aula, por obvio eu ja tenho uma experiencia previa com ela e foi por isso que eu a escolhi;

Pré-processamento: Redimensionamento, normalização para ter uma maior qualidade na validacao das imagens selecionadas;

Treino e teste separados : Para ter uma melhor qualidade na pesquisa;

Etapas realizadas:

Parte 1: Montagem do drive no colab;

Parte 2: Carreguei o dataset do cifar e normalizei os dados;

Parte 3:Construi a CNN com as camadas Conv2D, MaxPooling e Dense;

Parte 4: treinei o modelo com validação interna;

Parte 5: Avaliei o modelo treinado com o conjunto de teste do CIFAR-10 e observei a acurácia e perda;

Parte 6: Fiz o pré-processamento das imagens locais, com redimensionamento, filtro gaussiano e equalização  para melhorar a qualidade da imagem.

Parte 7: Separei as imagens locais em treino e teste, para garantir uma validação mais correta.

Parte 8: Treinei um modelo com as imagens locais e avaliei com relatório de classificação (precision, recall e f1-score).

Parte 9: Fiz a predição das imagens locais utilizando o modelo treinado com
CIFAR-10, para analisar como ele se comportaria com imagens reais fora do dataset.

Resultados obtidos:

O modelo treinado com CIFAR-10 atingiu uma acurácia de aproximadamente 71%.
O modelo treinado com as imagens locais de gatos e cachorros teve um desempenho de 100%.

Acredito que as curva de treino e validacao tiveram um bom desempenho com o passar das epocas.

Tempo total gasto:

Aproximadamente 2 horas e meia.


Dificuldades encontradas:

Poucas imagens locais,a validação não ficou tão robusta quanto poderia.

Demorou muito tempo para treinar o Cifar.

O aviso de UserWarning sobre o uso do input_shape → precisei ajustar utilizando a camada Input para evitar esse tipo de alerta.

 Predição com modelo CIFAR-10,esqueci de ter definido class_names, mas esta funcionando.
