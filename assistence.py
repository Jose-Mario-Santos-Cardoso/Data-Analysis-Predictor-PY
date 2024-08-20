# pip install pandas numpy scikit-learn
# import pandas as pd
# tabela = pd.read_cvs()
# tabela = pd.read_sql()
# tabela = pd.read_csv("clientes.csv")
# display(tabela)


#códigos executados de cima p/ baixo //
#da esquerda p/ direita


#pode dá apelido p/ LabelEncoder:

# from sklearn.preprocessing import LabelEncoder as le

# codificador = le()


#Fit_transform => {tabela[""] = tabela[""] (Só que Transformada em n° pelo codificador)} => NovoValor é = antigo valor-sóq-transfrmd-n°-by codificador
#como transforma o antigo valor? -> usar função Fit_transform //
#ajustar o codificador e apply atransformação na table

# tabela[""] = codificador.fit_transform(tabela[""])

#dados x e dados y -> dados de x e y de treino/dados de x e y de teste
#importar ferramenta do scikit-learn:

# from sklearn.model_selection import train_text_split


#separar a base em dados de treino/de teste
#vai criar 4 caras:

# x_treino, x_teste, y_treino, y_teste = train_text_split(x, y)


#passando pro train_text_split =>(dados de[x e y])
#ele vai % -> y d treino/teste // x d treino/teste
#ele sabe quem é quem por conta-da-ordem
#o codigo de alguem em inglês, vai ser na msm ordem:

# x_train, x_test, y_train, y_test = train_test_split(x, y)


# vai ser na msm ordem
#O 1°=(x_treino)/
#2°=(x_teste)/
#3°=(y_treino)/
#4°=(y_teste)/
#o automatico do tain_test_split é 75% - 25%
#mas pode definir quantos % vai p/ test e
#quants % vai p/ treino

# (x, y, test_size=0.3)


# 30% p/ teste e 70% p/ treino
#volume de teste p/ garantir a consistencia dos dados da sua AI

# x_treino, x_teste, y_treino, y_teste = train_text_split(x, y, test_size=0.3)


#AI => (ordem[ from sklearn import]) -> importar

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier


#AI => (ordem[ name the model, where u wanna keep that AI,
#will be kept in that variable]) -> criar

# modelo_arvoredecisao = RandomForestClassifier()
# modelo_knn = KNeighborsClassifier()


#AI => (ordem[ "seu modelo".fit, e, os dados de treino(x_treino e y_treino)]) -> treinar

# modelo_arvoredecisao.fit(x_treino, y_treino)
# modelo_knn.fit(x_treino, y_treino)


#Site:  Kaggle
#Site C/ Base de Dados Ficticia p/ Testar e Treinar Sua IA
#Site:  Kaggle
#Site W/ Fictional Data Base to Test and Train Your AI

#arvore de decisão (tipo o Akinator)
#knn (vai analisar os vizinhos proximos -> separar a base de dados [credito bom e credito ruim])

#test to verify which is the best model
#testar pra ver qual o melhor modelo

#which model had the better sucess rate?
#calculate the model's prediction

#pegar a previsão do modelo de arvore de decisao e visinhos
#usar predict e o ponto(.)-> p/ ele prever os dados de teste dps de treinar... pegar o X de teste:

# previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
# previsao_knn = modelo_knn.predict(x_teste)


#dps das previsões
#verify the accuracy after the predictions
#verificar a acurácia das previsões

#tem que importar o accuracy_score:

# from sklearn.metrics import accuracy_score


#dps-> só dar um display dos resultados e comparar
#then-> use the display to compare the results:
#do y_teste e da previsão deles('previsao_arvoredecisao' e 'previsao_knn'):

# display(accuracy_score(y_teste, previisao_arvoredecisao))
# display(accuracy_score(y_teste, previsao_knn))


# nubak tem esse sist pra definir o limite do cartão de crédito
# vai rodar esse sistema e dar uma sugestão de qual deve ser sua nota de crédito
# "esse cliente pode ter(x {tanto}) de limite"
# to make a better percentage:
# - could use some higher data base to train w/ more info
# - if u give a little more trainning'base information,
# to see if w/ more trainning info -> would've better results
# - AI have parameters, u could use to adjust(and some tecnics)