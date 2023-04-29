#!/usr/bin/env python
# coding: utf-8

# In[92]:


import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt


# <h3>Upload dos dados</h3>

# In[93]:


df = pd.read_excel("TCC\Ciencia_Computacao-V4.xlsx")


# <h4> Removendo alunos sem nenhum CR/CRA ou disciplina cursada.</h4>

# In[94]:


df.dropna(subset=["CRA","disciplinasCursadas","crPorPeriodo","craPorPeriodo"], axis=0,inplace=True)


# <h4>Removendo Alunos com matrícula Ativa ou Trancada</h4>

# In[95]:


df = df.query('situacaoMatriculaAtual != "Ativa"')
df = df.query('situacaoMatriculaAtual != "Trancada"')


# In[96]:


df.shape


# <h4>Removendo alunos que entraram de 2007 para trás</h4>

# In[97]:


def filter_ano(crPorPeriodo):
    anos= ['2000','2001','2002','2003','2004','2005','2006','2007']
    for ano in anos:
        if ano in crPorPeriodo:
            return False
    return True


# In[98]:


ano_entrada_maior_2010 = [filter_ano(x) for x in df['crPorPeriodo']]
df["ano_entrada_maior_2010"] = ano_entrada_maior_2010
df = df.query("ano_entrada_maior_2010==True")


# <h4>Caracterização da Base</h4>

# In[99]:


import seaborn as sns
df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
df = df.query('id != 1523')


# In[100]:


def f(x):
    return x.count("-")


names = ['Cancelamento-Passivo','Cancelamento-Passivo']
fig, ax = plt.subplots()
periodo_evasao=[f(x) for x in df[df.situacaoMatriculaAtual.isin(names)]['crPorPeriodo']]
sns.histplot(data=periodo_evasao,discrete=True).set_title('Cancelamentos por periodo')


# <h4>Separando alunos em Conjuntos</h4>

# In[101]:


def diferenca_entrada_saida(crPorPeriodo):
    return len(str(crPorPeriodo).split('\n'))


# In[102]:


tempo_faculdade = [diferenca_entrada_saida(x) for x in df['crPorPeriodo']]
df["tempo_faculdade"] = tempo_faculdade


names = ['Cancelamento-Ativo','Cancelamento-Passivo']
df_cancelado = df.query("situacaoMatriculaAtual in @names")
df_cancelado.to_csv("cancelamento.csv",sep=";")
df_cancelado_apos_terceiro_periodo = df_cancelado.query("tempo_faculdade > 2")
df_cancelado_apos_quinto_periodo = df_cancelado.query("tempo_faculdade > 4")
df_cancelado_apos_setimo_periodo = df_cancelado.query("tempo_faculdade > 6")
df_cancelado_apos_primeiro_periodo = df_cancelado.query("tempo_faculdade >= 1")
df_graduado = df.query('situacaoMatriculaAtual == "Concluido"')
df_a3 = pd.concat([df_graduado, df_cancelado_apos_terceiro_periodo])
df_a5 = pd.concat([df_graduado, df_cancelado_apos_quinto_periodo])
df_a7 = pd.concat([df_graduado, df_cancelado_apos_setimo_periodo])
df_a9 = pd.concat([df_graduado, df_cancelado_apos_primeiro_periodo])
df_a9.to_csv("temporario_a9.csv",sep=";")


# <h3>Criação de novos atributos e Transformação dos Dados</h3>

# <h4>Conjunto A</h4>

# In[103]:


import re

def ano_periodos_validos(ano_periodo,periodo_corte):
    periodos_validos= []
    for i in range(periodo_corte):
        minimo = min(ano_periodo)
        periodos_validos.append(minimo)
        ano_periodo.remove(minimo)
    return periodos_validos

def cortar_disciplinas(disciplinasCursadas,periodo_corte):
    regexp = re.compile(r'\d{4}[\/][1-2]') #remove periodos com valor 0: 2012/0
    disciplinasCursadas = disciplinasCursadas.split("\n")
    temp = []
    ano_periodo = set()
    for linha in disciplinasCursadas:
        if regexp.search(linha):
            #print('matched')
            ano_periodo.add(regexp.search(linha).group())
            temp.append(linha)
        else:
            pass
    disciplinasCursadas = temp 
    #print(periodo_corte)
    #print(ano_periodo)
    periodos_validos = ano_periodos_validos(ano_periodo,periodo_corte)
    temp =[]
    for linha in disciplinasCursadas:
        if any(substring in linha for substring in periodos_validos):  
            temp.append(linha)
    disciplinasCursadas = temp  
    return disciplinasCursadas

def qtd_vezes_cursadas(disciplinasCursadas,periodo_corte,materia):
    disciplinasCursadas=cortar_disciplinas(disciplinasCursadas,periodo_corte)
    cnt=0
    for linha in disciplinasCursadas:
        if materia in linha:
            cnt=cnt+1
    return cnt
def nota_final(disciplinasCursadas,periodo_corte,materia):
    disciplinasCursadas=cortar_disciplinas(disciplinasCursadas,periodo_corte)
    regexp = re.compile(r'[-]\s\d{3}\s[-]') #remove periodos com valor 0: 2012/0
    cnt=0
    for linha in disciplinasCursadas:
        if materia in linha:
            cnt=cnt+1
            nota = regexp.search(linha).group()
            nota = int(nota[2:5])
            #print(nota)
    if cnt==0:return 0
    return nota

def ultimo_periodo_em_que_cursou(disciplinasCursadas,periodo_corte,materia):
    disciplinasCursadas=cortar_disciplinas(disciplinasCursadas,periodo_corte)
    regexp = re.compile(r'\d{4}[\/][1-2]') #remove periodos com valor 0: 2012/0
    cnt=0
    todos_os_periodos=set()
    for linha in disciplinasCursadas:
        todos_os_periodos.add(regexp.search(linha).group())
        if materia in linha:
            cnt=cnt+1
            periodo = regexp.search(linha).group()            
            #print(nota)
    if cnt==0:return 0
    #identificar qual periodo cursado
    tamanho_conjunto=len(todos_os_periodos)
    for i in range(tamanho_conjunto):
        minimo=min(todos_os_periodos)
        if periodo==minimo:
            return i+1
        todos_os_periodos.remove(minimo)
    raise ValueError('Periodo não encontrado')
    return "error"

def cr_periodo(crPorPeriodo,n_periodo):
    regexp = re.compile(r'\d{4}[\/][1-2]') #remove periodos com valor 0: 2012/0
    crPorPeriodo = crPorPeriodo.split("\n")
    crPorPeriodo=float(crPorPeriodo[n_periodo-1].split("-")[1].strip())
    return crPorPeriodo
        


# In[104]:


obrigatorias_a3 = ['MAB111','MAB112','MAB120','MAB624','MAE111',"MAB113","MAB240","MAB245","MAB352","MAE992"]
for materia in obrigatorias_a3:
    qtd_vezes = [qtd_vezes_cursadas(x,2,materia) for x in df_a3['disciplinasCursadas']]
    nota = [nota_final(x,2,materia) for x in df_a3['disciplinasCursadas']]
    ultimo_periodo = [ultimo_periodo_em_que_cursou(x,2,materia) for x in df_a3['disciplinasCursadas']]
    df_a3["".join([materia, "_vezes_cursadas"])] = qtd_vezes
    df_a3["".join([materia, "_nota_final"])] = nota
    df_a3["".join([materia, "_ultimo_periodo_em_que_cursou"])] = ultimo_periodo

cr_periodo_1 = [cr_periodo(x,1) for x in df_a3['crPorPeriodo']]
cr_periodo_2 = [cr_periodo(x,2) for x in df_a3['crPorPeriodo']]
df_a3["cr_periodo_1"] = cr_periodo_1
df_a3["cr_periodo_2"] = cr_periodo_2
df_a3.to_csv("a3_real_oficial.csv",sep=";")


# <h4>Conjunto B</h4>

# In[105]:


obrigatorias_a5 = ['MAB111','MAB112','MAB120','MAB624','MAE111',"MAB113","MAB240","MAB245","MAB352","MAE992","FIW125","MAB115","MAB116","MAB123","MAB353","MAE993","FIW230","MAB117","MAB230","MAB368","MAE994"]
for materia in obrigatorias_a5:
    qtd_vezes = [qtd_vezes_cursadas(x,4,materia) for x in df_a5['disciplinasCursadas']]
    nota = [nota_final(x,4,materia) for x in df_a5['disciplinasCursadas']]
    ultimo_periodo = [ultimo_periodo_em_que_cursou(x,4,materia) for x in df_a5['disciplinasCursadas']]
    df_a5["".join([materia, "_vezes_cursadas"])] = qtd_vezes
    df_a5["".join([materia, "_nota_final"])] = nota
    df_a5["".join([materia, "_ultimo_periodo_em_que_cursou"])] = ultimo_periodo

cr_periodo_1 = [cr_periodo(x,1) for x in df_a5['crPorPeriodo']]
cr_periodo_2 = [cr_periodo(x,2) for x in df_a5['crPorPeriodo']]
cr_periodo_3 = [cr_periodo(x,3) for x in df_a5['crPorPeriodo']]
cr_periodo_4 = [cr_periodo(x,4) for x in df_a5['crPorPeriodo']]
df_a5["cr_periodo_1"] = cr_periodo_1
df_a5["cr_periodo_2"] = cr_periodo_2
df_a5["cr_periodo_3"] = cr_periodo_3
df_a5["cr_periodo_4"] = cr_periodo_4
df_a5.to_csv("a5_real_oficial.csv",sep=";")


# <h4>Conjunto C</h4>

# In[106]:


#temp_1 = [cortar_disciplinas(x,2) for x in df_a3['disciplinasCursadas']]
obrigatorias_a7 = ['MAB111','MAB112','MAB120','MAB624','MAE111',"MAB113","MAB240","MAB245","MAB352","MAE992","FIW125","MAB115","MAB116","MAB123","MAB353","MAE993","FIW230","MAB117","MAB230","MAB368","MAE994","MAB236","MAB354","MAB355","MAB471","MAB489","MAB533","MAB122","MAB232","MAB508","MAD243"]

df_a7.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
df_a7 = df_a7.query('id != 1523')
for materia in obrigatorias_a7:
    qtd_vezes = [qtd_vezes_cursadas(x,6,materia) for x in df_a7['disciplinasCursadas']]
    nota = [nota_final(x,6,materia) for x in df_a7['disciplinasCursadas']]
    ultimo_periodo = [ultimo_periodo_em_que_cursou(x,6,materia) for x in df_a7['disciplinasCursadas']]
    df_a7["".join([materia, "_vezes_cursadas"])] = qtd_vezes
    df_a7["".join([materia, "_nota_final"])] = nota
    df_a7["".join([materia, "_ultimo_periodo_em_que_cursou"])] = ultimo_periodo

cr_periodo_1 = [cr_periodo(x,1) for x in df_a7['crPorPeriodo']]
cr_periodo_2 = [cr_periodo(x,2) for x in df_a7['crPorPeriodo']]
cr_periodo_3 = [cr_periodo(x,3) for x in df_a7['crPorPeriodo']]
cr_periodo_4 = [cr_periodo(x,4) for x in df_a7['crPorPeriodo']]
cr_periodo_5 = [cr_periodo(x,3) for x in df_a7['crPorPeriodo']]
cr_periodo_6 = [cr_periodo(x,4) for x in df_a7['crPorPeriodo']]
df_a7["cr_periodo_1"] = cr_periodo_1
df_a7["cr_periodo_2"] = cr_periodo_2
df_a7["cr_periodo_3"] = cr_periodo_3
df_a7["cr_periodo_4"] = cr_periodo_4
df_a7["cr_periodo_5"] = cr_periodo_5
df_a7["cr_periodo_6"] = cr_periodo_6
df_a7.to_csv("a7_real_oficial.csv",sep=";")


# <h4>Conjunto D</h4>

# In[107]:


#temp_1 = [cortar_disciplinas(x,2) for x in df_a3['disciplinasCursadas']]
obrigatorias_a9 = ['MAB111','MAB112','MAB120','MAB624','MAE111']
df_a9.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
#df_a9 = df_a9.query('id != 1523')
for materia in obrigatorias_a9:
    qtd_vezes = [qtd_vezes_cursadas(x,1,materia) for x in df_a9['disciplinasCursadas']]
    nota = [nota_final(x,1,materia) for x in df_a9['disciplinasCursadas']]
    ultimo_periodo = [ultimo_periodo_em_que_cursou(x,1,materia) for x in df_a9['disciplinasCursadas']]
    df_a9["".join([materia, "_vezes_cursadas"])] = qtd_vezes
    df_a9["".join([materia, "_nota_final"])] = nota
    df_a9["".join([materia, "_ultimo_periodo_em_que_cursou"])] = ultimo_periodo

cr_periodo_1 = [cr_periodo(x,1) for x in df_a9['crPorPeriodo']]
df_a9["cr_periodo_1"] = cr_periodo_1
#df_a9 = df_a9.query('cr_periodo_1 > 0')
df_a9.to_csv("a9_real_oficial_2.csv",sep=";")


# <h3>Prediction Time</h3>

# <h4>Model 1 - Apenas Dados Demográficos</h4>

# In[108]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#df_a3_model_1.drop(columns=["tempo_faculdade","reprovacoes","periodosCRMenor3","periodosCancelados","periodosTrancados","CRA","cargaHorariaAcumulada","periodosIntegralizados","crPorPeriodo","craPorPeriodo","notaEnem","dataNascimento","disciplinasCursadas"],inplace=True)
df_a3_model_1 = df_a3[['codCursoIngresso','situacaoMatriculaAtual','sexo','modalidadeCota']]
df_a5_model_1 = df_a5[['codCursoIngresso','situacaoMatriculaAtual','sexo','modalidadeCota']]
df_a7_model_1 = df_a7[['codCursoIngresso','situacaoMatriculaAtual','sexo','modalidadeCota']]
df_a9_model_1 = df_a9[['codCursoIngresso','situacaoMatriculaAtual','sexo','modalidadeCota']]

df_a3_model_1['situacaoMatriculaAtual'] = df_a3_model_1['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a3_model_1.to_csv("a3_model_1.csv",sep=";")
df_a3_model_1 = pd.get_dummies(df_a3_model_1)
df_a3_model_1.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)

df_a5_model_1['situacaoMatriculaAtual'] = df_a5_model_1['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a5_model_1 = pd.get_dummies(df_a5_model_1)
df_a5_model_1.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)

df_a7_model_1['situacaoMatriculaAtual'] = df_a7_model_1['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a7_model_1 = pd.get_dummies(df_a7_model_1)
df_a7_model_1.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)

df_a9_model_1['situacaoMatriculaAtual'] = df_a9_model_1['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a9_model_1 = pd.get_dummies(df_a9_model_1)
df_a9_model_1.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)


# <h5>Model 1 - Conjunto A </h5>

# In[109]:


import statistics
from sklearn.metrics import recall_score

X_train_a3, X_test_a3, y_train_a3, y_test_a3 = train_test_split(df_a3_model_1.drop(columns=['situacaoMatriculaAtual_Cancelamento']), df_a3_model_1["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a3, y_train_a3, scoring='f1',cv=10)
    scores_acurracy = cross_val_score(clf, X_train_a3, y_train_a3, scoring='accuracy',cv=10)
    scores_recall = cross_val_score(clf, X_train_a3, y_train_a3, scoring='recall',cv=10)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    
    


# In[110]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
    
clf= DecisionTreeClassifier(random_state=0,max_depth=3)
clf = clf.fit(X_train_a3, y_train_a3)
y_test_a3.to_csv("y_test_a3.csv",sep=";")
y_pred = clf.predict(X_test_a3)
print("f1_score="+str(f1_score(y_test_a3, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a3, y_pred)))
print("recall=",str(recall_score(y_test_a3, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a3, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a3,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()


importances_sk = clf.feature_importances_
features = X_train_a3.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")





# In[111]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a3.columns),filled = True,class_names=['Concluiu','Evadiu'], fontsize=15)
fig.savefig('arvore_exp_1_conj_A.png')


# <h5>Model 1 - Conjunto B </h5>

# In[112]:


import statistics

X_train_a5, X_test_a5, y_train_a5, y_test_a5 = train_test_split(df_a5_model_1.drop(columns=['situacaoMatriculaAtual_Cancelamento']), df_a5_model_1["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a5, y_train_a5, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a5, y_train_a5, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a3, y_train_a3, scoring='recall',cv=10)
    print("f1:"+str(100*statistics.mean(scores_f1)))
    print("acurracy:"+str(100*statistics.mean(scores_acurracy)))
    print("recall:"+str(100*statistics.mean(scores_recall)))
    print("==")    


# In[113]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score

    
clf= DecisionTreeClassifier(random_state=0,max_depth=3)
clf = clf.fit(X_train_a5, y_train_a5)
y_test_a5.to_csv("y_test_a5.csv",sep=";")
y_pred = clf.predict(X_test_a5)
print("f1_score="+str(round(100*f1_score(y_test_a5, y_pred),2)))
print("accuracy=",str(round(100*accuracy_score(y_test_a5, y_pred),2)))
print("accuracy=",str(round(100*recall_score(y_test_a5, y_pred),2)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a5, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a5,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()


importances_sk = clf.feature_importances_
features = X_train_a5.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[114]:


fig, axes = plt.subplots(figsize = (16,16), dpi=600)
tree.plot_tree(clf,feature_names = list(X_test_a5.columns),filled = True,class_names=['Concluiu','Evadiu'], fontsize=12)
fig.savefig('arvore_exp_1_conj_B.png')


# <h5>Model 1 - Conjunto D </h5>

# In[115]:


import statistics

X_train_a9, X_test_a9, y_train_a9, y_test_a9 = train_test_split(df_a9_model_1.drop(columns=['situacaoMatriculaAtual_Cancelamento']), df_a9_model_1["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a9, y_train_a9, scoring='f1',cv=5)
    scores_acurracy = cross_val_score(clf, X_train_a9, y_train_a9, scoring='accuracy',cv=5)
    scores_recall = cross_val_score(clf, X_train_a9, y_train_a9, scoring='recall',cv=5)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[116]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score

    
clf= DecisionTreeClassifier(random_state=0,max_depth=1)
clf = clf.fit(X_train_a9, y_train_a9)
y_test_a9.to_csv("y_test_a9.csv",sep=";")
y_pred = clf.predict(X_test_a9)
print("f1_score="+str(f1_score(y_test_a9, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a9, y_pred)))
print("recall=",str(recall_score(y_test_a9, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a9, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a9,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()


importances_sk = clf.feature_importances_
features = X_train_a9.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[119]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a9.columns),filled = True,class_names=['Concluiu','Evadiu'])
fig.savefig('arvore_exp_1_conj_D.png')


# <h5>Model 1 - Conjunto C </h5>

# In[120]:


import statistics

X_train_a7, X_test_a7, y_train_a7, y_test_a7 = train_test_split(df_a7_model_1.drop(columns=['situacaoMatriculaAtual_Cancelamento']), df_a7_model_1["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a7, y_train_a7, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a7, y_train_a7, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a7, y_train_a7, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[121]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
    
clf= DecisionTreeClassifier(random_state=0,max_depth=1)
clf = clf.fit(X_train_a7, y_train_a7)
#y_test_a3.to_csv("y_test_a3.csv",sep=";")
y_pred = clf.predict(X_test_a7)
print("f1_score="+str(f1_score(y_test_a7, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a7, y_pred)))
print("recall=",str(recall_score(y_test_a7, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a7, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a7,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()


importances_sk = clf.feature_importances_
features = X_train_a7.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[122]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a5.columns),filled = True,class_names=['Concluiu','Evadiu'])
fig.savefig('arvore_exp_1_conj_C.png')


# <h4>Model 2 - Dados Acadêmicos</h4>

# In[123]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#df_a3_model_1.drop(columns=["tempo_faculdade","reprovacoes","periodosCRMenor3","periodosCancelados","periodosTrancados","CRA","cargaHorariaAcumulada","periodosIntegralizados","crPorPeriodo","craPorPeriodo","notaEnem","dataNascimento","disciplinasCursadas"],inplace=True)
#df_a3.to_csv("formato_dados.csv",sep=";")
#df_a3_model_2 = df_a3[['codCursoIngresso','formaIngresso','situacaoMatriculaAtual','sexo','modalidadeCota']]
#df_a3_model_2 = df_a3.drop(columns=['periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade','Unnamed: 0','codCursoIngresso','formaIngresso','sexo','modalidadeCota'],inplace=False)
#df_a3_model_2.to_csv("formato_dados.csv",sep=";")

df_a3_model_2 = df_a3.drop(columns=['periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade','codCursoIngresso','formaIngresso','sexo','modalidadeCota'],inplace=False)
df_a5_model_2 = df_a5.drop(columns=['periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade','codCursoIngresso','formaIngresso','sexo','modalidadeCota'],inplace=False)
df_a7_model_2 = df_a7.drop(columns=['periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade','codCursoIngresso','formaIngresso','sexo','modalidadeCota'],inplace=False)
df_a9_model_2 = df_a9.drop(columns=['periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade','codCursoIngresso','formaIngresso','sexo','modalidadeCota'],inplace=False)

df_a3_model_2.to_csv("a3_model_2.csv",sep=";")

df_a3_model_2['situacaoMatriculaAtual'] = df_a3_model_2['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a3_model_2.to_csv("pre_dummies.csv",sep=";")
df_a3_model_2 = pd.get_dummies(df_a3_model_2)
#df_a3_model_2.to_csv("teste_new_novo.csv",sep=";")
df_a3_model_2.drop(columns=["situacaoMatriculaAtual_Concluido"],inplace=True)

df_a5_model_2['situacaoMatriculaAtual'] = df_a5_model_2['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a5_model_2 = pd.get_dummies(df_a5_model_2)
df_a5_model_2.drop(columns=["situacaoMatriculaAtual_Concluido"],inplace=True)

df_a7_model_2['situacaoMatriculaAtual'] = df_a7_model_2['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a7_model_2 = pd.get_dummies(df_a7_model_2)
df_a7_model_2.drop(columns=["situacaoMatriculaAtual_Concluido"],inplace=True)

df_a9_model_2['situacaoMatriculaAtual'] = df_a9_model_2['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a9_model_2 = pd.get_dummies(df_a9_model_2)
df_a9_model_2.drop(columns=["situacaoMatriculaAtual_Concluido"],inplace=True)


# <h5>Model 2 - Conjunto A </h5>

# In[124]:


import statistics

X_train_a3, X_test_a3, y_train_a3, y_test_a3 = train_test_split(df_a3_model_2.drop(columns=['situacaoMatriculaAtual_Cancelamento']), df_a3_model_2["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a3, y_train_a3, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a3, y_train_a3, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a3, y_train_a3, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    
    


# In[ ]:





# In[125]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=3)
clf = clf.fit(X_train_a3, y_train_a3)
y_pred = clf.predict(X_test_a3)
print("f1_score="+str(f1_score(y_test_a3, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a3, y_pred)))
print("recall=",str(recall_score(y_test_a3, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a3, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a3,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a3.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[126]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a3.columns),filled = True,class_names=['Concluiu','Evadiu'],fontsize=13)
fig.savefig('tree_m2_A.png')


# <h5>Model 2 - Conjunto B </h5>

# In[127]:


import statistics

X_train_a5, X_test_a5, y_train_a5, y_test_a5 = train_test_split(df_a5_model_2.drop(columns=['situacaoMatriculaAtual_Cancelamento','id']), df_a5_model_2["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a5, y_train_a5, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a5, y_train_a5, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a5, y_train_a5, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[128]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=2)
clf = clf.fit(X_train_a5, y_train_a5)
y_pred = clf.predict(X_test_a5)
print("f1_score="+str(f1_score(y_test_a5, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a5, y_pred)))
print("recall=",str(recall_score(y_test_a5, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a5, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a5,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a5.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[129]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a5.columns),filled = True,class_names=['Concluiu','Evadiu'])
fig.savefig('tree_m2_B.png')


# <h5>Model 2 - Conjunto D </h5>

# In[130]:


import statistics

X_train_a9, X_test_a9, y_train_a9, y_test_a9 = train_test_split(df_a9_model_2.drop(columns=['situacaoMatriculaAtual_Cancelamento','id']), df_a9_model_1["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(17):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a9, y_train_a9, scoring='f1',cv=5)
    scores_acurracy = cross_val_score(clf, X_train_a9, y_train_a9, scoring='accuracy',cv=5)
    scores_recall = cross_val_score(clf, X_train_a9, y_train_a9, scoring='recall',cv=5)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[131]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=2)
clf = clf.fit(X_train_a9, y_train_a9)
y_pred = clf.predict(X_test_a9)
print("f1_score="+str(f1_score(y_test_a9, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a9, y_pred)))
print("recall=",str(recall_score(y_test_a9, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a9, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a9,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a9.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[132]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a9.columns),filled = True,class_names=['Concluiu','Evadiu'])
fig.savefig('tree_m2_cD.png')


# <h5>Model 2 - Conjunto C </h5>

# In[133]:


import statistics

X_train_a7, X_test_a7, y_train_a7, y_test_a7 = train_test_split(df_a7_model_2.drop(columns=['situacaoMatriculaAtual_Cancelamento','id']), df_a7_model_1["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(17):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a7, y_train_a7, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a7, y_train_a7, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a7, y_train_a7, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[134]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=3)
clf = clf.fit(X_train_a7, y_train_a7)
y_pred = clf.predict(X_test_a7)
print("f1_score="+str(f1_score(y_test_a7, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a7, y_pred)))
print("recall=",str(recall_score(y_test_a7, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a7, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a7,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a7.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[135]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a7.columns),filled = True,class_names=['Concluiu','Evadiu'],fontsize=11)
fig.savefig('arv_m2cC.png')


# <h4>Model 3 - Dados Demográficos e Acadêmicos</h4>

# In[136]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#df_a3_model_1.drop(columns=["tempo_faculdade","reprovacoes","periodosCRMenor3","periodosCancelados","periodosTrancados","CRA","cargaHorariaAcumulada","periodosIntegralizados","crPorPeriodo","craPorPeriodo","notaEnem","dataNascimento","disciplinasCursadas"],inplace=True)
#df_a3.to_csv("formato_dados.csv",sep=";")
#df_a3_model_2 = df_a3[['codCursoIngresso','formaIngresso','situacaoMatriculaAtual','sexo','modalidadeCota']]
#df_a3_model_2 = df_a3.drop(columns=['periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade','Unnamed: 0','codCursoIngresso','formaIngresso','sexo','modalidadeCota'],inplace=False)
#df_a3_model_2.to_csv("formato_dados.csv",sep=";")

df_a3_model_3 = df_a3.drop(columns=['formaIngresso','periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade'],inplace=False)
df_a5_model_3 = df_a5.drop(columns=['formaIngresso','periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade'],inplace=False)
df_a7_model_3 = df_a7.drop(columns=['formaIngresso','periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade'],inplace=False)
df_a9_model_3 = df_a9.drop(columns=['formaIngresso','periodoIngressoUFRJ','cursoIngressoUFRJ','cursoAtual','codCursoAtual','periodoIngressoCursoAtual','notaEnem','dataNascimento','disciplinasCursadas','crPorPeriodo','craPorPeriodo','periodosIntegralizados','cargaHorariaAcumulada','CRA','periodosTrancados','periodosCancelados','periodosCRMenor3','reprovacoes','ano_entrada_maior_2010','tempo_faculdade'],inplace=False)

df_a3_model_3.to_csv("a3_model_2.csv",sep=";")

df_a3_model_3['situacaoMatriculaAtual'] = df_a3_model_3['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a3_model_3 = pd.get_dummies(df_a3_model_3)
df_a3_model_3.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)


df_a9_model_3['situacaoMatriculaAtual'] = df_a9_model_3['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a9_model_3 = pd.get_dummies(df_a9_model_3)
df_a9_model_3.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)

df_a5_model_3['situacaoMatriculaAtual'] = df_a5_model_3['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a5_model_3 = pd.get_dummies(df_a5_model_3)
df_a5_model_3.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)

df_a7_model_3['situacaoMatriculaAtual'] = df_a7_model_3['situacaoMatriculaAtual'].replace(['Cancelamento-Passivo', 'Cancelamento-Ativo'], 'Cancelamento')
df_a7_model_3 = pd.get_dummies(df_a7_model_3)
df_a7_model_3.drop(columns=["situacaoMatriculaAtual_Concluido","sexo_F"],inplace=True)


# <h5>Model 3 - Conjunto A </h5>

# In[137]:


import statistics

X_train_a3, X_test_a3, y_train_a3, y_test_a3 = train_test_split(df_a3_model_3.drop(columns=['id','situacaoMatriculaAtual_Cancelamento']), df_a3_model_3["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a3, y_train_a3, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a3, y_train_a3, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a3, y_train_a3, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    
    


# In[138]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=3)
clf = clf.fit(X_train_a3, y_train_a3)
y_pred = clf.predict(X_test_a3)
print("f1_score="+str(f1_score(y_test_a3, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a3, y_pred)))
print("recall=",str(recall_score(y_test_a3, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a3, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a3,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a3.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[139]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a3.columns),filled = True,class_names=['Concluiu','Evadiu'],fontsize=11)
fig.savefig('tree_m3_cA.png')


# <h5>Model 3 - Conjunto B </h5>

# In[140]:


import statistics

X_train_a5, X_test_a5, y_train_a5, y_test_a5 = train_test_split(df_a5_model_3.drop(columns=['id','situacaoMatriculaAtual_Cancelamento']), df_a5_model_3["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(15):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a5, y_train_a5, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a5, y_train_a5, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a5, y_train_a5, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[141]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=2)
clf = clf.fit(X_train_a5, y_train_a5)
y_pred = clf.predict(X_test_a5)
print("f1_score="+str(f1_score(y_test_a5, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a5, y_pred)))
print("recall=",str(recall_score(y_test_a5, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a5, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a5,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a5.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[142]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a5.columns),filled = True,class_names=['Concluiu','Evadiu'],fontsize=11)
fig.savefig('tree_m3_cB.png')


# <h5>Model 3 - Conjunto C </h5>

# In[143]:


import statistics

X_train_a7, X_test_a7, y_train_a7, y_test_a7 = train_test_split(df_a7_model_3.drop(columns=['id','situacaoMatriculaAtual_Cancelamento']), df_a7_model_3["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(17):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a7, y_train_a7, scoring='f1',cv=7)
    scores_acurracy = cross_val_score(clf, X_train_a7, y_train_a7, scoring='accuracy',cv=7)
    scores_recall = cross_val_score(clf, X_train_a7, y_train_a7, scoring='recall',cv=7)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[144]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=3)
clf = clf.fit(X_train_a5, y_train_a5)
y_pred = clf.predict(X_test_a5)
print("f1_score="+str(f1_score(y_test_a5, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a5, y_pred)))
print("recall=",str(recall_score(y_test_a5, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a5, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a5,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a5.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[145]:


fig, axes = plt.subplots(figsize = (16,16), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a7.columns),filled = True,class_names=['Concluiu','Evadiu'],fontsize=11)
fig.savefig('tree_m3_cC.png')


# <h5>Model 3 - Conjunto D </h5>

# In[146]:


import statistics

X_train_a9, X_test_a9, y_train_a9, y_test_a9 = train_test_split(df_a9_model_3.drop(columns=['id','situacaoMatriculaAtual_Cancelamento']), df_a9_model_3["situacaoMatriculaAtual_Cancelamento"], test_size=0.20, shuffle=True,random_state=42) 

for i in range(17):
    print("max_depth="+str(i+1))
    clf= DecisionTreeClassifier(random_state=0,max_depth=i+1)
    scores_f1 = cross_val_score(clf, X_train_a9, y_train_a9, scoring='f1',cv=5)
    scores_acurracy = cross_val_score(clf, X_train_a9, y_train_a9, scoring='accuracy',cv=5)
    scores_recall = cross_val_score(clf, X_train_a9, y_train_a9, scoring='recall',cv=5)
    print("f1:"+str(statistics.mean(scores_f1)))
    print("acurracy:"+str(statistics.mean(scores_acurracy)))
    print("recall:"+str(statistics.mean(scores_recall)))
    print("==")    


# In[147]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
    
clf= DecisionTreeClassifier(random_state=0,max_depth=4)
clf = clf.fit(X_train_a9, y_train_a9)
y_pred = clf.predict(X_test_a9)
print("f1_score="+str(f1_score(y_test_a9, y_pred)))
print("accuracy=",str(accuracy_score(y_test_a9, y_pred)))
print("recall=",str(recall_score(y_test_a9, y_pred)))

target_names = ['Concluido', 'Cancelado']
print(classification_report(y_test_a9, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_a9,y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
disp.plot()
plt.show()

importances_sk = clf.feature_importances_
features = X_train_a9.columns.tolist()
# Creating a dataframe with the feature importance by sklearn
feature_importance_sk = {}
for i, feature in enumerate(features):
    feature_importance_sk[feature] = round(importances_sk[i], 3)
    print(feature,importances_sk[i])

print(f"Feature importance by sklearn: {feature_importance_sk}")


# In[148]:


fig, axes = plt.subplots(figsize = (18,18), dpi=800)
tree.plot_tree(clf,feature_names = list(X_test_a3.columns),filled = True,class_names=['Concluiu','Evadiu'],fontsize=8)
fig.savefig('tree_m3_cD.png')

