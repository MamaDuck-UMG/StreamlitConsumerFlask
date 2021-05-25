import streamlit as st #funciones generales de streamlit
from bokeh.plotting import figure as grafica #para mostrar graficas de lineas
import plotnine as p9 #pip install plotnine, para graficas de puntos y de lineas
from PIL import Image #para abrir imagenes
import numpy as np#para arrays
import pandas as pd#para dataframes
import streamlit.components.v1 as components#para importar y exportar elementos de archivos
import datetime#libreria para usar formatos de fechas 
import json#libreria para usar json
from queue import PriorityQueue #libreria colas de prioridad
import math #Para los infinitos
import requests
import warnings
import matplotlib.pyplot as plt
from os import remove
warnings.filterwarnings('ignore')
image = Image.open('duck.png')#abro el ícono de Mama Duck
st.title("Reportes para Mama Duck")#encabezado
st.sidebar.title("Bienvenidos")#encabezado
st.sidebar.image(image, caption='Mama Duck',width=200)#subo la imagen con su tamaño y pie de foto
#server="143.198.146.111"#cambiar a localhost si se desea probar en un mismo host que flask
server="localhost"
puerto=5000#5000 es el puerto predeterminado de flask
#importamos el json que es actualmente un archivo de ejemplo
with open('ejemplo.json') as file:
    datajson = json.load(file)#cargamos el archivo a una variable
jnodes=datajson.get("nodes")
jn1=jnodes[0]#esta variable guarda todo el json, recordar que su acceso es similara a diccionaros de dict o listas


## ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA

def DataJSONtoGraph(numnodo):#funcion para extraer datos de acuerdo con el formato del algoritmo de recorrido
#exclusivo para algoritmo de recorrido
  ismamaduck=False
  numcalls=0
  activetime=None
  numcalls=len(jn1.get(numnodo)[0]['history'])#ya esta listo numcalls
  numConected=len(jn1.get(numnodo)[0]['conections'])
  if numnodo=='1':
    ismamaduck=True
  adys=jn1.get(numnodo)[0]['conections']
  return int(numnodo),numConected,None,ismamaduck,numcalls,activetime,adys

#funcion realizada el primer parcial de IA: recorrido de nodos por prioridad
def recorridonodos(n1):  # recibe una cadena de valores
    n0 = n1  # lo cofiguro como un nodo tipo puzzle
    Q = PriorityQueue()  # Q es una cola de prioridad
    aux = 0  # otro indice secundario de prioridad para Q
    Q.put((n0.f, aux, n0))  # la cola de prioridad se manejará mediante los f(n)
    visitados = []
    visitadoschart = []
    while not Q.empty():  # mientras la cola no este vacía
        u = Q.get()  # con el metodo get se guarda en u pero se quita de la cola el elemento
        u = u[2]  # porque una cola de prioridad almacena tuplas de prioridad,contador,nodo
        if u.tag not in visitados:
            visitados.append(u.tag)  # para evitar volverlo a visitar
            visitadoschart.append(u)
        ady = u.expand(u.adyacentes)  # expand me genera una lista de adyacencia, con heuristica y señalando a su padre, establece costo de 1 al generar neuvo nodo
        for v in ady:  # explorar los vecinos
            if v.tag not in visitados:  # si todavia no esta en visitados
                fp = v.h + v.g  # cálculo de funciones
                if fp < v.f:
                    v.f = fp
                    aux = aux + 1  # debo tener un entero antes de insertar un nodo en prioridad
                    Q.put((-(v.f), aux,
                           v))  # lo colocamos en la cola, para que en cada ciclo se evite agregar uno repetido
    return visitados, visitadoschart #visitados son solo los valores, visitadoschart son objetos nodo


class Nodo:#estructura de Nodo para recorrido de nodos
    w=None
    h = None  # heuristica
    f = math.inf  # f es infinito

    def __init__(self, tag, conexiones, padre, mamaduck, personasconect, tiempoactivos,adyacentes):  # inicializador
        self.tag = tag
        self.conexiones = conexiones
        self.padre = padre
        self.mamaduck = mamaduck
        self.tiempoactivos = tiempoactivos
        self.adyacentes = adyacentes#adyacentes=adyacentes primarios
        if padre is not None:  # si es un hijo
            self.g = padre.g + 1
        else:  # si es padre
            self.g = 0
            self.f = 0  # su f debe valer 0
        self.h = personasconect
    def expand(self, adyacentes):  # nos va a decir como explorar el grafo, obtiene la raiz en primera instancia
        ady = []  # aqui guardaremos los adyacentes que son objetos del tipo nodo
        for i in adyacentes:
          tag,conections,parent,mamaduck,people,timeactive,list1=DataJSONtoGraph(i)
          ady.append(Nodo(tag, conections, self, mamaduck, people, timeactive,list1))
        return ady  # retornamos los adyacentes o hijos del nodo expandido

def prepdatoLRML(e,f,g):#se preparan los datos para insertarse en el algoritmo de Regresion Logistica
#e son los nombres de emergencias presentadas en nodo
#f son los valores de los indices de riesgo
#g almacena el diccionario de riesgos
  max_item_index = f.index(max(f, key=int))#vamos a obtener el valor máximo de la lista, en su índice
  namemergency=e[max_item_index]#ese indice indicaremos en la lista de nombres de emergencias
  contador=0#para sumar valores
  tuplapred=[]
  for i in range(len(e)):
    contador+=f[i]#sumo el numero de emergencias para promediar indices
  porcentaje=f[max_item_index]/contador
  if namemergency=='fire':#al porcentaje le sumare valores enteros dependiendo el tipo de emergencia 
  #0=medic, 1=fire,2=security,3=sos,4=otros (aun no entrenado para otros)
    tuplapred.append([porcentaje+1,g[0].get('firerisk')])
  elif namemergency=='medic':
    tuplapred.append([porcentaje,g[0].get('medicrisk')])
  elif namemergency=='securityrisk':
    tuplapred.append([porcentaje+2,g[0].get('securityrisk')])
  elif namemergency=='sos':
    tuplapred.append([porcentaje+3,(g[0].get('other')+g[0].get('securityrisk')+g[0].get('medicrisk')+g[0].get('firerisk'))/4])
  return tuplapred

def prepdatoLRMLGen(e,f,g):#lo mismo que lo anterior, solo que itera sobre un diccionario mas grande (por ser general)
  max_item_index = f.index(max(f, key=int))
  namemergency=e[max_item_index]
  contador=0
  tuplapred=[]
  promfire=prommedic=promsec=promother=0
  promedios=list()#aqui se guardará el promedio de riesgos de todos los nodos
  for elem in range(len(g)):
    promfire+=g[elem].get('firerisk')
    prommedic+=g[elem].get('medicrisk')
    promsec+=g[elem].get('securityrisk')
    promother+=g[elem].get('other')
  promedios.append(promfire/len(g))
  promedios.append(prommedic/len(g))
  promedios.append(promsec/len(g))
  promedios.append(promother/len(g))
  for i in range(len(e)):
    contador+=f[i]
  porcentaje=f[max_item_index]/contador
  if namemergency=='fire':
    tuplapred.append([porcentaje+1,promedios[0]])
  elif namemergency=='medic':
    tuplapred.append([porcentaje,promedios[1]])
  elif namemergency=='medic':
    tuplapred.append([porcentaje+2,promedios[2]])
  elif namemergency=='sos':
    tuplapred.append([porcentaje+3,(promedios[3]+promedios[2]+promedios[1]+promedios[0])/4])
  return tuplapred


strindices=[]#este string va a guardar el nombre de los nodos (las funciones get del diccionario solo aceptan strings)
for i in range(len(jn1)):
  strindices.append(str(i+1))#esta ya puede utilizarse para otras funciones


def obtencionCoords():#obtengo las coordenadas de latitud y longitud de todos los nodos
  #lat y lon 1 es para coordenadas de llamadas
  #lat  lon2 son para coordenadas de NODOS
  list1=list()
  for s in strindices:
    for i in range(len(jn1.get(str(s))[0]['history'])):
      auxlat=jn1.get(str(s))[0]['history'][i]['localization'][0]['lat']
      auxlon=jn1.get(str(s))[0]['history'][i]['localization'][0]['long']
      list1.append([auxlat,auxlon])
    #esto era para la ubicacion arbitraria de los nodos
    #list2.append([jn1.get(str(s))[0]['localization'][0]['lat'],jn1.get(str(s))[0]['localization'][0]['long']])
  a1=np.array(list1)
  #a2=np.array(list2)
  #print(a1.reshape(-2,2))
  df1 = pd.DataFrame(
  a1,
  columns=['lat', 'lon'])
  #print(a1.reshape(-2,2))
  #df2 = pd.DataFrame(
  #a2,
  #columns=['lat', 'lon'])
  return df1


def obtencionlistasJS(numnodo):
  #obtiene información de los archivos JSON dado un nodo a explorar
  jnemergency=list()#guarda los strings de tipo de emergencia
  jnumsoc=list()#cuenta las ocurrencias de horas
  jhours=list()#guarda las horas (formato 0.00 a 23.00)
  jndate=list()#guarda fechas
  jrisks=list()#guarda diccionario de riesgos
  varisk=jn1.get(numnodo)[0]['risks'][0]
  jrisks.append(varisk)#guardo el directorio de riesgos para su posterior exploracion
  #para jhours
  #para jnemergency y jndate
  for i in range(len(jn1.get(numnodo)[0]['history'])):
    varauxem=jn1.get(numnodo)[0]['history'][i]['emergency']
    jnemergency.append(varauxem)
    varauxda=jn1.get(numnodo)[0]['history'][i]['date']
    jndate.append(varauxda)
    varauxhr=jn1.get(numnodo)[0]['history'][i]['hour']
    varauxhr=int(varauxhr[0:2])
    jhours.append(varauxhr)
  #preprocesamiento para datos no repetidos en jhours
  jhoursp=[]
  for item in jhours:
      if item not in jhoursp:
          jhoursp.append(item)
  #para jnumsoc
  for item in jhoursp:
    jnumsoc.append(jhours.count(item))#contamos cada hora que se haya presentado por cada 
    #item unico de jhours preprocesado (objetivo: contar emergencias)
  #para jndate yconvertir a datetime
  dtdate=list()
  for y in range(len(jndate)):
    dtdate.append(datetime.datetime.strptime(str(jndate[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  #para listar tipos de emergencia no repetidos y contarlos
  nremerg=[]#emergencias de cada tipo no repetidas
  jnemergency2=list()#una copia para tener una lista de strs separados SOLO EN ARR DE STRINGS
  countemetype=list()#aqui se depositan los numeros de ocurrencia de cada emergencia segun el orden de nremerg
  for a in jnemergency:
    if len(a)>1:
      jnemergency2.append(str(a[0]))
      jnemergency2.append(str(a[1]))
    else:
      jnemergency2.append(str(a[0]))

  for item in jnemergency2:
      if item not in nremerg:
          nremerg.append(item)#para que las emergencias se muestren como unicas
  #para obtener num de datos
  for item in nremerg:
    countemetype.append(jnemergency2.count(item))
  ##print(countemetype)

  return dtdate,jnemergency2,jhoursp,jnumsoc,nremerg,countemetype,jrisks
#en este orden devuelve...
#*fechas en formato datetime (solo fechas, no horas)
#*emergencias por tipo (incluyendo si aplica mas de un tipo de emergencia)formato arreglo de [[],[],...]
#*horas en las que se declararon las emergencias (sin repetirse)
#*numero de ocurrencias de las horas de jhoursp
#*tipo de emergencias en formato string y sin repetirse
#*numero de emergencias por cada tipo (correspondiendo al orden presentado en nremerg)
#*lista de riesgos con sus indices

def obtencionlistasJSGeneral():#obtiene los datos del json aplicado para todos los nodos, no solamente uno
  jnemergency=list()
  jnumsoc=list()#cuenta las ocurrencias de horas
  jhours=list()
  jndate=list()
  jrisks=list()
  for numnodo in strindices:
      varisk=jn1.get(numnodo)[0]['risks'][0]
      jrisks.append(varisk)

  for numnodo in strindices:
    for i in range(len(jn1.get(str(numnodo))[0]['history'])):
      varauxem=jn1.get(str(numnodo))[0]['history'][i]['emergency']
      jnemergency.append(varauxem)
      varauxda=jn1.get(str(numnodo))[0]['history'][i]['date']
      jndate.append(varauxda)
      varauxhr=jn1.get(str(numnodo))[0]['history'][i]['hour']
      varauxhr=int(varauxhr[0:2])
      jhours.append(varauxhr)
  #preproc no repetidos en jhours
  jhoursp=[]
  for item in jhours:
      if item not in jhoursp:
          jhoursp.append(item)
  #para jnumsoc
  for item in jhoursp:
    jnumsoc.append(jhours.count(item))
  #para jndate yconvertir a datetime
  dtdate=list()
  for y in range(len(jndate)):
    dtdate.append(datetime.datetime.strptime(str(jndate[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  #para listar tipos de emergencia no repetidos y contarlos
  nremerg=[]#emergencias de cada tipo no repetidas
  jnemergency2=list()#una copia para tener una lista de strs
  countemetype=list()#aqui se depositan los numeros de ocurrencia de cada emergencia segun el orden de nremerg
  for a in jnemergency:
    if len(a)>1:
      jnemergency2.append(str(a[0]))
      jnemergency2.append(str(a[1]))
    else:
      jnemergency2.append(str(a[0]))

  for item in jnemergency2:
      if item not in nremerg:
          nremerg.append(item)
  for item in nremerg:
    countemetype.append(jnemergency2.count(item))
    jdays=list()#cuenta las ocurrencias de horas
  jnums=list()
  for s in strindices:
    for i in range(len(jn1.get(str(s))[0]['history'])):
      ad=jn1.get(str(s))[0]['history'][i]['date']
      jdays.append(ad)
  jdaystab=list()
  for e in jdays:
    if e not in jdaystab:
      jdaystab.append(e)
  for item in jdaystab:
    jnums.append(jdays.count(item))
  dtdate2=list()
  for y in range(len(jdaystab)):
    dtdate2.append(datetime.datetime.strptime(str(jdaystab[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  return dtdate,jnemergency2,jhoursp,jnumsoc,nremerg,countemetype,jrisks,dtdate2,jnums

def plotgraphline(x,y,color1):#muestra un grafico dada una x, y y un color. Grafico lineal
  infonumcalls=[]
  for i in range(len(x)):
    infonumcalls.append([x[i],y[i]])
  frameinfo=pd.DataFrame(infonumcalls,columns=['Hora','Num. emergencias'])
  if st.checkbox('Mostrar tabla de número de emergencias'):
    st.table(frameinfo)
  st.header('Emergencias acumuladas mostradas por hora de ocurrencia')
  dotgraph = p9.ggplot(data=frameinfo,
                        mapping=p9.aes(x='Hora', y='Num. emergencias'))
  st.pyplot(p9.ggplot.draw(dotgraph + p9.geom_line(color=color1)))

def plotgraphpoints(c,d,color1):#muestra un grafico dada una x, y y un color. Grafico de puntos
  infonumcalls=[]
  for i in range(len(c)):
    infonumcalls.append([c[i],d[i]])
  frameinfo=pd.DataFrame(infonumcalls,columns=['Hora','Num. emergencias'])
  if st.checkbox('Mostrar tabla de número de emergencias'):
    st.table(frameinfo)
  st.header('Emergencias acumuladas mostradas por hora de ocurrencia')
  dotgraph = p9.ggplot(data=frameinfo,
                        mapping=p9.aes(x='Hora', y='Num. emergencias'))

  st.pyplot(p9.ggplot.draw(dotgraph + p9.geom_point(color=color1,alpha=0.5,size=2.7)))
def requestapiRegresionPolinomial(numnodo,hora):
    url = 'http://%s:%s/predictNumEmergencias'%(server,puerto)
    myobj = {'node': numnodo,'hour':int(hora)}
    req = requests.post(url.strip(), json = myobj)
    st.write(req.text)
    st.write("Método: regresión polinomial")
    url2 = 'http://%s:%s/getpng'%(server,puerto)
    reqimg = requests.post(url2.strip(), json = myobj)
    st.image(reqimg.content, caption='Prediccion general',width=700)
    st.success("Se ha completado el análisis para la hora seleccionada")

def regresionlogCalling(dato):
    url = 'http://%s:%s/predictemtype'%(server,puerto)
    myobj = {'test': dato}
    req = requests.post(url.strip(), json = myobj)
    st.write(req.text)
    st.write("Método: regresión Logistica")

## FIN ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
st.sidebar.header("Para iniciar, presione el siguiente botón")



@st.cache(allow_output_mutation=True)
def button_states():
    return {"pressed": None}
botonreload=st.button("Actualizar")


boton = st.sidebar.button("Inicializar modelos")
is_pressed = button_states()  # gets our cached dictionary
if botonreload:
  st.cache(allow_output_mutation=False)#para que a fuerza se tengan que volver a cargar los modelos
  is_pressed.update({"pressed": False})
  #if not server=="localhost":
    #remove('/home/mama/FlaskAPI/modelos/.')
  #else:
    #remove('c:/users/Ariel Yohaina/desktop/mamaduckflask/modelos/.')
if boton:
    # any changes need to be performed in place
    is_pressed.update({"pressed": True})

if is_pressed["pressed"]:  


  ra=requests.get('http://%s:%s/trainNumEmergencias'%(server,puerto))
  rb=requests.get('http://%s:%s/traincentroids'%(server,puerto))

  nodoseleccionado = st.sidebar.selectbox("Seleccione una de las siguientes opciones para efectuar los análisis",
  ("Mama Duck", "Nodo 2", "Nodo 3", "Nodo 4", "General")
  )
  st.sidebar.write("Se han entrenado y actualizado los modelos para implementar funciones de análisis y otros complementos")
  st.sidebar.success('Ahora puede continuar con el análisis')
else:
  st.warning("Presione el botón 'inicializar modelos' en la parte izquierda")
  st.stop()
if nodoseleccionado=='Mama Duck':
  st.header('Análisis de los datos del Nodo Mama Duck')
  a,b,c,d,e,f,g=obtencionlistasJS('1')#obtendre todas las variables que regresa en el orden documentado

  plotgraphpoints(c,d,'blue')#para que haga una grafica de lineas

  user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
  #no se usará la funcion time porque causa conflicto el tipo de datos datetime
  for x in user_input:#cada dato que seleccione el usuario, va a ejecutar el algoritmo de Regresion Lineal
    requestapiRegresionPolinomial("1", int(x))
  st.header('Análisis de emergencias recurrentes')
  if st.button("Obtener análisis de emergencia más probable"):#al presionar el boton ejecuta la regresion Logistica
    regresionlogCalling(prepdatoLRML(e,f,g))

elif nodoseleccionado=='Nodo 2':#opcion siguiente
  st.header('Análisis de los datos del Nodo 2')
  a,b,c,d,e,f,g=obtencionlistasJS('2')
#quiero intentar analizar equis tipo de llamada mas probable
  plotgraphpoints(c,d,'red')
#predTipoEmergencia(x,y)
  #if st.button("Estimar no. de emergencias dada una hora", key=None, 
  #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
  user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
  for x in user_input:
    requestapiRegresionPolinomial("2", int(x))
  st.header('Análisis de emergencias recurrentes')
  if st.button("Obtener análisis de emergencia más probable"):
    regresionlogCalling(prepdatoLRML(e,f,g))

elif nodoseleccionado=='Nodo 3':#opcion 3
    st.header('Análisis de los datos del Nodo 3')
    a,b,c,d,e,f,g=obtencionlistasJS('3')
    plotgraphpoints(c,d,'brown')
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
      requestapiRegresionPolinomial("3", int(x))
    st.header('Análisis de emergencias recurrentes')
    if st.button("Obtener análisis de emergencia más probable"):
      regresionlogCalling(prepdatoLRML(e,f,g))
elif nodoseleccionado=='Nodo 4':#opcion 4
    st.header('Análisis de los datos del Nodo 4')
    a,b,c,d,e,f,g=obtencionlistasJS('4')
    plotgraphpoints(c,d,'purple')
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
      requestapiRegresionPolinomial("4", int(x))
    st.header('Análisis de emergencias recurrentes')
    if st.button("Obtener análisis de emergencia más probable"):
      regresionlogCalling(prepdatoLRML(e,f,g))

elif nodoseleccionado=='General':#si se selecciona un analisis gneral
  st.header('Análisis general de la red Mama Duck')
  a,b,c,d,e,f,g,h,i=obtencionlistasJSGeneral()
  #x=fecha, y=emergencias sin repetir
  plotgraphline(c,d,'red')
  user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
  for x in user_input:
      requestapiRegresionPolinomial('general', int(x))
  listdt=list()
 #i y h tienen los datos necesarios de fecha y numero de repeticiones menesteres para el siguiente procedimiento
  for x in range(len(h)):
    stra=str(h[x])
    listdt.append([stra[5:10],i[x]])
  df=pd.DataFrame(listdt,columns=['Fecha','Emergencias'])
  st.header("Historial de emergencias por fecha")
  dotgraph = p9.ggplot(data=df,
                          mapping=p9.aes(x='Fecha', y='Emergencias'))

  st.pyplot(p9.ggplot.draw(dotgraph +p9.geom_point(color='green',alpha=0.5,size=2.7)))
  st.header('Análisis de emergencias recurrentes globales')
  if st.button("Análisis de emergencia más probable"):
    regresionlogCalling(prepdatoLRMLGen(e,f,g))
  st.header("Mapa de nodos y emergencias")
  st.write("Hexagonos: ubicacion de nodos, Círculo verde: casos de emergencia")
  df1=obtencionCoords()
  if st.checkbox('Mostrar tabla de coordenadas de emergencia'):
    st.write(df1)
  rmap = requests.get('http://%s:%s/map'%(server,puerto))
  
  components.html(rmap.text,height=900,width=1000)
  st.title("Análisis de clustering para predecir centroides")
  drawcentroids = requests.get('http://%s:%s/obtenerCentroidesgraph'%(server,puerto))
  st.image(drawcentroids.content, caption='Prediccion general',width=800)
  st.header("Una vez obtenidos los centroides señalados en estrellas, se mostrará la ubicación precisa de los mismos en el plano (Hexagonos) junto con las llamadas (puntos verdes)")
  setcentroids = requests.get('http://%s:%s/clustercentroids'%(server,puerto))
  components.html(setcentroids.text,height=900,width=900)
  st.header("Coordenadas de los nodos propuestos")
  dfcap = requests.get('http://%s:%s/obtenerCentroidesrender'%(server,puerto))
  st.image(dfcap.content, caption='Coordenadas propuestas, método Kmeans',width=500)
  st.title("Nodos receptores Mama Duck")
  st.write("A continución se muestra la distribución de nodos de Mama Duck, sus interconexiones e información")

  r = requests.get('http://%s:%s/nodes'%(server,puerto))
  
  components.html(r.text,height=400,width=1500)

  st.header("Monitoreo de nodos")
  #aqui empieza la implementacion de recorrido de Grafo
  tag,conections,parent,mamaduck,people,timeactive,list1=DataJSONtoGraph('1')#con el nodo 1 se inicia para empezar a
  #crear los demas grafos
  n1=Nodo(tag,conections,parent,mamaduck,people,timeactive,list1)
  res,restab=recorridonodos(n1)#iniciamos el algoritmo de A estrella en una variable de objeto
  st.write("El Nodo: %d "%res[1],"requiere ser monitoreado después del nodo principal Mama Duck, consulte la tabla para mas detalles")#en res 1 está el nodo elegido como prioritario, esta función imprimirá el porqué primero este y porque los demas
  st.header("Orden de prioridad")
  st.header(res)
  dfcap3 = requests.get('http://%s:%s/nodestrajectory'%(server,puerto))
  st.image(dfcap3.content, caption='Detalles de recorrido',width=700)