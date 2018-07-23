# -*- coding: utf-8 -*-
"""
@author: cordula eggerth, dariga ramazanova, lusine yeghiazaryan

abschlussbeispiel: k-means  

"""


import pandas as pd
from itertools import repeat
import random as rd
import math
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# -----------------------------------------------------------------------------
# TEST-DATEN AUS CSV-FILE faithful.csv EINLESEN & DATAFRAME ANLEGEN
# -----------------------------------------------------------------------------
# daten: eruptions (col1), waiting (col2)
path = "C:/Users/cordu/Desktop/faithful.csv"
# path = "C:/Users/lusin/OneDrive/Desktop/Abschlussarbeit CS/faithful.csv"
# path = "C:/faithful.csv"
daten = pd.read_csv(path, sep=",", decimal=".") 
daten.head()

## TEST-VARIABLEN
# k = 3
# x = daten
# trace = False
# maxiter = 10 
# change = 0.01


# -----------------------------------------------------------------------------
# K-MEANS FUNKTION
# -----------------------------------------------------------------------------

'''
  input-parameter:
    x ... dataframe (dimension nx2, d.h. n rows, 2 cols)
    k ... anzahl der zu bildenden teilgruppen
    trace=FALSE ... falls TRUE, dann soll für jeden zwischenschritt eine grafik produziert werden
    maxiter=10 ... maximale anzahl von iterationsschritten
    change=0.001 ... abbruchswert für die relative änderung

  output (list):
    iter ... anzahl d er durchgeführten iterationsschritte
    zentren ... matrix der dimension kx2 (enthält für jede teilmenge die mittelwerte der beiden variablen)
    index ... vektor der länge n (enthält für jeden datenpunkt info, zu welcher teilmenge er gehört)
    distanz ... vektor der länge n (enthält für jeden datenpunkt die distanz zum zentrum seiner teilmenge)
'''

def kmeans(x=daten, k=3, trace=False, maxiter=10, change=0.001):
  # OUTPUT INITIALISIERUNG:
  iter = 0
  zentren = None
  index = list(repeat(-1, len(x)))
  distanz = list(repeat(-1, len(x)))
  colnames = x.columns.values.tolist()
  outputliste = list()
    
  # weitere intialisierungen:
  distanzensumme = 0
  relativeAenderung_DistanzenSumme = change+1
  index_groupMeans = list(range(0, k, 1))
  cols_groupMeans = ["xvalue", "yvalue"]
  group_means = pd.DataFrame(index=index_groupMeans, columns=cols_groupMeans)
  group_means = group_means.fillna(0) # intialisiere mit 0 
  
  if (trace):
      mpl.style.use('seaborn')
      number_of_colors = k
      
      color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
      print(color)
      colours = np.array(color)
  
  
  # SCHRITT 1: 
  # wähle zufällig k punkte aus beobachtungen als startlösung für die gruppenmittelwerte
  # stelle sicher, dass keine identischen beobachtungspaare unter den k ausgewählten punkten
  if(k > len(x)):
      try:      
          raise Exception("k muss kleiner als anzahl der beobachtungen sein!")     
      except Exception as e:
          k=x-1


  x_unique = x.drop_duplicates() # duplikate herausnehmen
  randomStartingRowIndices = rd.sample(list(range(1, len(x_unique), 1)), k=k)
  randomStartingPoints = x_unique.loc[randomStartingRowIndices, ]
  randomStartingPoints = randomStartingPoints.reset_index() # ACHTUNG: clusternamen sind 0 bis (k-1)

# plot wenn trace ist TRUE
  if (trace):
      plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c = "xkcd:mauve",marker = "p",alpha = 0.6)
      plt.scatter(randomStartingPoints.iloc[:, 1], randomStartingPoints.iloc[:, 2],c = colours[0:k], marker = "X", s= 100, alpha = 0.9)
      plt.title("Starting points")
      plt.show()

  # SCHRITT 2:
  # bestimme für jeden punkt die euklidschen distanzen zu den aktuellen gruppenmittelwerten
  while(iter < maxiter and relativeAenderung_DistanzenSumme >= change): # check abbruchbedingungen 
                                                                         # (i.e. schritt 4)
        for i in range(0,len(x)): # i ... anzahl der beobachtungen
            
            distanzenZuClustern_proBeobachtung = list(repeat(float(0), k))
            
            for j in range(0,k): # j ... anzahl der k zu bildenden gruppen
                                
                ## berechne euklidsche distanzen
                if(iter==0): 
                    distanzenZuClustern_proBeobachtung[j] = math.sqrt( (float(x.iloc[i,0])-float(randomStartingPoints.iloc[j,1]))**2 + (float(x.iloc[i,1])-float(randomStartingPoints.iloc[j,2]))**2 )
                else: 
                    distanzenZuClustern_proBeobachtung[j] = math.sqrt( (float(x.iloc[i,0])-float(group_means.iloc[j,0]))**2 + (float(x.iloc[i,1])-float(group_means.iloc[j,1]))**2 )
                
            ## setze distanz (distanz zum gewählten cluster-mittelpunkt)
            distanz[i] = min(distanzenZuClustern_proBeobachtung)
            
            ## setze index (clusterzuordnung gemäß minimaler distanz)
            bool_isMin = False
            minDistanzClusterNummer_proBeobachtung = -1
            
            for l in range(0,k):
                bool_isMin = distanzenZuClustern_proBeobachtung[l] == min(distanzenZuClustern_proBeobachtung)
                if(bool_isMin):
                    minDistanzClusterNummer_proBeobachtung = l
                    break
            index[i] = minDistanzClusterNummer_proBeobachtung # ACHTUNG: clusternummern starten bei 0
            # CHECK: print("index i", index[i])    

        ## distanzsumme und relative änderung davon in laufender iteration
        if(iter!=0): 
            relativeAenderung_DistanzenSumme = abs(distanzensumme - sum(distanz)) / distanzensumme
        
        distanzensumme = sum(distanz)
        
        ## SCHRITT 3: bestimme aufgrund von aktueller gruppenzugehörigkeit der datenpunkte für jede 
        ##            der k gruppen durch anwendung von "mean" neue gruppenmittelwerte
        
        # neue col mit index (i.e. info über clusternummer pro beobachtung) an dataframe dranhängen
        index_series = pd.Series(index)
        x['cluster'] = index_series.values
        
        # pro cluster neuen mittelwert bilden
        for a in range(0,k):
            try:
                group_means.iloc[a,0] = mean((x[x.cluster == a]).iloc[::,0])
                group_means.iloc[a,1] = mean((x[x.cluster == a]).iloc[::,1])
            except:
                group_means.iloc[a,0] = 0
                group_means.iloc[a,1] = 0
        
        if (trace): 
            plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c = colours[index], marker = "p", alpha = 0.6)
            plt.scatter(group_means.iloc[:, 0], group_means.iloc[:, 1], 
                        c = colours[0:k], marker = "X", linewidths = 3, s= 100, alpha = 0.9)
            plt.title("Iteration" + str(iter + 1), fontsize = 14)
            plt.show()        
            
        ## iterationsschritte-anzahl erhöhen
        iter = iter + 1
        
        
   # setze inhalte der outputliste
  zentren = group_means
  outputliste.append(iter)
  outputliste.append(zentren)
  outputliste.append(index)
  outputliste.append(distanz)    
  
  # ergebnisse ausgebenv
  print("\n Iterationsschritte: ", outputliste[0], "\n \n", 
      "Zentren: ", outputliste[1], "\n  \n", 
      "Index (Cluster): ", outputliste[2], "\n \n",
      "Distanzen: ", outputliste[3], "\n  \n")
  print("Anmerkung: Falls Clusternummern ohne Punkte vorkommen, werden die Zentren-Koordinaten des betroffenen Cluster jeweils auf 0.0 gesetzt.")

        
  ## RETURN list of output
  return outputliste

    

# TEST AUFRUFE DER FUNCTION kmeans
ergebnis_fall0 = kmeans()
ergebnis_fall1 = kmeans(daten, 10, False, 10, 0.001)  
ergebnis_fall2 = kmeans(daten, 4, False, 10, 0.01) 

ergebnis_fall3 = kmeans(daten, 7, True, 10, 0.001) 


##  d) Überprüfen Sie die Funktionalität mit einem simulierten Datensatz:
#generieren Sie 4 Stichproben mit je 25 Beobachtungen (insgesamt n=100), und folgenden
#Mittelwerten (-1,1), (-1,-1), (1,1), (1,-1), die Werte für die beiden Variablen sollen jeweils 
#um den Mittelwert normalverteilt mit Standardabweichung 1 sein.
#ok = {"xvalue":np.random.normal(-1,1,25),"yvalue": np.random.normal(1,1,25)}
stichprobe1 = pd.DataFrame(data = {"xvalue":np.random.normal(-1,1,25),"yvalue": np.random.normal(1,1,25)})
stichprobe2 = pd.DataFrame(data = {"xvalue":np.random.normal(-1,1,25),"yvalue": np.random.normal(-1,1,25)})
stichprobe3 = pd.DataFrame(data = {"xvalue":np.random.normal(1,1,25),"yvalue": np.random.normal(1,1,25)})
stichprobe4 = pd.DataFrame(data = {"xvalue":np.random.normal(1,1,25),"yvalue": np.random.normal(-1,1,25)})
#rbind wie im R
temp1 =  stichprobe1.append(stichprobe2, ignore_index = True)
temp2 = temp1.append(stichprobe3, ignore_index =True)
RandomData = temp2.append(stichprobe4, ignore_index= True)

ergebniss_mitRandomData = kmeans(RandomData, 18, True, 10, 0.001)


# =======================================================================
#  Optionale Mehrleistungen:  Silhouetten-Plot
# =======================================================================

'''
 Diese Funktion berechnet die Silhouettenwerte & Silhouettenkoeffizienten 
 von einem gruppierten Datensatz. Die Silhouettenwerte werden mithilfe eines
 Silhouettenplots dargestellt.

 Silhouettenwerte werden folgenderweise berechnet:
         s(i) = (b[i] - a[i])/max{a[i], b[i]}, wobei
 a ... durchschnittliche distanz zwischen jedem punkt und allen restlichen 
       punkten im selben cluster
 b ... minimale mittlere distanz von jedem punkt i zu allen anderen punkten in 
       einem anderen cluster, in dem i nicht liegt.

 input-parameter:
   x ... dataframe (dimension nx2, d.h. n rows, 2 cols)
   erg ... list mit Ergebnissen vom KMEANS-Algorithmus


 output (list):
   vektor der laenge n mit Silhoettenwerten
   summary von Silhouettenwerten
   Silhouettenkoeffizienten von Clusters
 und 
   barplot von Silhouettenwerten
   plot von geclusterten Punkten
   
'''
ergebnis = kmeans(daten, 7, False, 10, 0.001)

def silhouetten(x = daten, erg = ergebnis):
    
    outputliste = list()
    
    # Anzahl der Punkte
    n = len(x.index)
    # Vektor von Clusternummern
    ind = np.array(erg[2])
    # Anzahl der Cluster
    k = len(np.unique(ind))
    # Initialisierung
    silhouetten_werte = np.zeros(n)
    
    
    for i in range(0, n):
        # mittlere Distanz von einem Punkt bis zur anderen in demselben Cluster
        bool_i = ind == ind[i]
        
        dist_within = ( (x.iloc[i,0] - x.iloc[bool_i,0])**2 + (x.iloc[i,1] - x.iloc[bool_i,1])**2 ) ** 0.5
        a = mean(dist_within[dist_within != 0])
        
        # kleinste mittlere Distanz von einem Punkt bis zur anderen in unterscheidenden Clustern
        dist_nextgroup = np.zeros(k)
        vek = np.array(range(0, k))
        
        for j in vek[vek != ind[i]]:
            bool_j = ind == j
            dist_nextgroup[j]  = mean(( (x.iloc[i,0] - x.iloc[bool_j,0])**2 + (x.iloc[i,1] - x.iloc[bool_j,1])**2 ) ** 0.5)
        
        b = min(dist_nextgroup[dist_nextgroup != 0])
        
        silhouetten_werte[i] = (b - a)/max(a, b)
    
    # Dataframe fuer Plot und Berechnung von Koeffizienten
    yy = pd.DataFrame({"cluster": ind, "werte": silhouetten_werte})
    yy = yy.sort_values(by = ['cluster', 'werte'])
    
    # Silhouettenkoeffiziente
    n_in_cluster = yy.groupby(["cluster"]).agg({'werte': np.size})
    sil_koef = yy.groupby(["cluster"]).agg({'werte': np.mean})
    
    # Vorbereitung fuer Plots
    mpl.style.use('seaborn')
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(k)]
    print(color)
    colours = np.array(color)
    
    # labels fuer legend      
    lbs = list()
    for i in range(k):
        txt = "{0} cluster: {1} | {2}".format(i+1, n_in_cluster.iloc[i, 0], round(sil_koef.iloc[i, 0], 3))
        lbs.append(txt)
    
    #Silhouettenplot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = False, sharex = False, figsize = (16,8))

    y_pos = np.arange(n)
    for i in range(k):
        bool_1 = yy['cluster'] == i
        ax1.barh(y_pos[bool_1], yy['werte'].loc[bool_1], color = colours[i], linewidth = 0, 
                     label = lbs[i], alpha = 0.7)
            
    ax1.set_title("Silhouettenplot", fontsize = 14)
    ax1.set_xlabel("Silhouettenwerte", fontsize = 12)
    ax1.set_xlim(min(yy['werte']), 1.5)
    ax1.legend(title = "Silhouettenkoeffizienten")
    
    # Geclusterte Punkte    
    zentren = erg[1]
    ax2.scatter(x.iloc[:, 0], x.iloc[:, 1], c = colours[ind], marker = "p", alpha = 0.6)
    ax2.scatter(zentren.iloc[:, 0], zentren.iloc[:, 1], 
                                c = colours[0:k], marker = "X", linewidths = 3, s= 100, alpha = 0.9)
    ax2.set_title("Clustering points", fontsize = 14)
        
    plt.show()    


    print("\n Silhouettenwerte: \n ", silhouetten_werte, "\n \n", 
      "Summary von Silhouettenwerten: \n ", yy['werte'].describe(), "\n  \n", 
      "Silhouettenkoeffizienten: \n", sil_koef, "\n \n")
    
    outputliste.append(silhouetten_werte)
    outputliste.append(yy['werte'].describe())
    outputliste.append(sil_koef)
    return outputliste

ergebnis_faith = kmeans(daten, 7, False, 10, 0.001)
ergebnis_faith[0] 
silhouetten(erg = ergebnis_faith)

ergebnis_RD = kmeans(RandomData, 4, False, 10, 0.001)
ergebnis_RD[0] 
silhouetten(x = RandomData, erg = ergebnis_RD)
