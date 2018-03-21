# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    return np.array([[np.count_nonzero(a!=b) for a in X_train.toarray()] for b in X.toarray()])
    #pass


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    return(y[Dist.argsort(kind='mergesort')])    
    #pass


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    result = np.zeros(shape=(y.shape[0],4))
    y=np.delete(y,range(k,y.shape[1]), axis=1) #take k closest
    for i in range(y.shape[0]):
        result[i] = np.bincount(y[i],minlength=4)
    return np.divide(result,k)
    #pass


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    result = 0
    p_y_x = np.fliplr(p_y_x) #class 3 first, to choose highest first
    for i in range(y_true.shape[0]):
        if(np.argmax(p_y_x[i])!=(3-y_true[i])): #3-true because we flipped the matrix
            result+=1
    return result/p_y_x.shape[0]
    #pass


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    labeled = sort_train_labels_knn(hamming_distance(Xval,Xtrain), ytrain)
    best_k = k_values[0]
    best_error = classification_error(p_y_x_knn(labeled,k_values[0]), yval)
    errors = []
    errors.append(best_error)
    for i in range(1,len(k_values)):
        curr_error = classification_error(p_y_x_knn(labeled, k_values[i]), yval)
        errors.append(curr_error)
        if(curr_error<best_error):
            best_error = curr_error
            best_k = k_values[i]
    return(best_error, best_k, errors)
    #pass


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    counts = np.bincount(ytrain)
    return counts/ytrain.shape[0]
    #pass


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    #ogarnac to
    Xtrain = Xtrain.A
    pxy = np.empty((4,Xtrain.shape[1]))
 
    for k in range(4):
        d = np.nonzero(ytrain == k)[0]
        counts = np.count_nonzero(Xtrain[d, :] == 1, axis=0)
        pxy[k,:] = (counts + a - 1) / (d.shape[0] + a + b - 2)
    #print(p_x_y)
    return pxy
    #pass


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X=X.A
    pyx = np.zeros(shape=(X.shape[0], 4))
    
    for i in range(X.shape[0]):
        for j in range(4):
            pyx[i][j] = np.prod( np.power(p_x_1_y, X[i,:]) * np.power((1-p_x_1_y),(1-X[i,:])), axis=1)[j] * p_y[j]
        pyx[i] /= np.sum(pyx[i])
    return pyx
    #pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    error_best = classification_error(p_y_x_nb(estimate_a_priori_nb(ytrain), estimate_p_x_y_nb(Xtrain, ytrain, a_values[0], b_values[0]), Xval), yval)
    best_a = a_values[0]
    best_b = b_values[0]
    errors = np.zeros(shape=(len(a_values),len(b_values)))
    for i in range(len(a_values)):
        for j in range(len(b_values)):
            curr_error = classification_error(p_y_x_nb(estimate_a_priori_nb(ytrain), estimate_p_x_y_nb(Xtrain, ytrain, a_values[i], b_values[j]), Xval), yval)
            errors[i][j] = curr_error
            if(curr_error<error_best):
                error_best = curr_error
                best_a = a_values[i]
                best_b = b_values[j]
    return(error_best, best_a, best_b, errors)
    #pass

