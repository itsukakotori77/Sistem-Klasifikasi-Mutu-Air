# Library Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

# Package
from flask import Flask, render_template, request, make_response, redirect, Response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_jsonpify import jsonpify

# Import Custom Package
from knn.knn import KNN


# Import File
import os
import pandas as pd
import numpy as np
import mysql.connector
import json
import io
import csv

# Mysql Connector
from mysql.connector import Error
from mysql.connector import errorcode

class DataBaru:

    # Function
    @staticmethod
    def index():

        if request.method == 'POST':

            # Variable
            Warna = request.form['Warna']
            TDS = request.form['TDS']
            Kekeruhan = request.form['Kekeruhan']
            Suhu = request.form['Suhu']
            E_Coli = request.form['E_Coli']
            Total_Koliform = request.form['Total_Koliform']
            Fluorida = request.form['Fluorida']
            Total_Kromium = request.form['Total_Kromium']
            Nitrit = request.form['Nitrit']
            Nitrat = request.form['Nitrat']
            Sianida = request.form['Sianida']
            Aluminium = request.form['Aluminium']
            Besi = request.form['Besi']
            Kesadahan = request.form['Kesadahan']
            Khlorida = request.form['Khlorida']
            Mangan = request.form['Mangan']
            PH = request.form['PH']
            Seng = request.form['Seng']
            Sulfat = request.form['Sulfat']
            Tembaga = request.form['Tembaga']
            Amonia = request.form['Amonia']
            Zat_Organik = request.form['Zat_Organik']
            Karbondioksida = request.form['Karbondioksida_Bebas']
            Alkalinitas = request.form['Alkalinitas_Total']

            # Nilai K
            K = request.form['Nilai_K']
            knn = KNeighborsClassifier(n_neighbors=int(K))

            if 'Warna' not in request.form:
                prediksi = 0
            else:
                # Query Dataset
                connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
                dataset = pd.read_sql('SELECT * FROM dataset', connection)
                dataset.fillna(dataset.mean(), inplace=True)

                # Memisahkan label dan dataset
                #Create x and y variables.
                X = dataset.drop(columns=['Label', 'id'])
                y = dataset['Label']

                # split datatrain and datatest 8:2
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

                # Fit x_test with y_test
                knn.fit(X_train, y_train)
                y_test = knn.predict(X_test)
                
                # Create Model & resolve missing value
                scaler = StandardScaler()  
                scaler.fit(X_train)
                x_train = scaler.transform(X_train)  
                x_test = scaler.transform(X_test)
                knn.fit(x_train, y_train)

                # Predict KNN
                prediksiData = np.array([[ Warna, 
                                            TDS, 
                                            Kekeruhan, 
                                            Suhu, 
                                            E_Coli, 
                                            Total_Koliform, 
                                            Fluorida, 
                                            Total_Kromium, 
                                            Nitrit, 
                                            Nitrat, 
                                            Sianida, 
                                            Aluminium, 
                                            Besi, 
                                            Kesadahan, 
                                            Khlorida, 
                                            Mangan, 
                                            PH, 
                                            Seng, 
                                            Sulfat, 
                                            Tembaga, 
                                            Amonia, 
                                            Zat_Organik, 
                                            Karbondioksida, 
                                            Alkalinitas 
                                        ]])
                                
                prediksi = knn.predict(prediksiData)
                prediksiList = prediksi.tolist()
                jsonPrediksiDump = json.dumps(prediksiList)
                jsonPrediksi = jsonPrediksiDump.strip('[""]')

                # Insert to Prediksi
                # insertDataWater(Warna, TDS, Kekeruhan, Suhu, E_Coli, Total_Koliform, Fluorida, Total_Kromium, Nitrit, Nitrat, Sianida, Aluminium, Besi, Kesadahan, Khlorida, Mangan, PH, Seng, Sulfat, Tembaga, Amonia, Zat_Organik, Karbondioksida, Alkalinitas, jsonPrediksi)

                return jsonPrediksi

    @staticmethod
    def store():
        # Check Variable
        if 'Warna' not in request.form:
            return Response('Gagal')
        else:
            Warna = request.form['Warna']
            TDS = request.form['TDS']
            Kekeruhan = request.form['Kekeruhan']
            Suhu = request.form['Suhu']
            E_Coli = request.form['E_Coli']
            Total_Koliform = request.form['Total_Koliform']
            Fluorida = request.form['Fluorida']
            Total_Kromium = request.form['Total_Kromium']
            Nitrit = request.form['Nitrit']
            Nitrat = request.form['Nitrat']
            Sianida = request.form['Sianida']
            Aluminium = request.form['Aluminium']
            Besi = request.form['Besi']
            Kesadahan = request.form['Kesadahan']
            Khlorida = request.form['Khlorida']
            Mangan = request.form['Mangan']
            PH = request.form['PH']
            Seng = request.form['Seng']
            Sulfat = request.form['Sulfat']
            Tembaga = request.form['Tembaga']
            Amonia = request.form['Amonia']
            Zat_Organik = request.form['Zat_Organik']
            Karbondioksida = request.form['Karbondioksida_Bebas']
            Alkalinitas = request.form['Alkalinitas_Total']
            Label = request.form['Label']

            # Simpan dalam database
            # insertDataWater(Warna, TDS, Kekeruhan, Suhu, E_Coli, Total_Koliform, Fluorida, Total_Kromium, Nitrit, Nitrat, Sianida, Aluminium, Besi, Kesadahan, Khlorida, Mangan, PH, Seng, Sulfat, Tembaga, Amonia, Zat_Organik, Karbondioksida, Alkalinitas, Label)

            return Response('Berhasil')