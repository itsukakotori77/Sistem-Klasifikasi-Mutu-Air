
# Library Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Package
from flask import Flask, render_template, request, make_response, redirect, Response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_jsonpify import jsonpify

# Class View
from flask.views import View
from flask_appbuilder import AppBuilder, expose, BaseView

# Class KNN
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

# Name App
app = Flask(__name__)

# Path to .env file
# dotenv_path = join(dirname(__file__), '.env')  
dotenv_path = '.env'  
load_dotenv(dotenv_path)

# Konfigurasi File Upload
ALLOWED_EXTENSION = set(['json', 'csv'])
app.config['UPLOAD_FOLDER'] = 'uploads'

# Function & Route
@app.route('/')
def index():
    return render_template('index.html')

# Fungsi Insert Data Water
def insertDataWater(Warna, TDS, Kekeruhan, Suhu, E_Coli, Total_Koliform, Fluorida, Total_Kromium, Nitrit, Nitrat, Sianida, Aluminium, Besi, Kesadahan, Khlorida, Mangan, PH, Seng, Sulfat, Tembaga, Amonia, Zat_Organik, Karbondioksida, Alkalinitas, Label):
    try:
        # Connection
        connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
        mySql_insert_query = """INSERT INTO prediksi (Warna, TDS, Kekeruhan, Suhu, E_Coli, Total_Koliform, Fluorida, Total_Kromium, Nitrit, Nitrat, Sianida, Aluminium, Besi, Kesadahan, Khlorida, Mangan, PH, Seng, Sulfat, Tembaga, Amonia, Zat_Organik, Karbondioksida_Bebas, Alkalinitas_Total, Label)
                                VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        val = (Warna, TDS, Kekeruhan, Suhu, E_Coli, Total_Koliform, Fluorida, Total_Kromium, Nitrit, Nitrat, Sianida, Aluminium, Besi, Kesadahan, Khlorida, Mangan, PH, Seng, Sulfat, Tembaga, Amonia, Zat_Organik, Karbondioksida, Alkalinitas, Label)

        # Cursor Mysql
        cursor = connection.cursor()
        cursor.execute(mySql_insert_query, val)
        connection.commit()
        cursor.close()

    finally:
        if (connection.is_connected()):
            connection.close()


# Dashboard
class Dashboard:
    
    # Route Home
    @staticmethod
    @app.route('/home')
    def home():
        return render_template('/Dashboard/home.html')

# Class Dataset
class Dataset:

    # Dataset
    @staticmethod
    @app.route('/datalatih', methods=['GET', 'POST'])
    def datalatihIndex():

        # Select Data
        connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
        sql_select_Query = "SELECT * FROM dataset"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()
                
        return render_template('/DataLatih/index.html', record=records)

# Class Databaru
class DataBaru:

    # View Databaru
    @staticmethod
    @app.route('/databaru', methods=['GET', 'POST'])
    def databaruIndex():

            # Check Request
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
                    knn.fit(X,y)
                    # knn.fit(x_train, y_train)

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
                    
            else:
                # Query Dataset
                connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
                dataset = pd.read_sql('SELECT * FROM dataset', connection)
                dataset.fillna(dataset.mean(), inplace=True)

                #Create x and y variables.
                X = dataset.drop(columns=['Label', 'id'])
                y = dataset['Label']

                # split datatrain and datatest
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
            
                # Check Data test to Train
                knn = KNeighborsClassifier()
                knn.fit(X_train, y_train)
                y_test = knn.predict(X_test)

                # Detect Missing Value and Insert it
                scaler = StandardScaler()  
                scaler.fit(X_train)

                x_train = scaler.transform(X_train)  
                x_test = scaler.transform(X_test)
                
                # Error Variable
                error = []

                # Calculating error for K values between 1 and 40
                for i in range(1, 40):  
                    knn = KNeighborsClassifier(n_neighbors=i)
                    knn.fit(x_train, y_train)
                    pred_i = knn.predict(x_test)
                    error.append(np.mean(pred_i != y_test))

                return render_template('/DataBaru/index.html', error=error)


    # Simpan dalam database
    @staticmethod
    @app.route('/databaru/simpan', methods=['POST'])
    def dataBaruSimpan():

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
            insertDataWater(Warna, TDS, Kekeruhan, Suhu, E_Coli, Total_Koliform, Fluorida, Total_Kromium, Nitrit, Nitrat, Sianida, Aluminium, Besi, Kesadahan, Khlorida, Mangan, PH, Seng, Sulfat, Tembaga, Amonia, Zat_Organik, Karbondioksida, Alkalinitas, Label)

            return Response('Berhasil')

# Class Hasil Uji Data Baru
class HasilDataBaru:

    # Hasil Data Uji
    @staticmethod
    @app.route('/hasildatabaru', methods=['GET', 'POST'])
    def indexHasilDataBaru():

        #  Query prediksi   
        connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
        sql_select_Query = "SELECT * FROM prediksi"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()

        # return render_template('/HasilDataBaru/index.html', recordLD=recordsPrediksiLD, recordTLD=recordsPrediksiTLD)
        return render_template('/HasilDataBaru/index.html', record=records)

    # Hasil uji data baru
    @staticmethod
    @app.route('/hasildatabaru/detail/<id>', methods=['GET'])
    def detailHasilUji(id):

        # Select Data
        connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
        sql_select_Query = "SELECT * FROM prediksi WHERE id = %s"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query, (id,))
        records = cursor.fetchall()

        return render_template('/HasilDataBaru/detail.html', record=records)

    # Download CSV
    @staticmethod
    @app.route('/convert/csv', methods=['GET'])
    def convertCSV():
        
        # Try Catch
        try:
                connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM prediksi")
                result = cursor.fetchall()

                output = io.StringIO()
                writer = csv.writer(output)
                
                line = [
                            'ID',
                            'Warna', 
                            'TDS', 
                            'Kekeruhan', 
                            'Suhu', 
                            'E.Coli', 
                            'Total Koliform', 
                            'Fluorida', 
                            'Total Kromium', 
                            'Nitrit', 
                            'Nitrat', 
                            'Sianida', 
                            'Aluminium', 
                            'Besi', 
                            'Kesadahan', 
                            'Khlorida', 
                            'Mangan', 
                            'pH', 
                            'Seng', 
                            'Sulfat', 
                            'Tembaga', 
                            'Amonia', 
                            'Zat Organik', 
                            'Karbondioksida Bebas', 
                            'Alkalinitas Total',
                            'Label'
                        ]
                writer.writerow(line)

                # Loop Data
                for row in result:
                    line = [str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + ',' + str(row[4]) + ',' + str(row[5]) + ',' + str(row[6]) + ',' + str(row[7]) + ',' + str(row[8]) + ',' + str(row[9]) + ',' + str(row[10]) + ',' + str(row[11]) + ',' + str(row[12]) + ',' + str(row[13]) + ',' + str(row[14]) + ',' + str(row[15]) + ',' + str(row[16]) + ',' + str(row[17]) + ',' + str(row[18]) + ',' + str(row[19]) + ',' + str(row[20]) + ',' + str(row[21]) + ',' + str(row[22]) + ',' + str(row[23]) + ',' + str(row[24]) + ',' + str(row[25])]
                    writer.writerow(line)

                # Output
                output.seek(0)
                
                # Return Value
                return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=dataset_report.csv"})

        finally:
            cursor.close() 
            connection.close()










    

