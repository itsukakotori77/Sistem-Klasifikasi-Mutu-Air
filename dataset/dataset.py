# Package
from flask import Flask, render_template, request, make_response, redirect, Response

# Import Mysql
import mysql.connector

# Mysql Connector
from mysql.connector import Error
from mysql.connector import errorcode


class DataLatih:

    # Function
    @staticmethod
    def index():
        # Select Data
        connection = mysql.connector.connect(user='root', password='', host='localhost', database='sistem_prediksi')
        sql_select_Query = "SELECT * FROM dataset"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()

        # Varible Local
        return render_template('/DataLatih/index.html', record=records)

    #