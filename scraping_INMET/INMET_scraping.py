#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Bibliotecas já instaladas no ambiente Python
import os
import time

#Importar as funções que iremos utilizar do Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

#Importando BeautifulSoup
from bs4 import BeautifulSoup

import pandas as pd


# In[ ]:


VALOR_ESTACAO='A101'

DATA_INIT='01/01/2023'
DATA_END='06/08/2023'


# In[12]:


options = webdriver.ChromeOptions()
options.add_argument('--headless')
#options.add_argument('--no-sandbox')
#options.add_argument('--disable-dev-shm-usage')
#options.add_argument('--no-startup-window')


site = 'https://tempo.inmet.gov.br/TabelaEstacoes/A101'
wb = webdriver.Chrome('chromedriver',chrome_options= options)

wb.get(site)
#Recomendado para evitar ban do servidor
time.sleep(3)

#Selecionar em "Produto" a opção "Tabela de Dados das Estações" 
wb.find_element_by_xpath('//*[@id="root"]/div[1]/div[1]/i').click()


wb.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div[2]/div[4]/input').clear()
wb.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div[2]/div[4]/input').send_keys(DATA_INIT)

#O mesmo para a data final

wb.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div[2]/div[5]/input').clear()
wb.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div[2]/div[5]/input').send_keys(DATA_END)

#Por fim, clicamos em "gerar tabela"
wb.find_element_by_xpath("/html/body/div[1]/div[2]/div[1]/div[2]/button").click()


#Pedimos para o Selenium aguardar por alguns segundos até que a tabela seja gerada pelo site
WebDriverWait(wb, 10).until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/table")))

#atribuimos a estrutura atual do site para uma variável para que o BeautifulSoup possa fazer sua mágica!
page_source = wb.page_source

wb.close()
df = pd.read_html(page_source)[0]
df

