from selenium import webdriver
from selenium.webdriver.common.by import By

def back_translate(text):
    driver = webdriver.Chrome(executable_path='D:\Class\KTPM\TH\chromedriver.exe')
    driver.get("https://translate.google.com/?hl=vi&tab=TT&sl=vi&tl=en&op=translate")

    inp = driver.find_element(By.TAG_NAME, 'textarea')
    inp.send_keys(text)

    driver.implicitly_wait(10)

    en_output = driver.find_element(By.CSS_SELECTOR, '.dePhmb .eyKpYb .J0lOec span.VIiyi').text

    driver.get("https://translate.google.com/?hl=vi&tab=TT&sl=en&tl=vi&op=translate")

    inp = driver.find_element(By.TAG_NAME, 'textarea')
    inp.send_keys(en_output)

    driver.implicitly_wait(10)

    vi_output = driver.find_element(By.CSS_SELECTOR, '.dePhmb .eyKpYb .J0lOec').text


    driver.close()

    return vi_output

if __name__ == '__main__':
    print(back_translate('chào'))

