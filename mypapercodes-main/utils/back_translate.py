from selenium import webdriver
from selenium.webdriver.common.by import By
# from preprocess.preprocess import remove_underline
# from preprocess.vi_news_preprocess import remove_punctuation_characters, remove_whitespace
# from preprocess.vi_sentiment_preprocess import (
#     handle_negation_form,
#     remove_repeated_characters,
# )

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
    print(back_translate("chào"))
    # text = "10 YAMAHA XSR700 The same as the XSR900 Yamaha XSR700 senior is basically a MT 07 with a few changes to create a more retro style in Europe XSR700's attraction is huge when more than 11,000 units have been sold Car equipped with 2-cylinder motor in parallel capacity 689 cc produces a capacity of 74 horsepower with this motor is a 6-speed gearbox and electronic fuel injection system with a price in the US market is 8 499 USD equivalent to 206 VND 4 million"
    # text = "10 . yamaha xsr700 : giống với đàn_anh xsr900 , yamaha xsr700 về cơ_bản là một chiếc mt - 07 với một_vài thay_đổi để tạo ra phong_cách retro hơn . tại châu âu , sức hút của xsr700 là rất lớn khi có hơn 11.000 chiếc đã được bán ra . xe trang_bị động_cơ 2 xy - lanh song_song , dung_tích 689 cc , sản_sinh công_suất 74 mã_lực . đi cùng với động_cơ này là hộp_số 6 cấp và hệ_thống phun xăng điện_tử . xe có_giá bán tại thị_trường mỹ là 8.499 usd , tương_đương 206,4 triệu đồng ."
    # print(back_translate(remove_whitespace(remove_punctuation_characters(remove_underline(text)))))
