from selenium import webdriver
import selenium
import time


driver = webdriver.Chrome()

driver.get("https://www.topcoder.com/challenges/30120953?tab=details")
time.sleep(3)
Challenge_Overview = driver.find_element_by_xpath('//div[@class="_39Q1z2"]/div/div/article[1]').text

while True:
    try:
        driver.get("https://www.topcoder.com/challenges/30120953?tab=registrants")
        time.sleep(3)
        REGISTRANTS = driver.find_elements_by_xpath('//div[@class="_237uoT"]//div[@class="pVqBVg"]/span')
        for REGISTRANT in REGISTRANTS:
            print(REGISTRANT.text)
        break
    except selenium.common.exceptions.NoSuchElementException as error:
        print("failed")

# while True:
#     try:
#         driver.get("https://www.topcoder.com/challenges/30120959?tab=submissions")
#         time.sleep(3)
#         SUBMISSIONS = driver.find_elements_by_xpath('//div[@class="_3hLZag"]/div[1]/a')
#         for SUBMISSION in SUBMISSIONS:
#             print(SUBMISSION.text)
#         break
#     except selenium.common.exceptions.NoSuchElementException as error:
#         print("failed")

