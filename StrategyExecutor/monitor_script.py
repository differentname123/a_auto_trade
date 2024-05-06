# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-03-24 22:54
:last_date:
    2024-03-24 22:54
:description:
    
"""
import subprocess
import time


def run_script():
    while True:
        print("Starting the main script...")
        process = subprocess.Popen(["python", "getAllRandomForestClassifierReport.py"])
        process.wait()

        if process.returncode != 0:
            print("The main script died. Restarting in 5 seconds...")
            time.sleep(5)
        else:
            print("The main script finished successfully.")
            break


if __name__ == "__main__":
    run_script()