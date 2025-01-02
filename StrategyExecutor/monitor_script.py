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


def run_script(script_name):
    while True:
        print(f"Starting the script: {script_name}")
        process = subprocess.Popen(["python", script_name])
        process.wait()

        if process.returncode != 0:
            print(f"The script {script_name} died. Restarting in 5 seconds...")
            time.sleep(5)
        else:
            print(f"The script {script_name} finished successfully.")
            break

def run_scripts():
    while True:
        run_script("getAllRandomForestClassifierReport.py")
        run_script("getAllRandomForestClassifierReportCUML.py")



if __name__ == "__main__":
    run_scripts()