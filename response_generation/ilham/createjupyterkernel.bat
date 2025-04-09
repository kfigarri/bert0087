@echo off
REM Check if the correct number of parameters are provided
IF "%~2"=="" (
    echo Usage: %0 ^<env_name^> ^<display_name^>
    exit /b 1
)

REM Assign parameters to variables
set ENV_NAME=%1
set DISPLAY_NAME=%2

REM Create the virtual environment
python -m venv %ENV_NAME%

REM Activate the virtual environment
call %ENV_NAME%\Scripts\activate

REM Install Jupyter and ipykernel in the virtual environment
pip install jupyter
pip install ipykernel

REM Add the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=%ENV_NAME% --display-name "%DISPLAY_NAME%"

echo Jupyter kernel "%DISPLAY_NAME%" has been created for the environment "%ENV_NAME%".
