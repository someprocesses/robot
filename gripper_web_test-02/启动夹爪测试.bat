@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 优先用 py 启动器（python.org 安装包自带），避开微软商店的 python 假命令
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

%PY% --version >nul 2>nul
if errorlevel 1 (
    echo 未检测到 Python，请先安装（只需一次）:
    echo   https://www.python.org/downloads/
    echo   安装时务必勾选 "Add python.exe to PATH"
    pause
    exit /b 1
)

%PY% -c "import flask, paramiko" >nul 2>nul
if errorlevel 1 (
    echo 首次运行，正在安装依赖 flask 和 paramiko ...
    %PY% -m pip install flask paramiko -i https://pypi.tuna.tsinghua.edu.cn/simple
)

%PY% gripper_test_web.py
pause
