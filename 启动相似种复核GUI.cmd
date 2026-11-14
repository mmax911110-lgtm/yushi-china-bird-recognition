@echo off
chcp 65001 >nul
setlocal
if not defined BIRD_NVIDIA_PACKAGE set "BIRD_NVIDIA_PACKAGE=D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA"
if not exist "%BIRD_NVIDIA_PACKAGE%\runtime\python.exe" (
  echo 未找到完整NVIDIA运行包。请将 BIRD_NVIDIA_PACKAGE 设置为该文件夹路径。
  pause
  exit /b 1
)
"%BIRD_NVIDIA_PACKAGE%\runtime\python.exe" -X utf8 -B -I -S "%~dp0desktop_review\launch.py" --package "%BIRD_NVIDIA_PACKAGE%" %*
if errorlevel 1 pause
