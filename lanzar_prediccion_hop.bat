@echo off
REM Activar entorno virtual
call C:\DADES\DOCLIB\CIE10\venv\Scripts\activate.bat

REM Capturar el texto pasado como argumento
set "TEXTO=%~1"

REM Ejecutar Python y devolver solo el CIE predicho
for /f "delims=" %%i in ('python C:\DADES\DOCLIB\CIE10\PrediccionHop.py "%TEXTO%"') do (
    echo %%i
)

