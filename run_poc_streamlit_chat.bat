@echo off
cd /d "%~dp0"
rem Separate launcher for chat PoC path (baseline launcher unchanged).
python -m streamlit run poc_streamlit_chat.py --server.fileWatcherType none %*
