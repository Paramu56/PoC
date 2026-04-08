@echo off
cd /d "%~dp0"
rem fileWatcherType avoids scanning transformers/torchvision in site-packages (see .streamlit\config.toml)
python -m streamlit run poc_streamlit.py --server.fileWatcherType none %*
