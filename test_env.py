from dotenv import load_dotenv
import os

# Caminho completo do .env
load_dotenv(r"C:\Users\alyso\Downloads\Nova pasta\.env")

print("News Filter:", os.getenv("ENABLE_NEWS_FILTER"))
print("Dynamic Params:", os.getenv("ENABLE_DYNAMIC_PARAMS"))
print("SR Detection:", os.getenv("ENABLE_SR_DETECTION"))
