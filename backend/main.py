from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware

import uvicorn

# from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic
import os

current_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
root_folder = current_folder


security = HTTPBasic()

def initialize_server():
    global _is_initialized
    if _is_initialized:
        return

    print("Performing one-time server initialization...")

    # Create tables only once
    from database.create_sql_tables import create_all_tables
    create_all_tables()

    _is_initialized = True



# Create FastAPI app
app = FastAPI(on_startup=[initialize_server])

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000',"http://localhost:63343", "http://localhost:55310", 'http://localhost:63342', 'http://localhost:63695', 'https://lex-chatbot.epfl.ch', 'https://lex-chatbot-test.epfl.ch', 'lex-chatbot-backend-service-test', 'lex-chatbot-backend-service'
'http://127.0.0.1:3000',"http://127.0.0.1:63343", "http://127.0.0.1:55310", 'http://127.0.0.1:63342', 'http://127.0.0.1:63695'
                   ],  # Specify the allowed origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from routes import auth, libraries, progress, websocket_connections, pdfs

app.include_router(auth.router)
app.include_router(libraries.router)
app.include_router(progress.router)
app.include_router(websocket_connections.router)
app.include_router(pdfs.router)

# Print routes only once during startup
def print_routes():
    for route in app.routes:
        try:
            print(f"Route: {route.path}, Methods: {route.methods}")
        except:
            try:
                print(f"Route: {route.path}")
            except:
                pass



# Add initialization guard
_is_initialized = False

import os

# Define the base directory for storing databases
DATABASE_DIR = os.path.join(root_folder, 'data', 'users')
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir


print('test')






if __name__ == "__main__":
    print_routes()
    uvicorn.run("main:app", host='0.0.0.0', port=8000, workers = 1
                # log_level="debug"
                )
