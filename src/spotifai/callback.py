import ssl
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from importlib import resources

from spotifai.__init__ import CERT_FILE, KEY_FILE, SPOTIFAI_CONFIG_DIR as CONFIG_DIR, SPOTIPY_REDIRECT_HOST as HOST, SPOTIPY_REDIRECT_PORT as PORT, SPOTIPY_REDIRECT_URI as REDIRECT_URI

# Variable global para almacenar el código de autorización
auth_code = None

class CallbackHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Sobrescribir para omitir los logs de acceso
        pass
    
    def do_GET(self):
        global auth_code
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Si la ruta no es /callback, devolver 404    
        if path != "/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        
        # Extraer el código de autorización de la URL
        query = parse_qs(parsed_path.query)
        auth_code = query.get("code", [None])[0]

        # Lee la página de respuesta usando importlib.resources
        try:
            # Para Python 3.9+
            with resources.files('spotifai').joinpath('callback.html').open('rb') as f:
                response_content = f.read()
        except AttributeError:
            # Fallback para Python 3.7-3.8
            with resources.open_binary('spotifai', 'callback.html') as f:
                response_content = f.read()

        # Responder al navegador
        self.send_response(200)
        self.end_headers()
        self.wfile.write(response_content)

class HTTPSServer(HTTPServer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__ensure_certs()
        self.socket = self.__get_ssl_context(CERT_FILE, KEY_FILE).wrap_socket(self.socket, server_side=True)

    def __get_ssl_context(self, certfile, keyfile):
        # Usar ssl.create_default_context() en lugar del protocolo deprecated
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile, keyfile)
        # Configurar para permitir conexiones locales (desarrollo)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    # 1. Crear certificados si no existen
    def __ensure_certs(self):
        if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
            os.makedirs(CONFIG_DIR, exist_ok=True)
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", KEY_FILE, "-out", CERT_FILE,
                    "-days", "36500", "-nodes", "-subj", f"/CN={HOST}"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

def run_mini_auth_https_server():
    global httpd, auth_code
    
    httpd = HTTPSServer((HOST, PORT), CallbackHandler)
    
    # Iniciar servidor en thread separado
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    
    # Esperar a que se reciba el código
    print(f"🌐 Autentícate en tu navegador, por favor")
    while auth_code is None:
        time.sleep(0.1)

    print("✅ Código de autorización recibido")
    
    # Detener servidor
    httpd.shutdown()
    
    return auth_code