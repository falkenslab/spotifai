import ssl
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from importlib import resources
from .utils import make_localhost_cert

from .__init__ import (
    CERT_FILE,
    KEY_FILE,
    SPOTIFAI_CONFIG_DIR as CONFIG_DIR,
    SPOTIPY_REDIRECT_HOST as HOST,
    SPOTIPY_REDIRECT_PORT as PORT,
    SPOTIPY_REDIRECT_URI as REDIRECT_URI,
)

# TODO  Variable global para almacenar el código de autorización 🙈 no digas nada, lo sé!!!
auth_code = None


class CallbackHandler(BaseHTTPRequestHandler):
    """
    Manejador de solicitudes HTTP para capturar el código de autorización OAuth2.
    """

    def log_message(self, format, *args):
        # Sobrescribir para omitir los logs de acceso
        pass

    def do_GET(self):
        """
        Maneja las solicitudes GET para capturar el código de autorización.
        """
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
        with (
            resources
                .files("spotify")
                .joinpath("assets")
                .joinpath("callback.html")
                .open("rb") as f
        ): response_content = f.read()

        # Responder al navegador
        self.send_response(200)
        self.end_headers()
        self.wfile.write(response_content)


class SpotifyCallbackServer(HTTPServer):
    """
    Servidor HTTPS que utiliza certificados SSL autofirmados.
    """

    def __init__(self):
        super().__init__((HOST, PORT), CallbackHandler)
        self.__ensure_certs()
        self.socket = self.__get_ssl_context(CERT_FILE, KEY_FILE).wrap_socket(
            self.socket, server_side=True
        )

    def __get_ssl_context(self, certfile, keyfile):
        """
        Crear un contexto SSL con los certificados dados.
        """
        # Usar ssl.create_default_context() en lugar del protocolo deprecated
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile, keyfile)
        # Configurar para permitir conexiones locales (desarrollo)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def __ensure_certs(self):
        """
        Crear certificados autofirmados si no existen.
        """
        if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
            os.makedirs(CONFIG_DIR, exist_ok=True)
            make_localhost_cert(CERT_FILE, KEY_FILE, days_valid=36500)

    def run(self):
        global auth_code

        # Iniciar servidor en thread separado
        server_thread = threading.Thread(target=self.serve_forever, daemon=True)
        server_thread.start()

        # Esperar a que se reciba el código
        print(f"🌐 Autentícate en tu navegador, por favor")
        while auth_code is None:
            time.sleep(0.1)

        print("✅ Código de autorización recibido")

        # Detener servidor
        self.shutdown()

        return auth_code
