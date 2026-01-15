import spotipy
import webbrowser
from typing import List, Optional
from spotipy.oauth2 import SpotifyPKCE

from .__init__ import (
    SPOTIPY_REDIRECT_URI,
    SPOTIFY_CLIENT_ID,
    SPOTIPY_CACHE_PATH,
)
from .spotify_callback import SpotifyCallbackServer


class SpotifyManager:
    """Gestor de autenticación y cliente de Spotify."""

    def __init__(self):

        # Configurar el gestor de autenticación PKCE
        auth_manager = SpotifyPKCE(
            client_id=SPOTIFY_CLIENT_ID,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope=" ".join(
                [
                    "user-library-read",            # Leer canciones, álbumes o episodios guardados por el usuario.
                    "playlist-read-private",        # Leer listas de reproducción privadas del usuario.
                    "playlist-read-collaborative",  # Leer listas de reproducción colaborativas del usuario.
                    "playlist-modify-public",       # Modificar listas de reproducción públicas del usuario.
                    "playlist-modify-private",      # Modificar listas de reproducción privadas del usuario.
                    "ugc-image-upload",             # Subir imágenes de contenido generado por el usuario.
                ]
            ),
            cache_path=SPOTIPY_CACHE_PATH,
            open_browser=False,
        )

        # Intentar cargar un token desde caché
        token_info = auth_manager.cache_handler.get_cached_token()

        if token_info is None or not auth_manager.validate_token(token_info):
            print("🔒 Autenticando con Spotify")
            # Abrir el navegador para que el usuario autorice la aplicación
            webbrowser.open(auth_manager.get_authorize_url())
            # Iniciar el servidor local para capturar el código de autorización
            code = SpotifyCallbackServer().run()
            # Intercambiar el código por un token de acceso
            token_info = auth_manager.get_access_token(code)
            print("🔓 Autenticación completada\n")

        # Crear el cliente de Spotify con el auth_manager (no el token directamente)
        self.client = spotipy.Spotify(auth_manager=auth_manager)

    def current_user(self) -> str:
        return self.client.current_user()["display_name"]

    def search_song(self, query: str, limit: int = 10) -> List[dict]:
        result = self.client.search(q=query, type="track", limit=limit)
        return result["tracks"]["items"]

    def create_playlist(
        self,
        name: str,
        description: str = "",
        public: bool = False,
        track_uris: Optional[List[str]] = None,
    ) -> dict:
        me = self.client.current_user()
        user_id = me["id"]
        playlist = self.client.user_playlist_create(
            user=user_id,
            name=name,
            public=public,
            description=description,
        )
        if track_uris:
            # La API permite añadir hasta 100 elementos por llamada
            for i in range(0, len(track_uris), 100):
                chunk = track_uris[i : i + 100]
                self.client.playlist_add_items(playlist_id=playlist["id"], items=chunk)
        return {
            "id": playlist["id"],
            "name": playlist.get("name"),
            "public": playlist.get("public"),
            "description": playlist.get("description"),
            "url": playlist.get("external_urls", {}).get("spotify"),
            "tracks_added": len(track_uris) if track_uris else 0,
        }

    def get_my_playlists(self, limit: int = 50, fetch_all: bool = True) -> List[dict]:
        playlists: List[dict] = []
        results = self.client.current_user_playlists(limit=max(1, min(limit, 50)))
        while True:
            for item in results.get("items", []):
                playlists.append(
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "public": item.get("public"),
                        "collaborative": item.get("collaborative"),
                        "description": item.get("description"),
                        "tracks_total": item.get("tracks", {}).get("total"),
                        "owner": item.get("owner", {}).get("display_name")
                        or item.get("owner", {}).get("id"),
                        "url": item.get("external_urls", {}).get("spotify"),
                        "uri": item.get("uri"),
                    }
                )
            if fetch_all and results.get("next"):
                results = self.client.next(results)
            else:
                break
        return playlists

    def add_tracks_to_playlist(
        self, playlist_id: str, track_uris: List[str], position: Optional[int] = None
    ) -> dict:
        added = 0
        current_pos = position
        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i : i + 100]
            self.client.playlist_add_items(
                playlist_id=playlist_id,
                items=chunk,
                position=current_pos,
            )
            added += len(chunk)
            # Si se especificó posición, avanzamos acorde al tamaño añadido
            if current_pos is not None:
                current_pos += len(chunk)
        return {"playlist_id": playlist_id, "tracks_added": added}

    def remove_tracks_from_playlist(
        self, playlist_id: str, track_uris: List[str]
    ) -> dict:
        snapshot_id = None
        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i : i + 100]
            resp = self.client.playlist_remove_all_occurrences_of_items(
                playlist_id=playlist_id, items=chunk
            )
            snapshot_id = resp.get("snapshot_id", snapshot_id)
        return {
            "playlist_id": playlist_id,
            "tracks_requested": len(track_uris),
            "snapshot_id": snapshot_id,
        }

    def get_playlist_tracks(
        self, playlist_id: str, limit: int = 100, fetch_all: bool = True
    ) -> List[dict]:
        limit = max(1, min(100, limit))
        tracks: List[dict] = []
        results = self.client.playlist_items(playlist_id=playlist_id, limit=limit)

        def _map_item(item: dict) -> Optional[dict]:
            track = item.get("track") if item else None
            if not track:
                return None
            artists = [a.get("name") for a in track.get("artists", []) if a]
            album = track.get("album", {})
            return {
                "id": track.get("id"),
                "name": track.get("name"),
                "artists": artists,
                "album": album.get("name"),
                "duration_ms": track.get("duration_ms"),
                "explicit": track.get("explicit"),
                "added_at": item.get("added_at"),
                "added_by": item.get("added_by", {}).get("id"),
                "uri": track.get("uri"),
                "url": track.get("external_urls", {}).get("spotify"),
            }

        while True:
            for it in results.get("items", []):
                mapped = _map_item(it)
                if mapped:
                    tracks.append(mapped)
            if fetch_all and results.get("next"):
                results = self.client.next(results)
            else:
                break
        return tracks

    def reorder_playlist_items(
        self,
        playlist_id: str,
        range_start: int,
        insert_before: int,
        range_length: int = 1,
        snapshot_id: Optional[str] = None,
    ) -> dict:
        resp = self.client.playlist_reorder_items(
            playlist_id=playlist_id,
            range_start=range_start,
            insert_before=insert_before,
            range_length=range_length,
            snapshot_id=snapshot_id,
        )
        return {
            "playlist_id": playlist_id,
            "range_start": range_start,
            "insert_before": insert_before,
            "range_length": range_length,
            "snapshot_id": resp.get("snapshot_id"),
        }
