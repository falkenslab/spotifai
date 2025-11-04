from typing import List, Optional
from langchain.tools import tool

from spotifai.spotify_manager import SpotifyManager

def init_spotify_manager() -> tuple[str, List[tool]]:
    global __manager
    __manager = SpotifyManager()
    current_user = __manager.current_user()
    return (current_user, SPOTIFY_TOOLS)

@tool
def search_song(query: str, limit: int = 10) -> List[dict]:
    """
    Busca una canción en Spotify y devuelve los detalles de la canción.
    Args:
        query (str): La consulta de búsqueda (nombre de la canción, artista, etc.).
        limit (int): Número máximo de resultados a devolver. Por defecto es 10.
    Returns:
        List[dict]: Una lista de diccionarios con los detalles de las canciones encontradas.
    """
    print("Buscando canción en Spotify:", query)
    return __manager.search_song(
        query=query,
        limit=limit
    )

@tool
def create_playlist(name: str, description: str = "", public: bool = False, track_uris: Optional[List[str]] = None) -> dict:
    """
    Crea una playlist en el Spotify del usuario y opcionalmente añade canciones.
    Args:
        name (str): Nombre de la playlist.
        description (str): Descripción de la playlist.
        public (bool): Si la playlist será pública (True) o privada (False).
        track_uris (List[str] | None): Lista opcional de URIs de tracks a añadir
            (formato 'spotify:track:<id>' o URLs de Spotify válidas).
    Returns:
        dict: Información básica de la playlist creada (id, url, etc.).
    """
    print(f"Creando playlist en Spotify: {name} con {len(track_uris) if track_uris else 0} canciones ...")
    return __manager.create_playlist(
        name=name,
        description=description,
        public=public,
        track_uris=track_uris
    )

@tool
def get_my_playlists(limit: int = 50, fetch_all: bool = True) -> List[dict]:
    """
    Recupera las playlists del usuario actual (propias y seguidas).
    Args:
        limit (int): Número de playlists por página (máx 50).
        fetch_all (bool): Si True, pagina hasta recuperar todas.
    Returns:
        List[dict]: Lista de playlists con información básica.
    """
    print("Recuperando playlists del usuario...")
    return __manager.get_my_playlists(
        limit=limit,
        fetch_all=fetch_all
    )

@tool
def add_tracks_to_playlist(playlist_id: str, track_uris: List[str], position: Optional[int] = None) -> dict:
    """
    Añade temas a una playlist.
    Args:
        playlist_id (str): ID de la playlist destino.
        track_uris (List[str]): URIs/URLs/IDs de los temas a añadir.
        position (int|None): Posición inicial donde insertar (opcional).
    Returns:
        dict: Resumen de la operación con número de temas añadidos.
    """
    print(f"Añadiendo {len(track_uris)} temas a la playlist {playlist_id}...")
    return __manager.add_tracks_to_playlist(
        playlist_id=playlist_id,
        track_uris=track_uris,
        position=position
    )


@tool
def remove_tracks_from_playlist(playlist_id: str, track_uris: List[str]) -> dict:
    """
    Elimina todos los ocurrencias de los temas dados en una playlist.
    Args:
        playlist_id (str): ID de la playlist.
        track_uris (List[str]): URIs/URLs/IDs de los temas a eliminar.
    Returns:
        dict: Resumen con número de temas solicitados a eliminar y último snapshot.
    """
    print(f"Eliminando hasta {len(track_uris)} temas de la playlist {playlist_id} (todas las ocurrencias)...")
    return __manager.remove_tracks_from_playlist(
        playlist_id=playlist_id,
        track_uris=track_uris
    )


@tool
def get_playlist_tracks(playlist_id: str, limit: int = 100, fetch_all: bool = True) -> List[dict]:
    """
    Lista las pistas de una playlist dada.
    Args:
        playlist_id (str): ID de la playlist.
        limit (int): Tamaño de página (1-100).
        fetch_all (bool): Si True, pagina hasta recuperar todas las pistas.
    Returns:
        List[dict]: Lista de pistas con metadatos básicos.
    """
    print(f"Listando pistas de la playlist {playlist_id}...")
    return __manager.get_playlist_tracks(
        playlist_id=playlist_id,
        limit=limit,
        fetch_all=fetch_all
    )


@tool
def reorder_playlist_items(playlist_id: str, range_start: int, insert_before: int, range_length: int = 1, snapshot_id: Optional[str] = None,) -> dict:
    """
    Reordena uno o varios elementos contiguos de una playlist.
    Mueve el bloque que empieza en `range_start` (longitud `range_length`) a la
    posición `insert_before`.
    Args:
        playlist_id (str): ID de la playlist.
        range_start (int): Índice inicial del bloque a mover (0-based).
        insert_before (int): Índice destino antes del cual insertar el bloque.
        range_length (int): Longitud del bloque a mover (por defecto 1).
        snapshot_id (str|None): Snapshot opcional para control de concurrencia.
    Returns:
        dict: `{playlist_id, range_start, insert_before, range_length, snapshot_id}`
    """
    print(
        f"Reordenando playlist {playlist_id}: start={range_start}, before={insert_before}, len={range_length}"
    )
    return __manager.reorder_playlist_items(
        playlist_id=playlist_id,
        range_start=range_start,
        insert_before=insert_before,
        range_length=range_length,
        snapshot_id=snapshot_id
    )

# Lista de herramientas disponibles
SPOTIFY_TOOLS = [
    search_song,
    create_playlist,
    add_tracks_to_playlist,
    remove_tracks_from_playlist,
    get_playlist_tracks,
    get_my_playlists,
    reorder_playlist_items
]