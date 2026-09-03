"""Rutas de datos y registro en fichero, en desarrollo y dentro del ``.exe``."""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

import pytest

from decam import registro


@pytest.fixture
def modulo():
    """Recarga ``registro`` limpio y lo restaura al final.

    El módulo guarda estado global (ruta del log, handlers, excepthooks), así que
    cada test parte de una recarga y deja los hooks de Python como estaban.
    """
    excepthook = sys.excepthook
    hilo_hook = threading.excepthook
    mod = importlib.reload(registro)
    yield mod
    for manejador in list(mod.log.handlers):
        manejador.close()
        mod.log.removeHandler(manejador)
    sys.excepthook = excepthook
    threading.excepthook = hilo_hook
    importlib.reload(registro)


def congelar(monkeypatch, exe_dir, localappdata):
    """Simula el ejecutable de PyInstaller instalado en ``exe_dir``."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "DeCam.exe"))
    if localappdata is None:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.delenv("APPDATA", raising=False)
    else:
        monkeypatch.setenv("LOCALAPPDATA", str(localappdata))


class TestCarpetaDatos:
    def test_en_desarrollo_es_la_raiz_del_proyecto(self, modulo):
        assert not modulo.congelado()
        raiz = Path(modulo.__file__).resolve().parent.parent
        assert modulo.carpeta_datos() == raiz
        assert (raiz / "app.py").is_file(), "la raíz es donde vive app.py"
        assert modulo.ruta_config().name == "config.json"

    def test_congelado_usa_localappdata_y_la_crea(self, modulo, monkeypatch, tmp_path):
        exe_dir = tmp_path / "instalado"
        exe_dir.mkdir()
        congelar(monkeypatch, exe_dir, tmp_path / "LocalAppData")
        destino = modulo.carpeta_datos()
        assert destino == tmp_path / "LocalAppData" / "DeCam"
        assert destino.is_dir()
        # Lo importante: NUNCA la carpeta del exe (que en --onefile es temporal).
        assert modulo.ruta_config().parent != exe_dir

    def test_portable_respeta_config_junto_al_exe(self, modulo, monkeypatch, tmp_path):
        exe_dir = tmp_path / "portable"
        exe_dir.mkdir()
        (exe_dir / "config.json").write_text("{}", encoding="utf-8")
        congelar(monkeypatch, exe_dir, tmp_path / "LocalAppData")
        assert modulo.ruta_config() == exe_dir / "config.json"
        assert not (tmp_path / "LocalAppData").exists()

    def test_sin_localappdata_cae_a_la_carpeta_del_exe(self, modulo, monkeypatch, tmp_path):
        congelar(monkeypatch, tmp_path, None)
        assert modulo.carpeta_datos() == tmp_path

    def test_si_no_puede_crear_la_carpeta_cae_al_exe(self, modulo, monkeypatch, tmp_path):
        fichero = tmp_path / "soy_un_fichero"
        fichero.write_text("x", encoding="utf-8")
        congelar(monkeypatch, tmp_path, fichero)  # mkdir bajo un fichero falla
        assert modulo.carpeta_datos() == tmp_path


class TestConfigurar:
    def test_crea_el_log_y_es_idempotente(self, modulo, monkeypatch, tmp_path):
        congelar(monkeypatch, tmp_path, tmp_path / "lad")
        ruta = modulo.configurar()
        assert ruta == tmp_path / "lad" / "DeCam" / "decam.log"
        assert modulo.configurar() is ruta
        assert modulo.ruta_log() == ruta
        assert len(modulo.log.handlers) == 1
        assert "Inicio de DeCam" in ruta.read_text(encoding="utf-8")

    def test_sin_carpeta_escribible_devuelve_none_y_no_revienta(
        self, modulo, monkeypatch, tmp_path
    ):
        congelar(monkeypatch, tmp_path / "no" / "existe", None)
        assert modulo.configurar() is None
        assert modulo.ruta_log() is None
        modulo.log.info("no debe fallar aunque no haya fichero")

    def test_excepcion_en_hilo_queda_en_el_log(self, modulo, monkeypatch, tmp_path):
        congelar(monkeypatch, tmp_path, tmp_path / "lad")
        # El hook por defecto imprimiría el traceback por stderr; se silencia.
        monkeypatch.setattr(threading, "excepthook", lambda args: None)
        ruta = modulo.configurar()

        def revienta():
            raise RuntimeError("fallo en hilo de prueba")

        hilo = threading.Thread(target=revienta, name="analisis")
        hilo.start()
        hilo.join()
        contenido = ruta.read_text(encoding="utf-8")
        assert "fallo en hilo de prueba" in contenido
        assert "Traceback" in contenido
        assert "analisis" in contenido

    def test_excepcion_de_tk_queda_en_el_log(self, modulo, monkeypatch, tmp_path):
        congelar(monkeypatch, tmp_path, tmp_path / "lad")
        ruta = modulo.configurar()
        try:
            raise ValueError("fallo en callback de prueba")
        except ValueError as exc:
            modulo.excepcion_de_tk(type(exc), exc, exc.__traceback__)
        contenido = ruta.read_text(encoding="utf-8")
        assert "callback de Tkinter" in contenido
        assert "fallo en callback de prueba" in contenido
