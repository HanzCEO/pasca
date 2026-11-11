"""Pasca desktop app: drop sample files, get back the author's writing system."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from imgui_bundle import hello_imgui, imgui, immapp, rich_md
from imgui_bundle import portable_file_dialogs as pfd

from . import config, llm, metrics
from .ingest import SUPPORTED_EXTENSIONS, Document, IngestError, load_document

FILE_FILTERS = ["Text documents", " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS), "All files", "*"]
ERROR_COLOR = imgui.ImVec4(0.95, 0.45, 0.4, 1.0)
WARN_COLOR = imgui.ImVec4(0.95, 0.75, 0.3, 1.0)


class Phase(Enum):
    IDLE = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class AppState:
    settings: config.Settings = field(default_factory=config.load)
    docs: list[Document] = field(default_factory=list)
    file_errors: list[str] = field(default_factory=list)
    loading: int = 0  # files currently being read
    events: queue.Queue = field(default_factory=queue.Queue)  # results from worker threads

    phase: Phase = Phase.IDLE
    result: str = ""
    status: str = ""
    warning: str = ""
    metrics: dict | None = None
    show_metrics: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)

    open_dialog: pfd.open_file | None = None
    save_dialog: pfd.save_file | None = None
    settings_saved_msg: str = ""


state = AppState()


# ---------------------------------------------------------------- background work


def add_paths(paths: list[str]) -> None:
    """Read files off the GUI thread; results arrive through ``state.events``."""
    known = {str(d.path.resolve()) for d in state.docs}
    fresh = [p for p in dict.fromkeys(paths) if str(Path(p).resolve()) not in known]
    if not fresh:
        return
    state.loading += len(fresh)

    def work() -> None:
        for p in fresh:
            try:
                state.events.put(("doc", load_document(p)))
            except IngestError as exc:
                state.events.put(("file_error", str(exc)))
            hello_imgui.request_refresh()

    threading.Thread(target=work, daemon=True).start()


def start_analysis() -> None:
    docs = list(state.docs)
    settings = config.Settings(**vars(state.settings))
    state.phase = Phase.RUNNING
    state.result = ""
    state.warning = ""
    state.status = "Measuring style…"
    state.cancel = threading.Event()
    cancel = state.cancel

    def work() -> None:
        try:
            corpus = "\n\n".join(d.text for d in docs)
            profile = metrics.profile(corpus)
            state.events.put(("metrics", profile))
            request = llm.prepare(docs, profile, settings)
            if request.truncated:
                state.events.put(("warning", f"Files exceed the excerpt budget ({settings.max_excerpt_chars:,} chars); "
                                             "the model saw trimmed excerpts. Metrics still cover all text."))
            state.events.put(("status", f"Waiting for {settings.model}…"))
            llm.stream_completion(settings, request.messages, lambda d: state.events.put(("delta", d)), cancel)
            state.events.put(("done", "Cancelled" if cancel.is_set() else "Done"))
        except Exception as exc:  # surface any API / network error in the UI
            state.events.put(("failed", f"{type(exc).__name__}: {exc}"))
        hello_imgui.request_refresh()

    threading.Thread(target=work, daemon=True).start()


def drain_events() -> None:
    while True:
        try:
            kind, payload = state.events.get_nowait()
        except queue.Empty:
            return
        if kind == "doc":
            state.docs.append(payload)
            state.loading -= 1
        elif kind == "file_error":
            state.file_errors.append(payload)
            state.loading -= 1
        elif kind == "metrics":
            state.metrics = payload
        elif kind == "warning":
            state.warning = payload
        elif kind == "status":
            state.status = payload
        elif kind == "delta":
            if not state.result:
                state.status = "Writing…"
            state.result += payload
            hello_imgui.request_refresh()
        elif kind == "done":
            state.phase = Phase.DONE
            state.status = payload
        elif kind == "failed":
            state.phase = Phase.FAILED
            state.status = payload


def poll_dialogs() -> None:
    if state.open_dialog is not None and state.open_dialog.ready(0):
        paths = state.open_dialog.result()
        state.open_dialog = None
        if paths:
            add_paths(paths)
    if state.save_dialog is not None and state.save_dialog.ready(0):
        target = state.save_dialog.result()
        state.save_dialog = None
        if target:
            path = Path(target)
            if not path.suffix:
                path = path.with_suffix(".md")
            try:
                path.write_text(state.result, encoding="utf-8")
                state.status = f"Saved to {path}"
            except OSError as exc:
                state.status = f"Save failed: {exc}"


# ---------------------------------------------------------------- GUI


def gui_files() -> None:
    imgui.separator_text("Sample files")
    if imgui.button("Add files…") and state.open_dialog is None:
        state.open_dialog = pfd.open_file("Add sample files", "", FILE_FILTERS, pfd.opt.multiselect)
    imgui.same_line()
    imgui.begin_disabled(not state.docs or state.phase == Phase.RUNNING)
    if imgui.button("Clear"):
        state.docs.clear()
        state.file_errors.clear()
    imgui.end_disabled()

    list_height = hello_imgui.em_size(12)
    imgui.begin_child("files", imgui.ImVec2(0, list_height), imgui.ChildFlags_.borders)
    if not state.docs and not state.loading:
        imgui.text_disabled("Drop files anywhere on this window")
        imgui.text_disabled(", ".join(SUPPORTED_EXTENSIONS))
    remove = None
    for i, doc in enumerate(state.docs):
        imgui.push_id(i)
        if imgui.small_button("x"):
            remove = i
        imgui.same_line()
        imgui.text(f"{doc.name}")
        imgui.same_line()
        imgui.text_disabled(f"{doc.word_count:,} words")
        if imgui.is_item_hovered():
            imgui.set_tooltip(str(doc.path))
        imgui.pop_id()
    if remove is not None and state.phase != Phase.RUNNING:
        state.docs.pop(remove)
    if state.loading:
        imgui.text_disabled(f"Reading {state.loading} file(s)…")
    for err in state.file_errors:
        imgui.push_text_wrap_pos(0)
        imgui.text_colored(ERROR_COLOR, err)
        imgui.pop_text_wrap_pos()
    imgui.end_child()

    total = sum(d.word_count for d in state.docs)
    imgui.text_disabled(f"{len(state.docs)} file(s), {total:,} words")
    if state.docs and total < 1500:
        imgui.text_colored(WARN_COLOR, "Tip: 1,500+ words gives more reliable results.")


def gui_settings() -> None:
    s = state.settings
    imgui.separator_text("Endpoint (OpenAI-compatible)")
    imgui.push_item_width(-hello_imgui.em_size(5))
    _, s.base_url = imgui.input_text("Base URL", s.base_url)
    _, s.model = imgui.input_text("Model", s.model)
    _, s.api_key = imgui.input_text("API key", s.api_key, imgui.InputTextFlags_.password)
    if not s.api_key and s.effective_api_key():
        imgui.text_disabled(f"Using ${config.API_KEY_ENV}")
    _, s.temperature = imgui.slider_float("Temp.", s.temperature, 0.0, 1.5, "%.2f")
    _, s.max_excerpt_chars = imgui.input_int("Excerpt", s.max_excerpt_chars, 5_000, 20_000)
    s.max_excerpt_chars = max(2_000, s.max_excerpt_chars)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Max characters of sample text sent to the model")
    imgui.pop_item_width()
    _, s.save_api_key = imgui.checkbox("Remember API key on disk", s.save_api_key)
    if imgui.button("Save settings"):
        try:
            state.settings_saved_msg = f"Saved to {config.save(s)}"
        except OSError as exc:
            state.settings_saved_msg = f"Could not save: {exc}"
    if state.settings_saved_msg:
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(state.settings_saved_msg)
        imgui.pop_text_wrap_pos()


def gui_actions() -> None:
    imgui.separator_text("Analyze")
    s = state.settings
    missing = [n for n, v in (("files", state.docs), ("model", s.model.strip()), ("base URL", s.base_url.strip())) if not v]
    running = state.phase == Phase.RUNNING
    if running:
        if imgui.button("Cancel", imgui.ImVec2(-1, 0)):
            state.cancel.set()
            state.status = "Cancelling…"
    else:
        imgui.begin_disabled(bool(missing) or state.loading > 0)
        if imgui.button("Extract writing system", imgui.ImVec2(-1, hello_imgui.em_size(2))):
            start_analysis()
        imgui.end_disabled()
        if missing:
            imgui.text_disabled("Needs: " + ", ".join(missing))


def gui_result() -> None:
    running = state.phase == Phase.RUNNING
    has_result = bool(state.result)

    imgui.begin_disabled(not has_result or running)
    if imgui.button("Copy"):
        imgui.set_clipboard_text(state.result)
        state.status = "Copied to clipboard"
    imgui.same_line()
    if imgui.button("Save .md…") and state.save_dialog is None:
        state.save_dialog = pfd.save_file("Save writing system", "writing-system.md", ["Markdown", "*.md"])
    imgui.end_disabled()
    imgui.same_line()
    imgui.begin_disabled(state.metrics is None)
    _, state.show_metrics = imgui.checkbox("Show metrics", state.show_metrics)
    imgui.end_disabled()
    imgui.same_line()
    color = ERROR_COLOR if state.phase == Phase.FAILED else imgui.get_style().color_(imgui.Col_.text_disabled)
    imgui.text_colored(color, state.status)
    if state.warning:
        imgui.push_text_wrap_pos(0)
        imgui.text_colored(WARN_COLOR, state.warning)
        imgui.pop_text_wrap_pos()

    imgui.begin_child("result", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders)
    if state.show_metrics and state.metrics is not None:
        rich_md.render_raw("```json\n" + json.dumps(state.metrics, indent=2) + "\n```")
    elif has_result:
        rich_md.render_raw(state.result)
    else:
        imgui.text_disabled("The writing system will appear here.")
    imgui.end_child()


def gui() -> None:
    drain_events()
    poll_dialogs()

    left_width = hello_imgui.em_size(24)
    imgui.begin_child("left", imgui.ImVec2(left_width, 0), imgui.ChildFlags_.resize_x)
    gui_files()
    gui_settings()
    gui_actions()
    imgui.end_child()
    imgui.same_line()
    imgui.begin_child("right", imgui.ImVec2(0, 0))
    gui_result()
    imgui.end_child()


def install_drop_handler() -> None:
    """Hello ImGui has no drop API; register a GLFW drop callback on its window."""
    try:
        from imgui_bundle import glfw_utils
        import glfw  # must be imported after imgui_bundle (uses its bundled GLFW library)

        window = glfw_utils.glfw_window_hello_imgui()
        glfw.set_drop_callback(window, lambda _w, paths: add_paths(list(paths)))
    except Exception as exc:  # e.g. a non-GLFW backend
        state.file_errors.append(f"Drag-and-drop unavailable ({exc}); use 'Add files…'.")


def main() -> None:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Pasca — writing system extractor"
    params.app_window_params.window_geometry.size = (1280, 820)
    params.app_window_params.restore_previous_geometry = True
    params.callbacks.show_gui = gui
    params.callbacks.post_init = install_drop_handler
    params.fps_idling.fps_idle = 10
    immapp.run(params, immapp.AddOnsParams(with_markdown=True))
