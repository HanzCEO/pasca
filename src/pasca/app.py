"""Pasca desktop app. Drop sample files and get back the author's writing system."""

from __future__ import annotations

import json
import os
import platform
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
ACCENT = imgui.ImVec4(0.18, 0.42, 0.85, 1.0)
ACCENT_HOVER = imgui.ImVec4(0.27, 0.53, 0.95, 1.0)


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


state = AppState()


# ---------------------------------------------------------------- background work


def add_paths(paths: list[str]) -> None:
    """Read files off the GUI thread. Results arrive through ``state.events``."""
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
                state.events.put(("warning", f"Files exceed the excerpt budget ({settings.max_excerpt_chars:,} chars). "
                                             "the model saw trimmed excerpts. Metrics still cover all text."))
            state.events.put(("status", f"Waiting for {settings.model}…"))
            llm.stream_completion(settings, request.messages, lambda d: state.events.put(("delta", d)), cancel)
            state.events.put(("done", "Cancelled" if cancel.is_set() else "Done"))
        except Exception as exc:  # surface any API / network error in the UI
            state.events.put(("failed", f"{type(exc).__name__} ({exc})"))
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
                state.status = f"Save failed ({exc})"


# ---------------------------------------------------------------- GUI
#
# Layout. All sizes are in em, which keeps it in scale with the font.
#
#   +-- sidebar (fixed 23em) ---+-- main -------------------------------------+
#   | SOURCES                   | Writing system        [Result|Metrics] Copy |
#   | [ file table, 13em     ]  | +-----------------------------------------+ |
#   | [Add files] [Clear]  1,234| |  markdown result                        | |
#   | MODEL                     | |                                         | |
#   | Endpoint  [............]  | |                                         | |
#   | Model     [............]  | |                                         | |
#   | ...                       | |                                         | |
#   | [ Extract writing system ]| +-----------------------------------------+ |
#   +---------------------------+---------------------------------------------+
#   | status                                         3 files | 4,210 words   |  (status bar)

SIDEBAR_WIDTH_EM = 23.0
LABEL_WIDTH_EM = 6.5
FILES_HEIGHT_EM = 13.0
FULL_WIDTH = -imgui.FLT_MIN  # fill the remaining width
RADIUS = 5.0  # one corner radius for every rounded element


def em(x: float) -> float:
    return hello_imgui.em_size(x)


def section_header(title: str) -> None:
    imgui.dummy(imgui.ImVec2(0, em(0.3)))
    imgui.text_disabled(title)
    imgui.separator()
    imgui.dummy(imgui.ImVec2(0, em(0.2)))


def right_aligned_x(width: float) -> None:
    """Move the cursor so an item of ``width`` ends at the right edge of the region."""
    imgui.same_line(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - width)


def button_width(label: str) -> float:
    return imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 2


def wrapped(color: imgui.ImVec4, text: str) -> None:
    imgui.push_text_wrap_pos(0)
    imgui.text_colored(color, text)
    imgui.pop_text_wrap_pos()


def begin_box(str_id: str, size: imgui.ImVec2) -> None:
    """A bordered, rounded panel. Used for both the file list and the document."""
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(em(0.6), em(0.5)))
    imgui.push_style_color(imgui.Col_.child_bg, imgui.get_style().color_(imgui.Col_.frame_bg))
    imgui.begin_child(str_id, size, imgui.ChildFlags_.borders | imgui.ChildFlags_.always_use_window_padding)


def end_box() -> None:
    imgui.end_child()
    imgui.pop_style_color()
    imgui.pop_style_var()


def centered_hint(lines: tuple[str, ...]) -> None:
    """Muted text centered in the remaining region of the current window."""
    avail = imgui.get_content_region_avail()
    line_h = imgui.get_text_line_height_with_spacing()
    x0, y0 = imgui.get_cursor_pos_x(), imgui.get_cursor_pos_y() + (avail.y - line_h * len(lines)) / 2
    for i, line in enumerate(lines):
        imgui.set_cursor_pos(imgui.ImVec2(x0 + (avail.x - imgui.calc_text_size(line).x) / 2, y0 + i * line_h))
        imgui.text_disabled(line)


def form_row(label: str) -> None:
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text(label)
    imgui.table_next_column()
    imgui.set_next_item_width(FULL_WIDTH)


def gui_sources() -> None:
    section_header("Sources")
    busy = state.phase == Phase.RUNNING
    begin_box("files_box", imgui.ImVec2(0, em(FILES_HEIGHT_EM)))
    if not state.docs:
        hint = f"Reading {state.loading} file(s)…" if state.loading else "Drop files here"
        centered_hint((hint, " ".join(SUPPORTED_EXTENSIONS)))
    elif imgui.begin_table("files", 3, imgui.TableFlags_.sizing_fixed_fit):
        imgui.table_setup_column("file", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("words", imgui.TableColumnFlags_.width_fixed)
        imgui.table_setup_column("remove", imgui.TableColumnFlags_.width_fixed)
        remove = None
        for i, doc in enumerate(state.docs):
            imgui.push_id(i)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            imgui.text(doc.name)
            if imgui.is_item_hovered():
                imgui.set_tooltip(str(doc.path))
            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(f"{doc.word_count:,} words")
            imgui.table_next_column()
            imgui.begin_disabled(busy)
            if imgui.button("×", imgui.ImVec2(imgui.get_frame_height(), 0)):
                remove = i
            imgui.end_disabled()
            imgui.pop_id()
        if remove is not None:
            state.docs.pop(remove)
        imgui.end_table()
    end_box()

    if imgui.button("Add files…") and state.open_dialog is None:
        state.open_dialog = pfd.open_file("Add sample files", "", FILE_FILTERS, pfd.opt.multiselect)
    imgui.same_line()
    imgui.begin_disabled(not state.docs or busy)
    if imgui.button("Clear"):
        state.docs.clear()
        state.file_errors.clear()
    imgui.end_disabled()
    total = f"{sum(d.word_count for d in state.docs):,} words"
    right_aligned_x(imgui.calc_text_size(total).x)
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(total)

    if state.loading and state.docs:
        imgui.text_disabled(f"Reading {state.loading} file(s)…")
    for err in state.file_errors[-3:]:
        wrapped(ERROR_COLOR, err)


def gui_model() -> None:
    section_header("Model")
    s = state.settings
    if imgui.begin_table("model_form", 2, imgui.TableFlags_.sizing_fixed_fit):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, em(LABEL_WIDTH_EM))
        imgui.table_setup_column("field", imgui.TableColumnFlags_.width_stretch)

        form_row("Endpoint")
        _, s.base_url = imgui.input_text("##base_url", s.base_url)
        if imgui.is_item_hovered():
            imgui.set_tooltip("OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
        form_row("Model")
        _, s.model = imgui.input_text("##model", s.model)
        form_row("API key")
        hint = f"from ${config.API_KEY_ENV}" if os.environ.get(config.API_KEY_ENV) else ""
        _, s.api_key = imgui.input_text_with_hint("##api_key", hint, s.api_key, imgui.InputTextFlags_.password)
        form_row("Temperature")
        _, s.temperature = imgui.slider_float("##temperature", s.temperature, 0.0, 1.5, "%.2f")
        form_row("Excerpt")
        _, s.max_excerpt_chars = imgui.input_int("##excerpt", s.max_excerpt_chars, 5_000, 20_000)
        s.max_excerpt_chars = max(2_000, s.max_excerpt_chars)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Max characters of sample text sent to the model")
        imgui.end_table()

    _, s.save_api_key = imgui.checkbox("Remember API key", s.save_api_key)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Store the key in the config file (otherwise it is kept in memory only)")
    label = "Save settings"
    right_aligned_x(button_width(label))
    if imgui.button(label):
        try:
            config.save(s)
            state.status = f"Settings saved to {config.config_path()}"
        except OSError as exc:
            state.status = f"Could not save settings ({exc})"


def gui_run_button() -> None:
    """Primary action, pinned to the bottom of the sidebar."""
    s = state.settings
    missing = [n for n, v in (("files", state.docs), ("a model", s.model.strip()), ("an endpoint", s.base_url.strip())) if not v]
    height = em(2.4)
    hint_h = imgui.get_text_line_height_with_spacing() if missing and state.phase != Phase.RUNNING else 0
    bottom = imgui.get_cursor_pos_y() + imgui.get_content_region_avail().y - height - hint_h
    imgui.set_cursor_pos_y(max(imgui.get_cursor_pos_y(), bottom))

    if state.phase == Phase.RUNNING:
        if imgui.button("Cancel", imgui.ImVec2(FULL_WIDTH, height)):
            state.cancel.set()
            state.status = "Cancelling…"
        return
    if hint_h:
        imgui.text_disabled("Add " + " and ".join(missing) + " to start")
    imgui.begin_disabled(bool(missing) or state.loading > 0)
    imgui.push_style_color(imgui.Col_.button, ACCENT)
    imgui.push_style_color(imgui.Col_.button_hovered, ACCENT_HOVER)
    imgui.push_style_color(imgui.Col_.button_active, ACCENT)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 1, 1))
    if imgui.button("Extract writing system", imgui.ImVec2(FULL_WIDTH, height)):
        start_analysis()
    imgui.pop_style_color(4)
    imgui.end_disabled()


def gui_sidebar() -> None:
    gui_sources()
    gui_model()
    gui_run_button()


def segmented(labels: list[str], current: int) -> int:
    """A row of toggle buttons. Returns the selected index."""
    for i, label in enumerate(labels):
        if i:
            imgui.same_line()
        active = i == current
        if active:
            imgui.push_style_color(imgui.Col_.button, imgui.get_style().color_(imgui.Col_.header_active))
        if imgui.button(label, imgui.ImVec2(em(5.5), 0)):
            current = i
        if active:
            imgui.pop_style_color()
    return current


def gui_main() -> None:
    running = state.phase == Phase.RUNNING
    has_result = bool(state.result)

    # Toolbar: title on the left, view switch and actions on the right.
    imgui.align_text_to_frame_padding()
    imgui.text("Writing system")
    spacing = imgui.get_style().item_spacing.x
    actions_w = em(11) + spacing * 2 + button_width("Copy") + spacing + button_width("Save .md…")
    right_aligned_x(actions_w)
    imgui.begin_disabled(state.metrics is None)
    state.show_metrics = segmented(["Result", "Metrics"], int(state.show_metrics)) == 1
    imgui.end_disabled()
    imgui.same_line()
    imgui.begin_disabled(not has_result or running)
    if imgui.button("Copy"):
        imgui.set_clipboard_text(state.result)
        state.status = "Copied to clipboard"
    imgui.same_line()
    if imgui.button("Save .md…") and state.save_dialog is None:
        state.save_dialog = pfd.save_file("Save writing system", "writing-system.md", ["Markdown", "*.md"])
    imgui.end_disabled()

    if state.warning:
        wrapped(WARN_COLOR, state.warning)

    begin_box("document", imgui.ImVec2(0, 0))
    imgui.indent(em(0.6))
    if state.show_metrics and state.metrics is not None:
        rich_md.render_raw("```json\n" + json.dumps(state.metrics, indent=2) + "\n```")
    elif has_result:
        rich_md.render_raw(state.result)
    elif running:
        centered_hint((state.status or "Working…",))
    else:
        centered_hint(("Nothing yet.", "Add sample files, set the model, then press Extract writing system."))
    imgui.unindent(em(0.6))
    end_box()


def gui_status() -> None:
    """Hello ImGui status bar (bottom of the window)."""
    color = {Phase.FAILED: ERROR_COLOR, Phase.RUNNING: ACCENT_HOVER}.get(
        state.phase, imgui.get_style().color_(imgui.Col_.text_disabled))
    imgui.text_colored(color, state.status or "Ready")
    words = sum(d.word_count for d in state.docs)
    model = state.settings.model.strip() or "no model"
    info = f"{len(state.docs)} file(s)   |   {words:,} words   |   {model}"
    right_aligned_x(imgui.calc_text_size(info).x + imgui.get_style().window_padding.x)
    imgui.text_disabled(info)


_style_ready = False


def gui() -> None:
    global _style_ready
    if not _style_ready:  # em_size() is only meaningful once fonts are loaded
        setup_style()
        _style_ready = True
    drain_events()
    poll_dialogs()

    imgui.begin_child("sidebar", imgui.ImVec2(em(SIDEBAR_WIDTH_EM), 0))
    gui_sidebar()
    imgui.end_child()
    imgui.same_line(0, em(1.2))
    imgui.begin_child("main", imgui.ImVec2(0, 0))
    gui_main()
    imgui.end_child()


def setup_style() -> None:
    style = imgui.get_style()
    style.window_padding = imgui.ImVec2(em(0.9), em(0.8))
    style.frame_padding = imgui.ImVec2(em(0.6), em(0.35))
    style.item_spacing = imgui.ImVec2(em(0.5), em(0.5))
    style.item_inner_spacing = imgui.ImVec2(em(0.4), em(0.3))
    style.cell_padding = imgui.ImVec2(em(0.4), em(0.25))
    for name in ("window_rounding", "child_rounding", "frame_rounding", "popup_rounding", "scrollbar_rounding",
                 "grab_rounding", "tab_rounding", "image_rounding"):
        setattr(style, name, RADIUS)
    style.frame_border_size = 1
    style.child_border_size = 1


def install_drop_handler() -> None:
    """Hello ImGui has no drop API. Register a GLFW drop callback on its window."""
    try:
        from imgui_bundle import glfw_utils
        import glfw  # must be imported after imgui_bundle (uses its bundled GLFW library)

        window = glfw_utils.glfw_window_hello_imgui()
        glfw.set_drop_callback(window, lambda _w, paths: add_paths(list(paths)))
    except Exception as exc:  # e.g. a non-GLFW backend
        state.file_errors.append(f"Drag-and-drop unavailable ({exc}). Use 'Add files…' instead.")


def use_wsl_gpu() -> None:
    """On WSL, Mesa defaults to llvmpipe (CPU rendering, several cores per frame).
    Route OpenGL through the Windows GPU instead, unless the user chose a driver."""
    if "microsoft" in platform.release().lower() and os.path.exists("/dev/dxg"):
        os.environ.setdefault("GALLIUM_DRIVER", "d3d12")


def make_params() -> hello_imgui.RunnerParams:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Pasca"
    params.app_window_params.window_geometry.size = (1280, 820)
    params.app_window_params.restore_previous_geometry = True
    params.callbacks.show_gui = gui
    params.callbacks.post_init = install_drop_handler
    params.callbacks.show_status = gui_status
    params.imgui_window_params.show_status_bar = True
    params.imgui_window_params.show_status_fps = False
    params.imgui_window_params.remember_status_bar_settings = False
    params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
    params.fps_idling.fps_idle = 10
    params.fps_idling.fps_max = 60  # vsync is often ignored (WSL, software GL): cap active frames
    return params


def main() -> None:
    use_wsl_gpu()  # must run before the OpenGL context is created
    immapp.run(make_params(), immapp.AddOnsParams(with_markdown=True))
